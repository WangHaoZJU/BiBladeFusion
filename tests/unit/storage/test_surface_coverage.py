from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
    SurfacePartitionConfig,
    SurfaceQualityConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.features import FinComponent
from biblade_fusion.perception.fusion import FusedBladeCloud
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    CurvedSurfacePatch,
    CurvedViewPlan,
    SurfaceRegion,
)
from biblade_fusion.perception.tsdf import (
    BilateralTSDFResult,
    SparseTSDFVolume,
    TriangleMesh,
)
from biblade_fusion.planning.surface_coverage import (
    SurfaceCoverageLedger,
    SurfacePatchEvidence,
    evaluate_surface_quality,
)
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import (
    read_surface_coverage_generation,
    write_coarse_model,
    write_reconstructed_view,
    write_surface_coverage_generation,
)
from biblade_fusion.workflows import AuthoritativeRobotPose, ReconstructedBladeView
from biblade_fusion.workflows.coarse_model import CoarseModelResult


def _plane_patch(
    patch_id: str,
    side: BladeSide,
    region: SurfaceRegion,
    center: np.ndarray,
    normal: np.ndarray,
    *,
    row: int = 0,
    column: int = 0,
) -> CurvedSurfacePatch:
    normal = np.asarray(normal, dtype=np.float64)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = seed - float(seed @ normal) * normal
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    offsets = np.linspace(-0.01, 0.01, 3)
    first_grid, second_grid = np.meshgrid(offsets, offsets, indexing="xy")
    points = (
        center
        + first_grid.ravel()[:, None] * first
        + second_grid.ravel()[:, None] * second
    )
    coordinates = np.column_stack((first_grid.ravel(), second_grid.ravel()))
    return CurvedSurfacePatch(
        patch_id,
        side,
        region,
        row,
        column,
        0,
        points,
        np.tile(normal, (len(points), 1)),
        coordinates,
        center,
        np.column_stack((first, second, normal)),
        np.array([0.02, 0.02, 0.001]),
        normal,
        0.0,
        0.0,
    )


def _fin_component(side: BladeSide, *, two_faces: bool = True) -> FinComponent:
    sign = 1.0 if side is BladeSide.FRONT else -1.0
    x = np.linspace(-0.01, 0.01, 6)
    z = 0.51 + sign * np.linspace(0.004, 0.018, 6)
    negative = np.column_stack((x, np.full(6, -0.001), z))
    positive = np.column_stack((x, np.full(6, 0.001), z))
    points = np.vstack((negative, positive))
    root = np.zeros(12, dtype=bool)
    root[[0, 6]] = True
    free = np.zeros(12, dtype=bool)
    free[[5, 11]] = True
    return FinComponent(
        f"{side.value}_fin",
        side,
        points,
        np.vstack(
            (
                np.tile([0.0, -1.0, 0.0], (6, 1)),
                np.tile([0.0, 1.0, 0.0], (6, 1)),
            )
        ),
        points[:, [0, 2]],
        np.full(12, 0.01),
        root,
        free,
        np.mean(points, axis=0),
        np.eye(3),
        np.array([0.02, 0.004, 0.02]),
        np.array([0.0, 1.0, 0.0]),
        0.0005,
        0.002,
        two_faces,
    )


def _camera_rotation(outward_normal: np.ndarray) -> np.ndarray:
    optical = -np.asarray(outward_normal, dtype=np.float64)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ optical)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    camera_x = seed - float(seed @ optical) * optical
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(optical, camera_x)
    return np.column_stack((camera_x, camera_y, optical))


def _surface_and_plan(
    *,
    include_fin: bool = False,
    include_all_fin_regions: bool = True,
    two_faces: bool = True,
) -> tuple[CurvedBladeSurface, CurvedViewPlan]:
    patches = [
        _plane_patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            np.array([0.0, 0.0, 0.50]),
            np.array([0.0, 0.0, 1.0]),
        ),
        _plane_patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            np.array([0.0, 0.0, 0.48]),
            np.array([0.0, 0.0, -1.0]),
        ),
    ]
    components: tuple[FinComponent, ...] = ()
    if include_fin:
        components = (
            _fin_component(BladeSide.FRONT, two_faces=two_faces),
            _fin_component(BladeSide.BACK, two_faces=two_faces),
        )
        regions = [SurfaceRegion.FIN_FACE, SurfaceRegion.FIN_ROOT]
        if include_all_fin_regions:
            regions.append(SurfaceRegion.FIN_FREE_EDGE)
        for side in (BladeSide.FRONT, BladeSide.BACK):
            sign = 1.0 if side is BladeSide.FRONT else -1.0
            for index, region in enumerate(regions):
                normal = (
                    np.array([0.0, 1.0, 0.0])
                    if region is SurfaceRegion.FIN_FACE
                    else np.array([0.0, 0.0, sign])
                )
                patches.append(
                    _plane_patch(
                        f"{side.value}_{region.value}",
                        side,
                        region,
                        np.array([0.03 * (index - 1), 0.0, 0.49 + 0.02 * sign]),
                        normal,
                        row=index + 1,
                    )
                )
    surface = CurvedBladeSurface(
        "base",
        tuple(patches),
        np.eye(3),
        np.array([0.0, 0.0, 0.49]),
        (0.02, 0.02),
        (0, 0),
        (1, 1),
        (0.05, 0.05),
        "configured_override",
        components,
    )
    angle = np.deg2rad(2.0)
    calibration = PoseSE3.from_rotation_translation(
        "left_rectified",
        "left_ir",
        np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        ),
        np.zeros(3),
    )
    candidates = []
    rectified_poses = []
    for patch in surface.patches:
        rotation = _camera_rotation(patch.main_normal)
        translation = patch.obb_center_m + 0.25 * patch.main_normal
        rectified = PoseSE3.from_rotation_translation(
            "base", "left_rectified", rotation, translation
        )
        raw = rectified.compose(calibration)
        candidates.append(
            CandidateView(
                patch.patch_id,
                SurfacePatch(
                    patch.patch_id,
                    patch.side,
                    patch.row,
                    patch.column,
                    patch.obb_center_m,
                    patch.main_normal,
                    patch.planar_extents_m,
                ),
                raw,
                0.25,
                (0.05, 0.05),
            )
        )
        rectified_poses.append(rectified)
    return surface, CurvedViewPlan(
        surface,
        tuple(candidates),
        tuple(rectified_poses),
        calibration,
        (0.05, 0.05),
    )


def _write_coarse_reference(
    tmp_path: Path,
    *,
    fin_mode: str = "disabled",
    include_fin: bool = False,
    include_all_fin_regions: bool = True,
    two_faces: bool = True,
    name: str = "coarse",
) -> tuple[Path, CurvedBladeSurface, CurvedViewPlan]:
    surface, plan = _surface_and_plan(
        include_fin=include_fin,
        include_all_fin_regions=include_all_fin_regions,
        two_faces=two_faces,
    )
    points = np.vstack([surface.patches[0].points_m, surface.patches[1].points_m])
    normals = np.vstack([surface.patches[0].normals, surface.patches[1].normals])
    fused = FusedBladeCloud(
        points,
        normals,
        np.concatenate(
            (
                np.ones(len(surface.patches[0].points_m), dtype=np.int8),
                -np.ones(len(surface.patches[1].points_m), dtype=np.int8),
            )
        ),
        surface.center_m,
        surface.axes,
        0.02,
        (),
    )
    mesh = TriangleMesh(
        surface.patches[0].points_m[[0, 2, 6, 8]],
        np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        np.ones(2, dtype=np.int8),
    )
    front_volume = SparseTSDFVolume(
        1,
        np.zeros(3),
        0.002,
        0.004,
        np.array([[0, 0, 0]], dtype=np.int32),
        np.array([0.0]),
        np.array([1.0]),
    )
    back_volume = SparseTSDFVolume(
        -1,
        np.zeros(3),
        0.002,
        0.004,
        np.array([[0, 0, 0]], dtype=np.int32),
        np.array([0.0]),
        np.array([1.0]),
    )
    tsdf = BilateralTSDFResult(front_volume, back_volume, mesh, 0.003)
    coarse_ledger = SurfaceCoverageLedger(
        tuple(
            SurfacePatchEvidence(
                patch.patch_id,
                np.zeros(len(patch.points_m)),
                np.ones(len(patch.points_m)),
                ("coarse:seed",),
            )
            for patch in surface.patches
        ),
        ("coarse:seed",),
    )
    quality_config = SurfaceQualityConfig(
        minimum_observed_points=3,
        completed_fraction=0.8,
    )
    result = CoarseModelResult(
        fused,
        surface,
        plan,
        tsdf,
        coarse_ledger,
        evaluate_surface_quality(coarse_ledger, surface, quality_config, mesh=mesh),
    )
    partition = SurfacePartitionConfig(
        derive_footprint_from_intrinsics=False,
        usable_footprint_m=(0.05, 0.05),
        fin_mode=fin_mode,
    )
    settings = load_settings("configs/default.yaml").model_copy(
        update={
            "surface_partition": partition,
            "surface_quality": quality_config,
        }
    )
    source = tmp_path / f"{name}_source"
    source.mkdir()
    (source / "metadata.json").write_text("{}\n", encoding="utf-8")
    output = write_coarse_model(
        tmp_path / name,
        result,
        settings,
        source_views=(source,),
    )
    return output, surface, plan


def _write_front_view(
    tmp_path: Path,
    surface: CurvedBladeSurface,
    plan: CurvedViewPlan,
    *,
    depth_source: str = "foundation_stereo",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    patch = surface.patches[0]
    camera_pose = plan.candidates[0].base_t_left_ir
    projection_pose = (
        plan.candidate_base_t_left_rectified[0]
        if depth_source == "foundation_stereo"
        else PoseSE3("base", "depth", camera_pose.matrix)
    )
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    flange_t_tcp = load_es68_flange_t_tcp()
    flange_t_left_ir = base_t_flange.inverse().compose(camera_pose)
    tcp_t_left_ir = flange_t_tcp.inverse().compose(flange_t_left_ir)
    hand_eye_source = tmp_path / "hand_eye.yaml"
    hand_eye_source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "test",
                "matrix": flange_t_left_ir.matrix.tolist(),
                "derived_runtime": {"tcp_T_left_ir": tcp_t_left_ir.matrix.tolist()},
                "quality": {
                    "sample_count": 20,
                    "translation_rmse_m": 0.001,
                    "rotation_rmse_deg": 0.2,
                    "rotation_span_deg": 45.0,
                    "translation_span_m": 0.1,
                    "rotation_axis_diversity": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "test",
        20,
        0.001,
        0.2,
        hand_eye_source,
        flange_t_left_ir=flange_t_left_ir,
    )
    authority = AuthoritativeRobotPose(
        base_t_flange,
        base_t_flange.compose(flange_t_tcp),
        base_t_flange.compose(flange_t_tcp),
        0.0,
        0.0,
        0.002,
        0.3,
        (0.0,) * 6,
    )
    pixels = np.column_stack(np.meshgrid(np.arange(3), np.arange(3))).reshape(-1, 2)
    cloud = PointCloud("base", patch.points_m, pixels, (3, 3))
    view = ReconstructedBladeView(
        patch.patch_id,
        1,
        11,
        CameraIntrinsics(3, 3, 100.0, 100.0, 1.0, 1.0, "none", ()),
        np.zeros(6),
        camera_pose,
        projection_pose,
        cloud,
        depth_source,
        authority,
    )
    return write_reconstructed_view(
        tmp_path / f"front_view_{depth_source}",
        view,
        np.ones((3, 3), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
        source_stereo_inference=(
            tmp_path / "stereo_inference"
            if depth_source == "foundation_stereo"
            else None
        ),
    )


def _fine_quality() -> SurfaceQualityConfig:
    return SurfaceQualityConfig(
        maximum_surface_distance_m=0.002,
        minimum_incidence_cosine=0.5,
        completed_fraction=0.8,
        maximum_rmse_m=0.002,
        minimum_normal_consistency=0.8,
        minimum_observed_points=3,
    )


def test_initial_generation_starts_empty_and_restores_typed_reference(
    tmp_path: Path,
) -> None:
    coarse, surface, _ = _write_coarse_reference(tmp_path)
    output = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    stored = read_surface_coverage_generation(output)

    assert stored.root == output.resolve()
    assert stored.reference.root == coarse.resolve()
    assert stored.surface.frame == "base"
    assert tuple(item.patch_id for item in stored.surface.patches) == tuple(
        item.patch_id for item in surface.patches
    )
    assert stored.view_plan.surface is stored.surface
    assert stored.ledger.observation_ids == ()
    assert all(not item.observation_ids for item in stored.ledger.evidence)
    assert all(np.isinf(item.minimum_distances_m).all() for item in stored.ledger.evidence)
    assert stored.quality.completion_fraction == 0.0
    assert stored.quality.mesh_triangle_count == 0
    assert stored.quality.mesh_watertight is False
    assert stored.metadata["summary"]["complete"] is False
    assert stored.previous_generation_path is None
    assert stored.current_reconstructed_view_path is None
    assert stored.required_patch_ids == ("front_surface", "back_surface")
    assert stored.required_regions == (SurfaceRegion.SURFACE,)
    assert stored.metadata["reference"]["path"] == str(coarse.resolve())


def test_successor_replays_one_view_and_uses_source_view_id(tmp_path: Path) -> None:
    coarse, surface, plan = _write_coarse_reference(tmp_path)
    initial = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    view = _write_front_view(tmp_path, surface, plan)
    successor = write_surface_coverage_generation(
        tmp_path / "generation_001",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
        previous_generation=initial,
        current_reconstructed_view=view,
        observation_id="front_surface",
    )
    stored = read_surface_coverage_generation(successor)

    assert stored.ledger.observation_ids == ("front_surface",)
    assert stored.ledger.evidence[0].observation_ids == ("front_surface",)
    assert stored.ledger.evidence[1].observation_ids == ()
    assert stored.previous_generation_path == initial.resolve()
    assert stored.current_reconstructed_view_path == view.resolve()
    assert stored.quality.patches[0].complete
    assert not stored.quality.patches[1].complete
    for rectified, raw in zip(
        stored.view_plan.candidate_base_t_left_rectified,
        stored.view_plan.candidates,
        strict=True,
    ):
        np.testing.assert_allclose(
            rectified.compose(stored.view_plan.left_rectified_t_left_ir).matrix,
            raw.base_t_left_ir.matrix,
        )
    np.testing.assert_allclose(
        stored.view_plan.left_rectified_t_left_ir.matrix,
        plan.left_rectified_t_left_ir.matrix,
    )


def test_writer_rejects_observation_id_not_equal_to_candidate_view_id(
    tmp_path: Path,
) -> None:
    coarse, surface, plan = _write_coarse_reference(tmp_path)
    initial = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    view = _write_front_view(tmp_path, surface, plan)

    with pytest.raises(ValueError, match="must equal the reconstructed source view ID"):
        write_surface_coverage_generation(
            tmp_path / "generation_001",
            reference_coarse_model=coarse,
            quality_config=_fine_quality(),
            previous_generation=initial,
            current_reconstructed_view=view,
            observation_id="fine:front_surface",
        )


def test_writer_rejects_native_depth_as_a_fine_successor(tmp_path: Path) -> None:
    coarse, surface, plan = _write_coarse_reference(tmp_path)
    initial = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    native_view = _write_front_view(
        tmp_path,
        surface,
        plan,
        depth_source="native_realsense",
    )

    with pytest.raises(ValueError, match="requires a FoundationStereo reconstructed view"):
        write_surface_coverage_generation(
            tmp_path / "generation_001",
            reference_coarse_model=coarse,
            quality_config=_fine_quality(),
            previous_generation=initial,
            current_reconstructed_view=native_view,
            observation_id="front_surface",
        )


def test_reader_rejects_forged_complete_summary_and_two_id_append(
    tmp_path: Path,
) -> None:
    coarse, surface, plan = _write_coarse_reference(tmp_path)
    initial = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    metadata_path = initial / "coverage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["summary"]["complete"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="completion summary is false"):
        read_surface_coverage_generation(initial)

    initial = write_surface_coverage_generation(
        tmp_path / "generation_000_valid",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    view = _write_front_view(tmp_path / "second", surface, plan)
    successor = write_surface_coverage_generation(
        tmp_path / "generation_001",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
        previous_generation=initial,
        current_reconstructed_view=view,
        observation_id="front_surface",
    )
    metadata_path = successor / "coverage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ledger"]["observation_ids"].append("back_surface")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="append exactly its one current observation"):
        read_surface_coverage_generation(successor)


def test_reader_rejects_array_escape_and_checksum_change(tmp_path: Path) -> None:
    coarse, _, _ = _write_coarse_reference(tmp_path)
    escaped = write_surface_coverage_generation(
        tmp_path / "escaped",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    metadata_path = escaped / "coverage.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["minimum_distances_m"]["path"] = "../outside.npy"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="path escapes"):
        read_surface_coverage_generation(escaped)

    changed = write_surface_coverage_generation(
        tmp_path / "changed",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    array_path = changed / "best_normal_cosines.npy"
    array = np.load(array_path, allow_pickle=False)
    array[0] = 0.0
    np.save(array_path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_surface_coverage_generation(changed)


def test_reference_must_be_exact_schema_5(tmp_path: Path) -> None:
    coarse, _, _ = _write_coarse_reference(tmp_path)
    legacy = tmp_path / "legacy"
    shutil.copytree(coarse, legacy)
    metadata_path = legacy / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="exact schema-5"):
        write_surface_coverage_generation(
            tmp_path / "generation_000",
            reference_coarse_model=legacy,
            quality_config=_fine_quality(),
        )


@pytest.mark.parametrize(
    ("include_fin", "include_all_regions", "two_faces", "message"),
    [
        (False, True, True, "one coarse fin component on each side"),
        (True, False, True, "fin patch regions are missing"),
        (True, True, False, "does not have two observed faces"),
    ],
)
def test_required_fin_reference_rejects_missing_semantics(
    tmp_path: Path,
    include_fin: bool,
    include_all_regions: bool,
    two_faces: bool,
    message: str,
) -> None:
    coarse, _, _ = _write_coarse_reference(
        tmp_path,
        fin_mode="required_single_per_side",
        include_fin=include_fin,
        include_all_fin_regions=include_all_regions,
        two_faces=two_faces,
    )

    with pytest.raises(ValueError, match=message):
        write_surface_coverage_generation(
            tmp_path / "generation_000",
            reference_coarse_model=coarse,
            quality_config=_fine_quality(),
        )


def test_required_fin_reference_round_trips_all_regions(tmp_path: Path) -> None:
    coarse, _, _ = _write_coarse_reference(
        tmp_path,
        fin_mode="required_single_per_side",
        include_fin=True,
    )
    output = write_surface_coverage_generation(
        tmp_path / "generation_000",
        reference_coarse_model=coarse,
        quality_config=_fine_quality(),
    )
    stored = read_surface_coverage_generation(output)

    assert {component.side for component in stored.surface.fin_components} == {
        BladeSide.FRONT,
        BladeSide.BACK,
    }
    assert {
        SurfaceRegion.FIN_FACE,
        SurfaceRegion.FIN_ROOT,
        SurfaceRegion.FIN_FREE_EDGE,
    }.issubset(set(stored.required_regions))
