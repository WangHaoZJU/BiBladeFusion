from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.core.settings import (
    MultiViewFusionConfig,
    SurfacePartitionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.features import FinSegmentationError, segment_single_fin
from biblade_fusion.perception.fusion import FusedBladeCloud, RegisteredCloudView
from biblade_fusion.perception.surface import (
    SurfaceRegion,
    generate_curved_view_plan,
    partition_curved_blade,
)
from biblade_fusion.perception.tsdf import integrate_bilateral_tsdf
from biblade_fusion.planning.surface_coverage import (
    create_surface_coverage_ledger,
    evaluate_surface_quality,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.storage.coarse_model import (
    read_coarse_model_summary,
    write_coarse_model,
)
from biblade_fusion.workflows.coarse_model import build_coarse_blade_model


def _main_and_fin(side_sign: int) -> tuple[np.ndarray, np.ndarray, int]:
    major = np.linspace(-0.12, 0.12, 49)
    minor = np.linspace(-0.045, 0.045, 31)
    x, y = np.meshgrid(major, minor, indexing="xy")
    camber = 0.002 * (x / 0.12) ** 2
    main_height = camber + side_sign * 0.004
    main = np.column_stack((x.ravel(), y.ravel(), main_height.ravel()))
    main_normals = np.tile([0.0, 0.0, float(side_sign)], (len(main), 1))

    fin_major = np.linspace(-0.047, 0.047, 33)
    extension = np.linspace(0.0, 0.032, 18)
    fin_x, distance = np.meshgrid(fin_major, extension, indexing="xy")
    base = 0.002 * (fin_x / 0.12) ** 2 + side_sign * 0.004
    fin_z = base + side_sign * distance
    negative = np.column_stack((fin_x.ravel(), np.full(fin_x.size, -0.0016), fin_z.ravel()))
    positive = np.column_stack((fin_x.ravel(), np.full(fin_x.size, 0.0016), fin_z.ravel()))
    fin = np.vstack((negative, positive))
    fin_normals = np.vstack(
        (
            np.tile([0.0, -1.0, 0.0], (len(negative), 1)),
            np.tile([0.0, 1.0, 0.0], (len(positive), 1)),
        )
    )
    return np.vstack((main, fin)), np.vstack((main_normals, fin_normals)), len(main)


def _config(**updates: object) -> SurfacePartitionConfig:
    values: dict[str, object] = {
        "voxel_size_m": 0.002,
        "maximum_points_per_side": 5000,
        "minimum_points_per_side": 250,
        "normal_neighbors": 10,
        "derive_footprint_from_intrinsics": False,
        "usable_footprint_m": (0.055, 0.035),
        "minimum_patch_points": 8,
        "curvature_split_threshold_deg": 6.0,
        "maximum_adaptive_depth": 1,
        "fin_mode": "required_single_per_side",
        "fin_main_normal_min_cosine": 0.65,
        "fin_seed_max_normal_cosine": 0.45,
        "fin_seed_min_height_m": 0.007,
        "fin_grow_min_height_m": 0.0015,
        "fin_connectivity_radius_m": 0.0055,
        "fin_minimum_points": 50,
        "fin_minimum_span_m": 0.015,
        "fin_root_band_m": 0.005,
        "fin_free_edge_band_m": 0.005,
        "fin_face_min_separation_m": 0.002,
    }
    values.update(updates)
    return SurfacePartitionConfig(**values)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(640, 480, 700.0, 700.0, 319.5, 239.5, "none", ())


def _planning() -> ViewPlanningConfig:
    return ViewPlanningConfig(
        standoff_distance_m=0.20,
        minimum_standoff_distance_m=0.14,
        maximum_standoff_distance_m=0.28,
        image_edge_margin_px=20,
        maximum_candidates=500,
    )


def _fused_fin_blade() -> FusedBladeCloud:
    front, front_normals, _ = _main_and_fin(1)
    back, back_normals, _ = _main_and_fin(-1)
    return FusedBladeCloud(
        np.vstack((front, back)),
        np.vstack((front_normals, back_normals)),
        np.concatenate((np.ones(len(front), dtype=np.int8), -np.ones(len(back), dtype=np.int8))),
        np.zeros(3),
        np.eye(3),
        0.008,
        (),
    )


def test_single_fin_segmentation_removes_fin_from_main_surface() -> None:
    points, normals, main_count = _main_and_fin(1)
    segmentation = segment_single_fin(
        points,
        normals,
        np.zeros(3),
        np.eye(3),
        BladeSide.FRONT,
        _config(),
    )

    component = segmentation.component
    assert component is not None
    assert np.count_nonzero(segmentation.main_mask) == main_count
    assert abs(component.normal_axis[1]) > 0.98
    assert component.two_faces_observed
    assert component.face_separation_m == pytest.approx(0.0032, abs=0.0003)
    assert np.count_nonzero(component.root_mask) > 0
    assert np.count_nonzero(component.free_edge_mask) > 0


def test_required_fin_mode_rejects_a_plain_blade_side() -> None:
    points, normals, main_count = _main_and_fin(1)
    with pytest.raises(FinSegmentationError, match="no valid protruding fin"):
        segment_single_fin(
            points[:main_count],
            normals[:main_count],
            np.zeros(3),
            np.eye(3),
            BladeSide.FRONT,
            _config(),
        )


def test_bilateral_fin_partition_generates_face_root_and_free_edge_views() -> None:
    surface = partition_curved_blade(_fused_fin_blade(), _config())

    assert {component.side for component in surface.fin_components} == {
        BladeSide.FRONT,
        BladeSide.BACK,
    }
    assert all(component.two_faces_observed for component in surface.fin_components)
    assert {
        SurfaceRegion.FIN_FACE,
        SurfaceRegion.FIN_ROOT,
        SurfaceRegion.FIN_FREE_EDGE,
    }.issubset({patch.region for patch in surface.patches})
    for component in surface.fin_components:
        face_normals = np.asarray(
            [
                patch.main_normal
                for patch in surface.for_side(component.side)
                if patch.region is SurfaceRegion.FIN_FACE
            ]
        )
        assert np.min(face_normals @ component.normal_axis) < -0.9
        assert np.max(face_normals @ component.normal_axis) > 0.9
        root_normals = np.asarray(
            [
                patch.main_normal
                for patch in surface.for_side(component.side)
                if patch.region is SurfaceRegion.FIN_ROOT
            ]
        )
        main_outward = (
            surface.axes[:, 2] if component.side is BladeSide.FRONT else -surface.axes[:, 2]
        )
        assert np.all(root_normals @ main_outward > 0.4)

    plan = generate_curved_view_plan(
        surface,
        _intrinsics(),
        _planning(),
        _config(),
    )
    assert len(plan.candidates) == len(surface.patches)
    assert all(
        np.allclose(candidate.optical_axis, -candidate.patch.outward_normal)
        for candidate in plan.candidates
    )

    report = evaluate_surface_quality(
        create_surface_coverage_ledger(surface),
        surface,
        SurfaceQualityConfig(minimum_observed_points=3),
    )
    assert SurfaceRegion.FIN_ROOT in report.edge_completion
    assert SurfaceRegion.FIN_FREE_EDGE in report.edge_completion


def test_fin_thickness_protects_tsdf_truncation_band() -> None:
    fused = _fused_fin_blade()
    front = fused.points_for_side(1)
    back = fused.points_for_side(-1)
    views = (
        RegisteredCloudView("front", front, [0.0, 0.0, 0.30]),
        RegisteredCloudView("back", back, [0.0, 0.0, -0.30]),
    )
    result = integrate_bilateral_tsdf(
        fused,
        views,
        TSDFConfig(
            voxel_size_m=0.001,
            truncation_distance_m=0.006,
            maximum_voxels=500_000,
            use_open3d_if_available=False,
        ),
        feature_thicknesses_m=(0.0032,),
    )

    assert result.feature_thicknesses_m == (0.0032,)
    assert result.protected_truncation_distance_m == pytest.approx(0.00128)


def test_fin_workflow_persists_component_geometry_and_quality(tmp_path) -> None:
    fused = _fused_fin_blade()
    views = (
        RegisteredCloudView("front", fused.points_for_side(1), [0.0, 0.0, 0.30]),
        RegisteredCloudView("back", fused.points_for_side(-1), [0.0, 0.0, -0.30]),
    )
    fusion = MultiViewFusionConfig(
        voxel_size_m=0.002,
        maximum_icp_points=5000,
        icp_iterations=0,
        minimum_correspondences=20,
        normal_neighbors=10,
    )
    partition = _config()
    planning = _planning()
    tsdf = TSDFConfig(
        voxel_size_m=0.001,
        truncation_distance_m=0.006,
        maximum_voxels=500_000,
        use_open3d_if_available=False,
    )
    quality = SurfaceQualityConfig(
        maximum_surface_distance_m=0.004,
        completed_fraction=0.70,
        maximum_rmse_m=0.004,
        minimum_normal_consistency=0.60,
        minimum_observed_points=3,
    )
    result = build_coarse_blade_model(
        views, _intrinsics(), fusion, partition, planning, tsdf, quality
    )
    settings = load_settings("configs/default.yaml").model_copy(
        update={
            "multi_view_fusion": fusion,
            "surface_partition": partition,
            "view_planning": planning,
            "tsdf": tsdf,
            "surface_quality": quality,
        }
    )
    sources = []
    for view in views:
        source = tmp_path / view.view_id
        source.mkdir()
        (source / "metadata.json").write_text("{}\n", encoding="utf-8")
        sources.append(source)
    output = write_coarse_model(
        tmp_path / "fin_coarse",
        result,
        settings,
        source_views=tuple(sources),
    )
    stored = read_coarse_model_summary(output)

    assert stored.metadata["schema_version"] == 4
    assert len(stored.metadata["surface"]["fin_components"]) == 2
    assert "fin_component_points_m" in stored.metadata["files"]
    assert len(stored.metadata["tsdf"]["feature_thicknesses_m"]) == 2
    assert "fin_root" in stored.metadata["quality"]["edge_completion"]
    assert "fin_free_edge" in stored.metadata["quality"]["edge_completion"]
