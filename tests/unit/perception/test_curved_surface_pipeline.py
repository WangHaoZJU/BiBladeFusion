from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    MultiViewFusionConfig,
    SurfacePartitionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.fusion import RegisteredCloudView, fuse_registered_views
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.surface import (
    SurfaceRegion,
    derive_usable_footprint,
    generate_curved_view_plan,
    partition_curved_blade,
)
from biblade_fusion.perception.tsdf import integrate_bilateral_tsdf
from biblade_fusion.planning.fine_plan_gui import _inspection_scene
from biblade_fusion.planning.surface_coverage import (
    create_surface_coverage_ledger,
    evaluate_surface_quality,
    update_surface_coverage,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.storage.coarse_model import (
    read_coarse_model_summary,
    write_coarse_model,
)
from biblade_fusion.storage.fine_plan_inspection import (
    read_fine_plan_inspection,
    write_fine_plan_inspection,
)
from biblade_fusion.workflows.coarse_model import (
    build_coarse_blade_model,
    derive_consistent_left_rectified_t_left_ir,
)
from biblade_fusion.workflows.fine_plan_inspection import inspect_fine_plan
from biblade_fusion.workflows.reconstruction import ReconstructedBladeView


def _blade_views() -> tuple[RegisteredCloudView, ...]:
    major = np.linspace(-0.12, 0.12, 61)
    minor = np.linspace(-0.04, 0.04, 31)
    x, y = np.meshgrid(major, minor, indexing="xy")
    camber = 0.010 * (x / 0.12) ** 2 + 0.002 * np.cos(np.pi * y / 0.08)
    front = np.column_stack((x.ravel(), y.ravel(), (camber + 0.004).ravel()))
    back = np.column_stack((x.ravel(), y.ravel(), (camber - 0.004).ravel()))
    left = x.ravel() <= 0.035
    right = x.ravel() >= -0.035
    shift = np.array([0.0007, -0.0003, 0.0002])
    return (
        RegisteredCloudView("front_left", front[left], [0.0, 0.0, 0.35]),
        RegisteredCloudView("front_right", front[right] + shift, [0.03, 0.0, 0.35]),
        RegisteredCloudView("back_left", back[left], [0.0, 0.0, -0.35]),
        RegisteredCloudView("back_right", back[right] - shift, [-0.03, 0.0, -0.35]),
    )


def _fusion_config() -> MultiViewFusionConfig:
    return MultiViewFusionConfig(
        voxel_size_m=0.003,
        maximum_icp_points=1800,
        icp_iterations=8,
        maximum_correspondence_distance_m=0.012,
        minimum_correspondences=80,
        normal_neighbors=12,
    )


def _partition_config() -> SurfacePartitionConfig:
    return SurfacePartitionConfig(
        voxel_size_m=0.003,
        maximum_points_per_side=2500,
        minimum_points_per_side=200,
        normal_neighbors=12,
        derive_footprint_from_intrinsics=False,
        usable_footprint_m=(0.055, 0.035),
        minimum_patch_points=12,
        curvature_split_threshold_deg=0.5,
        maximum_adaptive_depth=1,
        fin_mode="disabled",
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(640, 480, 700.0, 700.0, 319.5, 239.5, "none", ())


def _planning_config() -> ViewPlanningConfig:
    return ViewPlanningConfig(
        standoff_distance_m=0.25,
        minimum_standoff_distance_m=0.15,
        maximum_standoff_distance_m=0.30,
        image_edge_margin_px=20,
        maximum_candidates=500,
    )


def _identity_rectification() -> PoseSE3:
    return PoseSE3.identity("left_rectified", "left_ir")


def _non_identity_rectification() -> PoseSE3:
    angle = radians(7.0)
    return PoseSE3.from_rotation_translation(
        "left_rectified",
        "left_ir",
        np.array(
            [
                [cos(angle), 0.0, sin(angle)],
                [0.0, 1.0, 0.0],
                [-sin(angle), 0.0, cos(angle)],
            ]
        ),
        np.zeros(3),
    )


def _reconstructed_blade_views(
    left_rectified_t_left_ir: PoseSE3,
) -> tuple[ReconstructedBladeView, ...]:
    reconstructed = []
    for index, view in enumerate(_blade_views()):
        rotation = np.diag([1.0, -1.0, -1.0]) if view.camera_origin_m[2] > 0.0 else np.eye(3)
        base_t_left_rectified = PoseSE3.from_rotation_translation(
            "base", "left_rectified", rotation, view.camera_origin_m
        )
        reconstructed.append(
            ReconstructedBladeView(
                view.view_id,
                index,
                index,
                _intrinsics(),
                np.zeros(6),
                base_t_left_rectified.compose(left_rectified_t_left_ir),
                base_t_left_rectified,
                PointCloud(
                    "base",
                    view.points_m,
                    np.zeros((len(view.points_m), 2), dtype=np.int32),
                    (480, 640),
                ),
                "foundation_stereo",
            )
        )
    return tuple(reconstructed)


def test_fusion_separates_sides_and_bounds_pose_refinement() -> None:
    fused = fuse_registered_views(_blade_views(), _fusion_config())

    assert set(fused.side_labels) == {-1, 1}
    assert 0.004 < fused.median_thickness_m < 0.012
    assert all(
        np.linalg.norm(item.correction_matrix[:3, 3]) <= 0.008 + 1e-12 for item in fused.refinements
    )
    assert {item.side for item in fused.refinements} == {-1, 1}
    assert fused.refinements[0].side == 1


def test_baseline_footprint_is_derived_from_calibrated_intrinsics() -> None:
    config = _planning_config().model_copy(update={"footprint_utilization": 0.8})
    footprint = derive_usable_footprint(_intrinsics(), config)

    assert footprint[0] == pytest.approx(0.25 * (639 - 40) / 700.0 * 0.8)
    assert footprint[1] == pytest.approx(0.25 * (479 - 40) / 700.0 * 0.8)


def test_paper_partition_has_real_normals_obb_edges_and_adaptive_patches() -> None:
    fused = fuse_registered_views(_blade_views(), _fusion_config())
    surface = partition_curved_blade(fused, _partition_config())

    assert {patch.side for patch in surface.patches} == {BladeSide.FRONT, BladeSide.BACK}
    assert {patch.region for patch in surface.patches} == {
        SurfaceRegion.SURFACE,
        SurfaceRegion.LEADING_EDGE,
        SurfaceRegion.TRAILING_EDGE,
        SurfaceRegion.ROOT,
        SurfaceRegion.TIP,
    }
    assert surface.parameterization_methods == ("boundary_curves", "boundary_curves")
    assert len(surface.boundary_models) == 2
    assert not any(surface.boundary_fallback_reasons)
    assert min(surface.angle_boundary_counts) > 0
    assert all(np.isfinite(patch.obb_center_m).all() for patch in surface.patches)
    assert all(np.isclose(np.linalg.norm(patch.main_normal), 1.0) for patch in surface.patches)
    assert any(patch.adaptive_depth == 1 for patch in surface.patches)
    assert np.ptp([patch.main_normal[2] for patch in surface.patches]) > 0.01
    major_count, minor_count = surface.base_grid_counts
    assert major_count > 1 and minor_count > 1
    assert all(patch.column < major_count and patch.row < minor_count for patch in surface.patches)

    plan = generate_curved_view_plan(
        surface,
        _intrinsics(),
        _planning_config(),
        _partition_config(),
        left_rectified_t_left_ir=_identity_rectification(),
    )
    assert len(plan.candidates) == len(surface.patches)
    assert plan.motion_authorized is False
    for candidate in plan.candidates:
        assert np.allclose(candidate.optical_axis, -candidate.patch.outward_normal)
        assert 0.15 <= candidate.standoff_distance_m <= 0.30
        assert candidate.projection_fraction == pytest.approx(1.0)
        assert candidate.visibility_fraction >= 0.90
        assert candidate.distance_policy.startswith("adaptive_")


def test_non_identity_rectification_converts_rectified_look_at_pose_to_raw() -> None:
    left_rectified_t_left_ir = _non_identity_rectification()
    reconstructed = _reconstructed_blade_views(left_rectified_t_left_ir)
    derived = derive_consistent_left_rectified_t_left_ir(reconstructed)
    assert np.allclose(derived.matrix, left_rectified_t_left_ir.matrix)

    surface = partition_curved_blade(
        fuse_registered_views(_blade_views(), _fusion_config()), _partition_config()
    )
    plan = generate_curved_view_plan(
        surface,
        _intrinsics(),
        _planning_config(),
        _partition_config(),
        left_rectified_t_left_ir=derived,
    )
    assert np.allclose(plan.left_rectified_t_left_ir.matrix, left_rectified_t_left_ir.matrix)
    for base_t_left_rectified, candidate in zip(
        plan.candidate_base_t_left_rectified,
        plan.candidates,
        strict=True,
    ):
        assert np.allclose(
            base_t_left_rectified.rotation[:, 2],
            -candidate.patch.outward_normal,
        )
        assert np.allclose(
            candidate.base_t_left_ir.matrix,
            base_t_left_rectified.matrix @ left_rectified_t_left_ir.matrix,
        )
        assert not np.allclose(
            candidate.base_t_left_ir.rotation,
            base_t_left_rectified.rotation,
        )

    inconsistent = list(reconstructed)
    different_calibration = PoseSE3.identity("left_rectified", "left_ir")
    inconsistent[1] = replace(
        inconsistent[1],
        base_t_left_ir=inconsistent[1].base_t_projection_camera.compose(different_calibration),
    )
    with pytest.raises(ValueError, match="inconsistent left_rectified_T_left_ir"):
        build_coarse_blade_model(
            tuple(inconsistent),
            _intrinsics(),
            _fusion_config(),
            _partition_config(),
            _planning_config(),
            TSDFConfig(use_open3d_if_available=False),
            SurfaceQualityConfig(),
        )


def test_real_surface_coverage_and_thin_wall_tsdf() -> None:
    views = _blade_views()
    fused = fuse_registered_views(views, _fusion_config())
    surface = partition_curved_blade(fused, _partition_config())
    quality_config = SurfaceQualityConfig(
        maximum_surface_distance_m=0.006,
        completed_fraction=0.70,
        maximum_rmse_m=0.004,
        minimum_normal_consistency=0.60,
        minimum_observed_points=8,
    )
    ledger = create_surface_coverage_ledger(surface)
    for view in views:
        ledger = update_surface_coverage(ledger, surface, view, view.view_id, quality_config)

    tsdf = integrate_bilateral_tsdf(
        fused,
        views,
        TSDFConfig(
            voxel_size_m=0.003,
            truncation_distance_m=0.010,
            maximum_voxels=500_000,
        ),
    )
    report = evaluate_surface_quality(ledger, surface, quality_config, mesh=tsdf.mesh)

    assert tsdf.protected_truncation_distance_m < tsdf.front.truncation_distance_m + 1e-12
    assert tsdf.protected_truncation_distance_m < 0.5 * fused.median_thickness_m
    assert tsdf.front.voxel_count > 0 and tsdf.back.voxel_count > 0
    assert len(tsdf.mesh.triangles) > 0
    assert report.completion_fraction > 0.5
    assert set(report.edge_completion) == {
        SurfaceRegion.LEADING_EDGE,
        SurfaceRegion.TRAILING_EDGE,
        SurfaceRegion.ROOT,
        SurfaceRegion.TIP,
    }


def test_coarse_model_workflow_writes_checksums_and_forbids_motion(tmp_path) -> None:
    planning = _planning_config()
    fusion = _fusion_config()
    partition = _partition_config()
    tsdf = TSDFConfig(
        voxel_size_m=0.003,
        truncation_distance_m=0.010,
        maximum_voxels=500_000,
        use_open3d_if_available=False,
    )
    quality = SurfaceQualityConfig(
        maximum_surface_distance_m=0.006,
        completed_fraction=0.70,
        maximum_rmse_m=0.004,
        minimum_normal_consistency=0.60,
        minimum_observed_points=8,
    )
    left_rectified_t_left_ir = _non_identity_rectification()
    result = build_coarse_blade_model(
        _reconstructed_blade_views(left_rectified_t_left_ir),
        _intrinsics(),
        fusion,
        partition,
        planning,
        tsdf,
        quality,
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
    source_dirs = []
    for view in _blade_views():
        source = tmp_path / view.view_id
        source.mkdir()
        (source / "metadata.json").write_text("{}\n", encoding="utf-8")
        source_dirs.append(source)
    output = write_coarse_model(
        tmp_path / "coarse", result, settings, source_views=tuple(source_dirs)
    )
    stored = read_coarse_model_summary(output)

    assert stored.metadata["motion_authorized"] is False
    assert stored.metadata["schema_version"] == 5
    assert stored.metadata["surface"]["angle_boundary_counts"][0] > 0
    assert stored.metadata["surface"]["parameterization_methods"] == [
        "boundary_curves",
        "boundary_curves",
    ]
    assert len(stored.metadata["surface"]["boundary_models"]) == 2
    assert stored.metadata["surface"]["footprint_source"] == "configured_override"
    assert stored.metadata["view_plan"]["candidates"]
    assert np.allclose(
        stored.metadata["view_plan"]["left_rectified_T_left_ir"],
        left_rectified_t_left_ir.matrix,
    )
    assert "candidate_base_T_left_rectified" in stored.metadata["files"]
    assert all(
        "base_T_left_rectified" in item and "base_T_left_ir" in item
        for item in stored.metadata["view_plan"]["candidates"]
    )
    assert all(
        0.15 <= item["standoff_distance_m"] <= 0.30
        for item in stored.metadata["view_plan"]["candidates"]
    )
    assert "boundary_control_points_m" in stored.metadata["files"]
    assert stored.metadata["quality"]["mesh_triangle_count"] > 0

    inspection_filter = settings.view_filter.model_copy(
        update={
            "duplicate_translation_tolerance_m": 0.0001,
            "duplicate_rotation_tolerance_deg": 0.1,
        }
    )
    inspection = inspect_fine_plan(stored, inspection_filter)
    assert inspection.geometry_passed is True
    assert inspection.robot_feasibility == "unverified"
    assert all(item.accepted for item in inspection.views)
    assert np.allclose(
        inspection.views[0].base_t_left_ir,
        stored.metadata["view_plan"]["candidates"][0]["base_T_left_ir"],
    )
    assert not np.allclose(
        inspection.views[0].base_t_left_ir[:3, :3],
        np.asarray(stored.metadata["view_plan"]["candidates"][0]["base_T_left_rectified"])[:3, :3],
    )
    inspection_path = write_fine_plan_inspection(tmp_path / "inspection", inspection)
    stored_inspection = read_fine_plan_inspection(inspection_path)
    assert stored_inspection.metadata["motion_authorized"] is False
    assert stored_inspection.metadata["geometry_passed"] is True
    assert stored_inspection.metadata["robot_feasibility"] == "unverified"
    assert set(stored_inspection.metadata["files"]) == {
        "views.csv",
        "patches.ply",
        "view_frusta.obj",
        "overview.svg",
    }
    gui_payload, gui_points, gui_colors, gui_labels = _inspection_scene(inspection_path)
    assert gui_payload["geometry_passed"] is True
    assert gui_points.shape == gui_colors.shape == (len(gui_labels), 3)

    mesh_path = output / "mesh_triangles.npy"
    mesh_path.write_bytes(mesh_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_coarse_model_summary(output)
