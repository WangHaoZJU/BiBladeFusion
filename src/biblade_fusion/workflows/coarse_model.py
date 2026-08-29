"""End-to-end offline coarse-model, fine-view, TSDF, and quality workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    MultiViewFusionConfig,
    SurfacePartitionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
    ViewPlanningConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.fusion import (
    FusedBladeCloud,
    RegisteredCloudView,
    fuse_registered_views,
)
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    CurvedViewPlan,
    derive_usable_footprint,
    generate_curved_view_plan,
    partition_curved_blade,
)
from biblade_fusion.perception.tsdf import BilateralTSDFResult, integrate_bilateral_tsdf
from biblade_fusion.planning.surface_coverage import (
    SurfaceCoverageLedger,
    SurfaceQualityReport,
    create_surface_coverage_ledger,
    evaluate_surface_quality,
    update_surface_coverage,
)
from biblade_fusion.workflows.reconstruction import ReconstructedBladeView


@dataclass(frozen=True, slots=True)
class CoarseModelResult:
    fused_cloud: FusedBladeCloud
    surface: CurvedBladeSurface
    view_plan: CurvedViewPlan
    tsdf: BilateralTSDFResult
    coverage: SurfaceCoverageLedger
    quality: SurfaceQualityReport

    @property
    def motion_authorized(self) -> bool:
        return False


def registered_cloud_view(view: ReconstructedBladeView) -> RegisteredCloudView:
    """Adapt an existing pose-registered depth artifact to the fusion contract."""

    intrinsics = view.planning_intrinsics
    intrinsic_matrix = (
        (intrinsics.fx, 0.0, intrinsics.cx),
        (0.0, intrinsics.fy, intrinsics.cy),
        (0.0, 0.0, 1.0),
    )
    return RegisteredCloudView(
        view.source_view_id,
        view.base_cloud.points_m,
        view.base_t_projection_camera.translation_m,
        view.base_cloud.pixel_uv,
        view.base_cloud.source_image_shape,
        intrinsic_matrix,
        view.base_t_projection_camera.matrix,
    )


def _validate_left_rectified_t_left_ir(pose: PoseSE3) -> None:
    if (pose.parent_frame, pose.child_frame) != ("left_rectified", "left_ir"):
        raise ValueError("Coarse-model planning requires left_rectified_T_left_ir")


def derive_consistent_left_rectified_t_left_ir(
    views: tuple[ReconstructedBladeView, ...],
) -> PoseSE3:
    """Derive one rectified-to-raw calibration from reconstructed stereo views."""

    if not views:
        raise ValueError("At least one reconstructed coarse view is required")
    calibrations: list[PoseSE3] = []
    for view in views:
        if view.depth_source != "foundation_stereo":
            raise ValueError(
                f"Coarse view {view.source_view_id} does not expose a left-rectified "
                "projection camera"
            )
        if (
            view.base_t_projection_camera.parent_frame,
            view.base_t_projection_camera.child_frame,
        ) != ("base", "left_rectified"):
            raise ValueError(f"Coarse view {view.source_view_id} requires base_T_left_rectified")
        calibration = view.base_t_projection_camera.inverse().compose(view.base_t_left_ir)
        _validate_left_rectified_t_left_ir(calibration)
        calibrations.append(calibration)
    reference = calibrations[0]
    for view, calibration in zip(views[1:], calibrations[1:], strict=True):
        if not np.allclose(
            calibration.matrix,
            reference.matrix,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Coarse views use inconsistent left_rectified_T_left_ir calibration: "
                f"{view.source_view_id} differs from {views[0].source_view_id}"
            )
    return reference


def build_coarse_blade_model(
    views: tuple[RegisteredCloudView, ...] | tuple[ReconstructedBladeView, ...],
    planning_intrinsics: CameraIntrinsics,
    fusion_config: MultiViewFusionConfig,
    partition_config: SurfacePartitionConfig,
    planning_config: ViewPlanningConfig,
    tsdf_config: TSDFConfig,
    quality_config: SurfaceQualityConfig,
    *,
    left_rectified_t_left_ir: PoseSE3 | None = None,
) -> CoarseModelResult:
    """Run the offline chain, deriving stereo calibration from reconstructed views."""

    if not views:
        raise ValueError("At least one coarse view is required")
    if all(isinstance(view, ReconstructedBladeView) for view in views):
        reconstructed_views = cast(tuple[ReconstructedBladeView, ...], views)
        derived_calibration = derive_consistent_left_rectified_t_left_ir(reconstructed_views)
        if left_rectified_t_left_ir is not None:
            _validate_left_rectified_t_left_ir(left_rectified_t_left_ir)
            if not np.allclose(
                left_rectified_t_left_ir.matrix,
                derived_calibration.matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Explicit left_rectified_T_left_ir disagrees with reconstructed views"
                )
        left_rectified_t_left_ir = derived_calibration
        if any(view.planning_intrinsics != planning_intrinsics for view in reconstructed_views):
            raise ValueError("Coarse views use different planning intrinsics")
        registered_views = tuple(registered_cloud_view(view) for view in reconstructed_views)
    elif all(isinstance(view, RegisteredCloudView) for view in views):
        registered_views = cast(tuple[RegisteredCloudView, ...], views)
        if left_rectified_t_left_ir is None:
            raise ValueError("RegisteredCloudView inputs require explicit left_rectified_T_left_ir")
        _validate_left_rectified_t_left_ir(left_rectified_t_left_ir)
    else:
        raise TypeError("Coarse views must not mix reconstructed and registered view types")

    fused = fuse_registered_views(registered_views, fusion_config)
    if partition_config.derive_footprint_from_intrinsics:
        footprint = derive_usable_footprint(planning_intrinsics, planning_config)
        footprint_source = "calibrated_intrinsics"
    else:
        footprint = partition_config.usable_footprint_m
        if footprint is None:
            raise ValueError(
                "usable_footprint_m is required when intrinsic footprint derivation is disabled"
            )
        footprint_source = "configured_override"
    surface = partition_curved_blade(
        fused,
        partition_config,
        usable_footprint_m=footprint,
        footprint_source=footprint_source,
    )
    plan = generate_curved_view_plan(
        surface,
        planning_intrinsics,
        planning_config,
        partition_config,
        left_rectified_t_left_ir=left_rectified_t_left_ir,
    )
    # Visibility-driven subdivision is part of the final fine partition and must
    # therefore be used by coverage, quality, and persisted artifacts.
    surface = plan.surface
    feature_thicknesses = tuple(
        component.face_separation_m
        for component in surface.fin_components
        if component.two_faces_observed
    )
    tsdf = integrate_bilateral_tsdf(
        fused,
        registered_views,
        tsdf_config,
        feature_thicknesses_m=feature_thicknesses,
    )
    ledger = create_surface_coverage_ledger(surface)
    for view in registered_views:
        ledger = update_surface_coverage(
            ledger, surface, view, f"coarse:{view.view_id}", quality_config
        )
    quality = evaluate_surface_quality(ledger, surface, quality_config, mesh=tsdf.mesh)
    return CoarseModelResult(fused, surface, plan, tsdf, ledger, quality)
