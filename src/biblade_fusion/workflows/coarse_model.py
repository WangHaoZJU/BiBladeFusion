"""End-to-end offline coarse-model, fine-view, TSDF, and quality workflow."""

from __future__ import annotations

from dataclasses import dataclass

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


def build_coarse_blade_model(
    views: tuple[RegisteredCloudView, ...],
    planning_intrinsics: CameraIntrinsics,
    fusion_config: MultiViewFusionConfig,
    partition_config: SurfacePartitionConfig,
    planning_config: ViewPlanningConfig,
    tsdf_config: TSDFConfig,
    quality_config: SurfaceQualityConfig,
) -> CoarseModelResult:
    """Run the complete paper-derived offline chain on pose-registered coarse views."""

    fused = fuse_registered_views(views, fusion_config)
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
        surface, planning_intrinsics, planning_config, partition_config
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
        views,
        tsdf_config,
        feature_thicknesses_m=feature_thicknesses,
    )
    ledger = create_surface_coverage_ledger(surface)
    for view in views:
        ledger = update_surface_coverage(
            ledger, surface, view, f"coarse:{view.view_id}", quality_config
        )
    quality = evaluate_surface_quality(ledger, surface, quality_config, mesh=tsdf.mesh)
    return CoarseModelResult(fused, surface, plan, tsdf, ledger, quality)
