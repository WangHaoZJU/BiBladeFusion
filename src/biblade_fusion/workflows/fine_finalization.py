"""Replayable terminal reconstruction for a completed fine-scan lineage.

Fine reference coverage is only a trigger for this workflow.  Completion is
granted only after every foreground-bound fine view is replayed into a new
bilateral fusion/TSDF result and the resulting mesh passes explicit topology,
bilateral, fin, and surface-quality gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biblade_fusion.core.settings import (
    FineFinalizationConfig,
    MultiViewFusionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
)
from biblade_fusion.perception.fusion import (
    FusedBladeCloud,
    RegisteredCloudView,
    fuse_registered_views,
)
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.perception.tsdf import BilateralTSDFResult, integrate_bilateral_tsdf
from biblade_fusion.planning.surface_coverage import (
    SurfaceQualityReport,
    evaluate_surface_quality,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.storage.reconstructed_view import (
    SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION,
    read_reconstructed_view,
)
from biblade_fusion.storage.surface_coverage import (
    StoredSurfaceCoverageGeneration,
    read_surface_coverage_generation,
)
from biblade_fusion.workflows.coarse_model import registered_cloud_view


class FineFinalizationError(ValueError):
    """The fine lineage cannot prove a terminal reconstruction."""


@dataclass(frozen=True, slots=True)
class FineFinalizationGateReport:
    required_patch_count: int
    complete_patch_count: int
    front_source_view_count: int
    back_source_view_count: int
    front_mesh_triangle_count: int
    back_mesh_triangle_count: int
    front_fin_count: int
    back_fin_count: int
    mesh_boundary_edge_count: int
    mesh_boundary_loop_count: int
    mesh_watertight: bool
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class FinalFineReconstruction:
    coverage: StoredSurfaceCoverageGeneration
    source_view_roots: tuple[Path, ...]
    registered_views: tuple[RegisteredCloudView, ...]
    fused_cloud: FusedBladeCloud
    tsdf: BilateralTSDFResult
    quality: SurfaceQualityReport
    gates: FineFinalizationGateReport

    @property
    def motion_authorized(self) -> bool:
        return False


def _lineage_source_views(
    terminal: StoredSurfaceCoverageGeneration,
) -> tuple[tuple[Path, ...], tuple[RegisteredCloudView, ...]]:
    roots: list[Path] = []
    current = terminal
    while current.previous_generation_path is not None:
        if current.current_reconstructed_view_path is None:
            raise FineFinalizationError(
                "Fine coverage successor lacks its reconstructed observation"
            )
        roots.append(current.current_reconstructed_view_path)
        current = read_surface_coverage_generation(
            current.previous_generation_path,
            require_foreground_bound_science=True,
        )
    if current.current_reconstructed_view_path is not None:
        raise FineFinalizationError(
            "Initial fine coverage unexpectedly contains a reconstructed observation"
        )
    roots.reverse()
    views: list[RegisteredCloudView] = []
    ids: list[str] = []
    identities: set[tuple[str, int, int]] = set()
    for root in roots:
        stored = read_reconstructed_view(root)
        if (
            int(stored.metadata["schema_version"])
            != SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION
        ):
            raise FineFinalizationError(
                f"Final fine source is not a foreground-bound schema-3 view: {root}"
            )
        view = stored.view
        identity = (
            view.source_view_id,
            view.source_sequence_index,
            view.source_frame_number,
        )
        if identity in identities:
            raise FineFinalizationError("Final fine source identity is duplicated")
        identities.add(identity)
        ids.append(view.source_view_id)
        views.append(registered_cloud_view(view))
    if tuple(ids) != terminal.ledger.observation_ids:
        raise FineFinalizationError(
            "Final fine source order does not exactly match the coverage ledger"
        )
    if not views:
        raise FineFinalizationError("Final fine reconstruction has no measured source views")
    return tuple(roots), tuple(views)


def _gate_report(
    terminal: StoredSurfaceCoverageGeneration,
    views: tuple[RegisteredCloudView, ...],
    fused: FusedBladeCloud,
    tsdf: BilateralTSDFResult,
    quality: SurfaceQualityReport,
    config: FineFinalizationConfig,
) -> FineFinalizationGateReport:
    violations: list[str] = []
    quality_by_id = {item.patch_id: item for item in quality.patches}
    required = terminal.required_patch_ids
    complete_count = sum(
        bool(quality_by_id[patch_id].complete)
        for patch_id in required
        if patch_id in quality_by_id
    )
    if set(quality_by_id) != set(required) or complete_count != len(required):
        violations.append("not every fixed-reference surface patch passed quality gates")

    view_sides = tuple(
        1
        if (view.camera_origin_m - fused.center_m) @ fused.axes[:, 2] >= 0.0
        else -1
        for view in views
    )
    front_views = view_sides.count(1)
    back_views = view_sides.count(-1)
    if front_views < config.minimum_source_views_per_side:
        violations.append("too few independently registered front-side source views")
    if back_views < config.minimum_source_views_per_side:
        violations.append("too few independently registered back-side source views")

    mesh_sides = tsdf.mesh.triangle_sides
    front_triangles = int(np.count_nonzero(mesh_sides == 1))
    back_triangles = int(np.count_nonzero(mesh_sides == -1))
    if front_triangles < config.minimum_mesh_triangles_per_side:
        violations.append("front TSDF surface has too few mesh triangles")
    if back_triangles < config.minimum_mesh_triangles_per_side:
        violations.append("back TSDF surface has too few mesh triangles")

    fin_by_side = {
        side: tuple(
            component
            for component in terminal.surface.fin_components
            if component.side is side
        )
        for side in (BladeSide.FRONT, BladeSide.BACK)
    }
    if config.require_two_face_fin_per_side:
        for side in (BladeSide.FRONT, BladeSide.BACK):
            components = fin_by_side[side]
            if len(components) != 1:
                violations.append(
                    f"{side.value} reference does not contain exactly one measured fin"
                )
            elif not components[0].two_faces_observed:
                violations.append(
                    f"{side.value} fin does not contain two independently observed faces"
                )

    if config.require_fin_regions_complete:
        fin_regions = (
            SurfaceRegion.FIN_FACE,
            SurfaceRegion.FIN_ROOT,
            SurfaceRegion.FIN_FREE_EDGE,
        )
        for side in (BladeSide.FRONT, BladeSide.BACK):
            for region in fin_regions:
                region_quality = tuple(
                    item
                    for item in quality.patches
                    if item.side is side and item.region is region
                )
                if not region_quality:
                    violations.append(
                        f"{side.value} final surface lacks required {region.value} patches"
                    )
                elif not all(item.complete for item in region_quality):
                    violations.append(
                        f"{side.value} {region.value} patches are incomplete"
                    )

    if quality.mesh_boundary_edge_count > config.maximum_mesh_boundary_edges:
        violations.append("final mesh boundary-edge count exceeds the hole gate")
    if quality.mesh_boundary_loop_count > config.maximum_mesh_boundary_loops:
        violations.append("final mesh boundary-loop count exceeds the hole gate")
    if config.require_watertight_mesh and not quality.mesh_watertight:
        violations.append("final mesh is not watertight")

    return FineFinalizationGateReport(
        len(required),
        complete_count,
        front_views,
        back_views,
        front_triangles,
        back_triangles,
        len(fin_by_side[BladeSide.FRONT]),
        len(fin_by_side[BladeSide.BACK]),
        quality.mesh_boundary_edge_count,
        quality.mesh_boundary_loop_count,
        quality.mesh_watertight,
        tuple(violations),
    )


def build_final_fine_reconstruction(
    coverage_generation: str | Path,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
) -> FinalFineReconstruction:
    """Replay a fine lineage and build its terminal bilateral reconstruction."""

    terminal = read_surface_coverage_generation(
        coverage_generation,
        require_foreground_bound_science=True,
    )
    if terminal.quality_config.model_dump(mode="json") != (
        surface_quality_config.model_dump(mode="json")
    ):
        raise FineFinalizationError(
            "Terminal fine coverage uses a different surface-quality policy"
        )
    if terminal.quality.incomplete_patch_ids:
        raise FineFinalizationError(
            "Fine coverage is incomplete; final multi-view reconstruction is premature"
        )
    roots, views = _lineage_source_views(terminal)
    try:
        fused = fuse_registered_views(views, fusion_config)
        feature_thicknesses = tuple(
            float(component.face_separation_m)
            for component in terminal.surface.fin_components
            if component.two_faces_observed and component.face_separation_m > 0.0
        )
        tsdf = integrate_bilateral_tsdf(
            fused,
            views,
            tsdf_config,
            feature_thicknesses_m=feature_thicknesses,
        )
        quality = evaluate_surface_quality(
            terminal.ledger,
            terminal.surface,
            surface_quality_config,
            mesh=tsdf.mesh,
        )
    except (TypeError, ValueError) as exc:
        raise FineFinalizationError(
            f"Final multi-view fusion/TSDF reconstruction failed: {exc}"
        ) from exc
    gates = _gate_report(
        terminal,
        views,
        fused,
        tsdf,
        quality,
        finalization_config,
    )
    if not gates.passed:
        raise FineFinalizationError(
            "Final reconstruction failed terminal gates: " + "; ".join(gates.violations)
        )
    return FinalFineReconstruction(
        terminal,
        roots,
        views,
        fused,
        tsdf,
        quality,
        gates,
    )
