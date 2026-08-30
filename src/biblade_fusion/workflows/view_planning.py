"""Offline bilateral view-plan generation workflow."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.core.settings import ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.planning import (
    BilateralViewPlan,
    CandidateStatus,
    FilteredViewPlan,
    ReachabilityChecker,
    filter_candidate_views,
    generate_bilateral_view_plan,
    generate_oblique_coarse_fallback,
)
from biblade_fusion.workflows.initialization import InitialObservation


@dataclass(frozen=True, slots=True)
class OfflineViewPlanningResult:
    geometric_plan: BilateralViewPlan
    filtered_plan: FilteredViewPlan

    @property
    def motion_authorized(self) -> bool:
        return False


def plan_initial_observation(
    observation: InitialObservation,
    planning_config: ViewPlanningConfig,
    filter_config: ViewFilterConfig,
    reachability_checker: ReachabilityChecker | None = None,
) -> OfflineViewPlanningResult:
    """Partition both proxy faces and endpoint-filter candidates without moving hardware."""

    geometric = generate_bilateral_view_plan(
        observation.proxy,
        observation.planning_intrinsics,
        planning_config,
    )
    filtered = filter_candidate_views(
        geometric.candidates,
        observation.proxy,
        filter_config,
        reachability_checker,
    )
    if planning_config.coarse_reachability_fallbacks and reachability_checker is not None:
        geometric_candidates = list(geometric.candidates)
        evaluations = list(filtered.candidates)
        for index, evaluation in enumerate(filtered.candidates):
            if evaluation.status is not CandidateStatus.REJECTED:
                continue
            for fallback_index, fallback in enumerate(
                planning_config.coarse_reachability_fallbacks,
                start=1,
            ):
                candidate = generate_oblique_coarse_fallback(
                    evaluation.candidate,
                    fallback,
                    view_id=f"{evaluation.candidate.view_id}_fallback_{fallback_index:02d}",
                )
                checked = filter_candidate_views(
                    (candidate,),
                    observation.proxy,
                    filter_config,
                    reachability_checker,
                    deduplicate=False,
                ).candidates[0]
                if checked.status is CandidateStatus.ENDPOINT_FEASIBLE:
                    geometric_candidates[index] = candidate
                    evaluations[index] = checked
                    break
        geometric = BilateralViewPlan(
            tuple(geometric_candidates),
            geometric.rows,
            geometric.columns,
            geometric.footprint_m,
            geometric.effective_surface_extents_m,
        )
        filtered = FilteredViewPlan(tuple(evaluations), filtered.duplicate_view_ids)
    return OfflineViewPlanningResult(geometric, filtered)
