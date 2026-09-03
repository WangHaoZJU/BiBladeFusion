"""Offline bilateral view-plan generation workflow."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.core.settings import (
    PointCloudConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.planning import (
    AdaptiveViewSearchConfig,
    AdaptiveViewSearchResult,
    BilateralViewPlan,
    CandidateStatus,
    FilteredViewPlan,
    ReachabilityChecker,
    filter_candidate_views,
    generate_bilateral_view_plan,
    generate_oblique_coarse_fallback,
    search_adaptive_candidate_family,
)
from biblade_fusion.workflows.initialization import InitialObservation


@dataclass(frozen=True, slots=True)
class AdaptiveViewPlanTrace:
    """Auditable non-motion search evidence attached to one proxy plan."""

    config: AdaptiveViewSearchConfig
    current_joint_positions_rad: tuple[float, float, float, float, float, float]
    results: tuple[AdaptiveViewSearchResult, ...]


@dataclass(frozen=True, slots=True)
class OfflineViewPlanningResult:
    geometric_plan: BilateralViewPlan
    filtered_plan: FilteredViewPlan
    adaptive_trace: AdaptiveViewPlanTrace | None = None

    @property
    def motion_authorized(self) -> bool:
        return False


def plan_initial_observation(
    observation: InitialObservation,
    planning_config: ViewPlanningConfig,
    filter_config: ViewFilterConfig,
    reachability_checker: ReachabilityChecker | None = None,
    point_cloud_config: PointCloudConfig | None = None,
) -> OfflineViewPlanningResult:
    """Partition both proxy faces and endpoint-filter candidates without moving hardware."""

    geometric = generate_bilateral_view_plan(
        observation.proxy,
        observation.planning_intrinsics,
        planning_config,
    )
    adaptive_policy = planning_config.adaptive_ik_view_search
    # Adaptive search already begins at the nominal pose.  Geometry-only filtering
    # here avoids solving that same IK once before, then again inside its family.
    filtered = filter_candidate_views(
        geometric.candidates,
        observation.proxy,
        filter_config,
        None if adaptive_policy.enabled else reachability_checker,
    )
    adaptive_trace = None
    if adaptive_policy.enabled:
        if reachability_checker is None:
            raise ValueError("Adaptive IK view search requires an endpoint IK checker")
        if point_cloud_config is None:
            raise ValueError("Adaptive IK view search requires physical point-cloud limits")
        search_config = AdaptiveViewSearchConfig(
            minimum_optical_distance_m=point_cloud_config.minimum_depth_m,
            maximum_optical_distance_m=point_cloud_config.maximum_depth_m,
            distance_step_m=adaptive_policy.distance_step_m,
            maximum_distance_expansions=adaptive_policy.maximum_distance_expansions,
            tilt_samples_deg=adaptive_policy.tilt_samples_deg,
            azimuth_samples_deg=adaptive_policy.azimuth_samples_deg,
            roll_samples_deg=adaptive_policy.roll_samples_deg,
            maximum_generated_candidates=adaptive_policy.maximum_generated_candidates,
            # A regular surface patch needs one feasible endpoint; requesting all
            # eight alternatives multiplied IK work without changing the plan.
            maximum_ik_feasible_candidates=1,
            maximum_ik_attempts_per_family=(
                adaptive_policy.maximum_ik_attempts_per_family
            ),
            maximum_search_duration_s=adaptive_policy.maximum_search_duration_s,
        )
        geometric_candidates = list(geometric.candidates)
        evaluations = list(filtered.candidates)
        searches = []
        for index, nominal in enumerate(geometric.candidates):
            search = search_adaptive_candidate_family(
                nominal,
                observation.proxy,
                filter_config,
                (reachability_checker,),
                observation.seed_joint_positions_rad,
                search_config,
            )
            searches.append(search)
            if search.recommended is not None:
                geometric_candidates[index] = search.recommended.evaluated.candidate
                evaluations[index] = search.recommended.evaluated
        geometric = BilateralViewPlan(
            tuple(geometric_candidates),
            geometric.rows,
            geometric.columns,
            geometric.footprint_m,
            geometric.effective_surface_extents_m,
        )
        filtered = FilteredViewPlan(tuple(evaluations), ())
        adaptive_trace = AdaptiveViewPlanTrace(
            search_config,
            tuple(float(value) for value in observation.seed_joint_positions_rad),
            tuple(searches),
        )
    elif planning_config.coarse_reachability_fallbacks and reachability_checker is not None:
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
    return OfflineViewPlanningResult(geometric, filtered, adaptive_trace)
