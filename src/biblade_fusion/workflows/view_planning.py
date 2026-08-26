"""Offline bilateral view-plan generation workflow."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.core.settings import ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.planning import (
    BilateralViewPlan,
    FilteredViewPlan,
    ReachabilityChecker,
    filter_candidate_views,
    generate_bilateral_view_plan,
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
        observation.left_intrinsics,
        planning_config,
    )
    filtered = filter_candidate_views(
        geometric.candidates,
        observation.proxy,
        filter_config,
        reachability_checker,
    )
    return OfflineViewPlanningResult(geometric, filtered)
