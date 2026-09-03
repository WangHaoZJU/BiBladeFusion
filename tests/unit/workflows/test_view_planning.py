import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AdaptiveIkViewSearchConfig,
    AxisAlignedBoxConfig,
    CoarseReachabilityFallbackConfig,
    PointCloudConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    BladeSide,
    CandidateStatus,
    ReachabilityResult,
    ReachabilityState,
)
from biblade_fusion.workflows import InitialObservation, plan_initial_observation


class BackNormalRejectingChecker:
    def check(self, pose: PoseSE3) -> ReachabilityResult:
        if pose.translation_m[2] < -0.09:
            return ReachabilityResult(ReachabilityState.UNREACHABLE, "normal back unreachable")
        return ReachabilityResult(
            ReachabilityState.REACHABLE,
            "reachable",
            np.zeros(6),
        )


class RollBranchChecker:
    def check(self, pose: PoseSE3) -> ReachabilityResult:
        if abs(float(pose.rotation[1, 0])) < 0.5:
            return ReachabilityResult(
                ReachabilityState.UNREACHABLE,
                "nominal wrist branch unavailable",
            )
        return ReachabilityResult(
            ReachabilityState.REACHABLE,
            "rolled wrist branch reachable",
            np.full(6, 0.2),
        )


def test_initial_planning_replaces_unreachable_normal_with_oblique_fallback() -> None:
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.1, 0.1, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )
    observation = InitialObservation(
        "seed",
        CameraIntrinsics(101, 101, 50, 50, 50, 50, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        PointCloud(
            "base",
            np.zeros((3, 3)),
            np.array([[0, 0], [1, 0], [2, 0]]),
            (101, 101),
        ),
        proxy,
    )
    planning = ViewPlanningConfig(
        standoff_distance_m=0.1,
        minimum_standoff_distance_m=0.08,
        maximum_standoff_distance_m=0.12,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
        coarse_reachability_fallbacks=(
            CoarseReachabilityFallbackConfig(
                distance_offset_m=0.0,
                tilt_deg=60.0,
                azimuth_deg=0.0,
            ),
        ),
    )
    filtering = ViewFilterConfig(
        workspace=AxisAlignedBoxConfig(
            name="test",
            minimum_m=(-1.0, -1.0, -1.0),
            maximum_m=(1.0, 1.0, 1.0),
        ),
        camera_clearance_radius_m=0.01,
        minimum_incidence_cosine=0.4,
    )

    result = plan_initial_observation(
        observation,
        planning,
        filtering,
        BackNormalRejectingChecker(),
    )

    back = tuple(
        item
        for item in result.filtered_plan.candidates
        if item.candidate.patch.side is BladeSide.BACK
    )
    assert len(back) == 1
    assert back[0].status is CandidateStatus.ENDPOINT_FEASIBLE
    assert back[0].candidate.view_id.endswith("_fallback_01")
    assert back[0].candidate.projection_fraction == pytest.approx(0.5)


def test_initial_planning_selects_adaptive_ik_pose_and_records_search_trace() -> None:
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.1, 0.1, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )
    observation = InitialObservation(
        "seed",
        CameraIntrinsics(101, 101, 50, 50, 50, 50, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        PointCloud(
            "base",
            np.zeros((3, 3)),
            np.array([[0, 0], [1, 0], [2, 0]]),
            (101, 101),
        ),
        proxy,
    )
    planning = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
        adaptive_ik_view_search=AdaptiveIkViewSearchConfig(
            enabled=True,
            maximum_distance_expansions=0,
            tilt_samples_deg=(0.0,),
            azimuth_samples_deg=(0.0,),
            roll_samples_deg=(0.0, 45.0),
            maximum_ik_feasible_candidates=1,
        ),
    )
    filtering = ViewFilterConfig(
        workspace=AxisAlignedBoxConfig(
            name="empirical_camera_centres",
            minimum_m=(-0.05, -0.05, -0.05),
            maximum_m=(0.05, 0.05, 0.05),
        ),
        camera_clearance_radius_m=0.01,
    )

    result = plan_initial_observation(
        observation,
        planning,
        filtering,
        RollBranchChecker(),
        PointCloudConfig(minimum_depth_m=0.05, maximum_depth_m=0.5),
    )

    assert result.adaptive_trace is not None
    assert len(result.adaptive_trace.results) == 2
    assert len(result.geometric_plan.candidates) == 2
    assert len(result.filtered_plan.endpoint_feasible) == 2
    assert all(
        item.candidate.distance_policy == "adaptive_ik_aware_pose_family_v1"
        for item in result.filtered_plan.endpoint_feasible
    )
    assert all(
        "advisory workspace" in " ".join(item.reasons)
        for item in result.filtered_plan.endpoint_feasible
    )
    assert result.motion_authorized is False
