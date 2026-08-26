import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AxisAlignedBoxConfig, ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    CandidateStatus,
    ReachabilityResult,
    ReachabilityState,
    filter_candidate_views,
    generate_bilateral_view_plan,
)


def make_proxy(extents=(0.4, 0.2, 0.02)) -> BilateralBladeProxy:
    return BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.asarray(extents),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )


class AlwaysReachable:
    def check(self, base_t_left_ir):
        return ReachabilityResult(ReachabilityState.REACHABLE, "offline IK solution found")


class FailingChecker:
    def check(self, base_t_left_ir):
        raise RuntimeError("solver unavailable")


def candidates():
    intrinsics = CameraIntrinsics(101, 101, 50.0, 50.0, 50.0, 50.0, "none", ())
    plan = generate_bilateral_view_plan(
        make_proxy(),
        intrinsics,
        ViewPlanningConfig(
            standoff_distance_m=0.1,
            overlap_fraction=0.0,
            footprint_utilization=1.0,
            edge_margin_m=0.0,
        ),
    )
    return plan.candidates


def workspace() -> AxisAlignedBoxConfig:
    return AxisAlignedBoxConfig(name="cell", minimum_m=(-1, -1, -1), maximum_m=(1, 1, 1))


def test_missing_workspace_and_ik_stays_geometry_only() -> None:
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(camera_clearance_radius_m=0.05),
    )

    assert all(item.status is CandidateStatus.GEOMETRY_ONLY for item in result.accepted)
    assert not result.endpoint_feasible
    assert result.motion_authorized is False
    assert "workspace bounds" in " ".join(result.accepted[0].reasons)


def test_workspace_and_reachability_mark_endpoint_feasible() -> None:
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.05),
        AlwaysReachable(),
    )

    assert len(result.endpoint_feasible) == len(candidates())
    assert all(item.metrics.geometric_score == 1.0 for item in result.endpoint_feasible)
    assert result.motion_authorized is False


def test_forbidden_volume_rejects_front_candidates() -> None:
    forbidden = AxisAlignedBoxConfig(
        name="front_fixture",
        minimum_m=(-1, -1, 0.05),
        maximum_m=(1, 1, 0.2),
    )
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(
            workspace=workspace(),
            forbidden_volumes=(forbidden,),
            camera_clearance_radius_m=0.01,
        ),
        AlwaysReachable(),
    )

    front = [item for item in result.candidates if item.candidate.patch.side.value == "front"]
    back = [item for item in result.candidates if item.candidate.patch.side.value == "back"]
    assert all(item.status is CandidateStatus.REJECTED for item in front)
    assert all(item.status is CandidateStatus.ENDPOINT_FEASIBLE for item in back)


def test_clearance_sphere_rejects_camera_too_close_to_proxy() -> None:
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.11),
        AlwaysReachable(),
    )

    assert all(item.status is CandidateStatus.REJECTED for item in result.candidates)
    assert "clearance sphere" in " ".join(result.candidates[0].reasons)


def test_duplicate_candidate_pose_is_removed() -> None:
    original = candidates()
    result = filter_candidate_views(
        (original[0], original[0]),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.01),
        AlwaysReachable(),
    )

    assert len(result.candidates) == 1
    assert result.duplicate_view_ids == (original[0].view_id,)


def test_ik_failure_never_marks_endpoint_feasible() -> None:
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.01),
        FailingChecker(),
    )

    assert all(item.status is CandidateStatus.GEOMETRY_ONLY for item in result.candidates)
    assert "solver unavailable" in " ".join(result.candidates[0].reasons)
