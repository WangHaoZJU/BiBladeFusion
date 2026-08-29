from dataclasses import replace

import numpy as np
import pytest

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
        return ReachabilityResult(
            ReachabilityState.REACHABLE,
            "offline IK solution found",
            np.zeros(6),
        )


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


def test_fine_scan_can_preserve_distinct_patch_candidates_at_same_pose() -> None:
    original = candidates()
    duplicate_patch = replace(
        original[0].patch,
        patch_id=original[1].patch.patch_id,
        row=original[1].patch.row,
        column=original[1].patch.column,
    )
    duplicate_pose = replace(
        original[0],
        view_id=original[1].view_id,
        patch=duplicate_patch,
    )
    result = filter_candidate_views(
        (original[0], duplicate_pose),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.01),
        AlwaysReachable(),
        deduplicate=False,
    )

    assert len(result.endpoint_feasible) == 2
    assert result.duplicate_view_ids == ()


def test_reachable_result_requires_concrete_joint_solution() -> None:
    with pytest.raises(ValueError, match="requires a concrete joint solution"):
        ReachabilityResult(ReachabilityState.REACHABLE, "missing joints")


def test_ik_failure_never_marks_endpoint_feasible() -> None:
    result = filter_candidate_views(
        candidates(),
        make_proxy(),
        ViewFilterConfig(workspace=workspace(), camera_clearance_radius_m=0.01),
        FailingChecker(),
    )

    assert all(item.status is CandidateStatus.GEOMETRY_ONLY for item in result.candidates)
    assert "solver unavailable" in " ".join(result.candidates[0].reasons)


def test_rectified_projection_pose_drives_geometry_but_raw_pose_drives_ik() -> None:
    original = candidates()[0]
    raw_rotation = original.base_t_left_ir.rotation @ np.diag([-1.0, 1.0, -1.0])
    raw_pose = PoseSE3.from_rotation_translation(
        "base",
        original.base_t_left_ir.child_frame,
        raw_rotation,
        original.base_t_left_ir.translation_m,
    )
    candidate = replace(original, base_t_left_ir=raw_pose)
    rectified_pose = PoseSE3(
        "base",
        "left_rectified",
        original.base_t_left_ir.matrix,
    )
    config = ViewFilterConfig(
        workspace=workspace(),
        camera_clearance_radius_m=0.01,
    )

    raw_geometry = filter_candidate_views(
        (candidate,),
        make_proxy(),
        config,
        AlwaysReachable(),
    )
    rectified_geometry = filter_candidate_views(
        (candidate,),
        make_proxy(),
        config,
        AlwaysReachable(),
        projection_poses={candidate.view_id: rectified_pose},
    )

    assert raw_geometry.candidates[0].status is CandidateStatus.REJECTED
    assert rectified_geometry.candidates[0].status is CandidateStatus.ENDPOINT_FEASIBLE
