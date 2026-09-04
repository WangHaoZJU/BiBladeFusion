from __future__ import annotations

import numpy as np

from biblade_fusion.robotics.holorobot_joint_planner import (
    HoloRobotJointPlanStatus,
    HoloRobotOmplConfig,
    _anchor_ompl_solution_endpoints,
    plan_holorobot_rrtconnect,
    resample_joint_path,
)


def test_resample_joint_path_bounds_every_joint_step() -> None:
    path = resample_joint_path(
        ((0.0,) * 6, (0.11, -0.05, 0.0, 0.0, 0.0, 0.0)),
        maximum_joint_step_rad=0.02,
    )

    assert path[0] == (0.0,) * 6
    assert path[-1] == (0.11, -0.05, 0.0, 0.0, 0.0, 0.0)
    assert np.max(np.abs(np.diff(np.asarray(path), axis=0))) <= 0.02 + 1e-12


def test_ompl_solution_endpoints_are_reanchored_to_exact_robot_targets() -> None:
    start = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
    goal = (0.7, -0.8, 0.9, -1.0, 1.1, -1.2)
    raw = (
        tuple(value + 1e-15 for value in start),
        (0.4, -0.5, 0.6, -0.7, 0.8, -0.9),
        tuple(value - 1e-15 for value in goal),
    )

    anchored, start_drift, goal_drift = _anchor_ompl_solution_endpoints(
        raw,
        start=start,
        goal=goal,
    )

    assert anchored[0] == start
    assert anchored[-1] == goal
    assert anchored[1] == raw[1]
    assert 0.0 < start_drift < 1e-12
    assert 0.0 < goal_drift < 1e-12


def test_rrtconnect_rejects_invalid_goal_before_importing_ompl() -> None:
    goal = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)

    def state_validity(configuration: tuple[float, ...]):
        if configuration == goal:
            return False, ("environment_occupancy_unknown:wrist",)
        return True, ()

    result = plan_holorobot_rrtconnect(
        (0.0,) * 6,
        goal,
        joint_limits_rad=((-3.14, 3.14),) * 6,
        state_validity=state_validity,
        config=HoloRobotOmplConfig(),
    )

    assert result.status is HoloRobotJointPlanStatus.COLLISION_AT_GOAL
    assert result.blocking_reasons == ("environment_occupancy_unknown:wrist",)
