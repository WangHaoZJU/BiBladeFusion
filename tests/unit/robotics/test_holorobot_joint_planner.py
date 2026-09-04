from __future__ import annotations

import numpy as np

from biblade_fusion.robotics.holorobot_joint_planner import (
    HoloRobotJointPlanStatus,
    HoloRobotOmplConfig,
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
