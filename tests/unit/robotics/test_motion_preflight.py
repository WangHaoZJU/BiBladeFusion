from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.robotics import (
    Cs68KinematicModel,
    Cs68PinocchioCollisionChecker,
    MotionPreflightStatus,
    preflight_linear_joint_motion,
)


@pytest.fixture(scope="module")
def checker() -> Cs68PinocchioCollisionChecker:
    return Cs68PinocchioCollisionChecker.from_resources()


def test_preflight_fails_closed_without_collision_checker() -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=None,
    )

    assert report.status is MotionPreflightStatus.CHECKER_UNAVAILABLE
    assert report.blocking_reasons == ("checker_unavailable",)
    assert report.servoj_stream is None
    assert report.motion_authorized is False


def test_clear_preflight_builds_velocity_limited_servoj_stream(checker) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.05, -0.04, 0.03, -0.02, 0.01, 0.0),
        collision_checker=checker,
        maximum_joint_step_rad=0.02,
        servoj_dt_s=0.004,
        speed_scaling=0.08,
        velocity_margin=0.8,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.ready_for_approval is True
    assert report.motion_authorized is False
    assert report.servoj_stream is not None
    assert report.servoj_stream.commands[0] == (0.0,) * 6
    np.testing.assert_allclose(
        report.servoj_stream.commands[-1],
        [0.05, -0.04, 0.03, -0.02, 0.01, 0.0],
    )
    maximum_velocity = np.asarray(
        Cs68KinematicModel.from_resources().joint_velocity_limits_rad_s()
    )
    commands = np.asarray(report.servoj_stream.commands)
    observed_velocity = np.max(
        np.abs(np.diff(commands, axis=0)) / report.servoj_stream.dt_s,
        axis=0,
    )
    assert np.all(observed_velocity <= maximum_velocity * 0.08 * 0.8 + 1e-12)


def test_folded_goal_is_blocked_before_trajectory_generation(checker) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.0, -3.0, 3.0, -3.0, 0.0, 0.0),
        collision_checker=checker,
        maximum_joint_step_rad=0.1,
    )

    assert report.status is MotionPreflightStatus.BLOCKED
    assert report.ready_for_approval is False
    assert report.servoj_stream is None
    assert any(reason.startswith("self_collision:") for reason in report.blocking_reasons)


def test_invalid_joint_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="six-vector"):
        preflight_linear_joint_motion(
            (0.0,) * 5,
            (0.0,) * 6,
            collision_checker=None,
        )
