from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot import ServoJStreamConfig, StreamServoJResult
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import RobotCommandError
from biblade_fusion.robotics import (
    Cs68PinocchioCollisionChecker,
    GuardedEliteExecutor,
    preflight_linear_joint_motion,
)


@dataclass
class FakeGuardedArm:
    joint_positions_rad: np.ndarray = field(default_factory=lambda: np.zeros(6))
    prepared: bool = False
    streamed: bool = False
    stopped: bool = False

    def read_state(self) -> RobotState:
        return RobotState(
            monotonic_time_ns=1,
            controller_time_s=1.0,
            joint_positions_rad=self.joint_positions_rad,
            base_t_tcp=PoseSE3.identity("base", "tcp"),
            robot_mode="IDLE",
            safety_status="NORMAL",
            speed_scaling=0.3,
        )

    def prepare_servoj_stream(
        self, *, dt_s: float, warmup_duration_s: float = 0.0
    ) -> None:
        assert dt_s == 0.004
        assert warmup_duration_s == 0.0
        self.prepared = True

    def stream_servoj(self, stream, *, config, tracking_samples=None):
        assert config.dt_s == stream.dt_s
        assert tracking_samples is None
        self.streamed = True
        return StreamServoJResult(ok=True, commands_sent=len(stream.commands))

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(scope="module")
def checker() -> Cs68PinocchioCollisionChecker:
    return Cs68PinocchioCollisionChecker.from_resources()


@pytest.fixture(scope="module")
def clear_preflight(checker):
    return preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.03, -0.02, 0.01, 0.0, 0.0, 0.0),
        collision_checker=checker,
    )


def test_authorization_requires_exact_preflight_bound_confirmation(
    checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker)

    with pytest.raises(RobotCommandError, match="confirmation mismatch"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation="EXECUTE",
        )

    prompt = executor.approval_prompt(clear_preflight)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=prompt,
    )
    assert permit.preflight_fingerprint.startswith(prompt.removeprefix("EXECUTE "))


def test_execute_revalidates_and_consumes_one_shot_permit(
    checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    result = executor.execute(
        clear_preflight,
        permit,
        stream_config=ServoJStreamConfig(
            dt_s=0.004,
            tracking_check_every_n_commands=99,
        ),
    )

    assert result.ok is True
    assert arm.prepared is True
    assert arm.streamed is True
    with pytest.raises(RobotCommandError, match="already consumed"):
        executor.execute(clear_preflight, permit)


def test_live_start_mismatch_blocks_before_driver_prepare(
    checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(joint_positions_rad=np.full(6, 0.02))
    executor = GuardedEliteExecutor(arm, checker, live_start_tolerance_rad=0.01)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="no longer matches"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_expired_permit_is_consumed_without_motion(checker, clear_preflight) -> None:
    clock = {"now": 10.0}
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        permit_lifetime_s=1.0,
        clock=lambda: clock["now"],
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    clock["now"] = 12.0

    with pytest.raises(RobotCommandError, match="expired"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
