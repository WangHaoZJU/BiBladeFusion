from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot import (
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import (
    RobotCommandError,
    RobotMotionInterruptedError,
)
from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    Es68PinocchioCollisionChecker,
    GuardedEliteExecutor,
    OccupancyRobotCollisionChecker,
    preflight_linear_joint_motion,
)
from biblade_fusion.robotics.guarded_execution import EmergencyStopUnconfirmedError
from biblade_fusion.robotics.occupancy_collision import (
    _issue_occupancy_semantic_attestation,
)

_MOTION_ENVELOPE_ID = "6" * 64
_MOTION_ENVELOPE_METADATA_SHA256 = "7" * 64
_MOTION_ENVELOPE_RAD = (0.001,) * 6


class _SyntheticSweptEs68Checker(Es68PinocchioCollisionChecker):
    def check_path(self, *args, **kwargs):
        return replace(
            super().check_path(*args, **kwargs),
            continuous_swept_volume_verified=True,
        )


class _SyntheticContinuousOccupancyChecker(OccupancyRobotCollisionChecker):
    @property
    def continuous_swept_volume_supported(self) -> bool:
        return True

    def check_path(self, *args, **kwargs):
        report = super().check_path(*args, **kwargs)
        if report.status is not CollisionCheckStatus.CLEAR:
            return report
        return replace(
            report,
            continuous_swept_volume_verified=True,
            result=replace(
                report.result,
                diagnostics={
                    **report.result.diagnostics,
                    "continuous_swept_volume_verified": True,
                    "continuous_sweep_backend": "synthetic_test_only",
                },
            ),
        )


def _attested_occupancy_checker(
    checker,
    snapshot,
    provider,
    **kwargs,
) -> _SyntheticContinuousOccupancyChecker:
    attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256="e" * 64,
        snapshot=snapshot,
        robot_geometry_hash=checker.robot_geometry_hash,
    )
    return _SyntheticContinuousOccupancyChecker(
        checker,
        provider,
        semantic_attestation=attestation,
        accepted_joint_uncertainty_rad=_MOTION_ENVELOPE_RAD,
        motion_envelope_acceptance_id=_MOTION_ENVELOPE_ID,
        motion_envelope_metadata_sha256=_MOTION_ENVELOPE_METADATA_SHA256,
        **kwargs,
    )


def _changed_snapshot(snapshot):
    last_centre = snapshot.source_camera_centres_base_m[-1]
    return replace(
        snapshot,
        sequence=snapshot.sequence + 1,
        created_at_utc=snapshot.created_at_utc + timedelta(milliseconds=1),
        source_view_ids=(*snapshot.source_view_ids, "changed-view"),
        source_camera_centres_base_m=(
            *snapshot.source_camera_centres_base_m,
            (last_centre[0] + 0.03, last_centre[1], last_centre[2]),
        ),
        source_camera_axes_base=(
            *snapshot.source_camera_axes_base,
            snapshot.source_camera_axes_base[-1],
        ),
        content_hash="",
    )


@dataclass
class FakeGuardedArm:
    joint_positions_rad: np.ndarray = field(default_factory=lambda: np.zeros(6))
    recovered_joint_positions_rad: np.ndarray | None = None
    resumed: bool = False
    prepared: bool = False
    streamed: bool = False
    stopped: bool = True
    enabled: bool = True
    guarded_enable_calls: int = 0
    _stop_generation: int = 0
    events: list[str] = field(default_factory=list)
    resume_exception: BaseException | None = None
    prepare_exception: BaseException | None = None
    stream_exception: BaseException | None = None
    stop_exception: BaseException | None = None
    deadline_stop_exception: BaseException | None = None
    deadline_stop_calls: int = 0

    @property
    def stop_generation(self) -> int:
        return self._stop_generation

    @property
    def stop_snapshot(self) -> tuple[int, bool]:
        return self._stop_generation, self.stopped

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    def read_state(self) -> RobotState:
        self.events.append("read_state")
        return RobotState(
            monotonic_time_ns=1,
            controller_time_s=1.0,
            joint_positions_rad=self.joint_positions_rad,
            base_t_tcp=PoseSE3.identity("base", "tcp"),
            robot_mode="IDLE",
            safety_status="NORMAL",
            speed_scaling=0.3,
        )

    def _guarded_resume_servoj_control(
        self,
        *,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded=None,
    ) -> None:
        assert capability is not None
        assert deadline_exceeded is None or deadline_exceeded() is False
        self.events.append("resume")
        self.resumed = True
        if self.resume_exception is not None:
            raise self.resume_exception
        if self._stop_generation != expected_stop_generation:
            raise RobotCommandError("stop generation changed during fake recovery")
        self.stopped = False
        if self.recovered_joint_positions_rad is not None:
            self.joint_positions_rad = self.recovered_joint_positions_rad

    def _guarded_enable_for_servoj_control(
        self,
        *,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded=None,
    ) -> None:
        assert capability is not None
        assert deadline_exceeded is None or deadline_exceeded() is False
        self.events.append("guarded_enable")
        self.guarded_enable_calls += 1
        if self._stop_generation != expected_stop_generation or not self.stopped:
            raise RobotMotionInterruptedError("stop latch changed before guarded enable")
        self.enabled = True

    def _guarded_prepare_servoj_stream(
        self,
        *,
        dt_s: float,
        warmup_duration_s: float = 0.0,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded=None,
    ) -> None:
        assert capability is not None
        assert deadline_exceeded is None or deadline_exceeded() is False
        assert dt_s == 0.004
        assert warmup_duration_s == 0.2
        if self._stop_generation != expected_stop_generation or self.stopped:
            raise RobotMotionInterruptedError("stop generation changed before prepare")
        self.events.append("prepare")
        self.prepared = True
        if self.prepare_exception is not None:
            raise self.prepare_exception

    def _guarded_stream_servoj(
        self,
        stream,
        *,
        config,
        expected_stop_generation,
        capability,
        tracking_samples=None,
        deadline_exceeded=None,
    ):
        assert capability is not None
        assert deadline_exceeded is None or deadline_exceeded() is False
        assert config.dt_s == stream.dt_s
        assert tracking_samples is None
        if self._stop_generation != expected_stop_generation or self.stopped:
            raise RobotMotionInterruptedError("stop generation changed before stream")
        self.events.append("stream")
        self.streamed = True
        if self.stream_exception is not None:
            raise self.stream_exception
        return StreamServoJResult(ok=True, commands_sent=len(stream.commands))

    def stop(self) -> None:
        self.events.append("stop")
        self._stop_generation += 1
        self.stopped = True
        if self.stop_exception is not None:
            raise self.stop_exception

    def _guarded_deadline_stop(self, *, capability: object) -> None:
        assert capability is not None
        self.events.append("deadline_stop")
        self.deadline_stop_calls += 1
        self._stop_generation += 1
        self.stopped = True
        if self.deadline_stop_exception is not None:
            raise self.deadline_stop_exception


@pytest.fixture(scope="module")
def checker() -> Cs68PinocchioCollisionChecker:
    base = Cs68PinocchioCollisionChecker.from_resources()
    payload = {field.name: getattr(base, field.name) for field in fields(base)}
    payload.update(
        model_name="es68",
        collision_model_id="test-es68-d435i",
        collision_model_hash="1" * 64,
        robot_geometry_hash="2" * 64,
        motion_model_contract_hash="3" * 64,
        continuous_swept_volume_supported=True,
    )
    return _SyntheticSweptEs68Checker(**payload)


@pytest.fixture(scope="module")
def clear_preflight(checker, occupancy_checker):
    return preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.03, -0.02, 0.01, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
    )


def test_authorization_requires_exact_preflight_bound_confirmation(
    checker, occupancy_checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)

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


def test_authorization_requires_atomic_stopped_snapshot(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(stopped=False)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)

    with pytest.raises(RobotCommandError, match="verified stop latch"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(clear_preflight),
        )


def test_guarded_enable_occurs_only_after_valid_permit_and_only_once(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(enabled=False, stopped=True)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    prompt = executor.approval_prompt(clear_preflight)

    with pytest.raises(RobotCommandError, match="confirmation mismatch"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation="yes",
        )
    assert arm.guarded_enable_calls == 0

    first = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=prompt,
    )
    executor.execute(clear_preflight, first)
    assert arm.guarded_enable_calls == 1
    assert arm.events.index("guarded_enable") < arm.events.index("resume")

    second = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=prompt,
    )
    executor.execute(clear_preflight, second)
    assert arm.guarded_enable_calls == 1


def test_execute_revalidates_and_consumes_one_shot_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    result = executor.execute(clear_preflight, permit)

    assert result.ok is True
    assert arm.events == [
        "read_state",
        "resume",
        "read_state",
        "prepare",
        "stream",
        "stop",
    ]
    assert arm.resumed is True
    assert arm.prepared is True
    assert arm.streamed is True
    assert arm.stopped is True
    completed_events = list(arm.events)
    with pytest.raises(RobotCommandError, match="already consumed"):
        executor.execute(clear_preflight, permit)
    assert arm.events == completed_events


def test_execute_revalidation_is_bounded_by_geometric_legs_not_servoj_ticks(
    checker,
    occupancy_checker,
    clear_preflight,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    occupancy_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    original_mesh_check = _SyntheticSweptEs68Checker.check_path
    occupancy_checker_type = type(occupancy_checker)
    original_occupancy_check = occupancy_checker_type.check_path

    def counted_mesh_check(self, start, goal, **kwargs):
        mesh_calls.append((tuple(start), tuple(goal)))
        return original_mesh_check(self, start, goal, **kwargs)

    def counted_occupancy_check(self, start, goal, **kwargs):
        occupancy_calls.append((tuple(start), tuple(goal)))
        return original_occupancy_check(self, start, goal, **kwargs)

    monkeypatch.setattr(_SyntheticSweptEs68Checker, "check_path", counted_mesh_check)
    monkeypatch.setattr(
        occupancy_checker_type,
        "check_path",
        counted_occupancy_check,
    )
    arm = FakeGuardedArm(enabled=False)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    executor.execute(clear_preflight, permit)

    assert clear_preflight.servoj_stream is not None
    assert len(clear_preflight.servoj_stream.commands) > 2
    # Before enable, after enable, and after control recovery: two geometric
    # legs each, independent of the number of 4 ms ServoJ commands.
    assert len(mesh_calls) == 6
    assert len(occupancy_calls) == 6
    assert all(
        calls[0][1] == clear_preflight.start_joint_positions_rad
        for calls in (mesh_calls, occupancy_calls)
    )
    assert all(
        calls[1][1] == clear_preflight.goal_joint_positions_rad
        for calls in (mesh_calls, occupancy_calls)
    )


def test_execute_stops_arm_when_driver_prepare_raises(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(prepare_exception=RuntimeError("prepare failed"))
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is True
    assert arm.streamed is False
    assert arm.stopped is True


def test_execute_stops_arm_when_driver_stream_raises(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(stream_exception=RuntimeError("stream failed"))
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is True
    assert arm.streamed is True
    assert arm.stopped is True


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_execute_stops_arm_and_preserves_base_exception_from_driver_prepare(
    checker, occupancy_checker, clear_preflight, exception_type
) -> None:
    original = exception_type("prepare interrupted")
    arm = FakeGuardedArm(prepare_exception=original)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(exception_type) as raised:
        executor.execute(clear_preflight, permit)

    assert raised.value is original
    assert arm.prepared is True
    assert arm.streamed is False
    assert arm.stopped is True


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_execute_stops_arm_and_preserves_base_exception_from_driver_stream(
    checker, occupancy_checker, clear_preflight, exception_type
) -> None:
    original = exception_type("stream interrupted")
    arm = FakeGuardedArm(stream_exception=original)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(exception_type) as raised:
        executor.execute(clear_preflight, permit)

    assert raised.value is original
    assert arm.prepared is True
    assert arm.streamed is True
    assert arm.stopped is True


def test_execute_reports_unconfirmed_emergency_stop_without_losing_original_error(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = KeyboardInterrupt("stream interrupted")
    arm = FakeGuardedArm(
        stream_exception=original,
        stop_exception=SystemExit("stop interrupted"),
    )
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(EmergencyStopUnconfirmedError) as raised:
        executor.execute(clear_preflight, permit)

    assert raised.value.operation_error is original
    assert isinstance(raised.value.stop_errors[0], SystemExit)
    assert raised.value.error_code == "emergency_stop_unconfirmed"
    assert arm.stopped is True


def test_execute_stops_if_control_recovery_raises(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = KeyboardInterrupt("resume interrupted")
    arm = FakeGuardedArm(resume_exception=original)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        executor.execute(clear_preflight, permit)

    assert raised.value is original
    assert arm.events == ["read_state", "resume", "stop"]
    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_revalidates_live_start_after_control_recovery(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(
        recovered_joint_positions_rad=np.full(6, 0.02),
    )
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        live_start_tolerance_rad=0.01,
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="changed during control recovery"):
        executor.execute(clear_preflight, permit)

    assert arm.events == ["read_state", "resume", "read_state", "stop"]
    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_never_reports_success_when_same_arm_stop_fails(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = RuntimeError("writeIdle failed")
    arm = FakeGuardedArm(stop_exception=original)
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RuntimeError) as raised:
        executor.execute(clear_preflight, permit)

    assert raised.value is original
    assert arm.events[-3:] == ["prepare", "stream", "stop"]
    assert arm.stopped is True


def test_servoj_abort_and_stop_failure_are_reported_as_unconfirmed(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(stop_exception=RuntimeError("writeIdle failed"))
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    def aborted_stream(*args, **kwargs):
        del args, kwargs
        return StreamServoJResult(ok=False, commands_sent=1, abort_reason="tracking_error")

    arm._guarded_stream_servoj = aborted_stream  # type: ignore[method-assign]

    with pytest.raises(EmergencyStopUnconfirmedError) as raised:
        executor.execute(clear_preflight, permit)

    assert "tracking_error" in str(raised.value.operation_error)
    assert [str(error) for error in raised.value.stop_errors] == ["writeIdle failed"]


def test_execute_rejects_stop_generation_newer_than_one_shot_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    assert permit.stop_generation == 0
    arm.stop()

    with pytest.raises(RobotCommandError, match="stop latch changed"):
        executor.execute(clear_preflight, permit)

    assert arm.events == ["stop", "read_state", "stop"]
    assert arm.resumed is False
    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_detects_stop_race_after_prepare_even_if_callback_returns_false(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    calls = 0

    def concurrent_stop_without_cancel_flag() -> bool:
        nonlocal calls
        calls += 1
        if calls == 3:
            arm.stop()
        return False

    with pytest.raises(RobotCommandError, match="after_servoj_prepare"):
        executor.execute(
            clear_preflight,
            permit,
            cancellation_requested=concurrent_stop_without_cancel_flag,
        )

    assert arm.events == [
        "read_state",
        "resume",
        "read_state",
        "prepare",
        "stop",
        "stop",
    ]
    assert arm.streamed is False
    assert arm.stop_generation == permit.stop_generation + 2


def test_execute_checks_cancellation_in_required_order(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    def not_cancelled() -> bool:
        arm.events.append("cancel_check")
        return False

    result = executor.execute(
        clear_preflight,
        permit,
        cancellation_requested=not_cancelled,
    )

    assert result.ok is True
    assert arm.events == [
        "read_state",
        "cancel_check",
        "resume",
        "cancel_check",
        "read_state",
        "prepare",
        "cancel_check",
        "stream",
        "stop",
    ]


@pytest.mark.parametrize(
    ("cancel_on_call", "expected_events", "stage"),
    [
        (1, ["read_state", "stop"], "before_control_recovery"),
        (2, ["read_state", "resume", "stop"], "after_control_recovery"),
        (
            3,
            ["read_state", "resume", "read_state", "prepare", "stop"],
            "after_servoj_prepare",
        ),
    ],
)
def test_execute_cancellation_race_blocks_every_later_motion_stage(
    checker,
    occupancy_checker,
    clear_preflight,
    cancel_on_call,
    expected_events,
    stage,
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    calls = 0

    def cancellation_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls == cancel_on_call

    with pytest.raises(RobotCommandError, match=stage):
        executor.execute(
            clear_preflight,
            permit,
            cancellation_requested=cancellation_requested,
        )

    assert calls == cancel_on_call
    assert arm.events == expected_events
    assert arm.streamed is False
    assert arm.stopped is True


def test_execute_cancellation_callback_exception_stops_and_propagates(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = SystemExit("cancel callback failed")
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    def broken_callback() -> bool:
        raise original

    with pytest.raises(SystemExit) as raised:
        executor.execute(
            clear_preflight,
            permit,
            cancellation_requested=broken_callback,
        )

    assert raised.value is original
    assert arm.events == ["read_state", "stop"]
    assert arm.resumed is False
    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_non_boolean_cancellation_result_fails_closed(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="non-boolean"):
        executor.execute(
            clear_preflight,
            permit,
            cancellation_requested=lambda: 1,  # type: ignore[return-value]
        )

    assert arm.events == ["read_state", "stop"]
    assert arm.resumed is False


def test_execute_rechecks_permit_expiry_after_revalidation(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    clock_values = iter((0.0, 0.0, 0.0, 2.0))
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        permit_lifetime_s=1.0,
        clock=lambda: next(clock_values),
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="post-recovery revalidation"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False
    assert arm.stopped is True


def test_execute_rechecks_permit_expiry_after_servoj_prepare(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        permit_lifetime_s=1.0,
        clock=lambda: 2.0 if arm.prepared else 0.0,
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="ServoJ preparation"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is True
    assert arm.streamed is False


def test_execution_budget_expires_during_control_recovery(
    checker,
    occupancy_checker,
    clear_preflight,
) -> None:
    clock = SimpleNamespace(value=0.0)
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        clock=lambda: clock.value,
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    original_resume = arm._guarded_resume_servoj_control

    def slow_resume(**kwargs):
        original_resume(**kwargs)
        clock.value = 1.000001

    arm._guarded_resume_servoj_control = slow_resume  # type: ignore[method-assign]

    with pytest.raises(RobotCommandError, match="during control recovery"):
        executor.execute(clear_preflight, permit, maximum_duration_s=1.0)

    assert arm.stopped is True
    assert arm.prepared is False
    assert arm.streamed is False
    assert arm.stopped is True


def test_deadline_watchdog_and_boundary_stop_failures_are_typed(
    checker,
    occupancy_checker,
    clear_preflight,
) -> None:
    arm = FakeGuardedArm(
        deadline_stop_exception=RuntimeError("Dashboard stop failed"),
        stop_exception=RuntimeError("writeIdle stop failed"),
    )
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    def blocked_stream(*args, **kwargs):
        del args, kwargs
        deadline = time.monotonic() + 1.0
        while arm.deadline_stop_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        return StreamServoJResult(ok=False, commands_sent=0, abort_reason="deadline")

    arm._guarded_stream_servoj = blocked_stream  # type: ignore[method-assign]

    with pytest.raises(EmergencyStopUnconfirmedError) as raised:
        executor.execute(clear_preflight, permit, maximum_duration_s=0.01)

    assert raised.value.error_code == "emergency_stop_unconfirmed"
    assert arm.deadline_stop_calls == 1
    assert [str(error) for error in raised.value.stop_errors] == [
        "Dashboard stop failed",
        "writeIdle stop failed",
    ]


def test_live_start_mismatch_blocks_before_driver_prepare(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(joint_positions_rad=np.full(6, 0.02))
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        live_start_tolerance_rad=0.01,
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="no longer matches"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False
    assert arm.resumed is False
    assert arm.events == ["read_state"]


def test_expired_permit_is_consumed_without_motion(
    checker, occupancy_checker, clear_preflight
) -> None:
    clock = {"now": 10.0}
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
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
    assert arm.resumed is False
    assert arm.events == []


def test_caller_cannot_extend_expired_permit(checker, occupancy_checker, clear_preflight) -> None:
    clock = {"now": 10.0}
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        permit_lifetime_s=1.0,
        clock=lambda: clock["now"],
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    clock["now"] = 12.0

    with pytest.raises(RobotCommandError, match="payload was modified"):
        executor.execute(
            clear_preflight,
            replace(permit, expires_monotonic_s=1_000_000.0),
        )

    assert arm.prepared is False


def test_caller_cannot_mutate_authoritative_permit_through_returned_alias(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    # frozen=True blocks normal assignment but deliberately does not make the
    # object tamper-proof.  The internal permit must therefore be a distinct
    # value object, allowing this mutation to be detected on consumption.
    object.__setattr__(
        permit,
        "expires_monotonic_s",
        permit.expires_monotonic_s + 1_000_000.0,
    )

    with pytest.raises(RobotCommandError, match="permit payload was modified"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False
    assert arm.resumed is False
    assert arm.events == []


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("permit_lifetime_s", float("nan")),
        ("live_start_tolerance_rad", float("nan")),
    ],
)
def test_executor_rejects_nonfinite_limits(checker, occupancy_checker, keyword, value) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GuardedEliteExecutor(
            FakeGuardedArm(),
            checker,
            occupancy_checker,
            **{keyword: value},
        )


def test_executor_rejects_nonfinite_clock(checker, occupancy_checker, clear_preflight) -> None:
    executor = GuardedEliteExecutor(
        FakeGuardedArm(),
        checker,
        occupancy_checker,
        clock=lambda: float("nan"),
    )

    with pytest.raises(RobotCommandError, match="issue time must be finite"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(clear_preflight),
        )


def test_authorization_rejects_tampered_servoj_detour(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = clear_preflight.servoj_stream
    assert original is not None and len(original.commands) > 2
    commands = list(original.commands)
    midpoint = list(commands[len(commands) // 2])
    midpoint[1] += 0.001
    commands[len(commands) // 2] = tuple(midpoint)
    tampered = replace(
        clear_preflight,
        servoj_stream=ServoJStream(tuple(commands), original.dt_s),
    )
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)

    with pytest.raises(RobotCommandError, match="stream does not reproduce"):
        executor.authorize(
            tampered,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(tampered),
        )


def test_authorization_rejects_changed_occupancy_snapshot(checker, occupancy_snapshot) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    holder["snapshot"] = _changed_snapshot(occupancy_snapshot)
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_authorization_rejects_mutable_snapshot_provider_result(
    checker, occupancy_snapshot
) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )

    class MutableSnapshotLookalike:
        frame_id = "base"
        map_state = "map_ready"
        sequence = occupancy_snapshot.sequence
        content_hash = occupancy_snapshot.content_hash

    holder["snapshot"] = MutableSnapshotLookalike()
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)

    with pytest.raises(RobotCommandError, match="concrete_immutable_snapshot"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_execute_rejects_snapshot_change_after_permit(checker, occupancy_snapshot) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy)
    permit = executor.authorize(
        preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(preflight),
    )
    holder["snapshot"] = _changed_snapshot(occupancy_snapshot)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.execute(preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_permit_carries_explicit_occupancy_binding(
    checker, occupancy_checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    evidence = clear_preflight.occupancy.evidence

    assert evidence is not None
    assert permit.occupancy_sequence == evidence.sequence
    assert permit.occupancy_content_hash == evidence.content_hash
    assert permit.occupancy_mapping_context_hash == evidence.mapping_context_hash
    assert permit.occupancy_quality_evidence_hash == evidence.quality_evidence_hash
    assert permit.occupancy_metadata_sha256 == evidence.occupancy_metadata_sha256
    assert (
        permit.occupancy_semantic_verifier_contract_hash == evidence.semantic_verifier_contract_hash
    )
    assert permit.occupancy_semantic_attestation_hash == evidence.semantic_attestation_hash
    assert permit.continuous_occupancy_sweep_verified is True
    assert permit.stop_generation == 0
    assert permit.collision_model_id == checker.collision_model_id
    assert permit.collision_model_hash == checker.collision_model_hash
    assert permit.robot_geometry_hash == checker.robot_geometry_hash
    assert permit.motion_model_contract_hash == checker.motion_model_contract_hash
    assert len(permit.servoj_runtime_config_hash) == 64
    assert permit.occupancy_policy_contract_hash == occupancy_checker.policy_contract_hash


def test_execute_rejects_runtime_guard_relaxation(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="runtime config differs"):
        executor.execute(
            clear_preflight,
            permit,
            stream_config=ServoJStreamConfig(
                dt_s=0.004,
                tracking_error_rad=100.0,
                tracking_check_every_n_commands=99,
            ),
        )

    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_rejects_occupancy_policy_change_after_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    original = occupancy_checker.additional_clearance_m
    occupancy_checker.additional_clearance_m = original + 0.001
    try:
        with pytest.raises(RobotCommandError, match="policy differs"):
            executor.execute(clear_preflight, permit)
    finally:
        occupancy_checker.additional_clearance_m = original

    assert arm.prepared is False
    assert arm.streamed is False


def test_executor_rejects_distinct_mesh_checker_for_occupancy(checker) -> None:
    other_checker = Cs68PinocchioCollisionChecker.from_resources()
    occupancy = _SyntheticContinuousOccupancyChecker(
        other_checker,
        lambda: None,
        verified_robot_geometry_hash="8" * 64,
    )

    with pytest.raises(ValueError, match="share one checker instance"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)


def test_authorization_requires_freshness_for_whole_planned_stream(
    checker, occupancy_snapshot
) -> None:
    clock = {"utc": datetime(2026, 8, 28, 0, 0, 0, 100000, tzinfo=UTC)}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: occupancy_snapshot,
        maximum_map_age_s=5.0,
        utc_clock=lambda: clock["utc"],
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)
    clock["utc"] = datetime(2026, 8, 28, 0, 0, 4, 900000, tzinfo=UTC)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_execute_requires_freshness_for_whole_remaining_stream(checker, occupancy_snapshot) -> None:
    clock = {"utc": datetime(2026, 8, 28, 0, 0, 0, 100000, tzinfo=UTC)}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: occupancy_snapshot,
        maximum_map_age_s=5.0,
        utc_clock=lambda: clock["utc"],
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy)
    permit = executor.authorize(
        preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(preflight),
    )
    clock["utc"] = datetime(2026, 8, 28, 0, 0, 4, 900000, tzinfo=UTC)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.execute(preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_executor_rejects_current_discrete_occupancy_checker(checker) -> None:
    occupancy = OccupancyRobotCollisionChecker(checker, lambda: None)

    with pytest.raises(ValueError, match="semantic occupancy attestation"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)


def test_executor_rejects_protocol_only_occupancy_checker(checker) -> None:
    class FakeOccupancyChecker:
        robot_checker = checker
        ignored_geometry_names = ()
        motion_semantic_attestation_valid = True
        continuous_swept_volume_supported = True
        verified_robot_geometry_hash = checker.robot_geometry_hash

    with pytest.raises(ValueError, match="concrete occupancy collision checker"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, FakeOccupancyChecker())


@pytest.mark.parametrize(
    "permit_update",
    [
        {"occupancy_mapping_context_hash": "e" * 64},
        {"occupancy_quality_evidence_hash": "f" * 64},
        {"occupancy_metadata_sha256": "a" * 64},
        {"occupancy_semantic_verifier_contract_hash": "b" * 64},
        {"occupancy_semantic_attestation_hash": "c" * 64},
        {"continuous_occupancy_sweep_verified": False},
    ],
)
def test_execute_rejects_relaxed_or_rebound_occupancy_permit(
    checker, occupancy_checker, clear_preflight, permit_update
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="permit payload was modified"):
        executor.execute(clear_preflight, replace(permit, **permit_update))

    assert arm.prepared is False
    assert arm.streamed is False
