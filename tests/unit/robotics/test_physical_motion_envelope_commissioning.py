"""Tests for the physical commissioning execution boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.robotics.motion_envelope_commissioning as commissioning
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.robot import ServoJStream, StreamServoJResult
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.robotics import MotionPreflightStatus
from biblade_fusion.storage.motion_envelope_commissioning import (
    CommissioningTrialCandidate,
    StoredCommissioningTrialCandidate,
)


def _candidate() -> CommissioningTrialCandidate:
    start = (0.0,) * 6
    goal = (0.01, -0.01, 0.005, 0.0, -0.02, 0.01)
    return CommissioningTrialCandidate(
        candidate_id="a" * 64,
        target_view_id="front_r00_c00",
        start_view_id="start",
        start_joint_positions_rad=start,
        raw_target_joint_positions_rad=goal,
        normalized_target_joint_positions_rad=goal,
        target_joint_turn_offsets=(0,) * 6,
        goal_joint_positions_rad=goal,
        direction_scale=1.0,
        maximum_candidate_joint_delta_rad=0.02,
        maximum_remaining_target_joint_delta_rad=0.02,
        mesh_status="clear",
        mesh_continuous_swept_volume_verified=True,
        mesh_minimum_certificate_margin_m=0.01,
        estimated_servoj_duration_s=0.004,
        servoj_command_count=2,
        blocking_reasons=(),
        warnings=("occupancy_disabled_offline_diagnostic_only",),
    )


def _stored(tmp_path, settings) -> StoredCommissioningTrialCandidate:
    candidate = _candidate()
    return StoredCommissioningTrialCandidate(
        path=tmp_path,
        candidate=candidate,
        metadata={
            "configuration": {
                "motion_preflight": settings.motion_preflight.model_dump(mode="json"),
                "collision": settings.collision.model_dump(mode="json"),
                "joint_zero_offsets_rad": list(settings.kinematics.joint_zero_offsets_rad),
            }
        },
    )


def _clear_preflight(candidate: CommissioningTrialCandidate, settings):
    return SimpleNamespace(
        status=MotionPreflightStatus.CLEAR,
        servoj_stream=ServoJStream(
            commands=(
                candidate.start_joint_positions_rad,
                candidate.goal_joint_positions_rad,
            ),
            dt_s=settings.motion_preflight.servoj_dt_s,
        ),
        collision=SimpleNamespace(continuous_swept_volume_evidence_valid=True),
    )


def test_prepare_is_hardware_free_and_prints_candidate_bound_prompt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path, settings)
    checker = object()

    class FakeChecker:
        @staticmethod
        def from_es68_resources(*_args, **_kwargs):
            return checker

    monkeypatch.setattr(
        commissioning,
        "read_commissioning_trial_candidate",
        lambda _path: stored,
    )
    monkeypatch.setattr(commissioning, "Cs68PinocchioCollisionChecker", FakeChecker)
    monkeypatch.setattr(
        commissioning,
        "preflight_linear_joint_motion",
        lambda *_args, **_kwargs: _clear_preflight(stored.candidate, settings),
    )

    prepared = commissioning.prepare_motion_envelope_commissioning_trial(tmp_path, settings)

    assert prepared.approval_prompt == "EXECUTE COMMISSION aaaaaaaaaaaa"
    assert prepared.collision_checker is checker


def test_prepare_rejects_configuration_drift_before_preflight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path, settings)
    stored.metadata["configuration"]["motion_preflight"]["speed_scaling"] = 0.09
    monkeypatch.setattr(
        commissioning,
        "read_commissioning_trial_candidate",
        lambda _path: stored,
    )

    with pytest.raises(ValueError, match="configuration has drifted"):
        commissioning.prepare_motion_envelope_commissioning_trial(tmp_path, settings)


def test_execute_rejects_wrong_confirmation_before_fifo_or_arm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    prepared = SimpleNamespace(
        approval_prompt="EXECUTE COMMISSION aaaaaaaaaaaa",
    )
    monkeypatch.setattr(
        commissioning,
        "probe_fifo_scheduler",
        lambda: pytest.fail("FIFO must not be touched before confirmation"),
    )

    with pytest.raises(ValueError, match="confirmation mismatch"):
        commissioning.execute_motion_envelope_commissioning_trial(
            prepared,
            settings,
            operator_id="operator-1",
            confirmation="yes",
            output_path=tmp_path / "result",
            arm_factory=lambda _config: pytest.fail("arm must not be constructed"),
        )


def test_fifo_scheduler_restores_original_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(commissioning.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(
        commissioning.os,
        "sched_getparam",
        lambda _pid: commissioning.os.sched_param(0),
    )
    monkeypatch.setattr(
        commissioning.os,
        "sched_setscheduler",
        lambda pid, policy, parameter: calls.append((pid, policy, parameter.sched_priority)),
    )

    with commissioning.fifo_scheduler(priority=10):
        pass

    assert calls == [(0, commissioning.os.SCHED_FIFO, 10), (0, 0, 0)]


def test_fifo_scheduler_converts_permission_denial_to_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commissioning.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(
        commissioning.os,
        "sched_getparam",
        lambda _pid: commissioning.os.sched_param(0),
    )

    def deny(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(commissioning.os, "sched_setscheduler", deny)

    with pytest.raises(RuntimeError, match="PAM rtprio"):
        commissioning.probe_fifo_scheduler()


def _state(
    joints,
    *,
    time_s: float = 1.0,
    actual_joint_velocity_rad_s=None,
) -> RobotState:
    zeros = np.zeros(6, dtype=np.float64)
    return RobotState(
        monotonic_time_ns=int(time_s * 1e9),
        controller_time_s=time_s,
        joint_positions_rad=np.asarray(joints, dtype=np.float64),
        base_t_tcp=PoseSE3.identity("base", "tcp"),
        robot_mode="RUNNING",
        safety_status="NORMAL",
        speed_scaling=0.1,
        runtime_state="RUNNING",
        actual_joint_velocity_rad_s=(
            zeros
            if actual_joint_velocity_rad_s is None
            else np.asarray(actual_joint_velocity_rad_s, dtype=np.float64)
        ),
        target_joint_velocity_rad_s=zeros,
        actual_tcp_velocity=zeros,
        target_tcp_velocity=zeros,
    )


def test_execute_stops_after_one_exact_bounded_stream_and_writes_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path / "candidate", settings)
    stored.path.mkdir()
    (stored.path / "candidate.json").write_text("{}", encoding="utf-8")
    preflight = _clear_preflight(stored.candidate, settings)
    prepared = commissioning.PreparedMotionEnvelopeCommissioningTrial(
        stored_candidate=stored,
        preflight=preflight,
        collision_checker=object(),
        approval_prompt="EXECUTE COMMISSION aaaaaaaaaaaa",
        direction="forward",
        execution_start_joint_positions_rad=stored.candidate.start_joint_positions_rad,
        execution_goal_joint_positions_rad=stored.candidate.goal_joint_positions_rad,
    )
    prepared = commissioning.bind_motion_envelope_commissioning_output(prepared, tmp_path / "trial")
    calls: list[str] = []

    class FakeArm:
        def __init__(self, config) -> None:
            assert config.motion_enabled is True
            assert config.default_speed_scaling == settings.robot.default_speed_scaling
            assert config.maximum_speed_scaling == settings.robot.maximum_speed_scaling
            self.is_connected = False
            self.is_enabled = False
            self._stopped = False
            self._generation = 0
            self._joints = np.asarray(stored.candidate.start_joint_positions_rad)

        @property
        def stop_snapshot(self):
            return self._generation, self._stopped

        def connect(self, *, with_driver: bool) -> None:
            assert with_driver is True
            self.is_connected = True
            calls.append("connect")

        def read_state(self) -> RobotState:
            return _state(self._joints)

        def stop(self) -> None:
            self._generation += 1
            self._stopped = True
            calls.append("stop")

        def _guarded_enable_for_servoj_control(self, **kwargs) -> None:
            assert kwargs["expected_stop_generation"] == self._generation
            self.is_enabled = True
            calls.append("enable")

        def _guarded_resume_servoj_control(self, **kwargs) -> None:
            assert kwargs["expected_stop_generation"] == self._generation
            assert kwargs["deadline_exceeded"]() is False
            assert self._stopped is True
            self._stopped = False
            calls.append("resume")

        def _guarded_prepare_servoj_stream(self, **kwargs) -> None:
            assert kwargs["dt_s"] == settings.motion_preflight.servoj_dt_s
            assert kwargs["warmup_duration_s"] == 0.2
            assert kwargs["deadline_exceeded"]() is False
            calls.append("prepare")

        def _guarded_stream_servoj(self, stream, *, tracking_samples, **_kwargs):
            self._joints = np.asarray(stream.commands[-1])
            tracking_samples.append(
                {
                    "q_cmd": list(stream.commands[-1]),
                    "q_actual": list(stream.commands[-1]),
                }
            )
            calls.append("stream")
            return StreamServoJResult(ok=True, commands_sent=len(stream.commands))

        def _guarded_deadline_stop(self, **_kwargs) -> None:
            pytest.fail("watchdog must be cancelled after the short stream")

        def release(self) -> None:
            self.is_connected = False
            calls.append("release")

    monkeypatch.setattr(commissioning, "probe_fifo_scheduler", lambda: None)
    monkeypatch.setattr(
        commissioning,
        "fifo_scheduler",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        commissioning,
        "preflight_linear_joint_motion",
        lambda *_args, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        commissioning,
        "_wait_for_stationary_safe_window",
        lambda arm, _goal, *, settle_time_s, samples: (
            arm.read_state(),
            {
                "sample_count": 2,
                "duration_s": settle_time_s,
                "controller_duration_s": settle_time_s,
                "goal_error_rad": 0.0,
                "maximum_joint_speed_rad_s": 0.0,
                "maximum_tcp_speed": 0.0,
                "total_observed_sample_count": len(samples),
            },
        ),
    )
    output = tmp_path / "trial"

    result = commissioning.execute_motion_envelope_commissioning_trial(
        prepared,
        settings,
        operator_id="operator-1",
        confirmation=prepared.approval_prompt,
        output_path=output,
        arm_factory=FakeArm,
    )

    assert result.ok is True
    assert calls == [
        "connect",
        "stop",
        "enable",
        "resume",
        "prepare",
        "stream",
        "stop",
        "release",
    ]
    payload = commissioning.json.loads((output / "trial.json").read_text())
    assert payload["production_motion_authorized"] is False
    assert payload["ok"] is True
    assert payload["goal_error_rad"] == 0.0
    assert payload["effective_limits"]["sdk_fifo_priority"] == 99
    assert payload["effective_limits"]["servoj_warmup_duration_s"] == 0.2
    assert payload["maximum_tracking_deviation_rad"] == [0.0] * 6
    assert payload["settling_evidence"]["duration_s"] == settings.robot.settle_time_s


def test_intentional_tracking_fault_requires_expected_early_abort_and_stop(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path / "candidate", settings)
    stored.path.mkdir()
    (stored.path / "candidate.json").write_text("{}", encoding="utf-8")
    preflight = _clear_preflight(stored.candidate, settings)
    prepared = commissioning.PreparedMotionEnvelopeCommissioningTrial(
        stored_candidate=stored,
        preflight=preflight,
        collision_checker=object(),
        approval_prompt="EXECUTE COMMISSION aaaaaaaaaaaa",
        direction="forward",
        execution_start_joint_positions_rad=stored.candidate.start_joint_positions_rad,
        execution_goal_joint_positions_rad=stored.candidate.goal_joint_positions_rad,
    )
    prepared = commissioning.intentional_tracking_fault_motion_envelope_commissioning_trial(
        prepared
    )
    prepared = commissioning.bind_motion_envelope_commissioning_output(
        prepared, tmp_path / "fault-trial"
    )
    calls: list[str] = []

    class FakeArm:
        def __init__(self, config) -> None:
            assert config.motion_enabled is True
            self.is_connected = False
            self.is_enabled = False
            self._stopped = False
            self._generation = 0
            self._joints = np.asarray(stored.candidate.start_joint_positions_rad)
            self._joints[0] += 0.004

        @property
        def stop_snapshot(self):
            return self._generation, self._stopped

        def connect(self, *, with_driver: bool) -> None:
            assert with_driver is True
            self.is_connected = True

        def read_state(self) -> RobotState:
            return _state(self._joints)

        def stop(self) -> None:
            self._generation += 1
            self._stopped = True
            calls.append("stop")

        def _guarded_enable_for_servoj_control(self, **_kwargs) -> None:
            self.is_enabled = True

        def _guarded_resume_servoj_control(self, **_kwargs) -> None:
            self._stopped = False

        def _guarded_prepare_servoj_stream(self, **_kwargs) -> None:
            return None

        def _guarded_stream_servoj(self, stream, *, config, tracking_samples, **_kwargs):
            assert len(stream.commands) == 10
            assert config.tracking_error_rad == 0.001
            assert config.max_consecutive_tracking_violations == 1
            tracking_samples.append(
                {
                    "q_cmd": list(stream.commands[2]),
                    "q_actual": list(stream.commands[0]),
                }
            )
            self._stopped = True
            return StreamServoJResult(
                ok=False,
                commands_sent=3,
                max_tracking_error_rad=0.0015,
                abort_reason="tracking_error_exceeded",
                last_command_index=2,
            )

        def _guarded_deadline_stop(self, **_kwargs) -> None:
            pytest.fail("watchdog must be cancelled after the early abort")

        def release(self) -> None:
            self.is_connected = False

    monkeypatch.setattr(commissioning, "probe_fifo_scheduler", lambda: None)
    monkeypatch.setattr(
        commissioning,
        "fifo_scheduler",
        lambda *_args, **_kwargs: nullcontext(),
    )
    live_preflight_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def live_preflight(start, goal, **_kwargs):
        live_preflight_calls.append(
            (np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64))
        )
        return SimpleNamespace(
            status=MotionPreflightStatus.CLEAR,
            servoj_stream=ServoJStream(
                commands=(tuple(start), tuple(goal)),
                dt_s=settings.motion_preflight.servoj_dt_s,
            ),
            collision=SimpleNamespace(continuous_swept_volume_evidence_valid=True),
        )

    monkeypatch.setattr(commissioning, "preflight_linear_joint_motion", live_preflight)
    monkeypatch.setattr(
        commissioning,
        "_wait_for_stationary_safe_window",
        lambda arm, _goal, *, settle_time_s, samples: (
            arm.read_state(),
            {
                "sample_count": 2,
                "duration_s": settle_time_s,
                "controller_duration_s": settle_time_s,
                "goal_error_rad": 0.0,
                "maximum_joint_speed_rad_s": 0.0,
                "maximum_tcp_speed": 0.0,
                "total_observed_sample_count": len(samples),
            },
        ),
    )

    result = commissioning.execute_motion_envelope_commissioning_trial(
        prepared,
        settings,
        operator_id="operator-1",
        confirmation=prepared.approval_prompt,
        output_path=tmp_path / "fault-trial",
        arm_factory=FakeArm,
    )

    assert result.ok is True
    assert calls == ["stop", "stop"]
    assert len(live_preflight_calls) == 1
    live_start, live_goal = live_preflight_calls[0]
    assert np.max(np.abs(live_goal - live_start)) == pytest.approx(0.01)
    assert "TRACKING-FAULT" in prepared.approval_prompt
    payload = commissioning.json.loads(
        (tmp_path / "fault-trial" / "trial.json").read_text()
    )
    assert payload["candidate"]["trial_kind"] == "intentional_tracking_fault"
    assert payload["candidate"]["sealed_execution_goal_joint_positions_rad"] != payload[
        "candidate"
    ]["execution_goal_joint_positions_rad"]
    assert payload["stream_result"]["abort_reason"] == "tracking_error_exceeded"
    assert payload["effective_limits"]["maximum_fault_trial_commands"] == 10
    assert payload["fault_detection_to_stop_acknowledgement_s"] is not None
    assert payload["ok"] is True


def test_output_binding_changes_prompt_and_blocks_replay_to_another_path(
    tmp_path,
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path / "candidate", settings)
    prepared = commissioning.PreparedMotionEnvelopeCommissioningTrial(
        stored_candidate=stored,
        preflight=_clear_preflight(stored.candidate, settings),
        collision_checker=object(),
        approval_prompt="EXECUTE COMMISSION aaaaaaaaaaaa",
        direction="forward",
        execution_start_joint_positions_rad=stored.candidate.start_joint_positions_rad,
        execution_goal_joint_positions_rad=stored.candidate.goal_joint_positions_rad,
    )

    first = commissioning.bind_motion_envelope_commissioning_output(prepared, tmp_path / "trial-1")
    second = commissioning.bind_motion_envelope_commissioning_output(prepared, tmp_path / "trial-2")

    assert first.approval_prompt != second.approval_prompt
    assert first.bound_output_path == (tmp_path / "trial-1").resolve()
    with pytest.raises(ValueError, match="not bound to this output path"):
        commissioning.execute_motion_envelope_commissioning_trial(
            first,
            settings,
            operator_id="operator-1",
            confirmation=first.approval_prompt,
            output_path=tmp_path / "trial-2",
            arm_factory=lambda _config: pytest.fail("arm must not be constructed"),
        )


def test_reverse_revalidates_path_and_binds_direction_to_prompt(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings("configs/default.yaml")
    stored = _stored(tmp_path / "candidate", settings)
    forward = commissioning.PreparedMotionEnvelopeCommissioningTrial(
        stored_candidate=stored,
        preflight=_clear_preflight(stored.candidate, settings),
        collision_checker=object(),
        approval_prompt="EXECUTE COMMISSION aaaaaaaaaaaa",
        direction="forward",
        execution_start_joint_positions_rad=stored.candidate.start_joint_positions_rad,
        execution_goal_joint_positions_rad=stored.candidate.goal_joint_positions_rad,
    )
    calls = []

    def fake_preflight(start, goal, **_kwargs):
        calls.append((tuple(start), tuple(goal)))
        return SimpleNamespace(
            status=MotionPreflightStatus.CLEAR,
            servoj_stream=ServoJStream(
                commands=(tuple(start), tuple(goal)),
                dt_s=settings.motion_preflight.servoj_dt_s,
            ),
            collision=SimpleNamespace(continuous_swept_volume_evidence_valid=True),
        )

    monkeypatch.setattr(
        commissioning,
        "preflight_linear_joint_motion",
        fake_preflight,
    )

    reverse = commissioning.reverse_motion_envelope_commissioning_trial(forward, settings)
    bound = commissioning.bind_motion_envelope_commissioning_output(
        reverse, tmp_path / "reverse-trial"
    )

    assert calls == [
        (
            stored.candidate.goal_joint_positions_rad,
            stored.candidate.start_joint_positions_rad,
        )
    ]
    assert reverse.direction == "reverse"
    assert reverse.execution_start_joint_positions_rad == (
        stored.candidate.goal_joint_positions_rad
    )
    assert reverse.execution_goal_joint_positions_rad == (
        stored.candidate.start_joint_positions_rad
    )
    assert " REVERSE OUTPUT " in bound.approval_prompt


def test_goal_hold_keeps_exact_endpoint_and_extends_duration() -> None:
    stream = ServoJStream(commands=((0.0,) * 6, (0.02,) * 6), dt_s=0.004)

    extended = commissioning._with_goal_hold(stream)

    assert extended.commands[:2] == stream.commands
    assert extended.commands[-1] == stream.commands[-1]
    assert len(extended.commands) == 252


def test_settle_window_ignores_one_transient_velocity_sample() -> None:
    joints = np.asarray(_candidate().goal_joint_positions_rad, dtype=np.float64)

    class FakeClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = FakeClock()

    class FakeArm:
        reads = 0

        def read_state(self) -> RobotState:
            self.reads += 1
            velocity = np.zeros(6, dtype=np.float64)
            if self.reads == 1:
                velocity[4] = 0.001293
            return _state(
                joints,
                time_s=clock.now,
                actual_joint_velocity_rad_s=velocity,
            )

    samples: list[dict[str, object]] = []
    final, evidence = commissioning._wait_for_stationary_safe_window(
        FakeArm(),
        joints,
        settle_time_s=1.0,
        timeout_s=2.0,
        poll_period_s=0.05,
        samples=samples,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert final.joint_positions_rad.tolist() == joints.tolist()
    assert evidence["duration_s"] >= 1.0
    assert evidence["sample_count"] >= 21
    assert evidence["total_observed_sample_count"] == len(samples)
    assert samples[0]["maximum_joint_speed_rad_s"] == pytest.approx(0.001293)
