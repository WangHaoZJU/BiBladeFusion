from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field, replace

import numpy as np
import pytest

import biblade_fusion.robotics.stationarity as stationarity_module
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.robotics.stationarity import (
    BootstrapSafeStateEvidence,
    RobotStateSource,
    StationarityError,
    StationarityTimeoutError,
    validate_stationary_trace,
    wait_until_bootstrap_safe_state,
    wait_until_settled,
)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _state(
    time_s: float,
    *,
    joint: float = 0.0,
    translation_m: float = 0.0,
    rotation_rad: float = 0.0,
    controller_time_s: float | None = None,
) -> RobotState:
    return RobotState(
        monotonic_time_ns=round(time_s * 1e9),
        controller_time_s=(
            time_s if controller_time_s is None else controller_time_s
        ),
        joint_positions_rad=np.full(6, joint, dtype=np.float64),
        base_t_tcp=PoseSE3.from_rotation_translation(
            "base",
            "tcp",
            _rotation_z(rotation_rad),
            [translation_m, 0.0, 0.0],
        ),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.0,
    )


class StateSource:
    def __init__(self, states: list[RobotState]) -> None:
        if not states:
            raise ValueError("states must not be empty")
        self._states = states
        self._index = 0

    def read_state(self) -> RobotState:
        index = min(self._index, len(self._states) - 1)
        self._index += 1
        return self._states[index]


class BootstrapStateSource(StateSource):
    def __init__(self, states: list[RobotState]) -> None:
        super().__init__(states)
        self.stop_snapshot = (1, True)


def _bootstrap_state(
    time_s: float,
    *,
    actual_joint_velocity_rad_s: float = 0.0,
    target_joint_velocity_rad_s: float = 0.0,
    actual_tcp_linear_velocity_m_s: float = 0.0,
    runtime_state: str = "STOPPED",
) -> RobotState:
    return replace(
        _state(time_s),
        runtime_state=runtime_state,
        actual_joint_velocity_rad_s=np.full(6, actual_joint_velocity_rad_s),
        target_joint_velocity_rad_s=np.full(6, target_joint_velocity_rad_s),
        actual_tcp_velocity=np.array(
            [actual_tcp_linear_velocity_m_s, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        target_tcp_velocity=np.zeros(6),
    )


def _wait_bootstrap(
    source: BootstrapStateSource,
    fake_time: FakeTime,
) -> BootstrapSafeStateEvidence:
    return wait_until_bootstrap_safe_state(
        source,
        expected_stop_generation=1,
        settle_time_s=0.5,
        timeout_s=1.5,
        poll_period_s=0.25,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        maximum_robot_state_staleness_s=0.25,
        maximum_stopped_actual_joint_velocity_rad_s=0.002,
        maximum_stopped_target_joint_velocity_rad_s=0.002,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.002,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.002,
        monotonic_clock=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )


@dataclass
class FakeTime:
    now_s: float = 0.0
    sleep_calls: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.sleep_calls.append(duration_s)
        self.now_s += duration_s


def _wait(
    source: StateSource,
    fake_time: FakeTime,
    *,
    settle_time_s: float = 1.0,
    timeout_s: float = 2.0,
    poll_period_s: float = 0.25,
    maximum_staleness_s: float = 0.25,
):
    return wait_until_settled(
        source,
        np.zeros(6),
        settle_time_s=settle_time_s,
        timeout_s=timeout_s,
        poll_period_s=poll_period_s,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        goal_tolerance_rad=0.01,
        maximum_robot_state_staleness_s=maximum_staleness_s,
        monotonic_clock=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )


def test_narrow_robot_state_source_requires_only_read_state() -> None:
    source = StateSource([_state(0.0)])

    assert isinstance(source, RobotStateSource)


def test_bootstrap_stop_requires_full_stopped_controller_and_velocity_window() -> None:
    source = BootstrapStateSource(
        [
            _bootstrap_state(0.0, actual_joint_velocity_rad_s=0.01),
            _bootstrap_state(0.25),
            _bootstrap_state(0.50),
            _bootstrap_state(0.75),
        ]
    )
    fake_time = FakeTime()

    evidence = _wait_bootstrap(source, fake_time)

    assert evidence.stationarity.sample_count == 3
    assert evidence.stationarity.duration_s == pytest.approx(0.5)
    assert evidence.stop_generation == 1
    assert evidence.runtime_state == "STOPPED"
    assert len(evidence.evidence_sha256) == 64


@pytest.mark.parametrize("robot_mode", ("RUNNING", "7"))
def test_bootstrap_stop_accepts_powered_robot_with_stopped_runtime(
    robot_mode: str,
) -> None:
    source = BootstrapStateSource(
        [
            replace(_bootstrap_state(0.0), robot_mode=robot_mode),
            replace(_bootstrap_state(0.25), robot_mode=robot_mode),
            replace(_bootstrap_state(0.50), robot_mode=robot_mode),
        ]
    )

    evidence = _wait_bootstrap(source, FakeTime())

    assert evidence.robot_mode == robot_mode
    assert evidence.runtime_state == "STOPPED"


def test_bootstrap_stop_fails_closed_when_velocity_channels_are_missing() -> None:
    source = BootstrapStateSource([_state(0.0)])

    with pytest.raises(StationarityError, match="mandatory RTSI velocity"):
        _wait_bootstrap(source, FakeTime())


def test_bootstrap_stop_fails_when_stop_generation_changes() -> None:
    source = BootstrapStateSource([_bootstrap_state(0.0)])

    def change_generation(duration_s: float) -> None:
        fake_time.now_s += duration_s
        source.stop_snapshot = (2, True)

    fake_time = FakeTime()
    fake_time.sleep = change_generation  # type: ignore[method-assign]

    with pytest.raises(StationarityError, match="generation/latch changed"):
        _wait_bootstrap(source, fake_time)


def test_wait_until_settled_covers_the_full_sampled_window() -> None:
    states = [_state(index * 0.25) for index in range(5)]
    source = StateSource(states)
    fake_time = FakeTime()

    evidence = _wait(source, fake_time)

    assert evidence.final_state is states[-1]
    assert evidence.sample_count == 5
    assert evidence.duration_s == pytest.approx(1.0)
    assert evidence.controller_duration_s == pytest.approx(1.0)
    assert evidence.max_sample_gap_s == pytest.approx(0.25)
    assert evidence.max_joint_delta_rad == 0.0
    assert evidence.max_tcp_translation_delta_m == 0.0
    assert evidence.max_tcp_rotation_delta_rad == 0.0
    assert evidence.goal_error_rad == 0.0
    assert sum(fake_time.sleep_calls) == pytest.approx(1.0)
    with pytest.raises(FrozenInstanceError):
        evidence.sample_count = 0  # type: ignore[misc]


def test_wait_until_settled_allows_playing_to_transition_to_stopped() -> None:
    states = [
        replace(_state(0.0), robot_mode="RUNNING", runtime_state="PLAYING"),
        replace(_state(0.25), robot_mode="RUNNING", runtime_state="STOPPED"),
        replace(_state(0.50), robot_mode="RUNNING", runtime_state="STOPPED"),
        replace(_state(0.75), robot_mode="RUNNING", runtime_state="STOPPED"),
    ]

    evidence = _wait(
        StateSource(states),
        FakeTime(),
        settle_time_s=0.5,
        timeout_s=1.0,
    )

    assert evidence.final_state is states[-1]
    assert evidence.sample_count == 3
    assert evidence.duration_s == pytest.approx(0.5)


def test_wait_until_settled_still_rejects_unsafe_transition_state() -> None:
    state = replace(
        _state(0.0),
        robot_mode="RUNNING",
        runtime_state="PLAYING",
        safety_status="PROTECTIVE_STOP",
    )

    with pytest.raises(StationarityError, match="NORMAL or REDUCED"):
        _wait(StateSource([state]), FakeTime())


def test_zero_settle_time_accepts_one_goal_sample_without_sleeping() -> None:
    source = StateSource([_state(0.0, joint=0.0005)])
    fake_time = FakeTime()

    evidence = _wait(source, fake_time, settle_time_s=0.0)

    assert evidence.sample_count == 1
    assert evidence.duration_s == 0.0
    assert evidence.controller_duration_s == 0.0
    assert evidence.max_sample_gap_s == 0.0
    assert evidence.goal_error_rad == pytest.approx(0.0005)
    assert fake_time.sleep_calls == []


def test_none_goal_checks_stationarity_without_a_goal_gate() -> None:
    states = [_state(index * 0.25, joint=1.2) for index in range(3)]
    source = StateSource(states)
    fake_time = FakeTime()

    evidence = wait_until_settled(
        source,
        None,
        settle_time_s=0.5,
        timeout_s=1.0,
        poll_period_s=0.25,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        goal_tolerance_rad=0.0,
        maximum_robot_state_staleness_s=0.25,
        monotonic_clock=fake_time.monotonic,
        sleeper=fake_time.sleep,
    )

    assert evidence.final_state is states[-1]
    assert evidence.duration_s == pytest.approx(0.5)
    assert evidence.goal_error_rad == 0.0


def test_motion_resets_the_window_and_new_window_spans_settle_time() -> None:
    states = [
        _state(0.0, joint=0.0),
        _state(0.25, joint=0.0020),
        _state(0.50, joint=0.0022),
        _state(0.75, joint=0.0021),
        _state(1.00, joint=0.0021),
        _state(1.25, joint=0.0021),
    ]
    source = StateSource(states)
    fake_time = FakeTime()

    evidence = _wait(source, fake_time)

    assert evidence.final_state is states[-1]
    assert evidence.sample_count == 5
    assert evidence.duration_s == pytest.approx(1.0)
    assert evidence.max_joint_delta_rad == pytest.approx(0.0002)
    assert evidence.goal_error_rad == pytest.approx(0.0021)
    assert sum(fake_time.sleep_calls) == pytest.approx(1.25)


def test_goal_error_never_becomes_stationary_evidence() -> None:
    source = StateSource(
        [_state(index * 0.25, joint=0.02) for index in range(3)]
    )
    fake_time = FakeTime()

    with pytest.raises(StationarityTimeoutError, match="goal error"):
        _wait(
            source,
            fake_time,
            settle_time_s=0.25,
            timeout_s=0.5,
        )


def test_timeout_cannot_truncate_the_required_settle_window() -> None:
    source = StateSource([_state(index * 0.25) for index in range(3)])
    fake_time = FakeTime()

    with pytest.raises(StationarityTimeoutError, match="robot_mode='IDLE'.*goal_error="):
        _wait(source, fake_time, settle_time_s=1.0, timeout_s=0.5)


def test_timeout_reports_last_running_controller_state() -> None:
    states = [
        replace(
            _state(index * 0.25, joint=0.02),
            robot_mode="RUNNING",
            runtime_state="PLAYING",
            actual_joint_velocity_rad_s=np.full(6, 0.003),
            target_joint_velocity_rad_s=np.full(6, 0.002),
            actual_tcp_velocity=np.array([0.001, 0.0, 0.0, 0.0, 0.002, 0.0]),
        )
        for index in range(3)
    ]
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(
        StationarityTimeoutError,
        match=(
            "runtime_state='PLAYING'.*controller_stopped=False.*"
            "actual_joint_speed_max=0.003"
        ),
    ):
        _wait(source, fake_time, settle_time_s=0.25, timeout_s=0.5)


def test_clock_progress_cannot_substitute_for_robot_state_time_coverage() -> None:
    states = [_state(index * 0.1) for index in range(6)]
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(StationarityTimeoutError, match="timed out"):
        _wait(
            source,
            fake_time,
            settle_time_s=0.5,
            timeout_s=1.0,
            poll_period_s=0.25,
            maximum_staleness_s=2.0,
        )


def test_controller_time_must_cover_the_complete_settle_window() -> None:
    states = [
        _state(index * 0.25, controller_time_s=index * 0.10)
        for index in range(5)
    ]
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(StationarityTimeoutError, match="timed out"):
        _wait(
            source,
            fake_time,
            settle_time_s=1.0,
            timeout_s=1.0,
            maximum_staleness_s=2.0,
        )


def test_waiter_rejects_controller_feedback_frozen_beyond_limit() -> None:
    states = [
        _state(index * 0.1, controller_time_s=0.0)
        for index in range(3)
    ]
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(StationarityError, match="controller feedback.*stale|frozen"):
        _wait(
            source,
            fake_time,
            settle_time_s=0.5,
            timeout_s=1.0,
            poll_period_s=0.1,
            maximum_staleness_s=0.15,
        )


def test_waiter_rejects_replayed_host_packets_even_if_controller_advances() -> None:
    states = [
        _state(0.0, controller_time_s=index * 0.1)
        for index in range(3)
    ]
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(StationarityError, match="host feedback is stale"):
        _wait(
            source,
            fake_time,
            settle_time_s=0.5,
            timeout_s=1.0,
            poll_period_s=0.1,
            maximum_staleness_s=0.15,
        )


def test_waiter_rejects_a_monotonic_sampling_gap() -> None:
    source = StateSource([_state(0.0), _state(0.3)])
    fake_time = FakeTime()

    with pytest.raises(StationarityError, match="sample gap"):
        _wait(
            source,
            fake_time,
            settle_time_s=0.5,
            timeout_s=1.0,
            poll_period_s=0.3,
            maximum_staleness_s=0.25,
        )


def test_injected_monotonic_clock_rollback_fails_closed() -> None:
    source = StateSource([_state(0.0), _state(0.25)])
    fake_time = FakeTime(now_s=1.0)

    def backwards_sleep(duration_s: float) -> None:
        fake_time.now_s -= duration_s

    with pytest.raises(StationarityError, match="clock moved backwards"):
        wait_until_settled(
            source,
            np.zeros(6),
            settle_time_s=0.5,
            timeout_s=1.0,
            poll_period_s=0.25,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            goal_tolerance_rad=0.001,
            monotonic_clock=fake_time.monotonic,
            sleeper=backwards_sleep,
        )


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ([_state(1.0), _state(0.5)], "robot monotonic timestamp"),
        (
            [
                _state(1.0, controller_time_s=2.0),
                _state(1.5, controller_time_s=1.0),
            ],
            "controller timestamp",
        ),
    ],
)
def test_robot_timestamp_rollback_fails_closed(
    states: list[RobotState],
    message: str,
) -> None:
    source = StateSource(states)
    fake_time = FakeTime()

    with pytest.raises(StationarityError, match=message):
        _wait(source, fake_time)


def test_invalid_goal_contract_is_rejected_before_sampling() -> None:
    source = StateSource([_state(0.0)])
    fake_time = FakeTime()

    with pytest.raises(ValueError, match="finite six-vector"):
        wait_until_settled(
            source,
            [0.0] * 5,
            settle_time_s=1.0,
            timeout_s=2.0,
            poll_period_s=0.25,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            goal_tolerance_rad=0.001,
            monotonic_clock=fake_time.monotonic,
            sleeper=fake_time.sleep,
        )


def test_validate_stationary_trace_returns_fixed_reference_evidence() -> None:
    reference = _state(10.0)
    trace = [
        _state(
            10.2,
            joint=0.0002,
            translation_m=0.0003,
            rotation_rad=0.0004,
        ),
        _state(
            10.4,
            joint=0.0005,
            translation_m=0.0006,
            rotation_rad=0.0007,
        ),
    ]

    evidence = validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
    )

    assert evidence.final_state is trace[-1]
    assert evidence.sample_count == 3
    assert evidence.duration_s == pytest.approx(0.4)
    assert evidence.controller_duration_s == pytest.approx(0.4)
    assert evidence.max_sample_gap_s == pytest.approx(0.2)
    assert evidence.max_joint_delta_rad == pytest.approx(0.0005)
    assert evidence.max_tcp_translation_delta_m == pytest.approx(0.0006)
    assert evidence.max_tcp_rotation_delta_rad == pytest.approx(0.0007)
    assert evidence.goal_error_rad == pytest.approx(0.0005)


def test_trace_rejects_slow_cumulative_drift_from_reference() -> None:
    reference = _state(0.0)
    trace = [
        _state(0.1, joint=0.0006),
        _state(0.2, joint=0.0012),
    ]

    with pytest.raises(StationarityError, match="moved.*sample 2"):
        validate_stationary_trace(
            reference,
            trace,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
        )


def test_long_repeated_kinematic_trace_avoids_pairwise_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = stationarity_module._state_deltas

    def counted(reference: RobotState, current: RobotState):
        nonlocal calls
        calls += 1
        return original(reference, current)

    monkeypatch.setattr(stationarity_module, "_state_deltas", counted)
    trace = [_state(index * 0.05) for index in range(1, 7201)]

    evidence = validate_stationary_trace(
        _state(0.0),
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        maximum_robot_state_staleness_s=0.25,
    )

    assert calls == 0
    assert evidence.sample_count == 7201
    assert evidence.duration_s == pytest.approx(360.0)
    assert evidence.max_sample_gap_s == pytest.approx(0.05)


def test_trace_rejects_feedback_gap_before_pairwise_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stationarity_module,
        "_state_deltas",
        lambda *_args: pytest.fail("pairwise motion validation must not precede freshness"),
    )

    with pytest.raises(StationarityError, match="sample gap") as raised:
        validate_stationary_trace(
            _state(0.0),
            [_state(0.3), _state(0.35)],
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            maximum_robot_state_staleness_s=0.25,
        )

    assert "local=0.3 s" in str(raised.value)
    assert "host=0.3 s" in str(raised.value)
    assert "controller=0.3 s" in str(raised.value)


def test_trace_rejects_sampled_out_and_return_motion() -> None:
    reference = _state(0.0)
    trace = [
        _state(0.1, joint=0.0008),
        _state(0.2, joint=-0.0008),
        _state(0.3, joint=0.0),
    ]

    with pytest.raises(StationarityError, match="moved.*sample 2"):
        validate_stationary_trace(
            reference,
            trace,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
        )


@pytest.mark.parametrize(
    "trace",
    [
        [
            _state(0.1, translation_m=0.002),
            _state(0.2, translation_m=0.002),
        ],
        [
            _state(0.1, rotation_rad=0.002),
            _state(0.2, rotation_rad=0.002),
        ],
    ],
)
def test_trace_rejects_tcp_motion(trace: list[RobotState]) -> None:
    with pytest.raises(StationarityError, match="robot moved"):
        validate_stationary_trace(
            _state(0.0),
            trace,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
        )


def test_trace_rejects_empty_or_time_reversed_evidence() -> None:
    reference = _state(1.0)
    limits = {
        "max_joint_delta_rad": 0.001,
        "max_tcp_translation_delta_m": 0.001,
        "max_tcp_rotation_delta_rad": 0.001,
    }

    with pytest.raises(StationarityError, match="at least two"):
        validate_stationary_trace(reference, [], **limits)
    with pytest.raises(StationarityError, match="timestamp moved backwards"):
        validate_stationary_trace(
            reference,
            [_state(0.5), _state(1.5)],
            **limits,
        )


def test_trace_rejects_one_post_reference_sample() -> None:
    with pytest.raises(StationarityError, match="at least two"):
        validate_stationary_trace(
            _state(0.0),
            [_state(0.1)],
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("robot_mode", "RUNNING", "runtime_state=STOPPED"),
        ("safety_status", "PROTECTIVE_STOP", "NORMAL or REDUCED"),
    ],
)
def test_trace_rejects_non_idle_or_unsafe_controller_state(
    field_name: str,
    value: str,
    message: str,
) -> None:
    reference = replace(_state(0.0), **{field_name: value})

    with pytest.raises(StationarityError, match=message):
        validate_stationary_trace(
            reference,
            [_state(0.1), _state(0.2)],
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
        )


@pytest.mark.parametrize("robot_mode", ("RUNNING", "7"))
def test_trace_accepts_powered_mode_only_with_stopped_runtime(
    robot_mode: str,
) -> None:
    reference = replace(
        _state(0.0),
        robot_mode=robot_mode,
        runtime_state="STOPPED",
    )
    trace = [
        replace(_state(0.1), robot_mode=robot_mode, runtime_state="STOPPED"),
        replace(_state(0.2), robot_mode=robot_mode, runtime_state="STOPPED"),
    ]

    evidence = validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
    )

    assert evidence.sample_count == 3


def test_trace_rejects_controller_freeze_and_monotonic_gap() -> None:
    limits = {
        "max_joint_delta_rad": 0.001,
        "max_tcp_translation_delta_m": 0.001,
        "max_tcp_rotation_delta_rad": 0.001,
        "maximum_robot_state_staleness_s": 0.15,
    }
    with pytest.raises(StationarityError, match="controller feedback.*stale|frozen"):
        validate_stationary_trace(
            _state(0.0, controller_time_s=0.0),
            [
                _state(0.1, controller_time_s=0.0),
                _state(0.2, controller_time_s=0.0),
            ],
            **limits,
        )
    with pytest.raises(StationarityError, match="sample gap"):
        validate_stationary_trace(
            _state(0.0),
            [_state(0.2), _state(0.3)],
            **limits,
        )


def test_trace_rejects_controller_clock_falling_behind_without_freezing() -> None:
    with pytest.raises(StationarityError, match="controller feedback is stale"):
        validate_stationary_trace(
            _state(0.0, controller_time_s=0.0),
            [
                _state(0.05, controller_time_s=0.01),
                _state(0.10, controller_time_s=0.02),
                _state(0.15, controller_time_s=0.03),
                _state(0.20, controller_time_s=0.04),
            ],
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            maximum_robot_state_staleness_s=0.1,
        )


def test_trace_rejects_large_controller_observation_gap() -> None:
    with pytest.raises(StationarityError, match="sample gap"):
        validate_stationary_trace(
            _state(0.0, controller_time_s=0.0),
            [
                _state(0.05, controller_time_s=0.20),
                _state(0.10, controller_time_s=0.25),
            ],
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            maximum_robot_state_staleness_s=0.1,
        )
