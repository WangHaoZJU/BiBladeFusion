"""Fail-closed stationarity evidence for stop-and-capture robot workflows."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.devices.robot.base import RobotState

_STATIONARY_ROBOT_MODES = {"IDLE", "3"}
_POWERED_STATIONARY_ROBOT_MODES = {"RUNNING", "7"}
_ACCEPTED_SAFETY_STATUSES = {"NORMAL", "REDUCED"}
_BOOTSTRAP_SAFE_ROBOT_MODES = {"IDLE", "POWER_OFF", "RUNNING", "3", "5", "7"}
_BOOTSTRAP_STOPPED_RUNTIME_STATES = {"STOPPED", "3"}


class StationarityError(RuntimeError):
    """A stationary-state claim could not be established safely."""


class StationarityTimeoutError(StationarityError):
    """The requested bounded-sampling stationary window was not observed in time."""


@runtime_checkable
class RobotStateSource(Protocol):
    """Narrow read-only boundary required by the stationarity monitor."""

    def read_state(self) -> RobotState: ...


@runtime_checkable
class StopLatchedRobotStateSource(RobotStateSource, Protocol):
    """Read-only state plus the driver's atomic software stop generation."""

    @property
    def stop_snapshot(self) -> tuple[int, bool]: ...


@dataclass(frozen=True, slots=True)
class StationarityEvidence:
    """Immutable evidence for one accepted bounded-sampling stationary interval.

    ``sample_count`` counts only the samples in the accepted interval.  Samples
    observed while waiting for the robot to reach the goal, or before a window
    reset, are deliberately excluded from the evidence.
    """

    final_state: RobotState
    sample_count: int
    duration_s: float
    controller_duration_s: float
    max_sample_gap_s: float
    max_joint_delta_rad: float
    max_tcp_translation_delta_m: float
    max_tcp_rotation_delta_rad: float
    goal_error_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.final_state, RobotState):
            raise TypeError("final_state must be a RobotState")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        values = (
            self.duration_s,
            self.controller_duration_s,
            self.max_sample_gap_s,
            self.max_joint_delta_rad,
            self.max_tcp_translation_delta_m,
            self.max_tcp_rotation_delta_rad,
            self.goal_error_rad,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("stationarity evidence metrics must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BootstrapSafeStateEvidence:
    """Multi-channel evidence for the unpowered startup stop boundary."""

    stationarity: StationarityEvidence
    stop_generation: int
    runtime_state: str
    robot_mode: str
    safety_status: str
    max_actual_joint_velocity_rad_s: float
    max_target_joint_velocity_rad_s: float
    max_actual_tcp_linear_velocity_m_s: float
    max_actual_tcp_angular_velocity_rad_s: float
    max_target_tcp_linear_velocity_m_s: float
    max_target_tcp_angular_velocity_rad_s: float
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.stop_generation) is not int or self.stop_generation < 1:
            raise ValueError("bootstrap stop generation must be a positive integer")
        labels = (self.runtime_state, self.robot_mode, self.safety_status)
        if any(not str(value).strip() for value in labels):
            raise ValueError("bootstrap controller-state labels must be non-empty")
        metrics = (
            self.max_actual_joint_velocity_rad_s,
            self.max_target_joint_velocity_rad_s,
            self.max_actual_tcp_linear_velocity_m_s,
            self.max_actual_tcp_angular_velocity_rad_s,
            self.max_target_tcp_linear_velocity_m_s,
            self.max_target_tcp_angular_velocity_rad_s,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
            raise ValueError("bootstrap stopped-velocity evidence must be finite and non-negative")
        final = self.stationarity.final_state
        payload = {
            "schema": "biblade_fusion.bootstrap_safe_state_evidence.v1",
            "stop_generation": self.stop_generation,
            "runtime_state": self.runtime_state,
            "robot_mode": self.robot_mode,
            "safety_status": self.safety_status,
            "sample_count": self.stationarity.sample_count,
            "duration_s": self.stationarity.duration_s,
            "controller_duration_s": self.stationarity.controller_duration_s,
            "max_sample_gap_s": self.stationarity.max_sample_gap_s,
            "max_joint_delta_rad": self.stationarity.max_joint_delta_rad,
            "max_tcp_translation_delta_m": (
                self.stationarity.max_tcp_translation_delta_m
            ),
            "max_tcp_rotation_delta_rad": self.stationarity.max_tcp_rotation_delta_rad,
            "final_monotonic_time_ns": final.monotonic_time_ns,
            "final_controller_time_s": final.controller_time_s,
            "final_joint_positions_rad": final.joint_positions_rad.tolist(),
            "stopped_velocity_maxima": list(metrics),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "evidence_sha256", hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class _Thresholds:
    max_joint_delta_rad: float
    max_tcp_translation_delta_m: float
    max_tcp_rotation_delta_rad: float


def _nonnegative_finite(value: float, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _positive_finite(value: float, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _thresholds(
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
) -> _Thresholds:
    return _Thresholds(
        _nonnegative_finite(
            max_joint_delta_rad,
            label="max_joint_delta_rad",
        ),
        _nonnegative_finite(
            max_tcp_translation_delta_m,
            label="max_tcp_translation_delta_m",
        ),
        _nonnegative_finite(
            max_tcp_rotation_delta_rad,
            label="max_tcp_rotation_delta_rad",
        ),
    )


def _joint_vector(values: ArrayLike, *, label: str) -> np.ndarray:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite six-vector")
    return vector


def _goal_error_rad(state: RobotState, goal: np.ndarray | None) -> float:
    if goal is None:
        return 0.0
    return float(np.max(np.abs(state.joint_positions_rad - goal)))


def _rotation_delta_rad(reference: RobotState, current: RobotState) -> float:
    relative = reference.base_t_tcp.rotation.T @ current.base_t_tcp.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def _state_deltas(
    reference: RobotState,
    current: RobotState,
) -> tuple[float, float, float]:
    return (
        float(np.max(np.abs(current.joint_positions_rad - reference.joint_positions_rad))),
        float(
            np.linalg.norm(current.base_t_tcp.translation_m - reference.base_t_tcp.translation_m)
        ),
        _rotation_delta_rad(reference, current),
    )


def _kinematic_state_key(state: RobotState) -> tuple[bytes, bytes]:
    """Return an exact key for motion-relevant state, excluding timestamps."""

    joints = np.ascontiguousarray(state.joint_positions_rad, dtype=np.float64)
    pose = np.ascontiguousarray(state.base_t_tcp.matrix, dtype=np.float64)
    return joints.tobytes(), pose.tobytes()


def _maximum_trace_deltas(
    states: Sequence[RobotState],
) -> tuple[float, float, float]:
    """Return pairwise maxima, including out-and-return motion seen in samples."""

    maximum = (0.0, 0.0, 0.0)
    for current_index, current in enumerate(states):
        for reference in states[:current_index]:
            deltas = _state_deltas(reference, current)
            maximum = tuple(max(old, new) for old, new in zip(maximum, deltas, strict=True))
    return maximum


def _within_thresholds(
    deltas: tuple[float, float, float],
    limits: _Thresholds,
) -> bool:
    return (
        deltas[0] <= limits.max_joint_delta_rad
        and deltas[1] <= limits.max_tcp_translation_delta_m
        and deltas[2] <= limits.max_tcp_rotation_delta_rad
    )


def _validate_feedback_freshness(
    states: Sequence[RobotState],
    *,
    maximum_staleness_s: float,
    sample_times_s: Sequence[float] | None = None,
) -> float:
    """Reject controller or host feedback that falls behind local monotonic time."""

    if not states:
        raise StationarityError("robot feedback freshness requires at least one state")
    local_times = (
        tuple(state.monotonic_time_ns / 1e9 for state in states)
        if sample_times_s is None
        else tuple(float(value) for value in sample_times_s)
    )
    if len(local_times) != len(states):
        raise StationarityError("robot feedback states and sample times do not align")
    if not all(math.isfinite(value) for value in local_times):
        raise StationarityError("robot feedback sample times must be finite")

    local_start = local_times[0]
    host_start = states[0].monotonic_time_ns / 1e9
    controller_start = float(states[0].controller_time_s)
    controller_frozen_since = local_start
    previous_local = local_start
    previous_host = host_start
    previous_controller = controller_start
    maximum_sample_gap_s = 0.0
    for index, (state, local_time) in enumerate(
        zip(states[1:], local_times[1:], strict=True),
        start=1,
    ):
        if local_time < previous_local:
            raise StationarityError("robot feedback sample time moved backwards")
        controller_time = float(state.controller_time_s)
        host_time = state.monotonic_time_ns / 1e9
        controller_step_s = controller_time - previous_controller
        local_gap_s = local_time - previous_local
        host_gap_s = host_time - previous_host
        sample_gap_s = max(local_gap_s, host_gap_s, controller_step_s)
        maximum_sample_gap_s = max(maximum_sample_gap_s, sample_gap_s)
        if sample_gap_s > maximum_staleness_s:
            raise StationarityError(
                "robot feedback monotonic sample gap at sample "
                f"{index} is {sample_gap_s:.9g} s, exceeding "
                f"{maximum_staleness_s:.9g} s "
                f"(local={local_gap_s:.9g} s, host={host_gap_s:.9g} s, "
                f"controller={controller_step_s:.9g} s)"
            )
        local_elapsed = local_time - local_start
        host_elapsed = host_time - host_start
        controller_elapsed = controller_time - controller_start
        controller_lag_s = max(
            local_elapsed - controller_elapsed,
            host_elapsed - controller_elapsed,
        )
        host_packet_lag_s = local_elapsed - host_elapsed
        if controller_lag_s > maximum_staleness_s:
            raise StationarityError(
                "robot controller feedback is stale at sample "
                f"{index}: lag={controller_lag_s:.9g} s exceeds "
                f"{maximum_staleness_s:.9g} s"
            )
        if host_packet_lag_s > maximum_staleness_s:
            raise StationarityError(
                "robot host feedback is stale at sample "
                f"{index}: lag={host_packet_lag_s:.9g} s exceeds "
                f"{maximum_staleness_s:.9g} s"
            )

        if controller_time == previous_controller:
            frozen_duration_s = local_time - controller_frozen_since
            if frozen_duration_s > maximum_staleness_s:
                raise StationarityError(
                    "robot controller feedback remained frozen for "
                    f"{frozen_duration_s:.9g} s at sample {index}"
                )
        else:
            controller_frozen_since = local_time
        previous_local = local_time
        previous_host = host_time
        previous_controller = controller_time
    return maximum_sample_gap_s


def _read_state(source: RobotStateSource) -> RobotState:
    try:
        state = source.read_state()
    except Exception as exc:
        raise StationarityError(f"robot state read failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(state, RobotState):
        raise StationarityError("robot state source returned a non-RobotState value")
    _validate_state_contract(state)
    return state


def _read_settling_state(source: RobotStateSource) -> RobotState:
    """Read a safe state while allowing RUNNING/PLAYING to settle after stop."""

    try:
        state = source.read_state()
    except Exception as exc:
        raise StationarityError(f"robot state read failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(state, RobotState):
        raise StationarityError("robot state source returned a non-RobotState value")
    timestamp = state.monotonic_time_ns
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)) or timestamp < 0:
        raise StationarityError("robot monotonic timestamp must be a non-negative integer")
    if not math.isfinite(float(state.controller_time_s)):
        raise StationarityError("robot controller timestamp must be finite")
    if (state.base_t_tcp.parent_frame, state.base_t_tcp.child_frame) != ("base", "tcp"):
        raise StationarityError("stationarity requires robot base_T_tcp state")
    robot_mode = state.robot_mode.strip().upper()
    if robot_mode not in {*_STATIONARY_ROBOT_MODES, *_POWERED_STATIONARY_ROBOT_MODES}:
        raise StationarityError(
            "stationarity requires controller robot_mode=IDLE or RUNNING while settling; "
            f"got robot_mode={state.robot_mode!r}"
        )
    if robot_mode in _POWERED_STATIONARY_ROBOT_MODES and state.runtime_state is None:
        raise StationarityError("powered stationarity requires an RTSI runtime_state")
    if state.safety_status.upper() not in _ACCEPTED_SAFETY_STATUSES:
        raise StationarityError(
            f"stationarity requires NORMAL or REDUCED safety status, got {state.safety_status!r}"
        )
    return state


def _controller_stopped_for_stationarity(state: RobotState) -> bool:
    robot_mode = state.robot_mode.strip().upper()
    if robot_mode in _STATIONARY_ROBOT_MODES:
        return True
    runtime_state = str(state.runtime_state).strip().upper()
    return runtime_state in _BOOTSTRAP_STOPPED_RUNTIME_STATES


def _validate_state_contract(state: RobotState) -> None:
    timestamp = state.monotonic_time_ns
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)) or timestamp < 0:
        raise StationarityError("robot monotonic timestamp must be a non-negative integer")
    if not math.isfinite(float(state.controller_time_s)):
        raise StationarityError("robot controller timestamp must be finite")
    if (state.base_t_tcp.parent_frame, state.base_t_tcp.child_frame) != (
        "base",
        "tcp",
    ):
        raise StationarityError("stationarity requires robot base_T_tcp state")
    robot_mode = state.robot_mode.strip().upper()
    runtime_state = (
        None
        if state.runtime_state is None
        else state.runtime_state.strip().upper()
    )
    if robot_mode in _POWERED_STATIONARY_ROBOT_MODES:
        if runtime_state not in _BOOTSTRAP_STOPPED_RUNTIME_STATES:
            raise StationarityError(
                "stationarity in powered robot_mode RUNNING requires "
                f"runtime_state=STOPPED, got {state.runtime_state!r}"
            )
    elif robot_mode not in _STATIONARY_ROBOT_MODES:
        raise StationarityError(
            "stationarity requires controller robot_mode=IDLE, or RUNNING with "
            f"runtime_state=STOPPED; got robot_mode={state.robot_mode!r}"
        )
    if state.safety_status.upper() not in _ACCEPTED_SAFETY_STATUSES:
        raise StationarityError(
            f"stationarity requires NORMAL or REDUCED safety status, got {state.safety_status!r}"
        )


def _clock_value(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise StationarityError(f"monotonic clock failed: {type(exc).__name__}: {exc}") from exc
    if not math.isfinite(value):
        raise StationarityError("monotonic clock returned a non-finite value")
    return value


def _require_nondecreasing_state_time(
    previous: RobotState,
    current: RobotState,
) -> None:
    if current.monotonic_time_ns < previous.monotonic_time_ns:
        raise StationarityError("robot monotonic timestamp moved backwards")
    if current.controller_time_s < previous.controller_time_s:
        raise StationarityError("robot controller timestamp moved backwards")


def wait_until_settled(
    state_source: RobotStateSource,
    goal_joint_positions_rad: ArrayLike | None,
    *,
    settle_time_s: float,
    timeout_s: float,
    poll_period_s: float,
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
    goal_tolerance_rad: float,
    maximum_robot_state_staleness_s: float = 0.25,
    maximum_stopped_actual_joint_velocity_rad_s: float | None = None,
    maximum_stopped_target_joint_velocity_rad_s: float | None = None,
    maximum_stopped_actual_tcp_linear_velocity_m_s: float | None = None,
    maximum_stopped_actual_tcp_angular_velocity_rad_s: float | None = None,
    maximum_stopped_target_tcp_linear_velocity_m_s: float | None = None,
    maximum_stopped_target_tcp_angular_velocity_rad_s: float | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> StationarityEvidence:
    """Wait for a goal-bound sampled window spanning ``settle_time_s``.

    Full-window pairwise motion is checked rather than only adjacent samples.
    This detects sampled cumulative drift and sampled out-and-return motion, while
    the maximum-staleness gate bounds the observation gaps.  It is not a continuous
    controller-level immobility proof: motion completed entirely between samples is
    a residual risk.  A sampled motion or goal-tolerance violation resets the
    candidate window; clock rollback and timeout fail immediately.  Passing
    ``None`` as the goal disables only the goal gate and records a finite zero goal
    error.
    """

    goal = (
        None
        if goal_joint_positions_rad is None
        else _joint_vector(
            goal_joint_positions_rad,
            label="goal_joint_positions_rad",
        )
    )
    settle_time = _nonnegative_finite(settle_time_s, label="settle_time_s")
    timeout = _positive_finite(timeout_s, label="timeout_s")
    poll_period = _positive_finite(poll_period_s, label="poll_period_s")
    goal_tolerance = _nonnegative_finite(
        goal_tolerance_rad,
        label="goal_tolerance_rad",
    )
    maximum_staleness = _positive_finite(
        maximum_robot_state_staleness_s,
        label="maximum_robot_state_staleness_s",
    )
    limits = _thresholds(
        max_joint_delta_rad,
        max_tcp_translation_delta_m,
        max_tcp_rotation_delta_rad,
    )
    optional_velocity_limits = (
        maximum_stopped_actual_joint_velocity_rad_s,
        maximum_stopped_target_joint_velocity_rad_s,
        maximum_stopped_actual_tcp_linear_velocity_m_s,
        maximum_stopped_actual_tcp_angular_velocity_rad_s,
        maximum_stopped_target_tcp_linear_velocity_m_s,
        maximum_stopped_target_tcp_angular_velocity_rad_s,
    )
    if any(value is not None for value in optional_velocity_limits) and not all(
        value is not None for value in optional_velocity_limits
    ):
        raise ValueError("stopped-velocity limits must be configured as one complete set")
    velocity_limits = (
        None
        if optional_velocity_limits[0] is None
        else tuple(
            _positive_finite(float(value), label="stopped velocity limit")
            for value in optional_velocity_limits
        )
    )

    def velocity_gate(state: RobotState) -> bool:
        if velocity_limits is None:
            return True
        if any(
            value is None
            for value in (
                state.actual_joint_velocity_rad_s,
                state.target_joint_velocity_rad_s,
                state.actual_tcp_velocity,
                state.target_tcp_velocity,
            )
        ):
            raise StationarityError(
                "stopped-state proof lacks mandatory actual/target joint/TCP velocity channels"
            )
        return all(
            observed <= limit
            for observed, limit in zip(
                _bootstrap_velocity_metrics(state),
                velocity_limits,
                strict=True,
            )
        )

    started_at = _clock_value(monotonic_clock)
    deadline = started_at + timeout
    if not math.isfinite(deadline):
        raise ValueError("timeout deadline must be finite")

    previous_state = _read_settling_state(state_source)
    now = _clock_value(monotonic_clock)
    if now < started_at:
        raise StationarityError("monotonic clock moved backwards during initial read")
    if now > deadline:
        raise StationarityTimeoutError("stationarity timed out during initial state read")

    goal_error = _goal_error_rad(previous_state, goal)
    stable_states: list[RobotState] = []
    stable_sample_times: list[float] = []
    stable_started_at: float | None = None
    maximum_deltas = (0.0, 0.0, 0.0)
    feedback_states = [previous_state]
    feedback_sample_times = [now]
    if (
        _controller_stopped_for_stationarity(previous_state)
        and goal_error <= goal_tolerance
        and velocity_gate(previous_state)
    ):
        stable_states.append(previous_state)
        stable_sample_times.append(now)
        stable_started_at = now

    while True:
        if stable_states and stable_started_at is not None:
            clock_duration = now - stable_started_at
            state_duration = (
                stable_states[-1].monotonic_time_ns - stable_states[0].monotonic_time_ns
            ) / 1e9
            controller_duration = (
                stable_states[-1].controller_time_s - stable_states[0].controller_time_s
            )
            if (
                clock_duration >= settle_time
                and state_duration >= settle_time
                and controller_duration >= settle_time
            ):
                return StationarityEvidence(
                    final_state=previous_state,
                    sample_count=len(stable_states),
                    duration_s=min(clock_duration, state_duration),
                    controller_duration_s=controller_duration,
                    max_sample_gap_s=_validate_feedback_freshness(
                        stable_states,
                        maximum_staleness_s=maximum_staleness,
                        sample_times_s=stable_sample_times,
                    ),
                    max_joint_delta_rad=maximum_deltas[0],
                    max_tcp_translation_delta_m=maximum_deltas[1],
                    max_tcp_rotation_delta_rad=maximum_deltas[2],
                    goal_error_rad=goal_error,
                )

        remaining = deadline - now
        if remaining <= 0.0:
            raise StationarityTimeoutError(
                "stationarity timed out before a bounded sampled settled window was "
                f"observed; last goal error was {goal_error:.9g} rad"
            )
        sleep_duration = min(poll_period, remaining)
        try:
            sleeper(sleep_duration)
        except Exception as exc:
            raise StationarityError(
                f"stationarity sleeper failed: {type(exc).__name__}: {exc}"
            ) from exc

        wake_time = _clock_value(monotonic_clock)
        if wake_time < now:
            raise StationarityError("monotonic clock moved backwards while waiting")
        if wake_time == now:
            raise StationarityError("monotonic clock did not advance while waiting")
        if wake_time > deadline:
            raise StationarityTimeoutError(
                "stationarity timed out before the next robot-state sample"
            )

        current_state = _read_settling_state(state_source)
        sampled_at = _clock_value(monotonic_clock)
        if sampled_at < wake_time:
            raise StationarityError("monotonic clock moved backwards during state read")
        if sampled_at > deadline:
            raise StationarityTimeoutError("stationarity timed out during robot-state sampling")
        _require_nondecreasing_state_time(previous_state, current_state)
        feedback_states.append(current_state)
        feedback_sample_times.append(sampled_at)
        _validate_feedback_freshness(
            feedback_states,
            maximum_staleness_s=maximum_staleness,
            sample_times_s=feedback_sample_times,
        )

        goal_error = _goal_error_rad(current_state, goal)
        controller_stopped = _controller_stopped_for_stationarity(current_state)
        if stable_states and stable_started_at is not None:
            candidate_states = (*stable_states, current_state)
            deltas = _maximum_trace_deltas(candidate_states)
            velocity_stopped = controller_stopped and velocity_gate(current_state)
            if (
                goal_error <= goal_tolerance
                and velocity_stopped
                and _within_thresholds(deltas, limits)
            ):
                stable_states.append(current_state)
                stable_sample_times.append(sampled_at)
                maximum_deltas = deltas
            elif goal_error <= goal_tolerance and velocity_stopped:
                stable_states = [current_state]
                stable_sample_times = [sampled_at]
                stable_started_at = sampled_at
                maximum_deltas = (0.0, 0.0, 0.0)
            else:
                stable_states = []
                stable_sample_times = []
                stable_started_at = None
                maximum_deltas = (0.0, 0.0, 0.0)
        elif (
            controller_stopped
            and goal_error <= goal_tolerance
            and velocity_gate(current_state)
        ):
            stable_states = [current_state]
            stable_sample_times = [sampled_at]
            stable_started_at = sampled_at
            maximum_deltas = (0.0, 0.0, 0.0)

        previous_state = current_state
        now = sampled_at


def _bootstrap_stop_snapshot(
    source: StopLatchedRobotStateSource,
    expected_generation: int,
) -> None:
    try:
        snapshot = source.stop_snapshot
    except Exception as exc:
        raise StationarityError(
            f"robot stop snapshot read failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(snapshot, tuple)
        or len(snapshot) != 2
        or type(snapshot[0]) is not int
        or type(snapshot[1]) is not bool
    ):
        raise StationarityError("robot stop snapshot must be an (integer, boolean) tuple")
    if snapshot != (expected_generation, True):
        raise StationarityError(
            "bootstrap stop generation/latch changed while safe state was being proved"
        )


def _read_bootstrap_state(source: RobotStateSource) -> RobotState:
    try:
        state = source.read_state()
    except Exception as exc:
        raise StationarityError(f"robot state read failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(state, RobotState):
        raise StationarityError("robot state source returned a non-RobotState value")
    timestamp = state.monotonic_time_ns
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)) or timestamp < 0:
        raise StationarityError("robot monotonic timestamp must be a non-negative integer")
    if not math.isfinite(float(state.controller_time_s)):
        raise StationarityError("robot controller timestamp must be finite")
    if (state.base_t_tcp.parent_frame, state.base_t_tcp.child_frame) != ("base", "tcp"):
        raise StationarityError("bootstrap stop requires robot base_T_tcp state")
    safety = state.safety_status.strip().upper()
    if safety not in {*_ACCEPTED_SAFETY_STATUSES, "1", "2"}:
        raise StationarityError(
            "bootstrap stop requires NORMAL or REDUCED safety status, "
            f"got {state.safety_status!r}"
        )
    required = {
        "actual_joint_velocity_rad_s": state.actual_joint_velocity_rad_s,
        "target_joint_velocity_rad_s": state.target_joint_velocity_rad_s,
        "actual_tcp_velocity": state.actual_tcp_velocity,
        "target_tcp_velocity": state.target_tcp_velocity,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise StationarityError(
            "bootstrap stop lacks mandatory RTSI velocity channels: " + ", ".join(missing)
        )
    if state.runtime_state is None:
        raise StationarityError("bootstrap stop lacks mandatory RTSI runtime_state")
    return state


def _bootstrap_velocity_metrics(
    state: RobotState,
) -> tuple[float, float, float, float, float, float]:
    actual_joint = np.asarray(state.actual_joint_velocity_rad_s, dtype=np.float64)
    target_joint = np.asarray(state.target_joint_velocity_rad_s, dtype=np.float64)
    actual_tcp = np.asarray(state.actual_tcp_velocity, dtype=np.float64)
    target_tcp = np.asarray(state.target_tcp_velocity, dtype=np.float64)
    return (
        float(np.max(np.abs(actual_joint))),
        float(np.max(np.abs(target_joint))),
        float(np.linalg.norm(actual_tcp[:3])),
        float(np.linalg.norm(actual_tcp[3:])),
        float(np.linalg.norm(target_tcp[:3])),
        float(np.linalg.norm(target_tcp[3:])),
    )


def wait_until_bootstrap_safe_state(
    state_source: StopLatchedRobotStateSource,
    *,
    expected_stop_generation: int,
    settle_time_s: float,
    timeout_s: float,
    poll_period_s: float,
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
    maximum_robot_state_staleness_s: float,
    maximum_stopped_actual_joint_velocity_rad_s: float,
    maximum_stopped_target_joint_velocity_rad_s: float,
    maximum_stopped_actual_tcp_linear_velocity_m_s: float,
    maximum_stopped_actual_tcp_angular_velocity_rad_s: float,
    maximum_stopped_target_tcp_linear_velocity_m_s: float,
    maximum_stopped_target_tcp_angular_velocity_rad_s: float,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> BootstrapSafeStateEvidence:
    """Prove the Dashboard bootstrap STOPPED postcondition from independent RTSI channels.

    Velocity or controller-state observations that are still settling reset the
    candidate window. Missing channels, unsafe safety modes, stale feedback, time
    rollback, and a changed stop generation fail immediately.
    """

    if type(expected_stop_generation) is not int or expected_stop_generation < 1:
        raise ValueError("expected_stop_generation must be a positive integer")
    settle_time = _positive_finite(settle_time_s, label="settle_time_s")
    timeout = _positive_finite(timeout_s, label="timeout_s")
    poll_period = _positive_finite(poll_period_s, label="poll_period_s")
    maximum_staleness = _positive_finite(
        maximum_robot_state_staleness_s,
        label="maximum_robot_state_staleness_s",
    )
    limits = _thresholds(
        max_joint_delta_rad,
        max_tcp_translation_delta_m,
        max_tcp_rotation_delta_rad,
    )
    velocity_limits = tuple(
        _positive_finite(value, label=label)
        for value, label in (
            (
                maximum_stopped_actual_joint_velocity_rad_s,
                "maximum_stopped_actual_joint_velocity_rad_s",
            ),
            (
                maximum_stopped_target_joint_velocity_rad_s,
                "maximum_stopped_target_joint_velocity_rad_s",
            ),
            (
                maximum_stopped_actual_tcp_linear_velocity_m_s,
                "maximum_stopped_actual_tcp_linear_velocity_m_s",
            ),
            (
                maximum_stopped_actual_tcp_angular_velocity_rad_s,
                "maximum_stopped_actual_tcp_angular_velocity_rad_s",
            ),
            (
                maximum_stopped_target_tcp_linear_velocity_m_s,
                "maximum_stopped_target_tcp_linear_velocity_m_s",
            ),
            (
                maximum_stopped_target_tcp_angular_velocity_rad_s,
                "maximum_stopped_target_tcp_angular_velocity_rad_s",
            ),
        )
    )
    started_at = _clock_value(monotonic_clock)
    deadline = started_at + timeout
    stable_states: list[RobotState] = []
    stable_sample_times: list[float] = []
    stable_velocity_maxima = (0.0,) * 6
    maximum_deltas = (0.0, 0.0, 0.0)
    all_states: list[RobotState] = []
    all_sample_times: list[float] = []
    previous_state: RobotState | None = None
    last_conditions = "no sample"

    while True:
        _bootstrap_stop_snapshot(state_source, expected_stop_generation)
        state = _read_bootstrap_state(state_source)
        sampled_at = _clock_value(monotonic_clock)
        if sampled_at < started_at or (all_sample_times and sampled_at < all_sample_times[-1]):
            raise StationarityError("monotonic clock moved backwards during bootstrap stop proof")
        if sampled_at > deadline:
            raise StationarityTimeoutError(
                "bootstrap safe-state proof timed out during robot-state sampling"
            )
        if previous_state is not None:
            _require_nondecreasing_state_time(previous_state, state)
        all_states.append(state)
        all_sample_times.append(sampled_at)
        _validate_feedback_freshness(
            all_states,
            maximum_staleness_s=maximum_staleness,
            sample_times_s=all_sample_times,
        )
        _bootstrap_stop_snapshot(state_source, expected_stop_generation)

        runtime = str(state.runtime_state).strip().upper()
        mode = state.robot_mode.strip().upper()
        velocities = _bootstrap_velocity_metrics(state)
        controller_stopped = (
            runtime in _BOOTSTRAP_STOPPED_RUNTIME_STATES
            and mode in _BOOTSTRAP_SAFE_ROBOT_MODES
        )
        velocities_stopped = all(
            observed <= limit
            for observed, limit in zip(velocities, velocity_limits, strict=True)
        )
        candidate_ok = controller_stopped and velocities_stopped
        last_conditions = (
            f"runtime={runtime}, robot_mode={mode}, velocities={velocities!r}"
        )
        if candidate_ok:
            candidate_states = (*stable_states, state)
            deltas = _maximum_trace_deltas(candidate_states)
            if stable_states and not _within_thresholds(deltas, limits):
                stable_states = [state]
                stable_sample_times = [sampled_at]
                stable_velocity_maxima = velocities
                maximum_deltas = (0.0, 0.0, 0.0)
            else:
                stable_states.append(state)
                stable_sample_times.append(sampled_at)
                maximum_deltas = deltas
                stable_velocity_maxima = tuple(
                    max(old, current)
                    for old, current in zip(
                        stable_velocity_maxima,
                        velocities,
                        strict=True,
                    )
                )
        else:
            stable_states = []
            stable_sample_times = []
            stable_velocity_maxima = (0.0,) * 6
            maximum_deltas = (0.0, 0.0, 0.0)

        if stable_states:
            clock_duration = stable_sample_times[-1] - stable_sample_times[0]
            host_duration = (
                stable_states[-1].monotonic_time_ns
                - stable_states[0].monotonic_time_ns
            ) / 1e9
            controller_duration = (
                stable_states[-1].controller_time_s
                - stable_states[0].controller_time_s
            )
            if min(clock_duration, host_duration, controller_duration) >= settle_time:
                stationarity = StationarityEvidence(
                    final_state=state,
                    sample_count=len(stable_states),
                    duration_s=min(clock_duration, host_duration),
                    controller_duration_s=controller_duration,
                    max_sample_gap_s=_validate_feedback_freshness(
                        stable_states,
                        maximum_staleness_s=maximum_staleness,
                        sample_times_s=stable_sample_times,
                    ),
                    max_joint_delta_rad=maximum_deltas[0],
                    max_tcp_translation_delta_m=maximum_deltas[1],
                    max_tcp_rotation_delta_rad=maximum_deltas[2],
                    goal_error_rad=0.0,
                )
                _bootstrap_stop_snapshot(state_source, expected_stop_generation)
                return BootstrapSafeStateEvidence(
                    stationarity=stationarity,
                    stop_generation=expected_stop_generation,
                    runtime_state=runtime,
                    robot_mode=mode,
                    safety_status=state.safety_status.strip().upper(),
                    max_actual_joint_velocity_rad_s=stable_velocity_maxima[0],
                    max_target_joint_velocity_rad_s=stable_velocity_maxima[1],
                    max_actual_tcp_linear_velocity_m_s=stable_velocity_maxima[2],
                    max_actual_tcp_angular_velocity_rad_s=stable_velocity_maxima[3],
                    max_target_tcp_linear_velocity_m_s=stable_velocity_maxima[4],
                    max_target_tcp_angular_velocity_rad_s=stable_velocity_maxima[5],
                )

        remaining = deadline - sampled_at
        if remaining <= 0.0:
            raise StationarityTimeoutError(
                "bootstrap safe-state proof timed out before a full accepted window; "
                + last_conditions
            )
        sleep_duration = min(poll_period, remaining)
        try:
            sleeper(sleep_duration)
        except Exception as exc:
            raise StationarityError(
                f"bootstrap safe-state sleeper failed: {type(exc).__name__}: {exc}"
            ) from exc
        wake_time = _clock_value(monotonic_clock)
        if wake_time <= sampled_at:
            raise StationarityError("monotonic clock did not advance during bootstrap stop proof")
        if wake_time > deadline:
            raise StationarityTimeoutError(
                "bootstrap safe-state proof timed out before the next robot-state sample"
            )
        previous_state = state


def validate_stationary_trace(
    reference: RobotState,
    trace: Sequence[RobotState],
    *,
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
    maximum_robot_state_staleness_s: float = 0.25,
) -> StationarityEvidence:
    """Validate an inference-period trace against one fixed robot reference.

    Every trace sample is compared pairwise across the full window.  This is
    intentionally stricter than adjacent-sample checks and detects cumulative
    drift present in the samples during a long FoundationStereo inference.  It
    does not claim visibility of motion completed between two samples.
    """

    if not isinstance(reference, RobotState):
        raise TypeError("reference must be a RobotState")
    _validate_state_contract(reference)
    samples = tuple(trace)
    if len(samples) < 2:
        raise StationarityError(
            "stationary inference trace requires at least two post-reference samples"
        )
    limits = _thresholds(
        max_joint_delta_rad,
        max_tcp_translation_delta_m,
        max_tcp_rotation_delta_rad,
    )
    maximum_staleness = _positive_finite(
        maximum_robot_state_staleness_s,
        label="maximum_robot_state_staleness_s",
    )

    previous = reference
    for index, current in enumerate(samples, start=1):
        if not isinstance(current, RobotState):
            raise StationarityError(f"stationary trace sample {index} is not a RobotState")
        _validate_state_contract(current)
        _require_nondecreasing_state_time(previous, current)
        previous = current

    max_sample_gap_s = _validate_feedback_freshness(
        (reference, *samples),
        maximum_staleness_s=maximum_staleness,
    )

    # Repeated controller samples usually differ only by timestamp.  Comparing one
    # representative of each exact kinematic state preserves every pairwise motion
    # delta while keeping a long stopped trace linear in the common case.
    accepted_states: dict[tuple[bytes, bytes], RobotState] = {
        _kinematic_state_key(reference): reference
    }
    maximum_deltas = (0.0, 0.0, 0.0)
    for index, current in enumerate(samples, start=1):
        key = _kinematic_state_key(current)
        if key in accepted_states:
            continue
        current_deltas = (0.0, 0.0, 0.0)
        for earlier in accepted_states.values():
            pair = _state_deltas(earlier, current)
            current_deltas = tuple(
                max(old, new) for old, new in zip(current_deltas, pair, strict=True)
            )
        maximum_deltas = tuple(
            max(old, new) for old, new in zip(maximum_deltas, current_deltas, strict=True)
        )
        if not _within_thresholds(maximum_deltas, limits):
            raise StationarityError(
                "robot moved during stationary inference trace at sample "
                f"{index}: joint={maximum_deltas[0]:.9g} rad, "
                f"tcp_translation={maximum_deltas[1]:.9g} m, "
                f"tcp_rotation={maximum_deltas[2]:.9g} rad"
            )
        accepted_states[key] = current

    final_state = samples[-1]
    duration_s = (final_state.monotonic_time_ns - reference.monotonic_time_ns) / 1e9
    controller_duration_s = final_state.controller_time_s - reference.controller_time_s
    return StationarityEvidence(
        final_state=final_state,
        sample_count=len(samples) + 1,
        duration_s=duration_s,
        controller_duration_s=controller_duration_s,
        max_sample_gap_s=max_sample_gap_s,
        max_joint_delta_rad=maximum_deltas[0],
        max_tcp_translation_delta_m=maximum_deltas[1],
        max_tcp_rotation_delta_rad=maximum_deltas[2],
        goal_error_rad=_goal_error_rad(
            final_state,
            reference.joint_positions_rad,
        ),
    )
