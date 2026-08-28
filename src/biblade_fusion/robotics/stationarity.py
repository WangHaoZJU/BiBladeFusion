"""Fail-closed stationarity evidence for stop-and-capture robot workflows."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.devices.robot.base import RobotState

_STATIONARY_ROBOT_MODES = {"IDLE"}
_ACCEPTED_SAFETY_STATUSES = {"NORMAL", "REDUCED"}


class StationarityError(RuntimeError):
    """A stationary-state claim could not be established safely."""


class StationarityTimeoutError(StationarityError):
    """The requested bounded-sampling stationary window was not observed in time."""


@runtime_checkable
class RobotStateSource(Protocol):
    """Narrow read-only boundary required by the stationarity monitor."""

    def read_state(self) -> RobotState: ...


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
        float(
            np.max(
                np.abs(
                    current.joint_positions_rad
                    - reference.joint_positions_rad
                )
            )
        ),
        float(
            np.linalg.norm(
                current.base_t_tcp.translation_m
                - reference.base_t_tcp.translation_m
            )
        ),
        _rotation_delta_rad(reference, current),
    )


def _maximum_trace_deltas(
    states: Sequence[RobotState],
) -> tuple[float, float, float]:
    """Return pairwise maxima, including out-and-return motion seen in samples."""

    maximum = (0.0, 0.0, 0.0)
    for current_index, current in enumerate(states):
        for reference in states[:current_index]:
            deltas = _state_deltas(reference, current)
            maximum = tuple(
                max(old, new)
                for old, new in zip(maximum, deltas, strict=True)
            )
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
        sample_gap_s = max(
            local_time - previous_local,
            host_time - previous_host,
            controller_step_s,
        )
        maximum_sample_gap_s = max(maximum_sample_gap_s, sample_gap_s)
        if sample_gap_s > maximum_staleness_s:
            raise StationarityError(
                "robot feedback monotonic sample gap at sample "
                f"{index} is {sample_gap_s:.9g} s, exceeding "
                f"{maximum_staleness_s:.9g} s"
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
        raise StationarityError(
            f"robot state read failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(state, RobotState):
        raise StationarityError("robot state source returned a non-RobotState value")
    _validate_state_contract(state)
    return state


def _validate_state_contract(state: RobotState) -> None:
    timestamp = state.monotonic_time_ns
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, np.integer))
        or timestamp < 0
    ):
        raise StationarityError(
            "robot monotonic timestamp must be a non-negative integer"
        )
    if not math.isfinite(float(state.controller_time_s)):
        raise StationarityError("robot controller timestamp must be finite")
    if (state.base_t_tcp.parent_frame, state.base_t_tcp.child_frame) != (
        "base",
        "tcp",
    ):
        raise StationarityError("stationarity requires robot base_T_tcp state")
    if state.robot_mode.upper() not in _STATIONARY_ROBOT_MODES:
        raise StationarityError(
            "stationarity requires controller robot_mode=IDLE, got "
            f"{state.robot_mode!r}"
        )
    if state.safety_status.upper() not in _ACCEPTED_SAFETY_STATUSES:
        raise StationarityError(
            "stationarity requires NORMAL or REDUCED safety status, got "
            f"{state.safety_status!r}"
        )


def _clock_value(clock: Callable[[], float]) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise StationarityError(
            f"monotonic clock failed: {type(exc).__name__}: {exc}"
        ) from exc
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

    started_at = _clock_value(monotonic_clock)
    deadline = started_at + timeout
    if not math.isfinite(deadline):
        raise ValueError("timeout deadline must be finite")

    previous_state = _read_state(state_source)
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
    if goal_error <= goal_tolerance:
        stable_states.append(previous_state)
        stable_sample_times.append(now)
        stable_started_at = now

    while True:
        if stable_states and stable_started_at is not None:
            clock_duration = now - stable_started_at
            state_duration = (
                stable_states[-1].monotonic_time_ns
                - stable_states[0].monotonic_time_ns
            ) / 1e9
            controller_duration = (
                stable_states[-1].controller_time_s
                - stable_states[0].controller_time_s
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

        current_state = _read_state(state_source)
        sampled_at = _clock_value(monotonic_clock)
        if sampled_at < wake_time:
            raise StationarityError("monotonic clock moved backwards during state read")
        if sampled_at > deadline:
            raise StationarityTimeoutError(
                "stationarity timed out during robot-state sampling"
            )
        _require_nondecreasing_state_time(previous_state, current_state)
        feedback_states.append(current_state)
        feedback_sample_times.append(sampled_at)
        _validate_feedback_freshness(
            feedback_states,
            maximum_staleness_s=maximum_staleness,
            sample_times_s=feedback_sample_times,
        )

        goal_error = _goal_error_rad(current_state, goal)
        if stable_states and stable_started_at is not None:
            candidate_states = (*stable_states, current_state)
            deltas = _maximum_trace_deltas(candidate_states)
            if goal_error <= goal_tolerance and _within_thresholds(deltas, limits):
                stable_states.append(current_state)
                stable_sample_times.append(sampled_at)
                maximum_deltas = deltas
            elif goal_error <= goal_tolerance:
                stable_states = [current_state]
                stable_sample_times = [sampled_at]
                stable_started_at = sampled_at
                maximum_deltas = (0.0, 0.0, 0.0)
            else:
                stable_states = []
                stable_sample_times = []
                stable_started_at = None
                maximum_deltas = (0.0, 0.0, 0.0)
        elif goal_error <= goal_tolerance:
            stable_states = [current_state]
            stable_sample_times = [sampled_at]
            stable_started_at = sampled_at
            maximum_deltas = (0.0, 0.0, 0.0)

        previous_state = current_state
        now = sampled_at


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
    accepted_states: list[RobotState] = [reference]
    maximum_deltas = (0.0, 0.0, 0.0)
    for index, current in enumerate(samples, start=1):
        if not isinstance(current, RobotState):
            raise StationarityError(
                f"stationary trace sample {index} is not a RobotState"
        )
        _validate_state_contract(current)
        _require_nondecreasing_state_time(previous, current)
        # Only pairs ending at ``current`` are new.  Updating the running maxima
        # preserves the exact full-window criterion without recomputing every
        # earlier pair for every sample (O(n^2), rather than O(n^3)).
        current_deltas = (0.0, 0.0, 0.0)
        for earlier in accepted_states:
            pair = _state_deltas(earlier, current)
            current_deltas = tuple(
                max(old, new)
                for old, new in zip(current_deltas, pair, strict=True)
            )
        maximum_deltas = tuple(
            max(old, new)
            for old, new in zip(maximum_deltas, current_deltas, strict=True)
        )
        if not _within_thresholds(maximum_deltas, limits):
            raise StationarityError(
                "robot moved during stationary inference trace at sample "
                f"{index}: joint={maximum_deltas[0]:.9g} rad, "
                f"tcp_translation={maximum_deltas[1]:.9g} m, "
                f"tcp_rotation={maximum_deltas[2]:.9g} rad"
            )
        accepted_states.append(current)
        previous = current

    max_sample_gap_s = _validate_feedback_freshness(
        (reference, *samples),
        maximum_staleness_s=maximum_staleness,
    )

    final_state = samples[-1]
    duration_s = (
        final_state.monotonic_time_ns - reference.monotonic_time_ns
    ) / 1e9
    controller_duration_s = (
        final_state.controller_time_s - reference.controller_time_s
    )
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
