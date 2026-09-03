"""Attended, tightly bounded physical motion-envelope commissioning trials."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.settings import AppSettings
from biblade_fusion.devices.robot import (
    EliteArm,
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)
from biblade_fusion.devices.robot._motion_capability import _GUARDED_MOTION_CAPABILITY
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import RobotCommandError
from biblade_fusion.robotics.collision_template import Es68D435iCollisionResources
from biblade_fusion.robotics.motion_preflight import (
    JointMotionPreflight,
    MotionPreflightStatus,
    preflight_linear_joint_motion,
)
from biblade_fusion.robotics.pinocchio_collision import (
    Cs68PinocchioCollisionChecker,
    Es68PinocchioCollisionChecker,
)
from biblade_fusion.storage.motion_envelope_commissioning import (
    MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD,
    StoredCommissioningTrialCandidate,
    read_commissioning_trial_candidate,
)

COMMISSIONING_LIVE_START_TOLERANCE_RAD = 0.001
COMMISSIONING_FIFO_PRIORITY = 10
COMMISSIONING_SDK_FIFO_PRIORITY = os.sched_get_priority_max(os.SCHED_FIFO)
COMMISSIONING_MAXIMUM_EXECUTION_DURATION_S = 3.0
COMMISSIONING_GOAL_HOLD_DURATION_S = 1.0
COMMISSIONING_MAXIMUM_GOAL_ERROR_RAD = 0.002
COMMISSIONING_MAXIMUM_STATIONARY_JOINT_SPEED_RAD_S = 0.001
COMMISSIONING_MAXIMUM_STATIONARY_TCP_SPEED = 0.001
COMMISSIONING_SETTLE_POLL_PERIOD_S = 0.05
COMMISSIONING_MAXIMUM_SETTLE_TIMEOUT_S = 5.0
COMMISSIONING_NOMINAL_TRIAL_KIND = "nominal"
COMMISSIONING_TRACKING_FAULT_TRIAL_KIND = "intentional_tracking_fault"
COMMISSIONING_FAULT_TRACKING_ERROR_RAD = 0.001
COMMISSIONING_FAULT_MAXIMUM_COMMANDS = 10
COMMISSIONING_FAULT_LIVE_START_TOLERANCE_RAD = 0.005
COMMISSIONING_FAULT_MAXIMUM_LIVE_SEGMENT_RAD = 0.01


@dataclass(frozen=True, slots=True)
class PreparedMotionEnvelopeCommissioningTrial:
    stored_candidate: StoredCommissioningTrialCandidate
    preflight: JointMotionPreflight
    collision_checker: Cs68PinocchioCollisionChecker
    approval_prompt: str
    direction: str
    execution_start_joint_positions_rad: tuple[float, ...]
    execution_goal_joint_positions_rad: tuple[float, ...]
    bound_output_path: Path | None = None
    trial_kind: str = COMMISSIONING_NOMINAL_TRIAL_KIND


@dataclass(frozen=True, slots=True)
class MotionEnvelopeCommissioningTrialResult:
    output_path: Path
    ok: bool
    candidate_id: str
    stream_result: StreamServoJResult | None
    maximum_tracking_deviation_rad: tuple[float, ...] | None
    stop_drift_rad: tuple[float, ...] | None
    stop_acknowledgement_s: float | None
    error: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_payload(state: RobotState) -> dict[str, Any]:
    return {
        "monotonic_time_ns": state.monotonic_time_ns,
        "controller_time_s": state.controller_time_s,
        "joint_positions_rad": state.joint_positions_rad.tolist(),
        "base_T_tcp": state.base_t_tcp.matrix.tolist(),
        "robot_mode": state.robot_mode,
        "safety_status": state.safety_status,
        "runtime_state": state.runtime_state,
        "speed_scaling": state.speed_scaling,
        "actual_joint_velocity_rad_s": state.actual_joint_velocity_rad_s.tolist(),
        "target_joint_velocity_rad_s": state.target_joint_velocity_rad_s.tolist(),
        "actual_tcp_velocity": state.actual_tcp_velocity.tolist(),
        "target_tcp_velocity": state.target_tcp_velocity.tolist(),
    }


def _safe_state_speed_metrics(
    state: RobotState,
    *,
    stage: str,
) -> tuple[float, float]:
    if state.robot_mode != "RUNNING" or state.safety_status not in {"NORMAL", "REDUCED"}:
        raise RobotCommandError(
            f"{stage} requires RUNNING and NORMAL/REDUCED, got "
            f"{state.robot_mode}/{state.safety_status}"
        )
    joint_speed = max(
        float(np.max(np.abs(state.actual_joint_velocity_rad_s))),
        float(np.max(np.abs(state.target_joint_velocity_rad_s))),
    )
    tcp_speed = max(
        float(np.max(np.abs(state.actual_tcp_velocity))),
        float(np.max(np.abs(state.target_tcp_velocity))),
    )
    return joint_speed, tcp_speed


def _require_stationary_safe_state(
    state: RobotState,
    *,
    stage: str,
) -> None:
    joint_speed, tcp_speed = _safe_state_speed_metrics(state, stage=stage)
    if joint_speed > COMMISSIONING_MAXIMUM_STATIONARY_JOINT_SPEED_RAD_S:
        raise RobotCommandError(f"{stage} is not stationary ({joint_speed:.6f} rad/s)")
    if tcp_speed > COMMISSIONING_MAXIMUM_STATIONARY_TCP_SPEED:
        raise RobotCommandError(f"{stage} TCP is not stationary ({tcp_speed:.6f})")


def _wait_for_stationary_safe_window(
    arm: Any,
    goal_joint_positions_rad: np.ndarray,
    *,
    settle_time_s: float,
    samples: list[dict[str, Any]],
    timeout_s: float = COMMISSIONING_MAXIMUM_SETTLE_TIMEOUT_S,
    poll_period_s: float = COMMISSIONING_SETTLE_POLL_PERIOD_S,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> tuple[RobotState, dict[str, Any]]:
    """Require a continuous multi-sample stopped window after writeIdle."""

    if (
        not math.isfinite(settle_time_s)
        or settle_time_s <= 0.0
        or not math.isfinite(timeout_s)
        or timeout_s < settle_time_s + poll_period_s
        or not math.isfinite(poll_period_s)
        or poll_period_s <= 0.0
    ):
        raise ValueError("commissioning settle timing is invalid")
    started = float(clock())
    deadline = started + timeout_s
    stable_host_start: float | None = None
    stable_robot_start_ns: int | None = None
    stable_controller_start_s: float | None = None
    stable_sample_count = 0
    maximum_joint_speed = 0.0
    maximum_tcp_speed = 0.0
    previous_state: RobotState | None = None
    last_reason = "no sample"
    while True:
        state = arm.read_state()
        sampled_at = float(clock())
        if not math.isfinite(sampled_at) or sampled_at < started:
            raise RobotCommandError("commissioning settle clock is invalid")
        if previous_state is not None and (
            state.monotonic_time_ns < previous_state.monotonic_time_ns
            or state.controller_time_s < previous_state.controller_time_s
        ):
            raise RobotCommandError("commissioning settle feedback time moved backwards")
        joint_speed, tcp_speed = _safe_state_speed_metrics(
            state,
            stage="commissioning settle",
        )
        goal_error = float(np.max(np.abs(state.joint_positions_rad - goal_joint_positions_rad)))
        samples.append(
            {
                "sampled_at_monotonic_s": sampled_at,
                "goal_error_rad": goal_error,
                "maximum_joint_speed_rad_s": joint_speed,
                "maximum_tcp_speed": tcp_speed,
                "state": _state_payload(state),
            }
        )
        stationary = (
            joint_speed <= COMMISSIONING_MAXIMUM_STATIONARY_JOINT_SPEED_RAD_S
            and tcp_speed <= COMMISSIONING_MAXIMUM_STATIONARY_TCP_SPEED
            and goal_error <= COMMISSIONING_MAXIMUM_GOAL_ERROR_RAD
        )
        if stationary:
            if stable_host_start is None:
                stable_host_start = sampled_at
                stable_robot_start_ns = state.monotonic_time_ns
                stable_controller_start_s = state.controller_time_s
                stable_sample_count = 0
                maximum_joint_speed = 0.0
                maximum_tcp_speed = 0.0
            stable_sample_count += 1
            maximum_joint_speed = max(maximum_joint_speed, joint_speed)
            maximum_tcp_speed = max(maximum_tcp_speed, tcp_speed)
            host_duration = sampled_at - stable_host_start
            robot_duration = (state.monotonic_time_ns - int(stable_robot_start_ns)) / 1e9
            controller_duration = state.controller_time_s - float(stable_controller_start_s)
            if min(host_duration, robot_duration, controller_duration) >= settle_time_s:
                return state, {
                    "sample_count": stable_sample_count,
                    "duration_s": min(host_duration, robot_duration),
                    "controller_duration_s": controller_duration,
                    "goal_error_rad": goal_error,
                    "maximum_joint_speed_rad_s": maximum_joint_speed,
                    "maximum_tcp_speed": maximum_tcp_speed,
                    "total_observed_sample_count": len(samples),
                }
            last_reason = "stable window incomplete"
        else:
            stable_host_start = None
            stable_robot_start_ns = None
            stable_controller_start_s = None
            stable_sample_count = 0
            last_reason = (
                f"goal_error={goal_error:.6f} rad, joint_speed={joint_speed:.6f} rad/s, "
                f"tcp_speed={tcp_speed:.6f}"
            )
        previous_state = state
        remaining = deadline - sampled_at
        if remaining <= 0.0:
            raise RobotCommandError(
                "commissioning failed to establish a stationary settle window: " + last_reason
            )
        sleeper(min(poll_period_s, remaining))


def _require_current_configuration(
    stored: StoredCommissioningTrialCandidate,
    settings: AppSettings,
) -> None:
    configuration = stored.metadata["configuration"]
    if configuration["motion_preflight"] != settings.motion_preflight.model_dump(mode="json"):
        raise ValueError("commissioning candidate motion-preflight configuration has drifted")
    if configuration["collision"] != settings.collision.model_dump(mode="json"):
        raise ValueError("commissioning candidate collision configuration has drifted")
    if configuration["joint_zero_offsets_rad"] != list(settings.kinematics.joint_zero_offsets_rad):
        raise ValueError("commissioning candidate joint-zero offsets have drifted")
    if settings.robot.motion_enabled:
        raise ValueError(
            "commissioning requires production robot.motion_enabled=false; "
            "the dedicated command owns the only temporary motion capability"
        )


def prepare_motion_envelope_commissioning_trial(
    candidate_path: str | Path,
    settings: AppSettings,
) -> PreparedMotionEnvelopeCommissioningTrial:
    """Re-derive the exact mesh-only commissioning segment without touching hardware."""

    stored = read_commissioning_trial_candidate(candidate_path)
    _require_current_configuration(stored, settings)
    candidate = stored.candidate
    if (
        candidate.maximum_candidate_joint_delta_rad
        > MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD + 1e-12
        or candidate.mesh_status != MotionPreflightStatus.CLEAR.value
        or not candidate.mesh_continuous_swept_volume_verified
        or candidate.blocking_reasons
    ):
        raise ValueError("commissioning candidate is not bounded and continuously mesh-clear")
    checker = Es68PinocchioCollisionChecker.from_es68_resources(
        Es68D435iCollisionResources.packaged_template(),
        joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        environment_obstacles=settings.collision.obstacles,
        minimum_clearance_m=settings.collision.minimum_clearance_m,
    )
    preflight = preflight_linear_joint_motion(
        candidate.start_joint_positions_rad,
        candidate.goal_joint_positions_rad,
        collision_checker=checker,
        require_occupancy=False,
        maximum_joint_step_rad=settings.motion_preflight.maximum_joint_step_rad,
        servoj_dt_s=settings.motion_preflight.servoj_dt_s,
        speed_scaling=settings.motion_preflight.speed_scaling,
        velocity_margin=settings.motion_preflight.velocity_margin,
    )
    collision = preflight.collision
    if (
        preflight.status is not MotionPreflightStatus.CLEAR
        or preflight.servoj_stream is None
        or collision is None
        or not collision.continuous_swept_volume_evidence_valid
    ):
        raise ValueError("commissioning segment failed deterministic mesh revalidation")
    reproduced_goal = np.asarray(preflight.servoj_stream.commands[-1], dtype=np.float64)
    if not np.allclose(
        reproduced_goal,
        np.asarray(candidate.goal_joint_positions_rad),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("commissioning ServoJ endpoint differs from sealed candidate")
    return PreparedMotionEnvelopeCommissioningTrial(
        stored_candidate=stored,
        preflight=preflight,
        collision_checker=checker,
        approval_prompt=f"EXECUTE COMMISSION {candidate.candidate_id[:12]}",
        direction="forward",
        execution_start_joint_positions_rad=candidate.start_joint_positions_rad,
        execution_goal_joint_positions_rad=candidate.goal_joint_positions_rad,
    )


def reverse_motion_envelope_commissioning_trial(
    prepared: PreparedMotionEnvelopeCommissioningTrial,
    settings: AppSettings,
) -> PreparedMotionEnvelopeCommissioningTrial:
    """Reverse the same continuously proven candidate path without touching hardware."""

    if prepared.bound_output_path is not None or prepared.direction != "forward":
        raise ValueError("only an unbound forward commissioning trial may be reversed")
    preflight = preflight_linear_joint_motion(
        prepared.execution_goal_joint_positions_rad,
        prepared.execution_start_joint_positions_rad,
        collision_checker=prepared.collision_checker,
        require_occupancy=False,
        maximum_joint_step_rad=settings.motion_preflight.maximum_joint_step_rad,
        servoj_dt_s=settings.motion_preflight.servoj_dt_s,
        speed_scaling=settings.motion_preflight.speed_scaling,
        velocity_margin=settings.motion_preflight.velocity_margin,
    )
    collision = preflight.collision
    if (
        preflight.status is not MotionPreflightStatus.CLEAR
        or preflight.servoj_stream is None
        or collision is None
        or not collision.continuous_swept_volume_evidence_valid
    ):
        raise ValueError("reversed commissioning segment failed continuous mesh proof")
    return replace(
        prepared,
        preflight=preflight,
        approval_prompt=(
            f"EXECUTE COMMISSION {prepared.stored_candidate.candidate.candidate_id[:12]} REVERSE"
        ),
        direction="reverse",
        execution_start_joint_positions_rad=prepared.execution_goal_joint_positions_rad,
        execution_goal_joint_positions_rad=prepared.execution_start_joint_positions_rad,
    )


def intentional_tracking_fault_motion_envelope_commissioning_trial(
    prepared: PreparedMotionEnvelopeCommissioningTrial,
) -> PreparedMotionEnvelopeCommissioningTrial:
    """Bind an expected early tracking abort to an unbound proven segment."""

    if prepared.bound_output_path is not None:
        raise ValueError("only an unbound commissioning trial may select fault mode")
    if prepared.trial_kind != COMMISSIONING_NOMINAL_TRIAL_KIND:
        raise ValueError("commissioning trial mode was already selected")
    return replace(
        prepared,
        approval_prompt=f"{prepared.approval_prompt} TRACKING-FAULT",
        trial_kind=COMMISSIONING_TRACKING_FAULT_TRIAL_KIND,
    )


def bind_motion_envelope_commissioning_output(
    prepared: PreparedMotionEnvelopeCommissioningTrial,
    output_path: str | Path,
) -> PreparedMotionEnvelopeCommissioningTrial:
    """Bind the one-shot confirmation to a unique result directory."""

    output = Path(output_path).resolve()
    output_binding = hashlib.sha256(str(output).encode("utf-8")).hexdigest()[:12]
    candidate_id = prepared.stored_candidate.candidate.candidate_id
    mode_token = (
        " TRACKING-FAULT"
        if prepared.trial_kind == COMMISSIONING_TRACKING_FAULT_TRIAL_KIND
        else ""
    )
    return replace(
        prepared,
        approval_prompt=(
            f"EXECUTE COMMISSION {candidate_id[:12]} {prepared.direction.upper()}"
            f"{mode_token} "
            f"OUTPUT {output_binding}"
        ),
        bound_output_path=output,
    )


@contextmanager
def fifo_scheduler(priority: int = COMMISSIONING_FIFO_PRIORITY) -> Iterator[None]:
    """Enter a low FIFO priority and always restore the caller's scheduler."""

    maximum = os.sched_get_priority_max(os.SCHED_FIFO)
    if type(priority) is not int or not 1 <= priority <= maximum:
        raise ValueError(f"commissioning FIFO priority must be in [1, {maximum}]")
    original_policy = os.sched_getscheduler(0)
    original_parameter = os.sched_getparam(0)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError as exc:
        raise RuntimeError(
            "FIFO scheduling is unavailable; configure a positive PAM rtprio limit "
            "and start a new login session"
        ) from exc
    try:
        yield
    finally:
        os.sched_setscheduler(0, original_policy, original_parameter)


def probe_fifo_scheduler(priority: int = COMMISSIONING_SDK_FIFO_PRIORITY) -> None:
    with fifo_scheduler(priority):
        pass


def _with_goal_hold(stream: ServoJStream) -> ServoJStream:
    hold_count = max(1, math.ceil(COMMISSIONING_GOAL_HOLD_DURATION_S / stream.dt_s))
    return ServoJStream(
        commands=stream.commands + (stream.commands[-1],) * hold_count,
        dt_s=stream.dt_s,
    )


def _tracking_deviation(
    samples: list[dict[str, object]],
) -> tuple[float, float, float, float, float, float] | None:
    if not samples:
        return None
    deviations = [
        np.abs(
            np.asarray(sample["q_cmd"], dtype=np.float64)
            - np.asarray(sample["q_actual"], dtype=np.float64)
        )
        for sample in samples
    ]
    maximum = np.max(np.vstack(deviations), axis=0)
    return tuple(float(value) for value in maximum)  # type: ignore[return-value]


def _write_result(output: Path, payload: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"commissioning result already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        (temporary / "trial.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def execute_motion_envelope_commissioning_trial(
    prepared: PreparedMotionEnvelopeCommissioningTrial,
    settings: AppSettings,
    *,
    operator_id: str,
    confirmation: str,
    output_path: str | Path,
    arm_factory=EliteArm,
) -> MotionEnvelopeCommissioningTrialResult:
    """Execute one attended segment and persist evidence; never grant production motion."""

    operator = operator_id.strip()
    if not operator:
        raise ValueError("operator_id must be non-empty")
    if confirmation.strip() != prepared.approval_prompt:
        raise ValueError(
            f"commissioning confirmation mismatch; expected {prepared.approval_prompt!r}"
        )
    output = Path(output_path).resolve()
    if prepared.bound_output_path != output:
        raise ValueError("commissioning confirmation is not bound to this output path")
    if output.exists():
        raise FileExistsError(f"commissioning result already exists: {output}")
    if prepared.trial_kind not in {
        COMMISSIONING_NOMINAL_TRIAL_KIND,
        COMMISSIONING_TRACKING_FAULT_TRIAL_KIND,
    }:
        raise ValueError(f"unsupported commissioning trial kind {prepared.trial_kind!r}")
    probe_fifo_scheduler()
    candidate = prepared.stored_candidate.candidate
    execution_start = np.asarray(
        prepared.execution_start_joint_positions_rad,
        dtype=np.float64,
    )
    sealed_execution_goal = np.asarray(
        prepared.execution_goal_joint_positions_rad,
        dtype=np.float64,
    )
    execution_goal = sealed_execution_goal.copy()
    effective_robot = settings.robot.model_copy(update={"motion_enabled": True})
    arm = arm_factory(effective_robot)
    tracking_samples: list[dict[str, object]] = []
    stream_result: StreamServoJResult | None = None
    before_state: RobotState | None = None
    stop_request_state: RobotState | None = None
    after_state: RobotState | None = None
    stop_acknowledgement_s: float | None = None
    fault_detection_to_stop_acknowledgement_s: float | None = None
    error: BaseException | None = None
    watchdog_errors: list[str] = []
    settling_samples: list[dict[str, Any]] = []
    settling_evidence: dict[str, Any] | None = None
    goal_error_rad: float | None = None
    trial_started = time.monotonic()
    try:
        arm.connect(with_driver=True)
        before_state = arm.read_state()
        _require_stationary_safe_state(before_state, stage="commissioning start")
        intentional_tracking_fault = (
            prepared.trial_kind == COMMISSIONING_TRACKING_FAULT_TRIAL_KIND
        )
        live_start_tolerance = (
            COMMISSIONING_FAULT_LIVE_START_TOLERANCE_RAD
            if intentional_tracking_fault
            else COMMISSIONING_LIVE_START_TOLERANCE_RAD
        )
        live_error = float(np.max(np.abs(before_state.joint_positions_rad - execution_start)))
        if live_error > live_start_tolerance:
            raise RobotCommandError(f"live start differs from candidate by {live_error:.6f} rad")
        if intentional_tracking_fault:
            live_direction = sealed_execution_goal - before_state.joint_positions_rad
            live_direction_maximum = float(np.max(np.abs(live_direction)))
            if live_direction_maximum <= 1e-12:
                raise RobotCommandError("tracking-fault live start already equals candidate goal")
            live_scale = min(
                1.0,
                COMMISSIONING_FAULT_MAXIMUM_LIVE_SEGMENT_RAD / live_direction_maximum,
            )
            execution_goal = before_state.joint_positions_rad + live_scale * live_direction
        live_delta = float(np.max(np.abs(execution_goal - before_state.joint_positions_rad)))
        maximum_live_delta = (
            COMMISSIONING_FAULT_MAXIMUM_LIVE_SEGMENT_RAD
            if intentional_tracking_fault
            else MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD
        )
        if live_delta > maximum_live_delta + 1e-12:
            raise RobotCommandError(
                "live commissioning delta exceeds its trial bound "
                f"({live_delta:.6f} > {maximum_live_delta:.6f} rad)"
            )
        live_preflight = preflight_linear_joint_motion(
            before_state.joint_positions_rad,
            execution_goal,
            collision_checker=prepared.collision_checker,
            require_occupancy=False,
            maximum_joint_step_rad=settings.motion_preflight.maximum_joint_step_rad,
            servoj_dt_s=settings.motion_preflight.servoj_dt_s,
            speed_scaling=settings.motion_preflight.speed_scaling,
            velocity_margin=settings.motion_preflight.velocity_margin,
        )
        if (
            live_preflight.status is not MotionPreflightStatus.CLEAR
            or live_preflight.servoj_stream is None
            or live_preflight.collision is None
            or not live_preflight.collision.continuous_swept_volume_evidence_valid
        ):
            raise RobotCommandError("live commissioning segment is not continuously mesh-clear")
        commissioning_stream = _with_goal_hold(live_preflight.servoj_stream)
        if intentional_tracking_fault:
            commissioning_stream = ServoJStream(
                commands=commissioning_stream.commands[
                    :COMMISSIONING_FAULT_MAXIMUM_COMMANDS
                ],
                dt_s=commissioning_stream.dt_s,
            )
        runtime_config = ServoJStreamConfig(
            dt_s=commissioning_stream.dt_s,
            tracking_error_rad=(
                COMMISSIONING_FAULT_TRACKING_ERROR_RAD
                if intentional_tracking_fault
                else ServoJStreamConfig().tracking_error_rad
            ),
            max_consecutive_tracking_violations=(
                1
                if intentional_tracking_fault
                else ServoJStreamConfig().max_consecutive_tracking_violations
            ),
            tracking_check_every_n_commands=2,
        )
        runtime_config.validate()

        arm.stop()
        generation, stopped = arm.stop_snapshot
        if not stopped:
            raise RobotCommandError("commissioning failed to establish a local stop latch")
        if not arm.is_enabled:
            arm._guarded_enable_for_servoj_control(
                expected_stop_generation=generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
        deadline = time.monotonic() + COMMISSIONING_MAXIMUM_EXECUTION_DURATION_S

        def deadline_exceeded() -> bool:
            return time.monotonic() > deadline

        def watchdog_stop() -> None:
            try:
                arm._guarded_deadline_stop(capability=_GUARDED_MOTION_CAPABILITY)
            except BaseException as exc:
                watchdog_errors.append(f"{type(exc).__name__}: {exc}")

        watchdog = threading.Timer(
            COMMISSIONING_MAXIMUM_EXECUTION_DURATION_S,
            watchdog_stop,
        )
        watchdog.daemon = True
        watchdog.start()
        try:
            arm._guarded_resume_servoj_control(
                expected_stop_generation=generation,
                capability=_GUARDED_MOTION_CAPABILITY,
                deadline_exceeded=deadline_exceeded,
            )
            arm._guarded_prepare_servoj_stream(
                dt_s=commissioning_stream.dt_s,
                warmup_duration_s=runtime_config.warmup_duration_s,
                expected_stop_generation=generation,
                capability=_GUARDED_MOTION_CAPABILITY,
                deadline_exceeded=deadline_exceeded,
            )
            with fifo_scheduler():
                stream_result = arm._guarded_stream_servoj(
                    commissioning_stream,
                    config=runtime_config,
                    expected_stop_generation=generation,
                    capability=_GUARDED_MOTION_CAPABILITY,
                    tracking_samples=tracking_samples,
                    deadline_exceeded=deadline_exceeded,
                )
        finally:
            watchdog.cancel()
            watchdog.join(timeout=0.5)
        stream_aborted = stream_result is not None and not stream_result.ok
        if stream_aborted:
            fault_detected_at = time.monotonic()
            arm.stop()
            stop_acknowledged_at = time.monotonic()
            stop_acknowledgement_s = stop_acknowledged_at - fault_detected_at
            fault_detection_to_stop_acknowledgement_s = stop_acknowledgement_s
            stop_request_state = arm.read_state()
            settle_goal = stop_request_state.joint_positions_rad.copy()
        else:
            stop_request_state = arm.read_state()
            settle_goal = execution_goal
            stop_started = time.monotonic()
            arm.stop()
            stop_acknowledgement_s = time.monotonic() - stop_started
        goal_error_rad = float(
            np.max(np.abs(stop_request_state.joint_positions_rad - execution_goal))
        )
        after_state, settling_evidence = _wait_for_stationary_safe_window(
            arm,
            settle_goal,
            settle_time_s=settings.robot.settle_time_s,
            samples=settling_samples,
        )
        if intentional_tracking_fault:
            reason = stream_result.abort_reason if stream_result is not None else "no_result"
            if stream_result is None or stream_result.ok or reason != "tracking_error_exceeded":
                raise RobotCommandError(
                    "intentional tracking-fault trial did not produce the expected abort "
                    f"(observed {reason})"
                )
            if stream_result.commands_sent > COMMISSIONING_FAULT_MAXIMUM_COMMANDS:
                raise RobotCommandError(
                    "intentional tracking-fault abort exceeded the command bound "
                    f"({stream_result.commands_sent} > {COMMISSIONING_FAULT_MAXIMUM_COMMANDS})"
                )
        else:
            if stream_result is None or not stream_result.ok:
                reason = stream_result.abort_reason if stream_result is not None else "no_result"
                raise RobotCommandError(f"commissioning ServoJ stream aborted: {reason}")
            if goal_error_rad > COMMISSIONING_MAXIMUM_GOAL_ERROR_RAD:
                raise RobotCommandError(
                    "commissioning endpoint was not reached "
                    f"({goal_error_rad:.6f} rad > {COMMISSIONING_MAXIMUM_GOAL_ERROR_RAD:.6f} rad)"
                )
        if watchdog_errors:
            raise RobotCommandError(
                "commissioning deadline stop was unconfirmed: " + "; ".join(watchdog_errors)
            )
    except BaseException as exc:
        error = exc
        with suppress(BaseException):
            if arm.is_connected:
                arm.stop()
        with suppress(BaseException):
            if arm.is_connected:
                after_state = arm.read_state()
    finally:
        with suppress(BaseException):
            arm.release()

    tracking = _tracking_deviation(tracking_samples)
    stop_drift: tuple[float, ...] | None = None
    if stop_request_state is not None and after_state is not None:
        stop_drift = tuple(
            float(value)
            for value in np.abs(
                after_state.joint_positions_rad - stop_request_state.joint_positions_rad
            )
        )
    candidate_json = prepared.stored_candidate.path / "candidate.json"
    payload = {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.motion_envelope_commissioning_trial",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "production_motion_authorized": False,
        "operator_id": operator,
        "candidate": {
            "path": str(prepared.stored_candidate.path),
            "candidate_json_sha256": _sha256(candidate_json),
            "candidate_id": candidate.candidate_id,
            "direction": prepared.direction,
            "trial_kind": prepared.trial_kind,
            "execution_start_joint_positions_rad": execution_start.tolist(),
            "sealed_execution_goal_joint_positions_rad": sealed_execution_goal.tolist(),
            "execution_goal_joint_positions_rad": execution_goal.tolist(),
            "approval_prompt_sha256": hashlib.sha256(
                prepared.approval_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "effective_limits": {
            "maximum_joint_delta_rad": MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD,
            "live_start_tolerance_rad": COMMISSIONING_LIVE_START_TOLERANCE_RAD,
            "fault_live_start_tolerance_rad": (
                COMMISSIONING_FAULT_LIVE_START_TOLERANCE_RAD
            ),
            "fault_maximum_live_segment_rad": (
                COMMISSIONING_FAULT_MAXIMUM_LIVE_SEGMENT_RAD
            ),
            "fifo_priority": COMMISSIONING_FIFO_PRIORITY,
            "sdk_fifo_priority": COMMISSIONING_SDK_FIFO_PRIORITY,
            "maximum_execution_duration_s": COMMISSIONING_MAXIMUM_EXECUTION_DURATION_S,
            "servoj_warmup_duration_s": ServoJStreamConfig().warmup_duration_s,
            "goal_hold_duration_s": COMMISSIONING_GOAL_HOLD_DURATION_S,
            "settle_window_duration_s": settings.robot.settle_time_s,
            "settle_poll_period_s": COMMISSIONING_SETTLE_POLL_PERIOD_S,
            "maximum_settle_timeout_s": COMMISSIONING_MAXIMUM_SETTLE_TIMEOUT_S,
            "maximum_goal_error_rad": COMMISSIONING_MAXIMUM_GOAL_ERROR_RAD,
            "controller_speed_scaling": effective_robot.default_speed_scaling,
            "servoj_runtime": asdict(runtime_config) if stream_result is not None else None,
            "maximum_fault_trial_commands": (
                COMMISSIONING_FAULT_MAXIMUM_COMMANDS
                if prepared.trial_kind == COMMISSIONING_TRACKING_FAULT_TRIAL_KIND
                else None
            ),
        },
        "before_state": _state_payload(before_state) if before_state is not None else None,
        "stop_request_state": (
            _state_payload(stop_request_state) if stop_request_state is not None else None
        ),
        "after_state": _state_payload(after_state) if after_state is not None else None,
        "stream_result": stream_result.to_dict() if stream_result is not None else None,
        "tracking_samples": tracking_samples,
        "settling_samples": settling_samples,
        "settling_evidence": settling_evidence,
        "maximum_tracking_deviation_rad": list(tracking) if tracking is not None else None,
        "stop_drift_rad": list(stop_drift) if stop_drift is not None else None,
        "stop_acknowledgement_s": stop_acknowledgement_s,
        "fault_detection_to_stop_acknowledgement_s": (
            fault_detection_to_stop_acknowledgement_s
        ),
        "goal_error_rad": goal_error_rad,
        "watchdog_errors": watchdog_errors,
        "elapsed_s": time.monotonic() - trial_started,
        "ok": error is None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
    }
    _write_result(output, payload)
    return MotionEnvelopeCommissioningTrialResult(
        output_path=output,
        ok=error is None,
        candidate_id=candidate.candidate_id,
        stream_result=stream_result,
        maximum_tracking_deviation_rad=tracking,
        stop_drift_rad=stop_drift,
        stop_acknowledgement_s=stop_acknowledgement_s,
        error=payload["error"],
    )


__all__ = [
    "COMMISSIONING_FIFO_PRIORITY",
    "COMMISSIONING_FAULT_MAXIMUM_COMMANDS",
    "COMMISSIONING_FAULT_LIVE_START_TOLERANCE_RAD",
    "COMMISSIONING_FAULT_MAXIMUM_LIVE_SEGMENT_RAD",
    "COMMISSIONING_FAULT_TRACKING_ERROR_RAD",
    "COMMISSIONING_SDK_FIFO_PRIORITY",
    "COMMISSIONING_LIVE_START_TOLERANCE_RAD",
    "COMMISSIONING_MAXIMUM_EXECUTION_DURATION_S",
    "MotionEnvelopeCommissioningTrialResult",
    "PreparedMotionEnvelopeCommissioningTrial",
    "bind_motion_envelope_commissioning_output",
    "execute_motion_envelope_commissioning_trial",
    "fifo_scheduler",
    "intentional_tracking_fault_motion_envelope_commissioning_trial",
    "prepare_motion_envelope_commissioning_trial",
    "probe_fifo_scheduler",
    "reverse_motion_envelope_commissioning_trial",
]
