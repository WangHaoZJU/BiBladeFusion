"""HoloRobot-style conservative ES68 joint planning and offline preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.devices.robot.streaming import ServoJStream, ServoJStreamConfig
from biblade_fusion.diagnostics.performance_timing import performance_timed
from biblade_fusion.robotics.occupancy_collision import (
    JointPathOccupancyCollisionReport,
    OccupancyRobotCollisionChecker,
)
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    JointPathMeshCollisionReport,
)
from biblade_fusion.robotics.provenance import robot_stack_provenance


class MotionPreflightStatus(StrEnum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    CHECKER_UNAVAILABLE = "checker_unavailable"
    INVALID_CONTRACT = "invalid_contract"


@dataclass(frozen=True, slots=True)
class JointMotionPreflight:
    """Auditable result; a clear result is still not an execution authorization."""

    status: MotionPreflightStatus
    start_joint_positions_rad: tuple[float, ...]
    goal_joint_positions_rad: tuple[float, ...]
    planning_waypoints: tuple[tuple[float, ...], ...]
    servoj_stream: ServoJStream | None
    collision: JointPathMeshCollisionReport | None
    occupancy: JointPathOccupancyCollisionReport | None
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    servoj_runtime_config: ServoJStreamConfig | None = None
    occupancy_required: bool = True
    swept_mesh_required: bool = True
    continuous_occupancy_sweep_required: bool = True
    approval_required: bool = True

    @property
    def ready_for_approval(self) -> bool:
        return (
            self.status is MotionPreflightStatus.CLEAR
            and self.servoj_stream is not None
            and self.collision is not None
            and self.collision.status is CollisionCheckStatus.CLEAR
            and self.swept_mesh_required
            and self.collision.continuous_swept_volume_evidence_valid
            and self.collision.proof_evidence is not None
            and self.collision.proof_evidence.matches_path(
                self.start_joint_positions_rad,
                self.goal_joint_positions_rad,
            )
            and self.collision.result.diagnostics.get("model") == "elite_es68"
            and bool(self.collision.result.diagnostics.get("robot_geometry_hash"))
            and bool(self.collision.result.diagnostics.get("motion_model_contract_hash"))
            and self.occupancy_required
            and self.occupancy is not None
            and self.occupancy.status is CollisionCheckStatus.CLEAR
            and self.occupancy.evidence is not None
            and self.occupancy.evidence.semantic_attestation_valid
            and self.continuous_occupancy_sweep_required
            and self.occupancy.continuous_swept_volume_evidence_valid
            and self.occupancy.proof_evidence is not None
            and self.occupancy.proof_evidence.matches_path(
                self.start_joint_positions_rad,
                self.goal_joint_positions_rad,
            )
            and bool(self.occupancy.result.diagnostics.get("occupancy_policy_contract_hash"))
            and self.servoj_runtime_config is not None
            and self.approval_required
        )

    @property
    def motion_authorized(self) -> bool:
        return False


def _joint_vector(values: ArrayLike, *, label: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite ES68 six-vector")
    return tuple(float(value) for value in vector)


def _linear_waypoints(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    *,
    maximum_joint_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    if not math.isfinite(maximum_joint_step_rad) or maximum_joint_step_rad <= 0.0:
        raise ValueError("maximum_joint_step_rad must be finite and positive")
    maximum_delta = max(abs(end - begin) for begin, end in zip(start, goal, strict=True))
    segments = max(1, math.ceil(maximum_delta / maximum_joint_step_rad))
    return tuple(
        tuple(
            begin + index / segments * (end - begin) for begin, end in zip(start, goal, strict=True)
        )
        for index in range(segments + 1)
    )


def _velocity_limited_stream(
    waypoints: tuple[tuple[float, ...], ...],
    *,
    maximum_velocity_rad_s: tuple[float, ...],
    dt_s: float,
    speed_scaling: float,
    velocity_margin: float,
) -> ServoJStream:
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("ServoJ dt_s must be finite and positive")
    if not math.isfinite(speed_scaling) or not 0.0 < speed_scaling <= 1.0:
        raise ValueError("ServoJ speed scaling must be finite and in (0, 1]")
    if not math.isfinite(velocity_margin) or not 0.0 < velocity_margin <= 1.0:
        raise ValueError("ServoJ velocity margin must be finite and in (0, 1]")
    velocities = np.asarray(maximum_velocity_rad_s, dtype=np.float64)
    if velocities.shape != (6,) or not np.isfinite(velocities).all() or np.any(velocities <= 0.0):
        raise ValueError("Joint velocity limits must be a finite positive six-vector")
    maximum_steps = tuple(
        velocity * dt_s * speed_scaling * velocity_margin for velocity in maximum_velocity_rad_s
    )
    commands: list[tuple[float, ...]] = [waypoints[0]]
    for start, goal in zip(waypoints[:-1], waypoints[1:], strict=True):
        count = max(
            1,
            math.ceil(
                max(
                    abs(end - begin) / maximum_steps[index]
                    for index, (begin, end) in enumerate(zip(start, goal, strict=True))
                )
            ),
        )
        commands.extend(
            tuple(
                begin + sample / count * (end - begin)
                for begin, end in zip(start, goal, strict=True)
            )
            for sample in range(1, count + 1)
        )
    stream = ServoJStream(commands=tuple(commands), dt_s=dt_s)
    stream.validate()
    return stream


def validate_preflight_servoj_contract(
    preflight: JointMotionPreflight,
    collision_checker: Cs68PinocchioCollisionChecker,
) -> ServoJStream:
    """Reproduce the exact deterministic ServoJ stream from bound planner inputs.

    A stored :class:`JointMotionPreflight` is an evidence object, not an executable
    command container by construction.  This function independently rebuilds its
    waypoints and velocity-limited command stream using the active ES68 kinematic
    model.  Any caller-created or deserialisation-induced command detour therefore
    fails closed before an execution permit can be issued.
    """

    if preflight.diagnostics.get("planner") != "holorobot_conservative_linear_joint":
        raise ValueError("Unsupported motion preflight planner contract")
    if preflight.diagnostics.get("trajectory_generator") != "holorobot_velocity_limited_servoj":
        raise ValueError("Unsupported motion trajectory-generator contract")

    def diagnostic_float(name: str) -> float:
        raw = preflight.diagnostics.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.number)):
            raise ValueError(f"Motion preflight diagnostic {name} is not numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Motion preflight diagnostic {name} is not finite")
        return value

    maximum_step = diagnostic_float("maximum_joint_step_rad")
    dt_s = diagnostic_float("servoj_dt_s")
    speed_scaling = diagnostic_float("speed_scaling")
    velocity_margin = diagnostic_float("velocity_margin")
    freshness_margin = diagnostic_float("execution_freshness_margin_s")
    if freshness_margin < 0.0:
        raise ValueError("Execution freshness margin must be non-negative")
    raw_uncertainty = preflight.diagnostics.get("accepted_joint_uncertainty_rad")
    uncertainty = np.asarray(raw_uncertainty, dtype=np.float64)
    if uncertainty.shape != (6,) or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0.0):
        raise ValueError("Motion preflight accepted uncertainty is invalid")
    acceptance_id = preflight.diagnostics.get("motion_envelope_acceptance_id")
    acceptance_metadata_sha256 = preflight.diagnostics.get("motion_envelope_metadata_sha256")
    if np.any(uncertainty > 0.0):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (acceptance_id, acceptance_metadata_sha256)
        ):
            raise ValueError("Motion preflight acceptance identity is invalid")
    elif acceptance_id is not None or acceptance_metadata_sha256 is not None:
        raise ValueError("Zero motion uncertainty cannot carry acceptance hashes")
    uncertainty_tuple = tuple(float(value) for value in uncertainty)
    mesh_proof = preflight.collision.proof_evidence if preflight.collision is not None else None
    if mesh_proof is not None and (
        mesh_proof.accepted_joint_uncertainty_rad != uncertainty_tuple
        or mesh_proof.motion_envelope_acceptance_id != acceptance_id
        or mesh_proof.motion_envelope_metadata_sha256 != acceptance_metadata_sha256
    ):
        raise ValueError("Motion preflight mesh proof envelope differs from diagnostics")
    occupancy_proof = (
        preflight.occupancy.proof_evidence if preflight.occupancy is not None else None
    )
    if occupancy_proof is not None and (
        occupancy_proof.accepted_joint_uncertainty_rad != uncertainty_tuple
        or occupancy_proof.motion_envelope_acceptance_id != acceptance_id
        or occupancy_proof.motion_envelope_metadata_sha256 != acceptance_metadata_sha256
    ):
        raise ValueError("Motion preflight occupancy proof envelope differs from diagnostics")

    start = _joint_vector(
        preflight.start_joint_positions_rad,
        label="motion preflight start",
    )
    goal = _joint_vector(
        preflight.goal_joint_positions_rad,
        label="motion preflight goal",
    )
    expected_waypoints = _linear_waypoints(
        start,
        goal,
        maximum_joint_step_rad=maximum_step,
    )
    if preflight.planning_waypoints != expected_waypoints:
        raise ValueError("Motion preflight waypoints do not reproduce")
    expected_stream = _velocity_limited_stream(
        expected_waypoints,
        maximum_velocity_rad_s=(collision_checker.kinematic_model.joint_velocity_limits_rad_s()),
        dt_s=dt_s,
        speed_scaling=speed_scaling,
        velocity_margin=velocity_margin,
    )
    stream = preflight.servoj_stream
    if stream is None:
        raise ValueError("Motion preflight lacks a ServoJ stream")
    stream.validate()
    if stream != expected_stream:
        raise ValueError("Motion preflight ServoJ stream does not reproduce")
    runtime_config = preflight.servoj_runtime_config
    if runtime_config is None:
        raise ValueError("Motion preflight lacks a ServoJ runtime config")
    runtime_config.validate()
    if runtime_config.dt_s != expected_stream.dt_s:
        raise ValueError("Motion preflight runtime dt_s differs from its stream")

    expected_duration = max(0, len(expected_stream.commands) - 1) * expected_stream.dt_s
    expected_horizon = expected_duration + freshness_margin
    for name, expected in (
        ("planned_servoj_duration_s", expected_duration),
        ("required_freshness_horizon_s", expected_horizon),
    ):
        recorded = diagnostic_float(name)
        if not math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Motion preflight diagnostic {name} does not reproduce")
    return expected_stream


@performance_timed("planning.preflight_linear_joint_motion")
def preflight_linear_joint_motion(
    start_joint_positions_rad: ArrayLike,
    goal_joint_positions_rad: ArrayLike,
    *,
    collision_checker: Cs68PinocchioCollisionChecker | None,
    checker_unavailable_reason: str = "checker_unavailable",
    occupancy_checker: OccupancyRobotCollisionChecker | None = None,
    require_occupancy: bool = True,
    require_swept_mesh: bool = True,
    require_continuous_occupancy_sweep: bool = True,
    maximum_joint_step_rad: float = 0.02,
    servoj_dt_s: float = 0.004,
    speed_scaling: float = 0.08,
    velocity_margin: float = 0.8,
    execution_freshness_margin_s: float = 1.0,
    servoj_runtime_config: ServoJStreamConfig | None = None,
    accepted_joint_uncertainty_rad: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    motion_envelope_acceptance_id: str | None = None,
    motion_envelope_metadata_sha256: str | None = None,
) -> JointMotionPreflight:
    """Plan, collision-check, and time-parameterize one conservative linear joint leg."""

    start = _joint_vector(start_joint_positions_rad, label="motion start")
    goal = _joint_vector(goal_joint_positions_rad, label="motion goal")
    waypoints = _linear_waypoints(
        start,
        goal,
        maximum_joint_step_rad=maximum_joint_step_rad,
    )
    freshness_margin_s = float(execution_freshness_margin_s)
    if not math.isfinite(freshness_margin_s) or freshness_margin_s < 0.0:
        raise ValueError("execution_freshness_margin_s must be finite and non-negative")
    runtime_config = servoj_runtime_config or ServoJStreamConfig(
        dt_s=servoj_dt_s,
        tracking_check_every_n_commands=2,
    )
    runtime_config.validate()
    if runtime_config.dt_s != servoj_dt_s:
        raise ValueError("Preflight ServoJ runtime config dt_s differs from servoj_dt_s")
    if (
        occupancy_checker is not None
        and not any(float(value) != 0.0 for value in accepted_joint_uncertainty_rad)
        and motion_envelope_acceptance_id is None
        and motion_envelope_metadata_sha256 is None
        and any(value > 0.0 for value in occupancy_checker.accepted_joint_uncertainty_rad)
    ):
        accepted_joint_uncertainty_rad = (
            occupancy_checker.accepted_joint_uncertainty_rad
        )
        motion_envelope_acceptance_id = occupancy_checker.motion_envelope_acceptance_id
        motion_envelope_metadata_sha256 = (
            occupancy_checker.motion_envelope_metadata_sha256
        )
    uncertainty = np.asarray(accepted_joint_uncertainty_rad, dtype=np.float64)
    if uncertainty.shape != (6,) or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0.0):
        raise ValueError("accepted_joint_uncertainty_rad must be a non-negative six-vector")
    acceptance_id = motion_envelope_acceptance_id
    acceptance_metadata_sha256 = motion_envelope_metadata_sha256
    if np.any(uncertainty > 0.0):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (acceptance_id, acceptance_metadata_sha256)
        ):
            raise ValueError("Non-zero motion uncertainty requires acceptance/metadata SHA-256s")
    elif acceptance_id is not None or acceptance_metadata_sha256 is not None:
        raise ValueError("Motion-envelope hashes cannot bind a zero uncertainty vector")
    uncertainty_tuple = tuple(float(value) for value in uncertainty)
    diagnostics = {
        "planner": "holorobot_conservative_linear_joint",
        "trajectory_generator": "holorobot_velocity_limited_servoj",
        "maximum_joint_step_rad": maximum_joint_step_rad,
        "servoj_dt_s": servoj_dt_s,
        "speed_scaling": speed_scaling,
        "velocity_margin": velocity_margin,
        "execution_freshness_margin_s": freshness_margin_s,
        "require_occupancy": bool(require_occupancy),
        "require_swept_mesh": bool(require_swept_mesh),
        "require_continuous_occupancy_sweep": bool(require_continuous_occupancy_sweep),
        "motion_envelope_acceptance_id": acceptance_id,
        "motion_envelope_metadata_sha256": acceptance_metadata_sha256,
        "accepted_joint_uncertainty_rad": list(uncertainty_tuple),
        "provenance": robot_stack_provenance(),
        "motion_authorized": False,
    }
    unavailable_reason = str(checker_unavailable_reason).strip()
    if not unavailable_reason:
        raise ValueError("checker_unavailable_reason must be non-empty")
    if collision_checker is None:
        return JointMotionPreflight(
            status=MotionPreflightStatus.CHECKER_UNAVAILABLE,
            start_joint_positions_rad=start,
            goal_joint_positions_rad=goal,
            planning_waypoints=waypoints,
            servoj_stream=None,
            collision=None,
            occupancy=None,
            blocking_reasons=(unavailable_reason,),
            warnings=(),
            diagnostics=diagnostics,
            occupancy_required=bool(require_occupancy),
            swept_mesh_required=bool(require_swept_mesh),
            continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
        )
    collision = collision_checker.check_path(
        start,
        goal,
        maximum_joint_step_rad=maximum_joint_step_rad,
        maximum_joint_path_deviation_rad=uncertainty_tuple,
        motion_envelope_acceptance_id=acceptance_id,
        motion_envelope_metadata_sha256=acceptance_metadata_sha256,
    )
    if collision.status is not CollisionCheckStatus.CLEAR:
        reasons = collision.result.blocking_reasons or (
            f"collision_status:{collision.status.value}",
        )
        return JointMotionPreflight(
            status=MotionPreflightStatus.BLOCKED,
            start_joint_positions_rad=start,
            goal_joint_positions_rad=goal,
            planning_waypoints=waypoints,
            servoj_stream=None,
            collision=collision,
            occupancy=None,
            blocking_reasons=reasons,
            warnings=(),
            diagnostics=diagnostics,
            occupancy_required=bool(require_occupancy),
            swept_mesh_required=bool(require_swept_mesh),
            continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
        )
    if require_swept_mesh and not (
        collision.continuous_swept_volume_evidence_valid
        and collision.proof_evidence is not None
        and collision.proof_evidence.matches_path(start, goal)
    ):
        return JointMotionPreflight(
            status=MotionPreflightStatus.BLOCKED,
            start_joint_positions_rad=start,
            goal_joint_positions_rad=goal,
            planning_waypoints=waypoints,
            servoj_stream=None,
            collision=collision,
            occupancy=None,
            blocking_reasons=("continuous_swept_mesh_unavailable",),
            warnings=("discrete_mesh_samples_are_diagnostic_only",),
            diagnostics=diagnostics,
            occupancy_required=bool(require_occupancy),
            swept_mesh_required=True,
            continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
        )
    velocities = collision_checker.kinematic_model.joint_velocity_limits_rad_s()
    stream = _velocity_limited_stream(
        waypoints,
        maximum_velocity_rad_s=velocities,
        dt_s=servoj_dt_s,
        speed_scaling=speed_scaling,
        velocity_margin=velocity_margin,
    )
    stream_duration_s = max(0, len(stream.commands) - 1) * stream.dt_s
    required_freshness_horizon_s = stream_duration_s + freshness_margin_s
    diagnostics["planned_servoj_duration_s"] = stream_duration_s
    diagnostics["required_freshness_horizon_s"] = required_freshness_horizon_s
    occupancy: JointPathOccupancyCollisionReport | None = None
    if occupancy_checker is not None:
        if (
            occupancy_checker.accepted_joint_uncertainty_rad != uncertainty_tuple
            or occupancy_checker.motion_envelope_acceptance_id != acceptance_id
        ):
            raise ValueError("Mesh and occupancy preflight motion envelopes differ")
        occupancy = occupancy_checker.check_path(
            start,
            goal,
            maximum_joint_step_rad=maximum_joint_step_rad,
            required_freshness_horizon_s=required_freshness_horizon_s,
        )
        if occupancy.status is not CollisionCheckStatus.CLEAR:
            reasons = occupancy.result.blocking_reasons or (
                f"occupancy_status:{occupancy.status.value}",
            )
            return JointMotionPreflight(
                status=MotionPreflightStatus.BLOCKED,
                start_joint_positions_rad=start,
                goal_joint_positions_rad=goal,
                planning_waypoints=waypoints,
                servoj_stream=None,
                collision=collision,
                occupancy=occupancy,
                blocking_reasons=reasons,
                warnings=(),
                diagnostics=diagnostics,
                occupancy_required=bool(require_occupancy),
                swept_mesh_required=bool(require_swept_mesh),
                continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
            )
        if require_continuous_occupancy_sweep and not (
            occupancy.continuous_swept_volume_evidence_valid
            and occupancy.proof_evidence is not None
            and occupancy.proof_evidence.matches_path(start, goal)
        ):
            return JointMotionPreflight(
                status=MotionPreflightStatus.BLOCKED,
                start_joint_positions_rad=start,
                goal_joint_positions_rad=goal,
                planning_waypoints=waypoints,
                servoj_stream=None,
                collision=collision,
                occupancy=occupancy,
                blocking_reasons=("continuous_swept_occupancy_unavailable",),
                warnings=("discrete_occupancy_samples_are_diagnostic_only",),
                diagnostics=diagnostics,
                occupancy_required=bool(require_occupancy),
                swept_mesh_required=bool(require_swept_mesh),
                continuous_occupancy_sweep_required=True,
            )
    elif require_occupancy:
        return JointMotionPreflight(
            status=MotionPreflightStatus.CHECKER_UNAVAILABLE,
            start_joint_positions_rad=start,
            goal_joint_positions_rad=goal,
            planning_waypoints=waypoints,
            servoj_stream=None,
            collision=collision,
            occupancy=None,
            blocking_reasons=("occupancy_checker_unavailable",),
            warnings=(),
            diagnostics=diagnostics,
            occupancy_required=True,
            swept_mesh_required=bool(require_swept_mesh),
            continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
        )
    return JointMotionPreflight(
        status=MotionPreflightStatus.CLEAR,
        start_joint_positions_rad=start,
        goal_joint_positions_rad=goal,
        planning_waypoints=waypoints,
        servoj_stream=stream,
        collision=collision,
        occupancy=occupancy,
        blocking_reasons=(),
        warnings=(
            "acceleration_limits_unavailable",
            *(() if require_occupancy else ("occupancy_disabled_offline_diagnostic_only",)),
            *(
                ()
                if require_swept_mesh
                else ("continuous_swept_mesh_disabled_offline_diagnostic_only",)
            ),
            *(
                ()
                if require_continuous_occupancy_sweep
                else ("continuous_swept_occupancy_disabled_offline_diagnostic_only",)
            ),
            *(
                ()
                if require_occupancy
                else ("continuous_swept_occupancy_unavailable_offline_diagnostic_only",)
            ),
            *(
                ()
                if (
                    occupancy is not None
                    and occupancy.evidence is not None
                    and occupancy.evidence.semantic_attestation_valid
                )
                else ("occupancy_semantic_attestation_unavailable_diagnostic_only",)
            ),
        ),
        diagnostics=diagnostics,
        servoj_runtime_config=runtime_config,
        occupancy_required=bool(require_occupancy),
        swept_mesh_required=bool(require_swept_mesh),
        continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
    )
