"""HoloRobot-style conservative ES68 joint planning and offline preflight."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.core.planning_deadline import (
    PlanningDeadlineExceeded,
    require_planning_time,
)
from biblade_fusion.devices.robot.streaming import ServoJStream, ServoJStreamConfig
from biblade_fusion.diagnostics.performance_timing import performance_span, performance_timed
from biblade_fusion.robotics.holorobot_joint_planner import (
    HoloRobotJointPlanStatus,
    HoloRobotOmplConfig,
    ompl_available,
    plan_holorobot_rrtconnect,
)
from biblade_fusion.robotics.occupancy_collision import (
    JointPathOccupancyCollisionReport,
    OccupancyRobotCollisionChecker,
)
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckResult,
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    JointPathMeshCollisionReport,
    _joint_path_sha256,
)
from biblade_fusion.robotics.provenance import robot_stack_provenance


class MotionPreflightStatus(StrEnum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    CHECKER_UNAVAILABLE = "checker_unavailable"
    INVALID_CONTRACT = "invalid_contract"


CONTINUOUS_INTERVAL_VALIDATION = "continuous_interval_v1"
HOLOROBOT_SAMPLED_VALIDATION = "holorobot_sampled_joint_v2"
HOLOROBOT_NATIVE_MAX_JOINT_STEP_RAD = 0.1
HOLOROBOT_NATIVE_SEGMENT_SAMPLES = 5
HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD = (
    HOLOROBOT_NATIVE_MAX_JOINT_STEP_RAD / (HOLOROBOT_NATIVE_SEGMENT_SAMPLES - 1)
)


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
    path_validation_mode: str = CONTINUOUS_INTERVAL_VALIDATION
    path_validation_evidence_sha256: str | None = None

    @property
    def ready_for_approval(self) -> bool:
        common = (
            self.status is MotionPreflightStatus.CLEAR
            and self.servoj_stream is not None
            and self.collision is not None
            and self.collision.status is CollisionCheckStatus.CLEAR
            and self.collision.result.diagnostics.get("model") == "elite_es68"
            and bool(self.collision.result.diagnostics.get("robot_geometry_hash"))
            and bool(self.collision.result.diagnostics.get("motion_model_contract_hash"))
            and self.occupancy_required
            and self.occupancy is not None
            and self.occupancy.status is CollisionCheckStatus.CLEAR
            and self.occupancy.evidence is not None
            and self.occupancy.evidence.semantic_attestation_valid
            and bool(self.occupancy.result.diagnostics.get("occupancy_policy_contract_hash"))
            and self.servoj_runtime_config is not None
            and self.approval_required
        )
        if not common:
            return False
        if self.path_validation_mode == CONTINUOUS_INTERVAL_VALIDATION:
            return bool(
                self.swept_mesh_required
                and self.collision is not None
                and self.collision.continuous_swept_volume_evidence_valid
                and self.collision.proof_evidence is not None
                and self.collision.proof_evidence.matches_path(
                    self.start_joint_positions_rad,
                    self.goal_joint_positions_rad,
                )
                and self.continuous_occupancy_sweep_required
                and self.occupancy is not None
                and self.occupancy.continuous_swept_volume_evidence_valid
                and self.occupancy.proof_evidence is not None
                and self.occupancy.proof_evidence.matches_path(
                    self.start_joint_positions_rad,
                    self.goal_joint_positions_rad,
                )
            )
        if self.path_validation_mode == HOLOROBOT_SAMPLED_VALIDATION:
            return _holorobot_sampled_evidence_valid(self)
        return False

    @property
    def motion_authorized(self) -> bool:
        return False


def _joint_vector(values: ArrayLike, *, label: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite ES68 six-vector")
    return tuple(float(value) for value in vector)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _waypoints_sha256(waypoints: tuple[tuple[float, ...], ...]) -> str:
    return _canonical_sha256(
        {
            "schema": "biblade_fusion.joint_waypoints.v1",
            "waypoints": [list(item) for item in waypoints],
        }
    )


def _holorobot_sampled_configurations(
    waypoints: tuple[tuple[float, ...], ...],
    *,
    maximum_sample_step_rad: float = HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD,
) -> tuple[tuple[tuple[float, ...], float], ...]:
    """Match HoloRobot's effective ES68 resolution without double sampling.

    HoloRobot's ES68 profile uses a 0.1-rad planner step and five evenly spaced
    ``check_segment`` states, giving a worst-case 0.025-rad sample interval.
    BiBladeFusion waypoints are already interpolated at 0.02 rad by default, so
    checking every waypoint is slightly denser than HoloRobot.  Subdivide only
    when a caller supplies a coarser waypoint path.
    """

    sample_step = float(maximum_sample_step_rad)
    if not math.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("HoloRobot effective sample step must be finite and positive")
    if not waypoints:
        return ()
    segment_count = max(1, len(waypoints) - 1)
    result: list[tuple[tuple[float, ...], float]] = [(waypoints[0], 0.0)]
    for segment_index, (start, goal) in enumerate(
        zip(waypoints[:-1], waypoints[1:], strict=True)
    ):
        maximum_delta = max(
            abs(end - begin) for begin, end in zip(start, goal, strict=True)
        )
        subdivisions = max(1, math.ceil(maximum_delta / sample_step))
        for sample_index in range(1, subdivisions):
            alpha = sample_index / subdivisions
            configuration = tuple(
                begin + alpha * (end - begin)
                for begin, end in zip(start, goal, strict=True)
            )
            fraction = (segment_index + alpha) / segment_count
            result.append((configuration, fraction))
        result.append((goal, (segment_index + 1) / segment_count))
    return tuple(result)


def _holorobot_sampled_evidence_payload(
    preflight: JointMotionPreflight,
) -> dict[str, Any] | None:
    collision = preflight.collision
    occupancy = preflight.occupancy
    if collision is None or occupancy is None or occupancy.evidence is None:
        return None
    collision_diagnostics = collision.result.diagnostics
    occupancy_diagnostics = occupancy.result.diagnostics
    return {
        "schema": "biblade_fusion.holorobot_sampled_path_evidence.v3",
        "method": HOLOROBOT_SAMPLED_VALIDATION,
        "trajectory_sha256": _joint_path_sha256(
            preflight.start_joint_positions_rad,
            preflight.goal_joint_positions_rad,
        ),
        "planning_waypoints_sha256": _waypoints_sha256(
            preflight.planning_waypoints
        ),
        "maximum_joint_step_rad": collision.maximum_joint_step_rad,
        "occupancy_maximum_joint_step_rad": occupancy.maximum_joint_step_rad,
        "holorobot_native_max_joint_step_rad": HOLOROBOT_NATIVE_MAX_JOINT_STEP_RAD,
        "holorobot_native_segment_samples": HOLOROBOT_NATIVE_SEGMENT_SAMPLES,
        "effective_maximum_sample_step_rad": HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD,
        "mesh_sample_count": collision.sample_count,
        "occupancy_sample_count": occupancy.sample_count,
        "collision_model_binding": [
            collision_diagnostics.get("model"),
            collision_diagnostics.get("collision_model_id"),
            collision_diagnostics.get("robot_geometry_hash"),
            collision_diagnostics.get("motion_model_contract_hash"),
        ],
        "occupancy_binding": list(occupancy.evidence.binding),
        "occupancy_policy_contract_hash": occupancy_diagnostics.get(
            "occupancy_policy_contract_hash"
        ),
        "motion_envelope_acceptance_id": preflight.diagnostics.get(
            "motion_envelope_acceptance_id"
        ),
        "motion_envelope_metadata_sha256": preflight.diagnostics.get(
            "motion_envelope_metadata_sha256"
        ),
        "accepted_joint_uncertainty_rad": preflight.diagnostics.get(
            "accepted_joint_uncertainty_rad"
        ),
    }


def _holorobot_sampled_evidence_valid(preflight: JointMotionPreflight) -> bool:
    collision = preflight.collision
    occupancy = preflight.occupancy
    payload = _holorobot_sampled_evidence_payload(preflight)
    if collision is None or occupancy is None or payload is None:
        return False
    expected_sample_count = len(
        _holorobot_sampled_configurations(preflight.planning_waypoints)
    )
    evidence_sha256 = preflight.path_validation_evidence_sha256
    return bool(
        not preflight.swept_mesh_required
        and not preflight.continuous_occupancy_sweep_required
        and collision.proof_evidence is None
        and not collision.continuous_swept_volume_verified
        and occupancy.proof_evidence is None
        and not occupancy.continuous_swept_volume_verified
        and collision.sample_count == expected_sample_count
        and occupancy.sample_count == expected_sample_count
        and math.isclose(
            collision.maximum_joint_step_rad,
            occupancy.maximum_joint_step_rad,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and collision.result.diagnostics.get("effective_maximum_sample_step_rad")
        == HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD
        and occupancy.result.diagnostics.get("effective_maximum_sample_step_rad")
        == HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD
        and collision.result.diagnostics.get("path_validation_mode")
        == HOLOROBOT_SAMPLED_VALIDATION
        and occupancy.result.diagnostics.get("path_validation_mode")
        == HOLOROBOT_SAMPLED_VALIDATION
        and collision.result.diagnostics.get("sampled_path_verified") is True
        and occupancy.result.diagnostics.get("sampled_path_verified") is True
        and isinstance(evidence_sha256, str)
        and evidence_sha256 == _canonical_sha256(payload)
    )


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
    interior = tuple(
        tuple(
            begin + index / segments * (end - begin) for begin, end in zip(start, goal, strict=True)
        )
        for index in range(1, segments)
    )
    # Preserve the exact current-state and selected-IK tuples.  Computing the
    # boundaries through interpolation can change a component by one ULP and
    # make a valid path fail the identity/hash contract before motion.
    return (start, *interior, goal)


@dataclass(frozen=True, slots=True)
class _TimeParameterizedServoJ:
    stream: ServoJStream
    knot_count: int
    minimum_duration_s: float
    limiting_segment_index: int | None
    limiting_joint_index: int | None
    limiting_constraint: str | None


def _servoj_geometric_knots(
    waypoints: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Remove only redundant collinear samples; preserve every path corner."""

    if len(waypoints) < 2:
        raise ValueError("ServoJ path requires at least two waypoints")
    knots: list[tuple[float, ...]] = [waypoints[0]]
    for middle, following in zip(waypoints[1:-1], waypoints[2:], strict=True):
        previous = knots[-1]
        incoming = np.asarray(middle, dtype=np.float64) - np.asarray(
            previous, dtype=np.float64
        )
        outgoing = np.asarray(following, dtype=np.float64) - np.asarray(
            middle, dtype=np.float64
        )
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= 1e-12:
            continue
        if outgoing_norm > 1e-12 and np.allclose(
            incoming / incoming_norm,
            outgoing / outgoing_norm,
            rtol=0.0,
            atol=1e-9,
        ):
            continue
        knots.append(middle)
    if knots[-1] != waypoints[-1]:
        knots.append(waypoints[-1])
    return tuple(knots)


def _time_parameterized_servoj_stream(
    waypoints: tuple[tuple[float, ...], ...],
    *,
    maximum_velocity_rad_s: tuple[float, ...],
    maximum_acceleration_rad_s2: tuple[float, ...],
    dt_s: float,
    speed_scaling: float,
    velocity_margin: float,
) -> _TimeParameterizedServoJ:
    """Apply HoloRobot's velocity/acceleration duration rule to a joint polyline."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("ServoJ dt_s must be finite and positive")
    if not math.isfinite(speed_scaling) or not 0.0 < speed_scaling <= 1.0:
        raise ValueError("ServoJ speed scaling must be finite and in (0, 1]")
    if not math.isfinite(velocity_margin) or not 0.0 < velocity_margin <= 1.0:
        raise ValueError("ServoJ velocity margin must be finite and in (0, 1]")
    velocities = np.asarray(maximum_velocity_rad_s, dtype=np.float64)
    if velocities.shape != (6,) or not np.isfinite(velocities).all() or np.any(velocities <= 0.0):
        raise ValueError("Joint velocity limits must be a finite positive six-vector")
    accelerations = np.asarray(maximum_acceleration_rad_s2, dtype=np.float64)
    if (
        accelerations.shape != (6,)
        or not np.isfinite(accelerations).all()
        or np.any(accelerations <= 0.0)
    ):
        raise ValueError("Joint acceleration limits must be a finite positive six-vector")
    knots = _servoj_geometric_knots(waypoints)
    scale = speed_scaling * velocity_margin
    commands: list[tuple[float, ...]] = [knots[0]]
    minimum_duration_s = 0.0
    limiting_segment_index: int | None = None
    limiting_joint_index: int | None = None
    limiting_constraint: str | None = None
    largest_segment_minimum_s = -1.0
    for segment_index, (start, goal) in enumerate(
        zip(knots[:-1], knots[1:], strict=True)
    ):
        segment_minimum_s = 0.0
        segment_joint_index: int | None = None
        segment_constraint: str | None = None
        for joint_index, (begin, end) in enumerate(zip(start, goal, strict=True)):
            delta = abs(end - begin)
            if delta <= 1e-12:
                continue
            velocity_duration_s = delta / (velocities[joint_index] * scale)
            acceleration_duration_s = 2.0 * math.sqrt(
                delta / (accelerations[joint_index] * scale)
            )
            if velocity_duration_s > segment_minimum_s:
                segment_minimum_s = velocity_duration_s
                segment_joint_index = joint_index
                segment_constraint = "velocity"
            if acceleration_duration_s > segment_minimum_s:
                segment_minimum_s = acceleration_duration_s
                segment_joint_index = joint_index
                segment_constraint = "acceleration"
        minimum_duration_s += segment_minimum_s
        if segment_minimum_s > largest_segment_minimum_s:
            largest_segment_minimum_s = segment_minimum_s
            limiting_segment_index = segment_index
            limiting_joint_index = segment_joint_index
            limiting_constraint = segment_constraint
        count = max(1, math.ceil(segment_minimum_s / dt_s - 1e-12))
        commands.extend(
            tuple(
                begin + sample / count * (end - begin)
                for begin, end in zip(start, goal, strict=True)
            )
            for sample in range(1, count)
        )
        commands.append(goal)
    stream = ServoJStream(commands=tuple(commands), dt_s=dt_s)
    stream.validate()
    return _TimeParameterizedServoJ(
        stream=stream,
        knot_count=len(knots),
        minimum_duration_s=minimum_duration_s,
        limiting_segment_index=limiting_segment_index,
        limiting_joint_index=limiting_joint_index,
        limiting_constraint=limiting_constraint,
    )


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

    planner = preflight.diagnostics.get("planner")
    if planner not in {
        "holorobot_conservative_linear_joint",
        "holorobot_composite_ompl_rrtconnect",
    }:
        raise ValueError("Unsupported motion preflight planner contract")
    if (
        preflight.diagnostics.get("trajectory_generator")
        != "holorobot_velocity_acceleration_limited_servoj_v2"
    ):
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
    acceleration = np.asarray(
        preflight.diagnostics.get("maximum_joint_acceleration_rad_s2"),
        dtype=np.float64,
    )
    if (
        acceleration.shape != (6,)
        or not np.isfinite(acceleration).all()
        or np.any(acceleration <= 0.0)
    ):
        raise ValueError("Motion preflight acceleration limits are invalid")
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
    if planner == "holorobot_conservative_linear_joint":
        expected_waypoints = _linear_waypoints(
            start,
            goal,
            maximum_joint_step_rad=maximum_step,
        )
    else:
        expected_waypoints = tuple(
            _joint_vector(item, label="motion preflight waypoint")
            for item in preflight.planning_waypoints
        )
        if len(expected_waypoints) < 2:
            raise ValueError("OMPL motion preflight requires at least two waypoints")
        if expected_waypoints[0] != start or expected_waypoints[-1] != goal:
            raise ValueError("OMPL motion preflight endpoints do not reproduce")
        if any(
            max(abs(end - begin) for begin, end in zip(first, second, strict=True))
            > maximum_step + 1e-12
            for first, second in zip(
                expected_waypoints[:-1],
                expected_waypoints[1:],
                strict=True,
            )
        ):
            raise ValueError("OMPL motion preflight exceeds its resampling step")
        if preflight.diagnostics.get("planning_waypoints_sha256") != _waypoints_sha256(
            expected_waypoints
        ):
            raise ValueError("OMPL motion preflight waypoint identity changed")
    if preflight.planning_waypoints != expected_waypoints:
        raise ValueError("Motion preflight waypoints do not reproduce")
    expected_timing = _time_parameterized_servoj_stream(
        expected_waypoints,
        maximum_velocity_rad_s=(collision_checker.kinematic_model.joint_velocity_limits_rad_s()),
        maximum_acceleration_rad_s2=tuple(float(value) for value in acceleration),
        dt_s=dt_s,
        speed_scaling=speed_scaling,
        velocity_margin=velocity_margin,
    )
    expected_stream = expected_timing.stream
    if preflight.diagnostics.get("servoj_path_knot_count") != expected_timing.knot_count:
        raise ValueError("Motion preflight ServoJ knot count does not reproduce")
    if not math.isclose(
        diagnostic_float("minimum_dynamic_duration_s"),
        expected_timing.minimum_duration_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Motion preflight dynamic duration does not reproduce")
    for name, expected in (
        ("limiting_segment_index", expected_timing.limiting_segment_index),
        ("limiting_joint_index", expected_timing.limiting_joint_index),
        ("limiting_constraint", expected_timing.limiting_constraint),
    ):
        if preflight.diagnostics.get(name) != expected:
            raise ValueError(f"Motion preflight diagnostic {name} does not reproduce")
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


def _sample_mesh_path_holorobot_style(
    collision_checker: Cs68PinocchioCollisionChecker,
    waypoints: tuple[tuple[float, ...], ...],
    *,
    maximum_joint_step_rad: float,
) -> JointPathMeshCollisionReport:
    sampled = _holorobot_sampled_configurations(waypoints)
    last_result: CollisionCheckResult | None = None
    for sample_index, (configuration, fraction) in enumerate(sampled):
        require_planning_time(f"before sampled mesh pose {sample_index}")
        result = collision_checker.check(configuration)
        require_planning_time(f"after sampled mesh pose {sample_index}")
        last_result = result
        if result.status is not CollisionCheckStatus.CLEAR:
            enriched = replace(
                result,
                diagnostics={
                    **result.diagnostics,
                    "path_validation_mode": HOLOROBOT_SAMPLED_VALIDATION,
                    "sampled_path_verified": False,
                    "effective_maximum_sample_step_rad": (
                        HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD
                    ),
                },
            )
            return JointPathMeshCollisionReport(
                status=result.status,
                sample_count=sample_index + 1,
                blocked_sample_index=sample_index,
                blocked_path_fraction=fraction,
                result=enriched,
                maximum_joint_step_rad=maximum_joint_step_rad,
            )
    if last_result is None:
        raise ValueError("HoloRobot sampled path contains no configurations")
    clear = replace(
        last_result,
        diagnostics={
            **last_result.diagnostics,
            "path_validation_mode": HOLOROBOT_SAMPLED_VALIDATION,
            "sampled_path_verified": True,
            "effective_maximum_sample_step_rad": HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD,
            "sample_count": len(sampled),
        },
    )
    return JointPathMeshCollisionReport(
        status=CollisionCheckStatus.CLEAR,
        sample_count=len(sampled),
        blocked_sample_index=None,
        blocked_path_fraction=None,
        result=clear,
        maximum_joint_step_rad=maximum_joint_step_rad,
    )


def _sample_occupancy_path_holorobot_style(
    occupancy_checker: OccupancyRobotCollisionChecker,
    waypoints: tuple[tuple[float, ...], ...],
    *,
    maximum_joint_step_rad: float,
    required_freshness_horizon_s: float,
) -> JointPathOccupancyCollisionReport:
    sampled = _holorobot_sampled_configurations(waypoints)
    report = occupancy_checker.check_sampled_configurations(
        tuple(configuration for configuration, _fraction in sampled),
        tuple(fraction for _configuration, fraction in sampled),
        maximum_joint_step_rad=maximum_joint_step_rad,
        required_freshness_horizon_s=required_freshness_horizon_s,
        precheck_last_configuration=True,
    )
    return replace(
        report,
        result=replace(
            report.result,
            diagnostics={
                **report.result.diagnostics,
                "effective_maximum_sample_step_rad": (
                    HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD
                ),
            },
        ),
    )


def _preflight_joint_waypoints(
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
    maximum_joint_acceleration_rad_s2: tuple[
        float, float, float, float, float, float
    ] = (4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
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
    path_validation_mode: str = CONTINUOUS_INTERVAL_VALIDATION,
    planning_waypoints: tuple[tuple[float, ...], ...],
    planner: str = "holorobot_conservative_linear_joint",
    planner_diagnostics: dict[str, object] | None = None,
) -> JointMotionPreflight:
    """Collision-check and time-parameterize one already generated joint path."""

    require_planning_time("before joint-path preflight")
    start = _joint_vector(start_joint_positions_rad, label="motion start")
    goal = _joint_vector(goal_joint_positions_rad, label="motion goal")
    waypoints = tuple(
        _joint_vector(item, label="motion planning waypoint")
        for item in planning_waypoints
    )
    if len(waypoints) < 2 or waypoints[0] != start or waypoints[-1] != goal:
        raise ValueError("motion planning waypoints must preserve exact start and goal")
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
    validation_mode = str(path_validation_mode).strip()
    if validation_mode not in {
        CONTINUOUS_INTERVAL_VALIDATION,
        HOLOROBOT_SAMPLED_VALIDATION,
    }:
        raise ValueError(f"Unsupported path_validation_mode: {validation_mode!r}")
    if validation_mode == HOLOROBOT_SAMPLED_VALIDATION:
        require_swept_mesh = False
        require_continuous_occupancy_sweep = False
    diagnostics = {
        "planner": planner,
        "trajectory_generator": "holorobot_velocity_acceleration_limited_servoj_v2",
        "maximum_joint_step_rad": maximum_joint_step_rad,
        "servoj_dt_s": servoj_dt_s,
        "speed_scaling": speed_scaling,
        "velocity_margin": velocity_margin,
        "maximum_joint_acceleration_rad_s2": list(
            maximum_joint_acceleration_rad_s2
        ),
        "execution_freshness_margin_s": freshness_margin_s,
        "require_occupancy": bool(require_occupancy),
        "require_swept_mesh": bool(require_swept_mesh),
        "require_continuous_occupancy_sweep": bool(require_continuous_occupancy_sweep),
        "motion_envelope_acceptance_id": acceptance_id,
        "motion_envelope_metadata_sha256": acceptance_metadata_sha256,
        "accepted_joint_uncertainty_rad": list(uncertainty_tuple),
        "path_validation_mode": validation_mode,
        "provenance": robot_stack_provenance(),
        "motion_authorized": False,
        "planning_waypoints_sha256": _waypoints_sha256(waypoints),
        **(planner_diagnostics or {}),
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
            path_validation_mode=validation_mode,
        )
    with performance_span("planning.mesh_collision_preflight"):
        require_planning_time("before mesh collision preflight")
        if validation_mode == HOLOROBOT_SAMPLED_VALIDATION:
            collision = _sample_mesh_path_holorobot_style(
                collision_checker,
                waypoints,
                maximum_joint_step_rad=maximum_joint_step_rad,
            )
        else:
            collision = collision_checker.check_path(
                start,
                goal,
                maximum_joint_step_rad=maximum_joint_step_rad,
                maximum_joint_path_deviation_rad=uncertainty_tuple,
                motion_envelope_acceptance_id=acceptance_id,
                motion_envelope_metadata_sha256=acceptance_metadata_sha256,
            )
        require_planning_time("after mesh collision preflight")
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
            path_validation_mode=validation_mode,
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
            path_validation_mode=validation_mode,
        )
    with performance_span("planning.servoj_stream_generation"):
        require_planning_time("before ServoJ time parameterization")
        velocities = collision_checker.kinematic_model.joint_velocity_limits_rad_s()
        timing = _time_parameterized_servoj_stream(
            waypoints,
            maximum_velocity_rad_s=velocities,
            maximum_acceleration_rad_s2=maximum_joint_acceleration_rad_s2,
            dt_s=servoj_dt_s,
            speed_scaling=speed_scaling,
            velocity_margin=velocity_margin,
        )
        stream = timing.stream
        require_planning_time("after ServoJ time parameterization")
    stream_duration_s = max(0, len(stream.commands) - 1) * stream.dt_s
    required_freshness_horizon_s = stream_duration_s + freshness_margin_s
    diagnostics["servoj_path_knot_count"] = timing.knot_count
    diagnostics["minimum_dynamic_duration_s"] = timing.minimum_duration_s
    diagnostics["limiting_segment_index"] = timing.limiting_segment_index
    diagnostics["limiting_joint_index"] = timing.limiting_joint_index
    diagnostics["limiting_constraint"] = timing.limiting_constraint
    diagnostics["planned_servoj_duration_s"] = stream_duration_s
    diagnostics["required_freshness_horizon_s"] = required_freshness_horizon_s
    occupancy: JointPathOccupancyCollisionReport | None = None
    if occupancy_checker is not None:
        if (
            occupancy_checker.accepted_joint_uncertainty_rad != uncertainty_tuple
            or occupancy_checker.motion_envelope_acceptance_id != acceptance_id
        ):
            raise ValueError("Mesh and occupancy preflight motion envelopes differ")
        with performance_span("planning.occupancy_collision_preflight"):
            require_planning_time("before occupancy collision preflight")
            if validation_mode == HOLOROBOT_SAMPLED_VALIDATION:
                occupancy = _sample_occupancy_path_holorobot_style(
                    occupancy_checker,
                    waypoints,
                    maximum_joint_step_rad=maximum_joint_step_rad,
                    required_freshness_horizon_s=required_freshness_horizon_s,
                )
            else:
                occupancy = occupancy_checker.check_path(
                    start,
                    goal,
                    maximum_joint_step_rad=maximum_joint_step_rad,
                    required_freshness_horizon_s=required_freshness_horizon_s,
                )
            require_planning_time("after occupancy collision preflight")
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
                path_validation_mode=validation_mode,
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
                path_validation_mode=validation_mode,
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
            path_validation_mode=validation_mode,
        )
    preflight = JointMotionPreflight(
        status=MotionPreflightStatus.CLEAR,
        start_joint_positions_rad=start,
        goal_joint_positions_rad=goal,
        planning_waypoints=waypoints,
        servoj_stream=stream,
        collision=collision,
        occupancy=occupancy,
        blocking_reasons=(),
        warnings=(
            (
                "online_path_uses_holorobot_fixed_step_segment_sampling",
            )
            if validation_mode == HOLOROBOT_SAMPLED_VALIDATION
            else (
                *(
                    ()
                    if require_occupancy
                    else ("occupancy_disabled_offline_diagnostic_only",)
                ),
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
            )
        ),
        diagnostics=diagnostics,
        servoj_runtime_config=runtime_config,
        occupancy_required=bool(require_occupancy),
        swept_mesh_required=bool(require_swept_mesh),
        continuous_occupancy_sweep_required=bool(require_continuous_occupancy_sweep),
        path_validation_mode=validation_mode,
    )
    if validation_mode == HOLOROBOT_SAMPLED_VALIDATION:
        payload = _holorobot_sampled_evidence_payload(preflight)
        if payload is None:
            raise ValueError("HoloRobot sampled preflight lacks bound collision evidence")
        preflight = replace(
            preflight,
            path_validation_evidence_sha256=_canonical_sha256(payload),
        )
    return preflight


def _endpoint_blocked_preflight(
    *,
    start: tuple[float, ...],
    goal: tuple[float, ...],
    waypoints: tuple[tuple[float, ...], ...],
    collision: JointPathMeshCollisionReport,
    occupancy: JointPathOccupancyCollisionReport | None,
    blocking_reasons: tuple[str, ...],
    maximum_joint_step_rad: float,
    path_validation_mode: str,
    require_occupancy: bool,
    accepted_joint_uncertainty_rad: tuple[float, ...],
    motion_envelope_acceptance_id: str | None,
    motion_envelope_metadata_sha256: str | None,
) -> JointMotionPreflight:
    return JointMotionPreflight(
        status=MotionPreflightStatus.BLOCKED,
        start_joint_positions_rad=start,
        goal_joint_positions_rad=goal,
        planning_waypoints=waypoints,
        servoj_stream=None,
        collision=collision,
        occupancy=occupancy,
        blocking_reasons=blocking_reasons,
        warnings=(),
        diagnostics={
            "planner": "holorobot_composite_joint",
            "primary_planner": "holorobot_conservative_linear_joint",
            "fallback_planner": "holorobot_ompl_rrtconnect",
            "fallback_used": False,
            "failure_stage": "goal_endpoint",
            "maximum_joint_step_rad": maximum_joint_step_rad,
            "path_validation_mode": path_validation_mode,
            "require_occupancy": require_occupancy,
            "accepted_joint_uncertainty_rad": list(
                accepted_joint_uncertainty_rad
            ),
            "motion_envelope_acceptance_id": motion_envelope_acceptance_id,
            "motion_envelope_metadata_sha256": motion_envelope_metadata_sha256,
            "planning_waypoints_sha256": _waypoints_sha256(waypoints),
            "motion_authorized": False,
        },
        occupancy_required=require_occupancy,
        swept_mesh_required=False,
        continuous_occupancy_sweep_required=False,
        path_validation_mode=path_validation_mode,
    )


def _with_failed_fallback(
    primary: JointMotionPreflight,
    *,
    status: HoloRobotJointPlanStatus,
    reasons: tuple[str, ...],
    diagnostics: dict[str, object] | None,
) -> JointMotionPreflight:
    return replace(
        primary,
        blocking_reasons=tuple(
            dict.fromkeys(
                (
                    *primary.blocking_reasons,
                    *(f"ompl_fallback:{reason}" for reason in reasons),
                )
            )
        ),
        diagnostics={
            **primary.diagnostics,
            "planner": "holorobot_composite_joint",
            "primary_planner": "holorobot_conservative_linear_joint",
            "primary_status": primary.status.value,
            "fallback_planner": "holorobot_ompl_rrtconnect",
            "fallback_status": status.value,
            "fallback_used": False,
            "fallback_diagnostics": diagnostics or {},
        },
    )


def _primary_failed_on_interior_path(
    primary: JointMotionPreflight,
) -> bool:
    """Match HoloRobot's PATH_BLOCKED-only fallback transition.

    UNKNOWN checker/evidence states and collisions at either endpoint are not
    search problems.  Sending those states to OMPL used to hide the real cause
    behind a planning timeout and added avoidable latency.
    """

    if primary.status is not MotionPreflightStatus.BLOCKED:
        return False
    report: JointPathMeshCollisionReport | JointPathOccupancyCollisionReport | None
    if primary.collision is not None and (
        primary.collision.status is not CollisionCheckStatus.CLEAR
    ):
        report = primary.collision
    elif primary.occupancy is not None and (
        primary.occupancy.status is not CollisionCheckStatus.CLEAR
    ):
        report = primary.occupancy
    else:
        return False
    fraction = report.blocked_path_fraction
    return bool(
        report.status is CollisionCheckStatus.BLOCKED
        and fraction is not None
        and 0.0 < float(fraction) < 1.0
    )


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
    maximum_joint_acceleration_rad_s2: tuple[
        float, float, float, float, float, float
    ] = (4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
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
    path_validation_mode: str = CONTINUOUS_INTERVAL_VALIDATION,
    enable_ompl_fallback: bool = False,
    ompl_plan_timeout_s: float = 1.0,
    ompl_rrt_range_rad: float = 0.25,
    ompl_simplify_path: bool = True,
) -> JointMotionPreflight:
    """Use HoloRobot's straight-primary, RRTConnect-fallback planning order."""

    start = _joint_vector(start_joint_positions_rad, label="motion start")
    goal = _joint_vector(goal_joint_positions_rad, label="motion goal")
    waypoints = _linear_waypoints(
        start,
        goal,
        maximum_joint_step_rad=maximum_joint_step_rad,
    )
    common: dict[str, Any] = {
        "collision_checker": collision_checker,
        "checker_unavailable_reason": checker_unavailable_reason,
        "occupancy_checker": occupancy_checker,
        "require_occupancy": require_occupancy,
        "require_swept_mesh": require_swept_mesh,
        "require_continuous_occupancy_sweep": require_continuous_occupancy_sweep,
        "maximum_joint_step_rad": maximum_joint_step_rad,
        "servoj_dt_s": servoj_dt_s,
        "speed_scaling": speed_scaling,
        "velocity_margin": velocity_margin,
        "maximum_joint_acceleration_rad_s2": maximum_joint_acceleration_rad_s2,
        "execution_freshness_margin_s": execution_freshness_margin_s,
        "servoj_runtime_config": servoj_runtime_config,
        "accepted_joint_uncertainty_rad": accepted_joint_uncertainty_rad,
        "motion_envelope_acceptance_id": motion_envelope_acceptance_id,
        "motion_envelope_metadata_sha256": motion_envelope_metadata_sha256,
        "path_validation_mode": path_validation_mode,
    }

    # HoloRobot rejects an invalid goal before spending time on either a direct
    # path or RRTConnect.  Preserve that ordering for the online sampled mode.
    if (
        path_validation_mode == HOLOROBOT_SAMPLED_VALIDATION
        and collision_checker is not None
    ):
        require_planning_time("before goal mesh endpoint precheck")
        goal_mesh = collision_checker.check(goal)
        require_planning_time("after goal mesh endpoint precheck")
        goal_mesh_report = JointPathMeshCollisionReport(
            status=goal_mesh.status,
            sample_count=1,
            blocked_sample_index=(
                None if goal_mesh.status is CollisionCheckStatus.CLEAR else 0
            ),
            blocked_path_fraction=(
                None if goal_mesh.status is CollisionCheckStatus.CLEAR else 1.0
            ),
            result=replace(
                goal_mesh,
                diagnostics={
                    **goal_mesh.diagnostics,
                    "path_validation_mode": HOLOROBOT_SAMPLED_VALIDATION,
                    "sampled_path_verified": (
                        goal_mesh.status is CollisionCheckStatus.CLEAR
                    ),
                    "effective_maximum_sample_step_rad": (
                        HOLOROBOT_EFFECTIVE_SAMPLE_STEP_RAD
                    ),
                },
            ),
            maximum_joint_step_rad=maximum_joint_step_rad,
        )
        if goal_mesh.status is not CollisionCheckStatus.CLEAR:
            return _endpoint_blocked_preflight(
                start=start,
                goal=goal,
                waypoints=waypoints,
                collision=goal_mesh_report,
                occupancy=None,
                blocking_reasons=goal_mesh.blocking_reasons
                or (f"goal_collision_status:{goal_mesh.status.value}",),
                maximum_joint_step_rad=maximum_joint_step_rad,
                path_validation_mode=path_validation_mode,
                require_occupancy=require_occupancy,
                accepted_joint_uncertainty_rad=accepted_joint_uncertainty_rad,
                motion_envelope_acceptance_id=motion_envelope_acceptance_id,
                motion_envelope_metadata_sha256=motion_envelope_metadata_sha256,
            )
    primary = _preflight_joint_waypoints(
        start,
        goal,
        planning_waypoints=waypoints,
        **common,
    )
    if (
        primary.status is MotionPreflightStatus.CLEAR
        or not enable_ompl_fallback
        or path_validation_mode != HOLOROBOT_SAMPLED_VALIDATION
        or collision_checker is None
        or occupancy_checker is None
    ):
        return primary
    if not _primary_failed_on_interior_path(primary):
        return replace(
            primary,
            diagnostics={
                **primary.diagnostics,
                "planner": "holorobot_composite_joint",
                "primary_planner": "holorobot_conservative_linear_joint",
                "fallback_planner": "holorobot_ompl_rrtconnect",
                "fallback_used": False,
                "fallback_reason": "not_an_interior_path_block",
                "failure_stage": (
                    "goal_endpoint"
                    if any(
                        report is not None
                        and report.blocked_path_fraction == 1.0
                        for report in (primary.collision, primary.occupancy)
                    )
                    else "start_endpoint_or_checker_evidence"
                ),
            },
        )
    if not ompl_available():
        return _with_failed_fallback(
            primary,
            status=HoloRobotJointPlanStatus.PLANNER_UNAVAILABLE,
            reasons=("ompl_python_bindings_unavailable",),
            diagnostics=None,
        )

    try:
        with (
            performance_span("planning.holorobot_ompl_fallback"),
            occupancy_checker.bind_configuration_queries() as bound_occupancy,
        ):

            def state_validity(
                configuration: tuple[float, ...],
            ) -> tuple[bool, tuple[str, ...]]:
                require_planning_time("before RRT state mesh check")
                mesh = collision_checker.check(configuration)
                require_planning_time("after RRT state mesh check")
                if mesh.status is not CollisionCheckStatus.CLEAR:
                    return False, mesh.blocking_reasons or (
                        f"mesh_collision_status:{mesh.status.value}",
                    )
                require_planning_time("before RRT state occupancy check")
                occupancy = bound_occupancy.check(configuration)
                require_planning_time("after RRT state occupancy check")
                if occupancy.status is not CollisionCheckStatus.CLEAR:
                    return False, occupancy.blocking_reasons or (
                        f"occupancy_status:{occupancy.status.value}",
                    )
                return True, ()

            fallback = plan_holorobot_rrtconnect(
                start,
                goal,
                joint_limits_rad=collision_checker.kinematic_model.joint_limit_pairs(),
                state_validity=state_validity,
                config=HoloRobotOmplConfig(
                    maximum_joint_step_rad=maximum_joint_step_rad,
                    plan_timeout_s=ompl_plan_timeout_s,
                    rrt_range_rad=ompl_rrt_range_rad,
                    simplify_path=ompl_simplify_path,
                ),
            )
    except PlanningDeadlineExceeded:
        raise
    except Exception as exc:
        return _with_failed_fallback(
            primary,
            status=HoloRobotJointPlanStatus.PATH_BLOCKED,
            reasons=(f"planner_error:{type(exc).__name__}:{exc}",),
            diagnostics=None,
        )
    if not fallback.clear:
        return _with_failed_fallback(
            primary,
            status=fallback.status,
            reasons=fallback.blocking_reasons,
            diagnostics=fallback.diagnostics,
        )

    # The concrete OMPL adapter anchors these values exactly.  Keep this
    # boundary defensive so a malformed or third-party planner result becomes a
    # typed rejected candidate instead of aborting the complete scan runtime.
    if (
        len(fallback.waypoints) < 2
        or fallback.waypoints[0] != start
        or fallback.waypoints[-1] != goal
    ):
        return _with_failed_fallback(
            primary,
            status=HoloRobotJointPlanStatus.PATH_BLOCKED,
            reasons=("ompl_path_endpoint_contract_mismatch",),
            diagnostics=fallback.diagnostics,
        )

    verified = _preflight_joint_waypoints(
        start,
        goal,
        planning_waypoints=fallback.waypoints,
        planner="holorobot_composite_ompl_rrtconnect",
        planner_diagnostics={
            "primary_planner": "holorobot_conservative_linear_joint",
            "primary_status": primary.status.value,
            "primary_blocking_reasons": list(primary.blocking_reasons),
            "fallback_planner": "holorobot_ompl_rrtconnect",
            "fallback_status": fallback.status.value,
            "fallback_used": True,
            "fallback_diagnostics": fallback.diagnostics or {},
        },
        **common,
    )
    return verified
