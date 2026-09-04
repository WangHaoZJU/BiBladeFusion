"""HoloRobot-backed preflight for an explicit ordered blade-view sequence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import MotionPreflightConfig
from biblade_fusion.diagnostics.performance_timing import performance_timed
from biblade_fusion.planning import CandidateStatus, FilteredViewPlan
from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    JointMotionPreflight,
    MotionPreflightStatus,
    OccupancyRobotCollisionChecker,
    load_es68_flange_t_tcp,
    preflight_linear_joint_motion,
)
from biblade_fusion.robotics.motion_preflight import CONTINUOUS_INTERVAL_VALIDATION
from biblade_fusion.workflows.path_validation import PathSequenceError


@dataclass(frozen=True, slots=True)
class EndpointPoseConsistency:
    """ES68 FK agreement with the TCP target that produced one IK endpoint."""

    status: CollisionCheckStatus
    translation_error_m: float | None
    rotation_error_deg: float | None
    maximum_translation_error_m: float
    maximum_rotation_error_deg: float
    predicted_base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...] | None
    blocking_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreflightedMotionLeg:
    start_view_id: str
    end_view_id: str
    goal_base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...]
    endpoint_consistency: EndpointPoseConsistency
    preflight: JointMotionPreflight


@dataclass(frozen=True, slots=True)
class LiveJointSegmentPreflight:
    """One live-start, occupancy-bound segment; never an authorization by itself."""

    preflight: JointMotionPreflight
    endpoint_consistency: EndpointPoseConsistency | None
    final_target: bool

    @property
    def ready_for_approval(self) -> bool:
        return self.preflight.ready_for_approval and (
            not self.final_target
            or (
                self.endpoint_consistency is not None
                and self.endpoint_consistency.status is CollisionCheckStatus.CLEAR
            )
        )

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class MotionSequenceCost:
    estimated_servoj_duration_s: float
    total_joint_travel_l1_rad: float
    maximum_leg_joint_delta_rad: float


@dataclass(frozen=True, slots=True)
class ViewSequenceMotionPreflight:
    ordered_view_ids: tuple[str, ...]
    legs: tuple[PreflightedMotionLeg, ...]
    cost: MotionSequenceCost
    evaluated_at_utc: str | None = None

    @property
    def ready_for_approval(self) -> bool:
        return bool(self.legs) and all(
            leg.endpoint_consistency.status is CollisionCheckStatus.CLEAR
            and leg.preflight.ready_for_approval
            for leg in self.legs
        )

    @property
    def motion_authorized(self) -> bool:
        return False


def preflight_view_sequence_motion(
    filtered_plan: FilteredViewPlan,
    ordered_view_ids: tuple[str, ...],
    seed_joint_positions_rad: ArrayLike,
    config: MotionPreflightConfig,
    *,
    hand_eye: HandEyeCalibration,
    collision_checker: Cs68PinocchioCollisionChecker | None,
    collision_checker_unavailable_reason: str = "checker_unavailable",
    occupancy_checker: OccupancyRobotCollisionChecker | None = None,
    execution_freshness_margin_s: float = 1.0,
    evaluated_at_utc: datetime | None = None,
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
) -> ViewSequenceMotionPreflight:
    """Preflight seed-to-view and view-to-view legs without authorizing motion."""

    if not ordered_view_ids:
        raise PathSequenceError("At least one ordered view ID is required")
    if len(set(ordered_view_ids)) != len(ordered_view_ids):
        raise PathSequenceError("Ordered view IDs must be unique")
    evaluation_time = None
    if evaluated_at_utc is not None:
        if evaluated_at_utc.tzinfo is None:
            raise ValueError("Motion-preflight evaluation time must be timezone-aware")
        evaluation_time = evaluated_at_utc.astimezone(UTC).isoformat()
    evaluations = {item.candidate.view_id: item for item in filtered_plan.candidates}
    unknown = tuple(view_id for view_id in ordered_view_ids if view_id not in evaluations)
    if unknown:
        raise PathSequenceError(f"Ordered sequence contains unknown views: {unknown}")

    previous_id = "initialization_seed"
    previous_joints = seed_joint_positions_rad
    legs: list[PreflightedMotionLeg] = []
    duration_s = 0.0
    joint_travel_l1_rad = 0.0
    maximum_leg_delta_rad = 0.0
    try:
        flange_t_left_ir = hand_eye.require_flange_primary()
        flange_t_tcp = load_es68_flange_t_tcp()
    except (OSError, ValueError) as exc:
        raise PathSequenceError(
            f"Motion preflight requires authoritative flange-primary hand-eye: {exc}"
        ) from exc
    for view_id in ordered_view_ids:
        evaluation = evaluations[view_id]
        if (
            evaluation.status is not CandidateStatus.ENDPOINT_FEASIBLE
            or evaluation.joint_positions_rad is None
        ):
            raise PathSequenceError(f"View {view_id!r} has no endpoint-feasible joint solution")
        camera_pose = evaluation.candidate.base_t_left_ir
        canonical_camera_pose = type(camera_pose)(
            "base",
            "left_ir",
            camera_pose.matrix,
        )
        base_t_flange = canonical_camera_pose.compose(flange_t_left_ir.inverse())
        base_t_tcp = base_t_flange.compose(flange_t_tcp)
        goal_matrix = tuple(tuple(float(value) for value in row) for row in base_t_tcp.matrix)
        endpoint_consistency = evaluate_endpoint_pose_consistency(
            evaluation.joint_positions_rad,
            base_t_tcp.matrix,
            collision_checker,
            maximum_translation_error_m=(config.maximum_endpoint_translation_error_m),
            maximum_rotation_error_deg=(config.maximum_endpoint_rotation_error_deg),
        )
        preflight = preflight_linear_joint_motion(
            previous_joints,
            evaluation.joint_positions_rad,
            collision_checker=collision_checker,
            checker_unavailable_reason=collision_checker_unavailable_reason,
            occupancy_checker=occupancy_checker,
            maximum_joint_step_rad=config.maximum_joint_step_rad,
            servoj_dt_s=config.servoj_dt_s,
            speed_scaling=config.speed_scaling,
            velocity_margin=config.velocity_margin,
            maximum_joint_acceleration_rad_s2=(
                config.maximum_joint_acceleration_rad_s2
            ),
            execution_freshness_margin_s=execution_freshness_margin_s,
            accepted_joint_uncertainty_rad=accepted_joint_uncertainty_rad,
            motion_envelope_acceptance_id=motion_envelope_acceptance_id,
            motion_envelope_metadata_sha256=motion_envelope_metadata_sha256,
            enable_ompl_fallback=config.enable_ompl_fallback,
            ompl_plan_timeout_s=config.ompl_plan_timeout_s,
            ompl_rrt_range_rad=config.ompl_rrt_range_rad,
            ompl_simplify_path=config.ompl_simplify_path,
        )
        preflight = _apply_endpoint_gate(preflight, endpoint_consistency)
        if evaluation_time is not None:
            preflight = replace(
                preflight,
                diagnostics={
                    **preflight.diagnostics,
                    "evaluated_at_utc": evaluation_time,
                },
            )
        legs.append(
            PreflightedMotionLeg(
                previous_id,
                view_id,
                goal_matrix,
                endpoint_consistency,
                preflight,
            )
        )
        if preflight.servoj_stream is not None:
            duration_s += (
                max(0, len(preflight.servoj_stream.commands) - 1) * preflight.servoj_stream.dt_s
            )
        start_array = np.asarray(preflight.start_joint_positions_rad)
        goal_array = np.asarray(preflight.goal_joint_positions_rad)
        delta = np.abs(goal_array - start_array)
        joint_travel_l1_rad += float(np.sum(delta))
        maximum_leg_delta_rad = max(maximum_leg_delta_rad, float(np.max(delta)))
        previous_id = view_id
        previous_joints = evaluation.joint_positions_rad
    return ViewSequenceMotionPreflight(
        ordered_view_ids,
        tuple(legs),
        MotionSequenceCost(
            estimated_servoj_duration_s=duration_s,
            total_joint_travel_l1_rad=joint_travel_l1_rad,
            maximum_leg_joint_delta_rad=maximum_leg_delta_rad,
        ),
        evaluation_time,
    )


@performance_timed("planning.endpoint_fk_consistency")
def evaluate_endpoint_pose_consistency(
    joint_positions_rad: ArrayLike,
    target_base_t_tcp: ArrayLike,
    collision_checker: Cs68PinocchioCollisionChecker | None,
    *,
    maximum_translation_error_m: float,
    maximum_rotation_error_deg: float,
) -> EndpointPoseConsistency:
    translation_limit = float(maximum_translation_error_m)
    rotation_limit = float(maximum_rotation_error_deg)
    if (
        not np.isfinite((translation_limit, rotation_limit)).all()
        or translation_limit <= 0.0
        or not 0.0 < rotation_limit <= 180.0
    ):
        raise ValueError("Endpoint-consistency limits are invalid")
    if (
        collision_checker is None
        or collision_checker.model_name != "es68"
        or not collision_checker.robot_geometry_hash
        or not collision_checker.motion_model_contract_hash
    ):
        return EndpointPoseConsistency(
            CollisionCheckStatus.UNKNOWN,
            None,
            None,
            translation_limit,
            rotation_limit,
            None,
            ("endpoint_pose_consistency_es68_model_unavailable",),
        )
    try:
        base_t_flange = np.asarray(
            collision_checker.kinematic_model.forward_kinematics(joint_positions_rad),
            dtype=np.float64,
        )
        target_pose = PoseSE3("base", "tcp", target_base_t_tcp)
        predicted_pose = PoseSE3(
            "base",
            "tcp",
            base_t_flange @ load_es68_flange_t_tcp().matrix,
        )
        target = target_pose.matrix
        predicted = predicted_pose.matrix
        if (
            base_t_flange.shape != (4, 4)
            or target.shape != (4, 4)
            or predicted.shape != (4, 4)
            or not np.isfinite((base_t_flange, target, predicted)).all()
        ):
            raise ValueError("endpoint transforms must be finite 4x4 matrices")
        translation_error = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]))
        relative_rotation = predicted[:3, :3].T @ target[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error = float(np.degrees(np.arccos(cosine)))
        reasons = tuple(
            reason
            for exceeded, reason in (
                (
                    translation_error > translation_limit,
                    "endpoint_fk_tcp_translation_error_exceeded",
                ),
                (
                    rotation_error > rotation_limit,
                    "endpoint_fk_tcp_rotation_error_exceeded",
                ),
            )
            if exceeded
        )
        return EndpointPoseConsistency(
            (CollisionCheckStatus.BLOCKED if reasons else CollisionCheckStatus.CLEAR),
            translation_error,
            rotation_error,
            translation_limit,
            rotation_limit,
            tuple(tuple(float(value) for value in row) for row in predicted),
            reasons,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return EndpointPoseConsistency(
            CollisionCheckStatus.UNKNOWN,
            None,
            None,
            translation_limit,
            rotation_limit,
            None,
            (f"endpoint_pose_consistency_error:{type(exc).__name__}",),
        )


@performance_timed("planning.preflight_live_joint_segment")
def preflight_live_joint_segment(
    start_joint_positions_rad: ArrayLike,
    goal_joint_positions_rad: ArrayLike,
    config: MotionPreflightConfig,
    *,
    collision_checker: Cs68PinocchioCollisionChecker | None,
    occupancy_checker: OccupancyRobotCollisionChecker | None,
    final_target: bool,
    target_base_t_tcp_matrix: ArrayLike | None = None,
    collision_checker_unavailable_reason: str = "checker_unavailable",
    execution_freshness_margin_s: float = 1.0,
    evaluated_at_utc: datetime | None = None,
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
) -> LiveJointSegmentPreflight:
    """Preflight exactly one receding-horizon segment from measured live joints.

    Intermediate short segments have no geometric view endpoint.  The final segment
    additionally proves that ES68 FK reproduces the target ``base_T_tcp`` that produced
    the stored IK solution.  Every call creates a fresh preflight bound to the supplied
    occupancy checker; callers must never cache it across a map publication.
    """

    if final_target and target_base_t_tcp_matrix is None:
        raise ValueError("Final live segment requires target_base_t_tcp_matrix")
    if not final_target and target_base_t_tcp_matrix is not None:
        raise ValueError("Intermediate live segment must not claim a view endpoint")
    evaluation_time = None
    if evaluated_at_utc is not None:
        if evaluated_at_utc.tzinfo is None:
            raise ValueError("Motion-preflight evaluation time must be timezone-aware")
        evaluation_time = evaluated_at_utc.astimezone(UTC).isoformat()
    preflight = preflight_linear_joint_motion(
        start_joint_positions_rad,
        goal_joint_positions_rad,
        collision_checker=collision_checker,
        checker_unavailable_reason=collision_checker_unavailable_reason,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=config.maximum_joint_step_rad,
        servoj_dt_s=config.servoj_dt_s,
        speed_scaling=config.speed_scaling,
        velocity_margin=config.velocity_margin,
        maximum_joint_acceleration_rad_s2=(
            config.maximum_joint_acceleration_rad_s2
        ),
        execution_freshness_margin_s=execution_freshness_margin_s,
        accepted_joint_uncertainty_rad=accepted_joint_uncertainty_rad,
        motion_envelope_acceptance_id=motion_envelope_acceptance_id,
        motion_envelope_metadata_sha256=motion_envelope_metadata_sha256,
        path_validation_mode=path_validation_mode,
        enable_ompl_fallback=config.enable_ompl_fallback,
        ompl_plan_timeout_s=config.ompl_plan_timeout_s,
        ompl_rrt_range_rad=config.ompl_rrt_range_rad,
        ompl_simplify_path=config.ompl_simplify_path,
    )
    endpoint = None
    if final_target:
        endpoint = evaluate_endpoint_pose_consistency(
            goal_joint_positions_rad,
            target_base_t_tcp_matrix,
            collision_checker,
            maximum_translation_error_m=config.maximum_endpoint_translation_error_m,
            maximum_rotation_error_deg=config.maximum_endpoint_rotation_error_deg,
        )
        preflight = _apply_endpoint_gate(preflight, endpoint)
    if evaluation_time is not None:
        preflight = replace(
            preflight,
            diagnostics={**preflight.diagnostics, "evaluated_at_utc": evaluation_time},
        )
    return LiveJointSegmentPreflight(preflight, endpoint, bool(final_target))


def _apply_endpoint_gate(
    preflight: JointMotionPreflight,
    endpoint: EndpointPoseConsistency,
) -> JointMotionPreflight:
    diagnostics = {
        **preflight.diagnostics,
        "endpoint_pose_consistency": {
            "status": endpoint.status.value,
            "translation_error_m": endpoint.translation_error_m,
            "rotation_error_deg": endpoint.rotation_error_deg,
            "maximum_translation_error_m": endpoint.maximum_translation_error_m,
            "maximum_rotation_error_deg": endpoint.maximum_rotation_error_deg,
            "predicted_base_t_tcp_matrix": endpoint.predicted_base_t_tcp_matrix,
            "blocking_reasons": endpoint.blocking_reasons,
        },
    }
    if endpoint.status is CollisionCheckStatus.CLEAR:
        return replace(preflight, diagnostics=diagnostics)
    reasons = tuple(dict.fromkeys((*preflight.blocking_reasons, *endpoint.blocking_reasons)))
    return replace(
        preflight,
        status=(
            preflight.status
            if preflight.status is not MotionPreflightStatus.CLEAR
            else MotionPreflightStatus.BLOCKED
        ),
        servoj_stream=None,
        blocking_reasons=reasons,
        diagnostics=diagnostics,
        servoj_runtime_config=None,
    )
