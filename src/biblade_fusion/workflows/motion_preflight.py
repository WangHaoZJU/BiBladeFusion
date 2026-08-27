"""HoloRobot-backed preflight for an explicit ordered blade-view sequence."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from biblade_fusion.core.settings import MotionPreflightConfig
from biblade_fusion.planning import CandidateStatus, FilteredViewPlan
from biblade_fusion.robotics import (
    Cs68PinocchioCollisionChecker,
    JointMotionPreflight,
    preflight_linear_joint_motion,
)
from biblade_fusion.workflows.path_validation import PathSequenceError


@dataclass(frozen=True, slots=True)
class PreflightedMotionLeg:
    start_view_id: str
    end_view_id: str
    preflight: JointMotionPreflight


@dataclass(frozen=True, slots=True)
class ViewSequenceMotionPreflight:
    ordered_view_ids: tuple[str, ...]
    legs: tuple[PreflightedMotionLeg, ...]

    @property
    def ready_for_approval(self) -> bool:
        return bool(self.legs) and all(
            leg.preflight.ready_for_approval for leg in self.legs
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
    collision_checker: Cs68PinocchioCollisionChecker | None,
) -> ViewSequenceMotionPreflight:
    """Preflight seed-to-view and view-to-view legs without authorizing motion."""

    if not ordered_view_ids:
        raise PathSequenceError("At least one ordered view ID is required")
    if len(set(ordered_view_ids)) != len(ordered_view_ids):
        raise PathSequenceError("Ordered view IDs must be unique")
    evaluations = {item.candidate.view_id: item for item in filtered_plan.candidates}
    unknown = tuple(view_id for view_id in ordered_view_ids if view_id not in evaluations)
    if unknown:
        raise PathSequenceError(f"Ordered sequence contains unknown views: {unknown}")

    previous_id = "initialization_seed"
    previous_joints = seed_joint_positions_rad
    legs: list[PreflightedMotionLeg] = []
    for view_id in ordered_view_ids:
        evaluation = evaluations[view_id]
        if (
            evaluation.status is not CandidateStatus.ENDPOINT_FEASIBLE
            or evaluation.joint_positions_rad is None
        ):
            raise PathSequenceError(
                f"View {view_id!r} has no endpoint-feasible joint solution"
            )
        preflight = preflight_linear_joint_motion(
            previous_joints,
            evaluation.joint_positions_rad,
            collision_checker=collision_checker,
            maximum_joint_step_rad=config.maximum_joint_step_rad,
            servoj_dt_s=config.servoj_dt_s,
            speed_scaling=config.speed_scaling,
            velocity_margin=config.velocity_margin,
        )
        legs.append(PreflightedMotionLeg(previous_id, view_id, preflight))
        previous_id = view_id
        previous_joints = evaluation.joint_positions_rad
    return ViewSequenceMotionPreflight(ordered_view_ids, tuple(legs))
