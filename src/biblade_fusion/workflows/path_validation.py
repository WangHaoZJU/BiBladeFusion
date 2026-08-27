"""Offline collision validation for an explicit ordered view sequence."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.settings import CollisionConfig
from biblade_fusion.planning import (
    CandidateStatus,
    FilteredViewPlan,
    JointPathCollisionReport,
    validate_joint_path_collision,
)


class PathSequenceError(ValueError):
    """An ordered view sequence cannot be validated from the offline plan."""


@dataclass(frozen=True, slots=True)
class ValidatedPathLeg:
    start_view_id: str
    end_view_id: str
    collision: JointPathCollisionReport


@dataclass(frozen=True, slots=True)
class ViewSequenceCollisionReport:
    ordered_view_ids: tuple[str, ...]
    legs: tuple[ValidatedPathLeg, ...]

    @property
    def collision_free(self) -> bool:
        return all(leg.collision.collision_free for leg in self.legs)

    @property
    def motion_authorized(self) -> bool:
        return False


def validate_view_sequence_collision(
    filtered_plan: FilteredViewPlan,
    ordered_view_ids: tuple[str, ...],
    seed_joint_positions_rad,
    model: Cs68KinematicsModel,
    hand_eye: HandEyeCalibration,
    config: CollisionConfig,
) -> ViewSequenceCollisionReport:
    """Validate seed-to-view and view-to-view legs in the explicitly supplied order."""

    if not ordered_view_ids:
        raise PathSequenceError("At least one ordered view ID is required")
    if len(set(ordered_view_ids)) != len(ordered_view_ids):
        raise PathSequenceError("Ordered view IDs must be unique")
    evaluations = {
        item.candidate.view_id: item for item in filtered_plan.candidates
    }
    unknown = tuple(view_id for view_id in ordered_view_ids if view_id not in evaluations)
    if unknown:
        raise PathSequenceError(f"Ordered sequence contains unknown views: {unknown}")

    previous_id = "initialization_seed"
    previous_joints = seed_joint_positions_rad
    legs = []
    for view_id in ordered_view_ids:
        evaluation = evaluations[view_id]
        if (
            evaluation.status is not CandidateStatus.ENDPOINT_FEASIBLE
            or evaluation.joint_positions_rad is None
        ):
            raise PathSequenceError(
                f"View {view_id!r} has no endpoint-feasible joint solution"
            )
        collision = validate_joint_path_collision(
            previous_joints,
            evaluation.joint_positions_rad,
            model,
            hand_eye,
            config,
        )
        legs.append(ValidatedPathLeg(previous_id, view_id, collision))
        previous_id = view_id
        previous_joints = evaluation.joint_positions_rad
    return ViewSequenceCollisionReport(ordered_view_ids, tuple(legs))
