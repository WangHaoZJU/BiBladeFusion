"""Fail-closed capsule collision and continuous joint-path validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, sin

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CollisionConfig, CollisionObstacleConfig


class CollisionValidationError(ValueError):
    """Collision validation lacks trustworthy inputs or geometry."""


@dataclass(frozen=True, slots=True)
class Capsule:
    capsule_id: str
    start_m: NDArray[np.float64]
    end_m: NDArray[np.float64]
    radius_m: float

    def __post_init__(self) -> None:
        start = np.array(self.start_m, dtype=np.float64, copy=True)
        end = np.array(self.end_m, dtype=np.float64, copy=True)
        if start.shape != (3,) or end.shape != (3,) or not np.isfinite((start, end)).all():
            raise ValueError("Capsule endpoints must be finite three-vectors")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("Capsule radius must be finite and positive")
        start.setflags(write=False)
        end.setflags(write=False)
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "end_m", end)


@dataclass(frozen=True, slots=True)
class CollisionFinding:
    sample_index: int
    path_fraction: float
    kind: str
    first_body: str
    second_body: str
    message: str


@dataclass(frozen=True, slots=True)
class JointPathCollisionReport:
    sample_count: int
    findings: tuple[CollisionFinding, ...]
    maximum_joint_step_rad: float

    @property
    def collision_free(self) -> bool:
        return not self.findings

    @property
    def motion_authorized(self) -> bool:
        return False


def _rot_x(angle: float) -> NDArray[np.float64]:
    result = np.eye(4)
    result[1:3, 1:3] = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
    return result


def _rot_z(angle: float) -> NDArray[np.float64]:
    result = np.eye(4)
    result[:2, :2] = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
    return result


def cs68_mdh_joint_origins(
    model: Cs68KinematicsModel,
    joint_positions_rad: ArrayLike,
) -> tuple[NDArray[np.float64], PoseSE3]:
    """Reproduce the exact fixed-MDH then RotZ chain used by Elite's KDL plugin."""

    joints = np.asarray(joint_positions_rad, dtype=np.float64)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise CollisionValidationError("Joint positions must be a finite six-vector")
    transform = np.eye(4)
    origins = [np.zeros(3, dtype=np.float64)]
    for alpha, a, d, joint in zip(
        model.dh_alpha_rad,
        model.dh_a_m,
        model.dh_d_m,
        joints,
        strict=True,
    ):
        translation = np.eye(4)
        translation[:3, 3] = [a, 0.0, d]
        transform = transform @ _rot_x(float(alpha)) @ translation
        origins.append(transform[:3, 3].copy())
        transform = transform @ _rot_z(float(joint))
    points = np.asarray(origins, dtype=np.float64)
    points.setflags(write=False)
    return points, PoseSE3("base", "tcp", transform)


def _capsules(
    model: Cs68KinematicsModel,
    hand_eye: HandEyeCalibration,
    joints: NDArray[np.float64],
    config: CollisionConfig,
) -> tuple[Capsule, ...]:
    if config.link_radii_m is None or config.camera_tool_radius_m is None:
        raise CollisionValidationError(
            "Collision link radii and camera/tool radius must be configured"
        )
    origins, base_t_tcp = cs68_mdh_joint_origins(model, joints)
    links = tuple(
        Capsule(f"link_{index}", origins[index], origins[index + 1], radius)
        for index, radius in enumerate(config.link_radii_m)
    )
    camera = base_t_tcp.compose(hand_eye.tcp_t_left_ir).translation_m
    return (*links, Capsule("camera_tool", origins[-1], camera, config.camera_tool_radius_m))


def _segment_distance(first: Capsule, second: Capsule) -> float:
    u = first.end_m - first.start_m
    v = second.end_m - second.start_m
    w = first.start_m - second.start_m
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denominator = a * c - b * b
    small = 1e-12
    if a <= small and c <= small:
        return float(np.linalg.norm(w))
    if a <= small:
        first_t, second_t = 0.0, float(np.clip(e / c, 0.0, 1.0))
    elif c <= small:
        first_t, second_t = float(np.clip(-d / a, 0.0, 1.0)), 0.0
    else:
        first_t = (
            float(np.clip((b * e - c * d) / denominator, 0.0, 1.0))
            if abs(denominator) > small
            else 0.0
        )
        second_t = float(np.clip((b * first_t + e) / c, 0.0, 1.0))
        first_t = float(np.clip((b * second_t - d) / a, 0.0, 1.0))
    separation = w + first_t * u - second_t * v
    return float(np.linalg.norm(separation))


def _segment_intersects_expanded_box(
    capsule: Capsule,
    obstacle: CollisionObstacleConfig,
    clearance_m: float,
) -> bool:
    expansion = capsule.radius_m + clearance_m
    lower = np.asarray(obstacle.minimum_m, dtype=np.float64) - expansion
    upper = np.asarray(obstacle.maximum_m, dtype=np.float64) + expansion
    direction = capsule.end_m - capsule.start_m
    near, far = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if capsule.start_m[axis] < lower[axis] or capsule.start_m[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - capsule.start_m[axis]) / direction[axis]
        second = (upper[axis] - capsule.start_m[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if near > far:
            return False
    return True


def validate_joint_path_collision(
    start_joint_positions_rad: ArrayLike,
    end_joint_positions_rad: ArrayLike,
    model: Cs68KinematicsModel,
    hand_eye: HandEyeCalibration,
    config: CollisionConfig,
) -> JointPathCollisionReport:
    """Check joint limits and conservative capsule collisions along a linear joint path."""

    start = np.asarray(start_joint_positions_rad, dtype=np.float64)
    end = np.asarray(end_joint_positions_rad, dtype=np.float64)
    if start.shape != (6,) or end.shape != (6,) or not np.isfinite((start, end)).all():
        raise CollisionValidationError("Path endpoints must be finite joint six-vectors")
    if config.minimum_joint_positions_rad is None or config.maximum_joint_positions_rad is None:
        raise CollisionValidationError("Collision joint limits must be configured")
    if config.require_obstacles and not config.obstacles:
        raise CollisionValidationError("At least one workcell obstacle must be configured")
    segments = max(1, ceil(float(np.max(np.abs(end - start))) / config.maximum_joint_step_rad))
    findings: list[CollisionFinding] = []
    lower = np.asarray(config.minimum_joint_positions_rad)
    upper = np.asarray(config.maximum_joint_positions_rad)
    for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, segments + 1)):
        joints = start + fraction * (end - start)
        capsules = _capsules(model, hand_eye, joints, config)
        outside = np.flatnonzero((joints < lower) | (joints > upper))
        for joint_index in outside:
            findings.append(
                CollisionFinding(
                    sample_index,
                    float(fraction),
                    "joint_limit",
                    f"joint_{joint_index}",
                    "configured_limits",
                    "Joint position is outside configured limits",
                )
            )
        for first_index, first in enumerate(capsules):
            for second_index in range(first_index + 2, len(capsules)):
                second = capsules[second_index]
                collision_distance = (
                    first.radius_m + second.radius_m + config.minimum_clearance_m
                )
                if _segment_distance(first, second) <= collision_distance:
                    findings.append(
                        CollisionFinding(
                            sample_index,
                            float(fraction),
                            "self_collision",
                            first.capsule_id,
                            second.capsule_id,
                            "Capsule separation is below the configured clearance",
                        )
                    )
            for obstacle in config.obstacles:
                if first_index in obstacle.ignored_capsule_indices:
                    continue
                if _segment_intersects_expanded_box(first, obstacle, config.minimum_clearance_m):
                    findings.append(
                        CollisionFinding(
                            sample_index,
                            float(fraction),
                            "workcell_collision",
                            first.capsule_id,
                            obstacle.name,
                            "Capsule intersects the clearance-expanded workcell box",
                        )
                    )
    return JointPathCollisionReport(segments + 1, tuple(findings), config.maximum_joint_step_rad)
