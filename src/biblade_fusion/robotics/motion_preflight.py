"""HoloRobot-style conservative CS68 joint planning and offline preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.devices.robot.streaming import ServoJStream
from biblade_fusion.robotics.cs68_model import Cs68KinematicModel
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
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    approval_required: bool = True

    @property
    def ready_for_approval(self) -> bool:
        return (
            self.status is MotionPreflightStatus.CLEAR
            and self.servoj_stream is not None
            and self.collision is not None
            and self.collision.status is CollisionCheckStatus.CLEAR
        )

    @property
    def motion_authorized(self) -> bool:
        return False


def _joint_vector(values: ArrayLike, *, label: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite CS68 six-vector")
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
            begin + index / segments * (end - begin)
            for begin, end in zip(start, goal, strict=True)
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
    if dt_s <= 0.0:
        raise ValueError("ServoJ dt_s must be positive")
    if not 0.0 < speed_scaling <= 1.0:
        raise ValueError("ServoJ speed scaling must be in (0, 1]")
    if not 0.0 < velocity_margin <= 1.0:
        raise ValueError("ServoJ velocity margin must be in (0, 1]")
    maximum_steps = tuple(
        velocity * dt_s * speed_scaling * velocity_margin
        for velocity in maximum_velocity_rad_s
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


def preflight_linear_joint_motion(
    start_joint_positions_rad: ArrayLike,
    goal_joint_positions_rad: ArrayLike,
    *,
    collision_checker: Cs68PinocchioCollisionChecker | None,
    maximum_joint_step_rad: float = 0.02,
    servoj_dt_s: float = 0.004,
    speed_scaling: float = 0.08,
    velocity_margin: float = 0.8,
) -> JointMotionPreflight:
    """Plan, collision-check, and time-parameterize one conservative linear joint leg."""

    start = _joint_vector(start_joint_positions_rad, label="motion start")
    goal = _joint_vector(goal_joint_positions_rad, label="motion goal")
    waypoints = _linear_waypoints(
        start,
        goal,
        maximum_joint_step_rad=maximum_joint_step_rad,
    )
    diagnostics = {
        "planner": "holorobot_conservative_linear_joint",
        "trajectory_generator": "holorobot_velocity_limited_servoj",
        "maximum_joint_step_rad": maximum_joint_step_rad,
        "servoj_dt_s": servoj_dt_s,
        "speed_scaling": speed_scaling,
        "velocity_margin": velocity_margin,
        "provenance": robot_stack_provenance(),
        "motion_authorized": False,
    }
    if collision_checker is None:
        return JointMotionPreflight(
            status=MotionPreflightStatus.CHECKER_UNAVAILABLE,
            start_joint_positions_rad=start,
            goal_joint_positions_rad=goal,
            planning_waypoints=waypoints,
            servoj_stream=None,
            collision=None,
            blocking_reasons=("checker_unavailable",),
            warnings=(),
            diagnostics=diagnostics,
        )
    collision = collision_checker.check_path(
        start,
        goal,
        maximum_joint_step_rad=maximum_joint_step_rad,
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
            blocking_reasons=reasons,
            warnings=(),
            diagnostics=diagnostics,
        )
    velocities = Cs68KinematicModel.from_resources().joint_velocity_limits_rad_s()
    stream = _velocity_limited_stream(
        waypoints,
        maximum_velocity_rad_s=velocities,
        dt_s=servoj_dt_s,
        speed_scaling=speed_scaling,
        velocity_margin=velocity_margin,
    )
    return JointMotionPreflight(
        status=MotionPreflightStatus.CLEAR,
        start_joint_positions_rad=start,
        goal_joint_positions_rad=goal,
        planning_waypoints=waypoints,
        servoj_stream=stream,
        collision=collision,
        blocking_reasons=(),
        warnings=("acceleration_limits_unavailable",),
        diagnostics=diagnostics,
    )
