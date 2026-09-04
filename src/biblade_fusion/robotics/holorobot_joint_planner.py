"""Bounded HoloRobot RRTConnect fallback adapted to BiBladeFusion collision ports.

The planning order follows HoloRobot's ``CompositeMotionPlanner`` and
``OmplJointPlanner``: reject invalid endpoints, try the conservative straight
path first elsewhere, and invoke RRTConnect only for a blocked path.  The blade
runtime deliberately uses one bounded solve rather than HoloRobot's general
five-attempt default; the returned path is still resampled and independently
rechecked by :mod:`biblade_fusion.robotics.motion_preflight`.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Any

import numpy as np


class HoloRobotJointPlanStatus(StrEnum):
    CLEAR = "clear"
    PATH_BLOCKED = "path_blocked"
    COLLISION_AT_START = "collision_at_start"
    COLLISION_AT_GOAL = "collision_at_goal"
    PLANNER_UNAVAILABLE = "planner_unavailable"


StateValidityCallback = Callable[[tuple[float, ...]], tuple[bool, tuple[str, ...]]]


@dataclass(frozen=True, slots=True)
class HoloRobotOmplConfig:
    """Small experiment-time budget around HoloRobot's RRTConnect backend."""

    maximum_joint_step_rad: float = 0.02
    plan_timeout_s: float = 1.0
    rrt_range_rad: float = 0.25
    simplify_path: bool = True

    def __post_init__(self) -> None:
        values = (
            self.maximum_joint_step_rad,
            self.plan_timeout_s,
            self.rrt_range_rad,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("OMPL joint planner limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class HoloRobotJointPlan:
    status: HoloRobotJointPlanStatus
    waypoints: tuple[tuple[float, ...], ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    diagnostics: dict[str, object] | None = None

    @property
    def clear(self) -> bool:
        return self.status is HoloRobotJointPlanStatus.CLEAR


def ompl_available() -> bool:
    """Return whether the optional OMPL Python bindings are importable."""

    try:
        return (
            importlib.util.find_spec("ompl") is not None
            and importlib.util.find_spec("ompl.base") is not None
            and importlib.util.find_spec("ompl.geometric") is not None
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _joint_vector(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    joints = tuple(float(value) for value in values)
    if len(joints) != 6 or not np.isfinite(joints).all():
        raise ValueError(f"{label} must be a finite ES68 six-vector")
    return joints


def resample_joint_path(
    positions: Sequence[Sequence[float]],
    *,
    maximum_joint_step_rad: float,
) -> tuple[tuple[float, ...], ...]:
    """Port HoloRobot's joint-path resampling contract without its domain types."""

    step = float(maximum_joint_step_rad)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("maximum_joint_step_rad must be finite and positive")
    raw = tuple(_joint_vector(value, label="OMPL waypoint") for value in positions)
    if len(raw) < 2:
        raise ValueError("OMPL path requires at least start and goal")
    result: list[tuple[float, ...]] = [raw[0]]
    for start, goal in zip(raw[:-1], raw[1:], strict=True):
        maximum_delta = max(
            abs(end - begin) for begin, end in zip(start, goal, strict=True)
        )
        subdivisions = max(1, math.ceil(maximum_delta / step))
        result.extend(
            tuple(
                begin + sample_index / subdivisions * (end - begin)
                for begin, end in zip(start, goal, strict=True)
            )
            for sample_index in range(1, subdivisions)
        )
        # Keep the authoritative OMPL knot object itself.  Recomputing it as
        # ``start + 1.0 * (goal - start)`` is only numerically equivalent and
        # can differ by one ULP, which violates the downstream hash-bound
        # start/goal identity contract.
        result.append(goal)
    return tuple(result)


def _anchor_ompl_solution_endpoints(
    positions: tuple[tuple[float, ...], ...],
    *,
    start: tuple[float, ...],
    goal: tuple[float, ...],
) -> tuple[tuple[tuple[float, ...], ...], float, float]:
    """Restore the exact authoritative endpoints after OMPL/simplifier roundoff.

    OMPL stores states in its own floating representation and PathSimplifier may
    return numerically equivalent endpoint values that are not tuple-identical to
    the requested robot states.  The downstream preflight deliberately binds the
    exact current/goal vectors, so anchor them before resampling.  Every resulting
    segment is still checked in full by the caller.
    """

    if len(positions) < 2:
        raise ValueError("OMPL solution requires at least start and goal")
    start_drift = max(
        abs(actual - expected)
        for actual, expected in zip(positions[0], start, strict=True)
    )
    goal_drift = max(
        abs(actual - expected)
        for actual, expected in zip(positions[-1], goal, strict=True)
    )
    anchored = (start, *positions[1:-1], goal)
    return anchored, start_drift, goal_drift


_OmplValidityChecker: type[Any] | None = None


def _validity_checker_class() -> type[Any]:
    global _OmplValidityChecker
    if _OmplValidityChecker is not None:
        return _OmplValidityChecker

    import ompl.base as ob

    class OmplValidityChecker(ob.StateValidityChecker):
        def __init__(
            self,
            space_information: Any,
            *,
            state_validity: StateValidityCallback,
        ) -> None:
            super().__init__(space_information)
            self._state_validity = state_validity

        def isValid(self, state: Any) -> bool:
            configuration = tuple(float(state[index]) for index in range(6))
            clear, _reasons = self._state_validity(configuration)
            return bool(clear)

    _OmplValidityChecker = OmplValidityChecker
    return _OmplValidityChecker


def plan_holorobot_rrtconnect(
    start_joint_positions_rad: Sequence[float],
    goal_joint_positions_rad: Sequence[float],
    *,
    joint_limits_rad: Sequence[tuple[float, float]],
    state_validity: StateValidityCallback,
    config: HoloRobotOmplConfig,
    monotonic_clock: Callable[[], float] = monotonic,
) -> HoloRobotJointPlan:
    """Run one bounded HoloRobot-style RRTConnect search."""

    start = _joint_vector(start_joint_positions_rad, label="OMPL start")
    goal = _joint_vector(goal_joint_positions_rad, label="OMPL goal")
    limits = tuple((float(lower), float(upper)) for lower, upper in joint_limits_rad)
    if (
        len(limits) != 6
        or not np.isfinite(limits).all()
        or any(lower >= upper for lower, upper in limits)
    ):
        raise ValueError("OMPL joint limits must contain six finite ordered pairs")

    for label, configuration, status in (
        ("start", start, HoloRobotJointPlanStatus.COLLISION_AT_START),
        ("goal", goal, HoloRobotJointPlanStatus.COLLISION_AT_GOAL),
    ):
        clear, reasons = state_validity(configuration)
        if not clear:
            return HoloRobotJointPlan(
                status,
                blocking_reasons=reasons or (f"{label}_not_clear",),
                diagnostics={"planner": "holorobot_ompl_rrtconnect"},
            )

    if not ompl_available():
        return HoloRobotJointPlan(
            HoloRobotJointPlanStatus.PLANNER_UNAVAILABLE,
            blocking_reasons=("ompl_python_bindings_unavailable",),
            diagnostics={"planner": "holorobot_ompl_rrtconnect"},
        )

    import ompl.base as ob
    import ompl.geometric as og
    import ompl.util as ou

    # OMPL's default INFO stream is noisy in the supervised operator console.
    # Typed status/diagnostics below retain the actionable result.
    ou.setLogLevel(ou.LOG_WARN)

    started = float(monotonic_clock())
    space = ob.RealVectorStateSpace(6)
    bounds = ob.RealVectorBounds(6)
    for index, (lower, upper) in enumerate(limits):
        bounds.setLow(index, lower)
        bounds.setHigh(index, upper)
    space.setBounds(bounds)

    space_information = ob.SpaceInformation(space)
    space_information.setStateValidityChecker(
        _validity_checker_class()(
            space_information,
            state_validity=state_validity,
        )
    )
    extent = max(1e-6, float(space_information.getMaximumExtent()))
    space_information.setStateValidityCheckingResolution(
        config.maximum_joint_step_rad / extent
    )
    space_information.setup()

    start_state = space.allocState()
    goal_state = space.allocState()
    for index, value in enumerate(start):
        start_state[index] = value
    for index, value in enumerate(goal):
        goal_state[index] = value

    problem = ob.ProblemDefinition(space_information)
    problem.setStartAndGoalStates(start_state, goal_state)
    planner = og.RRTConnect(space_information)
    planner.setRange(config.rrt_range_rad)
    planner.setProblemDefinition(problem)
    planner.setup()
    solved = planner.solve(config.plan_timeout_s)
    if not solved or not problem.hasSolution():
        return HoloRobotJointPlan(
            HoloRobotJointPlanStatus.PATH_BLOCKED,
            blocking_reasons=("ompl_planning_timeout",),
            diagnostics={
                "planner": "holorobot_ompl_rrtconnect",
                "plan_timeout_s": config.plan_timeout_s,
                "elapsed_s": max(0.0, float(monotonic_clock()) - started),
            },
        )

    path = problem.getSolutionPath()
    if config.simplify_path:
        simplify_timeout = min(0.25, max(0.05, config.plan_timeout_s * 0.2))
        og.PathSimplifier(space_information).simplify(path, simplify_timeout)
    raw = tuple(
        tuple(float(path.getState(index)[joint]) for joint in range(6))
        for index in range(path.getStateCount())
    )
    raw, raw_start_drift_rad, raw_goal_drift_rad = (
        _anchor_ompl_solution_endpoints(raw, start=start, goal=goal)
    )
    waypoints = resample_joint_path(
        raw,
        maximum_joint_step_rad=config.maximum_joint_step_rad,
    )
    return HoloRobotJointPlan(
        HoloRobotJointPlanStatus.CLEAR,
        waypoints=waypoints,
        diagnostics={
            "planner": "holorobot_ompl_rrtconnect",
            "plan_timeout_s": config.plan_timeout_s,
            "rrt_range_rad": config.rrt_range_rad,
            "simplify_path": config.simplify_path,
            "raw_waypoint_count": len(raw),
            "resampled_waypoint_count": len(waypoints),
            "exact_endpoints_anchored": True,
            "raw_start_drift_rad": raw_start_drift_rad,
            "raw_goal_drift_rad": raw_goal_drift_rad,
            "elapsed_s": max(0.0, float(monotonic_clock()) - started),
        },
    )
