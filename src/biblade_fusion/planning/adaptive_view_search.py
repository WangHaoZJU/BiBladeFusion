"""Offline IK-aware pose-family search around one ideal camera view."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import product
from math import cos, radians, sin
from time import monotonic
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.planning_deadline import require_planning_time
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ViewFilterConfig
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning.filtering import (
    BladeClearanceEnvelope,
    CandidateStatus,
    EvaluatedCandidate,
    ReachabilityChecker,
    ReachabilityResult,
    ReachabilityState,
    filter_candidate_views,
)
from biblade_fusion.planning.views import CandidateView


class AdaptiveViewSearchError(ValueError):
    """An ideal view cannot define a finite deterministic search family."""


class AdaptiveSearchTermination(StrEnum):
    """Typed reason that a bounded pose-family search stopped.

    A budget-limited prefix is not evidence that the remaining camera poses are
    unreachable.  Keeping that distinction typed prevents the coarse selector
    from reporting a physical IK failure when it merely exhausted an experiment-
    time budget.
    """

    FAMILY_EXHAUSTED = "family_exhausted"
    GENERATED_CANDIDATE_LIMIT = "generated_candidate_limit"
    IK_ATTEMPT_LIMIT = "ik_attempt_limit"
    DURATION_LIMIT = "duration_limit"
    FEASIBLE_CANDIDATE_LIMIT = "feasible_candidate_limit"


def _finite_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True, slots=True)
class AdaptiveViewSearchConfig:
    """Bounded search policy; distance limits are sensor limits, not preferences."""

    minimum_optical_distance_m: float = 0.15
    maximum_optical_distance_m: float = 1.5
    distance_step_m: float = 0.04
    maximum_distance_expansions: int = 64
    tilt_samples_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0)
    azimuth_samples_deg: tuple[float, ...] = (
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
    )
    roll_samples_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0)
    maximum_generated_candidates: int = 512
    maximum_ik_feasible_candidates: int = 8
    maximum_ik_attempts_per_family: int = 32
    maximum_search_duration_s: float = 1.5
    sampling_order: Literal["tilt_major", "distance_major"] = "tilt_major"
    ranking_mode: Literal["surface_quality", "fin_discovery"] = "surface_quality"
    require_attempted_per_tilt: bool = False

    def __post_init__(self) -> None:
        lower = float(self.minimum_optical_distance_m)
        upper = float(self.maximum_optical_distance_m)
        step = float(self.distance_step_m)
        if not np.isfinite((lower, upper, step)).all() or lower <= 0.0 or lower >= upper:
            raise ValueError("adaptive optical distance limits must be finite and ordered")
        if step <= 0.0:
            raise ValueError("distance_step_m must be positive")
        if self.maximum_distance_expansions < 0:
            raise ValueError("maximum_distance_expansions must be non-negative")
        tilts = _finite_tuple(self.tilt_samples_deg, name="tilt_samples_deg")
        azimuths = _finite_tuple(self.azimuth_samples_deg, name="azimuth_samples_deg")
        rolls = _finite_tuple(self.roll_samples_deg, name="roll_samples_deg")
        if any(not 0.0 <= value < 90.0 for value in tilts):
            raise ValueError("tilt samples must lie in [0, 90)")
        if len(set(tilts)) != len(tilts):
            raise ValueError("tilt samples must be unique")
        if len(set(azimuths)) != len(azimuths) or len(set(rolls)) != len(rolls):
            raise ValueError("azimuth and roll samples must be unique")
        if self.maximum_generated_candidates < 1:
            raise ValueError("maximum_generated_candidates must be positive")
        if self.maximum_ik_feasible_candidates < 1:
            raise ValueError("maximum_ik_feasible_candidates must be positive")
        if self.maximum_ik_attempts_per_family < 1:
            raise ValueError("maximum_ik_attempts_per_family must be positive")
        if not np.isfinite(self.maximum_search_duration_s) or self.maximum_search_duration_s <= 0:
            raise ValueError("maximum_search_duration_s must be finite and positive")
        object.__setattr__(self, "tilt_samples_deg", tilts)
        object.__setattr__(self, "azimuth_samples_deg", azimuths)
        object.__setattr__(self, "roll_samples_deg", rolls)


@dataclass(frozen=True, slots=True)
class CandidatePoseParameters:
    distance_m: float
    tilt_deg: float
    azimuth_deg: float
    roll_deg: float
    expansion_index: int


@dataclass(frozen=True, slots=True)
class EndpointConfigurationCheck:
    """One exact IK branch's endpoint collision verdict.

    The validator is deliberately independent of the camera-pose IK interface so
    planning code can reuse the already loaded URDF/STL collision backend without
    importing a robotics implementation into this geometry module.
    """

    clear: bool
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reasons = tuple(str(value).strip() for value in self.blocking_reasons)
        if any(not value for value in reasons):
            raise ValueError("endpoint blocking reasons must be non-empty strings")
        if self.clear and reasons:
            raise ValueError("a clear endpoint cannot carry blocking reasons")
        if not self.clear and not reasons:
            raise ValueError("a blocked endpoint requires a blocking reason")
        object.__setattr__(self, "blocking_reasons", reasons)


EndpointConfigurationValidator = Callable[[NDArray[np.float64]], EndpointConfigurationCheck]


@dataclass(frozen=True, slots=True)
class MultiSeedIkEvaluation:
    result: ReachabilityResult
    solutions_rad: tuple[NDArray[np.float64], ...]
    chosen_solution_index: int | None
    messages: tuple[str, ...]
    endpoint_collision_checked: bool = False
    solution_blocking_reasons: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        solutions = []
        for value in self.solutions_rad:
            joints = np.array(value, dtype=np.float64, copy=True)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError("multi-seed IK solutions must be finite six-vectors")
            joints.setflags(write=False)
            solutions.append(joints)
        if self.chosen_solution_index is not None and not (
            0 <= self.chosen_solution_index < len(solutions)
        ):
            raise ValueError("chosen IK solution index is out of range")
        reasons = tuple(
            tuple(str(item) for item in values)
            for values in self.solution_blocking_reasons
        )
        if self.endpoint_collision_checked and len(reasons) != len(solutions):
            raise ValueError("endpoint collision results must match all IK solutions")
        if not self.endpoint_collision_checked and reasons:
            raise ValueError("unchecked IK solutions cannot carry collision results")
        if (
            self.endpoint_collision_checked
            and self.chosen_solution_index is not None
            and reasons[self.chosen_solution_index]
        ):
            raise ValueError("chosen IK solution cannot be endpoint-collision blocked")
        object.__setattr__(self, "solutions_rad", tuple(solutions))
        object.__setattr__(self, "solution_blocking_reasons", reasons)


@dataclass(frozen=True, slots=True)
class AdaptiveCandidateAttempt:
    parameters: CandidatePoseParameters
    evaluated: EvaluatedCandidate
    ik_solutions_rad: tuple[NDArray[np.float64], ...] = ()
    chosen_solution_index: int | None = None
    endpoint_collision_checked: bool = False
    solution_blocking_reasons: tuple[tuple[str, ...], ...] = ()

    @property
    def ik_feasible(self) -> bool:
        return self.evaluated.status is CandidateStatus.ENDPOINT_FEASIBLE


@dataclass(frozen=True, slots=True)
class AdaptiveViewSearchResult:
    nominal_view_id: str
    attempts: tuple[AdaptiveCandidateAttempt, ...]
    ranked_feasible: tuple[AdaptiveCandidateAttempt, ...]
    truncated: bool
    termination: AdaptiveSearchTermination = AdaptiveSearchTermination.FAMILY_EXHAUSTED

    @property
    def recommended(self) -> AdaptiveCandidateAttempt | None:
        return self.ranked_feasible[0] if self.ranked_feasible else None

    @property
    def motion_authorized(self) -> bool:
        return False


class _PrecomputedReachabilityChecker:
    def __init__(self, result: ReachabilityResult) -> None:
        self._result = result

    def check(self, _base_t_left_ir: PoseSE3) -> ReachabilityResult:
        return self._result


class EndpointCollisionAwareReachabilityChecker:
    """Expose IK reachability only after all returned branches are endpoint-clear."""

    def __init__(
        self,
        checker: ReachabilityChecker,
        current_joint_positions_rad: ArrayLike,
        endpoint_validator: EndpointConfigurationValidator,
    ) -> None:
        self._checker = checker
        self._current = np.asarray(current_joint_positions_rad, dtype=np.float64)
        if self._current.shape != (6,) or not np.isfinite(self._current).all():
            raise ValueError("current joints must be a finite six-vector")
        self._endpoint_validator = endpoint_validator

    def check(self, base_t_left_ir: PoseSE3) -> ReachabilityResult:
        return evaluate_multi_seed_ik(
            base_t_left_ir,
            (self._checker,),
            self._current,
            self._endpoint_validator,
        ).result


def evaluate_multi_seed_ik(
    pose: PoseSE3,
    checkers: Sequence[ReachabilityChecker],
    current_joint_positions_rad: ArrayLike,
    endpoint_validator: EndpointConfigurationValidator | None = None,
) -> MultiSeedIkEvaluation:
    """Keep all IK branches and select the nearest collision-clear endpoint."""

    current = np.asarray(current_joint_positions_rad, dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("current joints must be a finite six-vector")
    if not checkers:
        result = ReachabilityResult(
            ReachabilityState.UNKNOWN,
            "adaptive search has no endpoint IK checker",
        )
        return MultiSeedIkEvaluation(result, (), None, (result.message,))

    # The production HoloRobot adapter exposes its ordered seed sweep lazily.
    # Validate each branch immediately and stop on the first clear endpoint;
    # continue only when the preceding mathematical solution collides.  Generic
    # injected checkers keep the legacy collect-and-rank behavior below.
    if endpoint_validator is not None and len(checkers) == 1:
        iter_checks = getattr(checkers[0], "iter_checks", None)
        if callable(iter_checks):
            outcomes: list[ReachabilityResult] = []
            solutions: list[NDArray[np.float64]] = []
            blocking: list[tuple[str, ...]] = []
            for outcome in iter_checks(pose):
                require_planning_time("while evaluating HoloRobot IK branches")
                outcomes.append(outcome)
                if (
                    outcome.state is not ReachabilityState.REACHABLE
                    or outcome.joint_positions_rad is None
                ):
                    continue
                solution = outcome.joint_positions_rad
                require_planning_time("before IK endpoint collision check")
                check = endpoint_validator(solution)
                require_planning_time("after IK endpoint collision check")
                if type(check) is not EndpointConfigurationCheck:
                    raise TypeError(
                        "endpoint validator must return EndpointConfigurationCheck"
                    )
                solutions.append(solution)
                blocking.append(check.blocking_reasons)
                if check.clear:
                    chosen = len(solutions) - 1
                    result = ReachabilityResult(
                        ReachabilityState.REACHABLE,
                        f"{len(solutions)} IK solution(s) evaluated in HoloRobot seed "
                        "order; first endpoint-collision-clear branch selected; "
                        "trajectory remains unchecked",
                        solution,
                    )
                    return MultiSeedIkEvaluation(
                        result,
                        tuple(solutions),
                        chosen,
                        tuple(item.message for item in outcomes),
                        True,
                        tuple(blocking),
                    )
            messages = tuple(item.message for item in outcomes)
            if solutions:
                unique_reasons = tuple(
                    dict.fromkeys(
                        reason for branch in blocking for reason in branch
                    )
                )
                result = ReachabilityResult(
                    ReachabilityState.UNREACHABLE,
                    f"{len(solutions)} IK solution(s) found, but every endpoint is "
                    f"collision blocked: {' | '.join(unique_reasons)}",
                )
                return MultiSeedIkEvaluation(
                    result,
                    tuple(solutions),
                    None,
                    messages,
                    True,
                    tuple(blocking),
                )
            state = (
                ReachabilityState.UNKNOWN
                if any(item.state is ReachabilityState.UNKNOWN for item in outcomes)
                else ReachabilityState.UNREACHABLE
            )
            result = ReachabilityResult(
                state,
                "no IK solution found in bounded HoloRobot seed sweep: "
                + " | ".join(messages),
            )
            return MultiSeedIkEvaluation(result, (), None, messages)

    outcomes_list: list[ReachabilityResult] = []
    for checker in checkers:
        require_planning_time("before endpoint IK checker")
        check_all = getattr(checker, "check_all", None)
        if callable(check_all):
            outcomes_list.extend(check_all(pose))
        else:
            outcomes_list.append(checker.check(pose))
        require_planning_time("after endpoint IK checker")
    outcomes = tuple(outcomes_list)
    messages = tuple(outcome.message for outcome in outcomes)
    solutions = tuple(
        outcome.joint_positions_rad
        for outcome in outcomes
        if outcome.state is ReachabilityState.REACHABLE
        and outcome.joint_positions_rad is not None
    )
    solution_blocking_reasons: tuple[tuple[str, ...], ...] = ()
    eligible = tuple(range(len(solutions)))
    if solutions and endpoint_validator is not None:
        checked_solutions = []
        for solution in solutions:
            require_planning_time("before IK endpoint collision check")
            checked_solutions.append(endpoint_validator(solution))
            require_planning_time("after IK endpoint collision check")
        checks = tuple(checked_solutions)
        if any(type(check) is not EndpointConfigurationCheck for check in checks):
            raise TypeError("endpoint validator must return EndpointConfigurationCheck")
        solution_blocking_reasons = tuple(check.blocking_reasons for check in checks)
        eligible = tuple(index for index, check in enumerate(checks) if check.clear)
    if eligible:
        chosen = min(
            eligible,
            key=lambda index: (
                float(np.max(np.abs(solutions[index] - current))),
                float(np.sum(np.abs(solutions[index] - current))),
                index,
            ),
        )
        result = ReachabilityResult(
            ReachabilityState.REACHABLE,
            f"{len(solutions)} IK solution(s) found from {len(checkers)} checker(s); "
            + (
                f"{len(eligible)} endpoint-collision clear; trajectory remains unchecked"
                if endpoint_validator is not None
                else "endpoint collision and trajectory remain unchecked"
            ),
            solutions[chosen],
        )
        return MultiSeedIkEvaluation(
            result,
            solutions,
            chosen,
            messages,
            endpoint_validator is not None,
            solution_blocking_reasons,
        )

    if solutions and endpoint_validator is not None:
        unique_reasons = tuple(
            dict.fromkeys(
                reason
                for branch_reasons in solution_blocking_reasons
                for reason in branch_reasons
            )
        )
        result = ReachabilityResult(
            ReachabilityState.UNREACHABLE,
            f"{len(solutions)} IK solution(s) found, but every endpoint is collision "
            f"blocked: {' | '.join(unique_reasons)}",
        )
        return MultiSeedIkEvaluation(
            result,
            solutions,
            None,
            messages,
            True,
            solution_blocking_reasons,
        )

    state = (
        ReachabilityState.UNKNOWN
        if any(item.state is ReachabilityState.UNKNOWN for item in outcomes)
        else ReachabilityState.UNREACHABLE
    )
    result = ReachabilityResult(
        state,
        f"no IK solution found from {len(checkers)} checker(s): " + " | ".join(messages),
    )
    return MultiSeedIkEvaluation(result, (), None, messages)


def _distance_samples(
    nominal_distance_m: float,
    config: AdaptiveViewSearchConfig,
) -> tuple[float, ...]:
    if not config.minimum_optical_distance_m <= nominal_distance_m <= (
        config.maximum_optical_distance_m
    ):
        raise AdaptiveViewSearchError(
            "ideal candidate lies outside the physical optical distance limits"
        )
    values = [float(nominal_distance_m)]
    for expansion in range(1, config.maximum_distance_expansions + 1):
        inward = nominal_distance_m - expansion * config.distance_step_m
        outward = nominal_distance_m + expansion * config.distance_step_m
        if inward >= config.minimum_optical_distance_m:
            values.append(float(inward))
        if outward <= config.maximum_optical_distance_m:
            values.append(float(outward))
    return tuple(values)


def _tangent_basis(candidate: CandidateView) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal = candidate.patch.outward_normal
    tangent_x = candidate.base_t_left_ir.rotation[:, 0].copy()
    tangent_x -= normal * float(tangent_x @ normal)
    if np.linalg.norm(tangent_x) <= 1e-9:
        tangent_x = candidate.base_t_left_ir.rotation[:, 1].copy()
        tangent_x -= normal * float(tangent_x @ normal)
    norm = float(np.linalg.norm(tangent_x))
    if norm <= 1e-9:
        raise AdaptiveViewSearchError("ideal view cannot define a stable tangent basis")
    tangent_x /= norm
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    return tangent_x, tangent_y


def _candidate_from_parameters(
    nominal: CandidateView,
    parameters: CandidatePoseParameters,
    tangent_x: NDArray[np.float64],
    tangent_y: NDArray[np.float64],
    sequence_index: int,
) -> CandidateView:
    normal = nominal.patch.outward_normal
    tilt = radians(parameters.tilt_deg)
    azimuth = radians(parameters.azimuth_deg)
    roll = radians(parameters.roll_deg)
    tangent_direction = cos(azimuth) * tangent_x + sin(azimuth) * tangent_y
    camera_direction = cos(tilt) * normal + sin(tilt) * tangent_direction
    camera_direction /= np.linalg.norm(camera_direction)
    position = nominal.patch.target_m + parameters.distance_m * camera_direction

    camera_z = -camera_direction
    camera_x = nominal.base_t_left_ir.rotation[:, 0].copy()
    camera_x -= camera_z * float(camera_x @ camera_z)
    if np.linalg.norm(camera_x) <= 1e-9:
        camera_x = tangent_y - camera_z * float(tangent_y @ camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    rolled_x = cos(roll) * camera_x + sin(roll) * camera_y
    rolled_y = -sin(roll) * camera_x + cos(roll) * camera_y
    rotation = np.column_stack((rolled_x, rolled_y, camera_z))
    view_id = f"{nominal.view_id}_adaptive_{sequence_index:04d}"
    pose = PoseSE3.from_rotation_translation(
        "base",
        f"{view_id}_left_ir",
        rotation,
        position,
    )
    footprint_scale = parameters.distance_m / nominal.standoff_distance_m
    angular_support = float(np.clip(cos(tilt), 0.0, 1.0))
    return CandidateView(
        view_id,
        nominal.patch,
        pose,
        parameters.distance_m,
        tuple(float(value * footprint_scale) for value in nominal.footprint_m),
        angular_support,
        angular_support,
        "adaptive_ik_aware_pose_family_v1",
    )


def generate_adaptive_candidate_family(
    nominal: CandidateView,
    config: AdaptiveViewSearchConfig,
) -> tuple[tuple[CandidatePoseParameters, CandidateView], ...]:
    """Generate ideal-first candidates, preferring normal incidence over tilt."""

    tangent_x, tangent_y = _tangent_basis(nominal)
    distances = _distance_samples(nominal.standoff_distance_m, config)
    family = []
    tilts = config.tilt_samples_deg
    rolls = config.roll_samples_deg

    def azimuths_for(tilt: float) -> tuple[float, ...]:
        return (0.0,) if tilt == 0.0 else config.azimuth_samples_deg

    # A bounded IK budget must be a representative prefix of the search space.
    # Lexicographic Cartesian products starved real runs: 32 attempts could all
    # share one tilt/roll or one distance.  Seed each independent dimension first,
    # then expand the remaining Cartesian product in diagonal shells.
    ordered_indices: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add(indices: tuple[int, int, int, int]) -> None:
        if indices not in seen:
            seen.add(indices)
            ordered_indices.append(indices)

    add((0, 0, 0, 0))
    for tilt_index in range(1, len(tilts)):
        add((0, tilt_index, 0, 0))
    for roll_index in range(1, len(rolls)):
        add((0, 0, 0, roll_index))
    azimuth_anchor = next(
        (index for index, tilt in enumerate(tilts) if tilt != 0.0),
        0,
    )
    for azimuth_index in range(1, len(azimuths_for(tilts[azimuth_anchor]))):
        add((0, azimuth_anchor, azimuth_index, 0))
    # Distance indices 1 and 2 are the first inward/outward samples whenever both
    # lie inside the sensor limits.  Cover them before deeper expansions.
    for distance_index in range(1, min(3, len(distances))):
        add((distance_index, 0, 0, 0))

    remaining = []
    for distance_index, tilt_index, roll_index in product(
        range(len(distances)),
        range(len(tilts)),
        range(len(rolls)),
    ):
        for azimuth_index in range(len(azimuths_for(tilts[tilt_index]))):
            indices = (distance_index, tilt_index, azimuth_index, roll_index)
            if indices in seen:
                continue
            remaining.append(indices)
    remaining.sort(
        key=lambda item: (
            sum(item),
            max(item),
            item[0] if config.sampling_order == "distance_major" else item[1],
            item[1] if config.sampling_order == "distance_major" else item[0],
            item[2],
            item[3],
        )
    )
    ordered_indices.extend(remaining)

    ordered_parameters = (
        (
            tilts[tilt_index],
            distance_index,
            distances[distance_index],
            azimuths_for(tilts[tilt_index])[azimuth_index],
            rolls[roll_index],
        )
        for distance_index, tilt_index, azimuth_index, roll_index in ordered_indices
    )
    for sequence_index, (
        tilt,
        distance_index,
        distance,
        azimuth,
        roll,
    ) in enumerate(ordered_parameters):
        parameters = CandidatePoseParameters(
            distance,
            tilt,
            azimuth,
            roll,
            distance_index,
        )
        family.append(
            (
                parameters,
                _candidate_from_parameters(
                    nominal,
                    parameters,
                    tangent_x,
                    tangent_y,
                    sequence_index,
                ),
            )
        )
        if len(family) >= config.maximum_generated_candidates:
            return tuple(family)
    return tuple(family)


def _attempt_rank(
    attempt: AdaptiveCandidateAttempt,
    current: NDArray[np.float64],
    nominal_distance_m: float,
    ranking_mode: Literal["surface_quality", "fin_discovery"],
) -> tuple[float | str, ...]:
    joints = attempt.evaluated.joint_positions_rad
    assert joints is not None
    delta = np.abs(joints - current)
    if ranking_mode == "fin_discovery":
        tilt = radians(attempt.parameters.tilt_deg)
        # A thin fin face appears approximately with sin(tilt), while useful
        # support on the blade surface falls with cos(tilt). Their product is a
        # conservative geometry-only information proxy, peaking near 45 degrees.
        information_score = sin(tilt) * cos(tilt)
    else:
        information_score = attempt.evaluated.metrics.geometric_score
    return (
        -information_score,
        -attempt.evaluated.metrics.geometric_score,
        float(np.max(delta)),
        float(np.sum(delta)),
        abs(attempt.parameters.distance_m - nominal_distance_m),
        abs(attempt.parameters.tilt_deg),
        abs(attempt.parameters.roll_deg),
        abs(attempt.parameters.azimuth_deg),
        attempt.evaluated.candidate.view_id,
    )


def search_adaptive_candidate_family(
    nominal: CandidateView,
    proxy: BilateralBladeProxy | BladeClearanceEnvelope,
    filter_config: ViewFilterConfig,
    ik_checkers: Sequence[ReachabilityChecker],
    current_joint_positions_rad: ArrayLike,
    config: AdaptiveViewSearchConfig | None = None,
    *,
    endpoint_validator: EndpointConfigurationValidator | None = None,
    monotonic_clock=monotonic,
) -> AdaptiveViewSearchResult:
    """Find endpoint IK alternatives, optionally rejecting colliding IK branches."""

    policy = config or AdaptiveViewSearchConfig()
    current = np.asarray(current_joint_positions_rad, dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("current joints must be a finite six-vector")
    family = generate_adaptive_candidate_family(nominal, policy)
    attempts = []
    feasible = []
    attempted_tilts: set[float] = set()
    search_started = float(monotonic_clock())
    ik_attempts = 0
    termination = AdaptiveSearchTermination.FAMILY_EXHAUSTED
    for parameters, candidate in family:
        require_planning_time(f"before adaptive pose {candidate.view_id}")
        if ik_attempts >= policy.maximum_ik_attempts_per_family:
            termination = AdaptiveSearchTermination.IK_ATTEMPT_LIMIT
            break
        if float(monotonic_clock()) - search_started >= policy.maximum_search_duration_s:
            termination = AdaptiveSearchTermination.DURATION_LIMIT
            break
        attempted_tilts.add(parameters.tilt_deg)
        geometry = filter_candidate_views(
            (candidate,),
            proxy,
            filter_config,
            None,
            deduplicate=False,
        ).candidates[0]
        if geometry.status is CandidateStatus.REJECTED:
            attempt = AdaptiveCandidateAttempt(parameters, geometry)
        else:
            ik_attempts += 1
            ik = evaluate_multi_seed_ik(
                candidate.base_t_left_ir,
                ik_checkers,
                current,
                endpoint_validator,
            )
            evaluated = filter_candidate_views(
                (candidate,),
                proxy,
                filter_config,
                _PrecomputedReachabilityChecker(ik.result),
                deduplicate=False,
            ).candidates[0]
            attempt = AdaptiveCandidateAttempt(
                parameters,
                evaluated,
                ik.solutions_rad,
                ik.chosen_solution_index,
                ik.endpoint_collision_checked,
                ik.solution_blocking_reasons,
            )
        attempts.append(attempt)
        if attempt.ik_feasible:
            feasible.append(attempt)
            has_required_tilt_sampling = (
                not policy.require_attempted_per_tilt
                or attempted_tilts == set(policy.tilt_samples_deg)
            )
            if (
                len(feasible) >= policy.maximum_ik_feasible_candidates
                and has_required_tilt_sampling
            ):
                termination = AdaptiveSearchTermination.FEASIBLE_CANDIDATE_LIMIT
                break

    natural_family_size = sum(
        len(policy.azimuth_samples_deg) if tilt != 0.0 else 1
        for tilt in policy.tilt_samples_deg
    ) * len(_distance_samples(nominal.standoff_distance_m, policy)) * len(
        policy.roll_samples_deg
    )
    if (
        termination is AdaptiveSearchTermination.FAMILY_EXHAUSTED
        and len(family) < natural_family_size
    ):
        termination = AdaptiveSearchTermination.GENERATED_CANDIDATE_LIMIT

    ranked = tuple(
        sorted(
            feasible,
            key=lambda attempt: _attempt_rank(
                attempt,
                current,
                nominal.standoff_distance_m,
                policy.ranking_mode,
            ),
        )
    )
    truncated = termination is not AdaptiveSearchTermination.FAMILY_EXHAUSTED
    return AdaptiveViewSearchResult(
        nominal.view_id,
        tuple(attempts),
        ranked,
        truncated,
        termination,
    )


def adaptive_view_search_payload(
    result: AdaptiveViewSearchResult,
    config: AdaptiveViewSearchConfig,
    current_joint_positions_rad: ArrayLike,
    *,
    source_initialization: str | None = None,
    source_kinematics: str | None = None,
) -> dict[str, object]:
    """Create a transparent JSON-ready diagnostic report."""

    current = np.asarray(current_joint_positions_rad, dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("current joints must be a finite six-vector")
    rank_by_id = {
        item.evaluated.candidate.view_id: rank
        for rank, item in enumerate(result.ranked_feasible, start=1)
    }
    attempts = []
    for attempt in result.attempts:
        candidate = attempt.evaluated.candidate
        attempts.append(
            {
                "view_id": candidate.view_id,
                "patch_id": candidate.patch.patch_id,
                "parameters": asdict(attempt.parameters),
                "base_T_left_ir": candidate.base_t_left_ir.matrix.tolist(),
                "status": attempt.evaluated.status.value,
                "reasons": list(attempt.evaluated.reasons),
                "metrics": asdict(attempt.evaluated.metrics),
                "ik_solutions_rad": [item.tolist() for item in attempt.ik_solutions_rad],
                "chosen_solution_index": attempt.chosen_solution_index,
                "endpoint_collision_checked": attempt.endpoint_collision_checked,
                "ik_solution_blocking_reasons": [
                    list(reasons) for reasons in attempt.solution_blocking_reasons
                ],
                "chosen_joint_positions_rad": (
                    attempt.evaluated.joint_positions_rad.tolist()
                    if attempt.evaluated.joint_positions_rad is not None
                    else None
                ),
                "rank": rank_by_id.get(candidate.view_id),
            }
        )
    recommended = result.recommended
    return {
        "schema_version": 2,
        "artifact_kind": "biblade_fusion.offline_adaptive_view_search",
        "motion_authorized": False,
        "endpoint_collision_checked": any(
            attempt.ik_solutions_rad for attempt in result.attempts
        ) and all(
            attempt.endpoint_collision_checked
            for attempt in result.attempts
            if attempt.ik_solutions_rad
        ),
        "trajectory_checked": False,
        "nominal_view_id": result.nominal_view_id,
        "current_joint_positions_rad": current.tolist(),
        "configuration": asdict(config),
        "sources": {
            "initialization": source_initialization,
            "kinematics": source_kinematics,
        },
        "summary": {
            "attempt_count": len(result.attempts),
            "ik_feasible_count": len(result.ranked_feasible),
            "truncated": result.truncated,
            "termination": result.termination.value,
            "recommended_view_id": (
                recommended.evaluated.candidate.view_id if recommended else None
            ),
        },
        "attempts": attempts,
    }
