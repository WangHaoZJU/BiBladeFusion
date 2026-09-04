"""Offline-only Elite ES68 inverse-kinematics reachability checker.

The default numerical path is a scoped adaptation of HoloRobot's
``EliteCsKinematicModel.solve_ik`` at the pinned provenance commit.  The vendor
KDL object remains injectable for artifact-compatibility tests, but ordinary
candidate generation no longer loads the noisy SDK plugin.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.planning_deadline import (
    PlanningDeadlineExceeded,
    require_planning_time,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.devices.robot.conversions import se3_to_elite_kdl_pose
from biblade_fusion.planning.filtering import ReachabilityResult, ReachabilityState
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp


class EliteIkError(RuntimeError):
    """The offline Elite KDL solver could not be initialized."""


def _joint_seed(value: ArrayLike) -> NDArray[np.float64]:
    joints = np.array(value, dtype=np.float64, copy=True)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("Elite IK seed must be a finite six-vector")
    joints.setflags(write=False)
    return joints


_HOLOROBOT_IK_PRESET_SEEDS: tuple[tuple[float, ...], ...] = (
    (0.0, -1.57, 1.57, -1.57, -1.57, 0.0),
    (0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
    (0.0, -1.0, 1.0, -1.0, 0.0, 0.0),
)


def _holorobot_near_seed_perturbations(
    seed: NDArray[np.float64],
    magnitude_rad: float = 0.12,
) -> tuple[NDArray[np.float64], ...]:
    """Copy HoloRobot active-mapping's bounded nearby-seed sweep."""

    magnitude = abs(float(magnitude_rad))
    half = magnitude * 0.5
    deltas = (
        (magnitude, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-magnitude, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, magnitude, 0.0, 0.0),
        (0.0, 0.0, 0.0, -magnitude, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, magnitude, 0.0),
        (0.0, 0.0, 0.0, 0.0, -magnitude, 0.0),
        (half, 0.0, 0.0, 0.0, magnitude, 0.0),
        (-half, 0.0, 0.0, 0.0, magnitude, 0.0),
        (half, 0.0, 0.0, 0.0, -magnitude, 0.0),
        (-half, 0.0, 0.0, 0.0, -magnitude, 0.0),
        (half, 0.0, 0.0, magnitude, 0.0, 0.0),
        (-half, 0.0, 0.0, -magnitude, 0.0, 0.0),
    )
    return tuple(seed + np.asarray(delta, dtype=np.float64) for delta in deltas)

_JOINT_AXIS_Z = np.array((0.0, 0.0, 1.0), dtype=np.float64)


def _rot_x4(angle: float) -> NDArray[np.float64]:
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[1:3, 1:3] = ((cosine, -sine), (sine, cosine))
    return result


def _rot_z4(angle: float) -> NDArray[np.float64]:
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = ((cosine, -sine), (sine, cosine))
    return result


def _forward_pose_and_jacobian(
    model: Cs68KinematicsModel,
    joints: NDArray[np.float64],
) -> tuple[PoseSE3, NDArray[np.float64]]:
    """HoloRobot analytic segment Jacobian for the Elite fixed-MDH chain."""

    transform = np.eye(4, dtype=np.float64)
    axes: list[NDArray[np.float64]] = []
    origins: list[NDArray[np.float64]] = []
    for alpha, a, d, joint in zip(
        model.dh_alpha_rad,
        model.dh_a_m,
        model.dh_d_m,
        joints,
        strict=True,
    ):
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = (a, 0.0, d)
        transform = transform @ _rot_x4(float(alpha)) @ translation
        axes.append(transform[:3, :3] @ _JOINT_AXIS_Z)
        origins.append(transform[:3, 3].copy())
        transform = transform @ _rot_z4(float(joint))
    endpoint = transform[:3, 3]
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
        jacobian[:3, index] = np.cross(axis, endpoint - origin)
        jacobian[3:, index] = axis
    return PoseSE3("base", "flange", transform), jacobian


def _nearest_equivalent_joints(
    solution: ArrayLike,
    reference: ArrayLike,
    limits: tuple[tuple[float, float], ...],
) -> NDArray[np.float64]:
    """Choose the valid 2-pi representation nearest the current robot posture."""

    values = _joint_seed(solution)
    near = _joint_seed(reference)
    result = values.copy()
    period = 2.0 * math.pi
    for index, (minimum, maximum) in enumerate(limits):
        shifts = range(
            math.ceil((minimum - values[index]) / period),
            math.floor((maximum - values[index]) / period) + 1,
        )
        candidates = [float(values[index] + shift * period) for shift in shifts]
        if candidates:
            result[index] = min(candidates, key=lambda item: abs(item - near[index]))
    result.setflags(write=False)
    return result


def _so3_error(target: NDArray[np.float64], current: NDArray[np.float64]) -> NDArray[np.float64]:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(theta)
    if abs(sine) < 1e-8:
        # The adaptive view family never intentionally requests an exact pi
        # discontinuity, but a finite fallback keeps a bad candidate contained.
        eigenvalues, eigenvectors = np.linalg.eigh((relative + np.eye(3)) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return np.asarray(axis * theta, dtype=np.float64)
    return np.asarray(
        (
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ),
        dtype=np.float64,
    ) * (theta / (2.0 * sine))


def _damped_least_squares_step(
    jacobian: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    damping: float,
) -> NDArray[np.float64]:
    """Copy HoloRobot's bounded damped-least-squares update."""

    lhs = jacobian @ jacobian.T + (damping**2) * np.eye(6, dtype=np.float64)
    return jacobian.T @ np.linalg.solve(lhs, error)


class _HoloRobotMdhIkSolver:
    """HoloRobot numerical IK applied to the controller-specific MDH chain."""

    def __init__(self, model: Cs68KinematicsModel) -> None:
        self._model = model
        self._joint_limits = Es68KinematicModel.from_resources().joint_limit_pairs()
        self._cache: OrderedDict[
            tuple[float, ...], tuple[NDArray[np.float64], ...]
        ] = OrderedDict()

    def solve(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        return next(iter(self.iter_solutions(target_base_t_flange, seed)), None)

    def solve_all(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        key = tuple(round(float(value), 9) for value in target_base_t_flange.matrix.ravel())
        key += tuple(round(float(value), 6) for value in seed)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        return tuple(self.iter_solutions(target_base_t_flange, seed))

    def iter_solutions(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> Iterator[NDArray[np.float64]]:
        """Yield HoloRobot-ordered branches so a clear first branch can stop work."""

        key = tuple(round(float(value), 9) for value in target_base_t_flange.matrix.ravel())
        key += tuple(round(float(value), 6) for value in seed)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            for solution in cached:
                require_planning_time("while yielding cached MDH IK branches")
                yield solution
            return
        candidates = (
            seed,
            *_holorobot_near_seed_perturbations(seed),
            *(_joint_seed(value) for value in _HOLOROBOT_IK_PRESET_SEEDS),
        )
        seen: set[tuple[float, ...]] = set()
        solutions: list[NDArray[np.float64]] = []
        for candidate in candidates:
            require_planning_time("before MDH IK seed")
            initial = self._clamp(candidate)
            seed_key = tuple(round(float(value), 6) for value in initial)
            if seed_key in seen:
                continue
            seen.add(seed_key)
            solution = self._solve_single(target_base_t_flange, initial)
            require_planning_time("after MDH IK seed")
            if solution is not None:
                normalized = _nearest_equivalent_joints(
                    solution,
                    seed,
                    self._joint_limits,
                )
                if not any(
                    np.allclose(normalized, item, atol=1e-5, rtol=0.0)
                    for item in solutions
                ):
                    solutions.append(normalized)
                    yield normalized
        result = tuple(solutions)
        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > 1024:
            self._cache.popitem(last=False)
        return result

    def _solve_single(
        self,
        target: PoseSE3,
        seed: NDArray[np.float64],
        *,
        max_iterations: int = 120,
        position_tolerance_m: float = 1e-4,
        rotation_tolerance_rad: float = 1e-3,
    ) -> NDArray[np.float64] | None:
        joints = seed.copy()
        for _ in range(max_iterations):
            require_planning_time("during MDH IK iteration")
            current_pose, jacobian = _forward_pose_and_jacobian(self._model, joints)
            position_error = target.translation_m - current_pose.translation_m
            rotation_error = _so3_error(target.rotation, current_pose.rotation)
            if (
                np.linalg.norm(position_error) < position_tolerance_m
                and np.linalg.norm(rotation_error) < rotation_tolerance_rad
            ):
                return joints
            error = np.concatenate((position_error, rotation_error))
            try:
                delta = _damped_least_squares_step(
                    jacobian,
                    error,
                    damping=0.05,
                )
            except np.linalg.LinAlgError:
                return None
            joints = self._clamp(joints + delta)
        return None

    def _clamp(self, joints: ArrayLike) -> NDArray[np.float64]:
        result = _joint_seed(joints).copy()
        for index, (minimum, maximum) in enumerate(self._joint_limits):
            result[index] = float(np.clip(result[index], minimum, maximum))
        return result


class _HoloRobotPinocchioIkSolver:
    """HoloRobot's preferred URDF/Pinocchio IK with its active-mapping seed sweep."""

    def __init__(self, pinocchio_model: Any) -> None:
        self._model = pinocchio_model
        self._data = pinocchio_model.model.createData()
        self._pin = import_module("pinocchio")
        self._joint_limits = Es68KinematicModel.from_resources().joint_limit_pairs()
        self._cache: OrderedDict[
            tuple[float, ...], tuple[NDArray[np.float64], ...]
        ] = OrderedDict()

    def solve(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        return next(iter(self.iter_solutions(target_base_t_flange, seed)), None)

    def solve_all(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        return tuple(self.iter_solutions(target_base_t_flange, seed))

    def iter_solutions(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> Iterator[NDArray[np.float64]]:
        """Yield nearby branches lazily in the proven HoloRobot seed order."""

        key = tuple(round(float(value), 9) for value in target_base_t_flange.matrix.ravel())
        key += tuple(round(float(value), 6) for value in seed)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            for solution in cached:
                require_planning_time("while yielding cached Pinocchio IK branches")
                yield solution
            return

        candidates = (
            seed,
            *_holorobot_near_seed_perturbations(seed),
            *(_joint_seed(value) for value in _HOLOROBOT_IK_PRESET_SEEDS),
        )
        target = self._pin.SE3(
            target_base_t_flange.rotation,
            target_base_t_flange.translation_m,
        )
        seen: set[tuple[float, ...]] = set()
        solutions: list[NDArray[np.float64]] = []
        for candidate in candidates:
            require_planning_time("before Pinocchio IK seed")
            controller_seed = self._clamp(candidate)
            seed_key = tuple(round(float(value), 6) for value in controller_seed)
            if seed_key in seen:
                continue
            seen.add(seed_key)
            solution = self._solve_single(target, controller_seed)
            require_planning_time("after Pinocchio IK seed")
            if solution is not None:
                # Consumers may stop after the first clear endpoint.  If it is
                # blocked, iteration continues through the bounded HoloRobot seeds.
                normalized = _nearest_equivalent_joints(
                    solution,
                    seed,
                    self._joint_limits,
                )
                if not any(
                    np.allclose(normalized, item, atol=1e-5, rtol=0.0)
                    for item in solutions
                ):
                    solutions.append(normalized)
                    yield normalized

        result = tuple(solutions)

        self._cache[key] = result
        self._cache.move_to_end(key)
        while len(self._cache) > 1024:
            self._cache.popitem(last=False)
        return result

    def _solve_single(
        self,
        target: Any,
        controller_seed: NDArray[np.float64],
        *,
        maximum_iterations: int = 120,
        position_tolerance_m: float = 1e-4,
        rotation_tolerance_rad: float = 1e-3,
    ) -> NDArray[np.float64] | None:
        configuration = self._model._to_configuration(controller_seed)
        for _ in range(maximum_iterations):
            require_planning_time("during Pinocchio IK iteration")
            self._pin.forwardKinematics(self._model.model, self._data, configuration)
            self._pin.updateFramePlacements(self._model.model, self._data)
            current = self._data.oMf[self._model.tool_frame_id]
            error = self._pin.log6(current.inverse() * target).vector
            if (
                np.linalg.norm(error[:3]) <= position_tolerance_m
                and np.linalg.norm(error[3:]) <= rotation_tolerance_rad
            ):
                joints = self._model.controller_joints_from_configuration(configuration)
                normalized = _nearest_equivalent_joints(
                    joints,
                    controller_seed,
                    self._joint_limits,
                )
                if all(
                    minimum - 1e-9 <= value <= maximum + 1e-9
                    for value, (minimum, maximum) in zip(
                        normalized,
                        self._joint_limits,
                        strict=True,
                    )
                ):
                    return normalized
                return None
            jacobian = self._pin.computeFrameJacobian(
                self._model.model,
                self._data,
                configuration,
                self._model.tool_frame_id,
                self._pin.ReferenceFrame.LOCAL,
            )
            try:
                step = _damped_least_squares_step(
                    np.asarray(jacobian, dtype=np.float64),
                    np.asarray(error, dtype=np.float64),
                    damping=1e-4,
                )
            except np.linalg.LinAlgError:
                return None
            configuration = self._pin.integrate(
                self._model.model,
                configuration,
                step * 0.25,
            )
        return None

    def _clamp(self, joints: ArrayLike) -> NDArray[np.float64]:
        result = _joint_seed(joints).copy()
        for index, (minimum, maximum) in enumerate(self._joint_limits):
            result[index] = float(np.clip(result[index], minimum, maximum))
        return result


def _resolve_plugin(native_module: Any, configured_path: Path | None) -> Path:
    if configured_path is not None:
        path = configured_path.resolve()
    else:
        path = Path(native_module.__file__).resolve().parent / "libelite_kdl_kinematics.so"
    if not path.is_file():
        raise EliteIkError(f"Elite KDL kinematics plugin is missing: {path}")
    return path


class EliteCs68IkChecker:
    """Use HoloRobot numerical IK without connecting to or moving the robot."""

    def __init__(
        self,
        model: Cs68KinematicsModel,
        hand_eye: HandEyeCalibration,
        near_joint_positions_rad: ArrayLike,
        config: KinematicsConfig,
        *,
        native_module: Any | None = None,
        solver: Any | None = None,
        pinocchio_model: Any | None = None,
    ) -> None:
        try:
            self._flange_t_left_ir = hand_eye.require_flange_primary()
            self._flange_t_tcp = load_es68_flange_t_tcp()
        except (OSError, ValueError) as exc:
            raise EliteIkError(
                f"Elite IK requires authoritative flange-primary hand-eye: {exc}"
            ) from exc
        self._near = _joint_seed(near_joint_positions_rad)
        self._loader: Any | None = None
        self._solver_label = "Elite KDL"
        if solver is None and native_module is None:
            if pinocchio_model is not None:
                self._solver = _HoloRobotPinocchioIkSolver(pinocchio_model)
                self._solver_label = "HoloRobot Pinocchio/URDF"
            else:
                self._solver = _HoloRobotMdhIkSolver(model)
                self._solver_label = "HoloRobot analytic MDH"
            self._uses_vendor_solver = False
            return
        if solver is None:
            plugin_path = _resolve_plugin(native_module, config.plugin_path)
            loader = native_module.ClassLoader(str(plugin_path))
            if not loader.loadLib():
                raise EliteIkError(f"Failed to load Elite KDL plugin: {plugin_path}")
            solver = loader.createKinematicsInstance("ELITE::KdlKinematicsPlugin")
            if solver is None:
                raise EliteIkError("Failed to create Elite KDL kinematics instance")
            self._loader = loader
        try:
            solver.setMDH(model.dh_alpha_rad, model.dh_a_m, model.dh_d_m)
            solver.setDefaultTimeout(config.ik_timeout_s)
        except Exception as exc:
            raise EliteIkError(f"Failed to configure injected Elite KDL solver: {exc}") from exc
        self._solver = solver
        self._uses_vendor_solver = True

    def check(self, base_t_left_ir: PoseSE3) -> ReachabilityResult:
        if base_t_left_ir.parent_frame != "base" or not base_t_left_ir.child_frame.endswith(
            "left_ir"
        ):
            return ReachabilityResult(
                ReachabilityState.UNKNOWN,
                "Elite IK requires a base_T_left_ir candidate pose",
            )
        canonical_camera_pose = PoseSE3(
            "base",
            "left_ir",
            base_t_left_ir.matrix,
        )
        base_t_flange = canonical_camera_pose.compose(
            self._flange_t_left_ir.inverse()
        )
        if not self._uses_vendor_solver:
            require_planning_time("before HoloRobot IK solve")
            try:
                solution = self._solver.solve(base_t_flange, self._near)
            except PlanningDeadlineExceeded:
                raise
            except Exception as exc:
                return ReachabilityResult(
                    ReachabilityState.UNKNOWN,
                    f"{self._solver_label} IK call failed: {exc}",
                )
            require_planning_time("after HoloRobot IK solve")
            if solution is None:
                return ReachabilityResult(
                    ReachabilityState.UNREACHABLE,
                    f"{self._solver_label} IK found no endpoint solution across "
                    "the bounded seed sweep",
                )
            return ReachabilityResult(
                ReachabilityState.REACHABLE,
                f"{self._solver_label} endpoint IK solution found; collision and trajectory "
                "remain unchecked",
                _joint_seed(solution),
            )

        base_t_tcp = base_t_flange.compose(self._flange_t_tcp)
        target = se3_to_elite_kdl_pose(base_t_tcp)
        require_planning_time("before vendor KDL IK call")
        try:
            ok, solution, result = self._solver.getPositionIK(target, self._near)
        except PlanningDeadlineExceeded:
            raise
        except Exception as exc:
            return ReachabilityResult(
                ReachabilityState.UNKNOWN,
                f"Elite KDL IK call failed: {exc}",
            )
        require_planning_time("after vendor KDL IK call")
        error = str(getattr(result, "kinematic_error", "unknown"))
        if not ok:
            return ReachabilityResult(
                ReachabilityState.UNREACHABLE,
                f"Elite KDL found no endpoint IK solution ({error})",
            )
        try:
            joints = _joint_seed(solution)
        except ValueError as exc:
            return ReachabilityResult(
                ReachabilityState.UNKNOWN,
                f"Elite KDL returned an invalid IK solution: {exc}",
            )
        return ReachabilityResult(
            ReachabilityState.REACHABLE,
            "Elite KDL endpoint IK solution found; collision and trajectory remain unchecked",
            joints,
        )

    def check_all(self, base_t_left_ir: PoseSE3) -> tuple[ReachabilityResult, ...]:
        """Return distinct nearby numerical solutions for motion-aware ranking."""

        if self._uses_vendor_solver:
            return (self.check(base_t_left_ir),)
        if base_t_left_ir.parent_frame != "base" or not base_t_left_ir.child_frame.endswith(
            "left_ir"
        ):
            return (self.check(base_t_left_ir),)
        canonical = PoseSE3("base", "left_ir", base_t_left_ir.matrix)
        target = canonical.compose(self._flange_t_left_ir.inverse())
        require_planning_time("before complete HoloRobot IK branch sweep")
        try:
            solutions = self._solver.solve_all(target, self._near)
        except PlanningDeadlineExceeded:
            raise
        except Exception as exc:
            return (
                ReachabilityResult(
                    ReachabilityState.UNKNOWN,
                    f"{self._solver_label} IK call failed: {exc}",
                ),
            )
        require_planning_time("after complete HoloRobot IK branch sweep")
        if not solutions:
            return (
                ReachabilityResult(
                    ReachabilityState.UNREACHABLE,
                    f"{self._solver_label} IK found no endpoint solution across "
                    "the bounded seed sweep",
                ),
            )
        return tuple(
            ReachabilityResult(
                ReachabilityState.REACHABLE,
                f"{self._solver_label} endpoint IK solution found; collision and "
                "trajectory remain unchecked",
                solution,
            )
            for solution in solutions
        )

    def iter_checks(self, base_t_left_ir: PoseSE3) -> Iterator[ReachabilityResult]:
        """Yield IK branches lazily for endpoint-collision-aware selection."""

        if self._uses_vendor_solver or (
            base_t_left_ir.parent_frame != "base"
            or not base_t_left_ir.child_frame.endswith("left_ir")
        ):
            yield self.check(base_t_left_ir)
            return
        canonical = PoseSE3("base", "left_ir", base_t_left_ir.matrix)
        target = canonical.compose(self._flange_t_left_ir.inverse())
        yielded = False
        try:
            for solution in self._solver.iter_solutions(target, self._near):
                require_planning_time("before yielding HoloRobot IK branch")
                yielded = True
                yield ReachabilityResult(
                    ReachabilityState.REACHABLE,
                    f"{self._solver_label} endpoint IK solution found; collision and "
                    "trajectory remain unchecked",
                    solution,
                )
        except PlanningDeadlineExceeded:
            raise
        except Exception as exc:
            yield ReachabilityResult(
                ReachabilityState.UNKNOWN,
                f"{self._solver_label} IK call failed: {exc}",
            )
            return
        if not yielded:
            yield ReachabilityResult(
                ReachabilityState.UNREACHABLE,
                f"{self._solver_label} IK found no endpoint solution across "
                "the bounded seed sweep",
            )
