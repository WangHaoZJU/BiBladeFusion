"""Offline-only Elite ES68 inverse-kinematics reachability checker.

The default numerical path is a scoped adaptation of HoloRobot's
``EliteCsKinematicModel.solve_ik`` at the pinned provenance commit.  The vendor
KDL object remains injectable for artifact-compatibility tests, but ordinary
candidate generation no longer loads the noisy SDK plugin.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.devices.robot.conversions import se3_to_elite_kdl_pose
from biblade_fusion.planning.collision import cs68_mdh_joint_origins
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

    def solve(
        self,
        target_base_t_flange: PoseSE3,
        seed: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        candidates = (seed, *(_joint_seed(value) for value in _HOLOROBOT_IK_PRESET_SEEDS))
        seen: set[tuple[float, ...]] = set()
        for candidate in candidates:
            initial = self._clamp(candidate)
            key = tuple(round(float(value), 6) for value in initial)
            if key in seen:
                continue
            seen.add(key)
            solution = self._solve_single(target_base_t_flange, initial)
            if solution is not None:
                return solution
        return None

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
        epsilon = 1e-6
        for _ in range(max_iterations):
            _, current_pose = cs68_mdh_joint_origins(self._model, joints)
            position_error = target.translation_m - current_pose.translation_m
            rotation_error = _so3_error(target.rotation, current_pose.rotation)
            if (
                np.linalg.norm(position_error) < position_tolerance_m
                and np.linalg.norm(rotation_error) < rotation_tolerance_rad
            ):
                return joints
            jacobian = np.zeros((6, 6), dtype=np.float64)
            for joint_index in range(6):
                perturbed = joints.copy()
                perturbed[joint_index] += epsilon
                _, perturbed_pose = cs68_mdh_joint_origins(self._model, perturbed)
                jacobian[:3, joint_index] = (
                    perturbed_pose.translation_m - current_pose.translation_m
                ) / epsilon
                jacobian[3:, joint_index] = _so3_error(
                    perturbed_pose.rotation,
                    current_pose.rotation,
                ) / epsilon
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
        if solver is None and native_module is None:
            self._solver = _HoloRobotMdhIkSolver(model)
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
            try:
                solution = self._solver.solve(base_t_flange, self._near)
            except Exception as exc:
                return ReachabilityResult(
                    ReachabilityState.UNKNOWN,
                    f"HoloRobot MDH IK call failed: {exc}",
                )
            if solution is None:
                return ReachabilityResult(
                    ReachabilityState.UNREACHABLE,
                    "HoloRobot MDH IK found no endpoint solution",
                )
            return ReachabilityResult(
                ReachabilityState.REACHABLE,
                "HoloRobot MDH endpoint IK solution found; collision and trajectory "
                "remain unchecked",
                _joint_seed(solution),
            )

        base_t_tcp = base_t_flange.compose(self._flange_t_tcp)
        target = se3_to_elite_kdl_pose(base_t_tcp)
        try:
            ok, solution, result = self._solver.getPositionIK(target, self._near)
        except Exception as exc:
            return ReachabilityResult(
                ReachabilityState.UNKNOWN,
                f"Elite KDL IK call failed: {exc}",
            )
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
