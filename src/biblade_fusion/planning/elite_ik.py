"""Offline-only Elite CS68 inverse-kinematics reachability checker."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.devices.robot.conversions import se3_to_elite_kdl_pose
from biblade_fusion.planning.filtering import ReachabilityResult, ReachabilityState


class EliteIkError(RuntimeError):
    """The offline Elite KDL solver could not be initialized."""


def _joint_seed(value: ArrayLike) -> NDArray[np.float64]:
    joints = np.array(value, dtype=np.float64, copy=True)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("Elite IK seed must be a finite six-vector")
    joints.setflags(write=False)
    return joints


def _resolve_plugin(native_module: Any, configured_path: Path | None) -> Path:
    if configured_path is not None:
        path = configured_path.resolve()
    else:
        path = Path(native_module.__file__).resolve().parent / "libelite_kdl_kinematics.so"
    if not path.is_file():
        raise EliteIkError(f"Elite KDL kinematics plugin is missing: {path}")
    return path


class EliteCs68IkChecker:
    """Use the vendor KDL plugin for IK without connecting to or moving the robot."""

    def __init__(
        self,
        model: Cs68KinematicsModel,
        hand_eye: HandEyeCalibration,
        near_joint_positions_rad: ArrayLike,
        config: KinematicsConfig,
        *,
        native_module: ModuleType | Any | None = None,
        solver: Any | None = None,
    ) -> None:
        self._hand_eye = hand_eye
        self._near = _joint_seed(near_joint_positions_rad)
        self._loader: Any | None = None
        if solver is None:
            native = native_module or import_module("elite_cs_sdk.elite_cs_sdk_python")
            plugin_path = _resolve_plugin(native, config.plugin_path)
            loader = native.ClassLoader(str(plugin_path))
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
            raise EliteIkError(f"Failed to configure Elite KDL solver: {exc}") from exc
        self._solver = solver

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
        base_t_tcp = canonical_camera_pose.compose(
            self._hand_eye.tcp_t_left_ir.inverse()
        )
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
