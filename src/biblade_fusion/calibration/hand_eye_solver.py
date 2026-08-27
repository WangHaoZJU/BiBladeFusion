"""HoloRobot-aligned ES68 eye-in-hand solving and auditable sample artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.robotics import load_es68_flange_t_tcp
from biblade_fusion.robotics.provenance import robot_stack_provenance

HAND_EYE_SAMPLE_SCHEMA_VERSION = 2


class HandEyeSolveError(ValueError):
    """Hand-eye samples are invalid, degenerate, or fail configured quality limits."""


def _optional_vector6(value: Sequence[float] | None, name: str) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite six-vector")
    array.setflags(write=False)
    return array


def _optional_points(
    value: Any | None,
    columns: int,
    dtype: Any,
    name: str,
) -> NDArray[Any] | None:
    if value is None:
        return None
    array = np.array(value, dtype=dtype, copy=True).reshape(-1, columns)
    if len(array) < 4 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain at least four finite points")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class HandEyeSample:
    """One synchronized fixed-board observation in the calibrated flange chain."""

    sample_id: str
    base_t_flange: PoseSE3
    left_ir_t_target: PoseSE3
    source_session: str | None = None
    charuco_corner_count: int | None = None
    reprojection_rmse_px: float | None = None
    pose_ambiguity_ratio: float | None = None
    joint_positions_rad: NDArray[np.float64] | None = None
    base_t_tcp_observed: PoseSE3 | None = None
    charuco_ids: NDArray[np.int32] | None = None
    image_points_px: NDArray[np.float64] | None = None
    object_points_m: NDArray[np.float64] | None = None
    frame_number: int | None = None
    bracket_ms: float | None = None
    selected_robot_state_offset_ms: float | None = None
    controller_time_s: float | None = None
    robot_mode: str | None = None
    safety_status: str | None = None
    fk_tcp_translation_error_m: float | None = None
    fk_tcp_rotation_error_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("Hand-eye sample ID must be non-empty")
        if (self.base_t_flange.parent_frame, self.base_t_flange.child_frame) != (
            "base",
            "flange",
        ):
            raise ValueError("Hand-eye sample requires base_T_flange")
        if (self.left_ir_t_target.parent_frame, self.left_ir_t_target.child_frame) != (
            "left_ir",
            "target",
        ):
            raise ValueError("Hand-eye sample requires left_ir_T_target")
        if self.base_t_tcp_observed is not None and (
            self.base_t_tcp_observed.parent_frame,
            self.base_t_tcp_observed.child_frame,
        ) != ("base", "tcp"):
            raise ValueError("Observed controller pose must be base_T_tcp")
        if self.charuco_corner_count is not None and self.charuco_corner_count < 4:
            raise ValueError("Hand-eye ChArUco sample needs at least four corners")
        if self.reprojection_rmse_px is not None and self.reprojection_rmse_px < 0.0:
            raise ValueError("Hand-eye reprojection RMSE must be non-negative")
        if (self.charuco_corner_count is None) != (self.reprojection_rmse_px is None):
            raise ValueError("Hand-eye detection metrics must be provided together")
        if self.pose_ambiguity_ratio is not None and self.pose_ambiguity_ratio < 1.0:
            raise ValueError("Hand-eye pose ambiguity ratio must be at least one")
        if self.frame_number is not None and self.frame_number < 0:
            raise ValueError("D435i frame number must be non-negative")
        if self.bracket_ms is not None and self.bracket_ms < 0.0:
            raise ValueError("Capture bracket must be non-negative")
        if (
            self.selected_robot_state_offset_ms is not None
            and self.selected_robot_state_offset_ms < 0.0
        ):
            raise ValueError("Selected robot-state offset must be non-negative")
        if self.controller_time_s is not None and self.controller_time_s < 0.0:
            raise ValueError("Controller time must be non-negative")
        if self.fk_tcp_translation_error_m is not None and self.fk_tcp_translation_error_m < 0.0:
            raise ValueError("FK/TCP translation error must be non-negative")
        if self.fk_tcp_rotation_error_deg is not None and self.fk_tcp_rotation_error_deg < 0.0:
            raise ValueError("FK/TCP rotation error must be non-negative")

        joints = _optional_vector6(self.joint_positions_rad, "ES68 joint positions")
        ids = None
        if self.charuco_ids is not None:
            ids = np.array(self.charuco_ids, dtype=np.int32, copy=True).reshape(-1)
            if len(ids) < 4 or len(set(ids.tolist())) != len(ids):
                raise ValueError("ChArUco IDs must contain at least four unique values")
            ids.setflags(write=False)
        image = _optional_points(self.image_points_px, 2, np.float64, "image points")
        objects = _optional_points(self.object_points_m, 3, np.float64, "object points")
        corner_payload = (ids, image, objects)
        if any(item is None for item in corner_payload) and any(
            item is not None for item in corner_payload
        ):
            raise ValueError("ChArUco IDs, image points, and object points are required together")
        if ids is not None and (len(ids) != len(image) or len(ids) != len(objects)):
            raise ValueError("ChArUco IDs and point arrays must align")
        if ids is not None and self.charuco_corner_count not in (None, len(ids)):
            raise ValueError("ChArUco corner count does not match stored observations")
        object.__setattr__(self, "joint_positions_rad", joints)
        object.__setattr__(self, "charuco_ids", ids)
        object.__setattr__(self, "image_points_px", image)
        object.__setattr__(self, "object_points_m", objects)

    @property
    def has_bundle_adjustment_observation(self) -> bool:
        return self.charuco_ids is not None


@dataclass(frozen=True, slots=True)
class HandEyeSampleRejection:
    sample_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.reason:
            raise ValueError("Rejected hand-eye samples require an ID and reason")


@dataclass(frozen=True, slots=True)
class HandEyeObservability:
    rotation_span_deg: float
    translation_span_m: float
    rotation_axis_diversity: float


@dataclass(frozen=True, slots=True)
class HandEyeBundleAdjustment:
    enabled: bool
    success: bool
    initial_rmse_px: float | None
    final_rmse_px: float | None
    mean_error_px: float | None
    maximum_error_px: float | None
    message: str


@dataclass(frozen=True, slots=True)
class _LeastSquaresResult:
    x: NDArray[np.float64]
    fun: NDArray[np.float64]
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class HandEyeSolution:
    flange_t_left_ir: PoseSE3
    base_t_target: PoseSE3
    method: str
    sample_count: int
    translation_rmse_m: float
    rotation_rmse_deg: float
    rotation_max_deg: float
    observability: HandEyeObservability
    bundle_adjustment: HandEyeBundleAdjustment
    initial_translation_rmse_m: float
    initial_rotation_rmse_deg: float


def _rotation_angle_deg(rotation: NDArray[np.float64]) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pairwise_observability(samples: Sequence[HandEyeSample], cv2: Any) -> HandEyeObservability:
    rotations: list[float] = []
    translations: list[float] = []
    axes: list[NDArray[np.float64]] = []
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            relative = left.base_t_flange.inverse().compose(right.base_t_flange)
            angle = _rotation_angle_deg(relative.rotation)
            rotations.append(angle)
            translations.append(float(np.linalg.norm(relative.translation_m)))
            if angle > 1.0:
                rotation_vector, _ = cv2.Rodrigues(relative.rotation)
                vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
                axes.append(vector / np.linalg.norm(vector))
    if not rotations:
        raise HandEyeSolveError("at least two hand-eye samples are required")
    diversity = 0.0
    if len(axes) >= 2:
        singular_values = np.linalg.svd(np.stack(axes), compute_uv=False)
        if singular_values[0] > 0.0:
            diversity = float(singular_values[1] / singular_values[0])
    return HandEyeObservability(max(rotations), max(translations), diversity)


def _mean_pose(poses: Sequence[PoseSE3]) -> PoseSE3:
    translation = np.mean([pose.translation_m for pose in poses], axis=0)
    rotation_sum = np.sum([pose.rotation for pose in poses], axis=0)
    left, _, right_t = np.linalg.svd(rotation_sum)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return PoseSE3.from_rotation_translation("base", "target", rotation, translation)


def _loop_metrics(
    samples: Sequence[HandEyeSample], flange_t_left_ir: PoseSE3
) -> tuple[PoseSE3, float, float, float]:
    target_poses = [
        sample.base_t_flange.compose(flange_t_left_ir).compose(sample.left_ir_t_target)
        for sample in samples
    ]
    fused = _mean_pose(target_poses)
    translations = np.asarray(
        [np.linalg.norm(pose.translation_m - fused.translation_m) for pose in target_poses]
    )
    rotations = np.asarray(
        [_rotation_angle_deg(fused.rotation.T @ pose.rotation) for pose in target_poses]
    )
    return (
        fused,
        float(np.sqrt(np.mean(translations**2))),
        float(np.sqrt(np.mean(rotations**2))),
        float(np.max(rotations)),
    )


def _skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _se3_exp(vector: NDArray[np.float64], cv2: Any) -> NDArray[np.float64]:
    rho = np.asarray(vector[:3], dtype=np.float64)
    omega = np.asarray(vector[3:], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(omega)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        v_matrix = np.eye(3) + 0.5 * omega_hat + omega_hat @ omega_hat / 6.0
    else:
        theta2 = theta * theta
        v_matrix = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta2 * omega_hat
            + (theta - np.sin(theta)) / (theta2 * theta) * (omega_hat @ omega_hat)
        )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = v_matrix @ rho
    return matrix


def _se3_log(matrix: NDArray[np.float64], cv2: Any) -> NDArray[np.float64]:
    omega, _ = cv2.Rodrigues(matrix[:3, :3])
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        v_inverse = np.eye(3) - 0.5 * omega_hat + omega_hat @ omega_hat / 12.0
    else:
        coefficient = (1.0 - theta / (2.0 * np.tan(theta / 2.0))) / (theta * theta)
        v_inverse = np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
    return np.concatenate((v_inverse @ matrix[:3, 3], omega))


def _camera_matrix(intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ]
    )


def _numpy_levenberg_marquardt(
    residual_function: Any,
    initial: NDArray[np.float64],
    *,
    maximum_iterations: int = 100,
) -> _LeastSquaresResult:
    """Small dense LM fallback for the same 12-parameter objective used by HoloRobot."""

    parameters = np.asarray(initial, dtype=np.float64).copy()
    residual = np.asarray(residual_function(parameters), dtype=np.float64)
    cost = 0.5 * float(residual @ residual)
    damping = 1e-3
    for iteration in range(maximum_iterations):
        jacobian = np.empty((len(residual), len(parameters)), dtype=np.float64)
        for column in range(len(parameters)):
            step = 1e-7 * max(1.0, abs(float(parameters[column])))
            candidate = parameters.copy()
            candidate[column] += step
            jacobian[:, column] = (
                np.asarray(residual_function(candidate), dtype=np.float64) - residual
            ) / step
        gradient = jacobian.T @ residual
        if float(np.linalg.norm(gradient, ord=np.inf)) < 1e-9:
            return _LeastSquaresResult(
                parameters,
                residual,
                True,
                f"NumPy LM converged by gradient after {iteration} iterations",
            )
        normal = jacobian.T @ jacobian
        diagonal = np.maximum(np.diag(normal), 1e-12)
        try:
            delta = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        if float(np.linalg.norm(delta)) < 1e-11 * (float(np.linalg.norm(parameters)) + 1e-11):
            return _LeastSquaresResult(
                parameters,
                residual,
                True,
                f"NumPy LM converged by step after {iteration} iterations",
            )
        proposed = parameters + delta
        proposed_residual = np.asarray(residual_function(proposed), dtype=np.float64)
        proposed_cost = 0.5 * float(proposed_residual @ proposed_residual)
        if proposed_cost < cost:
            relative_drop = (cost - proposed_cost) / max(cost, 1e-30)
            parameters = proposed
            residual = proposed_residual
            cost = proposed_cost
            damping = max(damping / 3.0, 1e-12)
            if relative_drop < 1e-12:
                return _LeastSquaresResult(
                    parameters,
                    residual,
                    True,
                    f"NumPy LM converged by cost after {iteration + 1} iterations",
                )
        else:
            damping = min(damping * 10.0, 1e12)
    return _LeastSquaresResult(
        parameters,
        residual,
        False,
        f"NumPy LM reached {maximum_iterations} iterations",
    )


def _run_lm(residual_function: Any, initial: NDArray[np.float64]) -> _LeastSquaresResult:
    try:
        least_squares = import_module("scipy.optimize").least_squares
    except ModuleNotFoundError:
        return _numpy_levenberg_marquardt(residual_function, initial)
    result = least_squares(residual_function, initial, method="lm")
    return _LeastSquaresResult(
        np.asarray(result.x, dtype=np.float64),
        np.asarray(result.fun, dtype=np.float64),
        bool(result.success),
        str(result.message),
    )


def _refine_bundle_adjustment(
    samples: Sequence[HandEyeSample],
    initial_flange_t_left_ir: PoseSE3,
    intrinsics: CameraIntrinsics,
    cv2: Any,
) -> tuple[PoseSE3, PoseSE3, HandEyeBundleAdjustment]:
    if not all(sample.has_bundle_adjustment_observation for sample in samples):
        raise HandEyeSolveError(
            "LM bundle adjustment requires stored ChArUco IDs and 2D/3D points for every sample"
        )
    initial_base_t_target = (
        samples[0]
        .base_t_flange.compose(initial_flange_t_left_ir)
        .compose(samples[0].left_ir_t_target)
    )
    x0 = np.concatenate(
        (
            _se3_log(initial_flange_t_left_ir.matrix, cv2),
            _se3_log(initial_base_t_target.matrix, cv2),
        )
    )
    camera = _camera_matrix(intrinsics)
    distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        flange_t_left = _se3_exp(parameters[:6], cv2)
        base_t_target = _se3_exp(parameters[6:], cv2)
        left_t_flange = np.linalg.inv(flange_t_left)
        errors: list[NDArray[np.float64]] = []
        for sample in samples:
            left_t_target = (
                left_t_flange @ np.linalg.inv(sample.base_t_flange.matrix) @ base_t_target
            )
            rotation_vector, _ = cv2.Rodrigues(left_t_target[:3, :3])
            projected, _ = cv2.projectPoints(
                sample.object_points_m,
                rotation_vector,
                left_t_target[:3, 3],
                camera,
                distortion,
            )
            errors.append(projected.reshape(-1, 2) - sample.image_points_px)
        return np.concatenate(errors).reshape(-1)

    initial_residual = residuals(x0)
    result = _run_lm(residuals, x0)
    pixel_norms = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
    flange = PoseSE3("flange", "left_ir", _se3_exp(result.x[:6], cv2))
    target = PoseSE3("base", "target", _se3_exp(result.x[6:], cv2))
    metrics = HandEyeBundleAdjustment(
        enabled=True,
        success=bool(result.success),
        initial_rmse_px=float(np.sqrt(np.mean(initial_residual**2))),
        final_rmse_px=float(np.sqrt(np.mean(result.fun**2))),
        mean_error_px=float(np.mean(pixel_norms)),
        maximum_error_px=float(np.max(pixel_norms)),
        message=str(result.message),
    )
    if not metrics.success:
        raise HandEyeSolveError(f"LM bundle adjustment failed: {metrics.message}")
    return flange, target, metrics


def solve_hand_eye(
    samples: Sequence[HandEyeSample],
    config: HandEyeConfig,
    *,
    method: str | None = None,
    intrinsics: CameraIntrinsics | None = None,
    refine: bool | None = None,
) -> HandEyeSolution:
    """Solve ``flange_T_left_ir`` with Daniilidis initialization and HoloRobot LM BA."""

    sample_list = tuple(samples)
    if len(sample_list) < config.minimum_samples:
        raise HandEyeSolveError(
            f"hand-eye sample count {len(sample_list)} is below {config.minimum_samples}"
        )
    if len({sample.sample_id for sample in sample_list}) != len(sample_list):
        raise HandEyeSolveError("hand-eye sample IDs must be unique")
    for sample in sample_list:
        if (
            sample.fk_tcp_translation_error_m is not None
            and sample.fk_tcp_translation_error_m > config.maximum_fk_tcp_translation_error_m
        ):
            raise HandEyeSolveError(
                f"sample {sample.sample_id} FK/TCP translation error "
                f"{sample.fk_tcp_translation_error_m:.6f} m exceeds "
                f"{config.maximum_fk_tcp_translation_error_m:.6f} m"
            )
        if (
            sample.fk_tcp_rotation_error_deg is not None
            and sample.fk_tcp_rotation_error_deg > config.maximum_fk_tcp_rotation_error_deg
        ):
            raise HandEyeSolveError(
                f"sample {sample.sample_id} FK/TCP rotation error "
                f"{sample.fk_tcp_rotation_error_deg:.3f} deg exceeds "
                f"{config.maximum_fk_tcp_rotation_error_deg:.3f} deg"
            )
    cv2 = import_module("cv2")
    method_name = (method or config.initial_method).lower().replace("_", "-")
    methods = {
        "park": (cv2.CALIB_HAND_EYE_PARK, "OpenCV Park-Martin"),
        "tsai": (cv2.CALIB_HAND_EYE_TSAI, "OpenCV Tsai-Lenz"),
        "horaud": (cv2.CALIB_HAND_EYE_HORAUD, "OpenCV Horaud"),
        "andreff": (cv2.CALIB_HAND_EYE_ANDREFF, "OpenCV Andreff"),
        "daniilidis": (cv2.CALIB_HAND_EYE_DANIILIDIS, "OpenCV Daniilidis"),
    }
    if method_name not in methods:
        raise HandEyeSolveError(f"unsupported hand-eye method: {method_name}")

    observability = _pairwise_observability(sample_list, cv2)
    if observability.rotation_span_deg < config.minimum_rotation_span_deg:
        raise HandEyeSolveError(
            f"rotation span {observability.rotation_span_deg:.3f} deg is below "
            f"{config.minimum_rotation_span_deg:.3f} deg"
        )
    if observability.translation_span_m < config.minimum_translation_span_m:
        raise HandEyeSolveError(
            f"translation span {observability.translation_span_m:.6f} m is below "
            f"{config.minimum_translation_span_m:.6f} m"
        )
    if observability.rotation_axis_diversity < config.minimum_rotation_axis_diversity:
        raise HandEyeSolveError(
            f"rotation-axis diversity {observability.rotation_axis_diversity:.4f} is below "
            f"{config.minimum_rotation_axis_diversity:.4f}"
        )

    algorithm, display_name = methods[method_name]
    try:
        camera_to_flange_rotation, camera_to_flange_translation = cv2.calibrateHandEye(
            [sample.base_t_flange.rotation for sample in sample_list],
            [sample.base_t_flange.translation_m for sample in sample_list],
            [sample.left_ir_t_target.rotation for sample in sample_list],
            [sample.left_ir_t_target.translation_m for sample in sample_list],
            method=algorithm,
        )
        initial = PoseSE3.from_rotation_translation(
            "flange",
            "left_ir",
            camera_to_flange_rotation,
            np.asarray(camera_to_flange_translation).reshape(3),
        )
    except (ValueError, TypeError, getattr(cv2, "error", RuntimeError)) as exc:
        raise HandEyeSolveError(f"OpenCV hand-eye solve failed: {exc}") from exc
    if not np.isfinite(initial.matrix).all():
        raise HandEyeSolveError("OpenCV hand-eye solve returned non-finite values")

    _, initial_translation, initial_rotation, _ = _loop_metrics(sample_list, initial)
    use_refinement = config.enable_bundle_adjustment if refine is None else refine
    if use_refinement:
        if intrinsics is None:
            raise HandEyeSolveError("left-IR intrinsics are required for LM bundle adjustment")
        flange_t_left_ir, optimized_base_t_target, optimization = _refine_bundle_adjustment(
            sample_list, initial, intrinsics, cv2
        )
        method_label = f"{display_name} + LM bundle adjustment"
    else:
        flange_t_left_ir = initial
        optimized_base_t_target = None
        optimization = HandEyeBundleAdjustment(
            False, True, None, None, None, None, "bundle adjustment disabled"
        )
        method_label = display_name

    fused_base_t_target, translation_rmse, rotation_rmse, rotation_max = _loop_metrics(
        sample_list, flange_t_left_ir
    )
    base_t_target = optimized_base_t_target or fused_base_t_target
    if translation_rmse > config.maximum_translation_rmse_m:
        raise HandEyeSolveError(
            f"translation RMSE {translation_rmse:.6f} m exceeds "
            f"{config.maximum_translation_rmse_m:.6f} m"
        )
    if rotation_rmse > config.maximum_rotation_rmse_deg:
        raise HandEyeSolveError(
            f"rotation RMSE {rotation_rmse:.3f} deg exceeds "
            f"{config.maximum_rotation_rmse_deg:.3f} deg"
        )
    return HandEyeSolution(
        flange_t_left_ir,
        base_t_target,
        method_label,
        len(sample_list),
        translation_rmse,
        rotation_rmse,
        rotation_max,
        observability,
        optimization,
        initial_translation,
        initial_rotation,
    )


def _pose(item: dict[str, Any], key: str, parent: str, child: str) -> PoseSE3:
    return PoseSE3(parent, child, item[key])


def read_hand_eye_samples(path: str | Path) -> tuple[HandEyeSample, ...]:
    """Read schema-2 flange samples, with explicit migration of legacy TCP samples."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sample root must be a mapping")
        schema = int(payload["schema_version"])
        if schema not in (1, HAND_EYE_SAMPLE_SCHEMA_VERSION):
            raise ValueError(f"unsupported schema {schema}")
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list):
            raise TypeError("samples must be a list")
        flange_t_tcp = load_es68_flange_t_tcp()
        samples: list[HandEyeSample] = []
        for item in raw_samples:
            detection = item.get("detection")
            if schema == 1:
                base_t_tcp = _pose(item, "base_T_tcp", "base", "tcp")
                base_t_flange = base_t_tcp.compose(flange_t_tcp.inverse())
            else:
                base_t_flange = _pose(item, "base_T_flange", "base", "flange")
                base_t_tcp = (
                    _pose(item, "base_T_tcp_observed", "base", "tcp")
                    if item.get("base_T_tcp_observed") is not None
                    else None
                )
            samples.append(
                HandEyeSample(
                    sample_id=str(item["sample_id"]),
                    base_t_flange=base_t_flange,
                    left_ir_t_target=_pose(item, "left_ir_T_target", "left_ir", "target"),
                    source_session=(
                        str(item["source_session"])
                        if item.get("source_session") is not None
                        else None
                    ),
                    charuco_corner_count=(
                        int(detection["charuco_corner_count"]) if detection is not None else None
                    ),
                    reprojection_rmse_px=(
                        float(detection["reprojection_rmse_px"]) if detection is not None else None
                    ),
                    pose_ambiguity_ratio=(
                        float(detection["pose_ambiguity_ratio"])
                        if detection is not None
                        and detection.get("pose_ambiguity_ratio") is not None
                        else None
                    ),
                    joint_positions_rad=item.get("joint_positions_rad"),
                    base_t_tcp_observed=base_t_tcp,
                    charuco_ids=(detection.get("charuco_ids") if detection else None),
                    image_points_px=(detection.get("image_points_px") if detection else None),
                    object_points_m=(detection.get("object_points_m") if detection else None),
                    frame_number=(
                        int(item["frame_number"]) if item.get("frame_number") is not None else None
                    ),
                    bracket_ms=(
                        float(item["capture"]["bracket_ms"])
                        if item.get("capture") is not None
                        else None
                    ),
                    selected_robot_state_offset_ms=(
                        float(item["capture"]["selected_robot_state_offset_ms"])
                        if item.get("capture") is not None
                        and item["capture"].get("selected_robot_state_offset_ms") is not None
                        else None
                    ),
                    controller_time_s=(
                        float(item["controller_time_s"])
                        if item.get("controller_time_s") is not None
                        else None
                    ),
                    robot_mode=(
                        str(item["robot_mode"]) if item.get("robot_mode") is not None else None
                    ),
                    safety_status=(
                        str(item["safety_status"])
                        if item.get("safety_status") is not None
                        else None
                    ),
                    fk_tcp_translation_error_m=(
                        float(item["fk_tcp_validation"]["translation_error_m"])
                        if item.get("fk_tcp_validation") is not None
                        else None
                    ),
                    fk_tcp_rotation_error_deg=(
                        float(item["fk_tcp_validation"]["rotation_error_deg"])
                        if item.get("fk_tcp_validation") is not None
                        else None
                    ),
                )
            )
        return tuple(samples)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise HandEyeSolveError(f"invalid hand-eye sample set {source}: {exc}") from exc


def write_hand_eye_calibration(
    output: str | Path,
    solution: HandEyeSolution,
    *,
    intrinsics: CameraIntrinsics | None = None,
    stereo_calibration_path: str | Path | None = None,
    target_path: str | Path | None = None,
) -> Path:
    """Write the flange-primary schema consumed by the runtime loader."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Hand-eye calibration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flange_t_tcp = load_es68_flange_t_tcp()
    tcp_t_left_ir = flange_t_tcp.inverse().compose(solution.flange_t_left_ir)
    optimization = solution.bundle_adjustment

    def source_file(path: str | Path | None) -> dict[str, str] | None:
        if path is None:
            return None
        resolved = Path(path).resolve()
        return {
            "path": str(resolved),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }

    payload = {
        "schema_version": 2,
        "calibration_type": "es68_d435i_left_ir_eye_in_hand",
        "robot_model": "es68",
        "holorobot_provenance": robot_stack_provenance(),
        "camera_stream": "infrared/1",
        "parent_frame": "flange",
        "child_frame": "left_ir",
        "method": solution.method,
        "matrix": solution.flange_t_left_ir.matrix.tolist(),
        "flange_T_left_ir": solution.flange_t_left_ir.matrix.tolist(),
        "derived_runtime": {
            "flange_T_tcp_validation": flange_t_tcp.matrix.tolist(),
            "tcp_T_left_ir": tcp_t_left_ir.matrix.tolist(),
            "note": (
                "RTSI TCP is validation/runtime compatibility only; solve used FK flange poses."
            ),
        },
        "quality": {
            "sample_count": solution.sample_count,
            "translation_rmse_m": solution.translation_rmse_m,
            "rotation_rmse_deg": solution.rotation_rmse_deg,
            "rotation_max_deg": solution.rotation_max_deg,
            "rotation_span_deg": solution.observability.rotation_span_deg,
            "translation_span_m": solution.observability.translation_span_m,
            "rotation_axis_diversity": solution.observability.rotation_axis_diversity,
            "initial_translation_rmse_m": solution.initial_translation_rmse_m,
            "initial_rotation_rmse_deg": solution.initial_rotation_rmse_deg,
        },
        "bundle_adjustment": {
            "enabled": optimization.enabled,
            "success": optimization.success,
            "initial_rmse_px": optimization.initial_rmse_px,
            "final_rmse_px": optimization.final_rmse_px,
            "mean_error_px": optimization.mean_error_px,
            "maximum_error_px": optimization.maximum_error_px,
            "message": optimization.message,
        },
        "fixed_target": {"base_T_target": solution.base_t_target.matrix.tolist()},
        "input_provenance": {
            "stereo_calibration": source_file(stereo_calibration_path),
            "charuco_target": source_file(target_path),
            "left_ir_intrinsics": (
                {
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "camera_matrix": _camera_matrix(intrinsics).tolist(),
                    "distortion_model": intrinsics.distortion_model,
                    "distortion_coefficients": list(intrinsics.distortion_coefficients),
                    "factory_intrinsics_used": False,
                }
                if intrinsics is not None
                else None
            ),
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_hand_eye_samples(
    output: str | Path,
    samples: Sequence[HandEyeSample],
    rejected: Sequence[HandEyeSampleRejection] = (),
) -> Path:
    """Write raw solver inputs without overwriting an earlier experiment."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Hand-eye sample set already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def detection_payload(sample: HandEyeSample) -> dict[str, Any] | None:
        if sample.charuco_corner_count is None:
            return None
        return {
            "charuco_corner_count": sample.charuco_corner_count,
            "reprojection_rmse_px": sample.reprojection_rmse_px,
            "pose_ambiguity_ratio": sample.pose_ambiguity_ratio,
            "charuco_ids": (
                sample.charuco_ids.tolist() if sample.charuco_ids is not None else None
            ),
            "image_points_px": (
                sample.image_points_px.tolist() if sample.image_points_px is not None else None
            ),
            "object_points_m": (
                sample.object_points_m.tolist() if sample.object_points_m is not None else None
            ),
        }

    payload = {
        "schema_version": HAND_EYE_SAMPLE_SCHEMA_VERSION,
        "calibration_type": "es68_d435i_left_ir_eye_in_hand_samples",
        "robot_model": "es68",
        "holorobot_provenance": robot_stack_provenance(),
        "robot_pose_source": "HoloRobot calibrated ES68 FK (709 poses)",
        "camera_stream": "infrared/1",
        "samples": [
            {
                "sample_id": sample.sample_id,
                "base_T_flange": sample.base_t_flange.matrix.tolist(),
                "base_T_tcp_observed": (
                    sample.base_t_tcp_observed.matrix.tolist()
                    if sample.base_t_tcp_observed is not None
                    else None
                ),
                "joint_positions_rad": (
                    sample.joint_positions_rad.tolist()
                    if sample.joint_positions_rad is not None
                    else None
                ),
                "left_ir_T_target": sample.left_ir_t_target.matrix.tolist(),
                "source_session": sample.source_session,
                "frame_number": sample.frame_number,
                "capture": (
                    {
                        "bracket_ms": sample.bracket_ms,
                        "selected_robot_state_offset_ms": (sample.selected_robot_state_offset_ms),
                    }
                    if sample.bracket_ms is not None
                    else None
                ),
                "controller_time_s": sample.controller_time_s,
                "robot_mode": sample.robot_mode,
                "safety_status": sample.safety_status,
                "fk_tcp_validation": (
                    {
                        "translation_error_m": sample.fk_tcp_translation_error_m,
                        "rotation_error_deg": sample.fk_tcp_rotation_error_deg,
                    }
                    if sample.fk_tcp_translation_error_m is not None
                    else None
                ),
                "detection": detection_payload(sample),
            }
            for sample in samples
        ],
        "rejected": [
            {"sample_id": rejection.sample_id, "reason": rejection.reason} for rejection in rejected
        ],
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
