"""Offline eye-in-hand solving with fixed-target closure quality metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig

HAND_EYE_SAMPLE_SCHEMA_VERSION = 1


class HandEyeSolveError(ValueError):
    """Hand-eye samples are invalid, degenerate, or fail configured quality limits."""


@dataclass(frozen=True, slots=True)
class HandEyeSample:
    sample_id: str
    base_t_tcp: PoseSE3
    left_ir_t_target: PoseSE3

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("Hand-eye sample ID must be non-empty")
        if (self.base_t_tcp.parent_frame, self.base_t_tcp.child_frame) != ("base", "tcp"):
            raise ValueError("Hand-eye sample requires base_T_tcp")
        if (self.left_ir_t_target.parent_frame, self.left_ir_t_target.child_frame) != (
            "left_ir",
            "target",
        ):
            raise ValueError("Hand-eye sample requires left_ir_T_target")


@dataclass(frozen=True, slots=True)
class HandEyeObservability:
    rotation_span_deg: float
    translation_span_m: float
    rotation_axis_diversity: float


@dataclass(frozen=True, slots=True)
class HandEyeSolution:
    tcp_t_left_ir: PoseSE3
    base_t_target: PoseSE3
    method: str
    sample_count: int
    translation_rmse_m: float
    rotation_rmse_deg: float
    observability: HandEyeObservability


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pairwise_observability(samples: Sequence[HandEyeSample], cv2: Any) -> HandEyeObservability:
    rotations: list[float] = []
    translations: list[float] = []
    axes: list[np.ndarray] = []
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            relative = left.base_t_tcp.inverse().compose(right.base_t_tcp)
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


def solve_hand_eye(
    samples: Sequence[HandEyeSample],
    config: HandEyeConfig,
    *,
    method: str = "park",
) -> HandEyeSolution:
    """Solve ``tcp_T_left_ir`` using OpenCV and quality-gate fixed-target closure."""

    sample_list = tuple(samples)
    if len(sample_list) < config.minimum_samples:
        raise HandEyeSolveError(
            f"hand-eye sample count {len(sample_list)} is below {config.minimum_samples}"
        )
    if len({sample.sample_id for sample in sample_list}) != len(sample_list):
        raise HandEyeSolveError("hand-eye sample IDs must be unique")
    cv2 = import_module("cv2")
    method_name = method.lower().replace("_", "-")
    methods = {
        "park": (cv2.CALIB_HAND_EYE_PARK, "OpenCV Park-Martin"),
        "tsai": (cv2.CALIB_HAND_EYE_TSAI, "OpenCV Tsai-Lenz"),
        "horaud": (cv2.CALIB_HAND_EYE_HORAUD, "OpenCV Horaud"),
        "andreff": (cv2.CALIB_HAND_EYE_ANDREFF, "OpenCV Andreff"),
        "daniilidis": (cv2.CALIB_HAND_EYE_DANIILIDIS, "OpenCV Daniilidis"),
    }
    if method_name not in methods:
        raise HandEyeSolveError(f"unsupported hand-eye method: {method}")

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
        camera_to_tcp_rotation, camera_to_tcp_translation = cv2.calibrateHandEye(
            [sample.base_t_tcp.rotation for sample in sample_list],
            [sample.base_t_tcp.translation_m for sample in sample_list],
            [sample.left_ir_t_target.rotation for sample in sample_list],
            [sample.left_ir_t_target.translation_m for sample in sample_list],
            method=algorithm,
        )
        tcp_t_left_ir = PoseSE3.from_rotation_translation(
            "tcp",
            "left_ir",
            camera_to_tcp_rotation,
            np.asarray(camera_to_tcp_translation).reshape(3),
        )
    except (ValueError, TypeError, getattr(cv2, "error", RuntimeError)) as exc:
        raise HandEyeSolveError(f"OpenCV hand-eye solve failed: {exc}") from exc

    target_poses = [
        sample.base_t_tcp.compose(tcp_t_left_ir).compose(sample.left_ir_t_target)
        for sample in sample_list
    ]
    base_t_target = _mean_pose(target_poses)
    translation_errors = np.array(
        [np.linalg.norm(pose.translation_m - base_t_target.translation_m) for pose in target_poses]
    )
    rotation_errors_deg = np.array(
        [_rotation_angle_deg(base_t_target.rotation.T @ pose.rotation) for pose in target_poses]
    )
    translation_rmse_m = float(np.sqrt(np.mean(translation_errors**2)))
    rotation_rmse_deg = float(np.sqrt(np.mean(rotation_errors_deg**2)))
    if translation_rmse_m > config.maximum_translation_rmse_m:
        raise HandEyeSolveError(
            f"translation RMSE {translation_rmse_m:.6f} m exceeds "
            f"{config.maximum_translation_rmse_m:.6f} m"
        )
    if rotation_rmse_deg > config.maximum_rotation_rmse_deg:
        raise HandEyeSolveError(
            f"rotation RMSE {rotation_rmse_deg:.3f} deg exceeds "
            f"{config.maximum_rotation_rmse_deg:.3f} deg"
        )
    return HandEyeSolution(
        tcp_t_left_ir,
        base_t_target,
        display_name,
        len(sample_list),
        translation_rmse_m,
        rotation_rmse_deg,
        observability,
    )


def read_hand_eye_samples(path: str | Path) -> tuple[HandEyeSample, ...]:
    """Read explicit frame-aware sample matrices from YAML or JSON."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("sample root must be a mapping")
        if int(payload["schema_version"]) != HAND_EYE_SAMPLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list):
            raise TypeError("samples must be a list")
        return tuple(
            HandEyeSample(
                str(item["sample_id"]),
                PoseSE3("base", "tcp", item["base_T_tcp"]),
                PoseSE3("left_ir", "target", item["left_ir_T_target"]),
            )
            for item in raw_samples
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise HandEyeSolveError(f"invalid hand-eye sample set {source}: {exc}") from exc


def write_hand_eye_calibration(output: str | Path, solution: HandEyeSolution) -> Path:
    """Write the schema consumed by :func:`load_hand_eye_calibration`."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Hand-eye calibration already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "parent_frame": "tcp",
        "child_frame": "left_ir",
        "method": solution.method,
        "matrix": solution.tcp_t_left_ir.matrix.tolist(),
        "quality": {
            "sample_count": solution.sample_count,
            "translation_rmse_m": solution.translation_rmse_m,
            "rotation_rmse_deg": solution.rotation_rmse_deg,
            "rotation_span_deg": solution.observability.rotation_span_deg,
            "translation_span_m": solution.observability.translation_span_m,
            "rotation_axis_diversity": solution.observability.rotation_axis_diversity,
        },
        "fixed_target": {"base_T_target": solution.base_t_target.matrix.tolist()},
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


def write_hand_eye_samples(output: str | Path, samples: Sequence[HandEyeSample]) -> Path:
    """Write auditable solver inputs without overwriting an earlier sample set."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Hand-eye sample set already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HAND_EYE_SAMPLE_SCHEMA_VERSION,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "base_T_tcp": sample.base_t_tcp.matrix.tolist(),
                "left_ir_T_target": sample.left_ir_t_target.matrix.tolist(),
            }
            for sample in samples
        ],
    }
    # JSON is valid YAML and avoids implementation-specific YAML matrix tags.
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
