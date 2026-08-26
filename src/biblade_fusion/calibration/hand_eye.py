"""Validated eye-in-hand calibration artifact loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig

SCHEMA_VERSION = 1


class HandEyeCalibrationError(ValueError):
    """The eye-in-hand calibration is missing, invalid, or below quality thresholds."""


@dataclass(frozen=True, slots=True)
class HandEyeCalibration:
    tcp_t_left_ir: PoseSE3
    method: str
    sample_count: int | None
    translation_rmse_m: float | None
    rotation_rmse_deg: float | None
    source_path: Path
    rotation_span_deg: float | None = None
    translation_span_m: float | None = None
    rotation_axis_diversity: float | None = None

    def __post_init__(self) -> None:
        if self.tcp_t_left_ir.parent_frame != "tcp":
            raise ValueError("Hand-eye parent frame must be tcp")
        if self.tcp_t_left_ir.child_frame != "left_ir":
            raise ValueError("Hand-eye child frame must be left_ir")
        if not self.method:
            raise ValueError("Hand-eye calibration method must be non-empty")
        if self.sample_count is not None and self.sample_count < 3:
            raise ValueError("Hand-eye calibration requires at least three samples")
        if self.translation_rmse_m is not None and self.translation_rmse_m < 0.0:
            raise ValueError("Hand-eye translation RMSE must be non-negative")
        if self.rotation_rmse_deg is not None and self.rotation_rmse_deg < 0.0:
            raise ValueError("Hand-eye rotation RMSE must be non-negative")
        if self.rotation_span_deg is not None and self.rotation_span_deg < 0.0:
            raise ValueError("Hand-eye rotation span must be non-negative")
        if self.translation_span_m is not None and self.translation_span_m < 0.0:
            raise ValueError("Hand-eye translation span must be non-negative")
        if (
            self.rotation_axis_diversity is not None
            and not 0.0 <= self.rotation_axis_diversity <= 1.0
        ):
            raise ValueError("Hand-eye rotation-axis diversity must be in [0, 1]")


def load_hand_eye_calibration(config: HandEyeConfig) -> HandEyeCalibration:
    """Load and quality-gate a ``tcp_T_left_ir`` calibration artifact."""

    if config.calibration_path is None:
        raise HandEyeCalibrationError("Hand-eye calibration path is not configured")
    path = config.calibration_path
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HandEyeCalibrationError(f"Cannot read hand-eye calibration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandEyeCalibrationError("Hand-eye calibration root must be a mapping")

    try:
        schema_version = int(payload["schema_version"])
        parent_frame = str(payload["parent_frame"])
        child_frame = str(payload["child_frame"])
        method = str(payload["method"])
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        quality = payload.get("quality")
        if quality is not None and not isinstance(quality, dict):
            raise TypeError("quality must be a mapping")
        sample_count = int(quality["sample_count"]) if quality is not None else None
        translation_rmse_m = float(quality["translation_rmse_m"]) if quality is not None else None
        rotation_rmse_deg = float(quality["rotation_rmse_deg"]) if quality is not None else None
        rotation_span_deg = (
            float(quality["rotation_span_deg"])
            if quality is not None and quality.get("rotation_span_deg") is not None
            else None
        )
        translation_span_m = (
            float(quality["translation_span_m"])
            if quality is not None and quality.get("translation_span_m") is not None
            else None
        )
        rotation_axis_diversity = (
            float(quality["rotation_axis_diversity"])
            if quality is not None and quality.get("rotation_axis_diversity") is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HandEyeCalibrationError(f"Hand-eye calibration fields are invalid: {exc}") from exc

    if schema_version != SCHEMA_VERSION:
        raise HandEyeCalibrationError(f"Unsupported hand-eye schema {schema_version}")
    try:
        calibration = HandEyeCalibration(
            tcp_t_left_ir=PoseSE3(parent_frame, child_frame, matrix),
            method=method,
            sample_count=sample_count,
            translation_rmse_m=translation_rmse_m,
            rotation_rmse_deg=rotation_rmse_deg,
            source_path=path.resolve(),
            rotation_span_deg=rotation_span_deg,
            translation_span_m=translation_span_m,
            rotation_axis_diversity=rotation_axis_diversity,
        )
    except ValueError as exc:
        raise HandEyeCalibrationError(str(exc)) from exc

    metrics = (
        calibration.sample_count,
        calibration.translation_rmse_m,
        calibration.rotation_rmse_deg,
    )
    if config.require_quality_metrics and any(value is None for value in metrics):
        raise HandEyeCalibrationError("Hand-eye quality metrics are required")
    observability_metrics = (
        calibration.rotation_span_deg,
        calibration.translation_span_m,
        calibration.rotation_axis_diversity,
    )
    if config.require_observability_metrics and any(
        value is None for value in observability_metrics
    ):
        raise HandEyeCalibrationError("Hand-eye observability metrics are required")
    if calibration.sample_count is not None and calibration.sample_count < config.minimum_samples:
        raise HandEyeCalibrationError(
            f"Hand-eye sample count {calibration.sample_count} is below {config.minimum_samples}"
        )
    if (
        calibration.translation_rmse_m is not None
        and calibration.translation_rmse_m > config.maximum_translation_rmse_m
    ):
        raise HandEyeCalibrationError(
            f"Hand-eye translation RMSE {calibration.translation_rmse_m:.6f} m exceeds "
            f"{config.maximum_translation_rmse_m:.6f} m"
        )
    if (
        calibration.rotation_rmse_deg is not None
        and calibration.rotation_rmse_deg > config.maximum_rotation_rmse_deg
    ):
        raise HandEyeCalibrationError(
            f"Hand-eye rotation RMSE {calibration.rotation_rmse_deg:.3f} deg exceeds "
            f"{config.maximum_rotation_rmse_deg:.3f} deg"
        )
    if (
        calibration.rotation_span_deg is not None
        and calibration.rotation_span_deg < config.minimum_rotation_span_deg
    ):
        raise HandEyeCalibrationError(
            f"Hand-eye rotation span {calibration.rotation_span_deg:.3f} deg is below "
            f"{config.minimum_rotation_span_deg:.3f} deg"
        )
    if (
        calibration.translation_span_m is not None
        and calibration.translation_span_m < config.minimum_translation_span_m
    ):
        raise HandEyeCalibrationError(
            f"Hand-eye translation span {calibration.translation_span_m:.6f} m is below "
            f"{config.minimum_translation_span_m:.6f} m"
        )
    if (
        calibration.rotation_axis_diversity is not None
        and calibration.rotation_axis_diversity < config.minimum_rotation_axis_diversity
    ):
        raise HandEyeCalibrationError(
            "Hand-eye rotation-axis diversity "
            f"{calibration.rotation_axis_diversity:.4f} is below "
            f"{config.minimum_rotation_axis_diversity:.4f}"
        )
    return calibration
