"""Depth-consistent removal of the eye-in-hand robot from safety mapping input."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class RobotSelfMaskConfig:
    """Tolerances for matching measured depth to rendered robot depth."""

    front_tolerance_m: float = 0.01
    back_tolerance_m: float = 0.02
    dilation_px: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite((self.front_tolerance_m, self.back_tolerance_m)).all():
            raise ValueError("Robot self-mask tolerances must be finite")
        if self.front_tolerance_m < 0.0 or self.back_tolerance_m < 0.0:
            raise ValueError("Robot self-mask tolerances must be non-negative")
        if self.dilation_px < 0:
            raise ValueError("Robot self-mask dilation must be non-negative")


@dataclass(frozen=True, slots=True)
class RobotSelfMaskReport:
    """Auditable counts for one self-filtered depth frame."""

    projected_robot_pixels: int
    measured_valid_pixels: int
    depth_matched_pixels: int
    masked_valid_pixels: int
    retained_valid_pixels: int
    front_tolerance_m: float
    back_tolerance_m: float
    dilation_px: int


@dataclass(frozen=True, slots=True)
class RobotSelfMaskResult:
    """Boolean masks; masked pixels must remain UNKNOWN during ray integration."""

    robot_mask: NDArray[np.bool_]
    integration_valid_mask: NDArray[np.bool_]
    report: RobotSelfMaskReport

    def __post_init__(self) -> None:
        robot_mask = np.array(self.robot_mask, dtype=np.bool_, copy=True)
        valid_mask = np.array(self.integration_valid_mask, dtype=np.bool_, copy=True)
        if robot_mask.shape != valid_mask.shape:
            raise ValueError("Robot and integration masks must have the same shape")
        robot_mask.setflags(write=False)
        valid_mask.setflags(write=False)
        object.__setattr__(self, "robot_mask", robot_mask)
        object.__setattr__(self, "integration_valid_mask", valid_mask)


def depth_consistent_robot_self_mask(
    measured_depth_m: ArrayLike,
    predicted_robot_depth_m: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    config: RobotSelfMaskConfig | None = None,
) -> RobotSelfMaskResult:
    """Remove rays that would reach the rendered robot surface or pass behind it.

    A closer measured surface is deliberately retained: it may be the unknown blade in
    front of a projected robot link.  A matching or farther measurement is masked: the
    latter can be a stereo dropout that would otherwise ray-clear through known robot
    geometry. Masked pixels are removed from integration rather than set to a long
    range, so neither the robot nor its occluded background clears safety occupancy.
    """

    settings = config or RobotSelfMaskConfig()
    measured = np.asarray(measured_depth_m, dtype=np.float64)
    predicted = np.asarray(predicted_robot_depth_m, dtype=np.float64)
    if measured.ndim != 2 or predicted.shape != measured.shape:
        raise ValueError("Measured and predicted robot depth must be matching 2-D images")
    measured_valid = np.isfinite(measured) & (measured > 0.0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=np.bool_)
        if supplied.shape != measured.shape:
            raise ValueError("Depth valid mask must match the depth image")
        measured_valid &= supplied
    projected = np.isfinite(predicted) & (predicted > 0.0)
    matched = (
        projected
        & measured_valid
        & (measured >= predicted - settings.front_tolerance_m)
        & (measured <= predicted + settings.back_tolerance_m)
    )
    robot_or_occluded = (
        projected
        & measured_valid
        & (measured >= predicted - settings.front_tolerance_m)
    )
    robot_mask = _dilate(robot_or_occluded, settings.dilation_px)
    masked_valid = robot_mask & measured_valid
    integration_valid = measured_valid & ~robot_mask
    return RobotSelfMaskResult(
        robot_mask,
        integration_valid,
        RobotSelfMaskReport(
            projected_robot_pixels=int(np.count_nonzero(projected)),
            measured_valid_pixels=int(np.count_nonzero(measured_valid)),
            depth_matched_pixels=int(np.count_nonzero(matched)),
            masked_valid_pixels=int(np.count_nonzero(masked_valid)),
            retained_valid_pixels=int(np.count_nonzero(integration_valid)),
            front_tolerance_m=settings.front_tolerance_m,
            back_tolerance_m=settings.back_tolerance_m,
            dilation_px=settings.dilation_px,
        ),
    )


def _dilate(mask: NDArray[np.bool_], pixels: int) -> NDArray[np.bool_]:
    if pixels == 0 or not np.any(mask):
        return np.array(mask, dtype=np.bool_, copy=True)
    size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
