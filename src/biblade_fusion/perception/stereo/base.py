"""Backend-neutral stereo inference results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.devices.depth_camera.base import StereoCalibrationSnapshot
from biblade_fusion.perception.stereo.rectification import RectifiedStereoCalibration


@dataclass(frozen=True, slots=True)
class StereoResult:
    """A disparity estimate in full-resolution left-image pixel units."""

    disparity_px: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    confidence: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        disparity = np.array(self.disparity_px, dtype=np.float32, copy=True)
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        if disparity.ndim != 2:
            raise ValueError("Disparity must be a two-dimensional array")
        if valid.shape != disparity.shape:
            raise ValueError("Stereo valid mask must match disparity shape")

        confidence: NDArray[np.float32] | None = None
        if self.confidence is not None:
            confidence = np.array(self.confidence, dtype=np.float32, copy=True)
            if confidence.shape != disparity.shape:
                raise ValueError("Stereo confidence must match disparity shape")
            confidence.setflags(write=False)

        disparity.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "disparity_px", disparity)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "confidence", confidence)

    def depth_m(
        self, calibration: StereoCalibrationSnapshot | RectifiedStereoCalibration
    ) -> NDArray[np.float32]:
        return disparity_to_depth_m(self.disparity_px, calibration, self.valid_mask)


def disparity_to_depth_m(
    disparity_px: NDArray[np.floating[Any]],
    calibration: StereoCalibrationSnapshot | RectifiedStereoCalibration,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Convert rectified disparity into metric axial depth.

    Invalid values are represented as ``NaN`` rather than zero to avoid treating
    missing depth as a physical point at the camera origin.
    """

    disparity = np.asarray(disparity_px, dtype=np.float64)
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a two-dimensional array")
    valid = np.isfinite(disparity) & (disparity > 0.0)
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=np.bool_)
        if supplied_mask.shape != disparity.shape:
            raise ValueError("Valid mask must match disparity shape")
        valid &= supplied_mask

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (calibration.left.fx * calibration.baseline_m / disparity[valid]).astype(
        np.float32
    )
    depth.setflags(write=False)
    return depth


def constrain_to_rectified_valid_regions(
    result: StereoResult,
    calibration: RectifiedStereoCalibration,
) -> StereoResult:
    """Reject pixels outside either rectified image's calibrated valid region."""

    height, width = result.disparity_px.shape
    expected_shape = (calibration.left.height, calibration.left.width)
    if (height, width) != expected_shape:
        raise ValueError(
            f"Disparity shape {(height, width)} does not match calibration {expected_shape}"
        )
    vertical, horizontal = np.indices((height, width), dtype=np.float32)
    right_horizontal = horizontal - result.disparity_px
    left_x, left_y, left_width, left_height = calibration.left_valid_roi
    right_x, right_y, right_width, right_height = calibration.right_valid_roi
    valid = result.valid_mask.copy()
    valid &= (
        (horizontal >= left_x)
        & (horizontal < left_x + left_width)
        & (vertical >= left_y)
        & (vertical < left_y + left_height)
    )
    valid &= (
        (right_horizontal >= right_x)
        & (right_horizontal < right_x + right_width)
        & (vertical >= right_y)
        & (vertical < right_y + right_height)
    )
    metadata = {
        **result.metadata,
        "rectified_valid_regions_applied": True,
        "left_valid_roi": list(calibration.left_valid_roi),
        "right_valid_roi": list(calibration.right_valid_roi),
    }
    return StereoResult(result.disparity_px, valid, result.confidence, metadata)


@runtime_checkable
class StereoBackend(Protocol):
    """A stereo backend consuming rectified left/right images."""

    def infer(
        self,
        left_rectified: NDArray[np.uint8],
        right_rectified: NDArray[np.uint8],
    ) -> StereoResult: ...
