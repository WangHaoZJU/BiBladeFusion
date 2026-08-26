"""Backend-neutral stereo inference results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.devices.depth_camera.base import StereoCalibrationSnapshot


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

    def depth_m(self, calibration: StereoCalibrationSnapshot) -> NDArray[np.float32]:
        return disparity_to_depth_m(self.disparity_px, calibration, self.valid_mask)


def disparity_to_depth_m(
    disparity_px: NDArray[np.floating[Any]],
    calibration: StereoCalibrationSnapshot,
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


@runtime_checkable
class StereoBackend(Protocol):
    """A stereo backend consuming rectified left/right images."""

    def infer(
        self,
        left_rectified: NDArray[np.uint8],
        right_rectified: NDArray[np.uint8],
    ) -> StereoResult: ...
