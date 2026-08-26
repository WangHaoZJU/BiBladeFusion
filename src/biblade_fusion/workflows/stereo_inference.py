"""Offline calibrated stereo inference for one synchronized observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.core.settings import StereoRectificationConfig
from biblade_fusion.perception.stereo import (
    RectifiedStereoFrame,
    StereoBackend,
    StereoRectifier,
    StereoResult,
    constrain_to_rectified_valid_regions,
)


@dataclass(frozen=True, slots=True)
class StereoInferenceObservation:
    source_view_id: str
    source_sequence_index: int
    rectified: RectifiedStereoFrame
    result: StereoResult
    depth_m: NDArray[np.float32]

    def __post_init__(self) -> None:
        depth = np.array(self.depth_m, dtype=np.float32, copy=True)
        if depth.shape != self.result.disparity_px.shape:
            raise ValueError("Stereo depth and disparity shapes must match")
        if self.rectified.left_ir.shape != depth.shape:
            raise ValueError("Rectified images and inferred depth shapes must match")
        if np.isfinite(depth[~self.result.valid_mask]).any():
            raise ValueError("Invalid stereo pixels must have NaN depth")
        if np.any(depth[self.result.valid_mask] <= 0.0):
            raise ValueError("Valid stereo depth must be positive")
        depth.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)


def infer_rectified_stereo(
    bundle: SynchronizedFrameBundle,
    backend: StereoBackend,
    rectification_config: StereoRectificationConfig,
) -> StereoInferenceObservation:
    """Rectify, infer, calibrate, and mask one already captured stereo pair."""

    rectified = StereoRectifier(bundle.stereo.calibration, rectification_config).rectify(
        bundle.stereo
    )
    raw_result = backend.infer(rectified.left_ir, rectified.right_ir)
    result = constrain_to_rectified_valid_regions(raw_result, rectified.calibration)
    depth_m = result.depth_m(rectified.calibration)
    return StereoInferenceObservation(
        bundle.view_id,
        bundle.sequence_index,
        rectified,
        result,
        depth_m,
    )
