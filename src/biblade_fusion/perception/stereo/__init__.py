"""Stereo inference contracts."""

from biblade_fusion.perception.stereo.base import (
    StereoBackend,
    StereoResult,
    disparity_to_depth_m,
)
from biblade_fusion.perception.stereo.foundation_stereo import (
    FoundationStereoBackend,
    FoundationStereoError,
    run_foundation_stereo_doctor,
)
from biblade_fusion.perception.stereo.rectification import (
    RectifiedStereoCalibration,
    RectifiedStereoFrame,
    StereoRectifier,
)

__all__ = [
    "StereoBackend",
    "FoundationStereoBackend",
    "FoundationStereoError",
    "StereoRectifier",
    "StereoResult",
    "RectifiedStereoCalibration",
    "RectifiedStereoFrame",
    "disparity_to_depth_m",
    "run_foundation_stereo_doctor",
]
