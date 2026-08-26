"""Stereo inference contracts."""

from biblade_fusion.perception.stereo.base import (
    StereoBackend,
    StereoResult,
    disparity_to_depth_m,
)
from biblade_fusion.perception.stereo.foundation_stereo import (
    run_foundation_stereo_doctor,
)
from biblade_fusion.perception.stereo.rectification import (
    RectifiedStereoCalibration,
    RectifiedStereoFrame,
    StereoRectifier,
)

__all__ = [
    "StereoBackend",
    "StereoRectifier",
    "StereoResult",
    "RectifiedStereoCalibration",
    "RectifiedStereoFrame",
    "disparity_to_depth_m",
    "run_foundation_stereo_doctor",
]
