"""Stereo inference contracts."""

from biblade_fusion.perception.stereo.base import (
    StereoBackend,
    StereoResult,
    disparity_to_depth_m,
)
from biblade_fusion.perception.stereo.foundation_stereo import (
    run_foundation_stereo_doctor,
)

__all__ = [
    "StereoBackend",
    "StereoResult",
    "disparity_to_depth_m",
    "run_foundation_stereo_doctor",
]
