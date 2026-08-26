"""Depth-camera-independent data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive")
        values = (self.fx, self.fy, self.cx, self.cy, *self.distortion_coefficients)
        if not np.isfinite(values).all():
            raise ValueError("Camera intrinsics must be finite")


@dataclass(frozen=True, slots=True)
class StereoCalibrationSnapshot:
    left: CameraIntrinsics
    right: CameraIntrinsics
    right_t_left: PoseSE3
    native_depth_scale_m: float | None

    def __post_init__(self) -> None:
        if self.right_t_left.parent_frame != "right_ir":
            raise ValueError("Stereo extrinsic parent frame must be right_ir")
        if self.right_t_left.child_frame != "left_ir":
            raise ValueError("Stereo extrinsic child frame must be left_ir")
        if self.native_depth_scale_m is not None and self.native_depth_scale_m <= 0:
            raise ValueError("Native depth scale must be positive")

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.right_t_left.translation_m))


@dataclass(frozen=True, slots=True)
class StereoFrame:
    monotonic_time_ns: int
    frame_number: int
    left_device_time_ms: float
    right_device_time_ms: float
    left_ir: NDArray[np.uint8]
    right_ir: NDArray[np.uint8]
    native_depth: NDArray[np.uint16] | None
    calibration: StereoCalibrationSnapshot

    def __post_init__(self) -> None:
        left = np.array(self.left_ir, dtype=np.uint8, copy=True)
        right = np.array(self.right_ir, dtype=np.uint8, copy=True)
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("Infrared images must be two-dimensional grayscale arrays")
        if left.shape != right.shape:
            raise ValueError("Left and right infrared image shapes must match")
        expected_shape = (self.calibration.left.height, self.calibration.left.width)
        if left.shape != expected_shape:
            raise ValueError(f"Infrared image shape {left.shape} does not match {expected_shape}")

        depth: NDArray[np.uint16] | None = None
        if self.native_depth is not None:
            depth = np.array(self.native_depth, dtype=np.uint16, copy=True)
            if depth.shape != left.shape:
                raise ValueError("Native depth shape must match infrared images")
            depth.setflags(write=False)

        if self.monotonic_time_ns < 0 or self.frame_number < 0:
            raise ValueError("Frame timestamps and number must be non-negative")
        left.setflags(write=False)
        right.setflags(write=False)
        object.__setattr__(self, "left_ir", left)
        object.__setattr__(self, "right_ir", right)
        object.__setattr__(self, "native_depth", depth)


@runtime_checkable
class StereoCamera(Protocol):
    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def capture(self) -> StereoFrame: ...

