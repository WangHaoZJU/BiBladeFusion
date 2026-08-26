"""Calibrated D435i stereo rectification with explicit frame transforms."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import StereoRectificationConfig
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)


@dataclass(frozen=True, slots=True)
class RectifiedStereoCalibration:
    left: CameraIntrinsics
    right: CameraIntrinsics
    right_rectified_t_left_rectified: PoseSE3
    left_rectified_t_left_ir: PoseSE3
    right_rectified_t_right_ir: PoseSE3
    disparity_to_depth_q: NDArray[np.float64]
    left_valid_roi: tuple[int, int, int, int]
    right_valid_roi: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        expected_frames = (
            (self.right_rectified_t_left_rectified, "right_rectified", "left_rectified"),
            (self.left_rectified_t_left_ir, "left_rectified", "left_ir"),
            (self.right_rectified_t_right_ir, "right_rectified", "right_ir"),
        )
        for pose, parent, child in expected_frames:
            if pose.parent_frame != parent or pose.child_frame != child:
                raise ValueError(f"Expected {parent}_T_{child} rectification transform")
        q = np.array(self.disparity_to_depth_q, dtype=np.float64, copy=True)
        if q.shape != (4, 4) or not np.isfinite(q).all():
            raise ValueError("Disparity reprojection Q must be a finite 4x4 matrix")
        q.setflags(write=False)
        object.__setattr__(self, "disparity_to_depth_q", q)

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.right_rectified_t_left_rectified.translation_m))


@dataclass(frozen=True, slots=True)
class RectifiedStereoFrame:
    left_ir: NDArray[np.uint8]
    right_ir: NDArray[np.uint8]
    calibration: RectifiedStereoCalibration
    source_monotonic_time_ns: int
    source_frame_number: int

    def __post_init__(self) -> None:
        left = np.array(self.left_ir, dtype=np.uint8, copy=True)
        right = np.array(self.right_ir, dtype=np.uint8, copy=True)
        expected_shape = (self.calibration.left.height, self.calibration.left.width)
        if left.shape != expected_shape or right.shape != expected_shape:
            raise ValueError("Rectified images must match rectified calibration dimensions")
        left.setflags(write=False)
        right.setflags(write=False)
        object.__setattr__(self, "left_ir", left)
        object.__setattr__(self, "right_ir", right)


def _camera_matrix(intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rectified_intrinsics(
    projection: NDArray[np.float64], width: int, height: int
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width,
        height,
        float(projection[0, 0]),
        float(projection[1, 1]),
        float(projection[0, 2]),
        float(projection[1, 2]),
        "none",
        (),
    )


class StereoRectifier:
    """Precompute rectification maps for one immutable calibration snapshot."""

    def __init__(
        self,
        calibration: StereoCalibrationSnapshot,
        config: StereoRectificationConfig,
        cv2_module: ModuleType | Any | None = None,
    ) -> None:
        cv2 = cv2_module or import_module("cv2")
        self._cv2 = cv2
        self._source = calibration
        image_size = (calibration.left.width, calibration.left.height)
        if image_size != (calibration.right.width, calibration.right.height):
            raise ValueError("Stereo rectification requires equal left/right image dimensions")
        left_matrix = _camera_matrix(calibration.left)
        right_matrix = _camera_matrix(calibration.right)
        left_distortion = np.asarray(calibration.left.distortion_coefficients, dtype=np.float64)
        right_distortion = np.asarray(calibration.right.distortion_coefficients, dtype=np.float64)
        flags = cv2.CALIB_ZERO_DISPARITY if config.zero_disparity else 0
        (
            left_rotation,
            right_rotation,
            left_projection,
            right_projection,
            q,
            left_roi,
            right_roi,
        ) = cv2.stereoRectify(
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            image_size,
            calibration.right_t_left.rotation,
            calibration.right_t_left.translation_m,
            flags=flags,
            alpha=config.alpha,
        )
        map_type = cv2.CV_32FC1
        self._left_maps = cv2.initUndistortRectifyMap(
            left_matrix,
            left_distortion,
            left_rotation,
            left_projection[:, :3],
            image_size,
            map_type,
        )
        self._right_maps = cv2.initUndistortRectifyMap(
            right_matrix,
            right_distortion,
            right_rotation,
            right_projection[:, :3],
            image_size,
            map_type,
        )
        self._interpolation = {
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
        }[config.interpolation]

        left_rectified_t_left_ir = PoseSE3.from_rotation_translation(
            "left_rectified", "left_ir", left_rotation, np.zeros(3)
        )
        right_rectified_t_right_ir = PoseSE3.from_rotation_translation(
            "right_rectified", "right_ir", right_rotation, np.zeros(3)
        )
        right_rectified_t_left_rectified = right_rectified_t_right_ir.compose(
            calibration.right_t_left
        ).compose(left_rectified_t_left_ir.inverse())
        self.calibration = RectifiedStereoCalibration(
            left=_rectified_intrinsics(left_projection, *image_size),
            right=_rectified_intrinsics(right_projection, *image_size),
            right_rectified_t_left_rectified=right_rectified_t_left_rectified,
            left_rectified_t_left_ir=left_rectified_t_left_ir,
            right_rectified_t_right_ir=right_rectified_t_right_ir,
            disparity_to_depth_q=q,
            left_valid_roi=tuple(int(value) for value in left_roi),
            right_valid_roi=tuple(int(value) for value in right_roi),
        )

    def rectify(self, frame: StereoFrame) -> RectifiedStereoFrame:
        """Rectify one synchronized raw infrared pair using precomputed maps."""

        if frame.calibration is not self._source:
            raise ValueError("Stereo frame calibration does not match this rectifier")
        left = self._cv2.remap(
            frame.left_ir,
            *self._left_maps,
            self._interpolation,
            borderMode=self._cv2.BORDER_CONSTANT,
        )
        right = self._cv2.remap(
            frame.right_ir,
            *self._right_maps,
            self._interpolation,
            borderMode=self._cv2.BORDER_CONSTANT,
        )
        return RectifiedStereoFrame(
            left,
            right,
            self.calibration,
            frame.monotonic_time_ns,
            frame.frame_number,
        )
