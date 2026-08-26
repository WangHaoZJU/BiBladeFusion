"""Intel RealSense D435i raw stereo acquisition."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from types import ModuleType
from typing import Any

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import RealSenseConfig
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.depth_camera.errors import (
    DepthCameraConnectionError,
    DepthCameraFrameError,
    DepthCameraNotOpenError,
)


@dataclass(frozen=True, slots=True)
class RealSenseDeviceInfo:
    serial_number: str
    name: str
    product_line: str


def _device_info(device: Any, rs: Any, field: Any) -> str:
    return str(device.get_info(field)) if device.supports(field) else "unknown"


def list_realsense_devices(rs_module: ModuleType | Any | None = None) -> list[RealSenseDeviceInfo]:
    """Enumerate connected RealSense devices without starting a stream."""

    rs = rs_module or import_module("pyrealsense2")
    return [
        RealSenseDeviceInfo(
            serial_number=_device_info(device, rs, rs.camera_info.serial_number),
            name=_device_info(device, rs, rs.camera_info.name),
            product_line=_device_info(device, rs, rs.camera_info.product_line),
        )
        for device in rs.context().query_devices()
    ]


class RealSenseD435i:
    """Capture synchronized left/right IR and optional native depth frames."""

    def __init__(
        self,
        config: RealSenseConfig,
        rs_module: ModuleType | Any | None = None,
    ) -> None:
        self._config = config
        self._rs_module = rs_module
        self._pipeline: Any | None = None
        self._calibration: StereoCalibrationSnapshot | None = None
        self._lock = RLock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._pipeline is not None

    @property
    def calibration(self) -> StereoCalibrationSnapshot:
        with self._lock:
            if self._calibration is None:
                raise DepthCameraNotOpenError("RealSense camera is not open")
            return self._calibration

    def __enter__(self) -> RealSenseD435i:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return

            rs = self._rs_module or import_module("pyrealsense2")
            pipeline = rs.pipeline()
            stream_config = rs.config()
            if self._config.serial_number is not None:
                stream_config.enable_device(self._config.serial_number)
            stream_config.enable_stream(
                rs.stream.infrared,
                1,
                self._config.infrared_width,
                self._config.infrared_height,
                rs.format.y8,
                self._config.frames_per_second,
            )
            stream_config.enable_stream(
                rs.stream.infrared,
                2,
                self._config.infrared_width,
                self._config.infrared_height,
                rs.format.y8,
                self._config.frames_per_second,
            )
            if self._config.enable_native_depth:
                stream_config.enable_stream(
                    rs.stream.depth,
                    self._config.infrared_width,
                    self._config.infrared_height,
                    rs.format.z16,
                    self._config.frames_per_second,
                )

            try:
                pipeline_profile = pipeline.start(stream_config)
                calibration = self._read_calibration(pipeline_profile, rs)
                for _ in range(self._config.warmup_frames):
                    pipeline.wait_for_frames(self._config.timeout_ms)
            except Exception as exc:
                with suppress(Exception):
                    pipeline.stop()
                raise DepthCameraConnectionError(f"failed to start D435i streams: {exc}") from exc

            self._pipeline = pipeline
            self._calibration = calibration

    def close(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._calibration = None

    def capture(self) -> StereoFrame:
        with self._lock:
            if self._pipeline is None or self._calibration is None:
                raise DepthCameraNotOpenError("RealSense camera is not open")
            try:
                frames = self._pipeline.wait_for_frames(self._config.timeout_ms)
                left_frame = frames.get_infrared_frame(1)
                right_frame = frames.get_infrared_frame(2)
                depth_frame = frames.get_depth_frame() if self._config.enable_native_depth else None
            except Exception as exc:
                raise DepthCameraFrameError(f"failed to receive D435i frames: {exc}") from exc

            if not left_frame or not right_frame:
                raise DepthCameraFrameError("D435i returned an incomplete stereo pair")
            if self._config.enable_native_depth and not depth_frame:
                raise DepthCameraFrameError("D435i returned no native depth frame")

            native_depth = None
            if depth_frame is not None:
                native_depth = np.asanyarray(depth_frame.get_data()).copy()
            return StereoFrame(
                monotonic_time_ns=time.monotonic_ns(),
                frame_number=int(left_frame.get_frame_number()),
                left_device_time_ms=float(left_frame.get_timestamp()),
                right_device_time_ms=float(right_frame.get_timestamp()),
                left_ir=np.asanyarray(left_frame.get_data()).copy(),
                right_ir=np.asanyarray(right_frame.get_data()).copy(),
                native_depth=native_depth,
                calibration=self._calibration,
            )

    def _read_calibration(self, pipeline_profile: Any, rs: Any) -> StereoCalibrationSnapshot:
        left_profile = pipeline_profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right_profile = pipeline_profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        left_intrinsics = _intrinsics_from_profile(left_profile)
        right_intrinsics = _intrinsics_from_profile(right_profile)

        left_to_right = left_profile.get_extrinsics_to(right_profile)
        # librealsense exposes rs2_extrinsics.rotation as a column-major array.
        rotation = np.asarray(left_to_right.rotation, dtype=np.float64).reshape(
            (3, 3), order="F"
        )
        translation = np.asarray(left_to_right.translation, dtype=np.float64)
        right_t_left = PoseSE3.from_rotation_translation(
            "right_ir",
            "left_ir",
            rotation,
            translation,
        )

        depth_scale = None
        if self._config.enable_native_depth:
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
        return StereoCalibrationSnapshot(
            left=left_intrinsics,
            right=right_intrinsics,
            right_t_left=right_t_left,
            native_depth_scale_m=depth_scale,
        )


def _intrinsics_from_profile(profile: Any) -> CameraIntrinsics:
    intrinsics = profile.get_intrinsics()
    return CameraIntrinsics(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.ppx),
        cy=float(intrinsics.ppy),
        distortion_model=str(intrinsics.model),
        distortion_coefficients=tuple(float(value) for value in intrinsics.coeffs),
    )
