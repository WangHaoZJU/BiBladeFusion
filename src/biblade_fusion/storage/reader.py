"""Validated reader for reproducible acquisition sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.thermal_camera.base import ThermalFrame


class SessionFormatError(ValueError):
    """A stored session is incomplete, unsafe, or incompatible."""


@dataclass(frozen=True, slots=True)
class StoredViewDescriptor:
    sequence_index: int
    view_id: str
    relative_path: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionFormatError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SessionFormatError(f"JSON root must be an object: {path}")
    return payload


def _intrinsics(payload: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        distortion_model=str(payload["distortion_model"]),
        distortion_coefficients=tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _pose(parent_frame: str, child_frame: str, matrix: Any) -> PoseSE3:
    return PoseSE3(parent_frame, child_frame, np.asarray(matrix, dtype=np.float64))


def _robot_state(payload: dict[str, Any]) -> RobotState:
    return RobotState(
        monotonic_time_ns=int(payload["monotonic_time_ns"]),
        controller_time_s=float(payload["controller_time_s"]),
        joint_positions_rad=np.asarray(payload["joint_positions_rad"], dtype=np.float64),
        base_t_tcp=_pose("base", "tcp", payload["base_T_tcp"]),
        robot_mode=str(payload["robot_mode"]),
        safety_status=str(payload["safety_status"]),
        speed_scaling=float(payload["speed_scaling"]),
    )


class SessionReader:
    """Read atomic views while validating schema and path containment."""

    def __init__(self, session_dir: str | Path) -> None:
        self.path = Path(session_dir).resolve()
        manifest_path = self.path / "manifest.json"
        self.manifest = _read_json(manifest_path)
        try:
            self.schema_version = int(self.manifest["schema_version"])
            raw_views = self.manifest["views"]
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionFormatError("Session manifest is missing required fields") from exc
        if self.schema_version not in {1, 2}:
            raise SessionFormatError(f"Unsupported session schema {self.schema_version}")
        if not isinstance(raw_views, list):
            raise SessionFormatError("Session manifest views must be a list")

        descriptors: list[StoredViewDescriptor] = []
        try:
            for item in raw_views:
                descriptor = StoredViewDescriptor(
                    sequence_index=int(item["sequence_index"]),
                    view_id=str(item["view_id"]),
                    relative_path=str(item["path"]),
                )
                self._contained_path(descriptor.relative_path)
                descriptors.append(descriptor)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, SessionFormatError):
                raise
            raise SessionFormatError("Session manifest contains an invalid view") from exc
        if len({item.sequence_index for item in descriptors}) != len(descriptors):
            raise SessionFormatError("Session contains duplicate sequence indices")
        self.views = tuple(sorted(descriptors, key=lambda item: item.sequence_index))

    def _contained_path(self, relative_path: str, *, root: Path | None = None) -> Path:
        if Path(relative_path).is_absolute():
            raise SessionFormatError(f"Absolute stored path is forbidden: {relative_path}")
        base = (root or self.path).resolve()
        candidate = (base / relative_path).resolve()
        if not candidate.is_relative_to(base):
            raise SessionFormatError(f"Stored path escapes its session directory: {relative_path}")
        return candidate

    def descriptor(self, selector: int | str) -> StoredViewDescriptor:
        matches = [
            item
            for item in self.views
            if (isinstance(selector, int) and item.sequence_index == selector)
            or (isinstance(selector, str) and item.view_id == selector)
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one stored view for {selector!r}, found {len(matches)}")
        return matches[0]

    def load_bundle(self, selector: int | str) -> SynchronizedFrameBundle:
        descriptor = self.descriptor(selector)
        view_path = self._contained_path(descriptor.relative_path)
        metadata = _read_json(view_path / "metadata.json")
        try:
            bundle = self._parse_bundle(view_path, metadata)
        except SessionFormatError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise SessionFormatError(f"Stored view is invalid: {view_path}: {exc}") from exc
        if (
            bundle.sequence_index != descriptor.sequence_index
            or bundle.view_id != descriptor.view_id
        ):
            raise SessionFormatError("Manifest and view metadata identifiers do not match")
        return bundle

    def _array(self, view_path: Path, filename: str) -> np.ndarray:
        path = self._contained_path(filename, root=view_path)
        if not path.is_file():
            raise SessionFormatError(f"Stored array is missing: {path}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise SessionFormatError(f"Cannot load stored array {path}: {exc}") from exc
        if not isinstance(array, np.ndarray):
            array.close()
            raise SessionFormatError(f"Stored array must be a single .npy array: {path}")
        return array

    def _parse_bundle(self, view_path: Path, metadata: dict[str, Any]) -> SynchronizedFrameBundle:
        stereo_data = metadata["stereo"]
        calibration_data = stereo_data["calibration"]
        depth_data = calibration_data.get("depth")
        left_t_depth_data = calibration_data.get("left_T_depth")
        native_depth_file = stereo_data.get("native_depth_file")
        native_scale = calibration_data.get("native_depth_scale_m")
        if native_depth_file is not None and (depth_data is None or left_t_depth_data is None):
            raise SessionFormatError(
                "Native depth lacks depth-stream calibration; schema-v1 data cannot be "
                "safely reconstructed"
            )

        calibration = StereoCalibrationSnapshot(
            left=_intrinsics(calibration_data["left"]),
            right=_intrinsics(calibration_data["right"]),
            right_t_left=_pose("right_ir", "left_ir", calibration_data["right_T_left"]),
            native_depth_scale_m=(float(native_scale) if native_scale is not None else None),
            depth=_intrinsics(depth_data) if depth_data is not None else None,
            left_t_depth=(
                _pose("left_ir", "depth", left_t_depth_data)
                if left_t_depth_data is not None
                else None
            ),
        )
        stereo = StereoFrame(
            monotonic_time_ns=int(stereo_data["monotonic_time_ns"]),
            frame_number=int(stereo_data["frame_number"]),
            left_device_time_ms=float(stereo_data["left_device_time_ms"]),
            right_device_time_ms=float(stereo_data["right_device_time_ms"]),
            left_ir=self._array(view_path, str(stereo_data["left_file"])),
            right_ir=self._array(view_path, str(stereo_data["right_file"])),
            native_depth=(
                self._array(view_path, str(native_depth_file))
                if native_depth_file is not None
                else None
            ),
            calibration=calibration,
        )

        thermal_data = metadata.get("thermal")
        thermal = None
        if thermal_data is not None:
            raw_file = thermal_data.get("raw_counts_file")
            thermal = ThermalFrame(
                monotonic_time_ns=int(thermal_data["monotonic_time_ns"]),
                device_time_ms=(
                    float(thermal_data["device_time_ms"])
                    if thermal_data.get("device_time_ms") is not None
                    else None
                ),
                temperature_c=self._array(view_path, str(thermal_data["temperature_file"])),
                raw_counts=(
                    self._array(view_path, str(raw_file)) if raw_file is not None else None
                ),
            )

        robot_data = metadata["robot"]
        metrics_data = metadata["synchronization"]
        return SynchronizedFrameBundle(
            view_id=str(metadata["view_id"]),
            sequence_index=int(metadata["sequence_index"]),
            robot_state_before=_robot_state(robot_data["before"]),
            robot_state_after=_robot_state(robot_data["after"]),
            selected_robot_state=_robot_state(robot_data["selected"]),
            stereo=stereo,
            thermal=thermal,
            metrics=CaptureMetrics(
                bracket_ms=float(metrics_data["bracket_ms"]),
                max_joint_delta_rad=float(metrics_data["max_joint_delta_rad"]),
                tcp_translation_delta_m=float(metrics_data["tcp_translation_delta_m"]),
                tcp_rotation_delta_rad=float(metrics_data["tcp_rotation_delta_rad"]),
                selected_robot_state_offset_ms=float(
                    metrics_data["selected_robot_state_offset_ms"]
                ),
            ),
        )
