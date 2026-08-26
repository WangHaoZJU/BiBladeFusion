"""Atomic, self-describing scan-session writer."""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion import __version__
from biblade_fusion.acquisition.bundle import SynchronizedFrameBundle
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.devices.robot.base import RobotState

SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _safe_view_name(view_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", view_id).strip("_")
    if not sanitized:
        raise ValueError("View ID must contain at least one safe filename character")
    return sanitized


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion_model": intrinsics.distortion_model,
        "distortion_coefficients": list(intrinsics.distortion_coefficients),
    }


def _robot_state_payload(state: RobotState) -> dict[str, Any]:
    return {
        "monotonic_time_ns": state.monotonic_time_ns,
        "controller_time_s": state.controller_time_s,
        "joint_positions_rad": state.joint_positions_rad.tolist(),
        "base_T_tcp": state.base_t_tcp.matrix.tolist(),
        "robot_mode": state.robot_mode,
        "safety_status": state.safety_status,
        "speed_scaling": state.speed_scaling,
    }


class SessionWriter:
    """Write a scan session while preserving raw observations and provenance."""

    def __init__(self, session_dir: Path, settings: AppSettings) -> None:
        self.path = session_dir
        self._settings = settings
        self._manifest_path = self.path / "manifest.json"
        self._closed = False
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "project": "BiBladeFusion",
            "project_version": __version__,
            "session_id": self.path.name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "open",
            "views": [],
        }

    @classmethod
    def create(
        cls,
        root: str | Path,
        settings: AppSettings,
        *,
        label: str = "scan",
    ) -> SessionWriter:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        safe_label = _safe_view_name(label)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        session_dir = root_path / f"{timestamp}_{safe_label}_{uuid4().hex[:8]}"
        session_dir.mkdir()
        (session_dir / "views").mkdir()

        writer = cls(session_dir, settings)
        _atomic_json(session_dir / "config_snapshot.json", settings.model_dump(mode="json"))
        _atomic_json(writer._manifest_path, writer._manifest)
        return writer

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.close("failed" if exc_type is not None else "completed")

    def write_bundle(self, bundle: SynchronizedFrameBundle) -> Path:
        """Atomically append one synchronized observation to the session."""

        if self._closed:
            raise RuntimeError("Cannot write to a closed session")

        view_name = f"{bundle.sequence_index:04d}_{_safe_view_name(bundle.view_id)}"
        final_dir = self.path / "views" / view_name
        if final_dir.exists():
            raise FileExistsError(f"View already exists: {final_dir}")
        temporary_dir = self.path / "views" / f".{view_name}.{uuid4().hex}.partial"
        temporary_dir.mkdir()

        try:
            np.save(temporary_dir / "left_ir.npy", bundle.stereo.left_ir, allow_pickle=False)
            np.save(temporary_dir / "right_ir.npy", bundle.stereo.right_ir, allow_pickle=False)
            if bundle.stereo.native_depth is not None:
                np.save(
                    temporary_dir / "native_depth.npy",
                    bundle.stereo.native_depth,
                    allow_pickle=False,
                )
            if bundle.thermal is not None:
                np.save(
                    temporary_dir / "temperature_c.npy",
                    bundle.thermal.temperature_c,
                    allow_pickle=False,
                )
                if bundle.thermal.raw_counts is not None:
                    np.save(
                        temporary_dir / "thermal_raw_counts.npy",
                        bundle.thermal.raw_counts,
                        allow_pickle=False,
                    )

            _atomic_json(temporary_dir / "metadata.json", self._bundle_payload(bundle))
            temporary_dir.replace(final_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        self._manifest["views"].append(
            {
                "sequence_index": bundle.sequence_index,
                "view_id": bundle.view_id,
                "path": f"views/{view_name}",
            }
        )
        _atomic_json(self._manifest_path, self._manifest)
        return final_dir

    def close(self, status: str = "completed") -> None:
        if self._closed:
            return
        if status not in {"completed", "failed", "aborted"}:
            raise ValueError(f"Unsupported session status: {status}")
        self._manifest["status"] = status
        self._manifest["closed_at_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(self._manifest_path, self._manifest)
        self._closed = True

    def _bundle_payload(self, bundle: SynchronizedFrameBundle) -> dict[str, Any]:
        calibration = bundle.stereo.calibration
        thermal_payload = None
        if bundle.thermal is not None:
            thermal_payload = {
                "monotonic_time_ns": bundle.thermal.monotonic_time_ns,
                "device_time_ms": bundle.thermal.device_time_ms,
                "temperature_file": "temperature_c.npy",
                "raw_counts_file": (
                    "thermal_raw_counts.npy" if bundle.thermal.raw_counts is not None else None
                ),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "view_id": bundle.view_id,
            "sequence_index": bundle.sequence_index,
            "stereo": {
                "monotonic_time_ns": bundle.stereo.monotonic_time_ns,
                "frame_number": bundle.stereo.frame_number,
                "left_device_time_ms": bundle.stereo.left_device_time_ms,
                "right_device_time_ms": bundle.stereo.right_device_time_ms,
                "left_file": "left_ir.npy",
                "right_file": "right_ir.npy",
                "native_depth_file": (
                    "native_depth.npy" if bundle.stereo.native_depth is not None else None
                ),
                "calibration": {
                    "left": _intrinsics_payload(calibration.left),
                    "right": _intrinsics_payload(calibration.right),
                    "right_T_left": calibration.right_t_left.matrix.tolist(),
                    "native_depth_scale_m": calibration.native_depth_scale_m,
                },
            },
            "thermal": thermal_payload,
            "robot": {
                "before": _robot_state_payload(bundle.robot_state_before),
                "after": _robot_state_payload(bundle.robot_state_after),
                "selected": _robot_state_payload(bundle.selected_robot_state),
            },
            "synchronization": {
                "bracket_ms": bundle.metrics.bracket_ms,
                "max_joint_delta_rad": bundle.metrics.max_joint_delta_rad,
                "tcp_translation_delta_m": bundle.metrics.tcp_translation_delta_m,
                "tcp_rotation_delta_rad": bundle.metrics.tcp_rotation_delta_rad,
                "selected_robot_state_offset_ms": (
                    bundle.metrics.selected_robot_state_offset_ms
                ),
            },
        }

