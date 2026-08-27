"""Read-only RTSI adapter for the configured Elite arm (ES68 in this project).

This module intentionally does not construct ``EliteDriver`` and exposes no motion,
power-on, brake-release, I/O-write, or script-send operation.
"""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any

import numpy as np

from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.conversions import elite_tcp_pose_to_se3
from biblade_fusion.devices.robot.errors import (
    RobotConfigurationError,
    RobotConnectionError,
    RobotNotConnectedError,
)


def _enum_label(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", maxsplit=1)[-1]


class EliteReadOnlyRobot:
    """Read Elite state through RTSI without enabling robot motion."""

    def __init__(self, config: RobotConfig, sdk_module: ModuleType | Any | None = None) -> None:
        self._config = config
        self._sdk_module = sdk_module
        self._rtsi: Any | None = None
        self._lock = RLock()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._rtsi is not None

    def __enter__(self) -> EliteReadOnlyRobot:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    def connect(self) -> None:
        with self._lock:
            if self._rtsi is not None:
                return
            if self._config.robot_ip is None:
                raise RobotConfigurationError("robot.robot_ip must be configured")

            sdk = self._sdk_module or import_module("elite_cs_sdk")
            resource_dir = Path(__file__).resolve().parent / "resources"
            rtsi = sdk.RtsiIOInterface(
                str(resource_dir / "output_recipe.txt"),
                str(resource_dir / "input_recipe.txt"),
                self._config.rtsi_frequency_hz,
            )
            if not rtsi.connect(self._config.robot_ip):
                try:
                    rtsi.disconnect()
                finally:
                    raise RobotConnectionError(
                        f"failed to connect Elite RTSI at {self._config.robot_ip}:30004"
                    )
            self._rtsi = rtsi

    def disconnect(self) -> None:
        with self._lock:
            if self._rtsi is None:
                return
            try:
                self._rtsi.disconnect()
            finally:
                self._rtsi = None

    def controller_version(self) -> str:
        """Return the connected controller version as a display string."""

        with self._lock:
            rtsi = self._require_connection()
            version = rtsi.getControllerVersion()
            return (
                f"{int(version.major)}.{int(version.minor)}."
                f"{int(version.bugfix)}+{int(version.build)}"
            )

    def read_state(self) -> RobotState:
        """Read one internally synchronized state snapshot."""

        with self._lock:
            rtsi = self._require_connection()
            joints = np.asarray(rtsi.getActualJointPositions(), dtype=np.float64)
            tcp_pose = rtsi.getActualTCPPose()
            return RobotState(
                monotonic_time_ns=time.monotonic_ns(),
                controller_time_s=float(rtsi.getTimestamp()),
                joint_positions_rad=joints,
                base_t_tcp=elite_tcp_pose_to_se3(tcp_pose),
                robot_mode=_enum_label(rtsi.getRobotMode()),
                safety_status=_enum_label(rtsi.getSafetyStatus()),
                speed_scaling=float(rtsi.getActualSpeedScaling()),
            )

    def _require_connection(self) -> Any:
        if self._rtsi is None:
            raise RobotNotConnectedError("Elite RTSI is not connected")
        return self._rtsi
