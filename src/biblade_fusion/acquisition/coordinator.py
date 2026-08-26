"""Stop-and-capture synchronization for robot-mounted sensors."""

from __future__ import annotations

import numpy as np

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.acquisition.errors import AcquisitionRejectedError
from biblade_fusion.core.settings import AcquisitionConfig
from biblade_fusion.devices.depth_camera.base import StereoCamera
from biblade_fusion.devices.robot.base import RobotState, RobotStateSource
from biblade_fusion.devices.thermal_camera.base import ThermalCamera


def _rotation_delta_rad(before: RobotState, after: RobotState) -> float:
    relative = before.base_t_tcp.rotation.T @ after.base_t_tcp.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


class SynchronizedAcquirer:
    """Acquire sensors between two robot-state samples and validate stationarity."""

    def __init__(
        self,
        robot: RobotStateSource,
        stereo_camera: StereoCamera,
        thermal_camera: ThermalCamera,
        config: AcquisitionConfig,
        *,
        require_thermal: bool = False,
    ) -> None:
        self._robot = robot
        self._stereo_camera = stereo_camera
        self._thermal_camera = thermal_camera
        self._config = config
        self._require_thermal = require_thermal

    def capture(self, view_id: str, sequence_index: int) -> SynchronizedFrameBundle:
        """Capture one bundle or reject it if the robot moved during acquisition."""

        before = self._robot.read_state()
        stereo = self._stereo_camera.capture()
        thermal = self._thermal_camera.capture()
        after = self._robot.read_state()

        if self._require_thermal and thermal is None:
            raise AcquisitionRejectedError("thermal observation is required but unavailable")
        if not before.monotonic_time_ns <= stereo.monotonic_time_ns <= after.monotonic_time_ns:
            raise AcquisitionRejectedError("stereo host timestamp is outside robot-state bracket")
        if thermal is not None and not (
            before.monotonic_time_ns <= thermal.monotonic_time_ns <= after.monotonic_time_ns
        ):
            raise AcquisitionRejectedError("thermal host timestamp is outside robot-state bracket")

        bracket_ms = (after.monotonic_time_ns - before.monotonic_time_ns) / 1e6
        joint_delta = float(
            np.max(np.abs(after.joint_positions_rad - before.joint_positions_rad))
        )
        translation_delta = float(
            np.linalg.norm(after.base_t_tcp.translation_m - before.base_t_tcp.translation_m)
        )
        rotation_delta = _rotation_delta_rad(before, after)

        limits = (
            (bracket_ms, self._config.max_bracket_ms, "capture bracket", "ms"),
            (
                joint_delta,
                self._config.max_joint_delta_rad,
                "joint motion",
                "rad",
            ),
            (
                translation_delta,
                self._config.max_tcp_translation_delta_m,
                "TCP translation",
                "m",
            ),
            (
                rotation_delta,
                self._config.max_tcp_rotation_delta_rad,
                "TCP rotation",
                "rad",
            ),
        )
        for value, maximum, label, unit in limits:
            if value > maximum:
                raise AcquisitionRejectedError(
                    f"{label} {value:.9g} {unit} exceeds limit {maximum:.9g} {unit}"
                )

        before_offset = abs(stereo.monotonic_time_ns - before.monotonic_time_ns)
        after_offset = abs(after.monotonic_time_ns - stereo.monotonic_time_ns)
        selected = before if before_offset <= after_offset else after
        selected_offset_ms = min(before_offset, after_offset) / 1e6

        metrics = CaptureMetrics(
            bracket_ms=bracket_ms,
            max_joint_delta_rad=joint_delta,
            tcp_translation_delta_m=translation_delta,
            tcp_rotation_delta_rad=rotation_delta,
            selected_robot_state_offset_ms=selected_offset_ms,
        )
        return SynchronizedFrameBundle(
            view_id=view_id,
            sequence_index=sequence_index,
            robot_state_before=before,
            robot_state_after=after,
            selected_robot_state=selected,
            stereo=stereo,
            thermal=thermal,
            metrics=metrics,
        )

