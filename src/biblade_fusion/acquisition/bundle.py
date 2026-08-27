"""Immutable acquisition bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.devices.depth_camera.base import StereoFrame
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.thermal_camera.base import ThermalFrame


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    bracket_ms: float
    max_joint_delta_rad: float
    tcp_translation_delta_m: float
    tcp_rotation_delta_rad: float
    selected_robot_state_offset_ms: float


@dataclass(frozen=True, slots=True)
class SynchronizedFrameBundle:
    """One audited sensor observation bracketed by two robot states."""

    view_id: str
    sequence_index: int
    robot_state_before: RobotState
    robot_state_after: RobotState
    selected_robot_state: RobotState
    stereo: StereoFrame
    thermal: ThermalFrame | None
    metrics: CaptureMetrics

    def __post_init__(self) -> None:
        if not self.view_id:
            raise ValueError("View ID must be non-empty")
        if self.sequence_index < 0:
            raise ValueError("Sequence index must be non-negative")
