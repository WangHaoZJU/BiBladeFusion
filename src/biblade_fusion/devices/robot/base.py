"""Robot-independent state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3


@dataclass(frozen=True, slots=True)
class RobotState:
    """A synchronized snapshot of the robot state."""

    monotonic_time_ns: int
    controller_time_s: float
    joint_positions_rad: NDArray[np.float64]
    base_t_tcp: PoseSE3
    robot_mode: str
    safety_status: str
    speed_scaling: float
    runtime_state: str | None = None
    actual_joint_velocity_rad_s: NDArray[np.float64] | None = None
    target_joint_velocity_rad_s: NDArray[np.float64] | None = None
    actual_tcp_velocity: NDArray[np.float64] | None = None
    target_tcp_velocity: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        joints = np.array(self.joint_positions_rad, dtype=np.float64, copy=True)
        if joints.shape != (6,):
            raise ValueError(f"Expected six robot joints, got shape {joints.shape}")
        if not np.isfinite(joints).all():
            raise ValueError("Joint positions must be finite")
        if self.monotonic_time_ns < 0 or self.controller_time_s < 0:
            raise ValueError("Robot timestamps must be non-negative")
        if not 0.0 <= self.speed_scaling <= 1.0:
            raise ValueError("Robot speed scaling must be in [0, 1]")
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions_rad", joints)
        for name in (
            "actual_joint_velocity_rad_s",
            "target_joint_velocity_rad_s",
            "actual_tcp_velocity",
            "target_tcp_velocity",
        ):
            raw = getattr(self, name)
            if raw is None:
                continue
            vector = np.array(raw, dtype=np.float64, copy=True)
            if vector.shape != (6,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must be a finite six-vector")
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        if self.runtime_state is not None:
            runtime_state = str(self.runtime_state).strip()
            if not runtime_state:
                raise ValueError("runtime_state must be non-empty when present")
            object.__setattr__(self, "runtime_state", runtime_state)


@runtime_checkable
class RobotStateSource(Protocol):
    """A read-only source of robot state."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self) -> RobotState: ...
