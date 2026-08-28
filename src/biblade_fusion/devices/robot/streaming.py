"""HoloRobot-compatible single-arm ServoJ stream contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ServoJStreamConfig:
    """Runtime guards applied while sending a prevalidated ServoJ stream."""

    dt_s: float = 0.004
    tracking_error_rad: float = 0.03
    max_consecutive_tracking_violations: int = 5
    tracking_check_every_n_commands: int = 2
    max_consecutive_timing_violations: int = 5
    timing_violation_factor: float = 1.5

    def validate(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("ServoJ dt_s must be finite and positive")
        if (
            not math.isfinite(self.tracking_error_rad)
            or self.tracking_error_rad <= 0.0
        ):
            raise ValueError("ServoJ tracking_error_rad must be finite and positive")
        if self.max_consecutive_tracking_violations < 1:
            raise ValueError("ServoJ tracking violation count must be positive")
        if self.tracking_check_every_n_commands < 1:
            raise ValueError("ServoJ tracking check interval must be positive")
        if self.max_consecutive_timing_violations < 1:
            raise ValueError("ServoJ timing violation count must be positive")
        if (
            not math.isfinite(self.timing_violation_factor)
            or self.timing_violation_factor <= 1.0
        ):
            raise ValueError(
                "ServoJ timing_violation_factor must be finite and exceed one"
            )


@dataclass(frozen=True, slots=True)
class ServoJStream:
    """A fixed-rate CS68 joint-command sequence produced by offline planning."""

    commands: tuple[tuple[float, float, float, float, float, float], ...]
    dt_s: float

    def validate(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("ServoJStream.dt_s must be finite and positive")
        if not self.commands:
            raise ValueError("ServoJStream.commands must be non-empty")
        for index, command in enumerate(self.commands):
            values = np.asarray(command, dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(
                    f"ServoJ command {index} must be a finite six-vector"
                )


@dataclass(frozen=True, slots=True)
class StreamServoJResult:
    ok: bool
    commands_sent: int = 0
    duration_s: float = 0.0
    max_tracking_error_rad: float = 0.0
    abort_reason: str | None = None
    last_command_index: int | None = None
    timing_summary: dict[str, float | int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "commands_sent": self.commands_sent,
            "duration_s": self.duration_s,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "abort_reason": self.abort_reason,
            "last_command_index": self.last_command_index,
            "timing_summary": dict(self.timing_summary or {}),
        }
