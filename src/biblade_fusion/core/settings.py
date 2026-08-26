"""Validated project configuration."""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path = Path("data")
    log_root: Path = Path("logs")


class RobotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["cs68"] = "cs68"
    robot_ip: str | None = None
    local_ip: str | None = None
    headless_mode: bool = False
    motion_enabled: bool = False
    rtsi_frequency_hz: float = Field(default=250.0, gt=0.0, le=500.0)
    settle_time_s: float = Field(default=1.0, ge=0.0)
    sdk_wheel: Path

    @field_validator("robot_ip", "local_ip")
    @classmethod
    def validate_ip(cls, value: str | None) -> str | None:
        if value is not None:
            ip_address(value)
        return value


class RealSenseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial_number: str | None = None
    infrared_width: int = Field(default=1280, gt=0)
    infrared_height: int = Field(default=720, gt=0)
    frames_per_second: int = Field(default=30, gt=0, le=90)
    enable_native_depth: bool = True
    warmup_frames: int = Field(default=15, ge=0, le=300)
    timeout_ms: int = Field(default=5000, gt=0, le=60000)


class ThermalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    driver: str | None = None


class AppSettings(BaseModel):
    """Top-level BiBladeFusion settings."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    robot: RobotConfig
    realsense: RealSenseConfig
    thermal: ThermalConfig


def load_settings(path: str | Path) -> AppSettings:
    """Load and validate a YAML settings file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return AppSettings.model_validate(raw)
