"""Validated project configuration."""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class AcquisitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_bracket_ms: float = Field(default=250.0, gt=0.0)
    max_joint_delta_rad: float = Field(default=0.001, ge=0.0)
    max_tcp_translation_delta_m: float = Field(default=0.0005, ge=0.0)
    max_tcp_rotation_delta_rad: float = Field(default=0.001, ge=0.0)


class FoundationStereoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_path: Path = Path("third_party/FoundationStereo")
    checkpoint_path: Path = Path("models/foundation_stereo/23-51-11/model_best_bp2.pth")
    device: Literal["cuda", "cpu"] = "cuda"
    scale: float = Field(default=1.0, gt=0.0, le=1.0)
    valid_iterations: int = Field(default=32, gt=0, le=128)
    hierarchical: bool = False
    remove_invisible: bool = True


class ProxyModelConfig(BaseModel):
    """Conservative single-view geometry assumptions for bilateral planning."""

    model_config = ConfigDict(extra="forbid")

    voxel_size_m: float = Field(default=0.002, gt=0.0)
    minimum_points: int = Field(default=100, ge=6)
    estimated_planar_extents_m: tuple[float, float] | None = None
    estimated_thickness_m: float | None = Field(default=None, gt=0.0)
    tangential_margin_m: float = Field(default=0.01, ge=0.0)
    visible_side_margin_m: float = Field(default=0.003, ge=0.0)
    hidden_side_margin_m: float = Field(default=0.005, ge=0.0)
    minimum_camera_normal_cosine: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("estimated_planar_extents_m")
    @classmethod
    def validate_planar_extents(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if value[0] <= 0.0 or value[1] <= 0.0:
            raise ValueError("Estimated planar extents must be positive")
        if value[0] < value[1]:
            raise ValueError("Estimated planar extents must be ordered major then minor")
        return value


class PointCloudConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_depth_m: float = Field(default=0.15, gt=0.0)
    maximum_depth_m: float = Field(default=1.5, gt=0.0)
    pixel_stride: int = Field(default=1, ge=1, le=16)
    minimum_valid_points: int = Field(default=100, ge=3)

    @model_validator(mode="after")
    def validate_depth_range(self) -> Self:
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
        return self


class HandEyeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_path: Path | None = None
    minimum_samples: int = Field(default=15, ge=3)
    maximum_translation_rmse_m: float = Field(default=0.002, gt=0.0)
    maximum_rotation_rmse_deg: float = Field(default=0.5, gt=0.0)
    require_quality_metrics: bool = True


class ViewPlanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standoff_distance_m: float | None = Field(default=None, gt=0.0)
    overlap_fraction: float = Field(default=0.3, ge=0.0, lt=0.9)
    footprint_utilization: float = Field(default=0.8, gt=0.0, le=1.0)
    edge_margin_m: float = Field(default=0.005, ge=0.0)
    maximum_candidates: int = Field(default=200, ge=2, le=10000)


class AxisAlignedBoxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.name:
            raise ValueError("Axis-aligned box name must be non-empty")
        if not np.isfinite((*self.minimum_m, *self.maximum_m)).all():
            raise ValueError("Axis-aligned box bounds must be finite")
        if any(lower >= upper for lower, upper in zip(self.minimum_m, self.maximum_m, strict=True)):
            raise ValueError("Axis-aligned box minima must be below maxima")
        return self


class ViewFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: AxisAlignedBoxConfig | None = None
    forbidden_volumes: tuple[AxisAlignedBoxConfig, ...] = ()
    camera_clearance_radius_m: float = Field(default=0.05, gt=0.0)
    minimum_look_at_cosine: float = Field(default=0.999, gt=0.0, le=1.0)
    minimum_incidence_cosine: float = Field(default=0.95, gt=0.0, le=1.0)
    maximum_standoff_error_m: float = Field(default=0.005, ge=0.0)
    duplicate_translation_tolerance_m: float = Field(default=0.005, ge=0.0)
    duplicate_rotation_tolerance_deg: float = Field(default=2.0, ge=0.0, le=180.0)


class KinematicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path | None = None
    plugin_path: Path | None = None
    primary_timeout_ms: int = Field(default=1000, gt=0, le=30000)
    ik_timeout_s: float = Field(default=0.05, gt=0.0, le=5.0)


class AppSettings(BaseModel):
    """Top-level BiBladeFusion settings."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    robot: RobotConfig
    realsense: RealSenseConfig
    thermal: ThermalConfig
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    foundation_stereo: FoundationStereoConfig = Field(default_factory=FoundationStereoConfig)
    proxy_model: ProxyModelConfig = Field(default_factory=ProxyModelConfig)
    point_cloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)
    view_planning: ViewPlanningConfig = Field(default_factory=ViewPlanningConfig)
    view_filter: ViewFilterConfig = Field(default_factory=ViewFilterConfig)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)


def load_settings(path: str | Path) -> AppSettings:
    """Load and validate a YAML settings file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return AppSettings.model_validate(raw)
