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
    sdk_import_path: str = "elite_cs_sdk"
    headless_mode: bool = True
    motion_enabled: bool = False
    script_file_path: Path | None = None
    reverse_port: int = Field(default=50002, ge=1024, le=65535)
    script_sender_port: int = Field(default=50001, ge=1024, le=65535)
    trajectory_port: int = Field(default=50003, ge=1024, le=65535)
    script_command_port: int = Field(default=50004, ge=1024, le=65535)
    servoj_time_s: float = Field(default=0.004, gt=0.0)
    servoj_lookahead_time_s: float = Field(default=0.1, ge=0.03, le=0.2)
    servoj_gain: int = Field(default=2000, gt=0)
    stopj_acceleration_rad_s2: float = Field(default=2.0, gt=0.0)
    default_speed_scaling: float = Field(default=0.3, ge=0.0, le=1.0)
    maximum_speed_scaling: float = Field(default=1.0, gt=0.0, le=1.0)
    default_motion_timeout_s: float = Field(default=15.0, gt=0.0)
    maximum_motion_timeout_s: float = Field(default=60.0, gt=0.0)
    motion_poll_period_s: float = Field(default=0.01, gt=0.0)
    default_trajectory_time_s: float = Field(default=3.0, gt=0.0)
    default_blend_radius_m: float = Field(default=0.0, ge=0.0)
    rtsi_frequency_hz: float = Field(default=125.0, gt=0.0, le=500.0)
    settle_time_s: float = Field(default=1.0, ge=0.0)
    sdk_wheel: Path

    @field_validator("robot_ip", "local_ip")
    @classmethod
    def validate_ip(cls, value: str | None) -> str | None:
        if value is not None:
            ip_address(value)
        return value

    @model_validator(mode="after")
    def validate_motion_control(self) -> Self:
        ports = (
            self.reverse_port,
            self.script_sender_port,
            self.trajectory_port,
            self.script_command_port,
        )
        if len(set(ports)) != len(ports):
            raise ValueError("Elite external-control ports must be unique")
        if self.default_speed_scaling > self.maximum_speed_scaling:
            raise ValueError("Default speed scaling exceeds the configured maximum")
        if self.default_motion_timeout_s > self.maximum_motion_timeout_s:
            raise ValueError("Default motion timeout exceeds the configured maximum")
        if not self.sdk_import_path.strip():
            raise ValueError("Elite SDK import path must be non-empty")
        return self


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
    model_config_path: Path | None = None
    device: Literal["cuda", "cpu"] = "cuda"
    scale: float = Field(default=1.0, gt=0.0, le=1.0)
    valid_iterations: int = Field(default=32, gt=0, le=128)
    hierarchical: bool = False
    remove_invisible: bool = True


class StereoRectificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.0, ge=-1.0, le=1.0)
    zero_disparity: bool = True
    interpolation: Literal["linear", "nearest"] = "linear"


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


class CharucoTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    squares_x: int = Field(default=7, ge=3, le=30)
    squares_y: int = Field(default=5, ge=3, le=30)
    square_length_m: float | None = Field(default=None, gt=0.0)
    marker_length_m: float | None = Field(default=None, gt=0.0)
    dictionary: Literal[
        "DICT_4X4_50",
        "DICT_5X5_100",
        "DICT_6X6_250",
        "DICT_APRILTAG_36h11",
    ] = "DICT_5X5_100"
    legacy_pattern: bool = False
    minimum_corners: int = Field(default=12, ge=4)
    maximum_reprojection_rmse_px: float = Field(default=0.8, gt=0.0)
    minimum_pose_ambiguity_ratio: float = Field(default=1.5, gt=1.0)

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if (self.square_length_m is None) != (self.marker_length_m is None):
            raise ValueError("ChArUco square and marker lengths must be configured together")
        if (
            self.square_length_m is not None
            and self.marker_length_m is not None
            and self.marker_length_m >= self.square_length_m
        ):
            raise ValueError("ChArUco marker length must be below square length")
        maximum_corners = (self.squares_x - 1) * (self.squares_y - 1)
        if self.minimum_corners > maximum_corners:
            raise ValueError("ChArUco minimum corners exceeds the board's available corners")
        return self


class HandEyeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration_path: Path | None = None
    minimum_samples: int = Field(default=15, ge=3)
    maximum_translation_rmse_m: float = Field(default=0.002, gt=0.0)
    maximum_rotation_rmse_deg: float = Field(default=0.5, gt=0.0)
    minimum_rotation_span_deg: float = Field(default=20.0, gt=0.0, le=180.0)
    minimum_translation_span_m: float = Field(default=0.03, gt=0.0)
    minimum_rotation_axis_diversity: float = Field(default=0.1, gt=0.0, le=1.0)
    require_quality_metrics: bool = True
    require_observability_metrics: bool = True
    target: CharucoTargetConfig = Field(default_factory=CharucoTargetConfig)


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


class CoverageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bins_per_axis: int = Field(default=8, ge=2, le=64)
    minimum_points_per_bin: int = Field(default=2, ge=1, le=1000)
    completed_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    maximum_surface_distance_m: float = Field(default=0.02, gt=0.0)
    minimum_camera_side_offset_m: float = Field(default=0.02, gt=0.0)
    minimum_surface_points_per_view: int = Field(default=50, ge=1)


class DepthComparisonConfig(BaseModel):
    """Reproducible native-versus-stereo comparison settings."""

    model_config = ConfigDict(extra="forbid")

    minimum_overlap_points: int = Field(default=100, ge=3)
    agreement_thresholds_m: tuple[float, ...] = (0.005, 0.01, 0.02)
    minimum_camera_side_offset_m: float = Field(default=0.02, gt=0.0)
    incidence_bin_edges_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)

    @field_validator("agreement_thresholds_m")
    @classmethod
    def validate_agreement_thresholds(
        cls, value: tuple[float, ...]
    ) -> tuple[float, ...]:
        if not value or not np.isfinite(value).all() or any(item <= 0.0 for item in value):
            raise ValueError("Depth agreement thresholds must be finite and positive")
        if tuple(sorted(set(value))) != value:
            raise ValueError("Depth agreement thresholds must be unique and increasing")
        return value

    @field_validator("incidence_bin_edges_deg")
    @classmethod
    def validate_incidence_edges(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if (
            len(value) < 2
            or value[0] != 0.0
            or value[-1] != 90.0
            or not np.isfinite(value).all()
            or any(
                first >= second
                for first, second in zip(value, value[1:], strict=False)
            )
        ):
            raise ValueError("Incidence bin edges must increase from 0 to 90 degrees")
        return value


class KinematicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path | None = None
    plugin_path: Path | None = None
    primary_timeout_ms: int = Field(default=1000, gt=0, le=30000)
    ik_timeout_s: float = Field(default=0.05, gt=0.0, le=5.0)


class CollisionObstacleConfig(AxisAlignedBoxConfig):
    """Conservative workcell box with explicit robot-capsule exemptions."""

    ignored_capsule_indices: tuple[int, ...] = ()

    @field_validator("ignored_capsule_indices")
    @classmethod
    def validate_ignored_capsules(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(index < 0 or index > 6 for index in value) or len(set(value)) != len(value):
            raise ValueError("Ignored capsule indices must be unique values in [0, 6]")
        return value


class CollisionConfig(BaseModel):
    """Fail-closed geometry required for offline CS68 path validation."""

    model_config = ConfigDict(extra="forbid")

    link_radii_m: tuple[float, float, float, float, float, float] | None = None
    camera_tool_radius_m: float | None = Field(default=None, gt=0.0)
    minimum_joint_positions_rad: tuple[float, float, float, float, float, float] | None = None
    maximum_joint_positions_rad: tuple[float, float, float, float, float, float] | None = None
    obstacles: tuple[CollisionObstacleConfig, ...] = ()
    require_obstacles: bool = True
    minimum_clearance_m: float = Field(default=0.01, ge=0.0)
    maximum_joint_step_rad: float = Field(default=0.02, gt=0.0, le=0.2)

    @model_validator(mode="after")
    def validate_collision_geometry(self) -> Self:
        if self.link_radii_m is not None and (
            not np.isfinite(self.link_radii_m).all()
            or any(radius <= 0.0 for radius in self.link_radii_m)
        ):
            raise ValueError("Collision link radii must be finite and positive")
        limits = (self.minimum_joint_positions_rad, self.maximum_joint_positions_rad)
        if (limits[0] is None) != (limits[1] is None):
            raise ValueError("Both minimum and maximum joint limits must be configured")
        if (
            limits[0] is not None
            and limits[1] is not None
            and (
                not np.isfinite((*limits[0], *limits[1])).all()
                or any(
                lower >= upper for lower, upper in zip(*limits, strict=True)
                )
            )
        ):
            raise ValueError("Collision joint limits must be finite and ordered")
        return self


class MotionPreflightConfig(BaseModel):
    """HoloRobot-derived conservative planning and ServoJ generation settings."""

    model_config = ConfigDict(extra="forbid")

    maximum_joint_step_rad: float = Field(default=0.02, gt=0.0, le=0.2)
    servoj_dt_s: float = Field(default=0.004, gt=0.0, le=0.1)
    speed_scaling: float = Field(default=0.08, gt=0.0, le=1.0)
    velocity_margin: float = Field(default=0.8, gt=0.0, le=1.0)


class AppSettings(BaseModel):
    """Top-level BiBladeFusion settings."""

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    robot: RobotConfig
    realsense: RealSenseConfig
    thermal: ThermalConfig
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    foundation_stereo: FoundationStereoConfig = Field(default_factory=FoundationStereoConfig)
    stereo_rectification: StereoRectificationConfig = Field(
        default_factory=StereoRectificationConfig
    )
    proxy_model: ProxyModelConfig = Field(default_factory=ProxyModelConfig)
    point_cloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)
    view_planning: ViewPlanningConfig = Field(default_factory=ViewPlanningConfig)
    view_filter: ViewFilterConfig = Field(default_factory=ViewFilterConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    depth_comparison: DepthComparisonConfig = Field(default_factory=DepthComparisonConfig)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    collision: CollisionConfig = Field(default_factory=CollisionConfig)
    motion_preflight: MotionPreflightConfig = Field(default_factory=MotionPreflightConfig)


def load_settings(path: str | Path) -> AppSettings:
    """Load and validate a YAML settings file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return AppSettings.model_validate(raw)
