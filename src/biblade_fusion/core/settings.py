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

    model: Literal["es68", "cs68"] = "es68"
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
    infrared_emitter_enabled: bool = False
    stereo_calibration_path: Path | None = None


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
    left_right_consistency_threshold_px: float | None = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
    )


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


class BladeForegroundConfig(BaseModel):
    """Fail-closed reference-guided foreground extraction for fine scanning.

    The policy point-splats the immutable coarse blade surface into the current
    rectified left image and accepts only measured depths that agree with that
    prediction.  It deliberately contains no connected-component or erosion
    stage because either operation can discard a thin fin or a one-pixel edge.
    ``projection_radius_px`` is a sampling-envelope parameter, not proof of a
    continuous triangle-rasterised visibility surface.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    method: Literal["reference_projected"] = "reference_projected"
    projection_radius_px: int = Field(default=3, ge=0, le=20)
    minimum_projection_depth_m: float = Field(default=0.05, gt=0.0)
    maximum_projection_depth_m: float = Field(default=3.0, gt=0.0)
    front_depth_tolerance_m: float = Field(default=0.006, ge=0.0, le=0.10)
    back_depth_tolerance_m: float = Field(default=0.010, ge=0.0, le=0.20)
    minimum_target_incidence_cosine: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_reference_pixels: int = Field(default=30, ge=1)
    minimum_target_reference_pixels: int = Field(default=6, ge=1)
    minimum_mask_pixels: int = Field(default=30, ge=1)
    minimum_target_mask_pixels: int = Field(default=3, ge=1)
    minimum_reference_match_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_target_match_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_mask_fraction: float = Field(default=0.0001, ge=0.0, lt=1.0)
    maximum_mask_fraction: float = Field(default=0.80, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_reference_mask_policy(self) -> Self:
        if self.maximum_projection_depth_m <= self.minimum_projection_depth_m:
            raise ValueError("Blade-foreground maximum projection depth must exceed minimum")
        if self.maximum_mask_fraction <= self.minimum_mask_fraction:
            raise ValueError("Blade-foreground maximum mask fraction must exceed minimum")
        return self


class BootstrapForegroundSettings(BaseModel):
    """Unknown-blade foreground policy before a coarse reference exists.

    The algorithm keeps this policy separate from the later schema-5
    reference-projected foreground gate.  It rejects depth components touching
    the valid-domain boundary and refuses ambiguous automatic selections; a
    recorded rectangle or polygon can be supplied as an explicit fallback.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_depth_m: float = Field(default=0.15, gt=0.0)
    maximum_depth_m: float = Field(default=2.0, gt=0.0)
    maximum_neighbour_depth_jump_m: float = Field(default=0.030, ge=0.0)
    maximum_neighbour_relative_depth_jump: float = Field(
        default=0.035,
        ge=0.0,
        le=1.0,
    )
    connectivity: Literal[4, 8] = 8
    boundary_margin_px: int = Field(default=2, ge=1, le=100)
    minimum_valid_pixels: int = Field(default=1_000, ge=1)
    minimum_component_pixels: int = Field(default=100, ge=1)
    minimum_mask_pixels: int = Field(default=500, ge=1)
    minimum_mask_fraction: float = Field(default=0.001, ge=0.0, le=1.0)
    maximum_mask_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_unseeded_ambiguity_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )
    minimum_seed_valid_pixels: int = Field(default=25, ge=1)
    minimum_seed_valid_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_component_hint_selection_fraction: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_bootstrap_foreground_policy(self) -> Self:
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("Bootstrap-foreground maximum depth must exceed minimum")
        if self.maximum_mask_fraction < self.minimum_mask_fraction:
            raise ValueError("Bootstrap-foreground maximum mask fraction is below minimum")
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
    detector_params: dict[str, int | float | bool] = Field(default_factory=dict)

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
    maximum_fk_tcp_translation_error_m: float = Field(default=0.002, gt=0.0)
    maximum_fk_tcp_rotation_error_deg: float = Field(default=0.3, gt=0.0)
    minimum_novel_translation_m: float = Field(default=0.01, gt=0.0)
    minimum_novel_rotation_deg: float = Field(default=5.0, gt=0.0, le=180.0)
    validation_minimum_samples: int = Field(default=5, ge=3, le=50)
    validation_maximum_translation_rmse_m: float = Field(default=0.003, gt=0.0)
    validation_maximum_rotation_rmse_deg: float = Field(default=0.5, gt=0.0)
    validation_maximum_reprojection_rmse_px: float = Field(default=0.8, gt=0.0)
    require_quality_metrics: bool = True
    require_observability_metrics: bool = True
    initial_method: Literal["daniilidis", "park", "tsai", "horaud", "andreff"] = "park"
    enable_bundle_adjustment: bool = True
    target: CharucoTargetConfig = Field(default_factory=CharucoTargetConfig)


class CoarseReachabilityFallbackConfig(BaseModel):
    """Bounded oblique alternative for an unreachable normal coarse view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distance_offset_m: float = Field(ge=-0.08, le=0.08)
    tilt_deg: float = Field(gt=0.0, le=75.0)
    azimuth_deg: float = Field(ge=-180.0, le=180.0)


class ViewPlanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ``standoff_distance_m`` is the baseline optical-centre-to-patch-centre
    # distance.  Fine planning may move an individual candidate within the
    # explicitly bounded interval below; proxy planning continues to use the
    # baseline exactly.
    standoff_distance_m: float | None = Field(default=None, gt=0.0)
    adaptive_standoff_enabled: bool = True
    minimum_standoff_distance_m: float | None = Field(default=None, gt=0.0)
    maximum_standoff_distance_m: float | None = Field(default=None, gt=0.0)
    distance_search_step_m: float = Field(default=0.01, gt=0.0)
    overlap_fraction: float = Field(default=0.3, ge=0.0, lt=0.9)
    footprint_utilization: float = Field(default=0.8, gt=0.0, le=1.0)
    image_edge_margin_px: int = Field(default=40, ge=0)
    minimum_patch_projection_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    minimum_patch_visibility_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    occlusion_depth_tolerance_m: float = Field(default=0.003, ge=0.0)
    maximum_visibility_split_depth: int = Field(default=2, ge=0, le=5)
    edge_margin_m: float = Field(default=0.005, ge=0.0)
    maximum_candidates: int = Field(default=200, ge=2, le=10000)
    coarse_reachability_fallbacks: tuple[CoarseReachabilityFallbackConfig, ...] = ()

    @model_validator(mode="after")
    def validate_adaptive_standoff(self) -> Self:
        bounds = (self.minimum_standoff_distance_m, self.maximum_standoff_distance_m)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("Adaptive standoff minimum and maximum must be configured together")
        if bounds[0] is not None and bounds[1] is not None:
            if bounds[0] > bounds[1]:
                raise ValueError("Minimum standoff distance must not exceed maximum")
            if self.standoff_distance_m is not None and not (
                bounds[0] <= self.standoff_distance_m <= bounds[1]
            ):
                raise ValueError("Baseline standoff distance must lie inside adaptive bounds")
        if self.coarse_reachability_fallbacks and (
            bounds[0] is None or bounds[1] is None or self.standoff_distance_m is None
        ):
            raise ValueError(
                "Coarse reachability fallbacks require baseline and bounded standoff distances"
            )
        if self.standoff_distance_m is not None and bounds[0] is not None and bounds[1] is not None:
            for fallback in self.coarse_reachability_fallbacks:
                distance = self.standoff_distance_m + fallback.distance_offset_m
                if not bounds[0] <= distance <= bounds[1]:
                    raise ValueError(
                        "Coarse reachability fallback leaves the bounded standoff interval"
                    )
        return self


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


class MultiViewFusionConfig(BaseModel):
    """Bounded pose-prior refinement and thin-wall-aware fusion settings."""

    model_config = ConfigDict(extra="forbid")

    voxel_size_m: float = Field(default=0.0015, gt=0.0)
    maximum_icp_points: int = Field(default=2500, ge=100, le=20000)
    icp_iterations: int = Field(default=12, ge=0, le=100)
    maximum_correspondence_distance_m: float = Field(default=0.012, gt=0.0)
    minimum_correspondences: int = Field(default=80, ge=6)
    pose_prior_weight: float = Field(default=0.25, ge=0.0)
    maximum_translation_correction_m: float = Field(default=0.008, gt=0.0)
    maximum_rotation_correction_deg: float = Field(default=1.5, gt=0.0, le=15.0)
    normal_neighbors: int = Field(default=16, ge=6, le=64)


class SurfacePartitionConfig(BaseModel):
    """Coarse-model partition parameters derived from the blade-view paper."""

    model_config = ConfigDict(extra="forbid")

    voxel_size_m: float = Field(default=0.0015, gt=0.0)
    maximum_points_per_side: int = Field(default=6000, ge=200, le=50000)
    minimum_points_per_side: int = Field(default=150, ge=30)
    normal_neighbors: int = Field(default=16, ge=6, le=64)
    angle_criterion_threshold_deg: float = Field(default=90.0, gt=0.0, lt=180.0)
    boundary_curve_enabled: bool = True
    boundary_allow_fallback: bool = True
    boundary_angular_bins: int = Field(default=180, ge=24, le=720)
    boundary_min_points_per_curve: int = Field(default=10, ge=6, le=1000)
    boundary_control_points: int = Field(default=10, ge=4, le=64)
    boundary_robust_iterations: int = Field(default=6, ge=1, le=30)
    boundary_smoothing_lambda: float = Field(default=0.1, ge=0.0)
    boundary_huber_delta_m: float = Field(default=0.003, gt=0.0)
    boundary_max_fit_rmse_m: float = Field(default=0.006, gt=0.0)
    boundary_min_inlier_fraction: float = Field(default=0.70, gt=0.0, le=1.0)
    boundary_band_fraction: float = Field(default=0.08, gt=0.0, lt=0.25)
    overlap_fraction: float = Field(default=0.30, ge=0.0, lt=0.9)
    # Production coarse-model workflows derive this from calibrated left-IR
    # intrinsics and the baseline standoff.  An explicit value remains available
    # only for synthetic/offline fixtures and controlled footprint experiments.
    derive_footprint_from_intrinsics: bool = True
    usable_footprint_m: tuple[float, float] | None = None
    minimum_patch_points: int = Field(default=24, ge=6)
    curvature_split_threshold_deg: float = Field(default=8.0, gt=0.0, le=90.0)
    maximum_adaptive_depth: int = Field(default=2, ge=0, le=5)
    normal_azimuth_bins: int = Field(default=24, ge=4, le=72)
    normal_elevation_bins: int = Field(default=12, ge=3, le=36)
    fin_mode: Literal["disabled", "optional", "required_single_per_side"] = (
        "required_single_per_side"
    )
    fin_height_fit_iterations: int = Field(default=6, ge=1, le=30)
    fin_height_huber_delta_m: float = Field(default=0.003, gt=0.0)
    fin_main_normal_min_cosine: float = Field(default=0.70, ge=0.0, le=1.0)
    fin_seed_max_normal_cosine: float = Field(default=0.55, ge=0.0, le=1.0)
    fin_seed_min_height_m: float = Field(default=0.008, gt=0.0)
    fin_grow_min_height_m: float = Field(default=0.0025, gt=0.0)
    fin_connectivity_radius_m: float = Field(default=0.006, gt=0.0)
    fin_minimum_points: int = Field(default=36, ge=12)
    fin_minimum_span_m: float = Field(default=0.012, gt=0.0)
    fin_maximum_thickness_ratio: float = Field(default=0.35, gt=0.0, lt=1.0)
    fin_maximum_secondary_fraction: float = Field(default=0.35, ge=0.0, lt=1.0)
    fin_root_band_m: float = Field(default=0.006, gt=0.0)
    fin_free_edge_band_m: float = Field(default=0.006, gt=0.0)
    fin_face_min_separation_m: float = Field(default=0.0015, gt=0.0)
    fin_root_view_main_weight: float = Field(default=0.75, gt=0.0, le=2.0)

    @field_validator("usable_footprint_m")
    @classmethod
    def validate_usable_footprint(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if not np.isfinite(value).all() or any(item <= 0.0 for item in value):
            raise ValueError("Usable surface footprint must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_fin_partition(self) -> Self:
        if self.fin_grow_min_height_m >= self.fin_seed_min_height_m:
            raise ValueError("Fin grow height must be below the seed height")
        if self.fin_seed_max_normal_cosine >= self.fin_main_normal_min_cosine:
            raise ValueError("Fin seed normal cosine must be below the main-surface cosine")
        if self.fin_connectivity_radius_m < self.voxel_size_m:
            raise ValueError("Fin connectivity radius must be at least one voxel")
        return self


class TSDFConfig(BaseModel):
    """Sparse projective TSDF settings with a protected thin-wall truncation band."""

    model_config = ConfigDict(extra="forbid")

    voxel_size_m: float = Field(default=0.0015, gt=0.0)
    truncation_distance_m: float = Field(default=0.006, gt=0.0)
    minimum_weight: float = Field(default=1.0, gt=0.0)
    maximum_voxels: int = Field(default=2_000_000, ge=1000)
    use_open3d_if_available: bool = True
    thin_wall_band_fraction: float = Field(default=0.40, gt=0.0, lt=0.5)


class SurfaceQualityConfig(BaseModel):
    """Reference-surface coverage and reconstruction-quality gates."""

    model_config = ConfigDict(extra="forbid")

    maximum_surface_distance_m: float = Field(default=0.004, gt=0.0)
    minimum_incidence_cosine: float = Field(default=0.35, ge=0.0, le=1.0)
    completed_fraction: float = Field(default=0.85, gt=0.0, le=1.0)
    maximum_rmse_m: float = Field(default=0.003, gt=0.0)
    minimum_normal_consistency: float = Field(default=0.75, ge=0.0, le=1.0)
    minimum_observed_points: int = Field(default=30, ge=3)


class FineFinalizationConfig(BaseModel):
    """Terminal gates for the immutable fine multi-view reconstruction.

    Coverage of the schema-5 reference is necessary but is deliberately not a
    completion condition.  These gates are evaluated against a newly fused fine
    cloud and its bilateral TSDF mesh before a fine run may report completion.
    """

    model_config = ConfigDict(extra="forbid")

    minimum_source_views_per_side: int = Field(default=1, ge=1, le=1000)
    minimum_mesh_triangles_per_side: int = Field(default=1, ge=1)
    maximum_mesh_boundary_edges: int = Field(default=0, ge=0)
    maximum_mesh_boundary_loops: int = Field(default=0, ge=0)
    require_watertight_mesh: Literal[True] = True
    require_two_face_fin_per_side: Literal[True] = True
    require_fin_regions_complete: Literal[True] = True


SurfaceRegionName = Literal[
    "surface",
    "leading_edge",
    "trailing_edge",
    "root",
    "tip",
    "fin_face",
    "fin_root",
    "fin_free_edge",
]


class ReacquisitionPerturbationConfig(BaseModel):
    """One deterministic retry orbit relative to a patch's nominal view."""

    model_config = ConfigDict(extra="forbid")

    distance_offset_m: float = Field(ge=-0.08, le=0.08)
    tilt_deg: float = Field(ge=0.0, le=20.0)
    azimuth_deg: float = Field(ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_nonzero_perturbation(self) -> Self:
        if self.distance_offset_m == 0.0 and self.tilt_deg == 0.0:
            raise ValueError(
                "A reacquisition perturbation must change distance or tilt"
            )
        return self


class NextViewSelectionConfig(BaseModel):
    """Versioned, coverage-first policy for the bilateral fin-blade selector."""

    model_config = ConfigDict(extra="forbid")

    required_regions: tuple[SurfaceRegionName, ...] = (
        "surface",
        "leading_edge",
        "trailing_edge",
        "root",
        "tip",
        "fin_face",
        "fin_root",
        "fin_free_edge",
    )
    region_priority: tuple[SurfaceRegionName, ...] = (
        "fin_root",
        "fin_free_edge",
        "leading_edge",
        "trailing_edge",
        "root",
        "tip",
        "fin_face",
        "surface",
    )
    require_each_region_on_both_blade_sides: bool = True
    require_two_observed_fin_faces_per_side: bool = True
    exclude_already_captured_candidate_ids: bool = True
    use_joint_travel_only_as_tiebreak: bool = True
    maximum_reacquisition_attempts_per_patch: int = Field(default=3, ge=0, le=8)
    reacquisition_perturbations: tuple[ReacquisitionPerturbationConfig, ...] = (
        ReacquisitionPerturbationConfig(
            distance_offset_m=0.0,
            tilt_deg=6.0,
            azimuth_deg=0.0,
        ),
        ReacquisitionPerturbationConfig(
            distance_offset_m=0.02,
            tilt_deg=8.0,
            azimuth_deg=120.0,
        ),
        ReacquisitionPerturbationConfig(
            distance_offset_m=-0.02,
            tilt_deg=10.0,
            azimuth_deg=-120.0,
        ),
    )

    @model_validator(mode="after")
    def validate_region_contract(self) -> Self:
        if not self.required_regions or len(set(self.required_regions)) != len(
            self.required_regions
        ):
            raise ValueError("Next-view required regions must be non-empty and unique")
        if len(set(self.region_priority)) != len(self.region_priority):
            raise ValueError("Next-view region priority must contain unique values")
        if set(self.region_priority) != set(self.required_regions):
            raise ValueError("Next-view region priority must contain exactly the required regions")
        if not self.exclude_already_captured_candidate_ids:
            raise ValueError(
                "Captured candidate IDs must remain excluded so every acquisition ID is unique"
            )
        if len(self.reacquisition_perturbations) != (
            self.maximum_reacquisition_attempts_per_patch
        ):
            raise ValueError(
                "Reacquisition perturbations must exactly match the per-patch attempt budget"
            )
        return self


class DepthComparisonConfig(BaseModel):
    """Reproducible native-versus-stereo comparison settings."""

    model_config = ConfigDict(extra="forbid")

    minimum_overlap_points: int = Field(default=100, ge=3)
    agreement_thresholds_m: tuple[float, ...] = (0.005, 0.01, 0.02)
    minimum_camera_side_offset_m: float = Field(default=0.02, gt=0.0)
    incidence_bin_edges_deg: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)

    @field_validator("agreement_thresholds_m")
    @classmethod
    def validate_agreement_thresholds(cls, value: tuple[float, ...]) -> tuple[float, ...]:
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
            or any(first >= second for first, second in zip(value, value[1:], strict=False))
        ):
            raise ValueError("Incidence bin edges must increase from 0 to 90 degrees")
        return value


class NativeOverlapValidationConfig(BaseModel):
    """Static-scene native-depth registration validation without corrective ICP."""

    model_config = ConfigDict(extra="forbid")

    minimum_views: int = Field(default=3, ge=2, le=50)
    minimum_depth_m: float = Field(default=0.25, gt=0.0)
    maximum_depth_m: float = Field(default=0.75, gt=0.0)
    pixel_stride: int = Field(default=2, ge=1, le=16)
    edge_window_radius_px: int = Field(default=1, ge=0, le=5)
    maximum_local_depth_range_m: float = Field(default=0.010, gt=0.0)
    maximum_surface_residual_m: float = Field(default=0.020, gt=0.0)
    minimum_projected_points: int = Field(default=1000, ge=3)
    minimum_surface_inlier_fraction: float = Field(default=0.95, gt=0.0, le=1.0)
    agreement_thresholds_m: tuple[float, ...] = (0.002, 0.005)
    maximum_median_absolute_error_m: float = Field(default=0.002, gt=0.0)
    maximum_root_mean_square_error_m: float = Field(default=0.003, gt=0.0)
    maximum_p95_absolute_error_m: float = Field(default=0.006, gt=0.0)
    minimum_five_mm_agreement_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    minimum_translation_span_m: float = Field(default=0.03, gt=0.0)
    minimum_rotation_span_deg: float = Field(default=5.0, gt=0.0, le=180.0)
    diagnostic_icp_enabled: bool = True
    diagnostic_icp_voxel_size_m: float = Field(default=0.005, gt=0.0)
    diagnostic_icp_maximum_points: int = Field(default=1200, ge=100, le=5000)
    diagnostic_icp_iterations: int = Field(default=8, ge=1, le=50)
    diagnostic_icp_maximum_correspondence_m: float = Field(default=0.020, gt=0.0)
    diagnostic_icp_minimum_correspondences: int = Field(default=100, ge=20)
    diagnostic_icp_normal_neighbors: int = Field(default=12, ge=4, le=64)
    diagnostic_icp_pose_prior_weight: float = Field(default=0.05, ge=0.0)
    overlay_voxel_size_m: float = Field(default=0.004, gt=0.0)
    maximum_overlay_points_per_view: int = Field(default=30000, ge=100, le=200000)

    @model_validator(mode="after")
    def validate_native_overlap(self) -> Self:
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("native-overlap maximum depth must exceed minimum depth")
        if self.maximum_local_depth_range_m >= self.maximum_surface_residual_m:
            raise ValueError(
                "native-overlap local depth range must be below the surface residual gate"
            )
        thresholds = self.agreement_thresholds_m
        if (
            not thresholds
            or not np.isfinite(thresholds).all()
            or any(value <= 0.0 for value in thresholds)
            or tuple(sorted(set(thresholds))) != thresholds
        ):
            raise ValueError("native-overlap agreement thresholds must be increasing")
        if not any(np.isclose(value, 0.005, atol=1e-12) for value in thresholds):
            raise ValueError("native-overlap agreement thresholds must include 0.005 m")
        if self.diagnostic_icp_normal_neighbors >= self.diagnostic_icp_maximum_points:
            raise ValueError("diagnostic ICP needs more points than normal neighbors")
        return self


class KinematicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: Path | None = None
    plugin_path: Path | None = None
    primary_timeout_ms: int = Field(default=1000, gt=0, le=30000)
    ik_timeout_s: float = Field(default=0.05, gt=0.0, le=5.0)
    joint_zero_offsets_rad: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


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
    """Fail-closed geometry required for offline ES68 path validation."""

    model_config = ConfigDict(extra="forbid")

    link_radii_m: tuple[float, float, float, float, float, float] | None = None
    camera_tool_radius_m: float | None = Field(default=None, gt=0.0)
    minimum_joint_positions_rad: tuple[float, float, float, float, float, float] | None = None
    maximum_joint_positions_rad: tuple[float, float, float, float, float, float] | None = None
    obstacles: tuple[CollisionObstacleConfig, ...] = ()
    require_obstacles: bool = False
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
                or any(lower >= upper for lower, upper in zip(*limits, strict=True))
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
    maximum_endpoint_translation_error_m: float = Field(default=0.002, gt=0.0)
    maximum_endpoint_rotation_error_deg: float = Field(default=0.3, gt=0.0, le=180.0)
    motion_envelope_acceptance_path: Path | None = None
    motion_envelope_acceptance_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_motion_envelope_binding(self) -> Self:
        if (self.motion_envelope_acceptance_path is None) != (
            self.motion_envelope_acceptance_id is None
        ):
            raise ValueError(
                "Motion-envelope acceptance path and identity must be configured together"
            )
        return self


class ScienceAcceptanceConfig(BaseModel):
    """Immutable physical acceptance asset for the geometry-science pipeline."""

    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    acceptance_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_acceptance_binding(self) -> Self:
        if (self.path is None) != (self.acceptance_id is None):
            raise ValueError(
                "Science acceptance path and identity must be configured together"
            )
        return self


class CoarseScienceConfig(BaseModel):
    """Completion and conservative fin-discovery policy for unknown blades."""

    model_config = ConfigDict(extra="forbid")

    discovery_tilt_deg: float = Field(default=15.0, gt=0.0, lt=45.0)
    minimum_total_views: int = Field(default=6, ge=4)
    minimum_views_per_side: int = Field(default=3, ge=2)
    maximum_attempts_per_candidate: int = Field(default=2, ge=1)
    require_complete_proxy_coverage: bool = True
    maximum_discovery_translation_error_m: float = Field(default=0.020, gt=0.0)
    maximum_discovery_rotation_error_deg: float = Field(
        default=5.0,
        gt=0.0,
        le=30.0,
    )

    @model_validator(mode="after")
    def validate_bilateral_view_gate(self) -> Self:
        if self.minimum_total_views < 2 * self.minimum_views_per_side:
            raise ValueError("Total coarse-view gate is below the per-side requirement")
        return self


class StopAndCaptureConfig(BaseModel):
    """Fail-closed receding-horizon motion/perception coordination settings.

    The feature remains disabled until the workcell-specific segment bound has been
    measured and configured.  Native RealSense depth is deliberately not selectable:
    this coordinator has one scientific/safety depth backend, FoundationStereo.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    depth_backend: Literal["foundation_stereo"] = "foundation_stereo"
    bootstrap_mode: Literal["operator_guided"] = "operator_guided"
    maximum_segment_joint_delta_rad: float | None = Field(
        default=None,
        gt=0.0,
        le=0.5,
    )
    settle_timeout_s: float = Field(default=15.0, gt=0.0, le=300.0)
    settle_poll_period_s: float = Field(default=0.05, gt=0.0, le=1.0)
    maximum_robot_state_staleness_s: float = Field(
        default=0.25,
        gt=0.0,
        le=5.0,
    )
    maximum_goal_joint_error_rad: float = Field(default=0.01, gt=0.0, le=0.2)
    execution_freshness_margin_s: float = Field(default=1.0, ge=0.0, le=60.0)
    # These budgets are hardware measurements, not guessed software defaults.
    # The unknown-blade production entry remains offline-blocked until all four
    # have been measured for the deployed GPU/controller/workcell.
    maximum_perception_cycle_duration_s: float | None = Field(
        default=None,
        gt=0.0,
        le=3600.0,
    )
    maximum_operator_reposition_interval_s: float | None = Field(
        default=None,
        gt=0.0,
        le=3600.0,
    )
    maximum_segment_execution_duration_s: float | None = Field(
        default=None,
        gt=0.0,
        le=300.0,
    )
    maximum_schema5_handoff_duration_s: float | None = Field(default=None, gt=0.0)
    runtime_timing_acceptance_path: Path | None = None
    runtime_timing_acceptance_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    require_operator_approval: Literal[True] = True
    require_capture_after_every_segment: Literal[True] = True

    @model_validator(mode="after")
    def validate_enabled_contract(self) -> Self:
        if (self.runtime_timing_acceptance_path is None) != (
            self.runtime_timing_acceptance_id is None
        ):
            raise ValueError(
                "Runtime-timing acceptance path and identity must be configured together"
            )
        if self.enabled and self.maximum_segment_joint_delta_rad is None:
            raise ValueError(
                "Enabled stop-and-capture coordination requires a measured "
                "maximum_segment_joint_delta_rad"
            )
        if self.settle_poll_period_s > self.settle_timeout_s:
            raise ValueError("settle_poll_period_s must not exceed settle_timeout_s")
        if self.maximum_robot_state_staleness_s < self.settle_poll_period_s:
            raise ValueError(
                "maximum_robot_state_staleness_s must be at least settle_poll_period_s"
            )
        return self


class OccupancyConfig(BaseModel):
    """Fail-closed online environment mapping for the unknown blade workcell."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    frame_id: Literal["base"] = "base"
    mapping_mode: Literal["stop_and_capture"] = "stop_and_capture"
    voxel_size_m: float = Field(default=0.01, gt=0.0, le=0.05)
    workspace_bounds_min_m: tuple[float, float, float] | None = None
    workspace_bounds_max_m: tuple[float, float, float] | None = None
    minimum_depth_m: float = Field(default=0.15, gt=0.0)
    maximum_depth_m: float = Field(default=1.5, gt=0.0)
    integration_stride: int = Field(default=2, ge=1, le=16)
    free_space_margin_m: float = Field(default=0.01, ge=0.0, le=0.10)
    obstacle_inflation_m: float = Field(default=0.01, ge=0.0, le=0.20)
    maximum_map_age_s: float = Field(default=5.0, gt=0.0, le=300.0)
    unknown_policy: Literal["block"] = "block"
    require_robot_self_mask: bool = True
    # Optional hardware-accepted static free volumes solve the otherwise
    # unobservable robot-base/self-volume bootstrap without weakening UNKNOWN
    # elsewhere.  The acceptance ID is the SHA-256 identity of the separately
    # archived workcell acceptance record; empty-by-default keeps this feature off.
    accepted_static_free_aabbs: tuple[AxisAlignedBoxConfig, ...] = ()
    accepted_static_free_acceptance_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    accepted_static_free_acceptance_path: Path | None = None
    self_mask_front_tolerance_m: float = Field(default=0.01, ge=0.0, le=0.10)
    self_mask_back_tolerance_m: float = Field(default=0.02, ge=0.0, le=0.20)
    self_mask_dilation_px: int = Field(default=1, ge=0, le=20)
    maximum_fk_tcp_translation_error_m: float = Field(default=0.002, gt=0.0)
    maximum_fk_tcp_rotation_error_deg: float = Field(
        default=0.3,
        gt=0.0,
        le=180.0,
    )
    minimum_valid_depth_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_stereo_confidence: float = Field(default=0.5, gt=0.0, le=1.0)
    maximum_lr_consistency_error_px: float = Field(default=1.0, gt=0.0, le=10.0)
    minimum_source_views: int = Field(default=3, ge=3, le=10000)
    minimum_free_observations: int = Field(default=3, ge=2, le=10000)
    minimum_free_view_translation_m: float = Field(default=0.02, gt=0.0, le=1.0)
    minimum_free_view_direction_deg: float = Field(default=5.0, gt=0.0, le=180.0)
    maximum_grid_voxels: int = Field(default=8_000_000, ge=1000)

    @model_validator(mode="after")
    def validate_occupancy_contract(self) -> Self:
        bounds = (self.workspace_bounds_min_m, self.workspace_bounds_max_m)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("Occupancy workspace minimum and maximum must be configured together")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("Occupancy maximum depth must exceed minimum depth")
        if self.free_space_margin_m >= self.maximum_depth_m:
            raise ValueError("Occupancy free-space margin must be below maximum depth")
        if bounds[0] is not None and bounds[1] is not None:
            values = (*bounds[0], *bounds[1])
            if not np.isfinite(values).all() or any(
                lower >= upper for lower, upper in zip(*bounds, strict=True)
            ):
                raise ValueError("Occupancy workspace bounds must be finite and ordered")
            extents = np.asarray(bounds[1]) - np.asarray(bounds[0])
            grid_shape = np.ceil(extents / self.voxel_size_m).astype(np.int64)
            if int(np.prod(grid_shape)) > self.maximum_grid_voxels:
                raise ValueError("Occupancy workspace exceeds maximum_grid_voxels")
        if self.enabled and bounds[0] is None:
            raise ValueError("Enabled occupancy mapping requires measured workspace bounds")
        if self.enabled and not self.require_robot_self_mask:
            raise ValueError("Physical occupancy mapping requires robot self masking")
        accepted = self.accepted_static_free_aabbs
        acceptance_fields_present = (
            self.accepted_static_free_acceptance_id is not None
            and self.accepted_static_free_acceptance_path is not None
        )
        if bool(accepted) != acceptance_fields_present or (
            (self.accepted_static_free_acceptance_id is None)
            != (self.accepted_static_free_acceptance_path is None)
        ):
            raise ValueError(
                "Accepted static-free AABBs, acceptance ID, and immutable acceptance "
                "asset path must be configured together"
            )
        names = tuple(item.name for item in accepted)
        if len(names) != len(set(names)):
            raise ValueError("Accepted static-free AABB names must be unique")
        if accepted and bounds[0] is None:
            raise ValueError(
                "Accepted static-free AABBs require configured occupancy workspace bounds"
            )
        if accepted and bounds[0] is not None and bounds[1] is not None:
            for volume in accepted:
                if any(
                    lower < workspace_lower or upper > workspace_upper
                    for lower, upper, workspace_lower, workspace_upper in zip(
                        volume.minimum_m,
                        volume.maximum_m,
                        bounds[0],
                        bounds[1],
                        strict=True,
                    )
                ):
                    raise ValueError(
                        "Accepted static-free AABBs must lie inside the occupancy workspace"
                    )
        return self


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
    bootstrap_foreground: BootstrapForegroundSettings = Field(
        default_factory=BootstrapForegroundSettings
    )
    blade_foreground: BladeForegroundConfig = Field(default_factory=BladeForegroundConfig)
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)
    view_planning: ViewPlanningConfig = Field(default_factory=ViewPlanningConfig)
    view_filter: ViewFilterConfig = Field(default_factory=ViewFilterConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    multi_view_fusion: MultiViewFusionConfig = Field(default_factory=MultiViewFusionConfig)
    surface_partition: SurfacePartitionConfig = Field(default_factory=SurfacePartitionConfig)
    tsdf: TSDFConfig = Field(default_factory=TSDFConfig)
    surface_quality: SurfaceQualityConfig = Field(default_factory=SurfaceQualityConfig)
    fine_finalization: FineFinalizationConfig = Field(
        default_factory=FineFinalizationConfig
    )
    next_view_selection: NextViewSelectionConfig = Field(default_factory=NextViewSelectionConfig)
    depth_comparison: DepthComparisonConfig = Field(default_factory=DepthComparisonConfig)
    native_overlap_validation: NativeOverlapValidationConfig = Field(
        default_factory=NativeOverlapValidationConfig
    )
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    collision: CollisionConfig = Field(default_factory=CollisionConfig)
    motion_preflight: MotionPreflightConfig = Field(default_factory=MotionPreflightConfig)
    science_acceptance: ScienceAcceptanceConfig = Field(
        default_factory=ScienceAcceptanceConfig
    )
    coarse_science: CoarseScienceConfig = Field(default_factory=CoarseScienceConfig)
    stop_and_capture: StopAndCaptureConfig = Field(default_factory=StopAndCaptureConfig)
    occupancy: OccupancyConfig = Field(default_factory=OccupancyConfig)

    @model_validator(mode="after")
    def validate_stop_and_capture_dependencies(self) -> Self:
        if self.stop_and_capture.enabled and not self.occupancy.enabled:
            raise ValueError("Enabled stop-and-capture coordination requires occupancy mapping")
        if self.stop_and_capture.enabled and self.robot.model != "es68":
            raise ValueError("Enabled stop-and-capture coordination requires robot.model='es68'")
        if self.stop_and_capture.enabled and not self.robot.motion_enabled:
            raise ValueError(
                "Enabled stop-and-capture coordination requires "
                "robot.motion_enabled=true for its explicit stop boundary"
            )
        if self.stop_and_capture.enabled and self.robot.settle_time_s <= 0.0:
            raise ValueError(
                "Enabled stop-and-capture coordination requires robot.settle_time_s to be positive"
            )
        minimum_settle_timeout_s = (
            self.robot.settle_time_s + self.stop_and_capture.settle_poll_period_s
        )
        if (
            self.stop_and_capture.enabled
            and self.stop_and_capture.settle_timeout_s < minimum_settle_timeout_s
        ):
            raise ValueError(
                "Enabled stop-and-capture settle_timeout_s must be at least "
                "robot.settle_time_s + settle_poll_period_s"
            )
        if self.stop_and_capture.enabled and not np.isclose(
            self.robot.servoj_time_s,
            self.motion_preflight.servoj_dt_s,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Enabled stop-and-capture requires robot.servoj_time_s to equal "
                "motion_preflight.servoj_dt_s"
            )
        lr_threshold = self.foundation_stereo.left_right_consistency_threshold_px
        if self.stop_and_capture.enabled and (
            lr_threshold is None or lr_threshold > self.occupancy.maximum_lr_consistency_error_px
        ):
            raise ValueError(
                "Enabled stop-and-capture requires a FoundationStereo left-right "
                "consistency threshold no larger than the occupancy threshold"
            )
        return self


def load_settings(path: str | Path) -> AppSettings:
    """Load and validate a YAML settings file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return AppSettings.model_validate(raw)
