from pathlib import Path

import pytest
from pydantic import ValidationError

from biblade_fusion.core.settings import (
    AppSettings,
    BladeForegroundConfig,
    OccupancyConfig,
    RobotConfig,
    StopAndCaptureConfig,
    SurfacePartitionConfig,
    load_settings,
)


def test_default_settings_load_safely() -> None:
    settings = load_settings(Path("configs/default.yaml"))

    assert settings.robot.model == "es68"
    assert settings.robot.robot_ip == "192.168.6.60"
    assert settings.robot.local_ip == "192.168.6.29"
    assert settings.robot.motion_enabled is False
    assert settings.robot.headless_mode is True
    assert settings.robot.rtsi_frequency_hz == 125.0
    assert settings.robot.servoj_lookahead_time_s == 0.1
    assert settings.thermal.enabled is False
    assert settings.realsense.infrared_width == 1280
    assert settings.acquisition.max_bracket_ms == 250.0
    assert settings.foundation_stereo.device == "cuda"
    assert settings.foundation_stereo.valid_iterations == 32
    assert settings.foundation_stereo.left_right_consistency_threshold_px == 1.0
    assert settings.stereo_rectification.alpha == 0.0
    assert settings.proxy_model.estimated_thickness_m is None
    assert settings.collision.require_obstacles is False
    assert settings.occupancy.minimum_source_views == 3
    assert settings.occupancy.minimum_free_observations == 3
    assert settings.occupancy.minimum_free_view_translation_m == 0.02
    assert settings.occupancy.minimum_free_view_direction_deg == 5.0
    assert settings.proxy_model.tangential_margin_m == 0.01
    assert settings.point_cloud.minimum_depth_m == 0.15
    assert settings.point_cloud.maximum_depth_m == 1.5
    assert settings.blade_foreground.enabled is False
    assert settings.blade_foreground.method == "reference_projected"
    assert settings.blade_foreground.projection_radius_px == 3
    assert settings.blade_foreground.minimum_target_incidence_cosine == 0.10
    assert settings.hand_eye.calibration_path == Path(
        "data/calibrations/es68_left_ir_hand_eye_active.yaml"
    )
    assert settings.hand_eye.initial_method == "park"
    assert settings.hand_eye.validation_minimum_samples == 5
    assert settings.native_overlap_validation.minimum_views == 3
    assert settings.native_overlap_validation.maximum_p95_absolute_error_m == 0.006
    assert settings.native_overlap_validation.diagnostic_icp_enabled is True
    assert settings.view_planning.standoff_distance_m is None
    assert settings.view_planning.adaptive_standoff_enabled is True
    assert settings.view_planning.minimum_standoff_distance_m is None
    assert settings.view_planning.maximum_standoff_distance_m is None
    assert settings.view_planning.overlap_fraction == 0.3
    assert settings.view_filter.workspace is None
    assert settings.view_filter.camera_clearance_radius_m == 0.05
    assert settings.view_filter.minimum_incidence_cosine == 0.95
    assert settings.kinematics.model_path is None
    assert settings.kinematics.ik_timeout_s == 0.05
    assert settings.motion_preflight.servoj_dt_s == 0.004
    assert settings.motion_preflight.speed_scaling == 0.08
    assert settings.stop_and_capture.enabled is False
    assert settings.stop_and_capture.depth_backend == "foundation_stereo"
    assert settings.stop_and_capture.maximum_segment_joint_delta_rad is None
    assert settings.stop_and_capture.require_operator_approval is True
    assert settings.stop_and_capture.require_capture_after_every_segment is True
    assert settings.stop_and_capture.maximum_robot_state_staleness_s == 0.25
    assert settings.surface_partition.fin_mode == "required_single_per_side"
    assert settings.surface_partition.derive_footprint_from_intrinsics is True
    assert settings.surface_partition.usable_footprint_m is None
    assert settings.occupancy.enabled is False
    assert settings.occupancy.unknown_policy == "block"
    assert settings.occupancy.mapping_mode == "stop_and_capture"
    assert settings.occupancy.workspace_bounds_min_m is None


def test_unknown_configuration_key_is_rejected() -> None:
    raw = {
        "project": {"data_root": "data", "log_root": "logs"},
        "robot": {
            "model": "cs68",
            "sdk_wheel": "/tmp/elite.whl",
            "unexpected": True,
        },
        "realsense": {},
        "thermal": {},
    }

    with pytest.raises(ValidationError, match="unexpected"):
        AppSettings.model_validate(raw)


def test_invalid_robot_ip_is_rejected() -> None:
    raw = {
        "project": {},
        "robot": {"robot_ip": "not-an-ip", "sdk_wheel": "/tmp/elite.whl"},
        "realsense": {},
        "thermal": {},
    }

    with pytest.raises(ValidationError, match="robot_ip"):
        AppSettings.model_validate(raw)


def test_fin_growth_gate_must_be_below_seed_gate() -> None:
    with pytest.raises(ValidationError, match="Fin grow height"):
        SurfacePartitionConfig(
            fin_grow_min_height_m=0.009,
            fin_seed_min_height_m=0.008,
        )


def test_blade_foreground_ranges_fail_closed() -> None:
    with pytest.raises(ValidationError, match="projection depth"):
        BladeForegroundConfig(
            minimum_projection_depth_m=1.0,
            maximum_projection_depth_m=0.5,
        )
    with pytest.raises(ValidationError, match="mask fraction"):
        BladeForegroundConfig(
            minimum_mask_fraction=0.8,
            maximum_mask_fraction=0.2,
        )


def test_adaptive_standoff_bounds_must_be_complete_and_contain_baseline() -> None:
    from biblade_fusion.core.settings import ViewPlanningConfig

    with pytest.raises(ValidationError, match="configured together"):
        ViewPlanningConfig(
            standoff_distance_m=0.20,
            minimum_standoff_distance_m=0.15,
        )
    with pytest.raises(ValidationError, match="inside adaptive bounds"):
        ViewPlanningConfig(
            standoff_distance_m=0.20,
            minimum_standoff_distance_m=0.21,
            maximum_standoff_distance_m=0.30,
        )


def test_enabled_occupancy_requires_measured_bounds_and_self_mask() -> None:
    with pytest.raises(ValidationError, match="requires measured workspace bounds"):
        OccupancyConfig(enabled=True)
    with pytest.raises(ValidationError, match="requires robot self masking"):
        OccupancyConfig(
            enabled=True,
            workspace_bounds_min_m=(-1.0, -1.0, 0.0),
            workspace_bounds_max_m=(1.0, 1.0, 1.0),
            require_robot_self_mask=False,
        )


def test_occupancy_grid_budget_is_checked() -> None:
    with pytest.raises(ValidationError, match="maximum_grid_voxels"):
        OccupancyConfig(
            enabled=True,
            workspace_bounds_min_m=(-1.0, -1.0, 0.0),
            workspace_bounds_max_m=(1.0, 1.0, 1.0),
            voxel_size_m=0.001,
            maximum_grid_voxels=1_000_000,
        )


def test_occupancy_requires_at_least_three_source_views() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 3"):
        OccupancyConfig(minimum_source_views=2)


def test_occupancy_requires_multiple_free_observations() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        OccupancyConfig(minimum_free_observations=1)


def test_enabled_stop_and_capture_requires_measured_short_segment_bound() -> None:
    with pytest.raises(ValidationError, match="maximum_segment_joint_delta_rad"):
        StopAndCaptureConfig(enabled=True)


def test_enabled_stop_and_capture_requires_enabled_occupancy() -> None:
    raw = {
        "project": {},
        "robot": {"sdk_wheel": "/tmp/elite.whl"},
        "realsense": {},
        "thermal": {},
        "stop_and_capture": {
            "enabled": True,
            "maximum_segment_joint_delta_rad": 0.05,
        },
        "occupancy": {"enabled": False},
    }

    with pytest.raises(ValidationError, match="requires occupancy mapping"):
        AppSettings.model_validate(raw)


def _enabled_stop_and_capture_settings() -> dict[str, object]:
    return {
        "project": {},
        "robot": {
            "sdk_wheel": "/tmp/elite.whl",
            "settle_time_s": 1.0,
            "motion_enabled": True,
        },
        "realsense": {},
        "thermal": {},
        "stop_and_capture": {
            "enabled": True,
            "maximum_segment_joint_delta_rad": 0.05,
            "settle_timeout_s": 2.0,
            "settle_poll_period_s": 0.05,
            "maximum_robot_state_staleness_s": 0.25,
        },
        "occupancy": {
            "enabled": True,
            "workspace_bounds_min_m": (-0.2, -0.2, -0.2),
            "workspace_bounds_max_m": (0.2, 0.2, 0.2),
        },
    }


def test_enabled_stop_and_capture_requires_positive_robot_settle_time() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["robot"] = {
        "sdk_wheel": "/tmp/elite.whl",
        "settle_time_s": 0.0,
        "motion_enabled": True,
    }

    with pytest.raises(ValidationError, match=r"robot\.settle_time_s.*positive"):
        AppSettings.model_validate(raw)


def test_settle_timeout_must_cover_settle_window_and_one_poll() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["stop_and_capture"] = {
        "enabled": True,
        "maximum_segment_joint_delta_rad": 0.05,
        "settle_timeout_s": 1.049,
        "settle_poll_period_s": 0.05,
        "maximum_robot_state_staleness_s": 0.25,
    }

    with pytest.raises(
        ValidationError,
        match=r"settle_timeout_s.*settle_time_s \+ settle_poll_period_s",
    ):
        AppSettings.model_validate(raw)


def test_enabled_stop_scan_requires_one_servoj_period_contract() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["robot"] = {
        "sdk_wheel": "/tmp/elite.whl",
        "settle_time_s": 1.0,
        "servoj_time_s": 0.008,
        "motion_enabled": True,
    }

    with pytest.raises(
        ValidationError,
        match=r"robot\.servoj_time_s.*motion_preflight\.servoj_dt_s",
    ):
        AppSettings.model_validate(raw)


@pytest.mark.parametrize("lr_threshold", [None, 2.0])
def test_enabled_stop_scan_requires_occupancy_compatible_lr_threshold(
    lr_threshold: float | None,
) -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["foundation_stereo"] = {
        "left_right_consistency_threshold_px": lr_threshold,
    }

    with pytest.raises(
        ValidationError,
        match="FoundationStereo left-right consistency threshold",
    ):
        AppSettings.model_validate(raw)


def test_nested_model_construct_cannot_bypass_enabled_settle_contract() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["robot"] = RobotConfig.model_construct(
        sdk_wheel=Path("/tmp/elite.whl"),
        settle_time_s=0.0,
        motion_enabled=True,
    )

    with pytest.raises(ValidationError, match=r"robot\.settle_time_s.*positive"):
        AppSettings.model_validate(raw)


def test_nested_stop_config_cannot_bypass_settle_timeout_contract() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["stop_and_capture"] = StopAndCaptureConfig.model_construct(
        enabled=True,
        maximum_segment_joint_delta_rad=0.05,
        settle_timeout_s=1.0,
        settle_poll_period_s=0.05,
        maximum_robot_state_staleness_s=0.25,
    )

    with pytest.raises(
        ValidationError,
        match=r"settle_timeout_s.*settle_time_s \+ settle_poll_period_s",
    ):
        AppSettings.model_validate(raw)


def test_robot_state_staleness_must_cover_the_poll_period() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_robot_state_staleness_s",
    ):
        StopAndCaptureConfig(
            settle_poll_period_s=0.2,
            maximum_robot_state_staleness_s=0.1,
        )


def test_enabled_stop_scan_requires_motion_driver_boundary() -> None:
    raw = _enabled_stop_and_capture_settings()
    raw["robot"] = {
        "sdk_wheel": "/tmp/elite.whl",
        "settle_time_s": 1.0,
        "motion_enabled": False,
    }

    with pytest.raises(ValidationError, match="robot.motion_enabled=true"):
        AppSettings.model_validate(raw)


def test_free_vote_threshold_is_independent_from_source_view_readiness() -> None:
    config = OccupancyConfig(
        minimum_source_views=5,
        minimum_free_observations=2,
    )

    assert config.minimum_source_views == 5
    assert config.minimum_free_observations == 2


@pytest.mark.parametrize(
    "field",
    ["minimum_free_view_translation_m", "minimum_free_view_direction_deg"],
)
def test_free_view_independence_thresholds_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        OccupancyConfig(**{field: 0.0})
