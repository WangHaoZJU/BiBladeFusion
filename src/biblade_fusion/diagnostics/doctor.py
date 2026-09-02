"""Safe, non-moving environment diagnostics."""

from __future__ import annotations

import math
import platform
import re
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from biblade_fusion.calibration import (
    HandEyeCalibrationError,
    RobotKinematicsError,
    load_cs68_kinematics,
    load_hand_eye_calibration,
)
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.devices.thermal_camera import (
    ThermalSdkKind,
    audit_tsr605_usb_sdk,
)
from biblade_fusion.diagnostics.types import CheckLevel, CheckResult
from biblade_fusion.mapping.occupancy import OccupancyState
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    Es68D435iCollisionTemplate,
    es68_d435i_collision_content_hash,
    es68_d435i_robot_geometry_hash,
)
from biblade_fusion.storage.runtime_timing_acceptance import (
    read_runtime_timing_acceptance,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def _check_python() -> CheckResult:
    current = sys.version_info[:3]
    level = CheckLevel.PASS if current[:2] == (3, 12) else CheckLevel.FAIL
    return CheckResult(
        "python",
        level,
        platform.python_version(),
        {"executable": sys.executable, "required": ">=3.12,<3.13"},
    )


def _check_elite_sdk(settings: AppSettings) -> CheckResult:
    try:
        module = import_module("elite_cs_sdk")
    except Exception as exc:
        return CheckResult("elite_sdk", CheckLevel.FAIL, f"import failed: {exc}")

    wheel_exists = settings.robot.sdk_wheel.is_file()
    level = CheckLevel.PASS if wheel_exists else CheckLevel.WARN
    message = f"elite-cs-sdk {_package_version('elite-cs-sdk')} imported"
    return CheckResult(
        "elite_sdk",
        level,
        message,
        {
            "module": str(getattr(module, "__file__", "unknown")),
            "wheel": str(settings.robot.sdk_wheel),
            "wheel_exists": wheel_exists,
        },
    )


def _check_realsense() -> CheckResult:
    try:
        rs = import_module("pyrealsense2")
    except Exception as exc:
        return CheckResult("realsense", CheckLevel.FAIL, f"SDK import failed: {exc}")

    try:
        devices = list(rs.context().query_devices())
    except Exception as exc:
        return CheckResult(
            "realsense",
            CheckLevel.WARN,
            f"SDK imported; device enumeration unavailable: {exc}",
            {"version": _package_version("pyrealsense2")},
        )

    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    if not serials:
        return CheckResult(
            "realsense",
            CheckLevel.WARN,
            "SDK imported; no camera detected",
            {"version": _package_version("pyrealsense2"), "serials": []},
        )
    return CheckResult(
        "realsense",
        CheckLevel.PASS,
        f"detected {len(serials)} camera(s)",
        {"version": _package_version("pyrealsense2"), "serials": serials},
    )


def _check_robot_configuration(settings: AppSettings) -> CheckResult:
    if settings.robot.motion_enabled:
        return CheckResult(
            "robot_safety",
            CheckLevel.WARN,
            "motion is enabled in configuration",
        )
    return CheckResult(
        "robot_safety",
        CheckLevel.PASS,
        "motion is disabled",
    )


def _check_robot_address(settings: AppSettings) -> CheckResult:
    if settings.robot.robot_ip is None:
        return CheckResult(
            "robot_address",
            CheckLevel.WARN,
            "robot IP is not configured; no connection attempted",
        )
    return CheckResult(
        "robot_address",
        CheckLevel.PASS,
        f"configured as {settings.robot.robot_ip}; no connection attempted",
    )


def _check_thermal(settings: AppSettings) -> CheckResult:
    audit = audit_tsr605_usb_sdk(settings.thermal.sdk_root)
    details = {
        **audit.as_dict(),
        "enabled": settings.thermal.enabled,
        "driver": settings.thermal.driver,
        "model": settings.thermal.model,
        "transport": settings.thermal.transport,
        "serial_number_configured": settings.thermal.serial_number is not None,
        "expected_shape": (
            [settings.thermal.expected_height, settings.thermal.expected_width]
            if settings.thermal.expected_width is not None
            and settings.thermal.expected_height is not None
            else None
        ),
        "unknown_blade_motion_scope": "blocked",
    }
    if not settings.thermal.enabled:
        if audit.kind is ThermalSdkKind.HIKVISION_DEVICE_NETWORK:
            return CheckResult(
                "thermal_camera",
                CheckLevel.WARN,
                "disabled safely; configured HCNetSDK is not a verified TSR605 USB binding",
                details,
            )
        return CheckResult(
            "thermal_camera",
            CheckLevel.PASS,
            "disabled; TSR605 capture and thermal fusion remain closed",
            details,
        )
    if settings.thermal.driver is None:
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            "enabled without a configured driver",
            details,
        )
    if settings.thermal.driver != "tsr605_usb":
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            f"unsupported thermal driver {settings.thermal.driver!r}",
            details,
        )
    if settings.thermal.model.strip().casefold() != "tsr605":
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            "the reviewed USB adapter boundary is restricted to model TSR605",
            details,
        )
    if settings.thermal.serial_number is None:
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            "enabled TSR605 capture requires a pinned device serial number",
            details,
        )
    if audit.kind is ThermalSdkKind.HIKVISION_DEVICE_NETWORK:
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            audit.reason,
            details,
        )
    if not audit.compatible:
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            f"{audit.reason}; no reviewed native TSR605 USB backend is bundled",
            details,
        )
    return CheckResult(
        "thermal_camera",
        CheckLevel.FAIL,
        "SDK audit passed but live TSR605 USB probing is not implemented",
        details,
    )


def _check_hand_eye(settings: AppSettings) -> CheckResult:
    if settings.hand_eye.calibration_path is None:
        return CheckResult(
            "hand_eye",
            CheckLevel.WARN,
            "not configured; base-frame reconstruction and view planning are unavailable",
        )
    try:
        calibration = load_hand_eye_calibration(settings.hand_eye)
    except HandEyeCalibrationError as exc:
        return CheckResult("hand_eye", CheckLevel.FAIL, str(exc))
    return CheckResult(
        "hand_eye",
        CheckLevel.PASS,
        f"validated {calibration.method} calibration",
        {
            "path": str(calibration.source_path),
            "sample_count": calibration.sample_count,
            "translation_rmse_m": calibration.translation_rmse_m,
            "rotation_rmse_deg": calibration.rotation_rmse_deg,
        },
    )


def _check_kinematics(settings: AppSettings) -> CheckResult:
    path = settings.kinematics.model_path
    if path is None:
        return CheckResult(
            "es68_controller_kinematics",
            CheckLevel.WARN,
            "controller MDH artifact is not configured; offline IK is unavailable",
        )
    try:
        model = load_cs68_kinematics(path)
    except RobotKinematicsError as exc:
        return CheckResult("es68_controller_kinematics", CheckLevel.FAIL, str(exc))
    return CheckResult(
        "es68_controller_kinematics",
        CheckLevel.PASS,
        "validated controller-specific ES68 MDH artifact",
        {"path": str(path), "source": model.source},
    )


def _check_collision_configuration(settings: AppSettings) -> CheckResult:
    config = settings.collision
    missing = []
    if config.link_radii_m is None:
        missing.append("link_radii_m")
    if config.camera_tool_radius_m is None:
        missing.append("camera_tool_radius_m")
    if config.minimum_joint_positions_rad is None:
        missing.append("minimum_joint_positions_rad")
    if config.maximum_joint_positions_rad is None:
        missing.append("maximum_joint_positions_rad")
    if config.require_obstacles and not config.obstacles:
        missing.append("obstacles")
    if missing:
        return CheckResult(
            "collision_geometry",
            CheckLevel.WARN,
            "offline path validation is unavailable; collision geometry is incomplete",
            {"missing": missing},
        )
    return CheckResult(
        "collision_geometry",
        CheckLevel.PASS,
        "collision geometry and joint limits are configured",
        {
            "obstacle_count": len(config.obstacles),
            "minimum_clearance_m": config.minimum_clearance_m,
            "maximum_joint_step_rad": config.maximum_joint_step_rad,
            "motion_authorized": False,
        },
    )


def _check_occupancy_configuration(settings: AppSettings) -> CheckResult:
    """Check the fail-closed three-state map contract without reading a live map."""

    config = settings.occupancy
    semantic_states = tuple(state.value for state in OccupancyState)
    bounds_min = config.workspace_bounds_min_m
    bounds_max = config.workspace_bounds_max_m
    bounds_configured = bounds_min is not None and bounds_max is not None
    details: dict[str, object] = {
        "enabled": config.enabled,
        "frame_id": config.frame_id,
        "mapping_mode": config.mapping_mode,
        "semantic_states": list(semantic_states),
        "unknown_policy": config.unknown_policy,
        "unknown_blocks_motion": config.unknown_policy == "block",
        "workspace_bounds_configured": bounds_configured,
        "workspace_bounds_min_m": list(bounds_min) if bounds_min is not None else None,
        "workspace_bounds_max_m": list(bounds_max) if bounds_max is not None else None,
        "voxel_size_m": config.voxel_size_m,
        "minimum_free_observations": config.minimum_free_observations,
        "motion_authorized": False,
        "hardware_connection_attempted": False,
    }
    contract_errors = []
    if semantic_states != ("free", "occupied", "unknown"):
        contract_errors.append("semantic_states")
    if config.frame_id != "base":
        contract_errors.append("frame_id")
    if config.mapping_mode != "stop_and_capture":
        contract_errors.append("mapping_mode")
    if config.unknown_policy != "block":
        contract_errors.append("unknown_policy")
    if not config.require_robot_self_mask:
        contract_errors.append("require_robot_self_mask")
    if contract_errors:
        details["invalid"] = contract_errors
        return CheckResult(
            "occupancy_configuration",
            CheckLevel.FAIL,
            "occupancy safety contract is not fail-closed",
            details,
        )
    if not bounds_configured:
        details["missing"] = ["workspace_bounds_min_m", "workspace_bounds_max_m"]
        return CheckResult(
            "occupancy_configuration",
            CheckLevel.WARN,
            "three-state occupancy is fail-closed, but measured workspace bounds are missing",
            details,
        )

    assert bounds_min is not None and bounds_max is not None
    bounds_are_valid = all(
        math.isfinite(lower) and math.isfinite(upper) and lower < upper
        for lower, upper in zip(bounds_min, bounds_max, strict=True)
    )
    details["workspace_bounds_valid"] = bounds_are_valid
    if not bounds_are_valid:
        return CheckResult(
            "occupancy_configuration",
            CheckLevel.FAIL,
            "occupancy workspace bounds must be finite and strictly ordered",
            details,
        )
    grid_shape = tuple(
        int(math.ceil((upper - lower) / config.voxel_size_m))
        for lower, upper in zip(bounds_min, bounds_max, strict=True)
    )
    grid_voxels = math.prod(grid_shape)
    details.update(
        {
            "grid_shape": list(grid_shape),
            "grid_voxels": grid_voxels,
            "maximum_grid_voxels": config.maximum_grid_voxels,
        }
    )
    if grid_voxels <= 0 or grid_voxels > config.maximum_grid_voxels:
        return CheckResult(
            "occupancy_configuration",
            CheckLevel.FAIL,
            "occupancy workspace cannot be represented within the configured voxel limit",
            details,
        )
    if not config.enabled:
        return CheckResult(
            "occupancy_configuration",
            CheckLevel.WARN,
            "three-state occupancy workspace is configured but mapping is disabled",
            details,
        )
    return CheckResult(
        "occupancy_configuration",
        CheckLevel.PASS,
        "three-state stop-and-capture occupancy is configured; UNKNOWN blocks",
        details,
    )


def _check_final_collision_model(
    settings: AppSettings,
    resources: Es68D435iCollisionResources | None = None,
) -> CheckResult:
    """Validate and hash the local final ES68+D435i assets without loading FCL."""

    resolved = resources or Es68D435iCollisionResources.packaged_template()
    manifest_path = resolved.manifest_path
    details: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.is_file(),
        "ready": False,
        "motion_authorized": False,
        "hardware_connection_attempted": False,
    }
    if not manifest_path.is_file():
        details["manifest_template_path"] = str(resolved.manifest_template_path)
        return CheckResult(
            "es68_d435i_final_collision_model",
            CheckLevel.WARN,
            "final ES68+D435i manifest is absent; production collision checking is unavailable",
            details,
        )
    try:
        parsed = Es68D435iCollisionTemplate.load(
            manifest_path,
            model_root=resolved.root,
        )
    except Exception as exc:
        details["error"] = str(exc)
        return CheckResult(
            "es68_d435i_final_collision_model",
            CheckLevel.FAIL,
            "final ES68+D435i manifest is invalid",
            details,
        )
    details.update({"model_id": parsed.model_id, "ready": parsed.ready})
    if not parsed.ready:
        return CheckResult(
            "es68_d435i_final_collision_model",
            CheckLevel.WARN,
            "final ES68+D435i manifest exists but is not marked ready",
            details,
        )
    try:
        template = resolved.load_active()
        collision_hash_before = es68_d435i_collision_content_hash(template)
        robot_hash_before = es68_d435i_robot_geometry_hash(
            template,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        )
        collision_hash_after = es68_d435i_collision_content_hash(template)
        robot_hash_after = es68_d435i_robot_geometry_hash(
            template,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        )
    except Exception as exc:
        details["error"] = str(exc)
        return CheckResult(
            "es68_d435i_final_collision_model",
            CheckLevel.FAIL,
            "final ES68+D435i assets failed validation or hashing",
            details,
        )
    hashes_are_sha256 = all(
        _SHA256_PATTERN.fullmatch(value) is not None
        for value in (collision_hash_before, robot_hash_before)
    )
    hashes_are_stable = (
        collision_hash_before == collision_hash_after and robot_hash_before == robot_hash_after
    )
    details.update(
        {
            "collision_content_sha256": collision_hash_before,
            "robot_geometry_sha256": robot_hash_before,
            "hashes_are_sha256": hashes_are_sha256,
            "hashes_stable_across_recompute": hashes_are_stable,
            "hash_scope": "two-pass local manifest/mesh/FK recomputation",
            "joint_zero_offsets_rad": list(settings.kinematics.joint_zero_offsets_rad),
        }
    )
    if not hashes_are_sha256 or not hashes_are_stable:
        return CheckResult(
            "es68_d435i_final_collision_model",
            CheckLevel.FAIL,
            "final ES68+D435i asset hashes are invalid or changed during diagnosis",
            details,
        )
    return CheckResult(
        "es68_d435i_final_collision_model",
        CheckLevel.PASS,
        "final ES68+D435i manifest is ready and local asset hashes are stable",
        details,
    )


def _check_motion_readiness(settings: AppSettings) -> CheckResult:
    """Expose present release blockers without constructing a robot driver."""

    # The code backends now implement conservative, hash-bound interval proofs.  This
    # offline doctor still cannot establish that a particular physical workcell,
    # occupancy generation, trajectory, or operator approval is safe.
    level = CheckLevel.FAIL if settings.robot.motion_enabled else CheckLevel.WARN
    return CheckResult(
        "motion_readiness",
        level,
        "continuous proof backends are installed, but motion remains blocked until "
        "a live segment has current map-bound proofs and hardware acceptance",
        {
            "motion_enabled": settings.robot.motion_enabled,
            "motion_ready": False,
            "continuous_swept_mesh_supported": True,
            "continuous_swept_occupancy_supported": True,
            "available_path_checks": (
                "conservative adaptive interval proofs; evaluated per live segment"
            ),
            "accepted_static_free_configured": bool(
                settings.occupancy.accepted_static_free_aabbs
                and settings.occupancy.accepted_static_free_acceptance_id
            ),
            "doctor_authorizes_motion": False,
            "hardware_connection_attempted": False,
        },
    )


def _check_runtime_timing_acceptance(settings: AppSettings) -> CheckResult:
    """Verify the immutable four-budget authority without touching hardware."""

    timing = settings.stop_and_capture
    limits = {
        "maximum_perception_cycle_duration_s": timing.maximum_perception_cycle_duration_s,
        "maximum_operator_reposition_interval_s": (
            timing.maximum_operator_reposition_interval_s
        ),
        "maximum_segment_execution_duration_s": timing.maximum_segment_execution_duration_s,
        "maximum_schema5_handoff_duration_s": timing.maximum_schema5_handoff_duration_s,
    }
    path = timing.runtime_timing_acceptance_path
    acceptance_id = timing.runtime_timing_acceptance_id
    required = settings.robot.motion_enabled or timing.enabled
    missing = [name for name, value in limits.items() if value is None]
    if path is None or acceptance_id is None:
        missing.append("runtime_timing_acceptance_path/id")
    details: dict[str, object] = {
        "configured_limits_s": limits,
        "acceptance_path": str(path) if path is not None else None,
        "acceptance_id": acceptance_id,
        "hardware_connection_attempted": False,
        "motion_authorized": False,
    }
    if missing:
        details["missing"] = missing
        return CheckResult(
            "runtime_timing_acceptance",
            CheckLevel.FAIL if required else CheckLevel.WARN,
            "runtime timing authority is incomplete",
            details,
        )
    try:
        assert path is not None and acceptance_id is not None
        acceptance = read_runtime_timing_acceptance(path)
        acceptance.assert_matches(settings=settings, acceptance_id=acceptance_id)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        details["error"] = str(exc)
        return CheckResult(
            "runtime_timing_acceptance",
            CheckLevel.FAIL,
            "runtime timing authority is invalid or differs from settings",
            details,
        )
    details.update(
        {
            "verified": True,
            "metadata_sha256": acceptance.metadata_sha256,
            "trial_count": acceptance.trial_count,
            "raw_evidence_count": acceptance.raw_evidence_count,
        }
    )
    return CheckResult(
        "runtime_timing_acceptance",
        CheckLevel.PASS,
        "runtime timing authority matches all four configured budgets",
        details,
    )


def run_doctor(settings: AppSettings) -> list[CheckResult]:
    """Run non-moving local checks. This function never connects to the robot."""

    return [
        _check_python(),
        _check_elite_sdk(settings),
        _check_realsense(),
        _check_robot_configuration(settings),
        _check_robot_address(settings),
        _check_thermal(settings),
        _check_hand_eye(settings),
        _check_kinematics(settings),
        _check_collision_configuration(settings),
        _check_occupancy_configuration(settings),
        _check_final_collision_model(settings),
        _check_runtime_timing_acceptance(settings),
        _check_motion_readiness(settings),
    ]
