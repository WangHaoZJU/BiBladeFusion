"""Safe, non-moving environment diagnostics."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from biblade_fusion.core.settings import AppSettings


class CheckLevel(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    level: CheckLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)


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
    if not settings.thermal.enabled:
        return CheckResult(
            "thermal_camera",
            CheckLevel.PASS,
            "disabled; interface reserved",
        )
    if settings.thermal.driver is None:
        return CheckResult(
            "thermal_camera",
            CheckLevel.FAIL,
            "enabled without a configured driver",
        )
    return CheckResult(
        "thermal_camera",
        CheckLevel.WARN,
        f"driver configured as {settings.thermal.driver}; device probe not implemented",
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
    ]
