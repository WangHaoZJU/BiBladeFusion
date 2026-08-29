"""Offline readiness audit for the supervised unknown-blade scan runtime.

The audit deliberately does not connect to a camera or robot and never turns a
configuration into a motion authorization.  Its job is to separate missing code or
assets from workcell values that can only be accepted during the hardware campaign.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from biblade_fusion.calibration import load_hand_eye_calibration
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.diagnostics.types import CheckLevel, CheckResult
from biblade_fusion.perception.stereo import run_foundation_stereo_doctor
from biblade_fusion.robotics import (
    AcceptedStaticFreeAabb,
    Cs68PinocchioCollisionChecker,
    Es68D435iCollisionResources,
    OccupancyRobotCollisionChecker,
)
from biblade_fusion.storage import (
    read_coarse_model_summary,
    read_static_free_acceptance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_check(settings: AppSettings) -> CheckResult:
    missing: list[str] = []
    if settings.robot.model != "es68":
        missing.append("robot.model=es68")
    if settings.robot.robot_ip is None:
        missing.append("robot.robot_ip")
    if settings.robot.local_ip is None:
        missing.append("robot.local_ip")
    if not settings.robot.motion_enabled:
        missing.append("robot.motion_enabled")
    if not settings.stop_and_capture.enabled:
        missing.append("stop_and_capture.enabled")
    if settings.stop_and_capture.maximum_segment_joint_delta_rad is None:
        missing.append("stop_and_capture.maximum_segment_joint_delta_rad")
    if not settings.occupancy.enabled:
        missing.append("occupancy.enabled")
    if settings.occupancy.workspace_bounds_min_m is None:
        missing.append("occupancy.workspace_bounds_min_m")
    if settings.occupancy.workspace_bounds_max_m is None:
        missing.append("occupancy.workspace_bounds_max_m")
    if not settings.occupancy.accepted_static_free_aabbs:
        missing.append("occupancy.accepted_static_free_aabbs")
    if settings.occupancy.accepted_static_free_acceptance_id is None:
        missing.append("occupancy.accepted_static_free_acceptance_id")
    if settings.occupancy.accepted_static_free_acceptance_path is None:
        missing.append("occupancy.accepted_static_free_acceptance_path")
    return CheckResult(
        "supervised_scan_policy",
        CheckLevel.FAIL if missing else CheckLevel.PASS,
        (
            "supervised scan policy has unresolved fail-closed fields"
            if missing
            else "single-segment, operator-approved stop-and-capture policy is configured"
        ),
        {
            "missing": missing,
            "operator_approval_required": settings.stop_and_capture.require_operator_approval,
            "capture_after_every_segment": (
                settings.stop_and_capture.require_capture_after_every_segment
            ),
            "motion_authorized": False,
            "hardware_connection_attempted": False,
        },
    )


def _science_geometry_check(settings: AppSettings) -> CheckResult:
    missing: list[str] = []
    if settings.kinematics.model_path is None:
        missing.append("kinematics.model_path")
    elif not settings.kinematics.model_path.is_file():
        missing.append("kinematics.model_path:file_missing")
    if settings.view_planning.standoff_distance_m is None:
        missing.append("view_planning.standoff_distance_m")
    if settings.view_planning.adaptive_standoff_enabled and (
        settings.view_planning.minimum_standoff_distance_m is None
        or settings.view_planning.maximum_standoff_distance_m is None
    ):
        missing.append("view_planning.adaptive_standoff_bounds")
    if settings.view_filter.workspace is None:
        missing.append("view_filter.workspace")
    return CheckResult(
        "supervised_scan_science_geometry",
        CheckLevel.FAIL if missing else CheckLevel.PASS,
        (
            "view generation or controller-specific IK lacks accepted physical inputs"
            if missing
            else "standoff, workspace, and controller-specific IK inputs are configured"
        ),
        {"missing": missing, "motion_authorized": False},
    )


def _static_free_acceptance_check(settings: AppSettings) -> CheckResult:
    """Verify the physical declaration that may downgrade only UNKNOWN voxels."""

    path = settings.occupancy.accepted_static_free_acceptance_path
    acceptance_id = settings.occupancy.accepted_static_free_acceptance_id
    lower = settings.occupancy.workspace_bounds_min_m
    upper = settings.occupancy.workspace_bounds_max_m
    configured = tuple(
        AcceptedStaticFreeAabb(item.name, item.minimum_m, item.maximum_m)
        for item in settings.occupancy.accepted_static_free_aabbs
    )
    if path is None or acceptance_id is None or lower is None or upper is None or not configured:
        return CheckResult(
            "supervised_scan_static_free_acceptance",
            CheckLevel.FAIL,
            "immutable static-free workcell acceptance is not configured",
            {"motion_authorized": False},
        )
    try:
        resources = Es68D435iCollisionResources.packaged_template()
        checker = Cs68PinocchioCollisionChecker.from_es68_resources(
            resources,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
            environment_obstacles=settings.collision.obstacles,
            minimum_clearance_m=settings.collision.minimum_clearance_m,
        )
        stored = read_static_free_acceptance(path)
        stored.assert_matches(
            acceptance_id=acceptance_id,
            robot_geometry_hash=checker.robot_geometry_hash,
            workspace_minimum_m=lower,
            workspace_maximum_m=upper,
            regions=configured,
        )
    except Exception as exc:
        return CheckResult(
            "supervised_scan_static_free_acceptance",
            CheckLevel.FAIL,
            "static-free acceptance failed semantic verification",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "motion_authorized": False,
            },
        )
    return CheckResult(
        "supervised_scan_static_free_acceptance",
        CheckLevel.PASS,
        "static-free regions match the accepted workcell, robot geometry, and workspace",
        {
            "path": str(stored.path),
            "acceptance_id": stored.acceptance_id,
            "metadata_sha256": stored.metadata_sha256,
            "operator_id": stored.operator_id,
            "motion_authorized": False,
        },
    )


def _calibration_check(settings: AppSettings) -> CheckResult:
    stereo_path = settings.realsense.stereo_calibration_path
    hand_eye_path = settings.hand_eye.calibration_path
    missing = [
        name
        for name, path in (
            ("realsense.stereo_calibration_path", stereo_path),
            ("hand_eye.calibration_path", hand_eye_path),
        )
        if path is None or not path.is_file()
    ]
    details: dict[str, object] = {"missing": missing, "motion_authorized": False}
    if missing:
        return CheckResult(
            "supervised_scan_calibration",
            CheckLevel.FAIL,
            "active user calibration assets are missing",
            details,
        )
    assert stereo_path is not None and hand_eye_path is not None
    try:
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        hand_eye.require_flange_primary()
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        return CheckResult(
            "supervised_scan_calibration",
            CheckLevel.FAIL,
            "hand-eye asset failed semantic validation",
            details,
        )
    details.update(
        {
            "stereo_path": str(stereo_path.resolve()),
            "stereo_sha256": _sha256(stereo_path),
            "hand_eye_path": str(hand_eye_path.resolve()),
            "hand_eye_sha256": _sha256(hand_eye_path),
        }
    )
    return CheckResult(
        "supervised_scan_calibration",
        CheckLevel.PASS,
        "active stereo and flange-primary hand-eye assets are readable",
        details,
    )


def _collision_backend_check(settings: AppSettings) -> CheckResult:
    details: dict[str, object] = {
        "motion_authorized": False,
        "hardware_connection_attempted": False,
    }
    try:
        resources = Es68D435iCollisionResources.packaged_template()
        template = resources.load_active()
        checker = Cs68PinocchioCollisionChecker.from_es68_resources(
            resources,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
            environment_obstacles=settings.collision.obstacles,
            minimum_clearance_m=settings.collision.minimum_clearance_m,
        )
        mesh_supported = bool(checker.continuous_swept_volume_supported)
        occupancy_checker = OccupancyRobotCollisionChecker(checker, lambda: None)
        occupancy_supported = bool(occupancy_checker.continuous_swept_volume_supported)
        details.update(
            {
                "model_id": template.model_id,
                "robot_geometry_hash": checker.robot_geometry_hash,
                "motion_model_contract_hash": checker.motion_model_contract_hash,
                "continuous_swept_mesh_supported": mesh_supported,
                "continuous_swept_occupancy_supported": occupancy_supported,
            }
        )
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        return CheckResult(
            "supervised_scan_continuous_collision",
            CheckLevel.FAIL,
            "ES68+D435i collision backend could not be constructed",
            details,
        )
    if not mesh_supported or not occupancy_supported:
        return CheckResult(
            "supervised_scan_continuous_collision",
            CheckLevel.FAIL,
            "one or both conservative continuous sweep proofs are unavailable",
            details,
        )
    return CheckResult(
        "supervised_scan_continuous_collision",
        CheckLevel.PASS,
        "both conservative continuous sweep proof backends are available",
        details,
    )


def _reference_check(
    settings: AppSettings,
    reference_coarse_model: str | Path | None,
    mode: Literal["bootstrap", "fine"],
) -> CheckResult:
    if mode == "bootstrap":
        return CheckResult(
            "supervised_scan_reference",
            CheckLevel.PASS,
            "operator-guided unknown-blade bootstrap selected",
            {
                "reference_required": False,
                "motion_authorized": False,
            },
        )
    if reference_coarse_model is None:
        return CheckResult(
            "supervised_scan_reference",
            CheckLevel.FAIL,
            "fine scanning requires an explicitly pinned schema-5 coarse model",
            {"reference_required": True, "motion_authorized": False},
        )
    root = Path(reference_coarse_model).resolve()
    try:
        stored = read_coarse_model_summary(root)
        schema = int(stored.metadata["schema_version"])
        digest = _sha256(root / "metadata.json")
    except Exception as exc:
        return CheckResult(
            "supervised_scan_reference",
            CheckLevel.FAIL,
            "coarse-model reference failed semantic readback",
            {"error": f"{type(exc).__name__}: {exc}", "motion_authorized": False},
        )
    foreground_enabled = settings.blade_foreground.enabled
    return CheckResult(
        "supervised_scan_reference",
        CheckLevel.PASS if foreground_enabled else CheckLevel.FAIL,
        (
            "fine reference and reference-guided foreground are configured"
            if foreground_enabled
            else "fine reference exists but blade_foreground.enabled is false"
        ),
        {
            "root": str(root),
            "schema_version": schema,
            "metadata_sha256": digest,
            "blade_foreground_enabled": foreground_enabled,
            "motion_authorized": False,
        },
    )


def run_supervised_scan_readiness(
    settings: AppSettings,
    *,
    mode: Literal["bootstrap", "fine"],
    reference_coarse_model: str | Path | None = None,
) -> list[CheckResult]:
    """Audit every code/configuration prerequisite without touching hardware.

    A fully passing result only means the software can proceed to the separate
    hardware-acceptance gate.  It is never a permit and never claims that any physical
    threshold has been accepted.
    """

    foundation = [
        CheckResult(
            f"scan_{item.name}",
            item.level,
            item.message,
            {**item.details, "motion_authorized": False},
        )
        for item in run_foundation_stereo_doctor(settings.foundation_stereo)
    ]
    return [
        _policy_check(settings),
        _science_geometry_check(settings),
        _static_free_acceptance_check(settings),
        _calibration_check(settings),
        _reference_check(settings, reference_coarse_model, mode),
        *foundation,
        _collision_backend_check(settings),
        CheckResult(
            "hardware_acceptance_release",
            CheckLevel.WARN,
            "software prerequisites do not replace GPU, workcell, or guarded-motion acceptance",
            {
                "pending_by_design": True,
                "motion_authorized": False,
                "hardware_connection_attempted": False,
            },
        ),
    ]
