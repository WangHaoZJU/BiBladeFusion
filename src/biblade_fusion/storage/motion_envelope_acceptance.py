"""Immutable physical acceptance for ServoJ tracking and stop uncertainty.

The continuous collision proofs operate on a commanded joint path.  A physical
robot follows that path with finite error and can continue moving while a stop is
propagated.  This asset records a workcell-specific upper bound for both effects.
It is evidence only: by itself it never grants motion authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

MOTION_ENVELOPE_ACCEPTANCE_SCHEMA_VERSION = 2
_ASSET_TYPE = "biblade_fusion.motion_envelope_acceptance"
_DECLARATION = (
    "The recorded joint-path uncertainty bounds include measured ServoJ following "
    "error, feedback latency, stop-command latency and post-command stopping drift "
    "for the exact bound robot, collision and control contracts."
)
_CHECK_NAMES = (
    "final_collision_assembly_verified",
    "final_servoj_configuration_verified",
    "representative_workspace_paths_verified",
    "intentional_tracking_fault_stop_verified",
    "bootstrap_multichannel_stop_verified",
    "segment_boundary_stop_verified",
    "emergency_stop_verified",
)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _six_vector(
    value: object,
    *,
    label: str,
    strictly_positive: bool,
) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{label} must contain six values")
    array = np.asarray(value, dtype=np.float64)
    invalid = np.any(array <= 0.0) if strictly_positive else np.any(array < 0.0)
    if array.shape != (6,) or not np.isfinite(array).all() or invalid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{label} must be a finite {qualifier} six-vector")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def motion_control_contract_sha256(
    *,
    robot_control: Mapping[str, Any],
    motion_preflight: Mapping[str, Any],
    servoj_runtime: Mapping[str, Any],
) -> str:
    """Hash every numeric/software control input used by acceptance and runtime.

    Callers must pass explicit, JSON-compatible subsets rather than entire settings
    objects.  Network addresses, output paths and acceptance-navigation fields do
    not belong to this control contract.
    """

    payload = {
        "schema": "biblade_fusion.motion_control_contract.v1",
        "robot_control": dict(robot_control),
        "motion_preflight": dict(motion_preflight),
        "servoj_runtime": dict(servoj_runtime),
    }
    return _sha256_bytes(_canonical_json(payload))


def motion_control_contract_for_settings(settings: Any) -> str:
    """Reproduce the deployed Elite/ServoJ control contract from AppSettings."""

    from biblade_fusion.devices.robot.streaming import ServoJStreamConfig

    robot = settings.robot
    motion = settings.motion_preflight
    sdk_wheel = Path(robot.sdk_wheel).resolve()
    robot_control = {
        "model": robot.model,
        "sdk_import_path": robot.sdk_import_path,
        "sdk_wheel_sha256": _sha256_path(sdk_wheel) if sdk_wheel.is_file() else None,
        "servoj_time_s": robot.servoj_time_s,
        "servoj_lookahead_time_s": robot.servoj_lookahead_time_s,
        "servoj_gain": robot.servoj_gain,
        "stopj_acceleration_rad_s2": robot.stopj_acceleration_rad_s2,
        "default_speed_scaling": robot.default_speed_scaling,
        "maximum_speed_scaling": robot.maximum_speed_scaling,
        "rtsi_frequency_hz": robot.rtsi_frequency_hz,
        "settle_time_s": robot.settle_time_s,
    }
    motion_preflight = motion.model_dump(mode="json", exclude={
        "motion_envelope_acceptance_path",
        "motion_envelope_acceptance_id",
        # Planner search policy changes path shape, not the accepted ServoJ
        # tracking/stop envelope. Every returned route still uses the same
        # velocity-limited stream and exact collision preflight below.
        "enable_ompl_fallback",
        "ompl_plan_timeout_s",
        "ompl_rrt_range_rad",
        "ompl_simplify_path",
    })
    runtime = ServoJStreamConfig(
        dt_s=motion.servoj_dt_s,
        tracking_check_every_n_commands=2,
    )
    runtime.validate()
    return motion_control_contract_sha256(
        robot_control=robot_control,
        motion_preflight=motion_preflight,
        servoj_runtime=asdict(runtime),
    )


@dataclass(frozen=True, slots=True)
class StoredMotionEnvelopeAcceptance:
    path: Path
    acceptance_id: str
    workcell_id: str
    operator_id: str
    accepted_at_utc: datetime
    robot_geometry_hash: str
    motion_model_contract_hash: str
    motion_control_contract_hash: str
    maximum_tracking_deviation_rad: tuple[float, float, float, float, float, float]
    maximum_stop_drift_rad: tuple[float, float, float, float, float, float]
    safety_margin_factor: float
    maximum_feedback_interval_s: float
    maximum_stop_acknowledgement_s: float
    maximum_stopped_actual_joint_velocity_rad_s: float
    maximum_stopped_target_joint_velocity_rad_s: float
    maximum_stopped_actual_tcp_linear_velocity_m_s: float
    maximum_stopped_actual_tcp_angular_velocity_rad_s: float
    maximum_stopped_target_tcp_linear_velocity_m_s: float
    maximum_stopped_target_tcp_angular_velocity_rad_s: float
    trial_count: int
    metadata_sha256: str

    @property
    def accepted_joint_uncertainty_rad(
        self,
    ) -> tuple[float, float, float, float, float, float]:
        return tuple(
            self.safety_margin_factor * (tracking + drift)
            for tracking, drift in zip(
                self.maximum_tracking_deviation_rad,
                self.maximum_stop_drift_rad,
                strict=True,
            )
        )  # type: ignore[return-value]

    def assert_matches(
        self,
        *,
        acceptance_id: str,
        robot_geometry_hash: str,
        motion_model_contract_hash: str,
        motion_control_contract_hash: str,
    ) -> None:
        if self.acceptance_id != _digest(acceptance_id, label="acceptance_id"):
            raise ValueError("motion-envelope acceptance ID differs from configuration")
        if self.robot_geometry_hash != robot_geometry_hash:
            raise ValueError("motion-envelope robot geometry differs from runtime")
        if self.motion_model_contract_hash != motion_model_contract_hash:
            raise ValueError("motion-envelope collision contract differs from runtime")
        if self.motion_control_contract_hash != motion_control_contract_hash:
            raise ValueError("motion-envelope ServoJ control contract differs from runtime")


def _strict_load(path: Path) -> dict[str, Any]:
    def object_from_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("motion-envelope metadata must be an object")
    return payload


def _validated_payload(
    *,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    robot_geometry_hash: str,
    motion_model_contract_hash: str,
    motion_control_contract_hash: str,
    maximum_tracking_deviation_rad: tuple[float, ...],
    maximum_stop_drift_rad: tuple[float, ...],
    safety_margin_factor: float,
    maximum_feedback_interval_s: float,
    maximum_stop_acknowledgement_s: float,
    maximum_stopped_actual_joint_velocity_rad_s: float,
    maximum_stopped_target_joint_velocity_rad_s: float,
    maximum_stopped_actual_tcp_linear_velocity_m_s: float,
    maximum_stopped_actual_tcp_angular_velocity_rad_s: float,
    maximum_stopped_target_tcp_linear_velocity_m_s: float,
    maximum_stopped_target_tcp_angular_velocity_rad_s: float,
    trial_count: int,
    checklist: Mapping[str, bool],
) -> dict[str, Any]:
    workcell = str(workcell_id).strip()
    operator = str(operator_id).strip()
    if not workcell or not operator:
        raise ValueError("workcell_id and operator_id must be non-empty")
    if accepted_at_utc.tzinfo is None or accepted_at_utc.utcoffset() is None:
        raise ValueError("accepted_at_utc must be timezone-aware")
    tracking = _six_vector(
        maximum_tracking_deviation_rad,
        label="maximum_tracking_deviation_rad",
        strictly_positive=True,
    )
    stop_drift = _six_vector(
        maximum_stop_drift_rad,
        label="maximum_stop_drift_rad",
        strictly_positive=False,
    )
    margin = float(safety_margin_factor)
    feedback = float(maximum_feedback_interval_s)
    stop_latency = float(maximum_stop_acknowledgement_s)
    stopped_velocity_limits = tuple(
        float(value)
        for value in (
            maximum_stopped_actual_joint_velocity_rad_s,
            maximum_stopped_target_joint_velocity_rad_s,
            maximum_stopped_actual_tcp_linear_velocity_m_s,
            maximum_stopped_actual_tcp_angular_velocity_rad_s,
            maximum_stopped_target_tcp_linear_velocity_m_s,
            maximum_stopped_target_tcp_angular_velocity_rad_s,
        )
    )
    if not np.isfinite((margin, feedback, stop_latency, *stopped_velocity_limits)).all():
        raise ValueError("motion-envelope scalar measurements must be finite")
    if (
        margin < 1.0
        or feedback <= 0.0
        or stop_latency <= 0.0
        or any(value <= 0.0 for value in stopped_velocity_limits)
    ):
        raise ValueError("motion-envelope margin/latencies are outside their valid range")
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 3:
        raise ValueError("motion-envelope acceptance requires at least three trials")
    if set(checklist) != set(_CHECK_NAMES) or not all(
        checklist[name] is True for name in _CHECK_NAMES
    ):
        raise ValueError("all motion-envelope physical checks must be true")
    return {
        "schema_version": MOTION_ENVELOPE_ACCEPTANCE_SCHEMA_VERSION,
        "asset_type": _ASSET_TYPE,
        "workcell_id": workcell,
        "operator_id": operator,
        "accepted_at_utc": accepted_at_utc.astimezone(UTC).isoformat(),
        "robot_geometry_sha256": _digest(
            robot_geometry_hash,
            label="robot_geometry_hash",
        ),
        "motion_model_contract_sha256": _digest(
            motion_model_contract_hash,
            label="motion_model_contract_hash",
        ),
        "motion_control_contract_sha256": _digest(
            motion_control_contract_hash,
            label="motion_control_contract_hash",
        ),
        "measurements": {
            "maximum_tracking_deviation_rad": list(tracking),
            "maximum_stop_drift_rad": list(stop_drift),
            "safety_margin_factor": margin,
            "accepted_joint_uncertainty_rad": [
                margin * (following + drift)
                for following, drift in zip(tracking, stop_drift, strict=True)
            ],
            "maximum_feedback_interval_s": feedback,
            "maximum_stop_acknowledgement_s": stop_latency,
            "maximum_stopped_actual_joint_velocity_rad_s": stopped_velocity_limits[0],
            "maximum_stopped_target_joint_velocity_rad_s": stopped_velocity_limits[1],
            "maximum_stopped_actual_tcp_linear_velocity_m_s": stopped_velocity_limits[2],
            "maximum_stopped_actual_tcp_angular_velocity_rad_s": stopped_velocity_limits[3],
            "maximum_stopped_target_tcp_linear_velocity_m_s": stopped_velocity_limits[4],
            "maximum_stopped_target_tcp_angular_velocity_rad_s": stopped_velocity_limits[5],
            "trial_count": trial_count,
        },
        "checklist": {name: True for name in _CHECK_NAMES},
        "declaration": _DECLARATION,
        "motion_authorized": False,
    }


def write_motion_envelope_acceptance(
    path: str | Path,
    **values: Any,
) -> StoredMotionEnvelopeAcceptance:
    """Write one non-overwriting acceptance and verify it by independent readback."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"motion-envelope acceptance already exists: {destination}")
    payload = _validated_payload(**values)
    payload["acceptance_id"] = _sha256_bytes(_canonical_json(payload))
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        metadata = temporary / "metadata.json"
        metadata.write_bytes(_canonical_json(payload) + b"\n")
        with metadata.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return read_motion_envelope_acceptance(destination)


def read_motion_envelope_acceptance(
    path: str | Path,
) -> StoredMotionEnvelopeAcceptance:
    """Strictly validate and recompute one motion-envelope acceptance."""

    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    payload = _strict_load(metadata_path)
    expected_fields = {
        "schema_version",
        "asset_type",
        "workcell_id",
        "operator_id",
        "accepted_at_utc",
        "robot_geometry_sha256",
        "motion_model_contract_sha256",
        "motion_control_contract_sha256",
        "measurements",
        "checklist",
        "declaration",
        "motion_authorized",
        "acceptance_id",
    }
    if set(payload) != expected_fields:
        raise ValueError("motion-envelope acceptance fields differ from schema")
    acceptance_id = _digest(payload.pop("acceptance_id"), label="acceptance_id")
    if acceptance_id != _sha256_bytes(_canonical_json(payload)):
        raise ValueError("motion-envelope acceptance identity mismatch")
    if (
        payload["schema_version"] != MOTION_ENVELOPE_ACCEPTANCE_SCHEMA_VERSION
        or payload["asset_type"] != _ASSET_TYPE
        or payload["declaration"] != _DECLARATION
        or payload["motion_authorized"] is not False
    ):
        raise ValueError("motion-envelope schema contract is invalid")
    measurements = payload["measurements"]
    if not isinstance(measurements, dict) or set(measurements) != {
        "maximum_tracking_deviation_rad",
        "maximum_stop_drift_rad",
        "safety_margin_factor",
        "accepted_joint_uncertainty_rad",
        "maximum_feedback_interval_s",
        "maximum_stop_acknowledgement_s",
        "maximum_stopped_actual_joint_velocity_rad_s",
        "maximum_stopped_target_joint_velocity_rad_s",
        "maximum_stopped_actual_tcp_linear_velocity_m_s",
        "maximum_stopped_actual_tcp_angular_velocity_rad_s",
        "maximum_stopped_target_tcp_linear_velocity_m_s",
        "maximum_stopped_target_tcp_angular_velocity_rad_s",
        "trial_count",
    }:
        raise ValueError("motion-envelope measurement fields differ from schema")
    tracking = _six_vector(
        measurements["maximum_tracking_deviation_rad"],
        label="maximum_tracking_deviation_rad",
        strictly_positive=True,
    )
    stop_drift = _six_vector(
        measurements["maximum_stop_drift_rad"],
        label="maximum_stop_drift_rad",
        strictly_positive=False,
    )
    checklist = payload["checklist"]
    reproduced = _validated_payload(
        workcell_id=str(payload["workcell_id"]),
        operator_id=str(payload["operator_id"]),
        accepted_at_utc=datetime.fromisoformat(str(payload["accepted_at_utc"])),
        robot_geometry_hash=str(payload["robot_geometry_sha256"]),
        motion_model_contract_hash=str(payload["motion_model_contract_sha256"]),
        motion_control_contract_hash=str(payload["motion_control_contract_sha256"]),
        maximum_tracking_deviation_rad=tracking,
        maximum_stop_drift_rad=stop_drift,
        safety_margin_factor=float(measurements["safety_margin_factor"]),
        maximum_feedback_interval_s=float(measurements["maximum_feedback_interval_s"]),
        maximum_stop_acknowledgement_s=float(
            measurements["maximum_stop_acknowledgement_s"]
        ),
        maximum_stopped_actual_joint_velocity_rad_s=float(
            measurements["maximum_stopped_actual_joint_velocity_rad_s"]
        ),
        maximum_stopped_target_joint_velocity_rad_s=float(
            measurements["maximum_stopped_target_joint_velocity_rad_s"]
        ),
        maximum_stopped_actual_tcp_linear_velocity_m_s=float(
            measurements["maximum_stopped_actual_tcp_linear_velocity_m_s"]
        ),
        maximum_stopped_actual_tcp_angular_velocity_rad_s=float(
            measurements["maximum_stopped_actual_tcp_angular_velocity_rad_s"]
        ),
        maximum_stopped_target_tcp_linear_velocity_m_s=float(
            measurements["maximum_stopped_target_tcp_linear_velocity_m_s"]
        ),
        maximum_stopped_target_tcp_angular_velocity_rad_s=float(
            measurements["maximum_stopped_target_tcp_angular_velocity_rad_s"]
        ),
        trial_count=int(measurements["trial_count"]),
        checklist=dict(checklist) if isinstance(checklist, dict) else {},
    )
    if reproduced != payload:
        raise ValueError("motion-envelope acceptance does not reproduce canonically")
    accepted = _six_vector(
        measurements["accepted_joint_uncertainty_rad"],
        label="accepted_joint_uncertainty_rad",
        strictly_positive=True,
    )
    expected_accepted = tuple(
        float(measurements["safety_margin_factor"]) * (following + drift)
        for following, drift in zip(tracking, stop_drift, strict=True)
    )
    if accepted != expected_accepted:
        raise ValueError("accepted joint uncertainty does not reproduce")
    return StoredMotionEnvelopeAcceptance(
        path=root,
        acceptance_id=acceptance_id,
        workcell_id=str(payload["workcell_id"]),
        operator_id=str(payload["operator_id"]),
        accepted_at_utc=datetime.fromisoformat(str(payload["accepted_at_utc"])),
        robot_geometry_hash=str(payload["robot_geometry_sha256"]),
        motion_model_contract_hash=str(payload["motion_model_contract_sha256"]),
        motion_control_contract_hash=str(payload["motion_control_contract_sha256"]),
        maximum_tracking_deviation_rad=tracking,
        maximum_stop_drift_rad=stop_drift,
        safety_margin_factor=float(measurements["safety_margin_factor"]),
        maximum_feedback_interval_s=float(measurements["maximum_feedback_interval_s"]),
        maximum_stop_acknowledgement_s=float(
            measurements["maximum_stop_acknowledgement_s"]
        ),
        maximum_stopped_actual_joint_velocity_rad_s=float(
            measurements["maximum_stopped_actual_joint_velocity_rad_s"]
        ),
        maximum_stopped_target_joint_velocity_rad_s=float(
            measurements["maximum_stopped_target_joint_velocity_rad_s"]
        ),
        maximum_stopped_actual_tcp_linear_velocity_m_s=float(
            measurements["maximum_stopped_actual_tcp_linear_velocity_m_s"]
        ),
        maximum_stopped_actual_tcp_angular_velocity_rad_s=float(
            measurements["maximum_stopped_actual_tcp_angular_velocity_rad_s"]
        ),
        maximum_stopped_target_tcp_linear_velocity_m_s=float(
            measurements["maximum_stopped_target_tcp_linear_velocity_m_s"]
        ),
        maximum_stopped_target_tcp_angular_velocity_rad_s=float(
            measurements["maximum_stopped_target_tcp_angular_velocity_rad_s"]
        ),
        trial_count=int(measurements["trial_count"]),
        metadata_sha256=_sha256_path(metadata_path),
    )


__all__ = [
    "MOTION_ENVELOPE_ACCEPTANCE_SCHEMA_VERSION",
    "StoredMotionEnvelopeAcceptance",
    "motion_control_contract_sha256",
    "motion_control_contract_for_settings",
    "read_motion_envelope_acceptance",
    "write_motion_envelope_acceptance",
]
