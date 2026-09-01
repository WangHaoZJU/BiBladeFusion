"""Immutable FoundationStereo inference-stationarity evidence assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.robotics.stationarity import (
    StationarityEvidence,
    validate_stationary_trace,
)

INFERENCE_STATIONARITY_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, INFERENCE_STATIONARITY_SCHEMA_VERSION}
INFERENCE_STATIONARITY_TRACE_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class StoredInferenceStationarity:
    path: Path
    view_id: str
    sequence_index: int
    reference: RobotState
    trace: tuple[RobotState, ...]
    evidence: StationarityEvidence
    source_session_manifest_path: Path
    source_session_manifest_sha256: str
    thresholds: tuple[float, float, float, float]
    content_sha256: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class StoredInferenceStationarityTrace:
    """Diagnostic sampler trace that does not assert stationarity acceptance."""

    path: Path
    view_id: str
    sequence_index: int
    trace: tuple[RobotState, ...]
    source_session_manifest_path: Path
    source_session_manifest_sha256: str
    sampler_diagnostics: dict[str, Any]
    content_sha256: str
    file_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_once_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.partial"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json_loads(text: str) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def _state_payload(state: RobotState) -> dict[str, Any]:
    return {
        "monotonic_time_ns": int(state.monotonic_time_ns),
        "controller_time_s": float(state.controller_time_s),
        "joint_positions_rad": state.joint_positions_rad.tolist(),
        "base_T_tcp": state.base_t_tcp.matrix.tolist(),
        "robot_mode": state.robot_mode,
        "safety_status": state.safety_status,
        "speed_scaling": float(state.speed_scaling),
        "runtime_state": state.runtime_state,
    }


def _state_from_payload(payload: object, *, schema_version: int) -> RobotState:
    if not isinstance(payload, dict):
        raise ValueError("stationarity robot state must be an object")
    expected_fields = {
        "monotonic_time_ns",
        "controller_time_s",
        "joint_positions_rad",
        "base_T_tcp",
        "robot_mode",
        "safety_status",
        "speed_scaling",
    }
    if schema_version >= 2:
        expected_fields.add("runtime_state")
    if set(payload) != expected_fields:
        raise ValueError("stationarity robot-state fields differ from schema")
    return RobotState(
        monotonic_time_ns=int(payload["monotonic_time_ns"]),
        controller_time_s=float(payload["controller_time_s"]),
        joint_positions_rad=np.asarray(
            payload["joint_positions_rad"], dtype=np.float64
        ),
        base_t_tcp=PoseSE3("base", "tcp", payload["base_T_tcp"]),
        robot_mode=str(payload["robot_mode"]),
        safety_status=str(payload["safety_status"]),
        speed_scaling=float(payload["speed_scaling"]),
        runtime_state=(
            None
            if schema_version < 2 or payload["runtime_state"] is None
            else str(payload["runtime_state"])
        ),
    )


def _diagnostic_trace(states: tuple[RobotState, ...]) -> tuple[RobotState, ...]:
    if len(states) < 3:
        raise ValueError("stationarity diagnostic trace requires at least three states")
    previous_host_ns = states[0].monotonic_time_ns
    previous_controller_s = float(states[0].controller_time_s)
    for state in states[1:]:
        if state.monotonic_time_ns <= previous_host_ns:
            raise ValueError(
                "stationarity diagnostic host timestamps must increase"
            )
        controller_s = float(state.controller_time_s)
        if controller_s < previous_controller_s:
            raise ValueError(
                "stationarity diagnostic controller timestamps moved backwards"
            )
        previous_host_ns = state.monotonic_time_ns
        previous_controller_s = controller_s
    return states


def _diagnostics_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("sampler diagnostics must be an object")
    try:
        normalized = _strict_json_loads(_canonical_json(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("sampler diagnostics must contain finite JSON values") from exc
    if not isinstance(normalized, dict):
        raise ValueError("sampler diagnostics must be an object")
    return normalized


def write_inference_stationarity_trace(
    path: str | Path,
    *,
    view_id: str,
    sequence_index: int,
    trace: tuple[RobotState, ...],
    source_session_manifest: str | Path,
    sampler_diagnostics: dict[str, Any],
) -> StoredInferenceStationarityTrace:
    """Persist a write-once trace before applying any acceptance threshold."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Inference-stationarity diagnostic already exists: {destination}"
        )
    selected_view = str(view_id).strip()
    if not selected_view or sequence_index < 0:
        raise ValueError("stationarity diagnostic source identity is invalid")
    manifest = Path(source_session_manifest).resolve()
    if not manifest.is_file():
        raise ValueError("stationarity diagnostic source session manifest is missing")
    states = _diagnostic_trace(tuple(trace))
    diagnostics = _diagnostics_payload(sampler_diagnostics)
    body: dict[str, Any] = {
        "schema_version": INFERENCE_STATIONARITY_TRACE_SCHEMA_VERSION,
        "artifact_kind": "biblade_fusion.inference_stationarity_trace",
        "depth_backend": "foundation_stereo",
        "view_id": selected_view,
        "sequence_index": int(sequence_index),
        "source_session_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "robot_state_trace": [_state_payload(item) for item in states],
        "sampler_diagnostics": diagnostics,
    }
    content_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    _write_once_json(destination, {**body, "content_sha256": content_sha256})
    return read_inference_stationarity_trace(destination)


def read_inference_stationarity_trace(
    path: str | Path,
) -> StoredInferenceStationarityTrace:
    """Verify a diagnostic trace without treating it as motion authority."""

    source = Path(path).resolve()
    try:
        payload = _strict_json_loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("stationarity diagnostic root must be an object")
        expected_keys = {
            "schema_version",
            "artifact_kind",
            "depth_backend",
            "view_id",
            "sequence_index",
            "source_session_manifest",
            "robot_state_trace",
            "sampler_diagnostics",
            "content_sha256",
        }
        if set(payload) != expected_keys:
            raise ValueError("stationarity diagnostic fields differ from schema")
        if (
            payload["schema_version"]
            != INFERENCE_STATIONARITY_TRACE_SCHEMA_VERSION
            or payload["artifact_kind"]
            != "biblade_fusion.inference_stationarity_trace"
            or payload["depth_backend"] != "foundation_stereo"
        ):
            raise ValueError("unsupported stationarity diagnostic contract")
        declared_content = str(payload["content_sha256"])
        body = {
            key: value
            for key, value in payload.items()
            if key != "content_sha256"
        }
        if (
            len(declared_content) != _SHA256_LENGTH
            or hashlib.sha256(_canonical_json(body)).hexdigest()
            != declared_content
        ):
            raise ValueError("stationarity diagnostic content SHA-256 mismatch")
        view_id = str(payload["view_id"]).strip()
        sequence_index = int(payload["sequence_index"])
        if not view_id or sequence_index < 0:
            raise ValueError("stationarity diagnostic source identity is invalid")
        manifest_record = payload["source_session_manifest"]
        if not isinstance(manifest_record, dict) or set(manifest_record) != {
            "path",
            "sha256",
        }:
            raise ValueError("stationarity diagnostic manifest record is invalid")
        manifest = Path(str(manifest_record["path"])).resolve()
        manifest_sha = str(manifest_record["sha256"])
        if (
            len(manifest_sha) != _SHA256_LENGTH
            or not manifest.is_file()
            or _sha256(manifest) != manifest_sha
        ):
            raise ValueError("stationarity diagnostic source manifest changed")
        raw_trace = payload["robot_state_trace"]
        if not isinstance(raw_trace, list):
            raise ValueError("stationarity diagnostic trace must be an array")
        trace = _diagnostic_trace(
            tuple(
                _state_from_payload(
                    item,
                    schema_version=INFERENCE_STATIONARITY_SCHEMA_VERSION,
                )
                for item in raw_trace
            )
        )
        diagnostics = _diagnostics_payload(payload["sampler_diagnostics"])
        return StoredInferenceStationarityTrace(
            source,
            view_id,
            sequence_index,
            trace,
            manifest,
            manifest_sha,
            diagnostics,
            declared_content,
            _sha256(source),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid inference-stationarity diagnostic {source}: {exc}"
        ) from exc


def _evidence_payload(evidence: StationarityEvidence) -> dict[str, int | float]:
    return {
        "sample_count": evidence.sample_count,
        "duration_s": evidence.duration_s,
        "controller_duration_s": evidence.controller_duration_s,
        "max_sample_gap_s": evidence.max_sample_gap_s,
        "max_joint_delta_rad": evidence.max_joint_delta_rad,
        "max_tcp_translation_delta_m": evidence.max_tcp_translation_delta_m,
        "max_tcp_rotation_delta_rad": evidence.max_tcp_rotation_delta_rad,
        "goal_error_rad": evidence.goal_error_rad,
    }


def _positive_finite(value: object, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _nonnegative_finite(value: object, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _threshold_payload(
    *,
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
    maximum_robot_state_staleness_s: float,
) -> dict[str, float]:
    return {
        "max_joint_delta_rad": _nonnegative_finite(
            max_joint_delta_rad, label="max_joint_delta_rad"
        ),
        "max_tcp_translation_delta_m": _nonnegative_finite(
            max_tcp_translation_delta_m,
            label="max_tcp_translation_delta_m",
        ),
        "max_tcp_rotation_delta_rad": _nonnegative_finite(
            max_tcp_rotation_delta_rad,
            label="max_tcp_rotation_delta_rad",
        ),
        "maximum_robot_state_staleness_s": _positive_finite(
            maximum_robot_state_staleness_s,
            label="maximum_robot_state_staleness_s",
        ),
    }


def _recompute(
    reference: RobotState,
    trace: tuple[RobotState, ...],
    thresholds: dict[str, float],
) -> StationarityEvidence:
    return validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=thresholds["max_joint_delta_rad"],
        max_tcp_translation_delta_m=thresholds[
            "max_tcp_translation_delta_m"
        ],
        max_tcp_rotation_delta_rad=thresholds["max_tcp_rotation_delta_rad"],
        maximum_robot_state_staleness_s=thresholds[
            "maximum_robot_state_staleness_s"
        ],
    )


def write_inference_stationarity(
    path: str | Path,
    *,
    view_id: str,
    sequence_index: int,
    reference: RobotState,
    trace: tuple[RobotState, ...],
    evidence: StationarityEvidence,
    source_session_manifest: str | Path,
    max_joint_delta_rad: float,
    max_tcp_translation_delta_m: float,
    max_tcp_rotation_delta_rad: float,
    maximum_robot_state_staleness_s: float,
) -> StoredInferenceStationarity:
    """Write once and independently read back one stationary-inference proof."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Inference-stationarity evidence already exists: {destination}"
        )
    selected_view = str(view_id).strip()
    if not selected_view or sequence_index < 0:
        raise ValueError("stationarity source identity is invalid")
    manifest = Path(source_session_manifest).resolve()
    if not manifest.is_file():
        raise ValueError("stationarity source session manifest is missing")
    thresholds = _threshold_payload(
        max_joint_delta_rad=max_joint_delta_rad,
        max_tcp_translation_delta_m=max_tcp_translation_delta_m,
        max_tcp_rotation_delta_rad=max_tcp_rotation_delta_rad,
        maximum_robot_state_staleness_s=maximum_robot_state_staleness_s,
    )
    recomputed = _recompute(reference, tuple(trace), thresholds)
    if _evidence_payload(evidence) != _evidence_payload(recomputed):
        raise ValueError("stationarity evidence does not reproduce from its trace")
    body: dict[str, Any] = {
        "schema_version": INFERENCE_STATIONARITY_SCHEMA_VERSION,
        "depth_backend": "foundation_stereo",
        "view_id": selected_view,
        "sequence_index": int(sequence_index),
        "source_session_manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "thresholds": thresholds,
        "reference_robot_state": _state_payload(reference),
        "inference_robot_state_trace": [_state_payload(item) for item in trace],
        "evidence": _evidence_payload(recomputed),
    }
    content_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    payload = {**body, "content_sha256": content_sha256}
    _write_once_json(destination, payload)
    return read_inference_stationarity(destination)


def read_inference_stationarity(
    path: str | Path,
) -> StoredInferenceStationarity:
    """Verify schema, hashes, source manifest, trace freshness, and motion limits."""

    source = Path(path).resolve()
    try:
        payload = _strict_json_loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("stationarity asset root must be an object")
        expected_keys = {
            "schema_version",
            "depth_backend",
            "view_id",
            "sequence_index",
            "source_session_manifest",
            "thresholds",
            "reference_robot_state",
            "inference_robot_state_trace",
            "evidence",
            "content_sha256",
        }
        if set(payload) != expected_keys:
            raise ValueError("stationarity asset fields differ from schema")
        schema_version = int(payload["schema_version"])
        if (
            schema_version not in _SUPPORTED_SCHEMA_VERSIONS
            or payload["depth_backend"] != "foundation_stereo"
        ):
            raise ValueError("unsupported inference-stationarity contract")
        declared_content = str(payload["content_sha256"])
        body = {key: value for key, value in payload.items() if key != "content_sha256"}
        reproduced_content = hashlib.sha256(_canonical_json(body)).hexdigest()
        if (
            len(declared_content) != _SHA256_LENGTH
            or declared_content != reproduced_content
        ):
            raise ValueError("stationarity content SHA-256 mismatch")
        view_id = str(payload["view_id"]).strip()
        sequence_index = int(payload["sequence_index"])
        if not view_id or sequence_index < 0:
            raise ValueError("stationarity source identity is invalid")
        manifest_record = payload["source_session_manifest"]
        if not isinstance(manifest_record, dict) or set(manifest_record) != {
            "path",
            "sha256",
        }:
            raise ValueError("stationarity session-manifest record is invalid")
        manifest = Path(str(manifest_record["path"])).resolve()
        manifest_sha = str(manifest_record["sha256"])
        if (
            len(manifest_sha) != _SHA256_LENGTH
            or not manifest.is_file()
            or _sha256(manifest) != manifest_sha
        ):
            raise ValueError("stationarity source session manifest changed")
        raw_thresholds = payload["thresholds"]
        if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != {
            "max_joint_delta_rad",
            "max_tcp_translation_delta_m",
            "max_tcp_rotation_delta_rad",
            "maximum_robot_state_staleness_s",
        }:
            raise ValueError("stationarity thresholds differ from schema")
        thresholds = _threshold_payload(**raw_thresholds)
        reference = _state_from_payload(
            payload["reference_robot_state"],
            schema_version=schema_version,
        )
        raw_trace = payload["inference_robot_state_trace"]
        if not isinstance(raw_trace, list):
            raise ValueError("stationarity trace must be an array")
        trace = tuple(
            _state_from_payload(item, schema_version=schema_version)
            for item in raw_trace
        )
        evidence = _recompute(reference, trace, thresholds)
        if payload["evidence"] != _evidence_payload(evidence):
            raise ValueError("recorded stationarity metrics do not reproduce")
        return StoredInferenceStationarity(
            source,
            view_id,
            sequence_index,
            reference,
            trace,
            evidence,
            manifest,
            manifest_sha,
            (
                thresholds["max_joint_delta_rad"],
                thresholds["max_tcp_translation_delta_m"],
                thresholds["max_tcp_rotation_delta_rad"],
                thresholds["maximum_robot_state_staleness_s"],
            ),
            declared_content,
            _sha256(source),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid inference-stationarity asset {source}: {exc}") from exc
