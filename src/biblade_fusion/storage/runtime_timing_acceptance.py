"""Immutable physical acceptance for unknown-blade runtime timing budgets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from biblade_fusion.storage.motion_envelope_acceptance import (
    motion_control_contract_for_settings,
)
from biblade_fusion.storage.science_acceptance import (
    science_runtime_contract_for_settings,
)

RUNTIME_TIMING_ACCEPTANCE_SCHEMA_VERSION = 2
_ASSET_TYPE = "biblade_fusion.runtime_timing_acceptance"
_REPORT_SCHEMA = "biblade_fusion.runtime_timing_trials.v1"
_MANIFEST_SCHEMA = "biblade_fusion.runtime_timing_raw_manifest.v1"
_REPORT_NAME = "trial_report.json"
_MANIFEST_NAME = "raw_timing_manifest.json"
_TIMING_FIELDS = (
    "maximum_perception_cycle_duration_s",
    "maximum_operator_reposition_interval_s",
    "maximum_segment_execution_duration_s",
    "maximum_schema5_handoff_duration_s",
)
_CHECKS = (
    "target_gpu_and_filesystem_used",
    "target_robot_controller_used",
    "all_four_intervals_measured_monotonically",
    "cold_and_warm_trials_included",
    "raw_timing_evidence_archived",
    "independent_result_review_completed",
)
_EVIDENCE_ROLES = (
    "perception_cycle_trace",
    "operator_reposition_trace",
    "segment_execution_trace",
    "schema5_handoff_trace",
)
_TRACE_SCHEMA = "biblade_fusion.runtime_timing_trace.v2"
_MEASUREMENT_METHOD = "biblade_fusion.storage.measure_runtime_timing_trace.v2"
_MEASUREMENT_CONTRACT_SCHEMA = "biblade_fusion.runtime_timing_measurement_contract.v1"
_MEASUREMENT_SESSION_SCHEMA = "biblade_fusion.runtime_timing_measurement_session.v1"
_ROLE_TO_FIELD = dict(zip(_EVIDENCE_ROLES, _TIMING_FIELDS, strict=True))
_EVIDENCE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    """Read one stable regular-file snapshot without following a terminal symlink."""

    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(content) != after.st_size:
            raise ValueError(f"{label} changed while it was read")
        return content
    finally:
        os.close(descriptor)


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest string")
    text = value
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _without_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_paths(item)
            for key, item in value.items()
            if not str(key).endswith("path") and not str(key).endswith("_path")
        }
    if isinstance(value, (list, tuple)):
        return [_without_paths(item) for item in value]
    return value


def runtime_timing_contract_payload(settings: Any) -> dict[str, Any]:
    """Bind timing to perception, control and map semantics without local paths."""

    timing = settings.stop_and_capture.model_dump(mode="json")
    for field in (*_TIMING_FIELDS, "runtime_timing_acceptance_id"):
        timing.pop(field, None)
    timing.pop("runtime_timing_acceptance_path", None)
    return {
        "schema": "biblade_fusion.runtime_timing_contract.v1",
        "science_runtime_contract_sha256": science_runtime_contract_for_settings(settings),
        "motion_control_contract_sha256": motion_control_contract_for_settings(settings),
        "acquisition": _without_paths(settings.acquisition.model_dump(mode="json")),
        "stop_and_capture": _without_paths(timing),
        "occupancy": _without_paths(settings.occupancy.model_dump(mode="json")),
    }


def runtime_timing_contract_for_settings(settings: Any) -> str:
    return _sha256_bytes(_canonical_json(runtime_timing_contract_payload(settings)))


def _boot_id_sha256() -> str:
    """Return the current Linux boot identity without exposing the raw host UUID."""

    try:
        raw = Path("/proc/sys/kernel/random/boot_id").read_bytes().strip()
    except OSError as exc:
        raise ValueError("runtime timing measurement requires a readable Linux boot ID") from exc
    if not raw:
        raise ValueError("runtime timing measurement boot ID is empty")
    return _sha256_bytes(raw)


def _measurement_contract_sha256() -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "schema": _MEASUREMENT_CONTRACT_SCHEMA,
                "trace_schema": _TRACE_SCHEMA,
                "measurement_method": _MEASUREMENT_METHOD,
                "clock": "time.monotonic_ns",
                "operation_evidence": (
                    "canonical_json_asset_embedded_with_kind_sha256_and_size"
                ),
            }
        )
    )


def _operation_evidence_kind(payload: Mapping[str, Any]) -> str:
    raw_kind = next(
        (
            payload[key]
            for key in ("artifact_kind", "asset_type", "schema", "event_type")
            if key in payload
        ),
        None,
    )
    return _nonempty(raw_kind, label="timing operation evidence kind")


def _canonical_operation_evidence(path: Path) -> tuple[dict[str, Any], bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("timing operation evidence must be one regular JSON file")
    content = path.read_bytes()
    payload = _strict_json(content, label="timing operation evidence")
    if content != _canonical_json(payload) + b"\n":
        raise ValueError("timing operation evidence must use canonical JSON encoding")
    kind = _operation_evidence_kind(payload)
    return payload, content, kind


def timing_limits_for_settings(settings: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in _TIMING_FIELDS:
        raw = getattr(settings.stop_and_capture, field)
        if raw is None:
            raise ValueError(f"stop_and_capture.{field} is not measured")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"stop_and_capture.{field} must be finite and positive")
        values[field] = value
    return values


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    payload = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {label}: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def load_runtime_timing_acceptance_declaration(
    path: str | Path,
) -> dict[str, Any]:
    """Read the human declaration without accepting duplicate or non-finite JSON."""

    payload = _strict_json(
        Path(path).resolve().read_bytes(),
        label="runtime timing acceptance declaration",
    )
    expected = {"workcell_id", "operator_id", "accepted_at_utc", "checklist"}
    if set(payload) != expected:
        raise ValueError(
            "Runtime-timing declaration fields must be exactly: "
            + ", ".join(sorted(expected))
        )
    checklist = payload["checklist"]
    if not isinstance(checklist, Mapping):
        raise ValueError("Runtime-timing declaration checklist must be an object")
    _nonempty(payload["workcell_id"], label="runtime-timing declaration workcell_id")
    _nonempty(payload["operator_id"], label="runtime-timing declaration operator_id")
    _timestamp(
        payload["accepted_at_utc"],
        label="runtime-timing declaration accepted_at_utc",
    )
    if set(checklist) != set(_CHECKS) or not all(
        checklist[name] is True for name in _CHECKS
    ):
        raise ValueError("runtime-timing declaration checklist differs")
    return payload


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return result


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _validate_report(data: bytes, limits: Mapping[str, float]) -> tuple[int, str]:
    report = _strict_json(data, label="runtime timing trial report")
    if set(report) != {"schema", "captured_at_utc", "host_run_id", "trials"}:
        raise ValueError("runtime timing trial report fields differ")
    if report["schema"] != _REPORT_SCHEMA:
        raise ValueError("runtime timing trial report schema differs")
    _timestamp(report["captured_at_utc"], label="trial report captured_at_utc")
    host_run_id = _nonempty(report["host_run_id"], label="trial report host_run_id")
    trials = report["trials"]
    if not isinstance(trials, list) or len(trials) < 3:
        raise ValueError("runtime timing acceptance requires at least three complete trials")
    trial_ids: set[str] = set()
    modes: set[str] = set()
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping) or set(trial) != {
            "trial_id",
            "mode",
            *_TIMING_FIELDS,
        }:
            raise ValueError(f"runtime timing trial {index} fields differ")
        trial_id = _nonempty(trial["trial_id"], label=f"runtime timing trial {index} id")
        if trial_id in trial_ids:
            raise ValueError("runtime timing trial identifiers must be unique")
        trial_ids.add(trial_id)
        mode = str(trial["mode"])
        if mode not in {"cold", "warm"}:
            raise ValueError(f"runtime timing trial {index} mode must be cold or warm")
        modes.add(mode)
        for field in _TIMING_FIELDS:
            if isinstance(trial[field], bool) or not isinstance(trial[field], (int, float)):
                raise ValueError(f"runtime timing trial {index} {field} must be numeric")
            value = float(trial[field])
            if not math.isfinite(value) or value <= 0.0 or value > limits[field]:
                raise ValueError(f"runtime timing trial {index} exceeds {field}")
    if modes != {"cold", "warm"}:
        raise ValueError("runtime timing trials require at least one cold and one warm trial")
    if data != _canonical_json(report) + b"\n":
        raise ValueError("runtime timing trial report must use canonical JSON encoding")
    return len(trials), host_run_id


def _validate_manifest(data: bytes) -> tuple[int, str]:
    manifest = _strict_json(data, label="raw timing manifest")
    if set(manifest) != {"schema", "captured_at_utc", "host_run_id", "evidence"}:
        raise ValueError("raw timing manifest fields differ")
    if manifest["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("raw timing manifest schema differs")
    _timestamp(manifest["captured_at_utc"], label="raw timing manifest captured_at_utc")
    host_run_id = _nonempty(manifest["host_run_id"], label="raw timing manifest host_run_id")
    evidence = manifest["evidence"]
    if not isinstance(evidence, list) or len(evidence) < len(_EVIDENCE_ROLES):
        raise ValueError("raw timing manifest requires all four timing evidence roles")
    logical_keys: set[tuple[str, str]] = set()
    identities: set[tuple[str, int]] = set()
    observed_roles: set[str] = set()
    for index, entry in enumerate(evidence):
        if not isinstance(entry, Mapping) or set(entry) != {
            "role",
            "name",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"raw timing evidence {index} fields differ")
        role = _nonempty(entry["role"], label=f"raw timing evidence {index} role")
        if role not in _EVIDENCE_ROLES:
            raise ValueError(f"raw timing evidence {index} role is unsupported")
        observed_roles.add(role)
        name = _nonempty(entry["name"], label=f"raw timing evidence {index} name")
        if _EVIDENCE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"raw timing evidence {index} name is not a safe filename")
        logical_key = (role, name)
        if logical_key in logical_keys:
            raise ValueError("raw timing evidence logical keys must be unique")
        logical_keys.add(logical_key)
        digest = _digest(entry["sha256"], label=f"raw timing evidence {index} sha256")
        size = entry["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"raw timing evidence {index} size_bytes must be positive")
        identity = (digest, size)
        if identity in identities:
            raise ValueError("raw timing evidence identities must be unique")
        identities.add(identity)
    if observed_roles != set(_EVIDENCE_ROLES):
        raise ValueError("raw timing manifest does not cover all four timing roles")
    if data != _canonical_json(manifest) + b"\n":
        raise ValueError("raw timing manifest must use canonical JSON encoding")
    return len(evidence), host_run_id


def _aggregate_timing_traces(
    paths: tuple[Path, ...],
    *,
    limits: Mapping[str, float],
    runtime_contract_sha256: str,
    content_by_path: Mapping[Path, bytes] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, bytes], str]:
    if len(paths) < 3 * len(_EVIDENCE_ROLES):
        raise ValueError(
            "timing report aggregation requires four traces for at least three trials"
        )
    trials: dict[str, dict[str, object]] = {}
    evidence: list[dict[str, object]] = []
    evidence_content: dict[str, bytes] = {}
    host_run_id: str | None = None
    measurement_session_id: str | None = None
    measurement_session_payload: dict[str, Any] | None = None
    measurement_workcell_id: str | None = None
    boot_id_sha256: str | None = None
    latest_capture: datetime | None = None
    identities: set[tuple[str, int]] = set()
    operation_evidence_identities: set[str] = set()
    expected_runtime_contract = _digest(
        runtime_contract_sha256,
        label="runtime timing trace expected runtime contract",
    )
    for index, path in enumerate(paths):
        data = (
            path.read_bytes()
            if content_by_path is None
            else content_by_path[path]
        )
        trace = _strict_json(data, label=f"runtime timing trace {index}")
        if set(trace) != {
            "schema",
            "host_run_id",
            "trial_id",
            "mode",
            "role",
            "captured_at_utc",
            "duration_s",
            "measurement_method",
            "runtime_contract_sha256",
            "measurement_session_id",
            "measurement_session_payload",
            "boot_id_sha256",
            "operation_evidence_sha256",
            "operation_evidence_kind",
            "operation_evidence_size_bytes",
            "operation_evidence_payload",
            "measurement_contract_sha256",
            "started_monotonic_ns",
            "completed_monotonic_ns",
            "duration_ns",
        } or trace["schema"] != _TRACE_SCHEMA:
            raise ValueError(f"runtime timing trace {index} schema/fields differ")
        if data != _canonical_json(trace) + b"\n":
            raise ValueError(f"runtime timing trace {index} is not canonical JSON")
        current_host = _nonempty(
            trace["host_run_id"], label=f"timing trace {index} host"
        )
        if host_run_id is None:
            host_run_id = current_host
        elif current_host != host_run_id:
            raise ValueError("runtime timing traces span multiple host runs")
        if trace["measurement_method"] != _MEASUREMENT_METHOD:
            raise ValueError(f"runtime timing trace {index} measurement method differs")
        if _digest(
            trace["measurement_contract_sha256"],
            label=f"runtime timing trace {index} measurement contract",
        ) != _measurement_contract_sha256():
            raise ValueError(f"runtime timing trace {index} measurement contract differs")
        trace_contract = _digest(
            trace["runtime_contract_sha256"],
            label=f"runtime timing trace {index} runtime contract",
        )
        if trace_contract != expected_runtime_contract:
            raise ValueError(
                f"runtime timing trace {index} runtime contract differs from settings"
            )
        current_boot = _digest(
            trace["boot_id_sha256"],
            label=f"runtime timing trace {index} boot ID",
        )
        if boot_id_sha256 is None:
            boot_id_sha256 = current_boot
        elif current_boot != boot_id_sha256:
            raise ValueError("runtime timing traces span multiple boot identities")
        current_session = _digest(
            trace["measurement_session_id"],
            label=f"runtime timing trace {index} measurement session",
        )
        raw_session_payload = trace["measurement_session_payload"]
        if not isinstance(raw_session_payload, Mapping):
            raise ValueError(f"runtime timing trace {index} measurement session differs")
        (
            embedded_session_id,
            embedded_host,
            embedded_workcell,
            _embedded_created,
            embedded_runtime_contract,
            embedded_measurement_contract,
            embedded_boot,
        ) = _validate_measurement_session_payload(raw_session_payload)
        if (
            embedded_session_id != current_session
            or embedded_host != current_host
            or embedded_runtime_contract != expected_runtime_contract
            or embedded_measurement_contract != _measurement_contract_sha256()
            or embedded_boot != current_boot
        ):
            raise ValueError(f"runtime timing trace {index} measurement session binding differs")
        if measurement_session_id is None:
            measurement_session_id = current_session
            measurement_session_payload = dict(raw_session_payload)
            measurement_workcell_id = embedded_workcell
        elif current_session != measurement_session_id:
            raise ValueError("runtime timing traces span multiple measurement sessions")
        elif dict(raw_session_payload) != measurement_session_payload:
            raise ValueError("runtime timing traces changed the measurement session payload")
        elif embedded_workcell != measurement_workcell_id:
            raise ValueError("runtime timing traces span multiple workcells")
        operation_evidence = _digest(
            trace["operation_evidence_sha256"],
            label=f"runtime timing trace {index} operation evidence",
        )
        if operation_evidence in operation_evidence_identities:
            raise ValueError("runtime timing traces repeat operation evidence")
        operation_evidence_identities.add(operation_evidence)
        evidence_kind = _nonempty(
            trace["operation_evidence_kind"],
            label=f"runtime timing trace {index} operation evidence kind",
        )
        evidence_size = trace["operation_evidence_size_bytes"]
        evidence_payload = trace["operation_evidence_payload"]
        if (
            isinstance(evidence_size, bool)
            or not isinstance(evidence_size, int)
            or evidence_size <= 0
            or not isinstance(evidence_payload, Mapping)
        ):
            raise ValueError(f"runtime timing trace {index} operation evidence differs")
        if evidence_kind != _operation_evidence_kind(evidence_payload):
            raise ValueError(f"runtime timing trace {index} operation evidence kind differs")
        evidence_bytes = _canonical_json(evidence_payload) + b"\n"
        if (
            len(evidence_bytes) != evidence_size
            or _sha256_bytes(evidence_bytes) != operation_evidence
        ):
            raise ValueError(f"runtime timing trace {index} operation evidence identity differs")
        started_ns = trace["started_monotonic_ns"]
        completed_ns = trace["completed_monotonic_ns"]
        duration_ns = trace["duration_ns"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (started_ns, completed_ns, duration_ns)
        ) or started_ns < 0 or completed_ns <= started_ns or duration_ns <= 0:
            raise ValueError(f"runtime timing trace {index} monotonic nanoseconds differ")
        if completed_ns - started_ns != duration_ns:
            raise ValueError(f"runtime timing trace {index} duration nanoseconds differ")
        trial_id = _nonempty(trace["trial_id"], label=f"timing trace {index} trial_id")
        mode = str(trace["mode"])
        if mode not in {"cold", "warm"}:
            raise ValueError(f"runtime timing trace {index} mode must be cold or warm")
        role = str(trace["role"])
        if role not in _ROLE_TO_FIELD:
            raise ValueError(f"runtime timing trace {index} role is unsupported")
        raw_duration = trace["duration_s"]
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
            raise ValueError(f"runtime timing trace {index} duration must be numeric")
        duration = float(raw_duration)
        if (
            not math.isfinite(duration)
            or duration <= 0.0
            or duration != duration_ns / 1_000_000_000.0
        ):
            raise ValueError(f"runtime timing trace {index} duration must be positive")
        captured = _timestamp(
            trace["captured_at_utc"],
            label=f"runtime timing trace {index} captured_at_utc",
        )
        latest_capture = captured if latest_capture is None else max(latest_capture, captured)
        trial = trials.setdefault(trial_id, {"trial_id": trial_id, "mode": mode})
        if trial["mode"] != mode:
            raise ValueError(f"runtime timing trial {trial_id} mixes cold and warm traces")
        field = _ROLE_TO_FIELD[role]
        if field in trial:
            raise ValueError(f"runtime timing trial {trial_id} repeats role {role}")
        trial[field] = duration
        logical_name = f"{trial_id}__{role}.json"
        if _EVIDENCE_NAME_PATTERN.fullmatch(logical_name) is None:
            raise ValueError("runtime timing trial ID cannot form a safe evidence filename")
        digest = _sha256_bytes(data)
        identity = (digest, len(data))
        if logical_name in evidence_content or identity in identities:
            raise ValueError("runtime timing trace logical key or identity is duplicated")
        evidence_content[logical_name] = data
        identities.add(identity)
        evidence.append(
            {
                "role": role,
                "name": logical_name,
                "sha256": digest,
                "size_bytes": len(data),
            }
        )
    if len(trials) < 3 or any(
        set(trial) != {"trial_id", "mode", *_TIMING_FIELDS}
        for trial in trials.values()
    ):
        raise ValueError("every timing trial must contain each of the four timing roles")
    assert (
        host_run_id is not None
        and latest_capture is not None
        and measurement_workcell_id is not None
    )
    captured_at = latest_capture.astimezone(UTC).isoformat()
    report_payload: dict[str, object] = {
        "schema": _REPORT_SCHEMA,
        "captured_at_utc": captured_at,
        "host_run_id": host_run_id,
        "trials": [trials[name] for name in sorted(trials)],
    }
    manifest_payload: dict[str, object] = {
        "schema": _MANIFEST_SCHEMA,
        "captured_at_utc": captured_at,
        "host_run_id": host_run_id,
        "evidence": sorted(
            evidence,
            key=lambda item: (str(item["role"]), str(item["name"])),
        ),
    }
    _validate_report(_canonical_json(report_payload) + b"\n", limits)
    _validate_manifest(_canonical_json(manifest_payload) + b"\n")
    return report_payload, manifest_payload, evidence_content, measurement_workcell_id


def _publish_new_files(files: tuple[tuple[Path, bytes], ...]) -> None:
    """Exclusively publish complete files and remove this call's partial pair."""

    temporary: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for destination, content in files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_name(
                f".{destination.name}.{uuid4().hex}.partial"
            )
            with staged.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.append((staged, destination))
        for staged, destination in temporary:
            os.link(staged, destination)
            published.append((staged, destination))
        for parent in {destination.parent for _, destination in temporary}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for staged, destination in reversed(published):
            try:
                staged_stat = staged.stat()
                destination_stat = destination.stat()
                if (
                    staged_stat.st_dev == destination_stat.st_dev
                    and staged_stat.st_ino == destination_stat.st_ino
                ):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for staged, _ in temporary:
            staged.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_destination_claim(
    destination: Path,
    *,
    asset_label: str = "runtime timing acceptance",
) -> Iterator[None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_name(f".{destination.name}.claim")
    claim_identity: tuple[int, int] | None = None
    claim_payload = (
        f"biblade_fusion.{asset_label.replace(' ', '_')}.claim.{uuid4().hex}\n".encode()
    )
    try:
        with claim.open("xb") as stream:
            stream.write(claim_payload)
            stream.flush()
            os.fsync(stream.fileno())
            stat = os.fstat(stream.fileno())
            claim_identity = (stat.st_dev, stat.st_ino)
    except FileExistsError as exc:
        raise FileExistsError(
            f"{asset_label} destination is already claimed: {destination}"
        ) from exc
    _fsync_directory(destination.parent)
    try:
        if destination.exists():
            raise FileExistsError(
                f"{asset_label} already exists: {destination}"
            )
        yield
    finally:
        try:
            current = claim.stat(follow_symlinks=False)
            if (
                claim_identity == (current.st_dev, current.st_ino)
                and claim.read_bytes() == claim_payload
            ):
                claim.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(destination.parent)


@dataclass(frozen=True, slots=True)
class StoredRuntimeTimingMeasurementSession:
    path: Path
    measurement_session_id: str
    host_run_id: str
    workcell_id: str
    created_at_utc: datetime
    runtime_contract_sha256: str
    measurement_contract_sha256: str
    boot_id_sha256: str
    payload: dict[str, Any]

    def assert_current(self, settings: Any) -> None:
        if self.runtime_contract_sha256 != runtime_timing_contract_for_settings(settings):
            raise ValueError("runtime timing measurement session contract changed")
        if self.measurement_contract_sha256 != _measurement_contract_sha256():
            raise ValueError("runtime timing measurement implementation changed")
        if self.boot_id_sha256 != _boot_id_sha256():
            raise ValueError("runtime timing measurement session belongs to another boot")


def _validate_measurement_session_payload(
    payload: Mapping[str, Any],
) -> tuple[str, str, str, datetime, str, str, str]:
    expected = {
        "schema",
        "host_run_id",
        "workcell_id",
        "created_at_utc",
        "runtime_contract_sha256",
        "measurement_contract_sha256",
        "boot_id_sha256",
        "motion_authorized",
        "measurement_session_id",
    }
    if set(payload) != expected or payload["schema"] != _MEASUREMENT_SESSION_SCHEMA:
        raise ValueError("runtime timing measurement session schema differs")
    if payload["motion_authorized"] is not False:
        raise ValueError("runtime timing measurement session cannot authorize motion")
    session_id = _digest(
        payload["measurement_session_id"],
        label="measurement_session_id",
    )
    unsigned = dict(payload)
    unsigned.pop("measurement_session_id")
    if session_id != _sha256_bytes(_canonical_json(unsigned)):
        raise ValueError("runtime timing measurement session identity differs")
    return (
        session_id,
        _nonempty(payload["host_run_id"], label="measurement session host_run_id"),
        _nonempty(payload["workcell_id"], label="measurement session workcell_id"),
        _timestamp(payload["created_at_utc"], label="measurement session created_at_utc"),
        _digest(
            payload["runtime_contract_sha256"],
            label="measurement session runtime contract",
        ),
        _digest(
            payload["measurement_contract_sha256"],
            label="measurement session measurement contract",
        ),
        _digest(payload["boot_id_sha256"], label="measurement session boot ID"),
    )


def read_runtime_timing_measurement_session(
    path: str | Path,
) -> StoredRuntimeTimingMeasurementSession:
    source = Path(path).resolve()
    content = source.read_bytes()
    payload = _strict_json(content, label="runtime timing measurement session")
    if content != _canonical_json(payload) + b"\n":
        raise ValueError("runtime timing measurement session must use canonical JSON")
    (
        session_id,
        host_run_id,
        workcell_id,
        created_at_utc,
        runtime_contract,
        measurement_contract,
        boot_id,
    ) = _validate_measurement_session_payload(payload)
    return StoredRuntimeTimingMeasurementSession(
        source,
        session_id,
        host_run_id,
        workcell_id,
        created_at_utc,
        runtime_contract,
        measurement_contract,
        boot_id,
        payload,
    )


def write_runtime_timing_measurement_session(
    path: str | Path,
    *,
    settings: Any,
    host_run_id: str,
    workcell_id: str,
    created_at_utc: datetime | None = None,
) -> StoredRuntimeTimingMeasurementSession:
    created = created_at_utc or _utc_now()
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("measurement session created_at_utc must be timezone-aware")
    payload: dict[str, Any] = {
        "schema": _MEASUREMENT_SESSION_SCHEMA,
        "host_run_id": _nonempty(host_run_id, label="measurement session host_run_id"),
        "workcell_id": _nonempty(workcell_id, label="measurement session workcell_id"),
        "created_at_utc": created.astimezone(UTC).isoformat(),
        "runtime_contract_sha256": runtime_timing_contract_for_settings(settings),
        "measurement_contract_sha256": _measurement_contract_sha256(),
        "boot_id_sha256": _boot_id_sha256(),
        "motion_authorized": False,
    }
    payload["measurement_session_id"] = _sha256_bytes(_canonical_json(payload))
    destination = Path(path).resolve()
    with _exclusive_destination_claim(
        destination,
        asset_label="runtime timing measurement session",
    ):
        _publish_new_files(((destination, _canonical_json(payload) + b"\n"),))
    return read_runtime_timing_measurement_session(destination)


def measure_runtime_timing_trace(
    path: str | Path,
    *,
    trial_id: str,
    mode: str,
    role: str,
    settings: Any,
    measurement_session: str | Path,
    operation_evidence_path: Callable[[Any], str | Path],
    operation: Callable[[], Any],
    monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
    utc_clock: Callable[[], datetime] = _utc_now,
) -> tuple[Any, Path]:
    """Measure one successful operation and bind it to its runtime and evidence."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite a runtime timing trace: {destination}"
        )
    with _exclusive_destination_claim(
        destination,
        asset_label="runtime timing trace",
    ):
        stored_session = read_runtime_timing_measurement_session(measurement_session)
        stored_session.assert_current(settings)
        host = stored_session.host_run_id
        trial = _nonempty(trial_id, label="timing trace trial_id")
        if mode not in {"cold", "warm"}:
            raise ValueError("timing trace mode must be cold or warm")
        if role not in _ROLE_TO_FIELD:
            raise ValueError("timing trace role is unsupported")
        runtime_contract = runtime_timing_contract_for_settings(settings)
        boot_identity = _boot_id_sha256()
        started_ns = monotonic_ns_clock()
        if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns < 0:
            raise ValueError("timing trace monotonic start must be non-negative integer ns")
        result = operation()
        completed_ns = monotonic_ns_clock()
        if (
            isinstance(completed_ns, bool)
            or not isinstance(completed_ns, int)
            or completed_ns <= started_ns
        ):
            raise ValueError("timing trace monotonic interval must be positive integer ns")
        duration_ns = completed_ns - started_ns
        duration = duration_ns / 1_000_000_000.0
        captured = utc_clock()
        if captured.tzinfo is None or captured.utcoffset() is None:
            raise ValueError("timing trace UTC clock must be timezone-aware")
        if runtime_timing_contract_for_settings(settings) != runtime_contract:
            raise ValueError("runtime timing contract changed during measurement")
        if _boot_id_sha256() != boot_identity:
            raise ValueError("host boot identity changed during measurement")
        evidence_payload, evidence_content, evidence_kind = _canonical_operation_evidence(
            Path(operation_evidence_path(result)).resolve()
        )
        operation_evidence = _sha256_bytes(evidence_content)
        payload = {
            "schema": _TRACE_SCHEMA,
            "host_run_id": host,
            "trial_id": trial,
            "mode": mode,
            "role": role,
            "captured_at_utc": captured.astimezone(UTC).isoformat(),
            "duration_s": duration,
            "measurement_method": _MEASUREMENT_METHOD,
            "runtime_contract_sha256": runtime_contract,
            "measurement_session_id": stored_session.measurement_session_id,
            "measurement_session_payload": stored_session.payload,
            "boot_id_sha256": boot_identity,
            "operation_evidence_sha256": operation_evidence,
            "operation_evidence_kind": evidence_kind,
            "operation_evidence_size_bytes": len(evidence_content),
            "operation_evidence_payload": evidence_payload,
            "measurement_contract_sha256": _measurement_contract_sha256(),
            "started_monotonic_ns": started_ns,
            "completed_monotonic_ns": completed_ns,
            "duration_ns": duration_ns,
        }
        content = _canonical_json(payload) + b"\n"
        try:
            _publish_new_files(((destination, content),))
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite a runtime timing trace: {destination}"
            ) from exc
    return result, destination


def build_runtime_timing_reports(
    trace_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    settings: Any,
    trial_report: str | Path,
    raw_timing_manifest: str | Path,
) -> tuple[Path, Path]:
    """Aggregate canonical role traces into the two acceptance input assets."""

    paths = tuple(Path(path).resolve() for path in trace_paths)
    report_payload, manifest_payload, _, _ = _aggregate_timing_traces(
        paths,
        limits=timing_limits_for_settings(settings),
        runtime_contract_sha256=runtime_timing_contract_for_settings(settings),
    )
    report_bytes = _canonical_json(report_payload) + b"\n"
    manifest_bytes = _canonical_json(manifest_payload) + b"\n"
    report_path = Path(trial_report).resolve()
    manifest_path = Path(raw_timing_manifest).resolve()
    if report_path == manifest_path:
        raise ValueError("trial report and raw timing manifest paths must differ")
    try:
        _publish_new_files(
            ((report_path, report_bytes), (manifest_path, manifest_bytes))
        )
    except FileExistsError as exc:
        raise FileExistsError(
            "refusing to overwrite a runtime timing report asset"
        ) from exc
    return report_path, manifest_path


@dataclass(frozen=True, slots=True)
class StoredRuntimeTimingAcceptance:
    path: Path
    acceptance_id: str
    workcell_id: str
    operator_id: str
    accepted_at_utc: datetime
    timing_limits_s: dict[str, float]
    trial_count: int
    raw_evidence_count: int
    runtime_contract_sha256: str
    metadata_sha256: str

    def assert_matches(self, *, settings: Any, acceptance_id: str) -> None:
        if self.acceptance_id != _digest(acceptance_id, label="acceptance_id"):
            raise ValueError("runtime timing acceptance identity mismatch")
        if self.timing_limits_s != timing_limits_for_settings(settings):
            raise ValueError("runtime timing acceptance limits differ from settings")
        if self.runtime_contract_sha256 != runtime_timing_contract_for_settings(settings):
            raise ValueError("runtime timing acceptance contract differs from settings")


@dataclass(frozen=True, slots=True)
class RuntimeTimingAcceptanceAuthority:
    acceptance_path: Path
    acceptance_id: str
    metadata_sha256: str
    runtime_contract_sha256: str
    timing_limits_s: dict[str, float]

    @classmethod
    def from_stored(
        cls,
        stored: StoredRuntimeTimingAcceptance,
    ) -> RuntimeTimingAcceptanceAuthority:
        return cls(
            stored.path.resolve(),
            stored.acceptance_id,
            stored.metadata_sha256,
            stored.runtime_contract_sha256,
            dict(stored.timing_limits_s),
        )

    @classmethod
    def from_payload(cls, payload: object) -> RuntimeTimingAcceptanceAuthority:
        if not isinstance(payload, Mapping) or set(payload) != {
            "acceptance_path",
            "acceptance_id",
            "metadata_sha256",
            "runtime_contract_sha256",
            "timing_limits_s",
        }:
            raise ValueError("runtime timing authority payload fields differ")
        raw_path = payload["acceptance_path"]
        if not isinstance(raw_path, str):
            raise ValueError("runtime timing authority path must be a string")
        path = Path(raw_path)
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError("runtime timing authority path must be canonical and absolute")
        raw_limits = payload["timing_limits_s"]
        if not isinstance(raw_limits, Mapping) or set(raw_limits) != set(_TIMING_FIELDS):
            raise ValueError("runtime timing authority limit fields differ")
        for field in _TIMING_FIELDS:
            value = raw_limits[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("runtime timing authority limits must be numeric")
        limits = {field: float(raw_limits[field]) for field in _TIMING_FIELDS}
        if any(not math.isfinite(value) or value <= 0.0 for value in limits.values()):
            raise ValueError("runtime timing authority limits must be finite and positive")
        return cls(
            path,
            _digest(payload["acceptance_id"], label="acceptance_id"),
            _digest(payload["metadata_sha256"], label="metadata_sha256"),
            _digest(payload["runtime_contract_sha256"], label="runtime_contract_sha256"),
            limits,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "acceptance_path": str(self.acceptance_path),
            "acceptance_id": self.acceptance_id,
            "metadata_sha256": self.metadata_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "timing_limits_s": dict(self.timing_limits_s),
        }

    def assert_acceptance_asset_current(self) -> None:
        stored = read_runtime_timing_acceptance(self.acceptance_path)
        current = type(self).from_stored(stored)
        if current != self:
            raise ValueError("runtime timing acceptance authority changed on disk")

    def assert_current(self, settings: Any) -> None:
        self.assert_acceptance_asset_current()
        if self.timing_limits_s != timing_limits_for_settings(settings):
            raise ValueError("runtime timing authority limits differ from settings")
        if self.runtime_contract_sha256 != runtime_timing_contract_for_settings(settings):
            raise ValueError("runtime timing authority contract differs from settings")
        configured_path = settings.stop_and_capture.runtime_timing_acceptance_path
        configured_id = settings.stop_and_capture.runtime_timing_acceptance_id
        if (
            configured_path is None
            or configured_id is None
            or configured_path.resolve() != self.acceptance_path
            or configured_id != self.acceptance_id
        ):
            raise ValueError("runtime timing authority differs from configured path/identity")


def load_runtime_timing_acceptance_authority(
    settings: Any,
) -> RuntimeTimingAcceptanceAuthority:
    path = settings.stop_and_capture.runtime_timing_acceptance_path
    acceptance_id = settings.stop_and_capture.runtime_timing_acceptance_id
    if path is None or acceptance_id is None:
        raise ValueError("runtime timing acceptance path/identity is not configured")
    stored = read_runtime_timing_acceptance(path)
    stored.assert_matches(settings=settings, acceptance_id=acceptance_id)
    authority = RuntimeTimingAcceptanceAuthority.from_stored(stored)
    authority.assert_current(settings)
    return authority


def _write_claimed_runtime_timing_acceptance(
    destination: Path,
    *,
    settings: Any,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    trial_report: str | Path,
    raw_timing_manifest: str | Path,
    raw_timing_traces: list[str | Path] | tuple[str | Path, ...],
    checklist: Mapping[str, bool],
) -> StoredRuntimeTimingAcceptance:
    workcell = workcell_id.strip()
    operator = operator_id.strip()
    if not workcell or not operator:
        raise ValueError("workcell_id and operator_id must be non-empty")
    if accepted_at_utc.tzinfo is None or accepted_at_utc.utcoffset() is None:
        raise ValueError("accepted_at_utc must be timezone-aware")
    if set(checklist) != set(_CHECKS) or not all(checklist[name] is True for name in _CHECKS):
        raise ValueError("all runtime timing acceptance checks must be true")
    report_bytes = Path(trial_report).resolve().read_bytes()
    manifest_bytes = Path(raw_timing_manifest).resolve().read_bytes()
    limits = timing_limits_for_settings(settings)
    trial_count, report_run_id = _validate_report(report_bytes, limits)
    evidence_count, manifest_run_id = _validate_manifest(manifest_bytes)
    if report_run_id != manifest_run_id:
        raise ValueError("trial report and raw timing manifest host_run_id differ")
    (
        derived_report,
        derived_manifest,
        evidence_content,
        measurement_workcell_id,
    ) = _aggregate_timing_traces(
        tuple(Path(path).resolve() for path in raw_timing_traces),
        limits=limits,
        runtime_contract_sha256=runtime_timing_contract_for_settings(settings),
    )
    if report_bytes != _canonical_json(derived_report) + b"\n":
        raise ValueError("trial report does not reproduce from the supplied raw traces")
    if manifest_bytes != _canonical_json(derived_manifest) + b"\n":
        raise ValueError("raw timing manifest does not reproduce from supplied traces")
    if measurement_workcell_id != workcell:
        raise ValueError(
            "runtime timing measurement session workcell differs from acceptance workcell"
        )
    payload = {
        "schema_version": RUNTIME_TIMING_ACCEPTANCE_SCHEMA_VERSION,
        "asset_type": _ASSET_TYPE,
        "workcell_id": workcell,
        "operator_id": operator,
        "accepted_at_utc": accepted_at_utc.astimezone(UTC).isoformat(),
        "timing_limits_s": limits,
        "trial_count": trial_count,
        "raw_evidence_count": evidence_count,
        "runtime_contract_sha256": runtime_timing_contract_for_settings(settings),
        "trial_report": {
            "file": _REPORT_NAME,
            "sha256": _sha256_bytes(report_bytes),
            "size_bytes": len(report_bytes),
        },
        "raw_timing_manifest": {
            "file": _MANIFEST_NAME,
            "sha256": _sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
        "checklist": {name: True for name in _CHECKS},
        "motion_authorized": False,
    }
    payload["acceptance_id"] = _sha256_bytes(_canonical_json(payload))
    destination.mkdir()
    incomplete = destination / ".incomplete"
    incomplete_token = uuid4().hex
    try:
        with incomplete.open("x", encoding="ascii") as stream:
            stream.write(incomplete_token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        (destination / _REPORT_NAME).write_bytes(report_bytes)
        (destination / _MANIFEST_NAME).write_bytes(manifest_bytes)
        evidence_root = destination / "evidence"
        evidence_root.mkdir()
        for name, content in sorted(evidence_content.items()):
            (evidence_root / name).write_bytes(content)
        metadata = destination / "metadata.json"
        metadata.write_bytes(_canonical_json(payload) + b"\n")
        for item in (
            destination / _REPORT_NAME,
            destination / _MANIFEST_NAME,
            *evidence_root.iterdir(),
            metadata,
        ):
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(evidence_root)
        _fsync_directory(destination)
        incomplete.unlink()
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        owns_destination = False
        with suppress(FileNotFoundError, OSError, UnicodeError):
            owns_destination = incomplete.read_text(encoding="ascii") == incomplete_token + "\n"
        if owns_destination:
            evidence_root = destination / "evidence"
            if evidence_root.is_dir():
                for item in evidence_root.iterdir():
                    item.unlink()
                evidence_root.rmdir()
            for item in destination.iterdir():
                item.unlink()
            destination.rmdir()
            _fsync_directory(destination.parent)
        raise
    return read_runtime_timing_acceptance(destination)


def write_runtime_timing_acceptance(
    path: str | Path,
    *,
    settings: Any,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    trial_report: str | Path,
    raw_timing_manifest: str | Path,
    raw_timing_traces: list[str | Path] | tuple[str | Path, ...],
    checklist: Mapping[str, bool],
) -> StoredRuntimeTimingAcceptance:
    destination = Path(path).resolve()
    with _exclusive_destination_claim(destination):
        return _write_claimed_runtime_timing_acceptance(
            destination,
            settings=settings,
            workcell_id=workcell_id,
            operator_id=operator_id,
            accepted_at_utc=accepted_at_utc,
            trial_report=trial_report,
            raw_timing_manifest=raw_timing_manifest,
            raw_timing_traces=raw_timing_traces,
            checklist=checklist,
        )


def read_runtime_timing_acceptance(path: str | Path) -> StoredRuntimeTimingAcceptance:
    root = Path(path).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime timing acceptance root must be a non-symlink directory")
    if (root / ".incomplete").exists():
        raise ValueError("runtime timing acceptance publication is incomplete")
    if {item.name for item in root.iterdir()} != {
        _REPORT_NAME,
        _MANIFEST_NAME,
        "metadata.json",
        "evidence",
    }:
        raise ValueError("runtime timing acceptance root file set changed")
    metadata_bytes = _read_regular_file_once(
        root / "metadata.json",
        label="runtime timing acceptance metadata",
    )
    report_bytes = _read_regular_file_once(
        root / _REPORT_NAME,
        label="runtime timing trial report",
    )
    manifest_bytes = _read_regular_file_once(
        root / _MANIFEST_NAME,
        label="runtime timing raw manifest",
    )
    payload = _strict_json(metadata_bytes, label="metadata")
    expected = {
        "schema_version",
        "asset_type",
        "workcell_id",
        "operator_id",
        "accepted_at_utc",
        "timing_limits_s",
        "trial_count",
        "raw_evidence_count",
        "runtime_contract_sha256",
        "trial_report",
        "raw_timing_manifest",
        "checklist",
        "motion_authorized",
        "acceptance_id",
    }
    if set(payload) != expected:
        raise ValueError("runtime timing acceptance fields differ from schema")
    if metadata_bytes != _canonical_json(payload) + b"\n":
        raise ValueError("runtime timing acceptance metadata must use canonical JSON encoding")
    acceptance_id = _digest(payload.pop("acceptance_id"), label="acceptance_id")
    if acceptance_id != _sha256_bytes(_canonical_json(payload)):
        raise ValueError("runtime timing acceptance identity mismatch")
    if (
        payload["schema_version"] != RUNTIME_TIMING_ACCEPTANCE_SCHEMA_VERSION
        or payload["asset_type"] != _ASSET_TYPE
        or payload["motion_authorized"] is not False
    ):
        raise ValueError("runtime timing acceptance schema/declaration differs")
    workcell_id = _nonempty(payload["workcell_id"], label="workcell_id")
    operator_id = _nonempty(payload["operator_id"], label="operator_id")
    for field in ("trial_count", "raw_evidence_count"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"runtime timing acceptance {field} must be a positive integer")
    if not isinstance(payload["timing_limits_s"], Mapping):
        raise ValueError("runtime timing acceptance limits must be an object")
    if set(payload["timing_limits_s"]) != set(_TIMING_FIELDS):
        raise ValueError("runtime timing acceptance limit fields differ")
    for field in _TIMING_FIELDS:
        raw = payload["timing_limits_s"][field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("runtime timing acceptance limits must be numeric")
    limits = {field: float(payload["timing_limits_s"][field]) for field in _TIMING_FIELDS}
    if any(not math.isfinite(value) or value <= 0.0 for value in limits.values()):
        raise ValueError("runtime timing acceptance limits must be finite and positive")
    for label, filename in (
        ("trial_report", _REPORT_NAME),
        ("raw_timing_manifest", _MANIFEST_NAME),
    ):
        record = payload[label]
        if not isinstance(record, Mapping) or set(record) != {"file", "sha256", "size_bytes"}:
            raise ValueError(f"runtime timing {label} record differs")
        if (
            not isinstance(record["file"], str)
            or not isinstance(record["sha256"], str)
            or isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] <= 0
        ):
            raise ValueError(f"runtime timing {label} record types differ")
        content = report_bytes if filename == _REPORT_NAME else manifest_bytes
        if record["file"] != filename:
            raise ValueError(f"runtime timing {label} copy is missing")
        if (
            _sha256_bytes(content) != record["sha256"]
            or len(content) != record["size_bytes"]
        ):
            raise ValueError(f"runtime timing {label} copy changed")
    trial_count, report_run_id = _validate_report(report_bytes, limits)
    if trial_count != payload["trial_count"]:
        raise ValueError("runtime timing trial count changed")
    evidence_count, manifest_run_id = _validate_manifest(manifest_bytes)
    if evidence_count != payload["raw_evidence_count"]:
        raise ValueError("runtime timing raw evidence count changed")
    if report_run_id != manifest_run_id:
        raise ValueError("trial report and raw timing manifest host_run_id differ")
    manifest = _strict_json(manifest_bytes, label="raw timing manifest")
    evidence_root = root / "evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError("runtime timing raw evidence directory is missing")
    expected_names = {str(item["name"]) for item in manifest["evidence"]}
    actual_names = {item.name for item in evidence_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("runtime timing raw evidence file set changed")
    evidence_content_by_path: dict[Path, bytes] = {}
    for index, record in enumerate(manifest["evidence"]):
        evidence_path = evidence_root / str(record["name"])
        content = _read_regular_file_once(
            evidence_path,
            label=f"runtime timing raw evidence {index}",
        )
        if (
            _sha256_bytes(content) != record["sha256"]
            or len(content) != record["size_bytes"]
        ):
            raise ValueError(f"runtime timing raw evidence {index} changed")
        evidence_content_by_path[evidence_path] = content
    (
        derived_report,
        derived_manifest,
        _,
        measurement_workcell_id,
    ) = _aggregate_timing_traces(
        tuple(evidence_root / name for name in sorted(expected_names)),
        limits=limits,
        runtime_contract_sha256=_digest(
            payload["runtime_contract_sha256"],
            label="runtime_contract_sha256",
        ),
        content_by_path=evidence_content_by_path,
    )
    if measurement_workcell_id != workcell_id:
        raise ValueError(
            "runtime timing measurement session workcell differs from acceptance workcell"
        )
    if report_bytes != _canonical_json(derived_report) + b"\n":
        raise ValueError("runtime timing trial report no longer reproduces from evidence")
    if manifest_bytes != _canonical_json(derived_manifest) + b"\n":
        raise ValueError("raw timing manifest no longer reproduces from evidence")
    if not isinstance(payload["checklist"], Mapping) or set(payload["checklist"]) != set(
        _CHECKS
    ) or not all(
        payload["checklist"][name] is True for name in _CHECKS
    ):
        raise ValueError("runtime timing checklist differs")
    accepted = _timestamp(payload["accepted_at_utc"], label="accepted_at_utc")
    return StoredRuntimeTimingAcceptance(
        root,
        acceptance_id,
        workcell_id,
        operator_id,
        accepted,
        limits,
        trial_count,
        evidence_count,
        _digest(payload["runtime_contract_sha256"], label="runtime_contract_sha256"),
        _sha256_bytes(metadata_bytes),
    )


__all__ = [
    "RUNTIME_TIMING_ACCEPTANCE_SCHEMA_VERSION",
    "RuntimeTimingAcceptanceAuthority",
    "StoredRuntimeTimingAcceptance",
    "StoredRuntimeTimingMeasurementSession",
    "build_runtime_timing_reports",
    "load_runtime_timing_acceptance_declaration",
    "measure_runtime_timing_trace",
    "read_runtime_timing_acceptance",
    "read_runtime_timing_measurement_session",
    "load_runtime_timing_acceptance_authority",
    "runtime_timing_contract_for_settings",
    "runtime_timing_contract_payload",
    "timing_limits_for_settings",
    "write_runtime_timing_acceptance",
    "write_runtime_timing_measurement_session",
]
