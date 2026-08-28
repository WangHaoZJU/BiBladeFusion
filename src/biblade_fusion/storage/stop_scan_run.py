"""Append-only evidence events for one stop-and-scan run.

``run.json`` is deliberately only a navigation hint.  The authoritative record is
the immutable, forward-linked sequence in ``events/``; readers discover those files
directly and verify the complete chain without trusting the index.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

STOP_SCAN_RUN_INDEX_SCHEMA_VERSION = 1
_EVENT_FILENAME_DIGITS = 8
_EVENT_KEYS = frozenset(
    {
        "run_id",
        "sequence",
        "phase",
        "cycle_index",
        "event_type",
        "created_at_utc",
        "payload",
        "previous_event_sha256",
        "event_sha256",
    }
)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVENT_FILENAME_PATTERN = re.compile(r"([0-9]{8})\.json")


class StopScanRunFormatError(ValueError):
    """A stop-and-scan event chain is malformed, incomplete, or corrupted."""


def _validate_run_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("run_id must be a string")
    run_id = value
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must contain 1-128 ASCII letters, digits, '.', '_' or '-', "
            "and must start with a letter or digit"
        )
    return run_id


def _validate_token(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    token = value
    if _TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(f"{label} must be a lowercase snake-case token")
    return token


def _non_negative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _canonical_utc_text(value: datetime | str, *, require_canonical: bool) -> str:
    if isinstance(value, datetime):
        parsed = value
        original = None
    elif isinstance(value, str):
        original = value
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("created_at_utc must be an ISO-8601 datetime") from exc
    else:
        raise ValueError("created_at_utc must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0.0:
        raise ValueError("created_at_utc must use UTC, not another offset")
    canonical = parsed.astimezone(UTC).isoformat()
    if require_canonical and original != canonical:
        raise ValueError("created_at_utc must use canonical UTC '+00:00' notation")
    return canonical


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _normalise_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("event payload must be a JSON object")
    _validate_json_value(value, path="payload")
    # The round trip gives the frozen event its own deep copy while also exercising
    # Python's strict JSON encoder with NaN/Infinity disabled.
    normalised = json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if not isinstance(normalised, dict):  # defensive; checked above
        raise ValueError("event payload must remain a JSON object")
    return normalised


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event_hash_payload(
    *,
    run_id: str,
    sequence: int,
    phase: str,
    cycle_index: int,
    event_type: str,
    created_at_utc: str,
    payload: Mapping[str, Any],
    previous_event_sha256: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "sequence": sequence,
        "phase": phase,
        "cycle_index": cycle_index,
        "event_type": event_type,
        "created_at_utc": created_at_utc,
        "payload": dict(payload),
        "previous_event_sha256": previous_event_sha256,
    }


def _compute_event_sha256(**values: Any) -> str:
    return hashlib.sha256(_canonical_json(_event_hash_payload(**values))).hexdigest()


@dataclass(frozen=True, slots=True)
class StopScanRunEvent:
    """One immutable event whose digest commits to its predecessor."""

    run_id: str
    sequence: int
    phase: str
    cycle_index: int
    event_type: str
    created_at_utc: str
    payload: dict[str, Any]
    previous_event_sha256: str | None
    event_sha256: str

    def __post_init__(self) -> None:
        run_id = _validate_run_id(self.run_id)
        sequence = _non_negative_integer(self.sequence, label="sequence")
        phase = _validate_token(self.phase, label="phase")
        cycle_index = _non_negative_integer(self.cycle_index, label="cycle_index")
        event_type = _validate_token(self.event_type, label="event_type")
        created_at = _canonical_utc_text(self.created_at_utc, require_canonical=True)
        payload = _normalise_payload(self.payload)
        previous = self.previous_event_sha256
        if previous is not None and (
            not isinstance(previous, str)
            or _SHA256_PATTERN.fullmatch(previous) is None
        ):
            raise ValueError("previous_event_sha256 must be null or a SHA-256 digest")
        if sequence == 0 and previous is not None:
            raise ValueError("the first event must not have a predecessor")
        if sequence > 0 and previous is None:
            raise ValueError("every event after sequence zero requires a predecessor")
        if not isinstance(self.event_sha256, str) or _SHA256_PATTERN.fullmatch(
            self.event_sha256
        ) is None:
            raise ValueError("event_sha256 must be a lowercase SHA-256 digest")
        expected_hash = _compute_event_sha256(
            run_id=run_id,
            sequence=sequence,
            phase=phase,
            cycle_index=cycle_index,
            event_type=event_type,
            created_at_utc=created_at,
            payload=payload,
            previous_event_sha256=previous,
        )
        if self.event_sha256 != expected_hash:
            raise ValueError("event_sha256 does not match the canonical event content")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "created_at_utc", created_at)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        sequence: int,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: dict[str, Any],
        previous_event_sha256: str | None,
        created_at_utc: datetime | str | None = None,
    ) -> StopScanRunEvent:
        run = _validate_run_id(run_id)
        sequence_value = _non_negative_integer(sequence, label="sequence")
        phase_value = _validate_token(phase, label="phase")
        cycle_value = _non_negative_integer(cycle_index, label="cycle_index")
        event_value = _validate_token(event_type, label="event_type")
        created = _canonical_utc_text(
            datetime.now(UTC) if created_at_utc is None else created_at_utc,
            require_canonical=isinstance(created_at_utc, str),
        )
        body = _normalise_payload(payload)
        digest = _compute_event_sha256(
            run_id=run,
            sequence=sequence_value,
            phase=phase_value,
            cycle_index=cycle_value,
            event_type=event_value,
            created_at_utc=created,
            payload=body,
            previous_event_sha256=previous_event_sha256,
        )
        return cls(
            run,
            sequence_value,
            phase_value,
            cycle_value,
            event_value,
            created,
            body,
            previous_event_sha256,
            digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **_event_hash_payload(
                run_id=self.run_id,
                sequence=self.sequence,
                phase=self.phase,
                cycle_index=self.cycle_index,
                event_type=self.event_type,
                created_at_utc=self.created_at_utc,
                payload=self.payload,
                previous_event_sha256=self.previous_event_sha256,
            ),
            "event_sha256": self.event_sha256,
        }


@dataclass(frozen=True, slots=True)
class StoredStopScanRun:
    root: Path
    run_id: str
    events: tuple[StopScanRunEvent, ...]
    navigation_index: dict[str, Any] | None = None

    @property
    def latest_event(self) -> StopScanRunEvent:
        if not self.events:
            raise ValueError("stop-and-scan run contains no events")
        return self.events[-1]


def _event_filename(sequence: int) -> str:
    if sequence >= 10**_EVENT_FILENAME_DIGITS:
        raise ValueError("stop-and-scan event sequence exceeds filename capacity")
    return f"{sequence:0{_EVENT_FILENAME_DIGITS}d}.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON file atomically without ever replacing an existing file."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite stop-and-scan event: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite stop-and-scan event: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_navigation_index(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
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
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class StopScanRunWriter:
    """Append immutable events and maintain a non-authoritative navigation index."""

    def __init__(
        self,
        root: Path,
        run_id: str,
        events: tuple[StopScanRunEvent, ...],
    ) -> None:
        self.root = root
        self.run_id = _validate_run_id(run_id)
        self._events = list(events)

    @classmethod
    def create(cls, output_dir: str | Path, *, run_id: str) -> StopScanRunWriter:
        output = Path(output_dir).resolve()
        run = _validate_run_id(run_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        (output / "events").mkdir()
        writer = cls(output, run, ())
        writer._update_navigation_index()
        return writer

    @classmethod
    def resume(cls, output_dir: str | Path) -> StopScanRunWriter:
        stored = read_stop_scan_run(output_dir)
        return cls(stored.root, stored.run_id, stored.events)

    @property
    def events(self) -> tuple[StopScanRunEvent, ...]:
        return tuple(self._events)

    def append_event(
        self,
        *,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: dict[str, Any],
        created_at_utc: datetime | str | None = None,
    ) -> StopScanRunEvent:
        sequence = len(self._events)
        previous = self._events[-1].event_sha256 if self._events else None
        event = StopScanRunEvent.build(
            run_id=self.run_id,
            sequence=sequence,
            phase=phase,
            cycle_index=cycle_index,
            event_type=event_type,
            payload=payload,
            previous_event_sha256=previous,
            created_at_utc=created_at_utc,
        )
        destination = self.root / "events" / _event_filename(sequence)
        _write_new_json(destination, event.to_payload())
        self._events.append(event)
        self._update_navigation_index()
        return event

    def _update_navigation_index(self) -> None:
        latest = self._events[-1] if self._events else None
        updated_at = latest.created_at_utc if latest is not None else datetime.now(UTC).isoformat()
        payload = {
            "schema_version": STOP_SCAN_RUN_INDEX_SCHEMA_VERSION,
            "navigation_only": True,
            "safety_evidence": False,
            "run_id": self.run_id,
            "event_count": len(self._events),
            "latest_event": (
                {
                    "sequence": latest.sequence,
                    "path": f"events/{_event_filename(latest.sequence)}",
                    "sha256": latest.event_sha256,
                }
                if latest is not None
                else None
            ),
            "updated_at_utc": updated_at,
        }
        _replace_navigation_index(self.root / "run.json", payload)


def _read_event(path: Path) -> StopScanRunEvent:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("event root must be a JSON object")
        if set(payload) != _EVENT_KEYS:
            missing = sorted(_EVENT_KEYS - set(payload))
            extra = sorted(set(payload) - _EVENT_KEYS)
            raise ValueError(f"event keys differ: missing={missing}, extra={extra}")
        return StopScanRunEvent(
            run_id=payload["run_id"],
            sequence=payload["sequence"],
            phase=payload["phase"],
            cycle_index=payload["cycle_index"],
            event_type=payload["event_type"],
            created_at_utc=payload["created_at_utc"],
            payload=payload["payload"],
            previous_event_sha256=payload["previous_event_sha256"],
            event_sha256=payload["event_sha256"],
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StopScanRunFormatError(f"Invalid stop-and-scan event {path}: {exc}") from exc


def _read_navigation_index(root: Path) -> dict[str, Any] | None:
    """Best-effort navigation only; this result is never used to verify events."""

    try:
        payload = json.loads((root / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _strict_json_loads(text: str) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def read_stop_scan_run(path: str | Path) -> StoredStopScanRun:
    """Discover and verify the complete event chain without trusting ``run.json``."""

    root = Path(path).resolve()
    events_dir = root / "events"
    if not events_dir.is_dir() or events_dir.is_symlink():
        raise StopScanRunFormatError(f"Stop-and-scan events directory is missing: {events_dir}")
    event_paths: list[tuple[int, Path]] = []
    for candidate in events_dir.iterdir():
        if candidate.name.startswith("."):
            continue
        match = _EVENT_FILENAME_PATTERN.fullmatch(candidate.name)
        if match is None or not candidate.is_file() or candidate.is_symlink():
            raise StopScanRunFormatError(
                f"Unexpected entry in stop-and-scan events directory: {candidate}"
            )
        event_paths.append((int(match.group(1)), candidate))
    event_paths.sort(key=lambda item: item[0])
    if not event_paths:
        raise StopScanRunFormatError("Stop-and-scan run contains no events")

    events: list[StopScanRunEvent] = []
    run_id: str | None = None
    previous_hash: str | None = None
    previous_created: datetime | None = None
    for expected_sequence, (filename_sequence, event_path) in enumerate(event_paths):
        if filename_sequence != expected_sequence or event_path.name != _event_filename(
            expected_sequence
        ):
            raise StopScanRunFormatError(
                "Stop-and-scan event filenames must form a canonical contiguous sequence"
            )
        event = _read_event(event_path)
        if event.sequence != expected_sequence:
            raise StopScanRunFormatError(
                f"Event file {event_path.name} declares sequence {event.sequence}"
            )
        if run_id is None:
            run_id = event.run_id
        elif event.run_id != run_id:
            raise StopScanRunFormatError("Stop-and-scan event run_id changed within the chain")
        if event.previous_event_sha256 != previous_hash:
            raise StopScanRunFormatError(
                f"Stop-and-scan predecessor hash mismatch at sequence {event.sequence}"
            )
        created = datetime.fromisoformat(event.created_at_utc)
        if previous_created is not None and created < previous_created:
            raise StopScanRunFormatError("Stop-and-scan event UTC timestamps moved backwards")
        events.append(event)
        previous_hash = event.event_sha256
        previous_created = created

    assert run_id is not None  # non-empty chain established above
    return StoredStopScanRun(
        root=root,
        run_id=run_id,
        events=tuple(events),
        navigation_index=_read_navigation_index(root),
    )
