"""Atomic, immutable JSON serialization for safety occupancy snapshots."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

from biblade_fusion.mapping.occupancy import (
    OccupancyMapState,
    OccupancySnapshot,
    snapshot_hash_payload,
)

FORMAT_VERSION = 4
ARTIFACT_KIND = "biblade_fusion.safety_occupancy"


class OccupancySnapshotFormatError(ValueError):
    """Serialized occupancy data is malformed, unsupported, or corrupted."""


def save_occupancy_snapshot(path: str | Path, snapshot: OccupancySnapshot) -> Path:
    """Atomically create an immutable snapshot artifact.

    Re-saving the exact same content is idempotent.  Existing different content
    is never overwritten, preserving the versioned map as a digital asset.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = load_occupancy_snapshot(output)
        if existing.content_hash == snapshot.content_hash:
            return output
        raise FileExistsError(f"Refusing to overwrite occupancy snapshot: {output}")

    payload = {
        "format_version": FORMAT_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "units": "m",
        "snapshot": {
            **snapshot_hash_payload(snapshot),
            "content_hash": snapshot.content_hash,
        },
    }
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
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
            os.link(temporary, output)
        except FileExistsError:
            existing = load_occupancy_snapshot(output)
            if existing.content_hash != snapshot.content_hash:
                raise FileExistsError(
                    f"Refusing to overwrite occupancy snapshot: {output}"
                ) from None
        return output
    finally:
        temporary.unlink(missing_ok=True)


def load_occupancy_snapshot(path: str | Path) -> OccupancySnapshot:
    """Load and hash-verify one occupancy artifact."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            _fail("Occupancy artifact root must be an object")
        _exact_keys(
            payload,
            {"format_version", "artifact_kind", "units", "snapshot"},
            label="artifact root",
        )
        if payload.get("format_version") != FORMAT_VERSION:
            _fail(f"Unsupported occupancy format_version: {payload.get('format_version')!r}")
        if payload.get("artifact_kind") != ARTIFACT_KIND:
            _fail(f"Unexpected occupancy artifact_kind: {payload.get('artifact_kind')!r}")
        if payload.get("units") != "m":
            _fail("Occupancy artifact units must be 'm'")
        raw = payload.get("snapshot")
        if not isinstance(raw, Mapping):
            _fail("Occupancy artifact snapshot must be an object")
        _exact_keys(
            raw,
            {
                "frame_id",
                "voxel_size_m",
                "origin_m",
                "grid_shape",
                "free_indices",
                "free_observation_counts",
                "minimum_free_observations",
                "minimum_free_view_translation_m",
                "minimum_free_view_direction_deg",
                "occupied_indices",
                "sequence",
                "created_at_utc",
                "source_view_ids",
                "source_camera_centres_base_m",
                "source_camera_axes_base",
                "rebuild_started_at_utc",
                "map_state",
                "mapping_context_hash",
                "parent_evidence_hash",
                "quality_evidence_hash",
                "state_reason",
                "content_hash",
            },
            label="snapshot",
        )
        return OccupancySnapshot(
            frame_id=_string(raw, "frame_id"),
            voxel_size_m=_number(raw, "voxel_size_m"),
            origin_m=_float_triplet(raw, "origin_m"),
            grid_shape=_int_triplet(raw, "grid_shape"),
            free_indices=_indices(raw, "free_indices"),
            free_observation_counts=_free_observation_counts(
                raw,
                "free_observation_counts",
            ),
            minimum_free_observations=_integer(
                raw,
                "minimum_free_observations",
            ),
            minimum_free_view_translation_m=_number(
                raw,
                "minimum_free_view_translation_m",
            ),
            minimum_free_view_direction_deg=_number(
                raw,
                "minimum_free_view_direction_deg",
            ),
            occupied_indices=_indices(raw, "occupied_indices"),
            sequence=_integer(raw, "sequence"),
            created_at_utc=datetime.fromisoformat(_string(raw, "created_at_utc")),
            source_view_ids=tuple(_string_list(raw, "source_view_ids")),
            source_camera_centres_base_m=tuple(
                _float_triplet_list(raw, "source_camera_centres_base_m")
            ),
            source_camera_axes_base=tuple(
                _float_triplet_list(raw, "source_camera_axes_base")
            ),
            rebuild_started_at_utc=_optional_datetime(
                raw,
                "rebuild_started_at_utc",
            ),
            map_state=OccupancyMapState(_string(raw, "map_state")),
            mapping_context_hash=_optional_string(raw, "mapping_context_hash"),
            parent_evidence_hash=_optional_string(raw, "parent_evidence_hash"),
            quality_evidence_hash=_optional_string(raw, "quality_evidence_hash"),
            state_reason=_string(raw, "state_reason"),
            content_hash=_string(raw, "content_hash"),
        )
    except OccupancySnapshotFormatError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OccupancySnapshotFormatError(f"Invalid occupancy artifact {source}: {exc}") from exc


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        _fail(f"Missing occupancy snapshot field: {key}")
    return mapping[key]


def _exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(f"Occupancy {label} fields mismatch: missing={missing}, extra={extra}")


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or not value:
        _fail(f"Occupancy snapshot {key} must be a non-empty string")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = _required(mapping, key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail(f"Occupancy snapshot {key} must be null or a non-empty string")
    return value


def _optional_datetime(
    mapping: Mapping[str, Any],
    key: str,
) -> datetime | None:
    value = _required(mapping, key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail(f"Occupancy snapshot {key} must be null or a datetime string")
    return datetime.fromisoformat(value)


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = _required(mapping, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"Occupancy snapshot {key} must be numeric")
    return float(value)


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = _required(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"Occupancy snapshot {key} must be an integer")
    return value


def _sequence(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = _required(mapping, key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(f"Occupancy snapshot {key} must be an array")
    return value


def _float_triplet(mapping: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    values = _sequence(mapping, key)
    if len(values) != 3:
        _fail(f"Occupancy snapshot {key} must contain three values")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(f"Occupancy snapshot {key} values must be numeric")
        converted.append(float(value))
    return (converted[0], converted[1], converted[2])


def _float_triplet_list(
    mapping: Mapping[str, Any],
    key: str,
) -> list[tuple[float, float, float]]:
    values = _sequence(mapping, key)
    result: list[tuple[float, float, float]] = []
    for value in values:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            _fail(f"Occupancy snapshot {key} entries must be numeric triplets")
        holder = {key: value}
        result.append(_float_triplet(holder, key))
    return result


def _int_triplet(mapping: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    values = _sequence(mapping, key)
    invalid_values = any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    )
    if len(values) != 3 or invalid_values:
        _fail(f"Occupancy snapshot {key} must contain three integers")
    return (int(values[0]), int(values[1]), int(values[2]))


def _indices(mapping: Mapping[str, Any], key: str) -> frozenset[tuple[int, int, int]]:
    values = _sequence(mapping, key)
    result: set[tuple[int, int, int]] = set()
    for value in values:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            _fail(f"Occupancy snapshot {key} entries must be integer triplets")
        result.add((int(value[0]), int(value[1]), int(value[2])))
    if len(result) != len(values):
        _fail(f"Occupancy snapshot {key} must not contain duplicate entries")
    return frozenset(result)


def _free_observation_counts(
    mapping: Mapping[str, Any],
    key: str,
) -> tuple[tuple[tuple[int, int, int], int], ...]:
    values = _sequence(mapping, key)
    result: dict[tuple[int, int, int], int] = {}
    for value in values:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in value
            )
        ):
            _fail(
                f"Occupancy snapshot {key} entries must be "
                "three integer indices plus one integer count"
            )
        index = (int(value[0]), int(value[1]), int(value[2]))
        count = int(value[3])
        if index in result:
            _fail(f"Occupancy snapshot {key} must not contain duplicate voxels")
        result[index] = count
    return tuple(sorted(result.items()))


def _string_list(mapping: Mapping[str, Any], key: str) -> list[str]:
    values = _sequence(mapping, key)
    if any(not isinstance(value, str) or not value for value in values):
        _fail(f"Occupancy snapshot {key} entries must be non-empty strings")
    return list(values)


def _fail(message: str) -> NoReturn:
    raise OccupancySnapshotFormatError(message)
