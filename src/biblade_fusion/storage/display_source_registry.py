"""Append-only provenance for bounded live-display point-cloud sources.

The registry is deliberately separate from scientific reconstruction assets.  It
records which already-verified physical views contributed to the bounded display
union, but it cannot grant motion authority or replace the source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

DISPLAY_SOURCE_REGISTRY_SCHEMA_VERSION = 1
DISPLAY_SOURCE_REGISTRY_KIND = "biblade_fusion.display_source_registry_entry"
_ENTRY_NAME = re.compile(r"^(?P<sequence>[0-9]{8})_(?P<digest>[0-9a-f]{64})\.json$")


def _canonical_json(payload: dict[str, Any]) -> bytes:
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


def _strict_json(path: Path) -> dict[str, Any]:
    def object_from_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("display-source registry entry must be an object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolved_file(
    value: object,
    *,
    expected_sha256: object,
    label: str,
) -> tuple[Path, str]:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} file is missing: {resolved}")
    digest = _digest(expected_sha256, label=f"{label} SHA-256")
    if _sha256_path(resolved) != digest:
        raise ValueError(f"{label} file SHA-256 changed: {resolved}")
    return resolved, digest


def _points_content_sha256(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    return _sha256_bytes(canonical.tobytes())


@dataclass(frozen=True, slots=True)
class DisplaySourceEntry:
    """One verified physical source in the append-only display chain."""

    path: Path
    sequence: int
    entry_sha256: str
    previous_entry_sha256: str | None
    physical_source_id: str
    source_kind: str
    view_id: str
    source_sequence_index: int
    source_frame_number: int
    metadata_path: Path
    metadata_sha256: str
    point_array_path: Path
    point_array_file_sha256: str
    points_f64le_sha256: str
    raw_point_count: int
    voxel_point_count: int
    display_algorithm: str
    display_voxel_size_m: float
    maximum_current_points: int
    created_at_utc: datetime


@dataclass(frozen=True, slots=True)
class DisplaySourceRegistryHead:
    root: Path
    entry_count: int
    head_entry_sha256: str | None
    head_entry_path: Path | None
    head_entry_file_sha256: str | None


class AppendOnlyDisplaySourceRegistry:
    """Crash-safe hash chain with source-byte verification on every recovery."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries = self._read_all_verified()
        self._by_physical_id = {item.physical_source_id: item for item in self._entries}
        if len(self._by_physical_id) != len(self._entries):
            raise ValueError("display-source registry contains a duplicate physical source")

    @property
    def entries(self) -> tuple[DisplaySourceEntry, ...]:
        with self._lock:
            return self._entries

    @property
    def head(self) -> DisplaySourceRegistryHead:
        with self._lock:
            latest = self._entries[-1] if self._entries else None
            return DisplaySourceRegistryHead(
                root=self.root,
                entry_count=len(self._entries),
                head_entry_sha256=(latest.entry_sha256 if latest is not None else None),
                head_entry_path=(latest.path if latest is not None else None),
                head_entry_file_sha256=(
                    _sha256_path(latest.path) if latest is not None else None
                ),
            )

    def verify(self) -> tuple[DisplaySourceEntry, ...]:
        """Re-read the full append-only chain and every physical source file."""

        with self._lock:
            verified = self._read_all_verified()
            if verified != self._entries:
                raise ValueError("display-source registry changed after recovery")
            return verified

    def append(
        self,
        *,
        source_kind: str,
        view_id: str,
        source_sequence_index: int,
        source_frame_number: int,
        metadata_path: str | Path,
        metadata_sha256: str,
        point_array_path: str | Path,
        point_array_file_sha256: str,
        points_f64le_sha256: str,
        raw_point_count: int,
        voxel_point_count: int,
        display_algorithm: str,
        display_voxel_size_m: float,
        maximum_current_points: int,
        created_at_utc: datetime,
    ) -> DisplaySourceEntry:
        with self._lock:
            payload = self._validated_payload(
                sequence=len(self._entries),
                previous_entry_sha256=(
                    self._entries[-1].entry_sha256 if self._entries else None
                ),
                source_kind=source_kind,
                view_id=view_id,
                source_sequence_index=source_sequence_index,
                source_frame_number=source_frame_number,
                metadata_path=metadata_path,
                metadata_sha256=metadata_sha256,
                point_array_path=point_array_path,
                point_array_file_sha256=point_array_file_sha256,
                points_f64le_sha256=points_f64le_sha256,
                raw_point_count=raw_point_count,
                voxel_point_count=voxel_point_count,
                display_algorithm=display_algorithm,
                display_voxel_size_m=display_voxel_size_m,
                maximum_current_points=maximum_current_points,
                created_at_utc=created_at_utc,
            )
            physical_id = str(payload["physical_source_id"])
            existing = self._by_physical_id.get(physical_id)
            if existing is not None:
                display = payload["display"]
                if (
                    existing.display_algorithm != display["algorithm"]
                    or existing.display_voxel_size_m != display["voxel_size_m"]
                    or existing.maximum_current_points != display["maximum_current_points"]
                    or existing.voxel_point_count != display["voxel_point_count"]
                    or existing.raw_point_count != payload["point_array"]["raw_point_count"]
                ):
                    raise ValueError(
                        "duplicate physical display source changed its display contract"
                    )
                return existing
            entry_sha256 = _sha256_bytes(_canonical_json(payload))
            payload["entry_sha256"] = entry_sha256
            destination = self.root / f"{len(self._entries):08d}_{entry_sha256}.json"
            if destination.exists():
                raise FileExistsError(f"display-source registry entry exists: {destination}")
            temporary = self.root / f".entry-{uuid4().hex}.partial"
            try:
                with temporary.open("xb") as stream:
                    stream.write(_canonical_json(payload) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                # A hard-link publish gives us atomic no-replace semantics on the
                # same filesystem.  ``Path.replace`` would silently overwrite a
                # concurrently created entry and would violate append-only storage.
                os.link(temporary, destination)
                _fsync_directory(self.root)
                temporary.unlink()
                _fsync_directory(self.root)
                entry = self._read_entry(
                    destination,
                    expected_sequence=len(self._entries),
                    expected_previous=(self._entries[-1].entry_sha256 if self._entries else None),
                )
            finally:
                if temporary.exists():
                    temporary.unlink()
            self._entries = (*self._entries, entry)
            self._by_physical_id[entry.physical_source_id] = entry
            return entry

    def load_points(self, entry: DisplaySourceEntry) -> np.ndarray:
        """Re-read one physical point array and verify every registry binding."""

        with self._lock:
            if self._by_physical_id.get(entry.physical_source_id) != entry:
                raise ValueError("display-source entry does not belong to this registry")
            verified = self._read_entry(
                entry.path,
                expected_sequence=entry.sequence,
                expected_previous=entry.previous_entry_sha256,
            )
            points = np.load(verified.point_array_path, allow_pickle=False)
            array = np.asarray(points, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
                raise ValueError("display-source point array must be finite Nx3")
            if len(array) != verified.raw_point_count:
                raise ValueError("display-source raw point count changed")
            if _points_content_sha256(array) != verified.points_f64le_sha256:
                raise ValueError("display-source normalized point content changed")
            return array

    def _read_all_verified(self) -> tuple[DisplaySourceEntry, ...]:
        files: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            if path.name.startswith(".entry-") and path.name.endswith(".partial"):
                continue
            match = _ENTRY_NAME.fullmatch(path.name)
            if match is None or not path.is_file():
                raise ValueError(f"unexpected display-source registry entry: {path}")
            files.append((int(match.group("sequence")), path))
        files.sort(key=lambda item: item[0])
        entries: list[DisplaySourceEntry] = []
        previous: str | None = None
        for expected_sequence, (name_sequence, path) in enumerate(files):
            if name_sequence != expected_sequence:
                raise ValueError("display-source registry sequence is not contiguous")
            entry = self._read_entry(
                path,
                expected_sequence=expected_sequence,
                expected_previous=previous,
            )
            self.load_points_unbound(entry)
            entries.append(entry)
            previous = entry.entry_sha256
        return tuple(entries)

    def load_points_unbound(self, entry: DisplaySourceEntry) -> np.ndarray:
        points = np.load(entry.point_array_path, allow_pickle=False)
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("display-source point array must be finite Nx3")
        if len(array) != entry.raw_point_count:
            raise ValueError("display-source raw point count changed")
        if _points_content_sha256(array) != entry.points_f64le_sha256:
            raise ValueError("display-source normalized point content changed")
        return array

    def _read_entry(
        self,
        path: Path,
        *,
        expected_sequence: int,
        expected_previous: str | None,
    ) -> DisplaySourceEntry:
        match = _ENTRY_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid display-source entry filename: {path.name}")
        payload = _strict_json(path)
        entry_sha256 = _digest(payload.pop("entry_sha256"), label="entry_sha256")
        if _sha256_bytes(_canonical_json(payload)) != entry_sha256:
            raise ValueError(f"display-source entry content hash mismatch: {path}")
        if match.group("digest") != entry_sha256:
            raise ValueError(f"display-source entry filename hash mismatch: {path}")
        if int(payload.get("sequence", -1)) != expected_sequence:
            raise ValueError("display-source entry sequence changed")
        if payload.get("previous_entry_sha256") != expected_previous:
            raise ValueError("display-source predecessor hash changed")
        validated = self._validated_payload(**self._payload_arguments(payload))
        if validated != payload:
            raise ValueError("display-source entry canonical payload changed")
        created = datetime.fromisoformat(str(payload["created_at_utc"])).astimezone(UTC)
        return DisplaySourceEntry(
            path=path.resolve(),
            sequence=expected_sequence,
            entry_sha256=entry_sha256,
            previous_entry_sha256=expected_previous,
            physical_source_id=str(payload["physical_source_id"]),
            source_kind=str(payload["source_kind"]),
            view_id=str(payload["physical_source"]["view_id"]),
            source_sequence_index=int(payload["physical_source"]["sequence_index"]),
            source_frame_number=int(payload["physical_source"]["frame_number"]),
            metadata_path=Path(str(payload["metadata"]["path"])),
            metadata_sha256=str(payload["metadata"]["sha256"]),
            point_array_path=Path(str(payload["point_array"]["path"])),
            point_array_file_sha256=str(payload["point_array"]["file_sha256"]),
            points_f64le_sha256=str(payload["point_array"]["points_f64le_sha256"]),
            raw_point_count=int(payload["point_array"]["raw_point_count"]),
            voxel_point_count=int(payload["display"]["voxel_point_count"]),
            display_algorithm=str(payload["display"]["algorithm"]),
            display_voxel_size_m=float(payload["display"]["voxel_size_m"]),
            maximum_current_points=int(payload["display"]["maximum_current_points"]),
            created_at_utc=created,
        )

    @staticmethod
    def _payload_arguments(payload: dict[str, Any]) -> dict[str, Any]:
        physical = payload["physical_source"]
        metadata = payload["metadata"]
        point_array = payload["point_array"]
        display = payload["display"]
        return {
            "sequence": int(payload["sequence"]),
            "previous_entry_sha256": payload["previous_entry_sha256"],
            "source_kind": str(payload["source_kind"]),
            "view_id": str(physical["view_id"]),
            "source_sequence_index": int(physical["sequence_index"]),
            "source_frame_number": int(physical["frame_number"]),
            "metadata_path": str(metadata["path"]),
            "metadata_sha256": str(metadata["sha256"]),
            "point_array_path": str(point_array["path"]),
            "point_array_file_sha256": str(point_array["file_sha256"]),
            "points_f64le_sha256": str(point_array["points_f64le_sha256"]),
            "raw_point_count": int(point_array["raw_point_count"]),
            "voxel_point_count": int(display["voxel_point_count"]),
            "display_algorithm": str(display["algorithm"]),
            "display_voxel_size_m": float(display["voxel_size_m"]),
            "maximum_current_points": int(display["maximum_current_points"]),
            "created_at_utc": datetime.fromisoformat(str(payload["created_at_utc"])),
        }

    @staticmethod
    def _validated_payload(
        *,
        sequence: int,
        previous_entry_sha256: str | None,
        source_kind: str,
        view_id: str,
        source_sequence_index: int,
        source_frame_number: int,
        metadata_path: str | Path,
        metadata_sha256: str,
        point_array_path: str | Path,
        point_array_file_sha256: str,
        points_f64le_sha256: str,
        raw_point_count: int,
        voxel_point_count: int,
        display_algorithm: str,
        display_voxel_size_m: float,
        maximum_current_points: int,
        created_at_utc: datetime,
    ) -> dict[str, Any]:
        if isinstance(sequence, bool) or sequence < 0:
            raise ValueError("display-source sequence must be non-negative")
        if sequence == 0 and previous_entry_sha256 is not None:
            raise ValueError("first display-source entry cannot have a predecessor")
        if sequence > 0:
            previous_entry_sha256 = _digest(
                previous_entry_sha256,
                label="previous_entry_sha256",
            )
        kind = str(source_kind).strip()
        identity = str(view_id).strip()
        algorithm = str(display_algorithm).strip()
        if not kind or not identity or not algorithm:
            raise ValueError("display-source identities and algorithm must be non-empty")
        if source_sequence_index < 0 or source_frame_number < 0:
            raise ValueError("display-source physical indices must be non-negative")
        if raw_point_count < 1 or not 0 < voxel_point_count <= raw_point_count:
            raise ValueError("display-source point counts are invalid")
        if maximum_current_points < 1 or voxel_point_count > maximum_current_points:
            raise ValueError("display-source voxel count exceeds its configured cap")
        voxel_size = float(display_voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("display-source voxel size must be finite and positive")
        if created_at_utc.tzinfo is None or created_at_utc.utcoffset() is None:
            raise ValueError("display-source created_at_utc must be timezone-aware")
        metadata, metadata_digest = _resolved_file(
            metadata_path,
            expected_sha256=metadata_sha256,
            label="display-source metadata",
        )
        point_array, point_file_digest = _resolved_file(
            point_array_path,
            expected_sha256=point_array_file_sha256,
            label="display-source point array",
        )
        points_digest = _digest(points_f64le_sha256, label="points_f64le_sha256")
        physical = {
            "view_id": identity,
            "sequence_index": int(source_sequence_index),
            "frame_number": int(source_frame_number),
        }
        physical_source_id = _sha256_bytes(
            _canonical_json(
                {
                    "schema": "biblade_fusion.display_physical_source.v1",
                    "source_kind": kind,
                    "physical_source": physical,
                    "metadata_sha256": metadata_digest,
                    "point_array_file_sha256": point_file_digest,
                    "points_f64le_sha256": points_digest,
                }
            )
        )
        return {
            "schema_version": DISPLAY_SOURCE_REGISTRY_SCHEMA_VERSION,
            "artifact_kind": DISPLAY_SOURCE_REGISTRY_KIND,
            "sequence": int(sequence),
            "previous_entry_sha256": previous_entry_sha256,
            "physical_source_id": physical_source_id,
            "source_kind": kind,
            "physical_source": physical,
            "metadata": {"path": str(metadata), "sha256": metadata_digest},
            "point_array": {
                "path": str(point_array),
                "file_sha256": point_file_digest,
                "points_f64le_sha256": points_digest,
                "raw_point_count": int(raw_point_count),
            },
            "display": {
                "algorithm": algorithm,
                "voxel_size_m": voxel_size,
                "maximum_current_points": int(maximum_current_points),
                "voxel_point_count": int(voxel_point_count),
            },
            "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
            "motion_authorized": False,
            "scientific_fusion": False,
        }
