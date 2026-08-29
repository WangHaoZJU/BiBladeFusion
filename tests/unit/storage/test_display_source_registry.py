from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import biblade_fusion.storage.display_source_registry as registry_module
from biblade_fusion.storage.display_source_registry import (
    AppendOnlyDisplaySourceRegistry,
    DisplaySourceEntry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _points_sha256(points: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(points, dtype="<f8")).tobytes()
    ).hexdigest()


def _source(root: Path, name: str, points: np.ndarray) -> tuple[Path, Path]:
    source = root / name
    source.mkdir()
    metadata = source / "metadata.json"
    point_array = source / "base_points_m.npy"
    np.save(point_array, points, allow_pickle=False)
    metadata.write_text(
        json.dumps(
            {
                "source": name,
                "base_points_m_sha256": _sha256(point_array),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata, point_array


def _append(
    registry: AppendOnlyDisplaySourceRegistry,
    *,
    metadata: Path,
    point_array: Path,
    view_id: str,
    sequence_index: int,
    frame_number: int,
) -> DisplaySourceEntry:
    points = np.load(point_array, allow_pickle=False)
    return registry.append(
        source_kind="coarse_scan_view",
        view_id=view_id,
        source_sequence_index=sequence_index,
        source_frame_number=frame_number,
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        point_array_path=point_array,
        point_array_file_sha256=_sha256(point_array),
        points_f64le_sha256=_points_sha256(points),
        raw_point_count=len(points),
        voxel_point_count=len(points),
        display_algorithm="deterministic_bounded_display_voxel_union_v1",
        display_voxel_size_m=0.002,
        maximum_current_points=50_000,
        created_at_utc=datetime(2026, 8, 29, 4, 0, tzinfo=UTC),
    )


def test_registry_is_append_only_hash_chained_and_restart_verified(tmp_path: Path) -> None:
    points_a = np.asarray(((0.1, 0.0, 0.2), (0.2, 0.0, 0.3)), dtype=np.float64)
    points_b = np.asarray(((0.3, 0.0, 0.4),), dtype=np.float64)
    metadata_a, array_a = _source(tmp_path, "source-a", points_a)
    metadata_b, array_b = _source(tmp_path, "source-b", points_b)
    root = tmp_path / "registry"
    registry = AppendOnlyDisplaySourceRegistry(root)

    first = _append(
        registry,
        metadata=metadata_a,
        point_array=array_a,
        view_id="shared-name",
        sequence_index=0,
        frame_number=10,
    )
    second = _append(
        registry,
        metadata=metadata_b,
        point_array=array_b,
        view_id="shared-name",
        sequence_index=1,
        frame_number=11,
    )

    assert first.sequence == 0
    assert first.previous_entry_sha256 is None
    assert second.sequence == 1
    assert second.previous_entry_sha256 == first.entry_sha256
    assert first.physical_source_id != second.physical_source_id
    assert registry.head.entry_count == 2
    assert registry.head.head_entry_sha256 == second.entry_sha256

    restarted = AppendOnlyDisplaySourceRegistry(root)
    assert restarted.entries == registry.entries
    assert np.array_equal(restarted.load_points(restarted.entries[0]), points_a)
    assert restarted.verify() == restarted.entries


def test_duplicate_physical_source_is_idempotent_not_reappended(tmp_path: Path) -> None:
    points = np.asarray(((0.1, 0.0, 0.2),), dtype=np.float64)
    metadata, point_array = _source(tmp_path, "source", points)
    registry = AppendOnlyDisplaySourceRegistry(tmp_path / "registry")

    first = _append(
        registry,
        metadata=metadata,
        point_array=point_array,
        view_id="view-00",
        sequence_index=0,
        frame_number=3,
    )
    repeated = _append(
        registry,
        metadata=metadata,
        point_array=point_array,
        view_id="view-00",
        sequence_index=0,
        frame_number=3,
    )

    assert repeated == first
    assert registry.head.entry_count == 1


@pytest.mark.parametrize("tamper", ["entry", "metadata", "points"])
def test_restart_rejects_chain_or_physical_source_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    points = np.asarray(((0.1, 0.0, 0.2),), dtype=np.float64)
    metadata, point_array = _source(tmp_path, "source", points)
    root = tmp_path / "registry"
    registry = AppendOnlyDisplaySourceRegistry(root)
    entry = _append(
        registry,
        metadata=metadata,
        point_array=point_array,
        view_id="view-00",
        sequence_index=0,
        frame_number=3,
    )
    if tamper == "entry":
        payload = json.loads(entry.path.read_text(encoding="utf-8"))
        payload["physical_source"]["frame_number"] = 4
        entry.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif tamper == "metadata":
        metadata.write_text('{"changed":true}\n', encoding="utf-8")
    else:
        np.save(point_array, np.asarray(((9.0, 9.0, 9.0),)), allow_pickle=False)

    with pytest.raises(ValueError):
        AppendOnlyDisplaySourceRegistry(root)


def test_unexpected_registry_file_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    root.mkdir()
    (root / "latest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        AppendOnlyDisplaySourceRegistry(root)


def test_atomic_publish_never_replaces_a_concurrent_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray(((0.1, 0.0, 0.2),), dtype=np.float64)
    metadata, point_array = _source(tmp_path, "source", points)
    registry = AppendOnlyDisplaySourceRegistry(tmp_path / "registry")
    concurrent_bytes = b"concurrent-writer-owned-this-path\n"
    attempted_destination: Path | None = None

    def collide(_source_path: Path, destination_path: Path) -> None:
        nonlocal attempted_destination
        attempted_destination = Path(destination_path)
        attempted_destination.write_bytes(concurrent_bytes)
        raise FileExistsError(attempted_destination)

    monkeypatch.setattr(registry_module.os, "link", collide)

    with pytest.raises(FileExistsError):
        _append(
            registry,
            metadata=metadata,
            point_array=point_array,
            view_id="view-00",
            sequence_index=0,
            frame_number=3,
        )

    assert attempted_destination is not None
    assert attempted_destination.read_bytes() == concurrent_bytes
    assert registry.entries == ()
