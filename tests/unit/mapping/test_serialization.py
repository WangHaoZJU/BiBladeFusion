import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from biblade_fusion.mapping.occupancy import (
    OccupancyMapState,
    OccupancySnapshot,
)
from biblade_fusion.mapping.serialization import (
    OccupancySnapshotFormatError,
    load_occupancy_snapshot,
    save_occupancy_snapshot,
)

NOW = datetime(2026, 8, 28, 4, 0, tzinfo=UTC)
CONTEXT_HASH = "d" * 64
PARENT_HASH = "b" * 64


def snapshot() -> OccupancySnapshot:
    return OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.02,
        origin_m=(-0.5, -0.5, 0.0),
        grid_shape=(50, 50, 50),
        free_indices=frozenset({(1, 2, 3), (2, 2, 3)}),
        free_observation_counts=(((1, 2, 3), 3), ((2, 2, 3), 3)),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset({(3, 2, 3)}),
        sequence=4,
        created_at_utc=NOW,
        source_view_ids=("front-001", "front-002", "front-003"),
        source_camera_centres_base_m=(
            (0.0, 0.0, 0.0),
            (0.03, 0.0, 0.0),
            (0.06, 0.0, 0.0),
        ),
        source_camera_axes_base=((0.0, 0.0, 1.0),) * 3,
        rebuild_started_at_utc=NOW,
        map_state=OccupancyMapState.MAPPING,
        mapping_context_hash=CONTEXT_HASH,
        parent_evidence_hash=PARENT_HASH,
        state_reason="awaiting quality validation",
    ).promote_to_ready("c" * 64)


def test_snapshot_round_trip_creates_asset_directory(tmp_path: Path) -> None:
    output = tmp_path / "maps" / "versions" / "map-0005.json"
    original = snapshot()

    returned = save_occupancy_snapshot(output, original)
    loaded = load_occupancy_snapshot(output)

    assert returned == output
    assert loaded == original
    assert loaded.content_hash == original.content_hash
    assert json.loads(output.read_text())["units"] == "m"
    assert loaded.free_observation_counts == original.free_observation_counts
    assert loaded.minimum_free_observations == 3


def test_same_snapshot_save_is_idempotent_but_different_content_is_not_overwritten(
    tmp_path: Path,
) -> None:
    output = tmp_path / "map.json"
    original = snapshot()
    save_occupancy_snapshot(output, original)
    save_occupancy_snapshot(output, original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_occupancy_snapshot(output, original.mark_stale("fixture moved"))
    assert load_occupancy_snapshot(output) == original


def test_hash_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "map.json"
    save_occupancy_snapshot(output, snapshot())
    payload = json.loads(output.read_text())
    payload["snapshot"]["state_reason"] = "tampered"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OccupancySnapshotFormatError, match="content_hash mismatch"):
        load_occupancy_snapshot(output)


def test_wrong_units_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "map.json"
    save_occupancy_snapshot(output, snapshot())
    payload = json.loads(output.read_text())
    payload["units"] = "mm"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OccupancySnapshotFormatError, match="units must be 'm'"):
        load_occupancy_snapshot(output)


def test_legacy_format_and_missing_free_vote_fields_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "map.json"
    save_occupancy_snapshot(output, snapshot())
    payload = json.loads(output.read_text())
    payload["format_version"] = 3
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OccupancySnapshotFormatError, match="format_version"):
        load_occupancy_snapshot(output)

    payload["format_version"] = 4
    payload["snapshot"].pop("free_observation_counts")
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OccupancySnapshotFormatError, match="fields mismatch"):
        load_occupancy_snapshot(output)


def test_free_vote_count_tampering_is_hash_detected(tmp_path: Path) -> None:
    output = tmp_path / "map.json"
    save_occupancy_snapshot(output, snapshot())
    payload = json.loads(output.read_text())
    payload["snapshot"]["free_observation_counts"][0][3] = 2
    payload["snapshot"]["free_indices"] = [payload["snapshot"]["free_indices"][1]]
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OccupancySnapshotFormatError, match="content_hash mismatch"):
        load_occupancy_snapshot(output)


def test_camera_pose_evidence_tampering_is_hash_detected(tmp_path: Path) -> None:
    output = tmp_path / "map.json"
    save_occupancy_snapshot(output, snapshot())
    payload = json.loads(output.read_text())
    payload["snapshot"]["source_camera_centres_base_m"][1][0] += 0.001
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OccupancySnapshotFormatError, match="content_hash mismatch"):
        load_occupancy_snapshot(output)
