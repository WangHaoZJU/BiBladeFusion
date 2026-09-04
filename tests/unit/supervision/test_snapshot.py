from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.supervision import (
    CandidatePlanningSnapshot,
    LivePlanningUpdate,
    PlanningProgressSnapshot,
    discover_supervisory_snapshots,
    load_snapshot_array,
    read_live_planning_update,
    read_supervisory_snapshot,
    write_live_planning_update,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_snapshot(root: Path, *, sequence: int = 0) -> Path:
    root.mkdir(parents=True)
    (root / "assets").mkdir()
    asset_path = root / "assets" / "source.json"
    asset_path.write_text('{"source":"unit-test"}\n', encoding="utf-8")
    links = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.4), (0.2, 0.0, 0.7)),
        dtype=np.float32,
    )
    occupied = np.asarray(((0.4, 0.1, 0.4), (0.4, -0.1, 0.4)), dtype=np.float32)
    fused = np.asarray(((0.39, 0.0, 0.35), (0.40, 0.0, 0.45)), dtype=np.float32)
    for name, array in (("links.npy", links), ("occupied.npy", occupied), ("fused.npy", fused)):
        np.save(root / name, array, allow_pickle=False)

    def reference(name: str, array: np.ndarray, semantic: str) -> dict:
        path = root / name
        return {
            "path": name,
            "sha256": _sha256(path),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "semantic": semantic,
        }

    payload = {
        "schema_version": 2,
        "snapshot_id": f"snapshot-{sequence}",
        "sequence": sequence,
        "created_at_utc": f"2026-08-28T00:00:{sequence:02d}+00:00",
        "safety": {
            "system_state": "BLOCKED",
            "viewer_mode": "REPLAY",
            "viewer_motion_command_capable": False,
            "blocking_reasons": ["replay_only"],
        },
        "robot": {
            "model_id": "es68-d435i-test",
            "base_frame": "base",
            "joint_names": [f"joint_{index}" for index in range(1, 7)],
            "joint_positions_rad": [0.0] * 6,
            "link_origins_base_m": reference("links.npy", links, "robot_link_origins"),
            "camera_pose": {
                "parent_frame": "base",
                "child_frame": "left_ir",
                "matrix": [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.7],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        },
        "occupancy": {
            "frame_id": "base",
            "state": "READY",
            "version": "map-1",
            "content_sha256": "0" * 64,
            "voxel_size_m": 0.02,
            "bounds_min_m": [-1.0, -1.0, 0.0],
            "bounds_max_m": [1.0, 1.0, 2.0],
            "age_s": 0.1,
            "integrated_frame_count": 1,
            "occupied_centres_m": reference(
                "occupied.npy", occupied, "occupied_voxel_centres_base_m"
            ),
        },
        "reconstruction": {
            "frame_id": "base",
            "model_version": "reconstruction-1",
            "fused_points_m": reference("fused.npy", fused, "fused_blade_points_base_m"),
            "front_coverage": 0.4,
            "back_coverage": 0.2,
        },
        "assets": [
            {
                "logical_name": "source_manifest",
                "kind": "unit_test.source",
                "path": "assets/source.json",
                "sha256": _sha256(asset_path),
                "version": "1",
            }
        ],
    }
    path = root / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_snapshot_verifies_and_loads_non_pickle_array(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")

    stored = read_supervisory_snapshot(path)
    links = load_snapshot_array(stored, stored.snapshot.robot.link_origins_base_m)

    assert stored.snapshot.safety.viewer_motion_command_capable is False
    assert links is not None
    assert links.shape == (3, 3)
    assert links.flags.writeable is False
    assert stored.snapshot.plan.planning.phase == "unknown"
    assert stored.snapshot.plan.planning.candidates == ()


def test_live_planning_sidecar_round_trips_candidate_gate_evidence(tmp_path: Path) -> None:
    update = LivePlanningUpdate(
        generated_at_utc="2026-09-04T02:00:00+00:00",
        source_session_id="planning-test",
        latest_event_sha256="a" * 64,
        planning=PlanningProgressSnapshot(
            phase="preflighting",
            disposition="preflighting",
            cycle_index=1,
            phase_started_at_utc="2026-09-04T01:59:59+00:00",
            phase_elapsed_s=1.0,
            candidate_count=1,
            active_candidate_id="view-01",
            candidates=(
                CandidatePlanningSnapshot(
                    candidate_id="view-01",
                    rank=1,
                    science_rank=1,
                    science_score=0.75,
                    active=True,
                    ik_status="CLEAR",
                    endpoint_status="RUNNING",
                    straight_path_status="RUNNING",
                    rrt_status="PENDING",
                ),
            ),
        ),
    )

    path = write_live_planning_update(tmp_path / "live_planning.json", update)
    restored = read_live_planning_update(path)

    assert restored == update
    assert not (tmp_path / ".live_planning.json.tmp").exists()


def test_planning_progress_rejects_inconsistent_selected_candidate() -> None:
    with pytest.raises(ValueError, match="selected_candidate_id"):
        PlanningProgressSnapshot(
            candidate_count=1,
            selected_candidate_id="view-02",
            candidates=(
                CandidatePlanningSnapshot(
                    candidate_id="view-01",
                    rank=1,
                    science_rank=1,
                ),
            ),
        )


def test_snapshot_rejects_tampered_array(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    np.save(path.parent / "links.npy", np.ones((3, 3), dtype=np.float32), allow_pickle=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_supervisory_snapshot(path)


def test_snapshot_rejects_array_path_escape(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["robot"]["link_origins_base_m"]["path"] = "../outside.npy"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes snapshot root"):
        read_supervisory_snapshot(path)


def test_snapshot_rejects_tampered_asset(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    (path.parent / "assets" / "source.json").write_text(
        '{"source":"tampered"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="asset checksum mismatch"):
        read_supervisory_snapshot(path)


def test_snapshot_rejects_asset_path_escape(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assets"][0]["path"] = str(path.parent / "assets" / "source.json")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="asset escapes snapshot root"):
        read_supervisory_snapshot(path)


def test_snapshot_contract_forbids_motion_command_capability(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["safety"]["viewer_motion_command_capable"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="viewer_motion_command_capable"):
        read_supervisory_snapshot(path)


def test_snapshot_schema_one_is_rejected_after_contract_upgrade(tmp_path: Path) -> None:
    path = _write_snapshot(tmp_path / "snapshot")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_supervisory_snapshot(path)


def test_timeline_discovers_direct_child_snapshots_in_sequence_order(tmp_path: Path) -> None:
    _write_snapshot(tmp_path / "later", sequence=2)
    _write_snapshot(tmp_path / "earlier", sequence=1)

    timeline = discover_supervisory_snapshots(tmp_path)

    assert [item.snapshot.sequence for item in timeline.snapshots] == [1, 2]
