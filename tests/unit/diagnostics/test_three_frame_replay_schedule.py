from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import report_three_frame_replay_schedule as report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _attempt(tmp_path: Path) -> Path:
    root = tmp_path / "attempt"
    physical_ids: list[str] = []
    for index in range(3):
        cycle = root / "perception/coarse/cycles" / f"{index:06d}_view"
        attempt_id = f"attempt_{index:032x}"
        _write_json(
            cycle / "committed.json",
            {"accepted_attempt": {"attempt_id": attempt_id}},
        )
        physical_ids.append(f"physical-{index}")
        frames = [
            {"evidence": {"physical_source_id": value}}
            for value in physical_ids
        ]
        _write_json(
            cycle / attempt_id / "occupancy_mapping/metadata.json",
            {"frames": frames},
        )
        _write_json(
            root
            / "coarse_science/generations"
            / f"{index:06d}"
            / "generation.json",
            {
                "views": [{} for _ in range(index + 1)],
                "summary": {"schema5_ready": False},
            },
        )
    _write_json(
        root / "experiment_handoff/events/00000000.json",
        {"event_type": "experiment_initialized"},
    )
    for index in range(1, 4):
        _write_json(
            root / "experiment_handoff/events" / f"{index:08d}.json",
            {"event_type": "coarse_checkpoint"},
        )
    _write_json(
        root / "runs/coarse/events/00000000.json",
        {"event_type": "map_ready"},
    )
    return root


def test_three_frame_schedule_locks_before_and_after_counts() -> None:
    schedule = report.replay_schedule((1, 2, 3))

    assert [item.before_total for item in schedule] == [18, 44, 92]
    assert [item.after_total for item in schedule] == [13, 28, 57]
    assert schedule[2].after == {
        "perception_and_coarse_view": 33,
        "generation_accept": 6,
        "checkpoint": 6,
        "live_ingest": 0,
        "map_ready_transition_evaluation": 12,
    }


def test_attempt_report_requires_exact_committed_123_topology(tmp_path: Path) -> None:
    root = _attempt(tmp_path)

    payload = report.inspect_attempt(root)

    assert payload["source_counts"] == [1, 2, 3]
    assert payload["generation_view_counts"] == [1, 2, 3]
    assert payload["checkpoint_count"] == 3
    assert payload["map_ready_event_count"] == 1
    assert payload["final_schema5_ready"] is False
    assert payload["totals"] == {"before": 154, "after": 98, "eliminated": 56}
    assert payload["physical_source_ids"] == [
        ["physical-0"],
        ["physical-0", "physical-1"],
        ["physical-0", "physical-1", "physical-2"],
    ]


def test_schedule_rejects_non_cumulative_source_windows() -> None:
    with pytest.raises(report.ThreeFrameScheduleError, match="1..N"):
        report.replay_schedule((1, 3, 3))
