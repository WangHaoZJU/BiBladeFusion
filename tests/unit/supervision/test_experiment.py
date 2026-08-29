from __future__ import annotations

from pathlib import Path

from biblade_fusion.storage.stop_scan_run import StopScanRunWriter
from biblade_fusion.supervision.experiment import read_experiment_events


def test_event_cursor_returns_only_verified_new_events(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    first = writer.append_event(
        phase="idle",
        cycle_index=0,
        event_type="created",
        payload={},
    )
    second = writer.append_event(
        phase="capturing",
        cycle_index=0,
        event_type="capture_started",
        payload={},
    )

    initial = read_experiment_events(writer.root)
    update = read_experiment_events(
        writer.root,
        from_sequence=first.sequence + 1,
    )

    assert initial.events == (first, second)
    assert initial.next_sequence == 2
    assert update.events == (second,)
    assert update.next_sequence == 2

    writer.append_event(
        phase="publishing_map",
        cycle_index=0,
        event_type="map_published",
        payload={},
    )
    tail = read_experiment_events(writer.root, from_sequence=update.next_sequence)
    assert tuple(event.sequence for event in tail.events) == (2,)
    assert tail.next_sequence == 3
