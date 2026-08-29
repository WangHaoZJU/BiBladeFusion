from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from biblade_fusion.storage.stop_scan_run import (
    StopScanRunFormatError,
    StopScanRunWriter,
    read_stop_scan_run,
)


def _rehash(payload: dict[str, object]) -> None:
    content = {key: value for key, value in payload.items() if key != "event_sha256"}
    payload["event_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_writer_appends_and_reader_reverifies_full_chain(tmp_path: Path) -> None:
    start = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="blade-run_001")

    first = writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={"motion_authorized": False, "view_id": None},
        created_at_utc=start,
    )
    second = writer.append_event(
        phase="inferring",
        cycle_index=0,
        event_type="stereo_started",
        payload={"source": "foundation_stereo", "latency_ms": 12.5},
        created_at_utc=start + timedelta(seconds=1),
    )

    stored = read_stop_scan_run(writer.root)

    assert stored.run_id == "blade-run_001"
    assert stored.events == (first, second)
    assert first.sequence == 0
    assert first.previous_event_sha256 is None
    assert second.sequence == 1
    assert second.previous_event_sha256 == first.event_sha256
    assert stored.navigation_index is not None
    assert stored.navigation_index["navigation_only"] is True
    assert stored.navigation_index["safety_evidence"] is False
    assert stored.navigation_index["latest_event"]["sha256"] == second.event_sha256


def test_create_refuses_existing_run_directory(tmp_path: Path) -> None:
    output = tmp_path / "run"
    StopScanRunWriter.create(output, run_id="run-001")

    with pytest.raises(FileExistsError):
        StopScanRunWriter.create(output, run_id="run-001")


def test_two_resumed_writers_cannot_overwrite_same_event(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
    )
    first_resume = StopScanRunWriter.resume(writer.root)
    second_resume = StopScanRunWriter.resume(writer.root)
    committed = first_resume.append_event(
        phase="capturing",
        cycle_index=0,
        event_type="capture_started",
        payload={},
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        second_resume.append_event(
            phase="failed",
            cycle_index=0,
            event_type="competing_write",
            payload={},
        )

    stored = read_stop_scan_run(writer.root)
    assert stored.latest_event == committed
    assert len(stored.events) == 2


@pytest.mark.parametrize(
    "payload",
    (
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": (1, 2)},
        {1: "non-string key"},
    ),
)
def test_writer_rejects_non_json_or_nonfinite_payload(
    tmp_path: Path,
    payload: dict[object, object],
) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")

    with pytest.raises(ValueError):
        writer.append_event(
            phase="failed",
            cycle_index=0,
            event_type="invalid_payload",
            payload=payload,  # type: ignore[arg-type]
        )

    assert not tuple((writer.root / "events").glob("*.json"))


def test_writer_rejects_non_utc_or_noncanonical_timestamp(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        writer.append_event(
            phase="failed",
            cycle_index=0,
            event_type="invalid_time",
            payload={},
            created_at_utc=datetime(2026, 8, 28, 8, 0),
        )
    with pytest.raises(ValueError, match="canonical UTC"):
        writer.append_event(
            phase="failed",
            cycle_index=0,
            event_type="invalid_time",
            payload={},
            created_at_utc="2026-08-28T08:00:00Z",
        )


def test_writer_rejects_backwards_utc_before_publishing_event(tmp_path: Path) -> None:
    start = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    first = writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
        created_at_utc=start,
    )

    with pytest.raises(ValueError, match="precedes the previous event"):
        writer.append_event(
            phase="capturing",
            cycle_index=0,
            event_type="capture_started",
            payload={},
            created_at_utc=start - timedelta(seconds=1),
        )

    assert writer.events == (first,)
    assert [item.name for item in (writer.root / "events").iterdir()] == [
        "00000000.json"
    ]
    assert read_stop_scan_run(writer.root).events == (first,)


def test_writer_rejects_non_string_identity_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id must be a string"):
        StopScanRunWriter.create(tmp_path / "run", run_id=7)  # type: ignore[arg-type]


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
    )
    event_path = writer.root / "events" / "00000000.json"
    text = event_path.read_text(encoding="utf-8")
    event_path.write_text(
        text.replace(
            '{\n  "created_at_utc"',
            '{\n  "run_id": "run-001",\n  "created_at_utc"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(StopScanRunFormatError, match="duplicate JSON object key"):
        read_stop_scan_run(writer.root)


def test_reader_detects_payload_tampering(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={"safe": True},
    )
    event_path = writer.root / "events" / "00000000.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["payload"]["safe"] = False
    _write_payload(event_path, payload)

    with pytest.raises(StopScanRunFormatError, match="event_sha256"):
        read_stop_scan_run(writer.root)


def test_reader_detects_forged_but_broken_predecessor_chain(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
    )
    writer.append_event(
        phase="capturing",
        cycle_index=0,
        event_type="capture_started",
        payload={},
    )
    event_path = writer.root / "events" / "00000001.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["previous_event_sha256"] = "f" * 64
    _rehash(payload)
    _write_payload(event_path, payload)

    with pytest.raises(StopScanRunFormatError, match="predecessor hash mismatch"):
        read_stop_scan_run(writer.root)


def test_reader_detects_sequence_gap_and_noncanonical_filename(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
    )
    source = writer.root / "events" / "00000000.json"
    source.rename(writer.root / "events" / "00000001.json")

    with pytest.raises(StopScanRunFormatError, match="contiguous sequence"):
        read_stop_scan_run(writer.root)


def test_reader_ignores_navigation_index_claims(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    event = writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={"evidence": "event-chain"},
    )
    (writer.root / "run.json").write_text(
        '{"event_count": 999, "latest_event": "untrusted"}\n',
        encoding="utf-8",
    )

    stored = read_stop_scan_run(writer.root)

    assert stored.events == (event,)
    assert stored.navigation_index == {
        "event_count": 999,
        "latest_event": "untrusted",
    }


def test_reader_rejects_backwards_utc_even_when_event_is_rehashed(tmp_path: Path) -> None:
    start = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="stopped",
        cycle_index=0,
        event_type="run_started",
        payload={},
        created_at_utc=start,
    )
    writer.append_event(
        phase="capturing",
        cycle_index=0,
        event_type="capture_started",
        payload={},
        created_at_utc=start + timedelta(seconds=1),
    )
    event_path = writer.root / "events" / "00000001.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["created_at_utc"] = (start - timedelta(seconds=1)).isoformat()
    _rehash(payload)
    _write_payload(event_path, payload)

    with pytest.raises(StopScanRunFormatError, match="timestamps moved backwards"):
        read_stop_scan_run(writer.root)
