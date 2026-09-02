#!/usr/bin/env python3
"""Report the 1/2/3-view replay schedule without opening hardware.

This diagnostic reads only immutable JSON authorities.  It does not execute DDA,
FoundationStereo, IK, a camera, or a robot.  The report separates eliminated
duplicate replay from the MAP_READY transition readbacks that deliberately remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ThreeFrameScheduleError(RuntimeError):
    """The experiment topology cannot support an exact three-frame report."""


@dataclass(frozen=True, slots=True)
class FrameReplaySchedule:
    frame_number: int
    source_count: int
    before: dict[str, int]
    after: dict[str, int]

    @property
    def before_total(self) -> int:
        return sum(self.before.values())

    @property
    def after_total(self) -> int:
        return sum(self.after.values())

    def payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "before_total": self.before_total,
            "after_total": self.after_total,
            "eliminated": self.before_total - self.after_total,
        }


def _triangular(value: int) -> int:
    return value * (value + 1) // 2


def _tetrahedral(value: int) -> int:
    return value * (value + 1) * (value + 2) // 6


def replay_schedule(source_counts: tuple[int, ...]) -> tuple[FrameReplaySchedule, ...]:
    """Return logical source-integration counts before and after Phase 1."""

    if source_counts != tuple(range(1, len(source_counts) + 1)):
        raise ThreeFrameScheduleError(
            "Replay schedule requires cumulative source windows 1..N"
        )
    schedules: list[FrameReplaySchedule] = []
    final_frame = len(source_counts)
    for frame_number, source_count in enumerate(source_counts, start=1):
        prior_triangle = _triangular(source_count - 1)
        triangle = _triangular(source_count)
        transition_evaluation = 2 * triangle if frame_number == final_frame else 0
        before = {
            "perception_and_coarse_view": 12 * source_count,
            "generation_accept": (
                2 * source_count + 3 * prior_triangle + triangle
            ),
            "checkpoint": 2 * _tetrahedral(source_count),
            "live_ingest": source_count,
            "map_ready_transition_evaluation": transition_evaluation,
        }
        after = {
            "perception_and_coarse_view": 11 * source_count,
            "generation_accept": triangle,
            "checkpoint": triangle,
            "live_ingest": 0,
            # Two independent full generation reads remain at the exact
            # MAP_READY transition boundary by design.  In the inspected
            # three-view attempt the science gate returns COLLECTING before
            # any schema-5 generation is written.
            "map_ready_transition_evaluation": transition_evaluation,
        }
        schedules.append(
            FrameReplaySchedule(
                frame_number=frame_number,
                source_count=source_count,
                before=before,
                after=after,
            )
        )
    return tuple(schedules)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThreeFrameScheduleError(f"JSON authority is not an object: {path}")
    return payload


def _authority(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def inspect_attempt(root: str | Path) -> dict[str, Any]:
    """Read the exact committed-cycle/generation/checkpoint topology."""

    attempt = Path(root).resolve()
    cycle_markers = sorted((attempt / "perception/coarse/cycles").glob("*/committed.json"))
    generation_files = sorted(
        (attempt / "coarse_science/generations").glob("*/generation.json")
    )
    event_files = sorted((attempt / "experiment_handoff/events").glob("*.json"))
    checkpoint_files = [
        path for path in event_files if _json(path).get("event_type") == "coarse_checkpoint"
    ]
    if len(cycle_markers) != 3 or len(generation_files) != 3 or len(checkpoint_files) != 3:
        raise ThreeFrameScheduleError(
            "Expected exactly three committed cycles, generations, and checkpoints"
        )
    source_counts: list[int] = []
    physical_sources: list[list[str]] = []
    cycle_authorities: list[dict[str, Any]] = []
    for marker in cycle_markers:
        committed = _json(marker)
        accepted = committed.get("accepted_attempt")
        if not isinstance(accepted, dict):
            raise ThreeFrameScheduleError(f"Committed cycle lacks accepted_attempt: {marker}")
        attempt_id = str(accepted.get("attempt_id", ""))
        cycle_attempt = (marker.parent / attempt_id).resolve()
        if not attempt_id or cycle_attempt.parent != marker.parent.resolve():
            raise ThreeFrameScheduleError(f"Committed attempt path is invalid: {marker}")
        occupancy_path = cycle_attempt / "occupancy_mapping" / "metadata.json"
        occupancy = _json(occupancy_path)
        frames = occupancy.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ThreeFrameScheduleError(f"Occupancy source window is empty: {occupancy_path}")
        source_counts.append(len(frames))
        physical_sources.append(
            [str(frame["evidence"]["physical_source_id"]) for frame in frames]
        )
        cycle_authorities.append(
            {
                "committed": _authority(marker),
                "occupancy": _authority(occupancy_path),
            }
        )
    generation_payloads = [_json(path) for path in generation_files]
    generation_view_counts = [len(payload.get("views", [])) for payload in generation_payloads]
    expected = [1, 2, 3]
    if source_counts != expected or generation_view_counts != expected:
        raise ThreeFrameScheduleError(
            "Committed occupancy/generation topology is not the expected 1/2/3 sequence"
        )
    final_summary = generation_payloads[-1].get("summary")
    if not isinstance(final_summary, dict) or final_summary.get("schema5_ready") is not False:
        raise ThreeFrameScheduleError(
            "Final three-view generation must record schema5_ready=false"
        )
    run_event_files = sorted((attempt / "runs/coarse/events").glob("*.json"))
    map_ready_files = [
        path for path in run_event_files if _json(path).get("event_type") == "map_ready"
    ]
    if not map_ready_files:
        raise ThreeFrameScheduleError(
            "Three-frame transition model requires an immutable MAP_READY run event"
        )
    schedules = replay_schedule(tuple(source_counts))
    return {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.three_frame_replay_schedule",
        "authority": "diagnostic_only_not_safety_or_science_authority",
        "attempt_root": str(attempt),
        "source_counts": source_counts,
        "generation_view_counts": generation_view_counts,
        "checkpoint_count": len(checkpoint_files),
        "map_ready_event_count": len(map_ready_files),
        "final_schema5_ready": False,
        "physical_source_ids": physical_sources,
        "cycle_authorities": cycle_authorities,
        "generation_authorities": [_authority(path) for path in generation_files],
        "checkpoint_authorities": [_authority(path) for path in checkpoint_files],
        "map_ready_event_authorities": [_authority(path) for path in map_ready_files],
        "schedule": [item.payload() for item in schedules],
        "totals": {
            "before": sum(item.before_total for item in schedules),
            "after": sum(item.after_total for item in schedules),
            "eliminated": sum(
                item.before_total - item.after_total for item in schedules
            ),
        },
        "notes": [
            "Counts are logical source integrations, not measured seconds.",
            (
                "The diagnostic validates the immutable 1/2/3 topology and applies "
                "a source-reviewed call-count model; it does not instrument production "
                "reader calls."
            ),
            (
                "The final-frame term is the two-read MAP_READY transition evaluation; "
                "the inspected three-view generation is not schema-5-ready."
            ),
            (
                "Counts stop at that capture callback boundary and exclude the later "
                "select_next read, a future successful schema-5 writer, and fine handoff."
            ),
            "GPU backend evaluation must hold this schedule fixed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = inspect_attempt(arguments.attempt_root)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        output = arguments.output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
