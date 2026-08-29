"""Read-only status and event views for a supervised stop-scan experiment.

The contracts in this module deliberately expose no robot, executor, approval, or
command method.  They are safe projections of the append-only run evidence used by
the workflow composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from biblade_fusion.storage.stop_scan_run import (
    StopScanRunEvent,
    read_stop_scan_run,
)


class ExperimentDisposition(StrEnum):
    """Operator-facing disposition without granting a motion capability."""

    READY = "ready"
    NEEDS_CAPTURE = "needs_capture"
    WAITING_APPROVAL = "waiting_approval"
    BLOCKED = "blocked"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class ExperimentStatusSnapshot:
    """Immutable, read-only projection of one supervised workflow state."""

    run_id: str
    run_root: Path
    phase: str
    disposition: ExperimentDisposition
    cycle_index: int
    current_view_id: str | None
    proposed_view_id: str | None
    expected_capture_view_id: str | None
    expected_capture_purpose: str | None
    blocking_reasons: tuple[str, ...]
    event_count: int
    latest_event_sha256: str | None
    recovery_required: bool
    awaiting_external_approval: bool
    stop_requested: bool = False
    stop_transport_acknowledged: bool = False
    stop_stationarity_verified: bool = False
    motion_command_capable: Literal[False] = False

    def __post_init__(self) -> None:
        root = Path(self.run_root).resolve()
        if not self.run_id.strip() or not self.phase.strip():
            raise ValueError("Experiment status requires non-empty run and phase identities")
        if self.cycle_index < 0 or self.event_count < 0:
            raise ValueError("Experiment status counters must be non-negative")
        if self.latest_event_sha256 is not None and (
            len(self.latest_event_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.latest_event_sha256)
        ):
            raise ValueError("Latest experiment event identity must be a SHA-256 digest")
        if self.motion_command_capable is not False:
            raise ValueError("A supervision snapshot can never expose motion commands")
        if (
            type(self.stop_requested) is not bool
            or type(self.stop_transport_acknowledged) is not bool
            or type(self.stop_stationarity_verified) is not bool
        ):
            raise ValueError("Experiment stop status fields must be boolean")
        if self.stop_transport_acknowledged and not self.stop_requested:
            raise ValueError("Stop transport acknowledgement requires a stop request")
        if self.stop_stationarity_verified and not self.stop_transport_acknowledged:
            raise ValueError("Stop stationarity requires transport acknowledgement")
        reasons = tuple(str(value).strip() for value in self.blocking_reasons)
        if any(not value for value in reasons):
            raise ValueError("Experiment blocking reasons must be non-empty strings")
        if self.disposition is ExperimentDisposition.BLOCKED and not reasons:
            raise ValueError("A blocked experiment status requires a reason")
        if self.awaiting_external_approval != (
            self.disposition is ExperimentDisposition.WAITING_APPROVAL
        ):
            raise ValueError("Approval flag and experiment disposition disagree")
        object.__setattr__(self, "run_root", root)
        object.__setattr__(self, "blocking_reasons", reasons)


@dataclass(frozen=True, slots=True)
class ExperimentEventBatch:
    """Verified append-only events and the next exclusive sequence cursor."""

    run_id: str
    events: tuple[StopScanRunEvent, ...]
    next_sequence: int

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.next_sequence < 0:
            raise ValueError("Experiment event cursor is invalid")
        if self.events:
            sequences = tuple(event.sequence for event in self.events)
            if sequences != tuple(range(sequences[0], self.next_sequence)):
                raise ValueError("Experiment event batch must be contiguous")


def read_experiment_events(
    run_root: str | Path,
    *,
    from_sequence: int = 0,
) -> ExperimentEventBatch:
    """Read events at/after ``from_sequence`` and return the next cursor.

    The complete hash chain is verified before slicing.  Passing the returned
    ``next_sequence`` to the next call therefore yields exactly the new suffix.
    """

    if from_sequence < 0:
        raise ValueError("Experiment event cursor cannot be negative")
    stored = read_stop_scan_run(run_root)
    if from_sequence > len(stored.events):
        raise ValueError("Experiment event cursor is beyond the verified run")
    events = stored.events[from_sequence:]
    return ExperimentEventBatch(stored.run_id, events, len(stored.events))
