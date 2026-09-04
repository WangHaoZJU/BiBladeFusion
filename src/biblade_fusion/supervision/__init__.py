"""Read-only supervision contracts and user interfaces.

This package deliberately has no dependency on robot command or execution ports.
"""

from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentEventBatch,
    ExperimentStatusSnapshot,
    read_experiment_events,
)
from biblade_fusion.supervision.snapshot import (
    ArrayReference,
    CandidatePlanningSnapshot,
    LivePlanningUpdate,
    PlanningProgressSnapshot,
    StoredSupervisorySnapshot,
    SupervisorySnapshot,
    SupervisoryTimeline,
    discover_supervisory_snapshots,
    load_snapshot_array,
    read_live_planning_update,
    read_supervisory_snapshot,
    snapshot_array_references,
    write_live_planning_update,
)
from biblade_fusion.supervision.storage import AtomicSupervisorySnapshotWriter

__all__ = [
    "ArrayReference",
    "CandidatePlanningSnapshot",
    "AtomicSupervisorySnapshotWriter",
    "StoredSupervisorySnapshot",
    "LivePlanningUpdate",
    "PlanningProgressSnapshot",
    "SupervisorySnapshot",
    "SupervisoryTimeline",
    "discover_supervisory_snapshots",
    "load_snapshot_array",
    "read_live_planning_update",
    "read_supervisory_snapshot",
    "snapshot_array_references",
    "write_live_planning_update",
    "ExperimentDisposition",
    "ExperimentEventBatch",
    "ExperimentStatusSnapshot",
    "read_experiment_events",
]
