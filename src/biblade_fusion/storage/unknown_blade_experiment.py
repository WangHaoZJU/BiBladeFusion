"""Append-only coarse-to-fine handoff authority for one unknown-blade experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from biblade_fusion.diagnostics.performance_timing import performance_span
from biblade_fusion.storage.coarse_scan import read_coarse_scan_generation
from biblade_fusion.storage.fine_reconstruction import (
    replay_final_fine_reconstruction,
)
from biblade_fusion.storage.runtime_timing_acceptance import (
    RuntimeTimingAcceptanceAuthority,
)
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.stop_scan_run import (
    exclusive_stop_scan_run_authority,
    read_stop_scan_run,
    validate_stop_scan_run_id,
)
from biblade_fusion.storage.surface_coverage import read_surface_coverage_generation

UNKNOWN_BLADE_EXPERIMENT_SCHEMA_VERSION = 1
UNKNOWN_BLADE_EXPERIMENT_KIND = "biblade_fusion.unknown_blade_experiment_event"
UNKNOWN_BLADE_FINE_START_PROTOCOL = "candidate_commit.stop_scan_exclusive.v2"
_FINE_START_PREPUBLICATION_DURATION_SEMANTICS = (
    "lower_bound_sample_before_final_event_serialization"
)
_FINE_START_PUBLICATION_DEADLINE_CONTRACT = (
    "final_budget_check_after_event_fsync_before_atomic_publish"
)

UnknownBladeExperimentEventType = Literal[
    "experiment_initialized",
    "coarse_checkpoint",
    "handoff_prepared",
    "fine_start_candidate",
    "fine_started",
    "fine_checkpoint",
    "fine_completed",
]

_EVENT_DIGITS = 8
_EVENT_PATTERN = re.compile(r"([0-9]{8})\.json")
_EVENT_PARTIAL_PATTERN = re.compile(r"\.[0-9]{8}\.json\.[0-9a-f]{32}\.partial")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVENT_TYPES = (
    "experiment_initialized",
    "coarse_checkpoint",
    "handoff_prepared",
    "fine_start_candidate",
    "fine_started",
    "fine_checkpoint",
    "fine_completed",
)
_EVENT_KEYS = {
    "schema_version",
    "artifact_kind",
    "experiment_id",
    "sequence",
    "event_type",
    "created_at_utc",
    "payload",
    "previous_event_sha256",
    "event_sha256",
}


class UnknownBladeExperimentFormatError(ValueError):
    """The top-level unknown-blade experiment chain is incomplete or changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_loads(text: str) -> Any:
    """Decode JSON without accepting duplicate keys or non-finite constants."""

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )


def _canonical_utc(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(UTC)
    elif isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("event timestamp is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event timestamp must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def _canonical_root(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    resolved = raw.resolve()
    if not raw.is_absolute() or raw != resolved:
        raise ValueError(f"{label} must be an absolute canonical path")
    return resolved


def _authority_record(root: Path, filename: str) -> dict[str, Any]:
    root = root.resolve()
    relative = Path(filename)
    authority = (root / relative).resolve()
    if relative.is_absolute() or not authority.is_relative_to(root) or not authority.is_file():
        raise ValueError(f"Experiment handoff authority is missing: {authority}")
    return {
        "root": str(root),
        "authority": filename,
        "sha256": _sha256(authority),
        "size_bytes": authority.stat().st_size,
    }


def _verify_authority_record(record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"root", "authority", "sha256", "size_bytes"}:
        raise ValueError(f"{label} authority record keys changed")
    root_value = record["root"]
    authority_value = record["authority"]
    digest_value = record["sha256"]
    size_value = record["size_bytes"]
    if not isinstance(root_value, str) or not isinstance(authority_value, str):
        raise ValueError(f"{label} authority paths must be strings")
    if not isinstance(digest_value, str):
        raise ValueError(f"{label} authority digest must be a string")
    if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 0:
        raise ValueError(f"{label} authority size must be a non-negative integer")
    root = _canonical_root(root_value, label=f"{label} root")
    relative = Path(authority_value)
    authority = (root / relative).resolve()
    if relative.is_absolute() or not authority.is_relative_to(root) or not authority.is_file():
        raise ValueError(f"{label} authority path changed")
    if (
        _SHA256_PATTERN.fullmatch(digest_value) is None
        or _sha256(authority) != digest_value
        or authority.stat().st_size != size_value
    ):
        raise ValueError(f"{label} authority content changed")
    return root


def _validate_fine_start_bootstrap_run(
    fine: Any,
    *,
    require_exactly_one_event: bool,
) -> None:
    """Validate the real StopScan bootstrap event used as fine-run authority.

    ``read_stop_scan_run`` has already verified the event's strict JSON schema,
    canonical hash and predecessor chain.  This check adds the workflow semantic
    contract: a candidate can only be minted from the single initial event emitted
    by :meth:`StopScanCoordinator.start`, never from an already-used run.
    """

    if not fine.events:
        raise ValueError("fine run has no bootstrap event")
    if require_exactly_one_event and len(fine.events) != 1:
        raise ValueError("fine-start candidate requires exactly one fine-run event")
    first = fine.events[0]
    if (
        first.sequence != 0
        or first.previous_event_sha256 is not None
        or first.phase != "bootstrap_map_required"
        or first.cycle_index != 0
        or first.event_type != "run_started"
    ):
        raise ValueError(
            "fine run must begin with the cycle-0 bootstrap_map_required/run_started event"
        )
    payload = first.payload
    if set(payload) != {"depth_backend", "bootstrap_mode", "minimum_source_views"}:
        raise ValueError("fine run_started payload differs from the StopScan schema")
    minimum_source_views = payload["minimum_source_views"]
    if (
        payload["depth_backend"] != "foundation_stereo"
        or payload["bootstrap_mode"] != "operator_guided"
        or isinstance(minimum_source_views, bool)
        or not isinstance(minimum_source_views, int)
        or minimum_source_views < 1
    ):
        raise ValueError("fine run_started payload has invalid bootstrap semantics")


def _verify_fine_terminal_assets(
    fine: Any,
    *,
    coverage_root: Path,
    reconstruction_root: Path,
    science_authority: ScienceAcceptanceAuthority | None,
) -> None:
    terminal = fine.latest_event
    if terminal.phase != "complete" or terminal.event_type != "coverage_complete":
        raise ValueError("fine run lacks its terminal coverage_complete event")
    terminal_reconstruction = terminal.payload.get("final_reconstruction")
    if not isinstance(terminal_reconstruction, Mapping) or set(
        terminal_reconstruction
    ) != {"path", "artifact_id", "metadata_sha256"}:
        raise ValueError("fine terminal event lacks immutable reconstruction evidence")
    stored = (
        replay_final_fine_reconstruction(
            reconstruction_root,
            expected_science_authority=science_authority,
        )
        if science_authority is not None
        else replay_final_fine_reconstruction(reconstruction_root)
    )
    if (
        stored.root != reconstruction_root
        or stored.result.coverage.root != coverage_root
        or terminal.payload.get("surface_generation_id")
        != stored.result.coverage.generation_id
        or Path(str(terminal_reconstruction["path"])).resolve() != stored.root
        or terminal_reconstruction["artifact_id"] != stored.artifact_id
        or terminal_reconstruction["metadata_sha256"] != stored.metadata_sha256
        or getattr(stored, "science_authority", None) != science_authority
    ):
        raise ValueError("fine terminal event and replayed reconstruction disagree")


def _event_hash_payload(
    *,
    experiment_id: str,
    sequence: int,
    event_type: str,
    created_at_utc: str,
    payload: Mapping[str, Any],
    previous_event_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": UNKNOWN_BLADE_EXPERIMENT_SCHEMA_VERSION,
        "artifact_kind": UNKNOWN_BLADE_EXPERIMENT_KIND,
        "experiment_id": experiment_id,
        "sequence": sequence,
        "event_type": event_type,
        "created_at_utc": created_at_utc,
        "payload": dict(payload),
        "previous_event_sha256": previous_event_sha256,
    }


@dataclass(frozen=True, slots=True)
class UnknownBladeExperimentEvent:
    experiment_id: str
    sequence: int
    event_type: UnknownBladeExperimentEventType
    created_at_utc: str
    payload: dict[str, Any]
    previous_event_sha256: str | None
    event_sha256: str

    def __post_init__(self) -> None:
        experiment_id = validate_stop_scan_run_id(self.experiment_id)
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("experiment event sequence must be non-negative")
        if self.event_type not in _EVENT_TYPES:
            raise ValueError("unknown-blade experiment event type is unsupported")
        created = _canonical_utc(self.created_at_utc)
        if created != self.created_at_utc:
            raise ValueError("experiment event timestamp is not canonical UTC")
        previous = self.previous_event_sha256
        if (self.sequence == 0) != (previous is None):
            raise ValueError("experiment event predecessor/sequence mismatch")
        if previous is not None and _SHA256_PATTERN.fullmatch(previous) is None:
            raise ValueError("experiment event predecessor SHA-256 is malformed")
        if not isinstance(self.payload, dict):
            raise ValueError("experiment event payload must be an object")
        normalised = json.loads(_canonical_json(self.payload))
        expected = hashlib.sha256(
            _canonical_json(
                _event_hash_payload(
                    experiment_id=experiment_id,
                    sequence=self.sequence,
                    event_type=self.event_type,
                    created_at_utc=created,
                    payload=normalised,
                    previous_event_sha256=previous,
                )
            )
        ).hexdigest()
        if self.event_sha256 != expected:
            raise ValueError("experiment event SHA-256 does not match canonical content")
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "payload", normalised)

    @classmethod
    def build(
        cls,
        *,
        experiment_id: str,
        sequence: int,
        event_type: UnknownBladeExperimentEventType,
        payload: dict[str, Any],
        previous_event_sha256: str | None,
        created_at_utc: datetime | str | None = None,
    ) -> UnknownBladeExperimentEvent:
        created = _canonical_utc(created_at_utc)
        body = json.loads(_canonical_json(payload))
        digest = hashlib.sha256(
            _canonical_json(
                _event_hash_payload(
                    experiment_id=validate_stop_scan_run_id(experiment_id),
                    sequence=sequence,
                    event_type=event_type,
                    created_at_utc=created,
                    payload=body,
                    previous_event_sha256=previous_event_sha256,
                )
            )
        ).hexdigest()
        return cls(
            experiment_id,
            sequence,
            event_type,
            created,
            body,
            previous_event_sha256,
            digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **_event_hash_payload(
                experiment_id=self.experiment_id,
                sequence=self.sequence,
                event_type=self.event_type,
                created_at_utc=self.created_at_utc,
                payload=self.payload,
                previous_event_sha256=self.previous_event_sha256,
            ),
            "event_sha256": self.event_sha256,
        }


def _decode_event_payload(
    raw: Any,
    *,
    expected_sequence: int,
) -> UnknownBladeExperimentEvent:
    """Apply the reader's strict schema and canonical-hash checks to one event."""

    if not isinstance(raw, dict) or set(raw) != _EVENT_KEYS:
        raise ValueError("experiment event keys changed")
    schema_version = raw["schema_version"]
    artifact_kind = raw["artifact_kind"]
    experiment_id = raw["experiment_id"]
    sequence = raw["sequence"]
    event_type = raw["event_type"]
    created_at_utc = raw["created_at_utc"]
    payload = raw["payload"]
    previous_event_sha256 = raw["previous_event_sha256"]
    event_sha256 = raw["event_sha256"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("experiment schema_version must be an integer")
    if not isinstance(artifact_kind, str):
        raise ValueError("experiment artifact_kind must be a string")
    if not isinstance(experiment_id, str):
        raise ValueError("experiment experiment_id must be a string")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ValueError("experiment sequence must be an integer")
    if not isinstance(event_type, str):
        raise ValueError("experiment event_type must be a string")
    if not isinstance(created_at_utc, str):
        raise ValueError("experiment created_at_utc must be a string")
    if not isinstance(payload, dict):
        raise ValueError("experiment payload must be an object")
    if previous_event_sha256 is not None and not isinstance(
        previous_event_sha256,
        str,
    ):
        raise ValueError("experiment previous_event_sha256 must be null or a string")
    if not isinstance(event_sha256, str):
        raise ValueError("experiment event_sha256 must be a string")
    if (
        schema_version != UNKNOWN_BLADE_EXPERIMENT_SCHEMA_VERSION
        or artifact_kind != UNKNOWN_BLADE_EXPERIMENT_KIND
        or sequence != expected_sequence
    ):
        raise ValueError("experiment event schema or sequence changed")
    return UnknownBladeExperimentEvent(
        experiment_id=experiment_id,
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        created_at_utc=created_at_utc,
        payload=payload,
        previous_event_sha256=previous_event_sha256,
        event_sha256=event_sha256,
    )


@dataclass(frozen=True, slots=True)
class StoredUnknownBladeExperiment:
    root: Path
    experiment_id: str
    events: tuple[UnknownBladeExperimentEvent, ...]
    science_authority: ScienceAcceptanceAuthority | None = None
    runtime_timing_authority: RuntimeTimingAcceptanceAuthority | None = None
    fine_start_protocol: str | None = None
    placement_id: str | None = None

    @property
    def latest_event(self) -> UnknownBladeExperimentEvent:
        return self.events[-1]


def _event_filename(sequence: int) -> str:
    if sequence >= 10**_EVENT_DIGITS:
        raise ValueError("experiment event sequence exceeds filename capacity")
    return f"{sequence:0{_EVENT_DIGITS}d}.json"


def _write_new_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    before_publish: Callable[[], None] | None = None,
    after_publish: Callable[[], None] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment event: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # The callback is deliberately inside the storage linearization path:
        # the complete event bytes are durable in a same-directory temporary,
        # but no authoritative event filename exists yet.  A deadline failure
        # therefore removes only the temporary and cannot become resumable.
        if before_publish is not None:
            before_publish()
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if after_publish is not None:
            after_publish()
    finally:
        temporary.unlink(missing_ok=True)


class UnknownBladeExperimentWriter:
    """Append the checkpointed one-way coarse-to-fine experiment chain."""

    def __init__(
        self,
        stored: StoredUnknownBladeExperiment,
        *,
        production: bool = True,
    ) -> None:
        if production and (
            stored.science_authority is None or stored.runtime_timing_authority is None
        ):
            raise ValueError(
                "Production experiment writer requires science and runtime timing authorities; "
                "legacy chains are audit-readable only"
            )
        if production and stored.fine_start_protocol != UNKNOWN_BLADE_FINE_START_PROTOCOL:
            raise ValueError(
                "Legacy single-phase fine-start chains are audit-readable only and "
                "cannot be resumed by the production writer"
            )
        if not production and (
            stored.science_authority is not None
            or stored.runtime_timing_authority is not None
            or stored.fine_start_protocol is not None
        ):
            raise ValueError("Experimental coarse writer cannot carry production authorities")
        self.root = stored.root
        self.experiment_id = stored.experiment_id
        self.placement_id = stored.placement_id
        self._production = production
        self._events = list(stored.events)
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        *,
        experiment_id: str,
        coarse_run_root: str | Path,
        coarse_run_id: str | None = None,
        placement_id: str | None = None,
        science_authority: ScienceAcceptanceAuthority | None = None,
        runtime_timing_authority: RuntimeTimingAcceptanceAuthority | None = None,
        production: bool = True,
    ) -> UnknownBladeExperimentWriter:
        if production and (science_authority is None or runtime_timing_authority is None):
            raise ValueError(
                "New production experiment chains require science and runtime timing authorities"
            )
        if not production and (
            science_authority is not None or runtime_timing_authority is not None
        ):
            raise ValueError(
                "Experimental coarse chains cannot claim production acceptance authorities"
            )
        output = Path(output_dir).resolve()
        coarse_root = Path(coarse_run_root).resolve()
        expected_id = validate_stop_scan_run_id(experiment_id)
        bound_placement_id = (
            validate_stop_scan_run_id(placement_id)
            if placement_id is not None
            else None
        )
        events_root = coarse_root / "events"
        if events_root.is_dir() and not any(events_root.iterdir()):
            if coarse_run_id is None:
                raise ValueError(
                    "An eventless coarse run reservation requires its explicit run ID"
                )
            bound_run_id = validate_stop_scan_run_id(coarse_run_id)
            bound_run_root = coarse_root
        else:
            coarse = read_stop_scan_run(coarse_root)
            bound_run_id = coarse.run_id
            bound_run_root = coarse.root
        if bound_run_id != expected_id:
            raise ValueError("coarse run ID differs from experiment ID")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        (output / "events").mkdir()
        stored = StoredUnknownBladeExperiment(
            root=output,
            experiment_id=experiment_id,
            events=(),
            science_authority=science_authority,
            runtime_timing_authority=runtime_timing_authority,
            fine_start_protocol=(
                UNKNOWN_BLADE_FINE_START_PROTOCOL if production else None
            ),
            placement_id=bound_placement_id,
        )
        writer = cls(stored, production=production)
        writer._append(
            "experiment_initialized",
            {
                "experiment_id": experiment_id,
                "coarse_run_id": bound_run_id,
                "coarse_run_root": str(bound_run_root),
                **(
                    {"placement_id": bound_placement_id}
                    if bound_placement_id is not None
                    else {}
                ),
                **(
                    {"fine_start_protocol": UNKNOWN_BLADE_FINE_START_PROTOCOL}
                    if production
                    else {}
                ),
                **(
                    {"science_acceptance_authority": science_authority.to_payload()}
                    if science_authority is not None
                    else {}
                ),
                **(
                    {
                        "runtime_timing_acceptance_authority": (
                            runtime_timing_authority.to_payload()
                        )
                    }
                    if runtime_timing_authority is not None
                    else {}
                ),
            },
        )
        return writer

    @classmethod
    def resume(cls, output_dir: str | Path) -> UnknownBladeExperimentWriter:
        return cls(read_unknown_blade_experiment(output_dir))

    @property
    def events(self) -> tuple[UnknownBladeExperimentEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def prepare_handoff(
        self,
        *,
        schema5_generation: str | Path,
        reference_coarse_model: str | Path,
        schema5_prepare_duration_s: float | None = None,
    ) -> UnknownBladeExperimentEvent:
        with self._lock:
            self._require_current_chain()
            checkpoint = self._events[-1]
            if checkpoint.event_type != "coarse_checkpoint":
                raise ValueError("handoff PREPARED requires a preceding COARSE_CHECKPOINT")
            initialized = self._events[0].payload
            coarse = read_stop_scan_run(initialized["coarse_run_root"])
            if not coarse.events:
                raise ValueError("coarse run has no terminal handoff event")
            generation_root = Path(schema5_generation).resolve()
            reference_root = Path(reference_coarse_model).resolve()
            generation = read_coarse_scan_generation(generation_root)
            if generation.root != generation_root or generation.coarse_model_path != reference_root:
                raise ValueError("schema-5 generation/reference binding changed")
            generation_authority = _authority_record(generation_root, "generation.json")
            checkpoint_generation_root = _canonical_root(
                checkpoint.payload["coarse_generation"]["root"],
                label="final coarse checkpoint generation",
            )
            if (
                int(checkpoint.payload["coarse_event_count"]) != len(coarse.events)
                or checkpoint.payload["coarse_last_event_sha256"]
                != coarse.latest_event.event_sha256
                or generation.previous_generation_path is None
                or generation.previous_generation_path.resolve()
                != checkpoint_generation_root
            ):
                raise ValueError(
                    "PREPARED ready generation must derive from the final coarse checkpoint"
                )
            timing_authority = self._events_runtime_timing_authority()
            timing_payload: dict[str, object] = {}
            if timing_authority is not None:
                if (
                    schema5_prepare_duration_s is None
                    or not math.isfinite(schema5_prepare_duration_s)
                    or schema5_prepare_duration_s < 0.0
                ):
                    raise ValueError("PREPARED requires a finite schema-5 prepare duration")
                limit = timing_authority.timing_limits_s[
                    "maximum_schema5_handoff_duration_s"
                ]
                if schema5_prepare_duration_s > limit:
                    raise ValueError("PREPARED schema-5 duration exceeds accepted limit")
                timing_payload["schema5_prepare_timing"] = {
                    "actual_duration_s": schema5_prepare_duration_s,
                    "accepted_limit_s": limit,
                    "runtime_timing_acceptance_id": timing_authority.acceptance_id,
                    "runtime_timing_metadata_sha256": timing_authority.metadata_sha256,
                }
            elif schema5_prepare_duration_s is not None:
                raise ValueError("Legacy experiment cannot add unbound prepare timing")
            return self._append(
                "handoff_prepared",
                {
                    "coarse_checkpoint_event_sha256": checkpoint.event_sha256,
                    "coarse_run_id": coarse.run_id,
                    "coarse_run_root": str(coarse.root),
                    "coarse_last_event_sha256": coarse.latest_event.event_sha256,
                    "coarse_event_count": len(coarse.events),
                    "schema5_generation": generation_authority,
                    "reference_coarse_model": _authority_record(reference_root, "metadata.json"),
                    **timing_payload,
                },
            )

    def append_coarse_checkpoint(
        self,
        *,
        coarse_generation: str | Path,
    ) -> UnknownBladeExperimentEvent:
        """Bind one accepted coarse generation to the run event that produced it."""

        with self._lock:
            self._require_current_event_integrity()
            coarse_root = Path(self._events[0].payload["coarse_run_root"]).resolve()
            with exclusive_stop_scan_run_authority(coarse_root):
                previous = self._events[-1]
                if previous.event_type not in {
                    "experiment_initialized",
                    "coarse_checkpoint",
                }:
                    raise ValueError("COARSE_CHECKPOINT is only valid before PREPARED")
                coarse = read_stop_scan_run(coarse_root)
                if not coarse.events:
                    raise ValueError("coarse run has no checkpoint event")
                generation_root = Path(coarse_generation).resolve()
                authority = _authority_record(generation_root, "generation.json")
                if previous.event_type == "coarse_checkpoint" and (
                    previous.payload["coarse_last_event_sha256"]
                    == coarse.latest_event.event_sha256
                    and previous.payload["coarse_generation"] == authority
                ):
                    raise ValueError("coarse checkpoint run/generation pair must advance")
                return self._append(
                    "coarse_checkpoint",
                    {
                        "coarse_run_id": coarse.run_id,
                        "coarse_run_root": str(coarse.root),
                        "coarse_last_event_sha256": coarse.latest_event.event_sha256,
                        "coarse_event_count": len(coarse.events),
                        "coarse_generation": authority,
                    },
                    incremental_coarse_checkpoint=True,
                )

    def append_fine_start_candidate(
        self,
        *,
        fine_run_root: str | Path,
    ) -> UnknownBladeExperimentEvent:
        """Persist a non-authoritative fine-run candidate.

        More than one candidate may follow PREPARED so a stopped recovery can
        abandon an orphan fine-run directory and bind a newly created run.  A
        candidate alone is never sufficient to resume the experiment as FINE.
        """

        with self._lock:
            self._require_current_chain()
            previous = self._events[-1]
            if previous.event_type not in {"handoff_prepared", "fine_start_candidate"}:
                raise ValueError(
                    "FINE_START_CANDIDATE requires PREPARED or an uncommitted candidate"
                )
            prepared = next(
                event for event in self._events if event.event_type == "handoff_prepared"
            )
            with exclusive_stop_scan_run_authority(fine_run_root):
                fine = read_stop_scan_run(fine_run_root)
                if fine.run_id != self.experiment_id:
                    raise ValueError("fine run ID or first event is unavailable")
                _validate_fine_start_bootstrap_run(
                    fine,
                    require_exactly_one_event=True,
                )
                candidate_identity = (
                    str(fine.root),
                    fine.events[0].event_sha256,
                )
                for event in self._events:
                    if event.event_type != "fine_start_candidate":
                        continue
                    if (
                        event.payload["fine_run_root"],
                        event.payload["fine_first_event_sha256"],
                    ) == candidate_identity:
                        raise ValueError("fine-start candidate must bind a new fine run")
                return self._append(
                    "fine_start_candidate",
                    {
                        "prepared_event_sha256": prepared.event_sha256,
                        "fine_run_id": fine.run_id,
                        "fine_run_root": str(fine.root),
                        "fine_first_event_sha256": fine.events[0].event_sha256,
                        "fine_event_count_at_candidate": 1,
                    },
                )

    def append_fine_started(
        self,
        *,
        timing_scope: Literal["uninterrupted_total", "resume_fine_start"],
        budget_check: Callable[[], float],
    ) -> UnknownBladeExperimentEvent:
        """Atomically commit the latest candidate while its budget is valid.

        ``budget_check`` is invoked once before event construction and again by
        ``_write_new_json`` after the final event temporary has been flushed and
        fsynced, immediately before its no-replace hard-link publication.
        """

        with self._lock:
            self._require_current_chain()
            candidate = self._events[-1]
            if candidate.event_type != "fine_start_candidate":
                raise ValueError(
                    "FINE_STARTED requires exactly one latest FINE_START_CANDIDATE"
                )
            prepared = next(
                event for event in self._events if event.event_type == "handoff_prepared"
            )
            fine_run_root = candidate.payload["fine_run_root"]
            with exclusive_stop_scan_run_authority(fine_run_root):

                def require_single_event_fine_authority() -> Any:
                    current = read_stop_scan_run(fine_run_root)
                    _validate_fine_start_bootstrap_run(
                        current,
                        require_exactly_one_event=True,
                    )
                    if (
                        current.run_id != self.experiment_id
                        or candidate.payload["prepared_event_sha256"]
                        != prepared.event_sha256
                        or candidate.payload["fine_run_id"] != current.run_id
                        or candidate.payload["fine_run_root"] != str(current.root)
                        or candidate.payload["fine_first_event_sha256"]
                        != current.events[0].event_sha256
                        or _checkpoint_event_count(
                            candidate.payload["fine_event_count_at_candidate"],
                            label="fine-start candidate",
                        )
                        != 1
                    ):
                        raise ValueError(
                            "fine-start candidate authority changed before commit"
                        )
                    return current

                fine = require_single_event_fine_authority()
                timing_authority = self._events_runtime_timing_authority()
                if timing_authority is None:
                    raise ValueError("FINE_STARTED requires runtime timing authority")
                if timing_scope not in {"uninterrupted_total", "resume_fine_start"}:
                    raise ValueError("FINE_STARTED timing scope is unsupported")
                limit = timing_authority.timing_limits_s[
                    "maximum_schema5_handoff_duration_s"
                ]

                def checked_duration(*, minimum_s: float | None = None) -> float:
                    value = budget_check()
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                    ):
                        raise ValueError(
                            "FINE_STARTED budget callback returned invalid duration"
                        )
                    duration = float(value)
                    if minimum_s is not None and duration < minimum_s:
                        raise ValueError("FINE_STARTED budget clock moved backwards")
                    if duration > limit:
                        raise ValueError(
                            "FINE_STARTED handoff duration exceeds accepted limit"
                        )
                    return duration

                prepublication_check_duration_s = checked_duration()

                def require_budget_before_publish() -> None:
                    timing_authority.assert_acceptance_asset_current()
                    checked_duration(minimum_s=prepublication_check_duration_s)
                    # This is the final fine-run readback.  It executes after the
                    # outer event temporary is durable and while the shared run
                    # authority remains exclusively held, immediately before the
                    # outer hard-link publication linearization point.
                    require_single_event_fine_authority()

                return self._append(
                    "fine_started",
                    {
                        "fine_start_candidate_event_sha256": candidate.event_sha256,
                        "prepared_event_sha256": prepared.event_sha256,
                        "fine_run_id": fine.run_id,
                        "fine_run_root": str(fine.root),
                        "fine_first_event_sha256": fine.events[0].event_sha256,
                        "schema5_handoff_timing": {
                            "prepublication_check_duration_s": (
                                prepublication_check_duration_s
                            ),
                            "prepublication_duration_semantics": (
                                _FINE_START_PREPUBLICATION_DURATION_SEMANTICS
                            ),
                            "accepted_limit_s": limit,
                            "runtime_timing_acceptance_id": timing_authority.acceptance_id,
                            "runtime_timing_metadata_sha256": (
                                timing_authority.metadata_sha256
                            ),
                            "measurement_scope": timing_scope,
                            "publication_deadline_contract": (
                                _FINE_START_PUBLICATION_DEADLINE_CONTRACT
                            ),
                        },
                    },
                    before_publish=require_budget_before_publish,
                )

    def append_unaccepted_fine_started(
        self,
        *,
        fine_run_root: str | Path,
    ) -> UnknownBladeExperimentEvent:
        """Bind an experiment-only fine run without claiming release acceptance."""

        if self._production:
            raise ValueError("Production chains cannot use an unaccepted fine handoff")
        with self._lock:
            self._require_current_chain()
            if self._events[-1].event_type != "handoff_prepared":
                raise ValueError("Experimental FINE_STARTED requires PREPARED")
            prepared = self._events[-1]
            with exclusive_stop_scan_run_authority(fine_run_root):
                fine = read_stop_scan_run(fine_run_root)
                _validate_fine_start_bootstrap_run(
                    fine,
                    require_exactly_one_event=True,
                )
                if fine.run_id != self.experiment_id:
                    raise ValueError("experimental fine run ID differs from experiment ID")
                return self._append(
                    "fine_started",
                    {
                        "prepared_event_sha256": prepared.event_sha256,
                        "fine_run_id": fine.run_id,
                        "fine_run_root": str(fine.root),
                        "fine_first_event_sha256": fine.events[0].event_sha256,
                    },
                )

    def append_fine_checkpoint(
        self,
        *,
        accepted_surface_coverage_generation: str | Path,
    ) -> UnknownBladeExperimentEvent:
        """Bind the latest accepted fine coverage to the current fine terminal event."""

        with self._lock:
            self._require_current_chain()
            previous = self._events[-1]
            if previous.event_type not in {"fine_started", "fine_checkpoint"}:
                raise ValueError("FINE_CHECKPOINT is only valid after FINE_STARTED")
            started = next(
                event.payload for event in self._events if event.event_type == "fine_started"
            )
            fine = read_stop_scan_run(started["fine_run_root"])
            if fine.run_id != self.experiment_id or not fine.events:
                raise ValueError("fine run identity or checkpoint event is unavailable")
            coverage_root = Path(accepted_surface_coverage_generation).resolve()
            coverage = read_surface_coverage_generation(
                coverage_root,
                require_foreground_bound_science=True,
            )
            if coverage.root.resolve() != coverage_root:
                raise ValueError("accepted fine coverage readback changed its root")
            authority = _authority_record(coverage_root, "coverage.json")
            if previous.event_type == "fine_checkpoint" and (
                previous.payload["fine_last_event_sha256"]
                == fine.latest_event.event_sha256
                and previous.payload["accepted_surface_coverage_generation"] == authority
            ):
                raise ValueError("fine checkpoint run/coverage pair must advance")
            return self._append(
                "fine_checkpoint",
                {
                    "fine_run_id": fine.run_id,
                    "fine_run_root": str(fine.root),
                    "fine_last_event_sha256": fine.latest_event.event_sha256,
                    "fine_event_count": len(fine.events),
                    "accepted_surface_coverage_generation": authority,
                },
            )

    def append_fine_completed(
        self,
        *,
        final_surface_coverage_generation: str | Path,
        final_reconstruction_product: str | Path,
    ) -> UnknownBladeExperimentEvent:
        """Seal the fine run and bind its final coverage and reconstruction outputs."""

        with self._lock:
            self._require_current_chain()
            previous = self._events[-1]
            if previous.event_type not in {"fine_started", "fine_checkpoint"}:
                raise ValueError("FINE_COMPLETED requires a preceding fine phase event")
            started = next(
                event.payload for event in self._events if event.event_type == "fine_started"
            )
            fine = read_stop_scan_run(started["fine_run_root"])
            if fine.run_id != self.experiment_id or not fine.events:
                raise ValueError("fine run identity or terminal event is unavailable")
            coverage_root = Path(final_surface_coverage_generation).resolve()
            coverage = read_surface_coverage_generation(
                coverage_root,
                require_foreground_bound_science=True,
            )
            if coverage.root.resolve() != coverage_root:
                raise ValueError("final fine coverage readback changed its root")
            reconstruction_root = Path(final_reconstruction_product).resolve()
            coverage_authority = _authority_record(coverage_root, "coverage.json")
            _verify_fine_terminal_assets(
                fine,
                coverage_root=coverage_root,
                reconstruction_root=reconstruction_root,
                science_authority=self._events_science_authority(),
            )
            if previous.event_type == "fine_checkpoint" and (
                int(previous.payload["fine_event_count"]) != len(fine.events)
                or previous.payload["fine_last_event_sha256"]
                != fine.latest_event.event_sha256
                or previous.payload["accepted_surface_coverage_generation"]
                != coverage_authority
            ):
                raise ValueError("FINE_COMPLETED must inherit the final fine checkpoint")
            return self._append(
                "fine_completed",
                {
                    "fine_run_id": fine.run_id,
                    "fine_run_root": str(fine.root),
                    "fine_last_event_sha256": fine.latest_event.event_sha256,
                    "fine_event_count": len(fine.events),
                    "final_surface_coverage_generation": coverage_authority,
                    "final_reconstruction_product": _authority_record(
                        reconstruction_root,
                        "final_reconstruction.json",
                    ),
                },
            )

    def _require_current_chain(self) -> None:
        stored = read_unknown_blade_experiment(self.root)
        if stored.experiment_id != self.experiment_id or stored.events != tuple(self._events):
            raise ValueError("experiment chain changed since this writer opened it")

    def _require_current_event_integrity(
        self,
        expected_events: tuple[UnknownBladeExperimentEvent, ...] | None = None,
    ) -> None:
        """Verify the small event head without recursively reopening its sources."""

        expected_values = tuple(self._events) if expected_events is None else expected_events
        events_root = self.root / "events"
        all_files = sorted(item for item in events_root.iterdir() if item.is_file())
        unexpected = [
            item
            for item in all_files
            if _EVENT_PATTERN.fullmatch(item.name) is None
            and _EVENT_PARTIAL_PATTERN.fullmatch(item.name) is None
        ]
        if unexpected:
            raise ValueError("experiment event directory contains an unexpected file")
        files = [item for item in all_files if _EVENT_PATTERN.fullmatch(item.name)]
        expected_names = [_event_filename(index) for index in range(len(expected_values))]
        if [item.name for item in files] != expected_names:
            raise ValueError("experiment event files changed since this writer opened it")
        decoded: list[UnknownBladeExperimentEvent] = []
        for index, (path, expected) in enumerate(
            zip(files, expected_values, strict=True)
        ):
            raw = _strict_json_loads(path.read_text(encoding="utf-8"))
            event = _decode_event_payload(raw, expected_sequence=index)
            if decoded and event.previous_event_sha256 != decoded[-1].event_sha256:
                raise ValueError("experiment event predecessor chain is broken")
            if decoded and event.experiment_id != decoded[0].experiment_id:
                raise ValueError("experiment ID changed within the event chain")
            if event != expected:
                raise ValueError("experiment event content changed since this writer opened it")
            decoded.append(event)

    def _events_science_authority(self) -> ScienceAcceptanceAuthority | None:
        payload = self._events[0].payload
        raw = payload.get("science_acceptance_authority")
        return ScienceAcceptanceAuthority.from_payload(raw) if raw is not None else None

    def _events_runtime_timing_authority(
        self,
    ) -> RuntimeTimingAcceptanceAuthority | None:
        payload = self._events[0].payload
        raw = payload.get("runtime_timing_acceptance_authority")
        return (
            RuntimeTimingAcceptanceAuthority.from_payload(raw)
            if raw is not None
            else None
        )

    def _append(
        self,
        event_type: UnknownBladeExperimentEventType,
        payload: dict[str, Any],
        *,
        before_publish: Callable[[], None] | None = None,
        incremental_coarse_checkpoint: bool = False,
    ) -> UnknownBladeExperimentEvent:
        sequence = len(self._events)
        previous = self._events[-1].event_sha256 if self._events else None
        event = UnknownBladeExperimentEvent.build(
            experiment_id=self.experiment_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            previous_event_sha256=previous,
        )
        prospective_events = (*self._events, event)

        def validate_before_publish() -> None:
            if before_publish is not None:
                before_publish()
            if incremental_coarse_checkpoint:
                self._require_current_event_integrity()
                science_authority = self._events_science_authority()
                if science_authority is not None:
                    science_authority.assert_acceptance_asset_current()
                timing_authority = self._events_runtime_timing_authority()
                if timing_authority is not None:
                    timing_authority.assert_acceptance_asset_current()
                with performance_span("experiment.checkpoint_incremental_verify"):
                    _validate_incremental_coarse_checkpoint(tuple(self._events), event)

        def validate_after_publish() -> None:
            if incremental_coarse_checkpoint:
                self._require_current_event_integrity(prospective_events)
                _verify_authority_record(
                    event.payload["coarse_generation"],
                    label="coarse checkpoint generation",
                )

        _write_new_json(
            self.root / "events" / _event_filename(sequence),
            event.to_payload(),
            before_publish=(
                validate_before_publish
                if before_publish is not None or incremental_coarse_checkpoint
                else None
            ),
            after_publish=(
                validate_after_publish
                if incremental_coarse_checkpoint
                else None
            ),
        )
        self._events.append(event)
        return event


def read_unknown_blade_experiment(path: str | Path) -> StoredUnknownBladeExperiment:
    """Recompute the complete chain and every cross-run/source authority binding."""

    root = Path(path).resolve()
    try:
        events_root = root / "events"
        files = sorted(item for item in events_root.iterdir() if item.is_file())
        if not files:
            raise ValueError("experiment chain contains no events")
        expected_names = [_event_filename(index) for index in range(len(files))]
        if [item.name for item in files] != expected_names:
            raise ValueError(
                "experiment event filenames are missing, duplicated, or non-contiguous"
            )
        events: list[UnknownBladeExperimentEvent] = []
        for index, file_path in enumerate(files):
            if _EVENT_PATTERN.fullmatch(file_path.name) is None:
                raise ValueError("experiment event filename is invalid")
            raw = _strict_json_loads(file_path.read_text(encoding="utf-8"))
            event = _decode_event_payload(raw, expected_sequence=index)
            if events and event.previous_event_sha256 != events[-1].event_sha256:
                raise ValueError("experiment event predecessor chain is broken")
            if events and event.experiment_id != events[0].experiment_id:
                raise ValueError("experiment ID changed within the event chain")
            events.append(event)
        _validate_semantic_chain(tuple(events))
        raw_authority = events[0].payload.get("science_acceptance_authority")
        science_authority = (
            ScienceAcceptanceAuthority.from_payload(raw_authority)
            if raw_authority is not None
            else None
        )
        if science_authority is not None:
            science_authority.assert_acceptance_asset_current()
        raw_timing_authority = events[0].payload.get(
            "runtime_timing_acceptance_authority"
        )
        runtime_timing_authority = (
            RuntimeTimingAcceptanceAuthority.from_payload(raw_timing_authority)
            if raw_timing_authority is not None
            else None
        )
        if runtime_timing_authority is not None:
            runtime_timing_authority.assert_acceptance_asset_current()
        return StoredUnknownBladeExperiment(
            root=root,
            experiment_id=events[0].experiment_id,
            events=tuple(events),
            science_authority=science_authority,
            runtime_timing_authority=runtime_timing_authority,
            fine_start_protocol=events[0].payload.get("fine_start_protocol"),
            placement_id=events[0].payload.get("placement_id"),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UnknownBladeExperimentFormatError(
            f"Invalid unknown-blade experiment chain {root}: {exc}"
        ) from exc


def _validate_incremental_coarse_checkpoint(
    prefix: tuple[UnknownBladeExperimentEvent, ...],
    event: UnknownBladeExperimentEvent,
) -> None:
    """Validate one new coarse checkpoint against an already verified prefix."""

    if not prefix or prefix[0].event_type != "experiment_initialized":
        raise ValueError("coarse checkpoint prefix is not initialized")
    previous = prefix[-1]
    if (
        previous.event_type not in {"experiment_initialized", "coarse_checkpoint"}
        or event.event_type != "coarse_checkpoint"
        or event.sequence != len(prefix)
        or event.previous_event_sha256 != previous.event_sha256
        or event.experiment_id != prefix[0].experiment_id
    ):
        raise ValueError("incremental COARSE_CHECKPOINT chain binding changed")
    payload = event.payload
    if set(payload) != {
        "coarse_run_id",
        "coarse_run_root",
        "coarse_last_event_sha256",
        "coarse_event_count",
        "coarse_generation",
    }:
        raise ValueError("COARSE_CHECKPOINT payload changed")
    initialized = prefix[0].payload
    coarse_root = _canonical_root(
        initialized["coarse_run_root"],
        label="coarse run root",
    )
    coarse = read_stop_scan_run(coarse_root)
    count = _checkpoint_event_count(
        payload["coarse_event_count"],
        label="coarse checkpoint",
    )
    if (
        coarse.run_id != initialized["coarse_run_id"]
        or coarse.run_id != event.experiment_id
        or payload["coarse_run_id"] != coarse.run_id
        or _canonical_root(
            payload["coarse_run_root"],
            label="checkpoint coarse run root",
        )
        != coarse.root
        or not coarse.events
        or count != len(coarse.events)
        or payload["coarse_last_event_sha256"] != coarse.latest_event.event_sha256
    ):
        raise ValueError("COARSE_CHECKPOINT run binding changed or did not advance")
    generation_record = payload["coarse_generation"]
    generation_root = _verify_authority_record(
        generation_record,
        label="coarse checkpoint generation",
    )
    generation = read_coarse_scan_generation(generation_root)
    if generation.root.resolve() != generation_root:
        raise ValueError("coarse checkpoint generation readback changed its root")
    if previous.event_type == "coarse_checkpoint":
        previous_count = _checkpoint_event_count(
            previous.payload["coarse_event_count"],
            label="previous coarse checkpoint",
        )
        if (
            count < previous_count
            or (
                count == previous_count
                and previous.payload["coarse_generation"] == generation_record
            )
        ):
            raise ValueError("COARSE_CHECKPOINT run/generation pair did not advance")


def _validate_semantic_chain(events: tuple[UnknownBladeExperimentEvent, ...]) -> None:
    if events[0].event_type != "experiment_initialized":
        raise ValueError("experiment chain must begin with INIT")
    initialized = events[0].payload
    legacy_init_fields = {"experiment_id", "coarse_run_id", "coarse_run_root"}
    authority_init_fields = legacy_init_fields | {"science_acceptance_authority"}
    combined_authority_init_fields = authority_init_fields | {
        "runtime_timing_acceptance_authority"
    }
    candidate_commit_init_fields = combined_authority_init_fields | {
        "fine_start_protocol"
    }
    accepted_init_fields = {
        frozenset(legacy_init_fields),
        frozenset(authority_init_fields),
        frozenset(combined_authority_init_fields),
        frozenset(candidate_commit_init_fields),
    }
    accepted_init_fields.update(
        frozenset((*fields, "placement_id"))
        for fields in (
            legacy_init_fields,
            authority_init_fields,
            combined_authority_init_fields,
            candidate_commit_init_fields,
        )
    )
    if frozenset(initialized) not in accepted_init_fields:
        raise ValueError("experiment initialization payload changed")
    placement_id = initialized.get("placement_id")
    if placement_id is not None:
        if not isinstance(placement_id, str):
            raise ValueError("experiment placement_id must be a string")
        validate_stop_scan_run_id(placement_id)
    fine_start_protocol = initialized.get("fine_start_protocol")
    expected_candidate_fields = candidate_commit_init_fields | (
        {"placement_id"} if placement_id is not None else set()
    )
    if fine_start_protocol is not None and (
        fine_start_protocol != UNKNOWN_BLADE_FINE_START_PROTOCOL
        or set(initialized) != expected_candidate_fields
    ):
        raise ValueError("experiment fine-start protocol changed")
    science_authority = (
        ScienceAcceptanceAuthority.from_payload(
            initialized["science_acceptance_authority"]
        )
        if "science_acceptance_authority" in initialized
        else None
    )
    if science_authority is not None:
        science_authority.assert_acceptance_asset_current()
    runtime_timing_authority = (
        RuntimeTimingAcceptanceAuthority.from_payload(
            initialized["runtime_timing_acceptance_authority"]
        )
        if "runtime_timing_acceptance_authority" in initialized
        else None
    )
    if runtime_timing_authority is not None:
        runtime_timing_authority.assert_acceptance_asset_current()
    if initialized["experiment_id"] != events[0].experiment_id:
        raise ValueError("experiment initialization ID changed")
    coarse_root = _canonical_root(initialized["coarse_run_root"], label="coarse run root")
    coarse = read_stop_scan_run(coarse_root)
    if coarse.run_id != initialized["coarse_run_id"] or coarse.run_id != events[0].experiment_id:
        raise ValueError("initialized coarse run identity changed")
    if len(events) == 1:
        return
    index = 1
    previous_coarse_count = 0
    previous_coarse_authority: dict[str, Any] | None = None
    last_coarse_checkpoint: UnknownBladeExperimentEvent | None = None
    while index < len(events) and events[index].event_type == "coarse_checkpoint":
        checkpoint = events[index]
        checkpoint_payload = checkpoint.payload
        if set(checkpoint_payload) != {
            "coarse_run_id",
            "coarse_run_root",
            "coarse_last_event_sha256",
            "coarse_event_count",
            "coarse_generation",
        }:
            raise ValueError("COARSE_CHECKPOINT payload changed")
        count = _checkpoint_event_count(
            checkpoint_payload["coarse_event_count"],
            label="coarse checkpoint",
        )
        if (
            checkpoint_payload["coarse_run_id"] != coarse.run_id
            or _canonical_root(
                checkpoint_payload["coarse_run_root"],
                label="checkpoint coarse run root",
            )
            != coarse.root
            or count > len(coarse.events)
            or checkpoint_payload["coarse_last_event_sha256"]
            != coarse.events[count - 1].event_sha256
            or count < previous_coarse_count
        ):
            raise ValueError("COARSE_CHECKPOINT run binding changed or did not advance")
        generation_record = checkpoint_payload["coarse_generation"]
        generation_root = _verify_authority_record(
            generation_record,
            label="coarse checkpoint generation",
        )
        generation = read_coarse_scan_generation(generation_root)
        if generation.root.resolve() != generation_root:
            raise ValueError("coarse checkpoint generation readback changed its root")
        if count == previous_coarse_count and previous_coarse_authority == generation_record:
            raise ValueError("COARSE_CHECKPOINT run/generation pair did not advance")
        previous_coarse_count = count
        previous_coarse_authority = dict(generation_record)
        last_coarse_checkpoint = checkpoint
        index += 1
    if index == len(events):
        return
    if events[index].event_type != "handoff_prepared" or last_coarse_checkpoint is None:
        raise ValueError("PREPARED must follow one or more COARSE_CHECKPOINT events")
    prepared = events[index]
    payload = prepared.payload
    expected_prepared_fields = {
        "coarse_checkpoint_event_sha256",
        "coarse_run_id",
        "coarse_run_root",
        "coarse_last_event_sha256",
        "coarse_event_count",
        "schema5_generation",
        "reference_coarse_model",
    }
    if runtime_timing_authority is not None:
        expected_prepared_fields.add("schema5_prepare_timing")
    if set(payload) != expected_prepared_fields:
        raise ValueError("handoff PREPARED payload changed")
    if (
        prepared.previous_event_sha256 != last_coarse_checkpoint.event_sha256
        or payload["coarse_checkpoint_event_sha256"]
        != last_coarse_checkpoint.event_sha256
        or payload["coarse_run_id"] != coarse.run_id
        or _canonical_root(payload["coarse_run_root"], label="prepared coarse run root")
        != coarse.root
        or _checkpoint_event_count(
            payload["coarse_event_count"],
            label="prepared coarse",
        )
        != len(coarse.events)
        or not coarse.events
        or payload["coarse_last_event_sha256"] != coarse.latest_event.event_sha256
    ):
        raise ValueError("handoff PREPARED coarse terminal binding changed")
    generation_root = _verify_authority_record(
        payload["schema5_generation"], label="schema-5 generation"
    )
    reference_root = _verify_authority_record(
        payload["reference_coarse_model"], label="reference coarse model"
    )
    generation = read_coarse_scan_generation(generation_root)
    if generation.root != generation_root or generation.coarse_model_path != reference_root:
        raise ValueError("handoff PREPARED schema-5/reference binding changed")
    checkpoint_generation_root = _canonical_root(
        last_coarse_checkpoint.payload["coarse_generation"]["root"],
        label="final coarse checkpoint generation",
    )
    if (
        generation.previous_generation_path is None
        or generation.previous_generation_path.resolve() != checkpoint_generation_root
    ):
        raise ValueError("schema-5 READY predecessor differs from final coarse checkpoint")
    if runtime_timing_authority is not None:
        prepare_timing = payload["schema5_prepare_timing"]
        if not isinstance(prepare_timing, Mapping) or set(prepare_timing) != {
            "actual_duration_s",
            "accepted_limit_s",
            "runtime_timing_acceptance_id",
            "runtime_timing_metadata_sha256",
        }:
            raise ValueError("PREPARED schema-5 timing payload changed")
        actual = prepare_timing["actual_duration_s"]
        limit = runtime_timing_authority.timing_limits_s[
            "maximum_schema5_handoff_duration_s"
        ]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or float(actual) < 0.0
            or float(actual) > limit
            or prepare_timing["accepted_limit_s"] != limit
            or prepare_timing["runtime_timing_acceptance_id"]
            != runtime_timing_authority.acceptance_id
            or prepare_timing["runtime_timing_metadata_sha256"]
            != runtime_timing_authority.metadata_sha256
        ):
            raise ValueError("PREPARED schema-5 timing authority changed")
    index += 1
    if index == len(events):
        return
    latest_candidate: UnknownBladeExperimentEvent | None = None
    candidate_fine_roots: set[Path] = set()
    candidate_runs: list[tuple[UnknownBladeExperimentEvent, Any]] = []
    if fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL:
        while index < len(events) and events[index].event_type == "fine_start_candidate":
            candidate = events[index]
            candidate_payload = candidate.payload
            if set(candidate_payload) != {
                "prepared_event_sha256",
                "fine_run_id",
                "fine_run_root",
                "fine_first_event_sha256",
                "fine_event_count_at_candidate",
            }:
                raise ValueError("FINE_START_CANDIDATE payload changed")
            expected_predecessor = (
                prepared.event_sha256
                if latest_candidate is None
                else latest_candidate.event_sha256
            )
            if (
                candidate.previous_event_sha256 != expected_predecessor
                or candidate_payload["prepared_event_sha256"] != prepared.event_sha256
            ):
                raise ValueError(
                    "FINE_START_CANDIDATE does not inherit the PREPARED candidate chain"
                )
            candidate_root = _canonical_root(
                candidate_payload["fine_run_root"],
                label="fine-start candidate run root",
            )
            candidate_fine = read_stop_scan_run(candidate_root)
            _validate_fine_start_bootstrap_run(
                candidate_fine,
                require_exactly_one_event=False,
            )
            if (
                candidate_root in candidate_fine_roots
                or candidate_fine.run_id != events[0].experiment_id
                or candidate_payload["fine_run_id"] != candidate_fine.run_id
                or candidate_payload["fine_first_event_sha256"]
                != candidate_fine.events[0].event_sha256
                or _checkpoint_event_count(
                    candidate_payload["fine_event_count_at_candidate"],
                    label="fine-start candidate",
                )
                != 1
            ):
                raise ValueError("FINE_START_CANDIDATE fine-run binding changed or repeated")
            candidate_fine_roots.add(candidate_root)
            candidate_runs.append((candidate, candidate_fine))
            latest_candidate = candidate
            index += 1
        if index == len(events):
            assert candidate_runs
            _validate_fine_start_bootstrap_run(
                candidate_runs[-1][1],
                require_exactly_one_event=True,
            )
            return
        if latest_candidate is None:
            raise ValueError("candidate-commit FINE_STARTED requires a candidate")
    elif events[index].event_type == "fine_start_candidate":
        raise ValueError("legacy chain cannot contain a fine-start candidate")
    if events[index].event_type != "fine_started":
        raise ValueError("FINE_STARTED must follow PREPARED or its latest candidate")
    started = events[index]
    started_payload = started.payload
    expected_started_fields = {
        "prepared_event_sha256",
        "fine_run_id",
        "fine_run_root",
        "fine_first_event_sha256",
    }
    if fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL:
        expected_started_fields.update(
            {"fine_start_candidate_event_sha256", "schema5_handoff_timing"}
        )
    elif runtime_timing_authority is not None:
        expected_started_fields.add("schema5_handoff_timing")
    if set(started_payload) != expected_started_fields:
        raise ValueError("FINE_STARTED payload changed")
    expected_started_predecessor = (
        latest_candidate.event_sha256
        if latest_candidate is not None
        else prepared.event_sha256
    )
    if (
        started.previous_event_sha256 != expected_started_predecessor
        or started_payload["prepared_event_sha256"] != prepared.event_sha256
        or (
            latest_candidate is not None
            and (
                started_payload["fine_start_candidate_event_sha256"]
                != latest_candidate.event_sha256
                or any(
                    started_payload[field] != latest_candidate.payload[field]
                    for field in (
                        "prepared_event_sha256",
                        "fine_run_id",
                        "fine_run_root",
                        "fine_first_event_sha256",
                    )
                )
            )
        )
    ):
        raise ValueError("FINE_STARTED does not commit the latest fine-start candidate")
    fine_root = _canonical_root(started_payload["fine_run_root"], label="fine run root")
    fine = read_stop_scan_run(fine_root)
    if fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL:
        _validate_fine_start_bootstrap_run(
            fine,
            require_exactly_one_event=False,
        )
    if (
        fine.run_id != events[0].experiment_id
        or started_payload["fine_run_id"] != fine.run_id
        or not fine.events
        or started_payload["fine_first_event_sha256"] != fine.events[0].event_sha256
    ):
        raise ValueError("FINE_STARTED fine-run root or first-event binding changed")
    if runtime_timing_authority is not None:
        timing = started_payload["schema5_handoff_timing"]
        if not isinstance(timing, Mapping):
            raise ValueError("FINE_STARTED handoff timing payload must be an object")
        if fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL:
            expected_timing_fields = {
                "prepublication_check_duration_s",
                "prepublication_duration_semantics",
                "accepted_limit_s",
                "runtime_timing_acceptance_id",
                "runtime_timing_metadata_sha256",
                "measurement_scope",
                "publication_deadline_contract",
            }
            duration = timing.get("prepublication_check_duration_s")
        else:
            # Historical single-event FINE_STARTED chains remain audit-readable,
            # but their pre-publication sample was named ``actual_duration_s``.
            expected_timing_fields = {
                "actual_duration_s",
                "accepted_limit_s",
                "runtime_timing_acceptance_id",
                "runtime_timing_metadata_sha256",
                "measurement_scope",
            }
            duration = timing.get("actual_duration_s")
        if set(timing) != expected_timing_fields:
            raise ValueError("FINE_STARTED handoff timing payload changed")
        limit = runtime_timing_authority.timing_limits_s[
            "maximum_schema5_handoff_duration_s"
        ]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
            or float(duration) > limit
            or timing["accepted_limit_s"] != limit
            or timing["runtime_timing_acceptance_id"]
            != runtime_timing_authority.acceptance_id
            or timing["runtime_timing_metadata_sha256"]
            != runtime_timing_authority.metadata_sha256
            or timing["measurement_scope"]
            not in {"uninterrupted_total", "resume_fine_start"}
            or (
                fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL
                and timing["publication_deadline_contract"]
                != _FINE_START_PUBLICATION_DEADLINE_CONTRACT
            )
            or (
                fine_start_protocol == UNKNOWN_BLADE_FINE_START_PROTOCOL
                and timing["prepublication_duration_semantics"]
                != _FINE_START_PREPUBLICATION_DURATION_SEMANTICS
            )
        ):
            raise ValueError("FINE_STARTED handoff timing authority changed")
    index += 1
    if index == len(events):
        return
    previous_fine_count = 0
    previous_fine_authority: dict[str, Any] | None = None
    last_fine_checkpoint: UnknownBladeExperimentEvent | None = None
    while index < len(events) and events[index].event_type == "fine_checkpoint":
        checkpoint = events[index]
        checkpoint_payload = checkpoint.payload
        if set(checkpoint_payload) != {
            "fine_run_id",
            "fine_run_root",
            "fine_last_event_sha256",
            "fine_event_count",
            "accepted_surface_coverage_generation",
        }:
            raise ValueError("FINE_CHECKPOINT payload changed")
        count = _checkpoint_event_count(
            checkpoint_payload["fine_event_count"],
            label="fine checkpoint",
        )
        if (
            checkpoint_payload["fine_run_id"] != fine.run_id
            or _canonical_root(
                checkpoint_payload["fine_run_root"],
                label="checkpoint fine run root",
            )
            != fine.root
            or count > len(fine.events)
            or checkpoint_payload["fine_last_event_sha256"]
            != fine.events[count - 1].event_sha256
            or count < previous_fine_count
        ):
            raise ValueError("FINE_CHECKPOINT run binding changed or did not advance")
        coverage_record = checkpoint_payload["accepted_surface_coverage_generation"]
        coverage_root = _verify_authority_record(
            coverage_record,
            label="fine checkpoint coverage generation",
        )
        coverage = read_surface_coverage_generation(
            coverage_root,
            require_foreground_bound_science=True,
        )
        if coverage.root.resolve() != coverage_root:
            raise ValueError("fine checkpoint coverage readback changed its root")
        if count == previous_fine_count and previous_fine_authority == coverage_record:
            raise ValueError("FINE_CHECKPOINT run/coverage pair did not advance")
        previous_fine_count = count
        previous_fine_authority = dict(coverage_record)
        last_fine_checkpoint = checkpoint
        index += 1
    if index == len(events):
        return
    if events[index].event_type != "fine_completed" or index != len(events) - 1:
        raise ValueError("experiment phase events are repeated or out of order")
    completed = events[index]
    completed_payload = completed.payload
    if set(completed_payload) != {
        "fine_run_id",
        "fine_run_root",
        "fine_last_event_sha256",
        "fine_event_count",
        "final_surface_coverage_generation",
        "final_reconstruction_product",
    }:
        raise ValueError("FINE_COMPLETED payload changed")
    if (
        completed_payload["fine_run_id"] != fine.run_id
        or _canonical_root(
            completed_payload["fine_run_root"],
            label="completed fine run root",
        )
        != fine.root
        or _checkpoint_event_count(
            completed_payload["fine_event_count"],
            label="completed fine",
        )
        != len(fine.events)
        or completed_payload["fine_last_event_sha256"] != fine.latest_event.event_sha256
        or (
            last_fine_checkpoint is not None
            and (
                completed.previous_event_sha256 != last_fine_checkpoint.event_sha256
                or _checkpoint_event_count(
                    completed_payload["fine_event_count"],
                    label="completed fine",
                )
                != previous_fine_count
                or completed_payload["final_surface_coverage_generation"]
                != previous_fine_authority
            )
        )
    ):
        raise ValueError("FINE_COMPLETED fine terminal binding changed")
    coverage_root = _verify_authority_record(
        completed_payload["final_surface_coverage_generation"],
        label="final surface-coverage generation",
    )
    coverage = read_surface_coverage_generation(
        coverage_root,
        require_foreground_bound_science=True,
    )
    if coverage.root.resolve() != coverage_root:
        raise ValueError("final surface-coverage readback changed its root")
    reconstruction_root = _verify_authority_record(
        completed_payload["final_reconstruction_product"],
        label="final reconstruction product",
    )
    _verify_fine_terminal_assets(
        fine,
        coverage_root=coverage_root,
        reconstruction_root=reconstruction_root,
        science_authority=science_authority,
    )


def _checkpoint_event_count(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} event count must be a positive integer")
    return value


__all__ = [
    "StoredUnknownBladeExperiment",
    "UNKNOWN_BLADE_EXPERIMENT_SCHEMA_VERSION",
    "UNKNOWN_BLADE_FINE_START_PROTOCOL",
    "UnknownBladeExperimentEvent",
    "UnknownBladeExperimentEventType",
    "UnknownBladeExperimentFormatError",
    "UnknownBladeExperimentWriter",
    "read_unknown_blade_experiment",
]
