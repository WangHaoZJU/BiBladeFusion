"""Serial FoundationStereo stop-and-capture coordination for one short motion leg.

This module deliberately exposes no CLI and no raw robot command.  It composes the
existing full semantic occupancy reader, one-leg preflight, guarded executor, and
stationarity gate into a receding-horizon state machine.  The production collision
checkers currently lack both continuous sweep proofs, so the same code reaches
``MOTION_BLOCKED`` before any driver call until those independent proofs exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    MotionPreflightConfig,
    OccupancyConfig,
    RobotConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.devices.robot.base import RobotState, RobotStateSource
from biblade_fusion.devices.robot.streaming import StreamServoJResult
from biblade_fusion.mapping import OccupancyMapState, OccupancySnapshot
from biblade_fusion.planning import CandidateStatus, EvaluatedCandidate
from biblade_fusion.robotics import (
    Cs68PinocchioCollisionChecker,
    GuardedEliteExecutor,
    JointMotionPreflight,
    MotionExecutionPermit,
    OccupancyRobotCollisionChecker,
    load_es68_flange_t_tcp,
)
from biblade_fusion.robotics.guarded_execution import GuardedArm
from biblade_fusion.robotics.stationarity import (
    StationarityEvidence,
    validate_stationary_trace,
    wait_until_settled,
)
from biblade_fusion.storage.inference_stationarity import (
    read_inference_stationarity,
)
from biblade_fusion.storage.occupancy_mapping import (
    StoredOccupancyMapping,
    read_occupancy_mapping,
)
from biblade_fusion.workflows.motion_preflight import preflight_live_joint_segment

_SHA256 = re.compile(r"[0-9a-f]{64}")


class StopScanError(RuntimeError):
    """The coordinator contract or a state transition is invalid."""


class StopScanBlocked(StopScanError):
    """A fail-closed gate prevented the next motion segment."""


class NextViewUnavailable(StopScanBlocked):
    """Coverage is incomplete, but no endpoint-feasible next view exists."""


class BladePlanningAssetError(StopScanError):
    """The scientific reconstruction/coverage evidence is invalid or inconsistent."""


class StopScanAbortRequested(StopScanError):
    """An asynchronous operator stop request interrupted the active transaction."""


class StopScanPhase(StrEnum):
    IDLE = "idle"
    BOOTSTRAP_MAP_REQUIRED = "bootstrap_map_required"
    WAITING_SETTLED = "waiting_settled"
    CAPTURING = "capturing"
    INFERRING = "inferring"
    PUBLISHING_MAP = "publishing_map"
    MAP_READY = "map_ready"
    PLANNING = "planning"
    PREFLIGHTING = "preflighting"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTING = "executing"
    SETTLING = "settling"
    AWAITING_CAPTURE = "awaiting_capture"
    MOTION_BLOCKED = "motion_blocked"
    COMPLETE = "complete"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OccupancyBinding:
    """Complete identity of the motion-visible immutable occupancy generation."""

    frame_id: str
    sequence: int
    content_hash: str
    mapping_context_hash: str
    quality_evidence_hash: str
    robot_geometry_hash: str
    occupancy_metadata_sha256: str
    semantic_verifier_contract_hash: str
    semantic_attestation_hash: str

    def __post_init__(self) -> None:
        if self.frame_id != "base" or self.sequence < 0:
            raise ValueError("Occupancy binding must identify a base-frame generation")
        for name in (
            "content_hash",
            "mapping_context_hash",
            "quality_evidence_hash",
            "robot_geometry_hash",
            "occupancy_metadata_sha256",
            "semantic_verifier_contract_hash",
            "semantic_attestation_hash",
        ):
            value = getattr(self, name)
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"Occupancy binding {name} must be a SHA-256 digest")

    @classmethod
    def from_mapping(cls, mapping: StoredOccupancyMapping) -> OccupancyBinding:
        if type(mapping) is not StoredOccupancyMapping:
            raise ValueError("Coordinator requires a full StoredOccupancyMapping")
        if (
            mapping.motion_eligible is not True
            or mapping.verification_status
            != "full_semantic_verified_for_motion_preflight"
        ):
            raise ValueError("Occupancy mapping lacks full semantic motion verification")
        snapshot = mapping.snapshot
        attestation = mapping.semantic_attestation
        attestation.assert_matches(
            snapshot,
            robot_geometry_hash=attestation.robot_geometry_hash,
        )
        return cls(
            frame_id=snapshot.frame_id,
            sequence=snapshot.sequence,
            content_hash=snapshot.content_hash,
            mapping_context_hash=str(snapshot.mapping_context_hash),
            quality_evidence_hash=str(snapshot.quality_evidence_hash),
            robot_geometry_hash=attestation.robot_geometry_hash,
            occupancy_metadata_sha256=attestation.occupancy_metadata_sha256,
            semantic_verifier_contract_hash=(
                attestation.semantic_verifier_contract_hash
            ),
            semantic_attestation_hash=attestation.attestation_hash,
        )

    @property
    def tuple(self) -> tuple[object, ...]:
        return (
            self.frame_id,
            self.sequence,
            self.content_hash,
            self.mapping_context_hash,
            self.quality_evidence_hash,
            self.robot_geometry_hash,
            self.occupancy_metadata_sha256,
            self.semantic_verifier_contract_hash,
            self.semantic_attestation_hash,
        )


@dataclass(frozen=True, slots=True)
class OccupancyGeneration:
    generation_id: str
    artifact_path: Path
    mapping: StoredOccupancyMapping
    binding: OccupancyBinding
    inference_stationarity_path: Path
    inference_stationarity_sha256: str

    @classmethod
    def verified(
        cls,
        artifact_path: str | Path,
        mapping: StoredOccupancyMapping,
        *,
        inference_stationarity_path: str | Path,
        inference_stationarity_sha256: str,
    ) -> OccupancyGeneration:
        path = Path(artifact_path).resolve()
        stationarity_path = Path(inference_stationarity_path).resolve()
        binding = OccupancyBinding.from_mapping(mapping)
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Verified occupancy artifact has no metadata.json")
        if _file_sha256(metadata_path) != binding.occupancy_metadata_sha256:
            raise ValueError(
                "Occupancy artifact path differs from its semantic attestation"
            )
        if (
            _SHA256.fullmatch(inference_stationarity_sha256) is None
            or not stationarity_path.is_file()
            or _file_sha256(stationarity_path) != inference_stationarity_sha256
        ):
            raise ValueError(
                "Occupancy generation lacks its exact inference-stationarity asset"
            )
        stored_stationarity = read_inference_stationarity(stationarity_path)
        if stored_stationarity.file_sha256 != inference_stationarity_sha256:
            raise ValueError(
                "Inference-stationarity semantic reader returned another asset"
            )
        payload = json.dumps(
            {
                "artifact_path": str(path),
                "binding": binding.tuple,
                "inference_stationarity_path": str(stationarity_path),
                "inference_stationarity_sha256": inference_stationarity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return cls(
            hashlib.sha256(payload).hexdigest(),
            path,
            mapping,
            binding,
            stationarity_path,
            inference_stationarity_sha256,
        )

    @property
    def snapshot(self) -> OccupancySnapshot:
        return self.mapping.snapshot


class OccupancyGenerationPublisher:
    """Atomically publish a generation and forbid replacement while it is frozen."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: OccupancyGeneration | None = None
        self._frozen_generation_id: str | None = None

    @property
    def current(self) -> OccupancyGeneration:
        with self._lock:
            if self._current is None:
                raise StopScanBlocked("No verified occupancy generation is published")
            return self._current

    def current_if_available(self) -> OccupancyGeneration | None:
        """Return an accepted generation without ever waiting on a commit.

        This is intentionally only for status reporting.  Safety decisions must use
        :attr:`current`, whose blocking lock provides the transaction boundary.
        """

        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._current
        finally:
            self._lock.release()

    def current_snapshot(self) -> OccupancySnapshot:
        return self.current.snapshot

    def publish(
        self,
        generation: OccupancyGeneration,
    ) -> None:
        """Replace the accepted generation in one publisher operation.

        This low-level operation is useful for importing an already accepted map.
        Coordinator perception cycles use :meth:`publish_after_acceptance` so a
        candidate can never become visible before its source window is committed.
        """

        with self._lock:
            if self._frozen_generation_id is not None:
                raise StopScanBlocked("Occupancy generation is frozen for execution")
            self._current = generation

    def publish_after_acceptance(
        self,
        generation: OccupancyGeneration,
        accept: Callable[[], None],
    ) -> None:
        """Accept a source transaction before exposing its matching generation.

        The same lock guards ``current``, ``freeze`` and direct ``publish``.  While
        ``accept`` runs, concurrent consumers therefore keep seeing neither a staged
        candidate nor a half-committed generation: their read blocks until acceptance
        succeeds and publication completes.  If acceptance raises, ``_current`` is
        unchanged and no rollback is necessary.
        """

        with self._lock:
            if self._frozen_generation_id is not None:
                raise StopScanBlocked(
                    "Occupancy generation is frozen for execution"
                )
            accept()
            self._current = generation

    @contextmanager
    def freeze(
        self,
        *,
        expected_generation_id: str,
        expected_binding: OccupancyBinding,
        expected_inference_stationarity_sha256: str,
    ):
        with self._lock:
            current = self.current
            if (
                current.generation_id != expected_generation_id
                or current.binding != expected_binding
                or current.inference_stationarity_sha256
                != expected_inference_stationarity_sha256
            ):
                raise StopScanBlocked(
                    "Published occupancy/perception generation changed before freeze"
                )
            if self._frozen_generation_id is not None:
                raise StopScanBlocked("Occupancy generation is already frozen")
            self._frozen_generation_id = current.generation_id
        try:
            yield current
            with self._lock:
                if self.current.generation_id != current.generation_id:
                    raise StopScanBlocked("Occupancy generation changed while frozen")
        finally:
            with self._lock:
                self._frozen_generation_id = None


@dataclass(frozen=True, slots=True)
class CapturedStopScanView:
    """One closed, immutable single-view raw session produced while stopped."""

    bundle: SynchronizedFrameBundle
    raw_session_path: Path
    cycle_root: Path
    captured_at_utc: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_session_path", Path(self.raw_session_path).resolve())
        object.__setattr__(self, "cycle_root", Path(self.cycle_root).resolve())
        if self.captured_at_utc.tzinfo is None:
            raise ValueError("Captured view UTC timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "captured_at_utc",
            self.captured_at_utc.astimezone(UTC),
        )


@dataclass(frozen=True, slots=True)
class PerceptionCycleResult:
    """Assets produced by one stopped, FoundationStereo-only perception transaction."""

    bundle: SynchronizedFrameBundle
    raw_session_path: Path
    stereo_inference_path: Path
    occupancy_mapping_path: Path
    stored_occupancy: StoredOccupancyMapping
    stationarity_reference: RobotState
    inference_robot_state_trace: tuple[RobotState, ...]
    inference_stationarity: StationarityEvidence
    inference_stationarity_path: Path
    inference_stationarity_sha256: str
    depth_backend: Literal["foundation_stereo"] = "foundation_stereo"
    reconstructed_view_path: Path | None = None
    coverage_path: Path | None = None

    def __post_init__(self) -> None:
        if self.depth_backend != "foundation_stereo":
            raise ValueError("Stop-scan perception accepts only FoundationStereo")
        for name in (
            "raw_session_path",
            "stereo_inference_path",
            "occupancy_mapping_path",
            "inference_stationarity_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        for name in ("reconstructed_view_path", "coverage_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        if not self.inference_robot_state_trace:
            raise ValueError("FoundationStereo inference requires a stationarity trace")
        if type(self.stationarity_reference) is not RobotState:
            raise ValueError("FoundationStereo result requires a typed trace reference")
        if type(self.inference_stationarity) is not StationarityEvidence:
            raise ValueError("FoundationStereo result requires typed stationarity evidence")
        if (
            _SHA256.fullmatch(self.inference_stationarity_sha256) is None
            or not self.inference_stationarity_path.is_file()
            or _file_sha256(self.inference_stationarity_path)
            != self.inference_stationarity_sha256
        ):
            raise ValueError("FoundationStereo stationarity asset hash is invalid")


class FoundationStereoPerceptionEngine(Protocol):
    @property
    def robot_state_source(self) -> RobotStateSource: ...

    @property
    def acquisition_config(self) -> AcquisitionConfig: ...

    @property
    def occupancy_config(self) -> OccupancyConfig: ...

    @property
    def coordinator_config(self) -> StopAndCaptureConfig: ...

    def capture(
        self,
        view_id: str,
        sequence_index: int,
    ) -> CapturedStopScanView: ...

    def infer_and_update(
        self,
        captured: CapturedStopScanView,
    ) -> PerceptionCycleResult: ...

    def commit_perception_cycle(
        self,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
    ) -> None: ...

    def cancel_pending_capture(
        self,
        captured: CapturedStopScanView | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NextViewTarget:
    view_id: str
    joint_positions_rad: tuple[float, float, float, float, float, float]
    base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.view_id.strip():
            raise ValueError("Next view ID must be non-empty")
        joints = _joint_vector(self.joint_positions_rad, label="next-view joints")
        pose = PoseSE3("base", "tcp", self.base_t_tcp_matrix)
        object.__setattr__(self, "joint_positions_rad", joints)
        object.__setattr__(
            self,
            "base_t_tcp_matrix",
            tuple(tuple(float(value) for value in row) for row in pose.matrix),
        )


@dataclass(frozen=True, slots=True)
class NextViewSelection:
    """Auditable selector decision bound to one semantic surface generation."""

    target: NextViewTarget | None
    surface_generation_id: str
    reference_model_sha256: str
    selection_policy_sha256: str
    required_patch_count: int
    incomplete_patch_count: int
    coverage_complete: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "surface_generation_id",
            "reference_model_sha256",
            "selection_policy_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"Next-view decision {name} must be a SHA-256 digest")
        if self.required_patch_count <= 0:
            raise ValueError("Next-view decision requires at least one reference patch")
        if not 0 <= self.incomplete_patch_count <= self.required_patch_count:
            raise ValueError("Next-view incomplete-patch count is invalid")
        if self.coverage_complete:
            if self.target is not None or self.incomplete_patch_count != 0:
                raise ValueError(
                    "Completed next-view decision cannot contain a target or gaps"
                )
        elif self.target is None or self.incomplete_patch_count == 0:
            raise ValueError(
                "Incomplete next-view decision requires a concrete target and gaps"
            )
        diagnostics = tuple(str(value).strip() for value in self.diagnostics)
        if any(not value for value in diagnostics):
            raise ValueError("Next-view diagnostics must be non-empty strings")
        object.__setattr__(self, "diagnostics", diagnostics)


def next_view_target_from_candidate(
    candidate: EvaluatedCandidate,
    hand_eye: HandEyeCalibration,
) -> NextViewTarget:
    """Convert one endpoint-feasible camera candidate into an ES68 TCP target."""

    if (
        candidate.status is not CandidateStatus.ENDPOINT_FEASIBLE
        or candidate.joint_positions_rad is None
    ):
        raise ValueError("Next view must carry an endpoint-feasible IK solution")
    flange_t_left_ir = hand_eye.require_flange_primary()
    camera_pose = candidate.candidate.base_t_left_ir
    canonical_camera_pose = type(camera_pose)("base", "left_ir", camera_pose.matrix)
    base_t_flange = canonical_camera_pose.compose(flange_t_left_ir.inverse())
    base_t_tcp = base_t_flange.compose(load_es68_flange_t_tcp())
    return NextViewTarget(
        candidate.candidate.view_id,
        tuple(float(value) for value in candidate.joint_positions_rad),
        tuple(tuple(float(value) for value in row) for row in base_t_tcp.matrix),
    )


class NextViewSelector(Protocol):
    def select_next(
        self,
        observation: PerceptionCycleResult,
        generation: OccupancyGeneration,
    ) -> NextViewSelection: ...


@dataclass(frozen=True, slots=True)
class SegmentProposal:
    proposal_id: str
    target_view_id: str
    capture_view_id: str
    start_joint_positions_rad: tuple[float, float, float, float, float, float]
    goal_joint_positions_rad: tuple[float, float, float, float, float, float]
    final_target_joint_positions_rad: tuple[float, float, float, float, float, float]
    target_base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...]
    final_target: bool
    occupancy_binding: OccupancyBinding
    occupancy_generation_id: str
    inference_stationarity_sha256: str
    surface_generation_id: str
    reference_model_sha256: str
    selection_policy_sha256: str

    def __post_init__(self) -> None:
        if not self.target_view_id.strip() or not self.capture_view_id.strip():
            raise ValueError("Segment proposal view IDs must be non-empty")
        for name in (
            "proposal_id",
            "occupancy_generation_id",
            "inference_stationarity_sha256",
            "surface_generation_id",
            "reference_model_sha256",
            "selection_policy_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"Segment proposal {name} must be a SHA-256 digest")
        object.__setattr__(
            self,
            "start_joint_positions_rad",
            _joint_vector(self.start_joint_positions_rad, label="segment start"),
        )
        object.__setattr__(
            self,
            "goal_joint_positions_rad",
            _joint_vector(self.goal_joint_positions_rad, label="segment goal"),
        )
        object.__setattr__(
            self,
            "final_target_joint_positions_rad",
            _joint_vector(
                self.final_target_joint_positions_rad,
                label="segment final target",
            ),
        )
        pose = PoseSE3("base", "tcp", self.target_base_t_tcp_matrix)
        object.__setattr__(
            self,
            "target_base_t_tcp_matrix",
            tuple(tuple(float(value) for value in row) for row in pose.matrix),
        )

    @property
    def motion_authorized(self) -> bool:
        return False


class ApprovedSegmentExecutor(Protocol):
    def approval_prompt(self, preflight: JointMotionPreflight) -> str: ...

    def authorize(
        self,
        preflight: JointMotionPreflight,
        *,
        operator_id: str,
        confirmation: str,
    ) -> MotionExecutionPermit: ...

    def execute(
        self,
        preflight: JointMotionPreflight,
        permit: MotionExecutionPermit,
        *,
        cancellation_requested: Callable[[], bool] = lambda: False,
    ) -> StreamServoJResult: ...


@dataclass(frozen=True, slots=True)
class PreparedSegment:
    """Detached preparation summary; it intentionally exposes no executor."""

    proposal: SegmentProposal
    preflight: JointMotionPreflight

    @property
    def ready_for_approval(self) -> bool:
        return self.preflight.ready_for_approval

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _PreparedSegmentExecution:
    proposal: SegmentProposal
    preflight: JointMotionPreflight
    executor: ApprovedSegmentExecutor | None

    @property
    def ready_for_approval(self) -> bool:
        return self.preflight.ready_for_approval and self.executor is not None

    @property
    def public_summary(self) -> PreparedSegment:
        # JointMotionPreflight contains diagnostic dictionaries.  A deep copy keeps
        # caller-side inspection or annotation from mutating the authoritative
        # evidence retained for approval and execution.
        return PreparedSegment(deepcopy(self.proposal), deepcopy(self.preflight))


class SegmentSafetyFactory(Protocol):
    @property
    def motion_robot(self) -> GuardedArm: ...

    @property
    def occupancy_publisher(self) -> OccupancyGenerationPublisher: ...

    @property
    def occupancy_config(self) -> OccupancyConfig: ...

    @property
    def coordinator_config(self) -> StopAndCaptureConfig: ...

    @property
    def motion_config(self) -> MotionPreflightConfig: ...

    def prepare(
        self,
        proposal: SegmentProposal,
        generation: OccupancyGeneration,
    ) -> _PreparedSegmentExecution: ...


class StopController(Protocol):
    def stop(self) -> None: ...


class CoordinatedRobot(RobotStateSource, StopController, Protocol):
    """One physical robot instance used for state, stop, and guarded motion."""

    @property
    def robot_config(self) -> RobotConfig: ...


class RunEventSink(Protocol):
    def append_event(
        self,
        *,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> object: ...


class GuardedSegmentSafetyFactory:
    """Build one map-bound preflight and its same-checker guarded executor."""

    def __init__(
        self,
        arm: GuardedArm,
        collision_checker: Cs68PinocchioCollisionChecker,
        publisher: OccupancyGenerationPublisher,
        motion_config: MotionPreflightConfig,
        occupancy_config: OccupancyConfig,
        coordinator_config: StopAndCaptureConfig,
        *,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._arm = arm
        self._collision_checker = collision_checker
        self._publisher = publisher
        self._motion_config = motion_config.model_copy(deep=True)
        self._occupancy_config = occupancy_config.model_copy(deep=True)
        self._coordinator_config = coordinator_config.model_copy(deep=True)
        self._utc_clock = utc_clock

    @property
    def motion_robot(self) -> GuardedArm:
        return self._arm

    @property
    def occupancy_publisher(self) -> OccupancyGenerationPublisher:
        return self._publisher

    @property
    def occupancy_config(self) -> OccupancyConfig:
        return self._occupancy_config.model_copy(deep=True)

    @property
    def coordinator_config(self) -> StopAndCaptureConfig:
        return self._coordinator_config.model_copy(deep=True)

    @property
    def motion_config(self) -> MotionPreflightConfig:
        return self._motion_config.model_copy(deep=True)

    def prepare(
        self,
        proposal: SegmentProposal,
        generation: OccupancyGeneration,
    ) -> _PreparedSegmentExecution:
        if self._publisher.current.generation_id != generation.generation_id:
            raise StopScanBlocked("Cannot preflight a non-current occupancy generation")
        if proposal.occupancy_binding != generation.binding:
            raise StopScanBlocked("Segment proposal occupancy binding changed")
        if (
            proposal.occupancy_generation_id != generation.generation_id
            or proposal.inference_stationarity_sha256
            != generation.inference_stationarity_sha256
        ):
            raise StopScanBlocked("Segment proposal perception evidence changed")
        occupancy_checker = OccupancyRobotCollisionChecker(
            self._collision_checker,
            self._publisher.current_snapshot,
            maximum_map_age_s=self._occupancy_config.maximum_map_age_s,
            additional_clearance_m=(
                self._occupancy_config.obstacle_inflation_m
                + self._collision_checker.minimum_clearance_m
            ),
            semantic_attestation=generation.mapping.semantic_attestation,
            utc_clock=self._utc_clock,
        )
        live = preflight_live_joint_segment(
            proposal.start_joint_positions_rad,
            proposal.goal_joint_positions_rad,
            self._motion_config,
            collision_checker=self._collision_checker,
            occupancy_checker=occupancy_checker,
            final_target=proposal.final_target,
            target_base_t_tcp_matrix=(
                proposal.target_base_t_tcp_matrix if proposal.final_target else None
            ),
            execution_freshness_margin_s=(
                self._coordinator_config.execution_freshness_margin_s
            ),
            evaluated_at_utc=self._utc_clock(),
        )
        bound_preflight = replace(
            live.preflight,
            diagnostics={
                **live.preflight.diagnostics,
                "stop_scan_occupancy_generation_id": generation.generation_id,
                "inference_stationarity_sha256": (
                    generation.inference_stationarity_sha256
                ),
                "surface_generation_id": proposal.surface_generation_id,
                "reference_model_sha256": proposal.reference_model_sha256,
                "selection_policy_sha256": proposal.selection_policy_sha256,
            },
        )
        executor: ApprovedSegmentExecutor | None = None
        if live.ready_for_approval:
            executor = GuardedEliteExecutor(
                self._arm,
                self._collision_checker,
                occupancy_checker,
                execution_freshness_margin_s=(
                    self._coordinator_config.execution_freshness_margin_s
                ),
            )
        return _PreparedSegmentExecution(proposal, bound_preflight, executor)


@dataclass(frozen=True, slots=True)
class StopScanCheckpoint:
    phase: StopScanPhase
    cycle_index: int
    current_view_id: str | None
    proposed_view_id: str | None
    occupancy_binding: OccupancyBinding | None
    expected_capture_view_id: str | None
    blocking_reasons: tuple[str, ...]
    stop_requested: bool = False
    motion_authorized: bool = False


@dataclass(slots=True)
class _OperationFinalizer:
    """Marks an operation whose commit decision was linearized with stop."""

    linearized: bool = False


class StopScanCoordinator:
    """Synchronous stop/infer/map/one-leg/stop coordinator.

    Public methods are serialized by a non-reentrant operation lock.  A perception
    callback can therefore never overlap a motion callback in this process.
    """

    def __init__(
        self,
        *,
        config: StopAndCaptureConfig,
        acquisition_config: AcquisitionConfig,
        robot_config: RobotConfig,
        motion_config: MotionPreflightConfig,
        occupancy_config: OccupancyConfig,
        robot: CoordinatedRobot,
        perception: FoundationStereoPerceptionEngine,
        selector: NextViewSelector,
        safety_factory: SegmentSafetyFactory,
        publisher: OccupancyGenerationPublisher,
        event_sink: RunEventSink | None = None,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config.model_copy(deep=True)
        self._acquisition_config = acquisition_config.model_copy(deep=True)
        self._robot_config = robot_config.model_copy(deep=True)
        self._motion_config = motion_config.model_copy(deep=True)
        self._occupancy_config = occupancy_config.model_copy(deep=True)
        if self._robot_config.model != "es68":
            raise ValueError("Stop-scan coordination requires robot.model='es68'")
        if not self._robot_config.motion_enabled:
            raise ValueError(
                "Stop-scan coordination requires robot.motion_enabled=true"
            )
        if not math.isclose(
            self._robot_config.servoj_time_s,
            self._motion_config.servoj_dt_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Robot ServoJ time and motion-preflight ServoJ dt must be identical"
            )
        if robot.robot_config != self._robot_config:
            raise ValueError(
                "Robot driver and coordination robot policies must be identical"
            )
        if perception.robot_state_source is not robot:
            raise ValueError(
                "Perception sampling and coordination must share one robot instance"
            )
        if safety_factory.motion_robot is not robot:
            raise ValueError(
                "Guarded motion and coordination must share one robot instance"
            )
        if safety_factory.occupancy_publisher is not publisher:
            raise ValueError(
                "Preflight, execution, and coordination must share one map publisher"
            )
        if (
            perception.acquisition_config != self._acquisition_config
            or perception.occupancy_config != self._occupancy_config
            or perception.coordinator_config != self._config
        ):
            raise ValueError(
                "Perception and coordination safety policies must be identical"
            )
        if (
            safety_factory.motion_config != self._motion_config
            or safety_factory.occupancy_config != self._occupancy_config
            or safety_factory.coordinator_config != self._config
        ):
            raise ValueError(
                "Preflight and coordination safety policies must be identical"
            )
        self._robot = robot
        self._perception = perception
        self._selector = selector
        self._safety_factory = safety_factory
        self._publisher = publisher
        self._event_sink = event_sink
        self._utc_clock = utc_clock
        self._operation_lock = threading.Lock()
        # Reentrant because an event sink may synchronously surface another stop
        # while the terminal abort event is being recorded at the linearization point.
        self._stop_request_lock = threading.RLock()
        self._stop_reason_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._stop_request_reason: str | None = None
        self._phase = StopScanPhase.IDLE
        self._cycle_index = 0
        self._observation: PerceptionCycleResult | None = None
        self._observation_generation_id: str | None = None
        self._prepared: _PreparedSegmentExecution | None = None
        self._expected_capture_view_id: str | None = None
        self._run_reference_model_sha256: str | None = None
        self._run_selection_policy_sha256: str | None = None
        self._blocking_reasons: tuple[str, ...] = ()
        self._event_store_failure_reason: str | None = None

    @property
    def checkpoint(self) -> StopScanCheckpoint:
        # Status/UI reads must never delay an operator stop behind an in-progress
        # source-window commit.  A temporarily unavailable binding is reported as
        # None; all safety paths use the blocking, authoritative ``current`` API.
        generation = self._publisher.current_if_available()
        return StopScanCheckpoint(
            phase=self._phase,
            cycle_index=self._cycle_index,
            current_view_id=(
                self._observation.bundle.view_id if self._observation is not None else None
            ),
            proposed_view_id=(
                self._prepared.proposal.target_view_id
                if self._prepared is not None
                else None
            ),
            occupancy_binding=(generation.binding if generation is not None else None),
            expected_capture_view_id=self._expected_capture_view_id,
            blocking_reasons=self._blocking_reasons,
            stop_requested=self._stop_requested.is_set(),
        )

    def start(self) -> StopScanCheckpoint:
        with self._exclusive_operation():
            if self._phase is not StopScanPhase.IDLE:
                raise StopScanError("Stop-scan run has already started")
            if not self._config.enabled:
                raise StopScanBlocked("Stop-and-capture coordinator is disabled")
            if not self._occupancy_config.enabled:
                raise StopScanBlocked("Safety occupancy mapping is disabled")
            if self._config.maximum_segment_joint_delta_rad is None:
                raise StopScanBlocked("Short-segment joint bound is not configured")
            if self._robot_config.settle_time_s <= 0.0:
                raise StopScanBlocked("A positive robot settle_time_s is required")
            if self._config.settle_timeout_s < (
                self._robot_config.settle_time_s
                + self._config.settle_poll_period_s
            ):
                raise StopScanBlocked(
                    "Settle timeout cannot prove the configured stationary window"
                )
            with self._stop_request_lock, self._stop_reason_lock:
                self._stop_requested.clear()
                self._stop_request_reason = None
            self._transition(
                StopScanPhase.BOOTSTRAP_MAP_REQUIRED,
                "run_started",
                {
                    "depth_backend": "foundation_stereo",
                    "bootstrap_mode": self._config.bootstrap_mode,
                    "minimum_source_views": self._occupancy_config.minimum_source_views,
                },
            )
            return self.checkpoint

    def capture_infer_update(self, view_id: str | None = None) -> PerceptionCycleResult:
        with self._exclusive_operation() as operation:
            self._raise_if_stop_requested()
            allowed = {
                StopScanPhase.BOOTSTRAP_MAP_REQUIRED,
                StopScanPhase.AWAITING_CAPTURE,
                StopScanPhase.MOTION_BLOCKED,
            }
            if self._phase not in allowed:
                raise StopScanError(f"Cannot capture from phase {self._phase.value}")
            expected = self._expected_capture_view_id
            if expected is not None:
                selected_view_id = expected if view_id is None else view_id.strip()
                if selected_view_id != expected:
                    raise StopScanBlocked(
                        f"Expected post-segment capture {expected!r}, got {selected_view_id!r}"
                    )
            else:
                selected_view_id = str(view_id or "").strip()
                if not selected_view_id:
                    raise StopScanError("Operator-guided bootstrap capture needs a view ID")
            self._prepared = None
            self._blocking_reasons = ()
            captured: CapturedStopScanView | None = None
            try:
                # Terminate any previously running external-control program before
                # trying to prove a stationary acquisition window.  This also makes
                # bootstrap capture obey the same stop boundary as post-motion views.
                self._robot.stop()
                self._raise_if_stop_requested()
                self._transition(
                    StopScanPhase.WAITING_SETTLED,
                    "capture_stop_asserted_wait_settled",
                    {},
                )
                settled = self._wait_until_settled(None)
                self._raise_if_stop_requested()
                self._transition(
                    StopScanPhase.CAPTURING,
                    "capture_started",
                    {
                        "view_id": selected_view_id,
                        "stationarity": _stationarity_payload(settled),
                    },
                )
                captured = self._perception.capture(
                    selected_view_id,
                    self._cycle_index,
                )
                self._raise_if_stop_requested()
                self._validate_captured_view(selected_view_id, captured)
                self._transition(
                    StopScanPhase.INFERRING,
                    "foundation_stereo_started",
                    {"view_id": selected_view_id},
                )
                result = self._perception.infer_and_update(captured)
                self._raise_if_stop_requested()
                authoritative_mapping = self._validate_perception_result(
                    captured,
                    result,
                )
                result = replace(result, stored_occupancy=authoritative_mapping)
                self._transition(
                    StopScanPhase.PUBLISHING_MAP,
                    "foundation_stereo_completed",
                    {
                        "view_id": selected_view_id,
                        "inference_stationarity_samples": len(
                            result.inference_robot_state_trace
                        ),
                        "raw_session": str(result.raw_session_path),
                        "stereo_inference": str(result.stereo_inference_path),
                        "occupancy_mapping": str(result.occupancy_mapping_path),
                        "inference_stationarity": str(
                            result.inference_stationarity_path
                        ),
                        "inference_stationarity_sha256": (
                            result.inference_stationarity_sha256
                        ),
                    },
                )
                generation = OccupancyGeneration.verified(
                    result.occupancy_mapping_path,
                    result.stored_occupancy,
                    inference_stationarity_path=(
                        result.inference_stationarity_path
                    ),
                    inference_stationarity_sha256=(
                        result.inference_stationarity_sha256
                    ),
                )
                self._raise_if_stop_requested()
                snapshot = generation.snapshot
                if snapshot.map_state is OccupancyMapState.MAPPING:
                    next_phase = StopScanPhase.BOOTSTRAP_MAP_REQUIRED
                    next_event = "bootstrap_map_incomplete"
                    next_blocking_reasons = (
                        f"bootstrap_requires_{self._occupancy_config.minimum_source_views}_"
                        f"independent_views:current={len(snapshot.source_view_ids)}",
                    )
                    next_payload: Mapping[str, Any] = {
                        "source_view_count": len(snapshot.source_view_ids),
                        "generation_id": generation.generation_id,
                    }
                elif (
                    snapshot.map_state is OccupancyMapState.MAP_READY
                    and snapshot.is_usable_for_preflight(
                        self._aware_utc_now(),
                        self._occupancy_config.maximum_map_age_s,
                    )
                ):
                    next_phase = StopScanPhase.MAP_READY
                    next_event = "map_ready"
                    next_blocking_reasons = ()
                    next_payload = {
                        "generation_id": generation.generation_id,
                        "binding": list(generation.binding.tuple),
                        "inference_stationarity_sha256": (
                            generation.inference_stationarity_sha256
                        ),
                    }
                else:
                    next_phase = StopScanPhase.MOTION_BLOCKED
                    next_event = "map_not_motion_eligible"
                    next_blocking_reasons = (
                        f"occupancy_not_fresh_map_ready:{snapshot.map_state.value}",
                    )
                    next_payload = {"reason": next_blocking_reasons[0]}
                self._finalize_operation(
                    operation,
                    lambda: self._commit_perception_transaction(
                        captured,
                        result,
                        generation,
                    ),
                )
                self._raise_if_stop_requested()
                self._observation = result
                self._expected_capture_view_id = None
                self._cycle_index += 1
                self._blocking_reasons = next_blocking_reasons
                self._transition(next_phase, next_event, next_payload)
            except StopScanAbortRequested as exc:
                with suppress(BaseException):
                    self._perception.cancel_pending_capture(captured)
                self._observation = None
                self._observation_generation_id = None
                self._blocking_reasons = (str(exc),)
                self._transition(
                    StopScanPhase.ABORTED,
                    "operator_stop_observed",
                    {"reason": str(exc)},
                )
                raise
            except BaseException as exc:
                with suppress(BaseException):
                    self._perception.cancel_pending_capture(captured)
                self._observation = None
                self._observation_generation_id = None
                self._blocking_reasons = (
                    f"perception_cycle_failed:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.FAILED,
                    "perception_cycle_failed",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            return result

    def prepare_next_segment(self) -> PreparedSegment | None:
        with self._exclusive_operation():
            self._raise_if_stop_requested()
            if self._phase is not StopScanPhase.MAP_READY:
                raise StopScanError(f"Cannot plan from phase {self._phase.value}")
            if self._observation is None:
                raise StopScanError("No current perception observation")
            generation = self._publisher.current
            if generation.generation_id != self._observation_generation_id:
                raise StopScanBlocked(
                    "Published occupancy no longer matches the current observation"
                )
            self._transition(StopScanPhase.PLANNING, "next_view_selection_started", {})
            try:
                selection = self._selector.select_next(self._observation, generation)
                if type(selection) is not NextViewSelection:
                    raise BladePlanningAssetError(
                        "Next-view selector returned an untyped decision"
                    )
                self._validate_selection_run_binding(selection)
                self._raise_if_stop_requested()
                if selection.coverage_complete:
                    self._transition(
                        StopScanPhase.COMPLETE,
                        "coverage_complete",
                        {
                            "surface_generation_id": (
                                selection.surface_generation_id
                            ),
                            "reference_model_sha256": (
                                selection.reference_model_sha256
                            ),
                            "selection_policy_sha256": (
                                selection.selection_policy_sha256
                            ),
                            "required_patch_count": (
                                selection.required_patch_count
                            ),
                            "incomplete_patch_count": 0,
                            "diagnostics": list(selection.diagnostics),
                        },
                    )
                    return None
                target = selection.target
                if target is None:  # guarded again at the trust boundary
                    raise BladePlanningAssetError(
                        "Incomplete selector decision has no target"
                    )
                live_state = self._robot.read_state()
                self._raise_if_stop_requested()
                proposal = self._propose_short_segment(
                    target,
                    selection,
                    live_state,
                    generation,
                )
                self._transition(
                    StopScanPhase.PREFLIGHTING,
                    "single_segment_preflight_started",
                    {
                        "proposal_id": proposal.proposal_id,
                        "target_view_id": proposal.target_view_id,
                        "final_target": proposal.final_target,
                        "surface_generation_id": proposal.surface_generation_id,
                        "reference_model_sha256": proposal.reference_model_sha256,
                        "selection_policy_sha256": proposal.selection_policy_sha256,
                    },
                )
                prepared = self._safety_factory.prepare(proposal, generation)
                self._raise_if_stop_requested()
                self._validate_prepared_segment(prepared, generation)
            except StopScanAbortRequested as exc:
                self._prepared = None
                self._blocking_reasons = (str(exc),)
                self._transition(
                    StopScanPhase.ABORTED,
                    "operator_stop_observed",
                    {"reason": str(exc)},
                )
                raise
            except NextViewUnavailable as exc:
                self._prepared = None
                self._blocking_reasons = (
                    f"next_view_unavailable:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.MOTION_BLOCKED,
                    "next_view_unavailable",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            except BladePlanningAssetError as exc:
                self._prepared = None
                self._blocking_reasons = (
                    f"blade_planning_asset_failed:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.FAILED,
                    "blade_planning_asset_failed",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            except Exception as exc:
                self._prepared = None
                self._blocking_reasons = (
                    f"single_segment_preflight_failed:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.MOTION_BLOCKED,
                    "single_segment_preflight_failed",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            except BaseException as exc:
                self._prepared = None
                self._blocking_reasons = (
                    f"single_segment_preflight_interrupted:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.ABORTED,
                    "single_segment_preflight_interrupted",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            self._prepared = prepared
            if not prepared.ready_for_approval:
                reasons = prepared.preflight.blocking_reasons or (
                    "preflight_not_ready_for_approval",
                )
                self._blocking_reasons = tuple(reasons)
                self._transition(
                    StopScanPhase.MOTION_BLOCKED,
                    "single_segment_blocked",
                    {"blocking_reasons": list(self._blocking_reasons)},
                )
                return prepared.public_summary
            self._transition(
                StopScanPhase.WAITING_APPROVAL,
                "single_segment_waiting_approval",
                {
                    "proposal_id": proposal.proposal_id,
                    "approval_prompt": prepared.executor.approval_prompt(
                        prepared.preflight
                    ),
                },
            )
            return prepared.public_summary

    def approval_prompt(self) -> str:
        if (
            self._phase is not StopScanPhase.WAITING_APPROVAL
            or self._prepared is None
            or self._prepared.executor is None
        ):
            raise StopScanError("No approval-eligible single segment exists")
        return self._prepared.executor.approval_prompt(self._prepared.preflight)

    def execute_approved(
        self,
        *,
        operator_id: str,
        confirmation: str,
    ) -> StreamServoJResult:
        with self._exclusive_operation():
            prepared = self._prepared
            if (
                self._phase is not StopScanPhase.WAITING_APPROVAL
                or prepared is None
                or prepared.executor is None
            ):
                raise StopScanError("No approval-eligible single segment exists")
            try:
                self._raise_if_stop_requested()
                with self._publisher.freeze(
                    expected_generation_id=(
                        prepared.proposal.occupancy_generation_id
                    ),
                    expected_binding=prepared.proposal.occupancy_binding,
                    expected_inference_stationarity_sha256=(
                        prepared.proposal.inference_stationarity_sha256
                    ),
                ):
                    permit = prepared.executor.authorize(
                        prepared.preflight,
                        operator_id=operator_id,
                        confirmation=confirmation,
                    )
                    self._raise_if_stop_requested()
                    self._transition(
                        StopScanPhase.EXECUTING,
                        "single_segment_executing",
                        {"proposal_id": prepared.proposal.proposal_id},
                    )
                    result = prepared.executor.execute(
                        prepared.preflight,
                        permit,
                        cancellation_requested=self._stop_requested.is_set,
                    )
                    self._raise_if_stop_requested()
                    self._transition(
                        StopScanPhase.SETTLING,
                        "single_segment_wait_settled",
                        {"proposal_id": prepared.proposal.proposal_id},
                    )
                    settled = self._wait_until_settled(
                        prepared.proposal.goal_joint_positions_rad
                    )
                    self._raise_if_stop_requested()
            except BaseException as exc:
                with suppress(BaseException):
                    self._robot.stop()
                self._prepared = None
                self._blocking_reasons = (
                    f"single_segment_execution_failed:{type(exc).__name__}:{exc}",
                )
                self._transition(
                    StopScanPhase.ABORTED,
                    "single_segment_execution_failed",
                    {"reason": self._blocking_reasons[0]},
                )
                raise
            self._expected_capture_view_id = prepared.proposal.capture_view_id
            self._prepared = None
            self._blocking_reasons = ()
            self._transition(
                StopScanPhase.AWAITING_CAPTURE,
                "single_segment_complete",
                {
                    "next_capture_view_id": self._expected_capture_view_id,
                    "stationarity": _stationarity_payload(settled),
                    "commands_sent": result.commands_sent,
                },
            )
            return result

    def abort(self, reason: str) -> StopScanCheckpoint:
        return self.request_stop(reason)

    def request_stop(self, reason: str) -> StopScanCheckpoint:
        """Issue an immediate best-effort stop without waiting for the transaction lock.

        This is the software interruption path for a UI/operator thread.  It first
        latches an abort request, then calls the driver's stop boundary even while the
        main coordinator thread owns ``_operation_lock`` during ServoJ.  The active
        transaction observes the latch and records ``ABORTED`` before releasing its
        lock; if no transaction is active, this method records the state itself.
        """

        text = reason.strip()
        if not text:
            raise ValueError("Stop-request reason must be non-empty")
        if self._phase in {
            StopScanPhase.IDLE,
            StopScanPhase.COMPLETE,
            StopScanPhase.ABORTED,
            StopScanPhase.FAILED,
        }:
            raise StopScanError(
                f"Cannot request an active-run stop from phase {self._phase.value}"
            )
        with self._stop_reason_lock:
            if self._stop_request_reason is None:
                self._stop_request_reason = text
        # The abort latch and physical stop never wait for the transaction
        # linearization lock.  A blocked publisher/asset commit therefore cannot
        # delay the operator's stop command.
        self._stop_requested.set()
        stop_error: BaseException | None = None
        try:
            self._robot.stop()
        except BaseException as exc:
            stop_error = exc
        if self._operation_lock.acquire(blocking=False):
            try:
                with self._stop_request_lock:
                    self._record_requested_abort_if_needed()
            finally:
                self._operation_lock.release()
        if stop_error is not None:
            raise stop_error
        return self.checkpoint

    def _wait_until_settled(
        self,
        goal_joint_positions_rad: Sequence[float] | None,
    ) -> StationarityEvidence:
        return wait_until_settled(
            self._robot,
            goal_joint_positions_rad,
            settle_time_s=self._robot_config.settle_time_s,
            timeout_s=self._config.settle_timeout_s,
            poll_period_s=self._config.settle_poll_period_s,
            max_joint_delta_rad=self._acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                self._acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                self._acquisition_config.max_tcp_rotation_delta_rad
            ),
            goal_tolerance_rad=self._config.maximum_goal_joint_error_rad,
            maximum_robot_state_staleness_s=(
                self._config.maximum_robot_state_staleness_s
            ),
        )

    def _validate_captured_view(
        self,
        expected_view_id: str,
        captured: CapturedStopScanView,
    ) -> None:
        if type(captured) is not CapturedStopScanView:
            raise StopScanBlocked("Perception engine returned an untyped capture")
        if (
            captured.bundle.view_id != expected_view_id
            or captured.bundle.sequence_index != self._cycle_index
        ):
            raise StopScanBlocked("Captured view identity or sequence changed")
        self._verify_closed_single_view_session(
            captured.raw_session_path,
            expected_view_id,
            self._cycle_index,
        )

    def _validate_perception_result(
        self,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
    ) -> StoredOccupancyMapping:
        if type(result) is not PerceptionCycleResult:
            raise StopScanBlocked("Perception engine returned an untyped result")
        if result.depth_backend != "foundation_stereo":
            raise StopScanBlocked("Native depth is forbidden in stop-scan coordination")
        if result.bundle is not captured.bundle:
            raise StopScanBlocked("Inference did not consume the exact captured bundle")
        if result.raw_session_path != captured.raw_session_path:
            raise StopScanBlocked("Inference raw-session binding changed")
        cycle_root = captured.cycle_root.resolve()
        for label, path in (
            ("stereo inference", result.stereo_inference_path),
            ("occupancy mapping", result.occupancy_mapping_path),
            ("stationarity evidence", result.inference_stationarity_path),
        ):
            if not path.resolve().is_relative_to(cycle_root):
                raise StopScanBlocked(f"{label} escaped the immutable cycle root")
        reconstructed_path = result.reconstructed_view_path
        if reconstructed_path is not None and (
            not reconstructed_path.is_dir()
            or not reconstructed_path.resolve().is_relative_to(cycle_root)
        ):
            raise StopScanBlocked(
                "reconstructed blade view is missing or escaped the immutable cycle root"
            )
        coverage_path = result.coverage_path
        if coverage_path is not None:
            if not coverage_path.is_dir():
                raise StopScanBlocked("surface coverage generation is missing")
            coverage_is_local = coverage_path.resolve().is_relative_to(cycle_root)
            # A transit capture may carry the latest independently verified immutable
            # generation from an earlier cycle.  A cycle that did produce a new science
            # reconstruction must publish its matching successor transaction locally.
            # The concrete selector pins the exact carried path/generation before any
            # subsequent completion decision or motion proposal.
            if reconstructed_path is not None and not coverage_is_local:
                raise StopScanBlocked(
                    "new surface coverage escaped the immutable cycle root"
                )
            if not coverage_is_local and (
                self._expected_capture_view_id is None
                or not self._expected_capture_view_id.startswith("transit_")
            ):
                raise StopScanBlocked(
                    "external surface coverage is allowed only for an expected transit capture"
                )

        recomputed = validate_stationary_trace(
            result.stationarity_reference,
            result.inference_robot_state_trace,
            max_joint_delta_rad=self._acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                self._acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                self._acquisition_config.max_tcp_rotation_delta_rad
            ),
            maximum_robot_state_staleness_s=(
                self._config.maximum_robot_state_staleness_s
            ),
        )
        expected_thresholds = (
            float(self._acquisition_config.max_joint_delta_rad),
            float(self._acquisition_config.max_tcp_translation_delta_m),
            float(self._acquisition_config.max_tcp_rotation_delta_rad),
            float(self._config.maximum_robot_state_staleness_s),
        )
        try:
            stored = read_inference_stationarity(
                result.inference_stationarity_path
            )
        except ValueError as exc:
            raise StopScanBlocked(
                "Inference-stationarity evidence failed semantic verification"
            ) from exc
        expected_manifest = (captured.raw_session_path / "manifest.json").resolve()
        if (
            stored.path != result.inference_stationarity_path
            or stored.file_sha256 != result.inference_stationarity_sha256
            or stored.view_id != captured.bundle.view_id
            or stored.sequence_index != captured.bundle.sequence_index
            or stored.source_session_manifest_path != expected_manifest
            or stored.source_session_manifest_sha256 != _file_sha256(expected_manifest)
            or stored.thresholds != expected_thresholds
        ):
            raise StopScanBlocked(
                "Inference-stationarity asset identity, source, or thresholds changed"
            )
        if (
            not _robot_states_equal(stored.reference, result.stationarity_reference)
            or not _robot_state_traces_equal(
                stored.trace,
                result.inference_robot_state_trace,
            )
            or _stationarity_payload(stored.evidence)
            != _stationarity_payload(result.inference_stationarity)
            or _stationarity_payload(stored.evidence)
            != _stationarity_payload(recomputed)
        ):
            raise StopScanBlocked(
                "Inference-stationarity asset does not match the returned transaction"
            )
        full_trace = (stored.reference, *stored.trace)
        for label, state in (
            ("capture-before", captured.bundle.robot_state_before),
            ("capture-selected", captured.bundle.selected_robot_state),
            ("capture-after", captured.bundle.robot_state_after),
        ):
            if not any(_robot_states_equal(state, sample) for sample in full_trace):
                raise StopScanBlocked(
                    f"Stationarity evidence does not cover the {label} robot state"
                )
        try:
            authoritative_mapping = read_occupancy_mapping(
                result.occupancy_mapping_path
            )
        except (OSError, TypeError, ValueError) as exc:
            raise StopScanBlocked(
                "Occupancy asset failed independent semantic readback"
            ) from exc
        authoritative_binding = OccupancyBinding.from_mapping(
            authoritative_mapping
        )
        try:
            returned_binding = OccupancyBinding.from_mapping(
                result.stored_occupancy
            )
        except (TypeError, ValueError) as exc:
            raise StopScanBlocked(
                "Perception returned invalid in-memory occupancy evidence"
            ) from exc
        if returned_binding != authoritative_binding:
            raise StopScanBlocked(
                "Returned occupancy differs from its independently re-read asset"
            )
        self._validate_current_mapping_source(
            captured,
            result,
            authoritative_mapping,
        )
        return authoritative_mapping

    @staticmethod
    def _validate_current_mapping_source(
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
        mapping: StoredOccupancyMapping,
    ) -> None:
        """Require the current capture to be the last source of the fresh map."""

        if not mapping.frame_evidence:
            raise StopScanBlocked("Occupancy asset has no frame evidence")
        evidence = mapping.frame_evidence[-1]
        if (
            evidence.source_view_id != captured.bundle.view_id
            or evidence.source_sequence_index != captured.bundle.sequence_index
            or evidence.frame_number != captured.bundle.stereo.frame_number
            or evidence.source_session_manifest_sha256
            != _file_sha256(captured.raw_session_path / "manifest.json")
            or evidence.source_stereo_metadata_sha256
            != _file_sha256(result.stereo_inference_path / "metadata.json")
        ):
            raise StopScanBlocked(
                "Current capture is not the final frame of the occupancy asset"
            )
        try:
            frames = mapping.metadata["frames"]
            final_sources = frames[-1]["sources"]
            session_root = Path(final_sources["session"]["root"]).resolve()
            stereo_root = Path(
                final_sources["stereo_inference"]["root"]
            ).resolve()
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise StopScanBlocked(
                "Occupancy asset final source metadata is invalid"
            ) from exc
        if (
            not isinstance(frames, list)
            or len(frames) != len(mapping.frame_evidence)
            or session_root != captured.raw_session_path
            or stereo_root != result.stereo_inference_path
        ):
            raise StopScanBlocked(
                "Occupancy final source paths do not match the current transaction"
            )

    @staticmethod
    def _verify_closed_single_view_session(
        session_path: Path,
        expected_view_id: str,
        expected_sequence_index: int,
    ) -> None:
        manifest_path = session_path / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StopScanBlocked("Raw single-view session manifest is unreadable") from exc
        views = payload.get("views")
        closed_at = payload.get("closed_at_utc")
        try:
            closed = datetime.fromisoformat(str(closed_at))
        except ValueError as exc:
            raise StopScanBlocked("Raw session has no valid close timestamp") from exc
        if (
            payload.get("status") != "completed"
            or not isinstance(views, list)
            or closed.tzinfo is None
        ):
            raise StopScanBlocked("Raw session must be closed with completed status")
        if len(views) != 1:
            raise StopScanBlocked("Each stop-scan cycle requires one immutable raw session")
        view = views[0]
        if not isinstance(view, dict) or (
            view.get("view_id"), view.get("sequence_index")
        ) != (expected_view_id, expected_sequence_index):
            raise StopScanBlocked("Raw single-view session identity mismatch")

    def _propose_short_segment(
        self,
        target: NextViewTarget,
        selection: NextViewSelection,
        live_state: RobotState,
        generation: OccupancyGeneration,
    ) -> SegmentProposal:
        if selection.coverage_complete or selection.target != target:
            raise BladePlanningAssetError(
                "Segment target does not match the incomplete selector decision"
            )
        start = np.asarray(live_state.joint_positions_rad, dtype=np.float64)
        final = np.asarray(target.joint_positions_rad, dtype=np.float64)
        delta = final - start
        maximum_delta = float(np.max(np.abs(delta)))
        bound = self._config.maximum_segment_joint_delta_rad
        if bound is None or not math.isfinite(bound) or bound <= 0.0:
            raise StopScanBlocked("Short-segment joint bound is unavailable")
        scale = 1.0 if maximum_delta <= 1e-12 else min(1.0, bound / maximum_delta)
        goal = start + scale * delta
        final_target = bool(scale >= 1.0 - 1e-12)
        capture_view_id = (
            target.view_id
            if final_target
            else f"transit_{target.view_id}_cycle_{self._cycle_index:04d}"
        )
        payload = {
            "target_view_id": target.view_id,
            "capture_view_id": capture_view_id,
            "start_joint_positions_rad": start.tolist(),
            "goal_joint_positions_rad": goal.tolist(),
            "final_target_joint_positions_rad": final.tolist(),
            "final_target": final_target,
            "occupancy_binding": generation.binding.tuple,
            "occupancy_generation_id": generation.generation_id,
            "inference_stationarity_sha256": (
                generation.inference_stationarity_sha256
            ),
            "surface_generation_id": selection.surface_generation_id,
            "reference_model_sha256": selection.reference_model_sha256,
            "selection_policy_sha256": selection.selection_policy_sha256,
        }
        proposal_id = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        return SegmentProposal(
            proposal_id,
            target.view_id,
            capture_view_id,
            _joint_vector(start, label="live segment start"),
            _joint_vector(goal, label="live segment goal"),
            target.joint_positions_rad,
            target.base_t_tcp_matrix,
            final_target,
            generation.binding,
            generation.generation_id,
            generation.inference_stationarity_sha256,
            selection.surface_generation_id,
            selection.reference_model_sha256,
            selection.selection_policy_sha256,
        )

    def _validate_selection_run_binding(
        self,
        selection: NextViewSelection,
    ) -> None:
        if self._run_reference_model_sha256 is None:
            self._run_reference_model_sha256 = selection.reference_model_sha256
            self._run_selection_policy_sha256 = selection.selection_policy_sha256
            return
        if (
            selection.reference_model_sha256
            != self._run_reference_model_sha256
            or selection.selection_policy_sha256
            != self._run_selection_policy_sha256
        ):
            raise BladePlanningAssetError(
                "Next-view reference or selection policy changed within one run"
            )

    @staticmethod
    def _validate_prepared_segment(
        prepared: _PreparedSegmentExecution,
        generation: OccupancyGeneration,
    ) -> None:
        proposal = prepared.proposal
        preflight = prepared.preflight
        if proposal.occupancy_binding != generation.binding:
            raise StopScanBlocked("Prepared segment belongs to another occupancy map")
        if (
            proposal.occupancy_generation_id != generation.generation_id
            or proposal.inference_stationarity_sha256
            != generation.inference_stationarity_sha256
        ):
            raise StopScanBlocked("Prepared segment perception evidence changed")
        if not np.array_equal(
            np.asarray(preflight.start_joint_positions_rad),
            np.asarray(proposal.start_joint_positions_rad),
        ) or not np.array_equal(
            np.asarray(preflight.goal_joint_positions_rad),
            np.asarray(proposal.goal_joint_positions_rad),
        ):
            raise StopScanBlocked("Preflight start/goal differ from the short proposal")
        report = preflight.occupancy
        if report is None or report.evidence is None:
            if preflight.ready_for_approval:
                raise StopScanBlocked("Approval-eligible preflight lacks occupancy evidence")
            return
        if report.evidence.binding != generation.binding.tuple:
            raise StopScanBlocked("Preflight occupancy evidence differs from generation")
        if prepared.ready_for_approval and not preflight.ready_for_approval:
            raise StopScanBlocked("Prepared segment readiness is inconsistent")
        if prepared.ready_for_approval:
            diagnostics = preflight.diagnostics
            if (
                diagnostics.get("stop_scan_occupancy_generation_id")
                != generation.generation_id
                or diagnostics.get("inference_stationarity_sha256")
                != generation.inference_stationarity_sha256
                or diagnostics.get("surface_generation_id")
                != proposal.surface_generation_id
                or diagnostics.get("reference_model_sha256")
                != proposal.reference_model_sha256
                or diagnostics.get("selection_policy_sha256")
                != proposal.selection_policy_sha256
            ):
                raise StopScanBlocked(
                    "Approval-eligible preflight lacks perception/selection binding"
                )

    def _transition(
        self,
        phase: StopScanPhase,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self._event_store_failure_reason is not None:
            # This latch is intentionally irreversible for the lifetime of one
            # coordinator.  Business-level catch blocks must not convert an
            # unaudited terminal failure back into a capturable/executable phase.
            self._phase = StopScanPhase.FAILED
            self._prepared = None
            self._blocking_reasons = (self._event_store_failure_reason,)
            raise StopScanError(
                "Run-event persistence has already failed; coordinator is terminal"
            )
        if self._event_sink is not None:
            try:
                self._event_sink.append_event(
                    phase=phase.value,
                    cycle_index=self._cycle_index,
                    event_type=event_type,
                    payload=payload,
                )
            except BaseException as exc:
                # Never publish an unaudited operational phase.  The persistent
                # latch prevents outer exception handlers from recovering this
                # coordinator into MOTION_BLOCKED/MAP_READY/WAITING_APPROVAL.
                self._event_store_failure_reason = (
                    "run_event_persistence_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
                self._phase = StopScanPhase.FAILED
                self._prepared = None
                self._blocking_reasons = (self._event_store_failure_reason,)
                raise StopScanError(
                    "Run-event persistence failed; coordinator is terminal"
                ) from exc
        self._phase = phase

    def _commit_perception_transaction(
        self,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
        generation: OccupancyGeneration,
    ) -> None:
        """Publish and commit one prepared cycle at the stop-latch linearization."""

        self._publisher.publish_after_acceptance(
            generation,
            lambda: self._perception.commit_perception_cycle(captured, result),
        )
        self._observation_generation_id = generation.generation_id

    def _finalize_operation(
        self,
        operation: _OperationFinalizer,
        commit: Callable[[], None],
    ) -> None:
        """Linearize success against asynchronous stop and release the operation."""

        if operation.linearized:
            raise StopScanError("Operation was already linearized")
        with self._stop_request_lock:
            if self._stop_requested.is_set():
                raise StopScanAbortRequested(
                    self._stop_reason()
                )
            # This short decision is the transaction's linearization point.  The
            # potentially blocking disk/lock commit runs after releasing the stop
            # lock, so request_stop can always latch and call robot.stop promptly.
            operation.linearized = True
        commit()

    @contextmanager
    def _exclusive_operation(self):
        if not self._operation_lock.acquire(blocking=False):
            raise StopScanBlocked(
                "Another perception or motion transaction is already running"
            )
        operation = _OperationFinalizer()
        raised = False
        stop_observed = False
        stop_reason = "operator_stop_requested"
        try:
            yield operation
        except BaseException:
            raised = True
            raise
        finally:
            # The operation lock is released while the stop-request lock remains
            # held.  Requests latched before a capture commit's explicit
            # linearization cancel its pending source; requests after that point
            # still abort the run, but the already accepted digital asset remains.
            with self._stop_request_lock:
                stop_observed = self._stop_requested.is_set()
                stop_reason = self._stop_reason(default=stop_reason)
                try:
                    self._record_requested_abort_if_needed()
                finally:
                    self._operation_lock.release()
            if stop_observed and not raised:
                raise StopScanAbortRequested(stop_reason)

    def _raise_if_stop_requested(self) -> None:
        if not self._stop_requested.is_set():
            return
        raise StopScanAbortRequested(self._stop_reason())

    def _record_requested_abort_if_needed(self) -> None:
        if not self._stop_requested.is_set() or self._phase in {
            StopScanPhase.COMPLETE,
            StopScanPhase.ABORTED,
            StopScanPhase.FAILED,
        }:
            return
        reason = self._stop_reason()
        self._prepared = None
        self._blocking_reasons = (reason,)
        self._transition(
            StopScanPhase.ABORTED,
            "operator_stop_observed",
            {"reason": reason},
        )

    def _stop_reason(self, *, default: str = "operator_stop_requested") -> str:
        with self._stop_reason_lock:
            return self._stop_request_reason or default

    def _aware_utc_now(self) -> datetime:
        value = self._utc_clock()
        if value.tzinfo is None:
            raise StopScanError("Coordinator UTC clock must be timezone-aware")
        return value.astimezone(UTC)


def _joint_vector(
    values: Sequence[float] | np.ndarray,
    *,
    label: str,
) -> tuple[float, float, float, float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite ES68 six-vector")
    return tuple(float(value) for value in vector)  # type: ignore[return-value]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stationarity_payload(evidence: StationarityEvidence) -> dict[str, int | float]:
    return {
        "sample_count": evidence.sample_count,
        "duration_s": evidence.duration_s,
        "controller_duration_s": evidence.controller_duration_s,
        "max_sample_gap_s": evidence.max_sample_gap_s,
        "max_joint_delta_rad": evidence.max_joint_delta_rad,
        "max_tcp_translation_delta_m": evidence.max_tcp_translation_delta_m,
        "max_tcp_rotation_delta_rad": evidence.max_tcp_rotation_delta_rad,
        "goal_error_rad": evidence.goal_error_rad,
    }


def _robot_states_equal(left: RobotState, right: RobotState) -> bool:
    return (
        left.monotonic_time_ns == right.monotonic_time_ns
        and left.controller_time_s == right.controller_time_s
        and np.array_equal(left.joint_positions_rad, right.joint_positions_rad)
        and left.base_t_tcp.parent_frame == right.base_t_tcp.parent_frame
        and left.base_t_tcp.child_frame == right.base_t_tcp.child_frame
        and np.array_equal(left.base_t_tcp.matrix, right.base_t_tcp.matrix)
        and left.robot_mode == right.robot_mode
        and left.safety_status == right.safety_status
        and left.speed_scaling == right.speed_scaling
    )


def _robot_state_traces_equal(
    left: Sequence[RobotState],
    right: Sequence[RobotState],
) -> bool:
    return len(left) == len(right) and all(
        _robot_states_equal(left_state, right_state)
        for left_state, right_state in zip(left, right, strict=True)
    )
