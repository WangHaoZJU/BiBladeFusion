"""Production composition for one operator-supervised unknown-blade scan.

The runtime keeps the two different authorities deliberately separate:

* an operator triggers the initial stopped observations while space is UNKNOWN;
* after a verified MAP_READY generation, every short segment still requires the
  exact preflight fingerprint printed by the guarded executor.

There is no unattended-motion API in this module.  The interactive console is a
thin adapter over these typed gates and the live viewer receives read-only copied
evidence only.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import Literal, Protocol
from uuid import uuid4

from biblade_fusion.acquisition import SynchronizedAcquirer
from biblade_fusion.calibration import load_cs68_kinematics, load_hand_eye_calibration
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.devices.depth_camera import RealSenseD435i
from biblade_fusion.devices.robot import EliteArm
from biblade_fusion.devices.robot.elite_rtsi_sampler import EliteRtsiProcessSampler
from biblade_fusion.devices.thermal_camera import NullThermalCamera
from biblade_fusion.diagnostics.performance_timing import (
    activate_performance_timing,
    performance_span,
    try_create_performance_timing,
)
from biblade_fusion.diagnostics.supervised_scan import run_supervised_scan_readiness
from biblade_fusion.diagnostics.types import CheckLevel, CheckResult
from biblade_fusion.mapping import Es68D435iRobotDepthRenderer, OccupancyMapState
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapSeed,
)
from biblade_fusion.perception.stereo import FoundationStereoBackend
from biblade_fusion.planning import BladeSide, EliteCs68IkChecker
from biblade_fusion.robotics import (
    BootstrapSafeStateEvidence,
    Cs68PinocchioCollisionChecker,
    Es68D435iCollisionResources,
    Es68KinematicModel,
    wait_until_bootstrap_safe_state,
    wait_until_settled,
)
from biblade_fusion.storage.coarse_scan import read_coarse_scan_generation
from biblade_fusion.storage.fine_reconstruction import (
    replay_final_fine_reconstruction,
)
from biblade_fusion.storage.motion_envelope_acceptance import (
    StoredMotionEnvelopeAcceptance,
    motion_control_contract_for_settings,
    read_motion_envelope_acceptance,
)
from biblade_fusion.storage.runtime_timing_acceptance import (
    RuntimeTimingAcceptanceAuthority,
    load_runtime_timing_acceptance_authority,
    read_runtime_timing_acceptance,
)
from biblade_fusion.storage.science_acceptance import (
    read_science_acceptance,
    required_science_test_envelope_for_settings,
    science_runtime_contract_for_settings,
)
from biblade_fusion.storage.science_authority import (
    ScienceAcceptanceAuthority,
    load_science_acceptance_authority,
)
from biblade_fusion.storage.stop_scan_run import (
    read_stop_scan_run,
    validate_stop_scan_run_id,
)
from biblade_fusion.storage.unknown_blade_experiment import (
    UNKNOWN_BLADE_FINE_START_PROTOCOL,
    UnknownBladeExperimentWriter,
    read_unknown_blade_experiment,
)
from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentStatusSnapshot,
)
from biblade_fusion.supervision.live import (
    LiveCollisionGeometry,
    LiveSupervisionBridge,
    LiveSupervisionLayout,
)
from biblade_fusion.workflows.blade_next_view import BladeCoverageNextViewSelector
from biblade_fusion.workflows.foundation_stereo_cycle import (
    FoundationStereoOccupancyCycleEngine,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    GuardedSegmentSafetyFactory,
    NextViewSelection,
    OccupancyBinding,
    OccupancyGeneration,
    OccupancyGenerationPublisher,
    PerceptionCycleResult,
)
from biblade_fusion.workflows.supervised_experiment import (
    GuardedCoordinatorMotionExecutor,
    OperatorApproval,
    RecoveryConfirmation,
    SupervisedExperimentRunner,
)
from biblade_fusion.workflows.unknown_blade_coarse import (
    BootstrapSeedProvider,
    CoarsePhase,
    CoarsePhaseTransition,
    CoarseSciencePolicy,
    CoarseScienceSession,
)


class UnknownBladeRuntimeError(RuntimeError):
    """The production composition cannot continue without overstating safety."""


class UnknownBladeRuntimePhase(StrEnum):
    OPERATOR_BOOTSTRAP = "operator_bootstrap"
    COARSE_SCAN = "coarse_scan"
    FINE_SCAN = "fine_scan"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    STOPPED = "stopped"


class UnknownBladeResumePhase(StrEnum):
    COARSE = "coarse"
    PREPARED = "prepared"
    FINE = "fine"
    COMPLETE = "complete"


class _Runner(Protocol):
    @property
    def status(self) -> ExperimentStatusSnapshot: ...

    def start(self) -> ExperimentStatusSnapshot: ...

    def step(self, *, view_id: str | None = None) -> ExperimentStatusSnapshot: ...

    def run_until_attention(
        self,
        *,
        maximum_steps: int = 32,
    ) -> ExperimentStatusSnapshot: ...

    def execute_approved_segment(
        self,
        approval: OperatorApproval,
    ) -> ExperimentStatusSnapshot: ...

    def request_stop(self, reason: str) -> ExperimentStatusSnapshot: ...

    def acknowledge_recovery(
        self,
        confirmation: RecoveryConfirmation,
    ) -> ExperimentStatusSnapshot: ...

    def approval_prompt(self) -> str: ...


class _CoarseSession(Protocol):
    @property
    def current_generation_path(self) -> Path | None: ...

    def stage_operator_capture(
        self,
        *,
        seed: BootstrapSeed | None = None,
        seed_provider: BootstrapSeedProvider | None = None,
        operator_side: BladeSide | None = None,
    ) -> None: ...

    def stage_selected_capture(
        self,
        selection: NextViewSelection,
        *,
        seed: BootstrapSeed | None = None,
    ) -> None: ...

    def accept_cycle(self, result: PerceptionCycleResult) -> Path: ...

    def reject_cycle(self) -> None: ...

    def select_next(self) -> NextViewSelection: ...

    def evaluate_transition(self) -> CoarsePhaseTransition: ...


class _FineRunnerFactory(Protocol):
    def __call__(self, reference_coarse_model: Path) -> _Runner: ...


class _ExperimentHandoffChain(Protocol):
    root: Path

    def append_coarse_checkpoint(self, *, coarse_generation: str | Path) -> object: ...

    def prepare_handoff(
        self,
        *,
        schema5_generation: str | Path,
        reference_coarse_model: str | Path,
        schema5_prepare_duration_s: float | None = None,
    ) -> object: ...

    def append_fine_start_candidate(
        self,
        *,
        fine_run_root: str | Path,
    ) -> object: ...

    def append_fine_started(
        self,
        *,
        timing_scope: Literal["uninterrupted_total", "resume_fine_start"],
        budget_check: Callable[[], float],
    ) -> object: ...

    def append_unaccepted_fine_started(
        self,
        *,
        fine_run_root: str | Path,
    ) -> object: ...

    def append_fine_checkpoint(
        self,
        *,
        accepted_surface_coverage_generation: str | Path,
    ) -> object: ...

    def append_fine_completed(
        self,
        *,
        final_surface_coverage_generation: str | Path,
        final_reconstruction_product: str | Path,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class UnknownBladeRuntimeSnapshot:
    phase: UnknownBladeRuntimePhase
    runner_status: ExperimentStatusSnapshot
    operator_bootstrap_views: int
    fine_source_replenishment_views: int
    current_coarse_generation_path: Path | None
    reference_coarse_model_path: Path | None
    timeline_root: Path
    bootstrap_stop_verified: bool
    bootstrap_stop_evidence_sha256: str | None
    blocking_reason: str | None = None
    final_reconstruction_path: Path | None = None

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class UnknownBladeResumePlan:
    experiment_root: Path
    experiment_id: str
    phase: UnknownBladeResumePhase
    coarse_run_root: Path
    coarse_generation_path: Path | None = None
    schema5_generation_path: Path | None = None
    reference_coarse_model_path: Path | None = None
    fine_run_root: Path | None = None
    accepted_fine_coverage_path: Path | None = None
    accepted_fine_event_count: int | None = None
    accepted_fine_event_sha256: str | None = None
    final_reconstruction_path: Path | None = None
    science_authority: ScienceAcceptanceAuthority | None = None
    runtime_timing_authority: RuntimeTimingAcceptanceAuthority | None = None
    placement_id: str | None = None


def _record_root(payload: object, *, label: str) -> Path:
    if not isinstance(payload, dict) or "root" not in payload:
        raise UnknownBladeRuntimeError(f"{label} authority record is missing")
    root = Path(str(payload["root"])).resolve()
    if not root.is_absolute():
        raise UnknownBladeRuntimeError(f"{label} authority root is not absolute")
    return root


def load_unknown_blade_resume_plan(output_root: str | Path) -> UnknownBladeResumePlan:
    """Derive one recovery plan solely from the explicitly named handoff chain."""

    experiment_root = Path(output_root).resolve()
    handoff_root = experiment_root / "experiment_handoff"
    stored = read_unknown_blade_experiment(handoff_root)
    if getattr(stored, "fine_start_protocol", None) != UNKNOWN_BLADE_FINE_START_PROTOCOL:
        raise UnknownBladeRuntimeError(
            "Legacy single-phase unknown-blade chains are audit-readable only; "
            "production continuation requires the candidate-commit fine-start protocol"
        )
    events = stored.events
    initialized = events[0].payload
    coarse_run_root = Path(str(initialized["coarse_run_root"])).resolve()
    coarse_checkpoints = tuple(
        event for event in events if event.event_type == "coarse_checkpoint"
    )
    coarse_generation = (
        _record_root(
            coarse_checkpoints[-1].payload["coarse_generation"],
            label="coarse checkpoint generation",
        )
        if coarse_checkpoints
        else None
    )
    prepared = next(
        (event for event in events if event.event_type == "handoff_prepared"),
        None,
    )
    started = next(
        (event for event in events if event.event_type == "fine_started"),
        None,
    )
    checkpoints = tuple(
        event for event in events if event.event_type == "fine_checkpoint"
    )
    completed = (
        events[-1] if events[-1].event_type == "fine_completed" else None
    )
    schema5 = (
        _record_root(prepared.payload["schema5_generation"], label="schema-5 generation")
        if prepared is not None
        else None
    )
    reference = (
        _record_root(
            prepared.payload["reference_coarse_model"],
            label="reference coarse model",
        )
        if prepared is not None
        else None
    )
    fine_run = (
        Path(str(started.payload["fine_run_root"])).resolve()
        if started is not None
        else None
    )
    accepted_coverage = (
        _record_root(
            checkpoints[-1].payload["accepted_surface_coverage_generation"],
            label="accepted fine coverage",
        )
        if checkpoints
        else None
    )
    accepted_event_count = (
        int(checkpoints[-1].payload["fine_event_count"])
        if checkpoints
        else None
    )
    accepted_event_sha256 = (
        str(checkpoints[-1].payload["fine_last_event_sha256"])
        if checkpoints
        else None
    )
    if completed is not None:
        final_coverage = _record_root(
            completed.payload["final_surface_coverage_generation"],
            label="final fine coverage",
        )
        final_reconstruction = _record_root(
            completed.payload["final_reconstruction_product"],
            label="final reconstruction",
        )
        if accepted_coverage != final_coverage:
            raise UnknownBladeRuntimeError(
                "Completed experiment does not inherit its final fine checkpoint"
            )
        phase = UnknownBladeResumePhase.COMPLETE
    elif started is not None:
        final_reconstruction = None
        phase = UnknownBladeResumePhase.FINE
    elif prepared is not None:
        final_reconstruction = None
        phase = UnknownBladeResumePhase.PREPARED
    else:
        final_reconstruction = None
        phase = UnknownBladeResumePhase.COARSE
    return UnknownBladeResumePlan(
        experiment_root=experiment_root,
        experiment_id=stored.experiment_id,
        phase=phase,
        coarse_run_root=coarse_run_root,
        coarse_generation_path=coarse_generation,
        schema5_generation_path=schema5,
        reference_coarse_model_path=reference,
        fine_run_root=fine_run,
        accepted_fine_coverage_path=accepted_coverage,
        accepted_fine_event_count=accepted_event_count,
        accepted_fine_event_sha256=accepted_event_sha256,
        final_reconstruction_path=final_reconstruction,
        science_authority=getattr(stored, "science_authority", None),
        runtime_timing_authority=getattr(stored, "runtime_timing_authority", None),
        placement_id=getattr(stored, "placement_id", None),
    )


class CompletedUnknownBladeRuntime:
    """Read-only report for an already sealed experiment; it owns no devices."""

    def __init__(self, plan: UnknownBladeResumePlan) -> None:
        if (
            plan.phase is not UnknownBladeResumePhase.COMPLETE
            or plan.fine_run_root is None
            or plan.final_reconstruction_path is None
        ):
            raise ValueError("Completed runtime requires a sealed fine resume plan")
        if plan.science_authority is None or plan.runtime_timing_authority is None:
            raise ValueError(
                "Production COMPLETE requires science and runtime timing authorities; "
                "legacy chains are audit-readable only"
            )
        fine = read_stop_scan_run(plan.fine_run_root)
        final = replay_final_fine_reconstruction(
            plan.final_reconstruction_path,
            expected_science_authority=plan.science_authority,
        )
        if final.root != plan.final_reconstruction_path:
            raise ValueError("Completed runtime reconstruction authority changed")
        self._snapshot = UnknownBladeRuntimeSnapshot(
            UnknownBladeRuntimePhase.COMPLETE,
            ExperimentStatusSnapshot(
                run_id=fine.run_id,
                run_root=fine.root,
                phase="complete",
                disposition=ExperimentDisposition.COMPLETE,
                cycle_index=fine.latest_event.cycle_index,
                current_view_id=None,
                proposed_view_id=None,
                expected_capture_view_id=None,
                expected_capture_purpose=None,
                blocking_reasons=(),
                event_count=len(fine.events),
                latest_event_sha256=fine.latest_event.event_sha256,
                recovery_required=False,
                awaiting_external_approval=False,
                stop_requested=False,
                stop_transport_acknowledged=False,
                stop_stationarity_verified=False,
            ),
            0,
            0,
            plan.coarse_generation_path,
            plan.reference_coarse_model_path,
            plan.experiment_root / "live_timeline",
            False,
            None,
            None,
            plan.final_reconstruction_path,
        )

    @property
    def snapshot(self) -> UnknownBladeRuntimeSnapshot:
        return self._snapshot

    def start(self) -> UnknownBladeRuntimeSnapshot:
        return self._snapshot


class CoarseSessionNextViewAdapter:
    """Bind one coarse session to coordinator selection and accepted-cycle callbacks.

    A selected target stays staged across any short TRANSIT legs.  It is cleared only
    when the exact CANDIDATE science wrapper is accepted, preventing a split motion
    from silently selecting a different endpoint midway through the route.
    """

    def __init__(self, session: _CoarseSession) -> None:
        self._session = session
        self._pending_selection: NextViewSelection | None = None
        self._last_transition: CoarsePhaseTransition | None = None
        self._accepted_cycle_count = 0
        self._pending_transition_result: PerceptionCycleResult | None = None
        self._run_reference_model_sha256: str | None = None
        self._run_selection_policy_sha256: str | None = None
        self._checkpoint_sink: Callable[[Path], None] | None = None
        self._last_checkpoint_generation: Path | None = None

    @property
    def last_transition(self) -> CoarsePhaseTransition | None:
        return self._last_transition

    @property
    def accepted_cycle_count(self) -> int:
        return self._accepted_cycle_count

    @property
    def current_generation_path(self) -> Path | None:
        return self._session.current_generation_path

    @property
    def has_pending_selection(self) -> bool:
        return self._pending_selection is not None

    @property
    def last_checkpoint_generation(self) -> Path | None:
        return self._last_checkpoint_generation

    def bind_checkpoint_sink(self, sink: Callable[[Path], None]) -> None:
        if self._checkpoint_sink is not None or self._accepted_cycle_count:
            raise UnknownBladeRuntimeError(
                "Coarse checkpoint sink must be bound once before the first cycle"
            )
        self._checkpoint_sink = sink

    def stage_operator_capture(
        self,
        *,
        seed: BootstrapSeed | None = None,
        seed_provider: BootstrapSeedProvider | None = None,
        operator_side: BladeSide | None = None,
    ) -> None:
        if self._pending_selection is not None:
            raise UnknownBladeRuntimeError(
                "A planned coarse target cannot be replaced by an operator capture"
            )
        if seed_provider is None:
            self._session.stage_operator_capture(seed=seed, operator_side=operator_side)
        else:
            self._session.stage_operator_capture(
                seed=seed,
                seed_provider=seed_provider,
                operator_side=operator_side,
            )

    def reject_staged_cycle(self) -> None:
        self._session.reject_cycle()
        self._pending_selection = None

    def observe_perception(self, result: PerceptionCycleResult) -> None:
        if result.coarse_scan_view_path is None:
            # A short route may require one or more TRANSIT captures.  Those update
            # safety occupancy but intentionally do not consume the staged science
            # target.
            return
        generation = Path(self._session.accept_cycle(result)).resolve()
        if self._checkpoint_sink is None:
            raise UnknownBladeRuntimeError(
                "Accepted coarse science has no top-level checkpoint sink"
            )
        self._checkpoint_sink(generation)
        self._last_checkpoint_generation = generation
        self._pending_selection = None
        self._accepted_cycle_count += 1
        # Schema-5 promotion is intentionally deferred.  The outer runtime may
        # evaluate it only after this exact result is the current MAP_READY disk
        # authority; accepting a science asset alone is insufficient.
        self._pending_transition_result = result

    def promote_after_exact_map_ready(
        self,
        assert_exact_map_ready: Callable[[PerceptionCycleResult], None],
    ) -> CoarsePhaseTransition | None:
        result = self._pending_transition_result
        if result is None:
            return None
        assert_exact_map_ready(result)
        self._last_transition = self._session.evaluate_transition()
        self._pending_transition_result = None
        return self._last_transition

    def select_next(
        self,
        observation: PerceptionCycleResult,
        generation: OccupancyGeneration,
    ) -> NextViewSelection:
        del observation, generation
        if self._pending_selection is not None:
            return self._pending_selection
        if (
            self._last_transition is not None
            and self._last_transition.phase is CoarsePhase.READY_FOR_FINE
        ):
            raise UnknownBladeRuntimeError(
                "Schema-5 is ready for the outer fine handoff; the coarse run must "
                "not emit a completion decision with a different reference hash"
            )
        selection = self._session.select_next()
        if selection.coverage_complete or selection.target is None:
            raise UnknownBladeRuntimeError(
                "Coarse completion must be promoted through the schema-5 gate"
            )
        binding = (
            selection.reference_model_sha256,
            selection.selection_policy_sha256,
        )
        if self._run_reference_model_sha256 is None:
            (
                self._run_reference_model_sha256,
                self._run_selection_policy_sha256,
            ) = binding
        elif binding != (
            self._run_reference_model_sha256,
            self._run_selection_policy_sha256,
        ):
            raise UnknownBladeRuntimeError(
                "Coarse reference or selection policy changed within one run"
            )
        self._session.stage_selected_capture(selection)
        self._pending_selection = selection
        return selection


class _FineCheckpointRecorder:
    """Persist every advanced fine-run/coverage pair, including run-only advances."""

    def __init__(
        self,
        chain: _ExperimentHandoffChain,
        *,
        initial_coverage: Path | None = None,
        initial_event_count: int | None = None,
        initial_event_sha256: str | None = None,
        verify_chain: Callable[[], None] = lambda: None,
    ) -> None:
        present = (
            initial_coverage is not None,
            initial_event_count is not None,
            initial_event_sha256 is not None,
        )
        if any(present) and not all(present):
            raise ValueError("Recovered fine checkpoint binding is incomplete")
        self._chain = chain
        self._last_binding = (
            (
                initial_coverage.resolve(),
                int(initial_event_count),
                str(initial_event_sha256),
            )
            if initial_coverage is not None
            and initial_event_count is not None
            and initial_event_sha256 is not None
            else None
        )
        self._verify_chain = verify_chain

    def record(self, result: PerceptionCycleResult, *, fine_run_root: Path) -> bool:
        coverage = result.coverage_path
        if coverage is None:
            return False
        coverage = coverage.resolve()
        fine = read_stop_scan_run(fine_run_root)
        if not fine.events:
            raise UnknownBladeRuntimeError(
                "Fine perception callback has no persisted run authority"
            )
        binding = (coverage, len(fine.events), fine.latest_event.event_sha256)
        if binding == self._last_binding:
            return False
        self._chain.append_fine_checkpoint(
            accepted_surface_coverage_generation=coverage,
        )
        self._verify_chain()
        self._last_binding = binding
        return True


class UnknownBladeSupervisedRuntime:
    """Phase-aware shell around coarse and fine supervised runners."""

    def __init__(
        self,
        *,
        coarse_runner: _Runner,
        coarse_adapter: CoarseSessionNextViewAdapter,
        create_fine_runner: _FineRunnerFactory,
        operator_id: str,
        minimum_operator_bootstrap_views: int,
        timeline_root: str | Path,
        establish_initial_stop: Callable[[], BootstrapSafeStateEvidence],
        assert_exact_map_ready: Callable[[PerceptionCycleResult], None],
        experiment_handoff: _ExperimentHandoffChain,
        resume_phase: UnknownBladeResumePhase | None = None,
        recovered_reference: str | Path | None = None,
        recovered_fine_runner: _Runner | None = None,
        science_authority: ScienceAcceptanceAuthority | None = None,
        science_settings: AppSettings | None = None,
        runtime_timing_authority: RuntimeTimingAcceptanceAuthority | None = None,
        maximum_schema5_handoff_duration_s: float | None = None,
        experimental: bool = False,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not operator_id.strip():
            raise ValueError("Unknown-blade runtime requires an operator identity")
        if minimum_operator_bootstrap_views < 3:
            raise ValueError("Unknown-space bootstrap requires at least three views")
        if maximum_schema5_handoff_duration_s is not None and (
            not math.isfinite(maximum_schema5_handoff_duration_s)
            or maximum_schema5_handoff_duration_s <= 0.0
        ):
            raise ValueError("Schema-5 handoff timing budget must be finite and positive")
        if resume_phase in {
            UnknownBladeResumePhase.PREPARED,
            UnknownBladeResumePhase.FINE,
        } and (recovered_reference is None or recovered_fine_runner is None):
            raise ValueError("Fine recovery requires its pinned reference and runner")
        if resume_phase in {UnknownBladeResumePhase.COMPLETE}:
            raise ValueError("A sealed experiment uses CompletedUnknownBladeRuntime")
        self._resume_phase = resume_phase
        self._active_runner = recovered_fine_runner or coarse_runner
        self._coarse_runner = coarse_runner
        self._coarse_adapter = coarse_adapter
        self._create_fine_runner = create_fine_runner
        self._operator_id = operator_id.strip()
        self._minimum_bootstrap = minimum_operator_bootstrap_views
        self._timeline_root = Path(timeline_root).resolve()
        self._establish_initial_stop = establish_initial_stop
        self._assert_exact_map_ready = assert_exact_map_ready
        self._experiment_handoff = experiment_handoff
        if (science_authority is None) != (science_settings is None):
            raise ValueError("Science authority and settings must be supplied together")
        self._science_authority = science_authority
        self._science_settings = (
            science_settings.model_copy(deep=True)
            if science_settings is not None
            else None
        )
        if runtime_timing_authority is not None:
            if science_settings is None:
                raise ValueError("Runtime timing authority requires authoritative settings")
            runtime_timing_authority.assert_current(science_settings)
            accepted_schema5_limit = runtime_timing_authority.timing_limits_s[
                "maximum_schema5_handoff_duration_s"
            ]
            if maximum_schema5_handoff_duration_s != accepted_schema5_limit:
                raise ValueError(
                    "Schema-5 runtime limit differs from the timing acceptance authority"
                )
        self._runtime_timing_authority = runtime_timing_authority
        self._maximum_schema5_handoff_duration_s = maximum_schema5_handoff_duration_s
        self._experimental = experimental
        self._monotonic_clock = monotonic_clock
        self._bootstrap_count = 0
        self._fine_replenishment_count = 0
        self._phase = (
            UnknownBladeRuntimePhase.FINE_SCAN
            if resume_phase in {
                UnknownBladeResumePhase.PREPARED,
                UnknownBladeResumePhase.FINE,
            }
            else UnknownBladeRuntimePhase.OPERATOR_BOOTSTRAP
        )
        self._reference = (
            Path(recovered_reference).resolve()
            if recovered_reference is not None
            else None
        )
        self._blocking_reason: str | None = None
        self._started = False
        self._bootstrap_stop_evidence: BootstrapSafeStateEvidence | None = None
        # This latch is deliberately independent of any phase runner.  It is set
        # before asking a runner/controller to stop, and therefore also covers a
        # coarse-to-fine factory that is concurrently constructing a new runner.
        self._stop_requested = Event()
        self._stop_confirmed = False
        self._fine_runner = recovered_fine_runner
        self._constructing_runner: _Runner | None = None
        self._final_reconstruction_path: Path | None = None
        self._coarse_adapter.bind_checkpoint_sink(self._append_coarse_checkpoint)

    def _verify_handoff_chain(self) -> None:
        if isinstance(self._experiment_handoff, UnknownBladeExperimentWriter):
            read_unknown_blade_experiment(self._experiment_handoff.root)

    def _append_coarse_checkpoint(self, generation: Path) -> None:
        recorder = try_create_performance_timing(
            transaction_kind="unknown_blade_coarse_checkpoint",
            identity={"coarse_generation": str(generation.resolve())},
        )
        if recorder is None:
            self._append_coarse_checkpoint_transaction(generation)
            return
        status = "failed"
        error: str | None = None
        try:
            with activate_performance_timing(recorder):
                self._append_coarse_checkpoint_transaction(generation)
            status = "completed"
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            recorder.write_best_effort(
                self._timeline_root.parent
                / "performance_diagnostics"
                / f"coarse_checkpoint_{generation.name}.json",
                status=status,
                error=error,
            )

    def _append_coarse_checkpoint_transaction(self, generation: Path) -> None:
        with performance_span("experiment.checkpoint_append"):
            self._experiment_handoff.append_coarse_checkpoint(
                coarse_generation=generation,
            )
        with performance_span("experiment.checkpoint_full_verify"):
            self._verify_handoff_chain()

    @property
    def snapshot(self) -> UnknownBladeRuntimeSnapshot:
        status = self._active_runner.status
        phase = self._phase
        blocking_reason = self._blocking_reason
        if self._stop_requested.is_set():
            phase = (
                UnknownBladeRuntimePhase.STOPPED
                if self._stop_confirmed
                else UnknownBladeRuntimePhase.BLOCKED
            )
        elif phase in {
            UnknownBladeRuntimePhase.STOPPED,
            UnknownBladeRuntimePhase.BLOCKED,
            UnknownBladeRuntimePhase.COMPLETE,
        }:
            pass
        elif status.disposition is ExperimentDisposition.COMPLETE:
            phase = UnknownBladeRuntimePhase.BLOCKED
            blocking_reason = (
                "supervised runner completed before the outer experiment chain was sealed"
            )
        elif status.disposition is ExperimentDisposition.BLOCKED:
            phase = UnknownBladeRuntimePhase.BLOCKED
        return UnknownBladeRuntimeSnapshot(
            phase=phase,
            runner_status=status,
            operator_bootstrap_views=self._bootstrap_count,
            fine_source_replenishment_views=self._fine_replenishment_count,
            current_coarse_generation_path=self._coarse_adapter.current_generation_path,
            reference_coarse_model_path=self._reference,
            timeline_root=self._timeline_root,
            bootstrap_stop_verified=self._bootstrap_stop_evidence is not None,
            bootstrap_stop_evidence_sha256=(
                self._bootstrap_stop_evidence.evidence_sha256
                if self._bootstrap_stop_evidence is not None
                else None
            ),
            blocking_reason=blocking_reason,
            final_reconstruction_path=self._final_reconstruction_path,
        )

    def start(self) -> UnknownBladeRuntimeSnapshot:
        if self._started:
            raise UnknownBladeRuntimeError("Unknown-blade runtime was already started")
        resumed_handoff_started = (
            self._monotonic_now()
            if self._resume_phase is UnknownBladeResumePhase.PREPARED
            else None
        )
        # The first console prompt may mention a stopped robot only after a real
        # controller stop and a sampled RTSI stationarity window have succeeded.
        evidence = self._establish_initial_stop()
        if not isinstance(evidence, BootstrapSafeStateEvidence):
            raise UnknownBladeRuntimeError(
                "Initial stop callback did not return multi-channel bootstrap evidence"
            )
        self._bootstrap_stop_evidence = evidence
        if self._resume_phase in {
            UnknownBladeResumePhase.COARSE,
            UnknownBladeResumePhase.FINE,
        }:
            status = self._active_runner.acknowledge_recovery(
                RecoveryConfirmation(self._operator_id, True, True)
            )
        elif self._resume_phase is UnknownBladeResumePhase.PREPARED:
            assert resumed_handoff_started is not None
            try:
                status = self._active_runner.start()
                self._require_schema5_handoff_budget(resumed_handoff_started)
                self._experiment_handoff.append_fine_start_candidate(
                    fine_run_root=status.run_root,
                )
                self._verify_handoff_chain()
                self._require_schema5_handoff_budget(resumed_handoff_started)
                self._experiment_handoff.append_fine_started(
                    timing_scope="resume_fine_start",
                    budget_check=lambda: self._require_schema5_commit_budget(
                        resumed_handoff_started
                    ),
                )
                # The storage writer checked the deadline after fsync of the
                # completed event temporary and before its atomic publication.
                # A durable FINE_STARTED is therefore authoritative; do not add
                # a post-publication timing gate that could be reversed on restart.
                self._verify_handoff_chain()
            except BaseException as exc:
                self._started = True
                return self._block(
                    "fine handoff resume failed closed: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            status = self._active_runner.start()
        self._started = True
        self._apply_terminal_status(status)
        return self.snapshot

    def capture_operator_view(
        self,
        *,
        view_id: str | None = None,
        seed: BootstrapSeed | None = None,
        seed_provider: BootstrapSeedProvider | None = None,
        operator_side: BladeSide | None = None,
    ) -> UnknownBladeRuntimeSnapshot:
        """Accept exactly one explicit ``c`` action while the robot is stopped."""

        self._require_started()
        if self._phase is not UnknownBladeRuntimePhase.OPERATOR_BOOTSTRAP:
            raise UnknownBladeRuntimeError(
                "Operator-positioned captures are disabled after coarse motion starts"
            )
        if self._bootstrap_count == 0 and operator_side is BladeSide.BACK:
            raise UnknownBladeRuntimeError(
                "The first operator capture may use only an inferred side or FRONT"
            )
        status = self._active_runner.status
        if (
            status.disposition is not ExperimentDisposition.NEEDS_CAPTURE
            or status.expected_capture_view_id is not None
        ):
            raise UnknownBladeRuntimeError(
                "The current runner is not awaiting an operator-positioned capture"
            )
        selected_id = (view_id or f"operator_bootstrap_{self._bootstrap_count:03d}").strip()
        if not selected_id:
            raise ValueError("Operator bootstrap view ID must be non-empty")
        before = self._coarse_adapter.accepted_cycle_count
        self._coarse_adapter.stage_operator_capture(
            seed=seed,
            seed_provider=seed_provider,
            operator_side=operator_side,
        )
        try:
            updated = self._active_runner.step(view_id=selected_id)
        except BaseException:
            self._coarse_adapter.reject_staged_cycle()
            raise
        if self._coarse_adapter.accepted_cycle_count != before + 1:
            self._coarse_adapter.reject_staged_cycle()
            return self._block("operator capture lacked one accepted coarse-science cycle")
        self._bootstrap_count += 1
        if updated.disposition is ExperimentDisposition.READY:
            promotion_started_monotonic_s = self._monotonic_now()
            try:
                transition = self._coarse_adapter.promote_after_exact_map_ready(
                    self._assert_exact_map_ready
                )
            except BaseException as exc:
                return self._block(
                    "coarse schema promotion lacked its exact MAP_READY authority: "
                    f"{type(exc).__name__}: {exc}"
                )
            if self._bootstrap_count < self._minimum_bootstrap:
                return self._block(
                    "occupancy became MAP_READY before the required operator-view count"
                )
            self._set_phase(UnknownBladeRuntimePhase.COARSE_SCAN)
            if transition is not None and transition.phase is CoarsePhase.READY_FOR_FINE:
                self._activate_fine(
                    transition,
                    handoff_started_monotonic_s=promotion_started_monotonic_s,
                )
        elif updated.disposition is ExperimentDisposition.NEEDS_CAPTURE:
            if self._bootstrap_count >= self._minimum_bootstrap:
                # More manually repositioned views are allowed when the map has not
                # yet met its independent-view evidence gates.
                self._set_phase(UnknownBladeRuntimePhase.OPERATOR_BOOTSTRAP)
        elif updated.disposition is ExperimentDisposition.BLOCKED:
            return self._block("operator bootstrap capture was blocked")
        elif updated.disposition is ExperimentDisposition.COMPLETE:
            return self._block("coarse runner completed before a verified fine handoff")
        return self.snapshot

    def advance_to_attention(self) -> UnknownBladeRuntimeSnapshot:
        """Plan/preflight only; never execute a segment."""

        self._require_started()
        if self._phase is UnknownBladeRuntimePhase.OPERATOR_BOOTSTRAP:
            raise UnknownBladeRuntimeError("Press c after manually repositioning the stopped robot")
        if self._phase in {
            UnknownBladeRuntimePhase.BLOCKED,
            UnknownBladeRuntimePhase.COMPLETE,
            UnknownBladeRuntimePhase.STOPPED,
        }:
            return self.snapshot
        try:
            status = self._active_runner.run_until_attention(maximum_steps=32)
        except BaseException:
            self._coarse_adapter.reject_staged_cycle()
            raise
        self._apply_terminal_status(status)
        return self.snapshot

    def capture_fine_source_replenishment(
        self,
        *,
        view_id: str | None = None,
    ) -> UnknownBladeRuntimeSnapshot:
        """Add one explicit stopped observation when fine MAP_READY evidence is incomplete."""

        self._require_started()
        if self._phase is not UnknownBladeRuntimePhase.FINE_SCAN:
            raise UnknownBladeRuntimeError(
                "Fine source replenishment is available only after the verified handoff"
            )
        status = self._active_runner.status
        if (
            status.disposition is not ExperimentDisposition.NEEDS_CAPTURE
            or status.expected_capture_view_id is not None
        ):
            raise UnknownBladeRuntimeError(
                "The fine runner is not awaiting an operator-positioned safety refresh"
            )
        selected = (
            view_id
            or f"fine_source_refresh_{self._fine_replenishment_count:03d}"
        ).strip()
        if not selected:
            raise ValueError("Fine source-replenishment view ID must be non-empty")
        updated = self._active_runner.step(view_id=selected)
        if updated.disposition is ExperimentDisposition.BLOCKED:
            return self._block("fine source replenishment was blocked")
        if updated.disposition is ExperimentDisposition.COMPLETE:
            return self._block("fine runner completed while replenishing its safety map")
        if updated.disposition not in {
            ExperimentDisposition.NEEDS_CAPTURE,
            ExperimentDisposition.READY,
        }:
            return self._block(
                "fine source replenishment reached an invalid coordinator disposition"
            )
        self._fine_replenishment_count += 1
        return self.snapshot

    def execute_exact_approval(self, confirmation: str) -> UnknownBladeRuntimeSnapshot:
        """Execute one prepared short segment, then automatically stop and capture."""

        self._require_started()
        expected = self._active_runner.approval_prompt()
        if confirmation.strip() != expected:
            raise UnknownBladeRuntimeError(f"Approval mismatch; expected exactly {expected!r}")
        try:
            status = self._active_runner.execute_approved_segment(
                OperatorApproval(self._operator_id, confirmation)
            )
        except BaseException:
            self._coarse_adapter.reject_staged_cycle()
            raise
        if status.disposition is ExperimentDisposition.BLOCKED:
            return self._block("approved segment execution was blocked")
        if status.disposition is ExperimentDisposition.COMPLETE:
            self._apply_terminal_status(status)
            return self.snapshot
        if status.disposition is not ExperimentDisposition.NEEDS_CAPTURE:
            return self._block("guarded segment did not terminate at the capture gate")

        # One exact post-motion capture is automatic.  The coordinator owns the
        # expected view ID and purpose; no user string can redirect it.
        capture_purpose = status.expected_capture_purpose
        accepted_before = self._coarse_adapter.accepted_cycle_count
        try:
            captured = self._active_runner.step()
        except BaseException:
            self._coarse_adapter.reject_staged_cycle()
            raise
        if self._phase is UnknownBladeRuntimePhase.COARSE_SCAN:
            accepted_after = self._coarse_adapter.accepted_cycle_count
            if capture_purpose == "candidate" and accepted_after != accepted_before + 1:
                return self._block(
                    "automatic candidate capture lacked one accepted coarse-science cycle"
                )
            if capture_purpose == "transit" and accepted_after != accepted_before:
                return self._block("transit capture unexpectedly consumed coarse science")
        self._apply_terminal_status(captured)
        if (
            self._phase is UnknownBladeRuntimePhase.COARSE_SCAN
            and captured.disposition is ExperimentDisposition.READY
        ):
            promotion_started_monotonic_s = self._monotonic_now()
            try:
                transition = self._coarse_adapter.promote_after_exact_map_ready(
                    self._assert_exact_map_ready
                )
            except BaseException as exc:
                return self._block(
                    "coarse schema promotion lacked its exact MAP_READY authority: "
                    f"{type(exc).__name__}: {exc}"
                )
            if transition is not None and transition.phase is CoarsePhase.READY_FOR_FINE:
                self._activate_fine(
                    transition,
                    handoff_started_monotonic_s=promotion_started_monotonic_s,
                )
        if (
            self._phase not in {
            UnknownBladeRuntimePhase.BLOCKED,
            UnknownBladeRuntimePhase.COMPLETE,
            }
            and self._active_runner.status.disposition is ExperimentDisposition.READY
        ):
            try:
                planned = self._active_runner.run_until_attention(maximum_steps=32)
            except BaseException:
                self._coarse_adapter.reject_staged_cycle()
                raise
            self._apply_terminal_status(planned)
        return self.snapshot

    def approval_prompt(self) -> str:
        return self._active_runner.approval_prompt()

    def request_stop(self, reason: str) -> UnknownBladeRuntimeSnapshot:
        text = reason.strip()
        if not text:
            raise ValueError("Stop reason must be non-empty")
        if self._stop_confirmed:
            return self.snapshot
        self._stop_requested.set()
        self._blocking_reason = text
        self._set_phase(UnknownBladeRuntimePhase.BLOCKED)
        self._coarse_adapter.reject_staged_cycle()
        runners = tuple(
            dict.fromkeys(
                item
                for item in (
                    self._active_runner,
                    self._constructing_runner,
                    self._fine_runner,
                )
                if item is not None
            )
        )
        active_status: ExperimentStatusSnapshot | None = None
        stop_errors: list[BaseException] = []
        for runner in runners:
            try:
                current = runner.status
                stopped = (
                    current
                    if (
                        current.phase == "aborted"
                        and current.stop_transport_acknowledged
                        and current.stop_stationarity_verified
                    )
                    else runner.request_stop(text)
                )
                if runner is self._active_runner:
                    active_status = stopped
            except BaseException as exc:
                stop_errors.append(exc)
        if stop_errors:
            summary = ";".join(
                f"{type(exc).__name__}:{exc}" for exc in stop_errors
            )
            self._blocking_reason = f"stop_failed:{summary}"
            self._phase = UnknownBladeRuntimePhase.BLOCKED
            grouped = BaseExceptionGroup(
                "One or more supervised runners failed their stop protocol",
                stop_errors,
            )
            raise UnknownBladeRuntimeError(self._blocking_reason) from grouped
        if active_status is None or not self._stop_status_confirmed(active_status):
            self._blocking_reason = "stop_requested_but_abort_not_confirmed"
            self._phase = UnknownBladeRuntimePhase.BLOCKED
            raise UnknownBladeRuntimeError(self._blocking_reason)
        self._stop_confirmed = True
        self._set_phase(UnknownBladeRuntimePhase.STOPPED)
        return self.snapshot

    def _activate_fine(
        self,
        transition: CoarsePhaseTransition,
        *,
        handoff_started_monotonic_s: float,
    ) -> None:
        if self._stop_requested.is_set():
            self._set_phase(UnknownBladeRuntimePhase.STOPPED)
            return
        try:
            ready, reference = self._verify_fine_transition(transition)
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            if self._stop_requested.is_set():
                self._set_phase(UnknownBladeRuntimePhase.STOPPED)
                return
            self._coarse_runner.request_stop("schema-5 handoff invalidated the coarse coordinator")
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            if self._stop_requested.is_set():
                self._set_phase(UnknownBladeRuntimePhase.STOPPED)
                return
            checkpoint_generation = self._coarse_adapter.last_checkpoint_generation
            if (
                checkpoint_generation is None
                or checkpoint_generation.resolve()
                != transition.source_generation_path.resolve()
            ):
                raise UnknownBladeRuntimeError(
                    "Fine handoff has no checkpoint for its source coarse generation"
                )
            self._experiment_handoff.append_coarse_checkpoint(
                coarse_generation=checkpoint_generation,
            )
            self._verify_handoff_chain()
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            prepare_kwargs: dict[str, object] = {
                "schema5_generation": ready,
                "reference_coarse_model": reference,
            }
            if self._runtime_timing_authority is not None:
                prepare_kwargs["schema5_prepare_duration_s"] = (
                    self._require_schema5_handoff_budget(
                        handoff_started_monotonic_s
                    )
                )
            self._experiment_handoff.prepare_handoff(**prepare_kwargs)
            self._verify_handoff_chain()
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            fine_runner = self._create_fine_runner(reference)
            self._constructing_runner = fine_runner
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            if self._stop_requested.is_set():
                self._stop_unstarted_runner(fine_runner)
                self._constructing_runner = None
                self._set_phase(UnknownBladeRuntimePhase.STOPPED)
                return
            fine_status = fine_runner.start()
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            if self._stop_requested.is_set():
                self._stop_unstarted_runner(fine_runner)
                self._constructing_runner = None
                self._set_phase(UnknownBladeRuntimePhase.STOPPED)
                return
            self._require_schema5_handoff_budget(handoff_started_monotonic_s)
            if self._experimental:
                self._experiment_handoff.append_unaccepted_fine_started(
                    fine_run_root=fine_status.run_root,
                )
            else:
                self._experiment_handoff.append_fine_start_candidate(
                    fine_run_root=fine_status.run_root,
                )
                self._verify_handoff_chain()
                self._require_schema5_handoff_budget(handoff_started_monotonic_s)
                self._experiment_handoff.append_fine_started(
                    timing_scope="uninterrupted_total",
                    budget_check=lambda: self._require_schema5_commit_budget(
                        handoff_started_monotonic_s
                    ),
                )
            # The final event's atomic publication is the activation
            # linearization point; its storage callback checked the accepted
            # deadline after fsync and before publication.
            self._verify_handoff_chain()
            self._reference = reference
            self._active_runner = fine_runner
            self._fine_runner = fine_runner
            self._constructing_runner = None
            self._set_phase(UnknownBladeRuntimePhase.FINE_SCAN)
            initial = fine_runner.step(view_id="fine_transition_bootstrap_000")
            if self._stop_requested.is_set():
                self._set_phase(UnknownBladeRuntimePhase.STOPPED)
                return
        except BaseException as exc:
            constructing = self._constructing_runner
            if constructing is not None:
                try:
                    self._stop_unstarted_runner(constructing)
                except BaseException as stop_exc:
                    exc = BaseExceptionGroup(
                        "fine handoff and orphan-runner stop both failed",
                        [exc, stop_exc],
                    )
                finally:
                    self._constructing_runner = None
            self._block(f"fine handoff failed closed: {type(exc).__name__}: {exc}")
            return
        if initial.disposition is ExperimentDisposition.NEEDS_CAPTURE:
            # The FINE_STARTED chain is already durable, so explicit stopped `c`
            # captures may now replenish the new coordinator without inheriting a
            # coarse publication, permit, or prepared segment.
            return
        if initial.disposition is not ExperimentDisposition.READY:
            self._block("fine handoff failed to enter a recoverable safety-map state")
            return

    def _monotonic_now(self) -> float:
        value = float(self._monotonic_clock())
        if not math.isfinite(value):
            raise UnknownBladeRuntimeError("Monotonic runtime clock returned a non-finite value")
        return value

    def _require_schema5_handoff_budget(self, started_monotonic_s: float) -> float:
        elapsed = self._monotonic_now() - started_monotonic_s
        if elapsed < 0.0:
            raise UnknownBladeRuntimeError(
                "Monotonic runtime clock moved backwards during schema-5 handoff"
            )
        limit = self._maximum_schema5_handoff_duration_s
        if limit is not None and elapsed > limit:
            raise UnknownBladeRuntimeError(
                "schema-5 handoff exceeded accepted timing budget: "
                f"actual={elapsed:.9g}s, limit={limit:.9g}s"
            )
        return elapsed

    def _require_schema5_commit_budget(self, started_monotonic_s: float) -> float:
        """Revalidate timing authority and return elapsed time at commit."""

        if self._runtime_timing_authority is not None:
            if self._science_settings is None:
                raise UnknownBladeRuntimeError(
                    "Runtime timing authority lost its authoritative settings"
                )
            self._runtime_timing_authority.assert_current(self._science_settings)
        return self._require_schema5_handoff_budget(started_monotonic_s)

    def _block(self, reason: str) -> UnknownBladeRuntimeSnapshot:
        self._blocking_reason = reason
        self._set_phase(UnknownBladeRuntimePhase.BLOCKED)
        self._coarse_adapter.reject_staged_cycle()
        try:
            if self._active_runner.status.phase != "aborted":
                self._active_runner.request_stop(reason)
        except BaseException as exc:
            self._blocking_reason = f"{reason}; stop_failed:{type(exc).__name__}:{exc}"
        return self.snapshot

    def _verify_fine_transition(
        self,
        transition: CoarsePhaseTransition,
    ) -> tuple[Path, Path]:
        ready = transition.ready_generation_path
        reference = transition.reference_coarse_model_path
        if ready is None or reference is None:
            raise UnknownBladeRuntimeError("schema-5 transition omitted its bound assets")
        ready = ready.resolve()
        reference = reference.resolve()
        current = self._coarse_adapter.current_generation_path
        if current is None or current.resolve() != ready:
            raise UnknownBladeRuntimeError(
                "schema-5 ready generation is not the adapter's current generation"
            )
        stored = read_coarse_scan_generation(ready)
        if stored.root != ready:
            raise UnknownBladeRuntimeError("schema-5 generation readback changed its root")
        if stored.previous_generation_path != transition.source_generation_path.resolve():
            raise UnknownBladeRuntimeError(
                "schema-5 predecessor differs from the transition source generation"
            )
        if stored.coarse_model_path != reference:
            raise UnknownBladeRuntimeError(
                "schema-5 generation coarse model differs from the transition reference"
            )
        return ready, reference

    def _apply_terminal_status(self, status: ExperimentStatusSnapshot) -> None:
        if self._stop_requested.is_set():
            self._set_phase(UnknownBladeRuntimePhase.STOPPED)
        elif status.disposition is ExperimentDisposition.BLOCKED:
            self._block("active supervised runner entered a blocked state")
        elif status.disposition is ExperimentDisposition.COMPLETE:
            if (
                self._phase is UnknownBladeRuntimePhase.FINE_SCAN
                and self._reference is not None
                and self._fine_runner is self._active_runner
            ):
                try:
                    self._seal_fine_completion(status)
                except BaseException as exc:
                    self._block(
                        "fine completion failed outer-chain verification: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                self._block("coarse runner completed before a verified fine handoff")

    def _seal_fine_completion(self, status: ExperimentStatusSnapshot) -> None:
        if (
            isinstance(self._experiment_handoff, UnknownBladeExperimentWriter)
            and not self._experimental
            and (self._science_authority is None or self._science_settings is None)
        ):
            raise UnknownBladeRuntimeError(
                "Production completion lacks its science acceptance authority"
            )
        if self._science_authority is not None and self._science_settings is not None:
            self._science_authority.assert_current(self._science_settings)
        if self._runtime_timing_authority is not None and self._science_settings is not None:
            self._runtime_timing_authority.assert_current(self._science_settings)
        fine = read_stop_scan_run(status.run_root)
        if (
            fine.run_id != status.run_id
            or not fine.events
            or fine.latest_event.event_sha256 != status.latest_event_sha256
            or fine.latest_event.event_type != "coverage_complete"
            or fine.latest_event.phase != "complete"
        ):
            raise UnknownBladeRuntimeError(
                "Fine COMPLETE status is not its persisted coverage_complete authority"
            )
        payload = fine.latest_event.payload
        terminal = payload.get("final_reconstruction")
        if not isinstance(terminal, dict) or set(terminal) != {
            "path",
            "artifact_id",
            "metadata_sha256",
        }:
            raise UnknownBladeRuntimeError(
                "Fine completion event lacks immutable final-reconstruction evidence"
            )
        terminal_path = Path(str(terminal["path"]))
        if not terminal_path.is_absolute() or terminal_path != terminal_path.resolve():
            raise UnknownBladeRuntimeError(
                "Fine completion reconstruction path is not absolute and canonical"
            )
        reconstruction = (
            replay_final_fine_reconstruction(
                terminal_path,
                expected_science_authority=self._science_authority,
            )
            if self._science_authority is not None
            else replay_final_fine_reconstruction(terminal_path)
        )
        coverage = reconstruction.result.coverage
        if (
            reconstruction.root != terminal_path
            or reconstruction.artifact_id != terminal["artifact_id"]
            or reconstruction.metadata_sha256 != terminal["metadata_sha256"]
            or payload.get("surface_generation_id") != coverage.generation_id
        ):
            raise UnknownBladeRuntimeError(
                "Fine completion event differs from replayed reconstruction evidence"
            )
        self._experiment_handoff.append_fine_checkpoint(
            accepted_surface_coverage_generation=coverage.root,
        )
        self._verify_handoff_chain()
        if self._science_authority is not None and self._science_settings is not None:
            self._science_authority.assert_current(self._science_settings)
        self._experiment_handoff.append_fine_completed(
            final_surface_coverage_generation=coverage.root,
            final_reconstruction_product=reconstruction.root,
        )
        self._verify_handoff_chain()
        if isinstance(self._experiment_handoff, UnknownBladeExperimentWriter):
            sealed = read_unknown_blade_experiment(self._experiment_handoff.root)
            if sealed.latest_event.event_type != "fine_completed":
                raise UnknownBladeRuntimeError(
                    "Outer experiment chain did not persist FINE_COMPLETED"
                )
            if sealed.science_authority != self._science_authority:
                raise UnknownBladeRuntimeError(
                    "Sealed outer chain changed its science acceptance authority"
                )
            if sealed.runtime_timing_authority != self._runtime_timing_authority:
                raise UnknownBladeRuntimeError(
                    "Sealed outer chain changed its runtime timing authority"
                )
        if self._science_authority is not None and self._science_settings is not None:
            self._science_authority.assert_current(self._science_settings)
        if self._runtime_timing_authority is not None and self._science_settings is not None:
            self._runtime_timing_authority.assert_current(self._science_settings)
        self._final_reconstruction_path = reconstruction.root
        self._set_phase(UnknownBladeRuntimePhase.COMPLETE)

    def _set_phase(self, phase: UnknownBladeRuntimePhase) -> bool:
        if self._stop_requested.is_set():
            self._phase = (
                UnknownBladeRuntimePhase.STOPPED
                if self._stop_confirmed
                else UnknownBladeRuntimePhase.BLOCKED
            )
            return False
        self._phase = phase
        # Close the store/check race without holding a lock that could delay stop.
        if self._stop_requested.is_set():
            self._phase = (
                UnknownBladeRuntimePhase.STOPPED
                if self._stop_confirmed
                else UnknownBladeRuntimePhase.BLOCKED
            )
            return False
        return True

    @staticmethod
    def _stop_status_confirmed(status: ExperimentStatusSnapshot) -> bool:
        return (
            status.phase == "aborted"
            and status.disposition is ExperimentDisposition.BLOCKED
            and status.stop_requested
            and status.stop_transport_acknowledged
            and status.stop_stationarity_verified
        )

    @staticmethod
    def _stop_unstarted_runner(runner: _Runner) -> None:
        # A factory-created IDLE runner has not acquired motion authority.  If it
        # was already started concurrently, its ordinary stop boundary must work.
        if runner.status.phase == "idle":
            return
        runner.request_stop("operator stop arrived during fine-runner construction")

    def _require_started(self) -> None:
        if not self._started:
            raise UnknownBladeRuntimeError("Unknown-blade runtime is not started")


def unknown_blade_runtime_readiness(
    settings: AppSettings,
    *,
    require_release_acceptance: bool = True,
) -> tuple[CheckResult, ...]:
    """Audit the complete coarse-to-fine runtime without touching hardware."""

    results = list(run_supervised_scan_readiness(settings, mode="bootstrap"))
    missing: list[str] = []
    if settings.proxy_model.estimated_thickness_m is None:
        missing.append("proxy_model.estimated_thickness_m")
    if settings.proxy_model.blade_envelope_min_m is None:
        missing.append(
            "proxy_model.blade_envelope_min_m/max_m/minimum_envelope_retained_fraction"
        )
    if not settings.blade_foreground.enabled:
        missing.append("blade_foreground.enabled")
    if settings.thermal.enabled or settings.thermal.driver is not None:
        missing.append("thermal must be disabled")
    envelope_motion_eligible = False
    motion_path = settings.motion_preflight.motion_envelope_acceptance_path
    motion_id = settings.motion_preflight.motion_envelope_acceptance_id
    if motion_path is None or motion_id is None:
        missing.append("motion_preflight.motion_envelope_acceptance")
    else:
        try:
            checker = Cs68PinocchioCollisionChecker.from_es68_resources(
                Es68D435iCollisionResources.packaged_template(),
                joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
                environment_obstacles=settings.collision.obstacles,
                minimum_clearance_m=settings.collision.minimum_clearance_m,
            )
            _load_motion_envelope(settings, checker)
            envelope_motion_eligible = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            missing.append(f"motion envelope invalid: {exc}")
    science_eligible = False
    science_path = settings.science_acceptance.path
    science_id = settings.science_acceptance.acceptance_id
    if require_release_acceptance and (science_path is None or science_id is None):
        missing.append("science_acceptance.path/id")
    elif require_release_acceptance:
        try:
            required_envelope = required_science_test_envelope_for_settings(settings)
            science = read_science_acceptance(science_path)
            science.assert_matches(
                acceptance_id=science_id,
                runtime_contract_sha256=science_runtime_contract_for_settings(settings),
                required_test_envelope=required_envelope,
            )
            science_eligible = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            missing.append(f"science acceptance invalid: {exc}")
    timing = settings.stop_and_capture
    measured = {
        "maximum_perception_cycle_duration_s": timing.maximum_perception_cycle_duration_s,
        "maximum_operator_reposition_interval_s": (timing.maximum_operator_reposition_interval_s),
        "maximum_segment_execution_duration_s": (timing.maximum_segment_execution_duration_s),
        "maximum_schema5_handoff_duration_s": timing.maximum_schema5_handoff_duration_s,
    }
    if require_release_acceptance:
        missing.extend(
            f"stop_and_capture.{name}" for name, value in measured.items() if value is None
        )
    timing_acceptance_eligible = False
    timing_path = timing.runtime_timing_acceptance_path
    timing_id = timing.runtime_timing_acceptance_id
    if require_release_acceptance and (timing_path is None or timing_id is None):
        missing.append("stop_and_capture.runtime_timing_acceptance_path/id")
    elif require_release_acceptance and all(value is not None for value in measured.values()):
        try:
            accepted_timing = read_runtime_timing_acceptance(timing_path)
            accepted_timing.assert_matches(settings=settings, acceptance_id=timing_id)
            timing_acceptance_eligible = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            missing.append(f"runtime timing acceptance invalid: {exc}")
    required_map_age_s: float | None = None
    if require_release_acceptance and all(value is not None for value in measured.values()):
        segment = float(measured["maximum_segment_execution_duration_s"])
        required_map_age_s = (
            segment
            + settings.stop_and_capture.execution_freshness_margin_s
        )
        maximum_map_age_s = settings.occupancy.maximum_map_age_s
        if maximum_map_age_s is not None and maximum_map_age_s <= required_map_age_s:
            missing.append("occupancy.maximum_map_age_s exceeds measured runtime budget")
    results.append(
        CheckResult(
            "unknown_blade_coarse_to_fine",
            CheckLevel.FAIL if missing else CheckLevel.PASS,
            (
                "unknown-blade runtime cannot complete its schema-5 fine handoff"
                if missing
                else "coarse proxy and schema-5 fine foreground policies are configured"
            ),
            {
                "missing": missing,
                "hardware_connection_attempted": False,
                "motion_authorized": False,
                "measured_runtime_budgets_s": measured,
                "required_map_age_exclusive_lower_bound_s": required_map_age_s,
                "thermal_scope": (
                    "geometry-only; thermal requires a separate validated radiometric adapter"
                ),
                "tracking_stop_envelope_motion_eligible": envelope_motion_eligible,
                "bootstrap_controller_stop_motion_eligible": envelope_motion_eligible,
                "geometry_science_acceptance_eligible": science_eligible,
                "runtime_timing_acceptance_eligible": timing_acceptance_eligible,
                "release_scope": (
                    "production" if require_release_acceptance else "experimental"
                ),
                "release_acceptance_bypassed": not require_release_acceptance,
            },
        )
    )
    return tuple(results)


def require_unknown_blade_runtime_ready(
    settings: AppSettings,
    *,
    require_release_acceptance: bool = True,
) -> tuple[CheckResult, ...]:
    results = unknown_blade_runtime_readiness(
        settings,
        require_release_acceptance=require_release_acceptance,
    )
    failures = tuple(item for item in results if item.level is CheckLevel.FAIL)
    if failures:
        summary = "; ".join(f"{item.name}: {item.message}" for item in failures)
        raise UnknownBladeRuntimeError(
            "Offline readiness failed before any hardware connection: " + summary
        )
    return results


def _bootstrap_foreground_config(settings: AppSettings) -> BootstrapForegroundConfig:
    return BootstrapForegroundConfig(**settings.bootstrap_foreground.model_dump(mode="python"))


def _new_fine_recovery_run_root(experiment_root: str | Path) -> Path:
    """Allocate a name without inspecting or reusing an unbound crashed fine run."""

    return (
        Path(experiment_root).resolve()
        / "runs"
        / f"fine_recovery_{uuid4().hex}"
    )


def _load_motion_envelope(
    settings: AppSettings,
    collision_checker: Cs68PinocchioCollisionChecker,
) -> tuple[StoredMotionEnvelopeAcceptance, str]:
    path = settings.motion_preflight.motion_envelope_acceptance_path
    acceptance_id = settings.motion_preflight.motion_envelope_acceptance_id
    if path is None or acceptance_id is None:
        raise UnknownBladeRuntimeError("Motion-envelope acceptance is not configured")
    try:
        control_hash = motion_control_contract_for_settings(settings)
        acceptance = read_motion_envelope_acceptance(path)
        acceptance.assert_matches(
            acceptance_id=acceptance_id,
            robot_geometry_hash=collision_checker.robot_geometry_hash,
            motion_model_contract_hash=collision_checker.motion_model_contract_hash,
            motion_control_contract_hash=control_hash,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise UnknownBladeRuntimeError(f"Motion-envelope acceptance is invalid: {exc}") from exc
    return acceptance, control_hash


def _finalize_production_runtime(
    runtime: UnknownBladeSupervisedRuntime,
    arm: EliteArm,
    settings: AppSettings,
    motion_envelope: StoredMotionEnvelopeAcceptance,
) -> None:
    """Always attempt the physical stop even if outer supervision stop fails."""

    stop_errors: list[BaseException] = []
    try:
        snapshot = runtime.snapshot
        runner_status = getattr(snapshot, "runner_status", None)
        if (
            snapshot.phase is not UnknownBladeRuntimePhase.COMPLETE
            and getattr(runner_status, "phase", None) != "failed"
        ):
            runtime.request_stop("production runtime context closed")
    except BaseException as exc:
        stop_errors.append(exc)
    try:
        arm.stop()
    except BaseException as exc:
        stop_errors.append(exc)
    else:
        try:
            wait_until_settled(
                arm,
                None,
                settle_time_s=settings.robot.settle_time_s,
                timeout_s=settings.stop_and_capture.settle_timeout_s,
                poll_period_s=settings.stop_and_capture.settle_poll_period_s,
                max_joint_delta_rad=settings.acquisition.max_joint_delta_rad,
                max_tcp_translation_delta_m=(
                    settings.acquisition.max_tcp_translation_delta_m
                ),
                max_tcp_rotation_delta_rad=settings.acquisition.max_tcp_rotation_delta_rad,
                goal_tolerance_rad=settings.stop_and_capture.maximum_goal_joint_error_rad,
                maximum_robot_state_staleness_s=min(
                    settings.stop_and_capture.maximum_robot_state_staleness_s,
                    motion_envelope.maximum_feedback_interval_s,
                ),
                maximum_stopped_actual_joint_velocity_rad_s=(
                    motion_envelope.maximum_stopped_actual_joint_velocity_rad_s
                ),
                maximum_stopped_target_joint_velocity_rad_s=(
                    motion_envelope.maximum_stopped_target_joint_velocity_rad_s
                ),
                maximum_stopped_actual_tcp_linear_velocity_m_s=(
                    motion_envelope.maximum_stopped_actual_tcp_linear_velocity_m_s
                ),
                maximum_stopped_actual_tcp_angular_velocity_rad_s=(
                    motion_envelope.maximum_stopped_actual_tcp_angular_velocity_rad_s
                ),
                maximum_stopped_target_tcp_linear_velocity_m_s=(
                    motion_envelope.maximum_stopped_target_tcp_linear_velocity_m_s
                ),
                maximum_stopped_target_tcp_angular_velocity_rad_s=(
                    motion_envelope.maximum_stopped_target_tcp_angular_velocity_rad_s
                ),
            )
        except BaseException as exc:
            stop_errors.append(exc)
    if len(stop_errors) == 1:
        raise stop_errors[0]
    if stop_errors:
        grouped = BaseExceptionGroup(
            "Production runtime cleanup had multiple stop failures",
            stop_errors,
        )
        details = "; ".join(_failure_summary(exc) for exc in stop_errors)
        raise UnknownBladeRuntimeError(
            f"production runtime cleanup stop failures: {details}"
        ) from grouped


def _failure_summary(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        nested = "; ".join(_failure_summary(item) for item in exc.exceptions)
        return f"{type(exc).__name__}[{nested}]"
    return f"{type(exc).__name__}: {exc}"


def _raise_combined_runtime_failure(
    primary: BaseException,
    cleanup: BaseException,
) -> None:
    grouped = BaseExceptionGroup(
        "Unknown-blade runtime and cleanup both failed",
        [primary, cleanup],
    )
    raise UnknownBladeRuntimeError(
        "runtime failure: "
        f"{_failure_summary(primary)}; cleanup failure: {_failure_summary(cleanup)}"
    ) from grouped


@contextmanager
def open_production_unknown_blade_runtime(
    settings: AppSettings,
    *,
    output_root: str | Path,
    operator_id: str,
    run_id: str | None = None,
    placement_id: str | None = None,
    resume: bool = False,
    experimental: bool = False,
) -> Iterator[UnknownBladeSupervisedRuntime | CompletedUnknownBladeRuntime]:
    """Open the real ES68+D435i runtime after a complete offline gate.

    Construction is ordered intentionally: every read-only software/asset check and
    output-directory reservation happens before ``EliteArm.connect`` or
    ``RealSenseD435i.open``.  Any later exception stops/releases already opened
    devices and is propagated; there is no degraded runtime.
    """

    operator = validate_stop_scan_run_id(operator_id)
    requested_placement_id = (
        validate_stop_scan_run_id(placement_id)
        if placement_id is not None
        else None
    )
    root = Path(output_root).resolve()
    if experimental and resume:
        raise UnknownBladeRuntimeError("Experimental unknown-blade runs cannot be resumed")
    resume_plan: UnknownBladeResumePlan | None = None
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(
                f"Unknown-blade resume root does not exist: {root}"
            )
        resume_plan = load_unknown_blade_resume_plan(root)
        identity = validate_stop_scan_run_id(run_id or resume_plan.experiment_id)
        if identity != resume_plan.experiment_id:
            raise UnknownBladeRuntimeError(
                "Requested run ID differs from the experiment handoff authority"
            )
        if (
            requested_placement_id is not None
            and requested_placement_id != resume_plan.placement_id
        ):
            raise UnknownBladeRuntimeError(
                "Requested placement ID differs from the experiment handoff authority"
            )
        bound_placement_id = resume_plan.placement_id
    else:
        identity = validate_stop_scan_run_id(
            run_id or f"unknown-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        if root.exists():
            raise FileExistsError(
                f"Unknown-blade experiment root already exists: {root}"
            )
        bound_placement_id = requested_placement_id
    if resume_plan is not None and resume_plan.phase is UnknownBladeResumePhase.COMPLETE:
        if (
            resume_plan.science_authority is None
            or resume_plan.runtime_timing_authority is None
        ):
            raise UnknownBladeRuntimeError(
                "Production COMPLETE resume requires science and runtime timing authorities; "
                "legacy chains remain available only through the read-only audit API"
            )
        science_authority = load_science_acceptance_authority(settings)
        if resume_plan.science_authority != science_authority:
            raise UnknownBladeRuntimeError(
                "Resume science authority differs from the experiment INIT authority"
            )
        timing_authority = load_runtime_timing_acceptance_authority(settings)
        if resume_plan.runtime_timing_authority != timing_authority:
            raise UnknownBladeRuntimeError(
                "Resume timing authority differs from the experiment INIT authority"
            )
        yield CompletedUnknownBladeRuntime(resume_plan)
        return
    if experimental:
        require_unknown_blade_runtime_ready(
            settings,
            require_release_acceptance=False,
        )
    else:
        require_unknown_blade_runtime_ready(settings)
    # The exact accepted science authority is loaded and the current executable,
    # dependency, model, calibration and policy contract is recomputed before any
    # hardware object is constructed.  Resume may inherit only this exact authority.
    science_authority = (
        None if experimental else load_science_acceptance_authority(settings)
    )
    timing_authority = (
        None if experimental else load_runtime_timing_acceptance_authority(settings)
    )
    science_authority_settings = settings if science_authority is not None else None
    if resume_plan is not None and resume_plan.science_authority != science_authority:
        raise UnknownBladeRuntimeError(
            "Resume science authority differs from the experiment INIT authority"
        )
    if resume_plan is not None and resume_plan.runtime_timing_authority != timing_authority:
        raise UnknownBladeRuntimeError(
            "Resume timing authority differs from the experiment INIT authority"
        )

    fine_settings = settings.model_copy(deep=True)
    coarse_settings = settings.model_copy(
        update={"blade_foreground": settings.blade_foreground.model_copy(update={"enabled": False})}
    )
    hand_eye = load_hand_eye_calibration(settings.hand_eye)
    hand_eye.require_flange_primary()
    resources = Es68D435iCollisionResources.packaged_template()
    collision_checker = Cs68PinocchioCollisionChecker.from_es68_resources(
        resources,
        joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        environment_obstacles=settings.collision.obstacles,
        minimum_clearance_m=settings.collision.minimum_clearance_m,
    )
    live_collision_geometry = LiveCollisionGeometry.from_active_resources(
        resources,
        joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        expected_model_id=str(collision_checker.collision_model_id),
        expected_collision_model_hash=str(collision_checker.collision_model_hash),
        expected_robot_geometry_hash=str(collision_checker.robot_geometry_hash),
    )
    motion_envelope, motion_control_hash = _load_motion_envelope(
        settings,
        collision_checker,
    )
    renderer = Es68D435iRobotDepthRenderer.from_active_resources(
        resources,
        joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
    )
    display_kinematics = Es68KinematicModel.from_resources(
        joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad
    )
    bounds_min = settings.occupancy.workspace_bounds_min_m
    bounds_max = settings.occupancy.workspace_bounds_max_m
    model_path = settings.kinematics.model_path
    if bounds_min is None or bounds_max is None or model_path is None:
        raise UnknownBladeRuntimeError("Readiness accepted incomplete runtime geometry")
    # Model construction can dominate the first inference and can expose checkpoint
    # incompatibilities that a path/dependency doctor cannot.  Complete it before
    # reserving the run root or opening either hardware endpoint.
    backend = FoundationStereoBackend(settings.foundation_stereo)
    backend.prepare()
    # Reserve the immutable experiment root only after every offline asset has
    # independently loaded, but still before either hardware endpoint is opened.
    if not resume:
        root.mkdir(parents=True, exist_ok=False)

    with ExitStack() as stack:
        arm = EliteArm(settings.robot)
        # Bootstrap is strictly read-only/unpowered.  The reverse driver is created
        # lazily by the capability-gated enable path only after permit consumption.
        arm.connect(with_driver=False)
        stack.callback(arm.release)
        camera = RealSenseD435i(settings.realsense)
        camera.open()
        stack.callback(camera.close)
        thermal = NullThermalCamera()
        state = arm.read_state()
        reachability = EliteCs68IkChecker(
            load_cs68_kinematics(model_path),
            hand_eye,
            state.joint_positions_rad,
            settings.kinematics,
        )
        coarse_session = CoarseScienceSession(
            settings=coarse_settings,
            hand_eye=hand_eye,
            reachability_checker=reachability,
            source_kinematics=model_path,
            output_root=root / "coarse_science",
            foreground_config=_bootstrap_foreground_config(settings),
            policy=CoarseSciencePolicy(
                **settings.coarse_science.model_dump(mode="python")
            ),
            recovered_generation=(
                resume_plan.coarse_generation_path
                if resume_plan is not None
                else None
            ),
        )
        coarse_adapter = CoarseSessionNextViewAdapter(coarse_session)
        acquirer = SynchronizedAcquirer(
            arm,
            camera,
            thermal,
            settings.acquisition,
            require_thermal=False,
        )

        def robot_state_sampler_factory() -> EliteRtsiProcessSampler:
            return EliteRtsiProcessSampler(
                settings.robot,
                evidence_period_s=settings.stop_and_capture.settle_poll_period_s,
            )

        coarse_engine = FoundationStereoOccupancyCycleEngine(
            settings=coarse_settings,
            acquirer=acquirer,
            state_source=arm,
            backend=backend,
            hand_eye=hand_eye,
            renderer=renderer,
            output_root=root / "perception" / "coarse",
            coarse_science_preparer=coarse_session.prepare_engine_cycle,
            coarse_science_preflighter=coarse_session.preflight_engine_cycle,
            science_authority=science_authority,
            science_authority_settings=science_authority_settings,
            robot_state_sampler_factory=robot_state_sampler_factory,
        )
        publisher = OccupancyGenerationPublisher()

        def assert_exact_map_ready(result: PerceptionCycleResult) -> None:
            generation = publisher.current
            expected_binding = OccupancyBinding.from_mapping(result.stored_occupancy)
            if generation.snapshot.map_state is not OccupancyMapState.MAP_READY:
                raise UnknownBladeRuntimeError(
                    "Coarse schema promotion requires the current generation to be MAP_READY"
                )
            if (
                generation.artifact_path != result.occupancy_mapping_path.resolve()
                or generation.binding != expected_binding
                or generation.inference_stationarity_path
                != result.inference_stationarity_path.resolve()
                or generation.inference_stationarity_sha256
                != result.inference_stationarity_sha256
            ):
                raise UnknownBladeRuntimeError(
                    "Coarse schema promotion result differs from the published MAP_READY authority"
                )
        safety_factory = GuardedSegmentSafetyFactory(
            arm,
            collision_checker,
            publisher,
            settings.motion_preflight,
            settings.occupancy,
            settings.stop_and_capture,
            motion_envelope,
            motion_control_hash,
        )
        layout = LiveSupervisionLayout(
            model_id=str(collision_checker.collision_model_id),
            occupancy_bounds_min_m=bounds_min,
            occupancy_bounds_max_m=bounds_max,
            occupancy_voxel_size_m=settings.occupancy.voxel_size_m,
        )
        timeline_root = root / "live_timeline"
        coarse_bridge = LiveSupervisionBridge(
            timeline_root,
            layout=layout,
            kinematics=display_kinematics,
            collision_geometry=live_collision_geometry,
        )
        coarse_runner: SupervisedExperimentRunner | None = None
        if resume_plan is None or resume_plan.phase is UnknownBladeResumePhase.COARSE:
            coarse_runner_kwargs = {
                "config": settings.stop_and_capture,
                "acquisition_config": settings.acquisition,
                "robot_config": settings.robot,
                "motion_config": settings.motion_preflight,
                "occupancy_config": settings.occupancy,
                "robot": arm,
                "perception": coarse_engine,
                "selector": coarse_adapter,
                "safety_factory": safety_factory,
                "publisher": publisher,
                "motion_executor": GuardedCoordinatorMotionExecutor(),
                "status_callbacks": (coarse_bridge,),
                "event_callbacks": (coarse_bridge.observe_event,),
                "perception_callbacks": (
                    coarse_adapter.observe_perception,
                    coarse_bridge.observe_perception,
                ),
                "prepared_segment_callbacks": (
                    coarse_bridge.observe_prepared_segment,
                ),
            }
            if resume_plan is None:
                coarse_runner = SupervisedExperimentRunner.create(
                    run_root=root / "runs" / "coarse",
                    run_id=identity,
                    **coarse_runner_kwargs,
                )
            else:
                coarse_runner = SupervisedExperimentRunner.resume(
                    run_root=resume_plan.coarse_run_root,
                    **coarse_runner_kwargs,
                )
        experiment_handoff = (
            UnknownBladeExperimentWriter.resume(root / "experiment_handoff")
            if resume_plan is not None
            else UnknownBladeExperimentWriter.create(
                root / "experiment_handoff",
                experiment_id=identity,
                coarse_run_root=coarse_runner.status.run_root,  # type: ignore[union-attr]
                coarse_run_id=coarse_runner.status.run_id,  # type: ignore[union-attr]
                placement_id=bound_placement_id,
                science_authority=science_authority,
                runtime_timing_authority=timing_authority,
                production=not experimental,
            )
        )
        fine_checkpoint_recorder = _FineCheckpointRecorder(
            experiment_handoff,
            initial_coverage=(
                resume_plan.accepted_fine_coverage_path
                if resume_plan is not None
                else None
            ),
            initial_event_count=(
                resume_plan.accepted_fine_event_count
                if resume_plan is not None
                else None
            ),
            initial_event_sha256=(
                resume_plan.accepted_fine_event_sha256
                if resume_plan is not None
                else None
            ),
            verify_chain=lambda: read_unknown_blade_experiment(
                experiment_handoff.root
            ),
        )

        def build_fine_runner(
            reference: Path,
            *,
            accepted_coverage: Path | None,
            resume_run_root: Path | None,
            fork_live_sources: bool,
            create_run_root: Path | None = None,
        ) -> SupervisedExperimentRunner:
            effective_run_root = (
                resume_run_root
                or create_run_root
                or (root / "runs" / "fine")
            ).resolve()
            fine_engine = (
                coarse_engine.fork_for_fine_science(
                    settings=fine_settings,
                    reference_coarse_model=reference,
                    output_root=root / "perception" / "fine",
                    replace_latest_source_on_first_capture=True,
                )
                if fork_live_sources
                else FoundationStereoOccupancyCycleEngine(
                    settings=fine_settings,
                    acquirer=acquirer,
                    state_source=arm,
                    backend=backend,
                    hand_eye=hand_eye,
                    renderer=renderer,
                    output_root=root / "perception" / "fine",
                    reference_coarse_model=reference,
                    accepted_coverage_path=accepted_coverage,
                    science_authority=science_authority,
                    science_authority_settings=science_authority_settings,
                    robot_state_sampler_factory=robot_state_sampler_factory,
                )
            )
            selector = BladeCoverageNextViewSelector.from_settings(
                fine_settings,
                hand_eye,
                reference_coarse_model=reference,
                science_authority=science_authority,
                experimental=experimental,
            )
            # Coarse and fine coordinators are distinct motion authorities.  Fine
            # may reuse verified perception sources, but it must publish its own
            # first MAP_READY generation before any fine preflight.
            fine_publisher = OccupancyGenerationPublisher()
            fine_safety_factory = GuardedSegmentSafetyFactory(
                arm,
                collision_checker,
                fine_publisher,
                settings.motion_preflight,
                settings.occupancy,
                settings.stop_and_capture,
                motion_envelope,
                motion_control_hash,
            )
            coarse_bridge.begin_new_event_stream(run_id=identity)
            runner_kwargs = {
                "config": settings.stop_and_capture,
                "acquisition_config": settings.acquisition,
                "robot_config": settings.robot,
                "motion_config": settings.motion_preflight,
                "occupancy_config": settings.occupancy,
                "robot": arm,
                "perception": fine_engine,
                "selector": selector,
                "safety_factory": fine_safety_factory,
                "publisher": fine_publisher,
                "motion_executor": GuardedCoordinatorMotionExecutor(),
                "status_callbacks": (coarse_bridge,),
                "event_callbacks": (coarse_bridge.observe_event,),
                "perception_callbacks": (
                    lambda result: fine_checkpoint_recorder.record(
                        result,
                        fine_run_root=effective_run_root,
                    ),
                    coarse_bridge.observe_perception,
                ),
                "prepared_segment_callbacks": (
                    coarse_bridge.observe_prepared_segment,
                ),
            }
            if resume_run_root is not None:
                return SupervisedExperimentRunner.resume(
                    run_root=resume_run_root,
                    **runner_kwargs,
                )
            return SupervisedExperimentRunner.create(
                run_root=effective_run_root,
                run_id=identity,
                **runner_kwargs,
            )

        def create_fine_runner(reference: Path) -> SupervisedExperimentRunner:
            return build_fine_runner(
                reference,
                accepted_coverage=None,
                resume_run_root=None,
                fork_live_sources=True,
                create_run_root=None,
            )

        recovered_fine_runner: SupervisedExperimentRunner | None = None
        if resume_plan is not None and resume_plan.phase in {
            UnknownBladeResumePhase.PREPARED,
            UnknownBladeResumePhase.FINE,
        }:
            reference = resume_plan.reference_coarse_model_path
            if reference is None:
                raise UnknownBladeRuntimeError("Fine resume plan lacks its reference")
            recovered_fine_runner = build_fine_runner(
                reference,
                accepted_coverage=resume_plan.accepted_fine_coverage_path,
                resume_run_root=(
                    resume_plan.fine_run_root
                    if resume_plan.phase is UnknownBladeResumePhase.FINE
                    else None
                ),
                fork_live_sources=False,
                create_run_root=(
                    _new_fine_recovery_run_root(root)
                    if resume_plan.phase is UnknownBladeResumePhase.PREPARED
                    else None
                ),
            )
            if coarse_runner is None:
                coarse_runner = recovered_fine_runner

        def establish_initial_stop() -> BootstrapSafeStateEvidence:
            stop_generation = arm.establish_bootstrap_controller_stop()
            return wait_until_bootstrap_safe_state(
                arm,
                expected_stop_generation=stop_generation,
                settle_time_s=settings.robot.settle_time_s,
                timeout_s=settings.stop_and_capture.settle_timeout_s,
                poll_period_s=settings.stop_and_capture.settle_poll_period_s,
                max_joint_delta_rad=settings.acquisition.max_joint_delta_rad,
                max_tcp_translation_delta_m=(settings.acquisition.max_tcp_translation_delta_m),
                max_tcp_rotation_delta_rad=(settings.acquisition.max_tcp_rotation_delta_rad),
                maximum_robot_state_staleness_s=min(
                    settings.stop_and_capture.maximum_robot_state_staleness_s,
                    motion_envelope.maximum_feedback_interval_s,
                ),
                maximum_stopped_actual_joint_velocity_rad_s=(
                    motion_envelope.maximum_stopped_actual_joint_velocity_rad_s
                ),
                maximum_stopped_target_joint_velocity_rad_s=(
                    motion_envelope.maximum_stopped_target_joint_velocity_rad_s
                ),
                maximum_stopped_actual_tcp_linear_velocity_m_s=(
                    motion_envelope.maximum_stopped_actual_tcp_linear_velocity_m_s
                ),
                maximum_stopped_actual_tcp_angular_velocity_rad_s=(
                    motion_envelope.maximum_stopped_actual_tcp_angular_velocity_rad_s
                ),
                maximum_stopped_target_tcp_linear_velocity_m_s=(
                    motion_envelope.maximum_stopped_target_tcp_linear_velocity_m_s
                ),
                maximum_stopped_target_tcp_angular_velocity_rad_s=(
                    motion_envelope.maximum_stopped_target_tcp_angular_velocity_rad_s
                ),
            )

        if coarse_runner is None:
            raise UnknownBladeRuntimeError("Runtime composition produced no active runner")
        runtime = UnknownBladeSupervisedRuntime(
            coarse_runner=coarse_runner,
            coarse_adapter=coarse_adapter,
            create_fine_runner=create_fine_runner,
            operator_id=operator,
            minimum_operator_bootstrap_views=settings.occupancy.minimum_source_views,
            timeline_root=timeline_root,
            establish_initial_stop=establish_initial_stop,
            assert_exact_map_ready=assert_exact_map_ready,
            experiment_handoff=experiment_handoff,
            resume_phase=(resume_plan.phase if resume_plan is not None else None),
            recovered_reference=(
                resume_plan.reference_coarse_model_path
                if resume_plan is not None
                else None
            ),
            recovered_fine_runner=recovered_fine_runner,
            science_authority=science_authority,
            science_settings=science_authority_settings,
            runtime_timing_authority=timing_authority,
            maximum_schema5_handoff_duration_s=(
                settings.stop_and_capture.maximum_schema5_handoff_duration_s
            ),
            experimental=experimental,
        )
        try:
            yield runtime
        except BaseException as primary:
            try:
                _finalize_production_runtime(runtime, arm, settings, motion_envelope)
            except BaseException as cleanup:
                if isinstance(primary, Exception) and isinstance(cleanup, Exception):
                    _raise_combined_runtime_failure(primary, cleanup)
                raise primary from cleanup
            raise
        else:
            _finalize_production_runtime(runtime, arm, settings, motion_envelope)


def _read_hard_roi_seed(path: str | Path) -> BootstrapSeed:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    vertices: object = None
    if isinstance(payload, dict):
        vertices = payload.get("vertices_uv")
        if vertices is None:
            shapes = payload.get("shapes")
            if isinstance(shapes, list):
                matches = [
                    item
                    for item in shapes
                    if isinstance(item, dict)
                    and item.get("label") == "blade"
                    and item.get("shape_type") == "polygon"
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "annotation must contain exactly one polygon labelled 'blade'"
                    )
                vertices = matches[0].get("points")
    elif isinstance(payload, list):
        vertices = payload
    if not isinstance(vertices, list):
        raise ValueError(
            "annotation must be vertices_uv JSON or an X-AnyLabeling blade polygon"
        )
    return BootstrapSeed.polygon(vertices, mode="hard_roi")


def run_unknown_blade_operator_console(
    runtime: UnknownBladeSupervisedRuntime | CompletedUnknownBladeRuntime,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    initial_bootstrap_seed: BootstrapSeed | None = None,
    initial_operator_side: BladeSide | None = None,
) -> int:
    """Run the blocking operator console; return zero only on completion/stop."""

    initial = runtime.start()
    if initial.phase is UnknownBladeRuntimePhase.COMPLETE:
        output_fn("Experiment chain is already sealed COMPLETE; no hardware was opened.")
        return 0
    output_fn(
        "Read-only GUI (another terminal): uv run bbf supervise replay "
        f"--snapshot {runtime.snapshot.timeline_root} --follow"
    )

    def formal_frame_seed_provider(
        _captured: CapturedStopScanView,
        image_path: Path,
    ) -> BootstrapSeed:
        default_json = image_path.with_suffix(".json")
        output_fn(
            "Formal bootstrap frame captured. Annotate exactly one polygon labelled "
            f"'blade' on: {image_path}"
        )
        output_fn(
            "Save the annotation without moving robot/camera/blade; default JSON: "
            f"{default_json}"
        )
        while True:
            raw = input_fn(
                "Enter the hard_roi annotation JSON path (Enter uses default, q aborts): "
            ).strip()
            if raw.lower() == "q":
                raise UnknownBladeRuntimeError(
                    "operator aborted while the formal bootstrap frame awaited annotation"
                )
            candidate = default_json if not raw else Path(raw)
            try:
                return _read_hard_roi_seed(candidate)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                output_fn(f"Annotation rejected: {type(exc).__name__}: {exc}")
    while True:
        snapshot = runtime.snapshot
        status = snapshot.runner_status
        output_fn(
            f"phase={snapshot.phase.value} runner={status.phase} "
            f"disposition={status.disposition.value} cycle={status.cycle_index}"
        )
        if snapshot.phase is UnknownBladeRuntimePhase.COMPLETE:
            output_fn("Scan completed; no further motion is authorized.")
            return 0
        if snapshot.phase is UnknownBladeRuntimePhase.BLOCKED:
            reasons = status.blocking_reasons or (
                (snapshot.blocking_reason,) if snapshot.blocking_reason else ()
            )
            output_fn("BLOCKED: " + "; ".join(reasons))
            return 1
        if snapshot.phase is UnknownBladeRuntimePhase.STOPPED:
            return 0
        if (
            status.disposition is ExperimentDisposition.NEEDS_CAPTURE
            and status.expected_capture_view_id is None
        ):
            command = input_fn(
                "Manually reposition the stopped robot. Enter c, c front, c back, "
                "or q to stop: "
            )
            parts = command.strip().lower().split()
            if parts == ["q"]:
                runtime.request_stop("operator requested stop")
            elif parts and parts[0] == "c" and len(parts) <= 2:
                explicit_side: BladeSide | None = None
                if len(parts) == 2:
                    try:
                        explicit_side = BladeSide(parts[1])
                    except ValueError:
                        output_fn("Unknown side: use exactly front or back.")
                        continue
                if snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN:
                    if explicit_side is not None:
                        output_fn("Fine safety replenishment does not accept a side label.")
                        continue
                    runtime.capture_fine_source_replenishment()
                else:
                    first_capture = snapshot.operator_bootstrap_views == 0
                    runtime.capture_operator_view(
                        seed=initial_bootstrap_seed if first_capture else None,
                        seed_provider=formal_frame_seed_provider,
                        operator_side=(
                            explicit_side
                            if explicit_side is not None
                            else initial_operator_side if first_capture else None
                        ),
                    )
            else:
                output_fn("No capture: enter exactly c, c front, c back, or q.")
            continue
        if status.disposition is ExperimentDisposition.WAITING_APPROVAL:
            expected = runtime.approval_prompt()
            output_fn(f"Prepared one segment. Exact approval token: {expected}")
            command = input_fn("Paste the exact token, or q to stop: ").strip()
            if command.lower() == "q":
                runtime.request_stop("operator requested stop")
            elif command == expected:
                runtime.execute_exact_approval(command)
            else:
                output_fn("Token mismatch; motion was not requested.")
            continue
        if status.disposition is ExperimentDisposition.READY:
            runtime.advance_to_attention()
            continue
        output_fn("Runtime reached an unsupported attention state and stopped fail-closed.")
        runtime.request_stop(f"unsupported operator disposition: {status.disposition.value}")
        return 1


__all__ = [
    "CoarseSessionNextViewAdapter",
    "UnknownBladeRuntimeError",
    "UnknownBladeRuntimePhase",
    "UnknownBladeRuntimeSnapshot",
    "UnknownBladeSupervisedRuntime",
    "open_production_unknown_blade_runtime",
    "require_unknown_blade_runtime_ready",
    "run_unknown_blade_operator_console",
    "unknown_blade_runtime_readiness",
]
