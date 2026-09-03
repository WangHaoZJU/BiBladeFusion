"""Concrete single-backend perception transaction for the stop-scan coordinator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np

from biblade_fusion.acquisition import SynchronizedAcquirer, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    AppSettings,
    OccupancyConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.devices.robot.base import RobotState, RobotStateSource
from biblade_fusion.diagnostics.performance_timing import (
    activate_performance_timing,
    performance_span,
    try_create_performance_timing,
)
from biblade_fusion.perception.stereo import FoundationStereoBackend
from biblade_fusion.robotics.stationarity import validate_stationary_trace
from biblade_fusion.storage.blade_foreground import read_blade_foreground_mask
from biblade_fusion.storage.inference_stationarity import (
    read_inference_stationarity,
    write_inference_stationarity,
    write_inference_stationarity_trace,
)
from biblade_fusion.storage.occupancy_mapping import read_occupancy_mapping
from biblade_fusion.storage.occupancy_mapping import (
    write_live_occupancy_mapping as write_occupancy_mapping,
)
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.session import SessionWriter
from biblade_fusion.storage.stereo_inference import (
    read_stereo_inference,
    verify_stereo_inference_source,
    write_stereo_inference,
)
from biblade_fusion.storage.surface_coverage import read_surface_coverage_generation
from biblade_fusion.workflows.fine_science import (
    PreparedFineScienceAssets,
    prepare_fine_science_assets,
    validate_fine_science_startup,
)
from biblade_fusion.workflows.occupancy_mapping import (
    OccupancyFrameUpdate,
    PreparedOccupancyFrame,
    RobotDepthRenderer,
    integrate_foundation_stereo_occupancy,
    integrate_prepared_foundation_stereo_occupancy,
    prepare_foundation_stereo_occupancy_frame,
)
from biblade_fusion.workflows.stereo_inference import (
    StereoInferenceObservation,
    infer_rectified_stereo,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    CapturePurpose,
    OccupancyBinding,
    PerceptionCycleResult,
)


class FoundationStereoCycleError(RuntimeError):
    """A stopped FoundationStereo asset transaction could not be committed."""


_ROBOT_STATE_SAMPLER_FIFO_PRIORITY = 10


class CoarseSciencePreparer(Protocol):
    """Prepare one coarse wrapper inside the current stopped transaction."""

    def __call__(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        occupancy_update: OccupancyFrameUpdate,
        occupancy_path: Path,
    ) -> str | Path: ...


class CoarseSciencePreflighter(Protocol):
    """Validate current-frame coarse foreground before occupancy ray integration."""

    def __call__(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        prepared_occupancy: PreparedOccupancyFrame,
    ) -> None: ...


class RobotStateSampler(Protocol):
    """One-shot continuous state trace owned by one perception transaction."""

    @property
    def is_alive(self) -> bool: ...

    @property
    def diagnostics(self) -> dict[str, object]: ...

    def start(self) -> None: ...

    def finish(self) -> tuple[RobotState, ...]: ...

    def cancel(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _VerifiedSource:
    captured: CapturedStopScanView
    stereo: StereoInferenceObservation
    stereo_path: Path
    stereo_metadata_sha256: str
    session_manifest_sha256: str
    session_view_metadata_sha256: str
    camera_center_base_m: tuple[float, float, float]
    camera_axis_base: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _PendingPerceptionCommit:
    """One materialized cycle not yet accepted by the coordinator."""

    key: tuple[str, int]
    cycle_root: Path
    raw_session_path: Path
    raw_session_manifest_sha256: str
    raw_session_view_metadata_sha256: str
    stereo_inference_path: Path
    stereo_metadata_sha256: str
    occupancy_mapping_path: Path
    occupancy_metadata_sha256: str
    occupancy_binding: tuple[object, ...]
    inference_stationarity_path: Path
    inference_stationarity_sha256: str
    sources: tuple[_VerifiedSource, ...]
    updates: tuple[OccupancyFrameUpdate, ...]
    blade_foreground_path: Path | None
    reconstructed_view_path: Path | None
    coverage_path: Path | None
    coverage_metadata_sha256: str | None
    accepted_coverage_path_after_commit: Path | None
    coarse_scan_view_path: Path | None = None
    coarse_scan_metadata_sha256: str | None = None


class FoundationStereoOccupancyCycleEngine:
    """Capture one immutable session, infer once, and rebuild a fresh map window.

    The engine owns no motion interface.  A continuous sampler records read-only robot
    state while the main thread performs FoundationStereo inference.  The coordinator
    validates the returned trace and its capture-state binding before publishing the map.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        acquirer: SynchronizedAcquirer,
        state_source: RobotStateSource,
        backend: FoundationStereoBackend,
        hand_eye: HandEyeCalibration,
        renderer: RobotDepthRenderer,
        output_root: str | Path,
        reference_coarse_model: str | Path | None = None,
        accepted_coverage_path: str | Path | None = None,
        coarse_science_preparer: CoarseSciencePreparer | None = None,
        coarse_science_preflighter: CoarseSciencePreflighter | None = None,
        science_authority: ScienceAcceptanceAuthority | None = None,
        science_authority_settings: AppSettings | None = None,
        robot_state_sampler_factory: Callable[[], RobotStateSampler] | None = None,
        utc_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if settings.stop_and_capture.depth_backend != "foundation_stereo":
            raise ValueError("Stop-scan cycle engine requires FoundationStereo")
        if type(backend) is not FoundationStereoBackend:
            raise ValueError(
                "Stop-scan cycle engine requires the concrete FoundationStereo backend"
            )
        if backend.config != settings.foundation_stereo:
            raise ValueError("FoundationStereo backend configuration differs from AppSettings")
        if not settings.occupancy.enabled:
            raise ValueError("Stop-scan cycle engine requires enabled occupancy mapping")
        if acquirer.robot_state_source is not state_source:
            raise ValueError(
                "Camera brackets and perception sampling must share one robot instance"
            )
        if acquirer.acquisition_config != settings.acquisition:
            raise ValueError(
                "Camera brackets and perception sampling must share one acquisition policy"
            )
        calibration_path = settings.realsense.stereo_calibration_path
        if calibration_path is None or not calibration_path.resolve().is_file():
            raise ValueError("A user-calibrated stereo asset is required")
        if not hand_eye.source_path.resolve().is_file():
            raise ValueError("A persisted flange-primary hand-eye asset is required")
        hand_eye.require_flange_primary()
        if settings.science_acceptance.path is not None and science_authority is None:
            raise ValueError(
                "A configured science acceptance requires its preloaded runtime authority"
            )
        if (science_authority is None) != (science_authority_settings is None):
            raise ValueError(
                "Science authority and its authoritative AppSettings must be supplied together"
            )
        if science_authority is not None and science_authority_settings is not None:
            science_authority.assert_current(science_authority_settings)
        reference = (
            Path(reference_coarse_model).resolve() if reference_coarse_model is not None else None
        )
        accepted_coverage = (
            Path(accepted_coverage_path).resolve() if accepted_coverage_path is not None else None
        )
        if settings.blade_foreground.enabled != (reference is not None):
            raise ValueError(
                "Fine science requires both blade_foreground.enabled and a pinned "
                "reference_coarse_model"
            )
        if accepted_coverage is not None and reference is None:
            raise ValueError("An accepted fine-coverage generation requires fine science")
        if reference is not None and coarse_science_preparer is not None:
            raise ValueError("A cycle cannot prepare coarse and fine science together")
        if (coarse_science_preparer is None) != (coarse_science_preflighter is None):
            raise ValueError("Coarse science preparation and preflight must be supplied together")
        if reference is not None:
            reference, accepted_coverage = validate_fine_science_startup(
                settings,
                hand_eye,
                reference_coarse_model=reference,
                accepted_coverage_path=accepted_coverage,
            )
        self._settings = settings.model_copy(deep=True)
        self._acquirer = acquirer
        self._state_source = state_source
        self._backend = backend
        self._hand_eye = hand_eye
        self._renderer = renderer
        self._output_root = Path(output_root).resolve()
        self._reference_coarse_model = reference
        self._accepted_coverage_path = accepted_coverage
        self._coarse_science_preparer = coarse_science_preparer
        self._coarse_science_preflighter = coarse_science_preflighter
        self._science_authority = science_authority
        self._science_authority_settings = (
            science_authority_settings.model_copy(deep=True)
            if science_authority_settings is not None
            else None
        )
        self._robot_state_sampler_factory = robot_state_sampler_factory
        self._utc_clock = utc_clock
        self._sources: list[_VerifiedSource] = []
        self._updates: list[OccupancyFrameUpdate] = []
        # A logical (view_id, sequence) becomes occupied only at successful commit.
        # Every physical attempt is written below its own UUID root and is retained
        # even when capture/inference is cancelled or fails.
        self._capture_roots: dict[tuple[str, int], Path] = {}
        self._pending_lock = threading.Lock()
        self._pending_key: tuple[str, int] | None = None
        self._pending_attempt_root: Path | None = None
        self._pending_sampler: RobotStateSampler | None = None
        self._pending_commit: _PendingPerceptionCommit | None = None
        self._poisoned_reason: str | None = None

    @property
    def robot_state_source(self) -> RobotStateSource:
        return self._state_source

    @property
    def acquisition_config(self) -> AcquisitionConfig:
        return self._settings.acquisition.model_copy(deep=True)

    @property
    def occupancy_config(self) -> OccupancyConfig:
        return self._settings.occupancy.model_copy(deep=True)

    @property
    def coordinator_config(self) -> StopAndCaptureConfig:
        return self._settings.stop_and_capture.model_copy(deep=True)

    @property
    def accepted_coverage_path(self) -> Path | None:
        """Return the exact committed fine generation, never an inferred "latest" path."""

        with self._pending_lock:
            return self._accepted_coverage_path

    def _require_science_authority_settings(self) -> AppSettings:
        settings = self._science_authority_settings
        if settings is None:
            raise FoundationStereoCycleError(
                "Science authority lost its authoritative runtime settings"
            )
        return settings

    def fork_for_fine_science(
        self,
        *,
        settings: AppSettings,
        reference_coarse_model: str | Path,
        output_root: str | Path,
        replace_latest_source_on_first_capture: bool = False,
    ) -> FoundationStereoOccupancyCycleEngine:
        """Create a fine-science engine from this engine's committed source window.

        This is a perception-source fork, not a coordinator-state migration.  It
        deliberately carries only independently verified, already committed raw and
        stereo sources.  It does *not* carry an occupancy publication, prepared
        segment, motion permit, stop latch, run event, or fine-coverage generation.
        The new coordinator must start normally and publish a fresh MAP_READY
        generation before it can preflight any motion.

        The supplied settings may differ only in ``blade_foreground``.  This keeps
        every acquisition, calibration, occupancy, stationarity and safety policy
        identical while allowing the coarse engine's disabled reference mask to be
        replaced by the schema-5-guided fine mask.
        """

        if not settings.blade_foreground.enabled:
            raise FoundationStereoCycleError(
                "Fine source-window fork requires blade_foreground.enabled=true"
            )
        comparable = settings.model_copy(
            update={"blade_foreground": self._settings.blade_foreground}
        )
        if comparable != self._settings:
            raise FoundationStereoCycleError(
                "Fine source-window fork changed a non-foreground runtime policy"
            )
        with self._pending_lock:
            if (
                self._pending_key is not None
                or self._pending_attempt_root is not None
                or self._pending_sampler is not None
                or self._pending_commit is not None
            ):
                raise FoundationStereoCycleError(
                    "Cannot fork a perception engine with a pending transaction"
                )
            sources = tuple(self._sources)
            updates = tuple(self._updates)
        if len(updates) != len(sources):
            raise FoundationStereoCycleError(
                "Committed occupancy update prefix differs from its source window"
            )
        if not sources:
            raise FoundationStereoCycleError(
                "Cannot fork fine science before a committed coarse source exists"
            )
        if (
            replace_latest_source_on_first_capture
            and len(sources) < settings.occupancy.minimum_source_views
        ):
            raise FoundationStereoCycleError(
                "Fine transition cannot replace the latest source before a MAP_READY "
                "coarse source window exists"
            )
        verified_sources: list[_VerifiedSource] = []
        for source in sources:
            if (
                _sha256(source.stereo_path / "metadata.json") != source.stereo_metadata_sha256
                or _sha256(source.captured.raw_session_path / "manifest.json")
                != source.session_manifest_sha256
                or _single_view_metadata_hash(
                    source.captured.raw_session_path,
                    source.captured.bundle,
                )
                != source.session_view_metadata_sha256
            ):
                raise FoundationStereoCycleError(
                    "Committed coarse source evidence changed before fine handoff"
                )
            stored = read_stereo_inference(source.stereo_path)
            verify_stereo_inference_source(
                stored,
                expected_session=source.captured.raw_session_path,
            )
            verified_sources.append(source)

        forked = type(self)(
            settings=settings,
            acquirer=self._acquirer,
            state_source=self._state_source,
            backend=self._backend,
            hand_eye=self._hand_eye,
            renderer=self._renderer,
            output_root=output_root,
            reference_coarse_model=reference_coarse_model,
            accepted_coverage_path=None,
            science_authority=self._science_authority,
            science_authority_settings=self._science_authority_settings,
            robot_state_sampler_factory=self._robot_state_sampler_factory,
            utc_clock=self._utc_clock,
        )
        # The source records are immutable value/evidence bindings.  Copying the
        # list prevents either engine from mutating the other's source window.
        # Source retention is generation-driven, not tied to the motion-authorization
        # clock.  The fine coordinator atomically replaces this source generation
        # only after a later perception cycle is accepted.
        # Production fine activation immediately captures the still-stopped coarse
        # endpoint again to create fine coverage generation zero.  Omit that exact
        # preceding viewpoint so the replacement frame is not rejected as a
        # geometrically duplicate occupancy source.
        forked._sources = (
            verified_sources[:-1]
            if replace_latest_source_on_first_capture
            else verified_sources
        )
        forked._updates = list(
            updates[:-1] if replace_latest_source_on_first_capture else updates
        )
        return forked

    def capture(
        self,
        view_id: str,
        sequence_index: int,
        *,
        purpose: CapturePurpose,
    ) -> CapturedStopScanView:
        """Capture and close exactly one raw session; no inference occurs here."""

        if type(purpose) is not CapturePurpose:
            raise ValueError("Capture purpose must be assigned by the coordinator")

        safe_view = _safe_name(view_id)
        logical_root = self._output_root / "cycles" / (f"{sequence_index:06d}_{safe_view}")
        commit_marker = logical_root / "committed.json"
        attempt_id = uuid4().hex
        cycle_root = (logical_root / f"attempt_{attempt_id}").resolve()
        key = (view_id, sequence_index)
        sampler = (
            self._robot_state_sampler_factory()
            if self._robot_state_sampler_factory is not None
            else _RobotStateSampler(
                self._state_source,
                self._settings.stop_and_capture.settle_poll_period_s,
                prefer_fifo=self._settings.stop_and_capture.enabled,
            )
        )
        with self._pending_lock:
            if self._poisoned_reason is not None:
                raise FoundationStereoCycleError(
                    "Perception engine is fail-closed after sampler cleanup failure: "
                    f"{self._poisoned_reason}"
                )
            if key in self._capture_roots or commit_marker.exists():
                raise FoundationStereoCycleError("Capture identity was already committed")
            if self._pending_sampler is not None or self._pending_commit is not None:
                raise FoundationStereoCycleError(
                    "A prior capture transaction is still awaiting inference or commit"
                )
            self._pending_key = key
            self._pending_attempt_root = cycle_root
            self._pending_sampler = sampler
        writer: SessionWriter | None = None
        try:
            logical_root.mkdir(parents=True, exist_ok=True)
            cycle_root.mkdir()
            writer = SessionWriter.create(
                cycle_root / "raw",
                self._settings,
                label=f"cycle_{sequence_index:06d}_{safe_view}_attempt_{attempt_id}",
            )
            # Sampling starts before camera exposure and remains active across raw
            # persistence, FoundationStereo, map rebuild, and evidence persistence.
            sampler.start()
            bundle = self._acquirer.capture(view_id, sequence_index)
            if (bundle.view_id, bundle.sequence_index) != key:
                raise FoundationStereoCycleError("Acquirer changed the requested capture identity")
            writer.write_bundle(bundle)
            writer.close("completed")
        except BaseException:
            if writer is not None:
                with suppress(BaseException):
                    writer.close("failed")
            with suppress(BaseException):
                self._cancel_pending_sampler_instance(sampler)
            raise
        assert writer is not None  # narrowed by the successful transaction above
        # The session manifest is the persisted capture-time authority consumed by
        # full occupancy semantic verification.  Creating the session before camera
        # exposure keeps that timestamp conservative, while reading it back prevents
        # an independently sampled clock value from diverging by microseconds.
        captured_at = _session_created_at_utc(writer.path)
        return CapturedStopScanView(
            bundle=bundle,
            raw_session_path=writer.path,
            cycle_root=cycle_root,
            captured_at_utc=captured_at,
            purpose=purpose,
        )

    def cancel_pending_capture(
        self,
        captured: CapturedStopScanView | None = None,
    ) -> None:
        """Cancel and join the continuous sampler for an uncommitted transaction."""

        expected = (
            None if captured is None else (captured.bundle.view_id, captured.bundle.sequence_index)
        )
        with self._pending_lock:
            sampler = self._pending_sampler
            pending_key = (
                self._pending_key
                if sampler is not None
                else (self._pending_commit.key if self._pending_commit is not None else None)
            )
            if pending_key is None:
                return
            if expected is not None and expected != pending_key:
                raise FoundationStereoCycleError("Cannot cancel another perception transaction")
            if sampler is None:
                # Inference finished, but the coordinator has not accepted this
                # source window.  Rollback is therefore a metadata-state operation;
                # immutable raw/inference assets remain on disk as rejected evidence.
                self._pending_commit = None
                return
        self._cancel_pending_sampler_instance(sampler)

    def infer_and_update(
        self,
        captured: CapturedStopScanView,
    ) -> PerceptionCycleResult:
        """Infer and stage one cycle while emitting non-authoritative timings."""

        recorder = try_create_performance_timing(
            transaction_kind="foundation_stereo_occupancy_cycle",
            identity={
                "view_id": captured.bundle.view_id,
                "sequence_index": captured.bundle.sequence_index,
                "frame_number": captured.bundle.stereo.frame_number,
                "purpose": captured.purpose.value,
                "attempt_root": str(captured.cycle_root.resolve()),
            },
        )
        if recorder is None:
            return self._infer_and_update_transaction(captured)
        status = "failed"
        error: str | None = None
        try:
            with activate_performance_timing(recorder), performance_span(
                "perception.cycle"
            ):
                result = self._infer_and_update_transaction(captured)
            status = "completed"
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            # This file is deliberately absent from every safety/science source
            # record and hash chain.  Ordinary diagnostic failures cannot replace
            # the transaction result; its small synchronous latency remains visible
            # to the caller's conservative perception-duration gate.
            recorder.write_best_effort(
                captured.cycle_root / "performance_timing.json",
                status=status,
                error=error,
            )

    def _infer_and_update_transaction(
        self,
        captured: CapturedStopScanView,
    ) -> PerceptionCycleResult:
        """Infer FoundationStereo and stage a reverified map candidate."""

        key = (captured.bundle.view_id, captured.bundle.sequence_index)
        with self._pending_lock:
            if (
                self._pending_key != key
                or self._pending_attempt_root != captured.cycle_root
                or self._pending_sampler is None
            ):
                raise FoundationStereoCycleError(
                    "Captured attempt is not the engine's active logical transaction"
                )
        sampler = self._require_pending_sampler(key)
        sampler_finished = False
        try:
            if self._science_authority is not None:
                # This check is intentionally adjacent to the actual backend call.
                # Constructor/readiness validation cannot close a source/checkpoint/
                # calibration TOCTOU window during a long-running experiment.
                self._science_authority.assert_current(
                    self._require_science_authority_settings()
                )
            with performance_span("stereo.backend"):
                observation = infer_rectified_stereo(
                    captured.bundle,
                    self._backend,
                    self._settings.stereo_rectification,
                )
            if self._science_authority is not None:
                self._science_authority.assert_current(
                    self._require_science_authority_settings()
                )
                self._science_authority.assert_inference_observation(observation)
            stereo_path = captured.cycle_root / "stereo_inference"
            with performance_span("stereo.artifact_write"):
                write_stereo_inference(
                    stereo_path,
                    observation,
                    self._settings.foundation_stereo,
                    self._settings.stereo_rectification,
                    source_session=captured.raw_session_path,
                    source_stereo_calibration=(self._settings.realsense.stereo_calibration_path),
                )
            with performance_span("stereo.artifact_readback"):
                stored_stereo = read_stereo_inference(stereo_path)
                verify_stereo_inference_source(
                    stored_stereo,
                    expected_session=captured.raw_session_path,
                )
            if self._science_authority is not None:
                self._science_authority.assert_stereo_artifact(stored_stereo)
                self._science_authority.assert_current(
                    self._require_science_authority_settings()
                )
            source = _VerifiedSource(
                captured=captured,
                stereo=stored_stereo.observation,
                stereo_path=stereo_path.resolve(),
                stereo_metadata_sha256=_sha256(stereo_path / "metadata.json"),
                session_manifest_sha256=_sha256(captured.raw_session_path / "manifest.json"),
                session_view_metadata_sha256=_single_view_metadata_hash(
                    captured.raw_session_path,
                    captured.bundle,
                ),
                **self._camera_view_evidence(captured, stored_stereo.observation),
            )
            prepared_current = None
            if self._coarse_science_preflighter is not None:
                with performance_span("occupancy.current_frame_prepare"):
                    prepared_current = prepare_foundation_stereo_occupancy_frame(
                        source.captured.bundle,
                        source.stereo,
                        self._hand_eye,
                        self._settings.occupancy,
                        self._settings.acquisition,
                        self._renderer,
                        captured_at_utc=source.captured.captured_at_utc,
                        source_stereo_metadata_sha256=source.stereo_metadata_sha256,
                        source_session_manifest_sha256=source.session_manifest_sha256,
                        source_session_view_metadata_sha256=(
                            source.session_view_metadata_sha256
                        ),
                    )
                with performance_span("coarse.foreground_preflight"):
                    self._preflight_coarse_science(
                        captured,
                        stored_stereo.observation,
                        stereo_path,
                        prepared_current,
                    )
            with performance_span("occupancy.source_window_selection"):
                candidates = self._fresh_rebuild_sources(source)
            with performance_span("occupancy.source_window_rebuild"):
                updates = (
                    self._rebuild_updates(candidates)
                    if prepared_current is None
                    else self._rebuild_updates(
                        candidates,
                        prepared_current=prepared_current,
                    )
                )
            occupancy_path = captured.cycle_root / "occupancy_mapping"
            with performance_span("occupancy.artifact_write"):
                written_mapping = write_occupancy_mapping(
                    occupancy_path,
                    updates,
                    self._settings.occupancy,
                    self._settings.acquisition,
                    source_stereo_inferences=[item.stereo_path for item in candidates],
                    source_sessions=[item.captured.raw_session_path for item in candidates],
                    source_hand_eye=self._hand_eye.source_path,
                )
                if isinstance(written_mapping, tuple):
                    occupancy_path, stored_mapping = written_mapping
                else:
                    # Compatibility for injected/test storage adapters that still
                    # implement the historical path-only writer contract.  The
                    # production alias above always returns the already-verified
                    # live mapping and therefore avoids an immediate ray replay.
                    occupancy_path = Path(written_mapping or occupancy_path).resolve()
                    stored_mapping = read_occupancy_mapping(occupancy_path)
            # ``stored_mapping`` is the motion-grade authority produced by this
            # exact live integration transaction.  Subsequent semantic readers in
            # this process use its immutable cache entry instead of replaying every
            # depth ray; a new process still performs the complete disk replay.
            with performance_span("science.fine_assets"):
                science = self._prepare_science_assets(
                    captured,
                    stored_stereo.observation,
                    stereo_path,
                    updates[-1],
                    occupancy_path,
                )
            with performance_span("coarse.scan_view_prepare"):
                coarse_scan_view_path = self._prepare_coarse_science_asset(
                    captured,
                    stored_stereo.observation,
                    stereo_path,
                    updates[-1],
                    occupancy_path,
                )
            with performance_span("stationarity.sampler_finish"):
                independent_trace = _ordered_unique_robot_states(sampler.finish())
            sampler_finished = True
            with performance_span("stationarity.trace_write"):
                write_inference_stationarity_trace(
                    captured.cycle_root / "inference_stationarity_trace.json",
                    view_id=captured.bundle.view_id,
                    sequence_index=captured.bundle.sequence_index,
                    trace=independent_trace,
                    source_session_manifest=captured.raw_session_path / "manifest.json",
                    sampler_diagnostics=sampler.diagnostics,
                )
            capture_states = (
                captured.bundle.robot_state_before,
                captured.bundle.selected_robot_state,
                captured.bundle.robot_state_after,
            )
            with performance_span("stationarity.validation"):
                self._validate_capture_binding(independent_trace, capture_states)
                full_trace = _build_authoritative_stationarity_trace(
                    independent_trace,
                    capture_states,
                )
                stationarity_reference = full_trace[0]
                state_trace = full_trace[1:]
                stationarity = validate_stationary_trace(
                    stationarity_reference,
                    state_trace,
                    max_joint_delta_rad=self._settings.acquisition.max_joint_delta_rad,
                    max_tcp_translation_delta_m=(
                        self._settings.acquisition.max_tcp_translation_delta_m
                    ),
                    max_tcp_rotation_delta_rad=(
                        self._settings.acquisition.max_tcp_rotation_delta_rad
                    ),
                    maximum_robot_state_staleness_s=(
                        self._settings.stop_and_capture.maximum_robot_state_staleness_s
                    ),
                )
            with performance_span("stationarity.authority_write_readback"):
                stored_stationarity = write_inference_stationarity(
                    captured.cycle_root / "inference_stationarity.json",
                    view_id=captured.bundle.view_id,
                    sequence_index=captured.bundle.sequence_index,
                    reference=stationarity_reference,
                    trace=state_trace,
                    evidence=stationarity,
                    source_session_manifest=captured.raw_session_path / "manifest.json",
                    max_joint_delta_rad=self._settings.acquisition.max_joint_delta_rad,
                    max_tcp_translation_delta_m=(
                        self._settings.acquisition.max_tcp_translation_delta_m
                    ),
                    max_tcp_rotation_delta_rad=(
                        self._settings.acquisition.max_tcp_rotation_delta_rad
                    ),
                    maximum_robot_state_staleness_s=(
                        self._settings.stop_and_capture.maximum_robot_state_staleness_s
                    ),
                )
        except BaseException as exc:
            if not sampler_finished:
                try:
                    self._cancel_pending_sampler_instance(sampler)
                except BaseException as sampling_exc:
                    exc.add_note(
                        "Robot-state sampling cleanup also failed: "
                        f"{type(sampling_exc).__name__}: {sampling_exc}"
                    )
            else:
                self._clear_pending_sampler(sampler)
            raise
        result = PerceptionCycleResult(
            bundle=captured.bundle,
            raw_session_path=captured.raw_session_path,
            stereo_inference_path=stereo_path,
            occupancy_mapping_path=occupancy_path,
            stored_occupancy=stored_mapping,
            stationarity_reference=stationarity_reference,
            inference_robot_state_trace=state_trace,
            inference_stationarity=stationarity,
            inference_stationarity_path=stored_stationarity.path,
            inference_stationarity_sha256=stored_stationarity.file_sha256,
            purpose=captured.purpose,
            depth_backend="foundation_stereo",
            blade_foreground_path=science.blade_foreground_path,
            reconstructed_view_path=science.reconstructed_view_path,
            coverage_path=science.coverage_path,
            coarse_scan_view_path=coarse_scan_view_path,
        )
        # Preparing a valid asset does not make it a source for a later map.  Only
        # the coordinator can accept it after independent readback and stop-latch
        # checks; the publisher transaction exposes the generation only after this
        # engine has committed the matching source window.
        self._stage_pending_commit(
            sampler,
            _PendingPerceptionCommit(
                key=key,
                cycle_root=captured.cycle_root,
                raw_session_path=captured.raw_session_path,
                raw_session_manifest_sha256=source.session_manifest_sha256,
                raw_session_view_metadata_sha256=(source.session_view_metadata_sha256),
                stereo_inference_path=result.stereo_inference_path,
                stereo_metadata_sha256=source.stereo_metadata_sha256,
                occupancy_mapping_path=result.occupancy_mapping_path,
                occupancy_metadata_sha256=_sha256(result.occupancy_mapping_path / "metadata.json"),
                occupancy_binding=OccupancyBinding.from_mapping(stored_mapping).tuple,
                inference_stationarity_path=result.inference_stationarity_path,
                inference_stationarity_sha256=(result.inference_stationarity_sha256),
                sources=candidates,
                updates=updates,
                blade_foreground_path=result.blade_foreground_path,
                reconstructed_view_path=result.reconstructed_view_path,
                coverage_path=result.coverage_path,
                coverage_metadata_sha256=(
                    _sha256(result.coverage_path / "coverage.json")
                    if result.coverage_path is not None
                    else None
                ),
                accepted_coverage_path_after_commit=(
                    science.coverage_path
                    if science.advances_coverage
                    else self._accepted_coverage_path
                ),
                coarse_scan_view_path=result.coarse_scan_view_path,
                coarse_scan_metadata_sha256=(
                    _sha256(result.coarse_scan_view_path / "metadata.json")
                    if result.coarse_scan_view_path is not None
                    else None
                ),
            ),
        )
        return result

    def commit_perception_cycle(
        self,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
        *,
        before_commit: Callable[[str], None] = lambda _stage: None,
    ) -> None:
        """Commit exactly one coordinator-accepted perception transaction."""

        key = (captured.bundle.view_id, captured.bundle.sequence_index)
        with self._pending_lock:
            before_commit("before_pending_authority_validation")
            pending = self._pending_commit
            if (
                pending is None
                or pending.key != key
                or pending.cycle_root != captured.cycle_root
                or pending.raw_session_path != captured.raw_session_path
                or pending.raw_session_path != result.raw_session_path
                or pending.stereo_inference_path != result.stereo_inference_path
                or pending.occupancy_mapping_path != result.occupancy_mapping_path
                or pending.inference_stationarity_path != result.inference_stationarity_path
                or pending.inference_stationarity_sha256 != result.inference_stationarity_sha256
                or pending.blade_foreground_path != result.blade_foreground_path
                or pending.reconstructed_view_path != result.reconstructed_view_path
                or pending.coverage_path != result.coverage_path
                or pending.coarse_scan_view_path != result.coarse_scan_view_path
            ):
                raise FoundationStereoCycleError(
                    "Perception commit does not match the prepared asset transaction"
                )
            if pending.coverage_path is not None and (
                pending.coverage_metadata_sha256 is None
                or _sha256(pending.coverage_path / "coverage.json")
                != pending.coverage_metadata_sha256
            ):
                raise FoundationStereoCycleError(
                    "Prepared fine-coverage metadata changed before commit"
                )
            if pending.coarse_scan_view_path is not None and (
                pending.coarse_scan_metadata_sha256 is None
                or _sha256(pending.coarse_scan_view_path / "metadata.json")
                != pending.coarse_scan_metadata_sha256
            ):
                raise FoundationStereoCycleError(
                    "Prepared coarse-scan metadata changed before commit"
                )
            try:
                if pending.blade_foreground_path is not None:
                    read_blade_foreground_mask(pending.blade_foreground_path)
                if pending.reconstructed_view_path is not None:
                    read_reconstructed_view(pending.reconstructed_view_path)
                if pending.coverage_path is not None:
                    read_surface_coverage_generation(
                        pending.coverage_path,
                        require_foreground_bound_science=True,
                    )
                if pending.coarse_scan_view_path is not None:
                    from biblade_fusion.storage.coarse_scan import (
                        read_coarse_scan_view,
                    )

                    read_coarse_scan_view(pending.coarse_scan_view_path)
            except (OSError, TypeError, ValueError) as exc:
                raise FoundationStereoCycleError(
                    f"Prepared science asset changed before commit: {exc}"
                ) from exc
            self._reverify_pending_authority(captured, result, pending)
            before_commit("after_pending_authority_validation")
            if key in self._capture_roots:
                self._poisoned_reason = "logical capture identity committed twice"
                raise FoundationStereoCycleError(self._poisoned_reason)
            self._write_logical_commit_marker(
                pending,
                before_commit=before_commit,
            )
            self._sources = list(pending.sources)
            self._updates = list(pending.updates)
            self._accepted_coverage_path = pending.accepted_coverage_path_after_commit
            self._capture_roots[key] = pending.cycle_root
            self._pending_commit = None

    @staticmethod
    def _write_logical_commit_marker(
        pending: _PendingPerceptionCommit,
        *,
        before_commit: Callable[[str], None] = lambda _stage: None,
    ) -> None:
        logical_root = pending.cycle_root.parent
        marker = logical_root / "committed.json"
        payload = {
            "schema_version": 1,
            "artifact_kind": "biblade_fusion.foundation_stereo_logical_commit",
            "logical_identity": {
                "view_id": pending.key[0],
                "sequence_index": pending.key[1],
            },
            "accepted_attempt": {
                "attempt_id": pending.cycle_root.name,
                "root": str(pending.cycle_root),
            },
            "authority": {
                "raw_session_manifest_sha256": pending.raw_session_manifest_sha256,
                "raw_session_view_metadata_sha256": (pending.raw_session_view_metadata_sha256),
                "stereo_metadata_sha256": pending.stereo_metadata_sha256,
                "inference_stationarity_sha256": (pending.inference_stationarity_sha256),
                "occupancy_metadata_sha256": pending.occupancy_metadata_sha256,
                "occupancy_binding": list(pending.occupancy_binding),
            },
        }
        temporary = logical_root / f".committed.{uuid4().hex}.partial"
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            before_commit("before_logical_commit_temporary_write")
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # This is the final fallible deadline gate before the immutable logical
            # identity becomes visible.  A raised timeout leaves only the temporary
            # file, which the ``finally`` block removes without advancing sources.
            before_commit("before_logical_commit_marker_link")
            os.link(temporary, marker)
            descriptor = os.open(logical_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except FileExistsError as exc:
            raise FoundationStereoCycleError(
                "Logical capture identity already has an immutable commit marker"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _reverify_pending_authority(
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
        pending: _PendingPerceptionCommit,
    ) -> None:
        """Re-read the full disk authority at the commit linearization point."""

        manifest_path = pending.raw_session_path / "manifest.json"
        stereo_metadata_path = pending.stereo_inference_path / "metadata.json"
        occupancy_metadata_path = pending.occupancy_mapping_path / "metadata.json"
        try:
            hashes = {
                "raw session manifest": (
                    _sha256(manifest_path),
                    pending.raw_session_manifest_sha256,
                ),
                "raw session view metadata": (
                    _single_view_metadata_hash(pending.raw_session_path, captured.bundle),
                    pending.raw_session_view_metadata_sha256,
                ),
                "stereo inference metadata": (
                    _sha256(stereo_metadata_path),
                    pending.stereo_metadata_sha256,
                ),
                "inference stationarity": (
                    _sha256(pending.inference_stationarity_path),
                    pending.inference_stationarity_sha256,
                ),
                "occupancy metadata": (
                    _sha256(occupancy_metadata_path),
                    pending.occupancy_metadata_sha256,
                ),
            }
        except (OSError, TypeError, ValueError) as exc:
            raise FoundationStereoCycleError(
                f"Prepared disk authority is missing or unreadable: {exc}"
            ) from exc
        changed = tuple(label for label, values in hashes.items() if values[0] != values[1])
        if changed:
            raise FoundationStereoCycleError(
                "Prepared disk authority changed before commit: " + ", ".join(changed)
            )

        try:
            stored_stereo = read_stereo_inference(pending.stereo_inference_path)
            verify_stereo_inference_source(
                stored_stereo,
                expected_session=pending.raw_session_path,
            )
            observation = stored_stereo.observation
            stored_stationarity = read_inference_stationarity(pending.inference_stationarity_path)
            authoritative_mapping = read_occupancy_mapping(pending.occupancy_mapping_path)
            authoritative_binding = OccupancyBinding.from_mapping(authoritative_mapping).tuple
            returned_binding = OccupancyBinding.from_mapping(result.stored_occupancy).tuple
        except (OSError, TypeError, ValueError) as exc:
            raise FoundationStereoCycleError(
                f"Prepared disk authority failed semantic readback: {exc}"
            ) from exc

        bundle = captured.bundle
        if (
            observation.source_view_id != bundle.view_id
            or observation.source_sequence_index != bundle.sequence_index
            or observation.rectified.source_frame_number != bundle.stereo.frame_number
            or stored_stationarity.view_id != bundle.view_id
            or stored_stationarity.sequence_index != bundle.sequence_index
            or stored_stationarity.source_session_manifest_path != manifest_path.resolve()
            or stored_stationarity.source_session_manifest_sha256
            != pending.raw_session_manifest_sha256
            or authoritative_binding != pending.occupancy_binding
            or returned_binding != pending.occupancy_binding
            or not authoritative_mapping.frame_evidence
        ):
            raise FoundationStereoCycleError(
                "Prepared disk authority identity or binding changed before commit"
            )
        current_evidence = authoritative_mapping.frame_evidence[-1]
        if (
            current_evidence.source_view_id != bundle.view_id
            or current_evidence.source_sequence_index != bundle.sequence_index
            or current_evidence.frame_number != bundle.stereo.frame_number
            or current_evidence.source_stereo_metadata_sha256 != pending.stereo_metadata_sha256
            or current_evidence.source_session_manifest_sha256
            != pending.raw_session_manifest_sha256
            or current_evidence.source_session_view_metadata_sha256
            != pending.raw_session_view_metadata_sha256
        ):
            raise FoundationStereoCycleError(
                "Occupancy authority is not bound to the committing physical attempt"
            )

    def _prepare_science_assets(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        occupancy_update: OccupancyFrameUpdate,
        occupancy_path: Path,
    ) -> PreparedFineScienceAssets:
        if self._reference_coarse_model is None:
            if (
                captured.purpose is CapturePurpose.CANDIDATE
                and self._coarse_science_preparer is None
            ):
                raise FoundationStereoCycleError(
                    "Candidate capture requires the configured fine-science pipeline"
                )
            return PreparedFineScienceAssets(None, None, None, False)
        return prepare_fine_science_assets(
            purpose=captured.purpose,
            captured=captured,
            stereo=stereo,
            stereo_path=stereo_path,
            occupancy_update=occupancy_update,
            occupancy_path=occupancy_path,
            settings=self._settings,
            hand_eye=self._hand_eye,
            reference_coarse_model=self._reference_coarse_model,
            accepted_coverage_path=self._accepted_coverage_path,
        )

    def _prepare_coarse_science_asset(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        occupancy_update: OccupancyFrameUpdate,
        occupancy_path: Path,
    ) -> Path | None:
        preparer = self._coarse_science_preparer
        if preparer is None or captured.purpose not in {
            CapturePurpose.BOOTSTRAP,
            CapturePurpose.CANDIDATE,
        }:
            return None
        path = Path(
            preparer(
                captured,
                stereo,
                stereo_path,
                occupancy_update,
                occupancy_path,
            )
        ).resolve()
        if path.parent != captured.cycle_root.resolve() or not path.is_dir():
            raise FoundationStereoCycleError(
                "Coarse-science preparer must write one direct child of the cycle root"
            )
        from biblade_fusion.storage.coarse_scan import read_coarse_scan_view

        with performance_span("coarse.scan_view_readback"):
            read_coarse_scan_view(path)
        return path

    def _preflight_coarse_science(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        prepared_occupancy: PreparedOccupancyFrame,
    ) -> None:
        preflighter = self._coarse_science_preflighter
        if preflighter is None or captured.purpose not in {
            CapturePurpose.BOOTSTRAP,
            CapturePurpose.CANDIDATE,
        }:
            return
        preflighter(captured, stereo, stereo_path, prepared_occupancy)

    def _require_pending_sampler(
        self,
        key: tuple[str, int],
    ) -> RobotStateSampler:
        with self._pending_lock:
            if self._pending_key != key or self._pending_sampler is None:
                raise FoundationStereoCycleError(
                    "Capture has no live continuous stationarity sampler"
                )
            return self._pending_sampler

    def _validate_capture_binding(
        self,
        sampled_trace: tuple[RobotState, ...],
        capture_states: tuple[RobotState, ...],
    ) -> None:
        """Bind the independent trace to the main RTSI camera bracket."""

        maximum_time_delta_s = (
            self._settings.stop_and_capture.maximum_robot_state_staleness_s
        )
        acquisition = self._settings.acquisition
        for capture_index, captured in enumerate(capture_states):
            nearest = min(
                sampled_trace,
                key=lambda state: abs(
                    state.controller_time_s - captured.controller_time_s
                ),
            )
            controller_delta_s = abs(
                nearest.controller_time_s - captured.controller_time_s
            )
            if controller_delta_s > maximum_time_delta_s:
                raise FoundationStereoCycleError(
                    "Independent RTSI trace does not bracket camera state "
                    f"{capture_index}: nearest controller delta "
                    f"{controller_delta_s:.9g} s exceeds {maximum_time_delta_s:.9g} s"
                )
            joint_delta_rad = float(
                np.max(
                    np.abs(
                        nearest.joint_positions_rad - captured.joint_positions_rad
                    )
                )
            )
            tcp_translation_delta_m = float(
                np.linalg.norm(
                    nearest.base_t_tcp.translation_m
                    - captured.base_t_tcp.translation_m
                )
            )
            relative_rotation = (
                nearest.base_t_tcp.rotation.T @ captured.base_t_tcp.rotation
            )
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            )
            tcp_rotation_delta_rad = float(math.acos(cosine))
            if (
                joint_delta_rad > acquisition.max_joint_delta_rad
                or tcp_translation_delta_m
                > acquisition.max_tcp_translation_delta_m
                or tcp_rotation_delta_rad > acquisition.max_tcp_rotation_delta_rad
            ):
                raise FoundationStereoCycleError(
                    "Independent RTSI trace differs from camera bracket state "
                    f"{capture_index}: joint={joint_delta_rad:.9g} rad, "
                    f"tcp_translation={tcp_translation_delta_m:.9g} m, "
                    f"tcp_rotation={tcp_rotation_delta_rad:.9g} rad"
                )
            if (
                nearest.robot_mode.strip().upper()
                != captured.robot_mode.strip().upper()
                or nearest.safety_status.strip().upper()
                != captured.safety_status.strip().upper()
            ):
                raise FoundationStereoCycleError(
                    "Independent RTSI trace controller state differs from camera "
                    f"bracket state {capture_index}"
                )

    def _clear_pending_sampler(self, sampler: RobotStateSampler) -> None:
        with self._pending_lock:
            if self._pending_sampler is sampler:
                if sampler.is_alive:
                    self._poisoned_reason = "attempted to clear a live robot-state sampler"
                    raise FoundationStereoCycleError(self._poisoned_reason)
                self._pending_sampler = None
                self._pending_key = None
                self._pending_attempt_root = None

    def _stage_pending_commit(
        self,
        sampler: RobotStateSampler,
        pending: _PendingPerceptionCommit,
    ) -> None:
        """Atomically replace a finished sampler with an uncommitted asset set."""

        with self._pending_lock:
            if (
                self._pending_sampler is not sampler
                or self._pending_key != pending.key
                or self._pending_attempt_root != pending.cycle_root
                or self._pending_commit is not None
            ):
                self._poisoned_reason = "perception transaction ownership changed"
                raise FoundationStereoCycleError(self._poisoned_reason)
            if sampler.is_alive:
                self._poisoned_reason = (
                    "attempted to stage assets while robot-state sampler is live"
                )
                raise FoundationStereoCycleError(self._poisoned_reason)
            self._pending_sampler = None
            self._pending_key = None
            self._pending_attempt_root = None
            self._pending_commit = pending

    def _cancel_pending_sampler_instance(
        self,
        sampler: RobotStateSampler,
    ) -> None:
        try:
            sampler.cancel()
        except BaseException as exc:
            with self._pending_lock:
                if self._pending_sampler is sampler:
                    self._poisoned_reason = f"{type(exc).__name__}: {exc}"
            raise
        self._clear_pending_sampler(sampler)

    def _fresh_rebuild_sources(self, current: _VerifiedSource) -> tuple[_VerifiedSource, ...]:
        retained = (*self._sources, current)
        maximum_gap_s = (
            self._settings.stop_and_capture.maximum_operator_reposition_interval_s
        )
        suffix_start = 0
        previous_monotonic_ns = retained[0].captured.bundle.stereo.monotonic_time_ns
        maximum_gap_ns = maximum_gap_s * 1e9 if maximum_gap_s is not None else None
        for index, source in enumerate(retained[1:], start=1):
            captured_monotonic_ns = source.captured.bundle.stereo.monotonic_time_ns
            gap_ns = captured_monotonic_ns - previous_monotonic_ns
            if gap_ns <= 0:
                raise FoundationStereoCycleError(
                    "Committed source monotonic capture timestamps did not advance"
                )
            if maximum_gap_ns is not None and gap_ns > maximum_gap_ns:
                # The earlier safety window is no longer accepted evidence.  Rebuild
                # only from the continuous tail, which returns the map to MAPPING
                # until the configured independent-view count is reacquired.
                suffix_start = index
            previous_monotonic_ns = captured_monotonic_ns
        continuous = retained[suffix_start:]
        # A viewpoint motion can end too close to its preceding stopped frame to add
        # an independent FREE vote.  Replace conflicting older viewpoints instead
        # of rejecting the whole mandatory capture or counting duplicate evidence.
        independent = tuple(
            source
            for source in continuous[:-1]
            if self._views_are_independent(source, current)
        )
        refreshed = (*independent, current)
        return refreshed[-self._settings.occupancy.maximum_source_views :]

    def _camera_view_evidence(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
    ) -> dict[str, tuple[float, float, float]]:
        joints = captured.bundle.selected_robot_state.joint_positions_rad
        base_t_flange = PoseSE3(
            "base",
            "flange",
            self._renderer.base_t_flange_matrix(joints),
        )
        flange_t_left_ir = self._hand_eye.require_flange_primary()
        base_t_camera = base_t_flange.compose(flange_t_left_ir).compose(
            stereo.rectified.calibration.left_rectified_t_left_ir.inverse()
        )
        return {
            "camera_center_base_m": tuple(
                float(value) for value in base_t_camera.translation_m
            ),
            "camera_axis_base": tuple(
                float(value) for value in base_t_camera.rotation[:, 2]
            ),
        }

    def _views_are_independent(
        self,
        existing: _VerifiedSource,
        current: _VerifiedSource,
    ) -> bool:
        existing_center = np.asarray(existing.camera_center_base_m, dtype=np.float64)
        current_center = np.asarray(current.camera_center_base_m, dtype=np.float64)
        translation_m = float(np.linalg.norm(current_center - existing_center))
        existing_axis = np.asarray(existing.camera_axis_base, dtype=np.float64)
        current_axis = np.asarray(current.camera_axis_base, dtype=np.float64)
        cosine = float(np.clip(np.dot(current_axis, existing_axis), -1.0, 1.0))
        direction_deg = math.degrees(math.acos(cosine))
        config = self._settings.occupancy
        return (
            translation_m >= config.minimum_free_view_translation_m
            or direction_deg >= config.minimum_free_view_direction_deg
        )

    def _rebuild_updates(
        self,
        sources: tuple[_VerifiedSource, ...],
        *,
        prepared_current: PreparedOccupancyFrame | None = None,
    ) -> tuple[OccupancyFrameUpdate, ...]:
        prefix_is_unchanged = (
            len(sources) == len(self._sources) + 1
            and len(self._updates) == len(self._sources)
            and all(
                source is accepted
                for source, accepted in zip(
                    sources[:-1],
                    self._sources,
                    strict=True,
                )
            )
        )
        updates: list[OccupancyFrameUpdate] = (
            list(self._updates) if prefix_is_unchanged else []
        )
        previous = updates[-1].snapshot if updates else None
        previous_evidence_hash = (
            updates[-1].evidence.quality_evidence_hash if updates else None
        )
        start_index = len(updates)
        for index, source in enumerate(sources[start_index:], start=start_index):
            span_name = (
                "occupancy.current_source_integration"
                if index == len(sources) - 1
                else "occupancy.historical_source_replay"
            )
            with performance_span(span_name):
                if prepared_current is not None and index == len(sources) - 1:
                    identity = (
                        prepared_current.bundle.view_id,
                        prepared_current.bundle.sequence_index,
                        prepared_current.stereo.rectified.source_frame_number,
                    )
                    expected = (
                        source.captured.bundle.view_id,
                        source.captured.bundle.sequence_index,
                        source.stereo.rectified.source_frame_number,
                    )
                    if identity != expected:
                        raise FoundationStereoCycleError(
                            "Prepared occupancy frame differs from current rebuild source"
                        )
                    update = integrate_prepared_foundation_stereo_occupancy(
                        previous,
                        prepared_current,
                        self._settings.occupancy,
                        self._renderer,
                        previous_evidence_hash=previous_evidence_hash,
                    )
                else:
                    update = integrate_foundation_stereo_occupancy(
                        previous,
                        source.captured.bundle,
                        source.stereo,
                        self._hand_eye,
                        self._settings.occupancy,
                        self._settings.acquisition,
                        self._renderer,
                        captured_at_utc=source.captured.captured_at_utc,
                        source_stereo_metadata_sha256=source.stereo_metadata_sha256,
                        source_session_manifest_sha256=source.session_manifest_sha256,
                        source_session_view_metadata_sha256=(
                            source.session_view_metadata_sha256
                        ),
                        previous_evidence_hash=previous_evidence_hash,
                    )
            updates.append(update)
            previous = update.snapshot
            previous_evidence_hash = update.evidence.quality_evidence_hash
        return tuple(updates)

    def _aware_utc_now(self) -> datetime:
        value = self._utc_clock()
        if value.tzinfo is None:
            raise FoundationStereoCycleError("Cycle UTC clock must be timezone-aware")
        return value.astimezone(UTC)


def _single_view_metadata_hash(
    session_path: Path,
    bundle: SynchronizedFrameBundle,
) -> str:
    manifest_path = session_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        views = manifest["views"]
        if not isinstance(views, list) or len(views) != 1:
            raise ValueError("session does not contain exactly one view")
        record = views[0]
        if (
            record.get("view_id"),
            record.get("sequence_index"),
        ) != (bundle.view_id, bundle.sequence_index):
            raise ValueError("session view identity mismatch")
        relative = Path(str(record["path"])) / "metadata.json"
        metadata_path = (session_path / relative).resolve()
        if not metadata_path.is_relative_to(session_path.resolve()):
            raise ValueError("session view metadata escapes session root")
        return _sha256(metadata_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FoundationStereoCycleError("Cannot bind the single-view metadata asset") from exc


def _session_created_at_utc(session_path: Path) -> datetime:
    try:
        payload = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
        value = datetime.fromisoformat(str(payload["created_at_utc"]))
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session created_at_utc is not timezone-aware")
        return value.astimezone(UTC)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FoundationStereoCycleError(
            "Cannot recover the authoritative raw-session capture timestamp"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not safe:
        raise ValueError("View ID has no safe filename characters")
    return safe


class _RobotStateSampler:
    """Read-only trace spanning exposure, inference, and occupancy reconstruction."""

    def __init__(
        self,
        source: RobotStateSource,
        poll_period_s: float,
        *,
        prefer_fifo: bool,
    ) -> None:
        self._source = source
        self._poll_period_s = poll_period_s
        self._prefer_fifo = prefer_fifo
        self._trace: list[RobotState] = []
        self._errors: list[BaseException] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._sample,
            name="bbf-foundation-stereo-stationarity",
            daemon=True,
        )
        self._finished = False
        self._started = False

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "sampler_kind": "in_process_thread",
            "poll_period_s": self._poll_period_s,
            "retained_sample_count": len(self._trace),
            "fifo_requested": self._prefer_fifo,
        }

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        if self._started:
            raise FoundationStereoCycleError("Robot-state sampler was already started")
        self._started = True
        self._thread.start()
        if not self._ready.wait(timeout=max(1.0, self._poll_period_s * 4.0)):
            self._stop.set()
            self._thread.join(timeout=max(1.0, self._poll_period_s * 4.0))
            raise FoundationStereoCycleError(
                "Robot-state sampler did not produce initial readiness evidence"
            )
        if self._errors:
            error = self._errors[0]
            self._stop.set()
            self._thread.join(timeout=max(1.0, self._poll_period_s * 4.0))
            raise FoundationStereoCycleError(
                "Initial robot-state sampling failed before FoundationStereo inference: "
                f"{type(error).__name__}: {error}"
            ) from error

    def finish(self) -> tuple[RobotState, ...]:
        if self._finished:
            raise FoundationStereoCycleError("Robot-state sampler was already finished")
        if not self._started:
            raise FoundationStereoCycleError("Robot-state sampler was never started")
        self._finished = True
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._poll_period_s * 4.0))
        if self._thread.is_alive():
            raise FoundationStereoCycleError("Robot-state sampler did not terminate")
        if self._errors:
            error = self._errors[0]
            raise FoundationStereoCycleError(
                "Robot-state sampling failed during the perception transaction: "
                f"{type(error).__name__}: {error}"
            ) from error
        try:
            self._trace.append(self._source.read_state())
        except BaseException as exc:
            raise FoundationStereoCycleError("Post-transaction robot state is unavailable") from exc
        return _ordered_unique_robot_states(tuple(self._trace))

    def cancel(self) -> None:
        """Stop sampling without asserting that the transaction was stationary."""

        self._finished = True
        self._stop.set()
        if not self._started:
            return
        self._thread.join(timeout=max(1.0, self._poll_period_s * 4.0))
        if self._thread.is_alive():
            raise FoundationStereoCycleError("Robot-state sampler did not terminate")

    def _sample(self) -> None:
        original_scheduler: tuple[int, os.sched_param] | None = None
        try:
            if self._prefer_fifo:
                original_scheduler = _try_enter_robot_state_sampler_fifo()
            self._trace.append(self._source.read_state())
            self._ready.set()
            while not self._stop.wait(self._poll_period_s):
                self._trace.append(self._source.read_state())
        except BaseException as exc:
            self._errors.append(exc)
            self._stop.set()
            self._ready.set()
        finally:
            if original_scheduler is not None:
                try:
                    _restore_robot_state_sampler_scheduler(original_scheduler)
                except BaseException as exc:
                    self._errors.append(exc)
                    self._stop.set()
                    self._ready.set()


def _try_enter_robot_state_sampler_fifo() -> tuple[int, os.sched_param] | None:
    """Give the short, sleeping sampler priority over CPU-heavy perception work."""

    try:
        maximum = os.sched_get_priority_max(os.SCHED_FIFO)
        original = (os.sched_getscheduler(0), os.sched_getparam(0))
        if not 1 <= _ROBOT_STATE_SAMPLER_FIFO_PRIORITY <= maximum:
            return None
        os.sched_setscheduler(
            0,
            os.SCHED_FIFO,
            os.sched_param(_ROBOT_STATE_SAMPLER_FIFO_PRIORITY),
        )
    except (AttributeError, OSError):
        return None
    return original


def _restore_robot_state_sampler_scheduler(
    original: tuple[int, os.sched_param],
) -> None:
    try:
        os.sched_setscheduler(0, original[0], original[1])
    except (AttributeError, OSError) as exc:
        raise FoundationStereoCycleError(
            "Robot-state sampler could not restore its original scheduler"
        ) from exc


def _robot_state_values_equal(left: RobotState, right: RobotState) -> bool:
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


def _ordered_unique_robot_states(
    states: tuple[RobotState, ...],
    *,
    minimum_count: int = 3,
) -> tuple[RobotState, ...]:
    """Merge sampler and capture-bracket states into one exact monotonic trace."""

    ordered = sorted(
        enumerate(states),
        key=lambda item: (item[1].monotonic_time_ns, item[0]),
    )
    unique: list[RobotState] = []
    for _, state in ordered:
        if unique and state.monotonic_time_ns == unique[-1].monotonic_time_ns:
            if not _robot_state_values_equal(unique[-1], state):
                raise FoundationStereoCycleError(
                    "Conflicting robot states share one host monotonic timestamp"
                )
            continue
        unique.append(state)
    if len(unique) < minimum_count:
        raise FoundationStereoCycleError(
            "Perception transaction needs at least "
            f"{minimum_count} distinct robot-state samples"
        )
    return tuple(unique)


def _build_authoritative_stationarity_trace(
    independent_trace: tuple[RobotState, ...],
    capture_states: tuple[RobotState, ...],
) -> tuple[RobotState, ...]:
    """Merge exact capture evidence without admitting cross-clock inversions."""

    captures = _ordered_unique_robot_states(capture_states, minimum_count=1)
    capture_before = captures[0]
    capture_after = captures[-1]
    independent_before = tuple(
        state
        for state in independent_trace
        if (
            state.monotonic_time_ns < capture_before.monotonic_time_ns
            and state.controller_time_s <= capture_before.controller_time_s
        )
    )
    independent_after = tuple(
        state
        for state in independent_trace
        if (
            state.monotonic_time_ns > capture_after.monotonic_time_ns
            and state.controller_time_s >= capture_after.controller_time_s
        )
    )
    return _ordered_unique_robot_states(
        (*independent_before, *captures, *independent_after)
    )
