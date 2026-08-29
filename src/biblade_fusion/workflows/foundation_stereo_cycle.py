"""Concrete single-backend perception transaction for the stop-scan coordinator."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from biblade_fusion.acquisition import SynchronizedAcquirer, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    AppSettings,
    OccupancyConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.devices.robot.base import RobotState, RobotStateSource
from biblade_fusion.perception.stereo import FoundationStereoBackend
from biblade_fusion.robotics.stationarity import validate_stationary_trace
from biblade_fusion.storage.blade_foreground import read_blade_foreground_mask
from biblade_fusion.storage.inference_stationarity import (
    write_inference_stationarity,
)
from biblade_fusion.storage.occupancy_mapping import (
    read_occupancy_mapping,
    write_occupancy_mapping,
)
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
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
    RobotDepthRenderer,
    integrate_foundation_stereo_occupancy,
)
from biblade_fusion.workflows.stereo_inference import (
    StereoInferenceObservation,
    infer_rectified_stereo,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    CapturePurpose,
    PerceptionCycleResult,
)


class FoundationStereoCycleError(RuntimeError):
    """A stopped FoundationStereo asset transaction could not be committed."""


@dataclass(frozen=True, slots=True)
class _VerifiedSource:
    captured: CapturedStopScanView
    stereo: StereoInferenceObservation
    stereo_path: Path
    stereo_metadata_sha256: str
    session_manifest_sha256: str
    session_view_metadata_sha256: str


@dataclass(frozen=True, slots=True)
class _PendingPerceptionCommit:
    """One materialized cycle not yet accepted by the coordinator."""

    key: tuple[str, int]
    cycle_root: Path
    occupancy_mapping_path: Path
    inference_stationarity_sha256: str
    sources: tuple[_VerifiedSource, ...]
    blade_foreground_path: Path | None
    reconstructed_view_path: Path | None
    coverage_path: Path | None
    coverage_metadata_sha256: str | None
    accepted_coverage_path_after_commit: Path | None


class FoundationStereoOccupancyCycleEngine:
    """Capture one immutable session, infer once, and rebuild a fresh map window.

    The engine owns no motion interface.  A background thread samples read-only robot
    state while the main thread performs FoundationStereo inference.  The coordinator
    validates the returned trace against the capture pose before publishing the map.
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
        if reference is not None:
            reference, accepted_coverage = validate_fine_science_startup(
                settings,
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
        self._utc_clock = utc_clock
        self._sources: list[_VerifiedSource] = []
        self._capture_roots: dict[tuple[str, int], Path] = {}
        self._pending_lock = threading.Lock()
        self._pending_key: tuple[str, int] | None = None
        self._pending_sampler: _RobotStateSampler | None = None
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
        cycle_root = self._output_root / "cycles" / (f"{sequence_index:06d}_{safe_view}")
        key = (view_id, sequence_index)
        sampler = _RobotStateSampler(
            self._state_source,
            self._settings.stop_and_capture.settle_poll_period_s,
        )
        with self._pending_lock:
            if self._poisoned_reason is not None:
                raise FoundationStereoCycleError(
                    "Perception engine is fail-closed after sampler cleanup failure: "
                    f"{self._poisoned_reason}"
                )
            if self._pending_sampler is not None or self._pending_commit is not None:
                raise FoundationStereoCycleError(
                    "A prior capture transaction is still awaiting inference or commit"
                )
            self._pending_key = key
            self._pending_sampler = sampler
        writer: SessionWriter | None = None
        try:
            cycle_root.parent.mkdir(parents=True, exist_ok=True)
            cycle_root.mkdir()
            writer = SessionWriter.create(
                cycle_root / "raw",
                self._settings,
                label=f"cycle_{sequence_index:06d}_{safe_view}",
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
        if key in self._capture_roots:
            with suppress(BaseException):
                self._cancel_pending_sampler_instance(sampler)
            raise FoundationStereoCycleError("Capture identity was already committed")
        self._capture_roots[key] = cycle_root
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
        """Infer FoundationStereo and stage a reverified map candidate."""

        key = (captured.bundle.view_id, captured.bundle.sequence_index)
        if self._capture_roots.get(key) != captured.cycle_root:
            raise FoundationStereoCycleError("Captured view is not owned by this engine")
        sampler = self._require_pending_sampler(key)
        sampler_finished = False
        try:
            observation = infer_rectified_stereo(
                captured.bundle,
                self._backend,
                self._settings.stereo_rectification,
            )
            stereo_path = captured.cycle_root / "stereo_inference"
            write_stereo_inference(
                stereo_path,
                observation,
                self._settings.foundation_stereo,
                self._settings.stereo_rectification,
                source_session=captured.raw_session_path,
                source_stereo_calibration=(self._settings.realsense.stereo_calibration_path),
            )
            stored_stereo = read_stereo_inference(stereo_path)
            verify_stereo_inference_source(
                stored_stereo,
                expected_session=captured.raw_session_path,
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
            )
            candidates = self._fresh_rebuild_sources(source)
            updates = self._rebuild_updates(candidates)
            occupancy_path = captured.cycle_root / "occupancy_mapping"
            write_occupancy_mapping(
                occupancy_path,
                updates,
                self._settings.occupancy,
                self._settings.acquisition,
                source_stereo_inferences=[item.stereo_path for item in candidates],
                source_sessions=[item.captured.raw_session_path for item in candidates],
                source_hand_eye=self._hand_eye.source_path,
            )
            stored_mapping = read_occupancy_mapping(occupancy_path)
            science = self._prepare_science_assets(
                captured,
                stored_stereo.observation,
                stereo_path,
                updates[-1],
                occupancy_path,
            )
            full_trace = sampler.finish(
                additional_states=(
                    captured.bundle.robot_state_before,
                    captured.bundle.selected_robot_state,
                    captured.bundle.robot_state_after,
                )
            )
            sampler_finished = True
            stationarity_reference = full_trace[0]
            state_trace = full_trace[1:]
            stationarity = validate_stationary_trace(
                stationarity_reference,
                state_trace,
                max_joint_delta_rad=self._settings.acquisition.max_joint_delta_rad,
                max_tcp_translation_delta_m=(
                    self._settings.acquisition.max_tcp_translation_delta_m
                ),
                max_tcp_rotation_delta_rad=(self._settings.acquisition.max_tcp_rotation_delta_rad),
                maximum_robot_state_staleness_s=(
                    self._settings.stop_and_capture.maximum_robot_state_staleness_s
                ),
            )
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
                max_tcp_rotation_delta_rad=(self._settings.acquisition.max_tcp_rotation_delta_rad),
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
                occupancy_mapping_path=result.occupancy_mapping_path,
                inference_stationarity_sha256=(result.inference_stationarity_sha256),
                sources=candidates,
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
            ),
        )
        return result

    def commit_perception_cycle(
        self,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
    ) -> None:
        """Commit exactly one coordinator-accepted perception transaction."""

        key = (captured.bundle.view_id, captured.bundle.sequence_index)
        with self._pending_lock:
            pending = self._pending_commit
            if (
                pending is None
                or pending.key != key
                or pending.cycle_root != captured.cycle_root
                or pending.occupancy_mapping_path != result.occupancy_mapping_path
                or pending.inference_stationarity_sha256 != result.inference_stationarity_sha256
                or pending.blade_foreground_path != result.blade_foreground_path
                or pending.reconstructed_view_path != result.reconstructed_view_path
                or pending.coverage_path != result.coverage_path
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
            except (OSError, TypeError, ValueError) as exc:
                raise FoundationStereoCycleError(
                    f"Prepared fine-science asset changed before commit: {exc}"
                ) from exc
            self._sources = list(pending.sources)
            self._accepted_coverage_path = pending.accepted_coverage_path_after_commit
            self._pending_commit = None

    def _prepare_science_assets(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        occupancy_update: OccupancyFrameUpdate,
        occupancy_path: Path,
    ) -> PreparedFineScienceAssets:
        if self._reference_coarse_model is None:
            if captured.purpose is CapturePurpose.CANDIDATE:
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

    def _require_pending_sampler(
        self,
        key: tuple[str, int],
    ) -> _RobotStateSampler:
        with self._pending_lock:
            if self._pending_key != key or self._pending_sampler is None:
                raise FoundationStereoCycleError(
                    "Capture has no live continuous stationarity sampler"
                )
            return self._pending_sampler

    def _clear_pending_sampler(self, sampler: _RobotStateSampler) -> None:
        with self._pending_lock:
            if self._pending_sampler is sampler:
                if sampler.is_alive:
                    self._poisoned_reason = "attempted to clear a live robot-state sampler"
                    raise FoundationStereoCycleError(self._poisoned_reason)
                self._pending_sampler = None
                self._pending_key = None

    def _stage_pending_commit(
        self,
        sampler: _RobotStateSampler,
        pending: _PendingPerceptionCommit,
    ) -> None:
        """Atomically replace a finished sampler with an uncommitted asset set."""

        with self._pending_lock:
            if (
                self._pending_sampler is not sampler
                or self._pending_key != pending.key
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
            self._pending_commit = pending

    def _cancel_pending_sampler_instance(
        self,
        sampler: _RobotStateSampler,
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
        now = self._aware_utc_now()
        oldest = now - timedelta(seconds=self._settings.occupancy.maximum_map_age_s)
        sources = tuple(
            item
            for item in (*self._sources, current)
            if oldest <= item.captured.captured_at_utc <= now
        )
        if not sources or sources[-1] is not current:
            raise FoundationStereoCycleError(
                "Current FoundationStereo frame expired before map rebuild"
            )
        return sources

    def _rebuild_updates(
        self,
        sources: tuple[_VerifiedSource, ...],
    ) -> tuple[OccupancyFrameUpdate, ...]:
        updates: list[OccupancyFrameUpdate] = []
        previous = None
        previous_evidence_hash = None
        for source in sources:
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
                source_session_view_metadata_sha256=(source.session_view_metadata_sha256),
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

    def __init__(self, source: RobotStateSource, poll_period_s: float) -> None:
        self._source = source
        self._poll_period_s = poll_period_s
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

    def finish(
        self,
        *,
        additional_states: tuple[RobotState, ...] = (),
    ) -> tuple[RobotState, ...]:
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
        return _ordered_unique_robot_states((*self._trace, *additional_states))

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
        try:
            self._trace.append(self._source.read_state())
            self._ready.set()
            while not self._stop.wait(self._poll_period_s):
                self._trace.append(self._source.read_state())
        except BaseException as exc:
            self._errors.append(exc)
            self._stop.set()
            self._ready.set()


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
    if len(unique) < 3:
        raise FoundationStereoCycleError(
            "Perception transaction needs at least three distinct robot-state samples"
        )
    return tuple(unique)
