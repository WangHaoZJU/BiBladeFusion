"""Supervised, one-segment-at-a-time composition for blade experiments.

This module is an orchestration shell around :class:`StopScanCoordinator`; it does
not implement another perception, coverage, selection, collision, or execution
algorithm.  In particular, the injected perception engine and selector determine
whether a run is still bootstrapping an unknown blade or operating against a fixed
fine-scan reference.

No motion adapter is installed by default.  ``run_until_attention`` can advance
capture, FoundationStereo, occupancy, scientific assets, next-view selection and
single-segment preflight, but always returns at external approval.  Executing that
one prepared segment additionally requires both an injected motion adapter and a
typed operator approval.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from biblade_fusion.core.settings import (
    AcquisitionConfig,
    MotionPreflightConfig,
    OccupancyConfig,
    RobotConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.storage.stop_scan_run import (
    StopScanRunEvent,
    StopScanRunWriter,
)
from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentStatusSnapshot,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    CoordinatedRobot,
    FoundationStereoPerceptionEngine,
    NextViewSelector,
    OccupancyGenerationPublisher,
    PerceptionCycleResult,
    PreparedSegment,
    SegmentSafetyFactory,
    StopScanBlocked,
    StopScanCheckpoint,
    StopScanCoordinator,
    StopScanError,
    StopScanPhase,
)


class SupervisedExperimentBlocked(StopScanBlocked):
    """The outer supervision gate refused to advance the experiment."""


@dataclass(frozen=True, slots=True)
class OperatorApproval:
    """Explicit approval for exactly the currently prepared short segment."""

    operator_id: str
    confirmation: str

    def __post_init__(self) -> None:
        if not self.operator_id.strip() or not self.confirmation.strip():
            raise ValueError("Operator identity and confirmation must be non-empty")


@dataclass(frozen=True, slots=True)
class OperatorRejection:
    """Explicit refusal of the currently prepared segment."""

    operator_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.operator_id.strip() or not self.reason.strip():
            raise ValueError("Operator identity and rejection reason must be non-empty")


@dataclass(frozen=True, slots=True)
class RecoveryConfirmation:
    """Human assertion required before restarting from append-only run evidence.

    Recovery never restores a motion permit or an in-flight segment.  The new
    coordinator starts from bootstrap and must acquire fresh occupancy evidence.
    """

    operator_id: str
    physical_stop_confirmed: bool
    discard_pending_motion: bool

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ValueError("Recovery confirmation requires an operator identity")


class SupervisedMotionExecutor(Protocol):
    """Opt-in bridge to the coordinator's already guarded execution boundary."""

    def execute_segment(
        self,
        coordinator: StopScanCoordinator,
        approval: OperatorApproval,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class GuardedCoordinatorMotionExecutor:
    """Explicit adapter for the coordinator's guarded execution boundary.

    Constructing this adapter grants no approval and performs no motion.  Its only
    behavior is exact forwarding of a caller-supplied :class:`OperatorApproval` to
    the already guarded, one-segment coordinator method.
    """

    def execute_segment(
        self,
        coordinator: StopScanCoordinator,
        approval: OperatorApproval,
    ) -> object:
        return coordinator.execute_approved(
            operator_id=approval.operator_id,
            confirmation=approval.confirmation,
        )


class _CoordinatorPort(Protocol):
    @property
    def checkpoint(self) -> StopScanCheckpoint: ...

    def start(self) -> StopScanCheckpoint: ...

    def capture_infer_update(self, view_id: str | None = None) -> object: ...

    def prepare_next_segment(self) -> object | None: ...

    def approval_prompt(self) -> str: ...

    def request_stop(self, reason: str) -> StopScanCheckpoint: ...


StatusCallback = Callable[[ExperimentStatusSnapshot], None]
EventCallback = Callable[[StopScanRunEvent], None]
BootstrapViewProvider = Callable[[ExperimentStatusSnapshot], str | None]
PerceptionCallback = Callable[[PerceptionCycleResult], None]
PreparedSegmentCallback = Callable[[PreparedSegment | None], None]


class _ExperimentEventSink(Protocol):
    def append_event(
        self,
        *,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> StopScanRunEvent: ...


class _StreamingRunEventSink:
    """Persist before notifying, and propagate listener failures fail-closed."""

    def __init__(
        self,
        writer: StopScanRunWriter,
        callbacks: Iterable[EventCallback],
    ) -> None:
        self.writer = writer
        self._callbacks = tuple(callbacks)
        self._append_lock = threading.RLock()

    def append_event(
        self,
        *,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> StopScanRunEvent:
        with self._append_lock:
            event = self.writer.append_event(
                phase=phase,
                cycle_index=cycle_index,
                event_type=event_type,
                payload=dict(payload),
            )
            for callback in self._callbacks:
                callback(event)
            return event


class SupervisedExperimentRunner:
    """Advance a durable experiment only as far as the next human attention point."""

    def __init__(
        self,
        *,
        coordinator: _CoordinatorPort,
        run_writer: StopScanRunWriter,
        motion_executor: SupervisedMotionExecutor | None = None,
        status_callbacks: Iterable[StatusCallback] = (),
        perception_callbacks: Iterable[PerceptionCallback] = (),
        prepared_segment_callbacks: Iterable[PreparedSegmentCallback] = (),
        recovery_required: bool = False,
        event_sink: _ExperimentEventSink | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._writer = run_writer
        self._motion_executor = motion_executor
        self._status_callbacks = tuple(status_callbacks)
        self._perception_callbacks = tuple(perception_callbacks)
        self._prepared_segment_callbacks = tuple(prepared_segment_callbacks)
        self._event_sink = event_sink or run_writer
        self._recovery_required = bool(recovery_required)
        self._runner_blocking_reasons: tuple[str, ...] = (
            ("recovery_confirmation_required",) if recovery_required else ()
        )
        self._operation_lock = threading.Lock()

    @classmethod
    def create(
        cls,
        *,
        run_root: str | Path,
        run_id: str,
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
        motion_executor: SupervisedMotionExecutor | None = None,
        status_callbacks: Iterable[StatusCallback] = (),
        event_callbacks: Iterable[EventCallback] = (),
        perception_callbacks: Iterable[PerceptionCallback] = (),
        prepared_segment_callbacks: Iterable[PreparedSegmentCallback] = (),
        utc_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> SupervisedExperimentRunner:
        """Compose existing algorithms around a new append-only run asset."""

        writer = StopScanRunWriter.create(run_root, run_id=run_id)
        sink = _StreamingRunEventSink(writer, event_callbacks)
        coordinator_kwargs: dict[str, Any] = {}
        if utc_clock is not None:
            coordinator_kwargs["utc_clock"] = utc_clock
        if monotonic_clock is not None:
            coordinator_kwargs["monotonic_clock"] = monotonic_clock
        coordinator = StopScanCoordinator(
            config=config,
            acquisition_config=acquisition_config,
            robot_config=robot_config,
            motion_config=motion_config,
            occupancy_config=occupancy_config,
            robot=robot,
            perception=perception,
            selector=selector,
            safety_factory=safety_factory,
            publisher=publisher,
            event_sink=sink,
            **coordinator_kwargs,
        )
        return cls(
            coordinator=coordinator,
            run_writer=writer,
            motion_executor=motion_executor,
            status_callbacks=status_callbacks,
            perception_callbacks=perception_callbacks,
            prepared_segment_callbacks=prepared_segment_callbacks,
            event_sink=sink,
        )

    @classmethod
    def resume(
        cls,
        *,
        run_root: str | Path,
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
        motion_executor: SupervisedMotionExecutor | None = None,
        status_callbacks: Iterable[StatusCallback] = (),
        event_callbacks: Iterable[EventCallback] = (),
        perception_callbacks: Iterable[PerceptionCallback] = (),
        prepared_segment_callbacks: Iterable[PreparedSegmentCallback] = (),
        utc_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> SupervisedExperimentRunner:
        """Verify a run chain and return blocked until explicit recovery confirmation.

        The caller must inject perception/selector instances already configured with
        any accepted fixed coarse reference and fine-coverage generation.  This method
        intentionally does not infer those scientific assets from directory names.
        """

        writer = StopScanRunWriter.resume(run_root)
        previous = writer.events[-1] if writer.events else None
        sink = _StreamingRunEventSink(writer, event_callbacks)
        coordinator_kwargs: dict[str, Any] = {}
        if utc_clock is not None:
            coordinator_kwargs["utc_clock"] = utc_clock
        if monotonic_clock is not None:
            coordinator_kwargs["monotonic_clock"] = monotonic_clock
        coordinator = StopScanCoordinator(
            config=config,
            acquisition_config=acquisition_config,
            robot_config=robot_config,
            motion_config=motion_config,
            occupancy_config=occupancy_config,
            robot=robot,
            perception=perception,
            selector=selector,
            safety_factory=safety_factory,
            publisher=publisher,
            event_sink=sink,
            **coordinator_kwargs,
        )
        runner = cls(
            coordinator=coordinator,
            run_writer=writer,
            motion_executor=motion_executor,
            status_callbacks=status_callbacks,
            perception_callbacks=perception_callbacks,
            prepared_segment_callbacks=prepared_segment_callbacks,
            recovery_required=True,
            event_sink=sink,
        )
        sink.append_event(
            phase=StopScanPhase.MOTION_BLOCKED.value,
            cycle_index=(previous.cycle_index if previous is not None else 0),
            event_type="recovery_confirmation_required",
            payload={
                "previous_event_sha256": (previous.event_sha256 if previous is not None else None),
                "previous_phase": previous.phase if previous is not None else None,
                "pending_motion_restored": False,
                "occupancy_freshness_restored": False,
            },
        )
        runner._notify_status()
        return runner

    @property
    def status(self) -> ExperimentStatusSnapshot:
        checkpoint = self._coordinator.checkpoint
        events = self._writer.events
        phase = checkpoint.phase.value
        reasons = self._runner_blocking_reasons or checkpoint.blocking_reasons
        disposition = _disposition(
            checkpoint,
            recovery_required=self._recovery_required,
            runner_blocking_reasons=self._runner_blocking_reasons,
        )
        if checkpoint.stop_requested and not reasons:
            reasons = ("operator_stop_pending",)
        if disposition is ExperimentDisposition.BLOCKED and not reasons:
            reasons = (f"coordinator_{phase}",)
        return ExperimentStatusSnapshot(
            run_id=self._writer.run_id,
            run_root=self._writer.root,
            phase=phase,
            disposition=disposition,
            cycle_index=checkpoint.cycle_index,
            current_view_id=checkpoint.current_view_id,
            proposed_view_id=checkpoint.proposed_view_id,
            expected_capture_view_id=checkpoint.expected_capture_view_id,
            expected_capture_purpose=(
                checkpoint.expected_capture_purpose.value
                if checkpoint.expected_capture_purpose is not None
                else None
            ),
            blocking_reasons=tuple(reasons),
            event_count=len(events),
            latest_event_sha256=(events[-1].event_sha256 if events else None),
            recovery_required=self._recovery_required,
            awaiting_external_approval=(disposition is ExperimentDisposition.WAITING_APPROVAL),
            stop_requested=checkpoint.stop_requested,
            stop_transport_acknowledged=checkpoint.stop_transport_acknowledged,
            stop_stationarity_verified=checkpoint.stop_stationarity_verified,
        )

    @property
    def events(self) -> tuple[StopScanRunEvent, ...]:
        """Return the in-process view of already persisted run events."""

        return self._writer.events

    def start(self) -> ExperimentStatusSnapshot:
        with self._exclusive_operation():
            self._require_recovered()
            self._require_not_runner_blocked()
            try:
                self._coordinator.start()
            except BaseException as exc:
                self._latch_exception("start_failed", exc)
                raise
            return self._notify_status()

    def acknowledge_recovery(
        self,
        confirmation: RecoveryConfirmation,
    ) -> ExperimentStatusSnapshot:
        """Discard all prior motion authority, then restart from a fresh bootstrap."""

        with self._exclusive_operation():
            if not self._recovery_required:
                raise SupervisedExperimentBlocked("This run does not require recovery")
            if not confirmation.physical_stop_confirmed:
                return self._block(
                    "recovery_physical_stop_not_confirmed",
                    event_type="recovery_rejected",
                )
            if not confirmation.discard_pending_motion:
                return self._block(
                    "recovery_pending_motion_not_discarded",
                    event_type="recovery_rejected",
                )
            self._append_event(
                phase=StopScanPhase.IDLE.value,
                cycle_index=0,
                event_type="recovery_acknowledged",
                payload={
                    "operator_id": confirmation.operator_id,
                    "pending_motion_restored": False,
                    "requires_fresh_bootstrap_capture": True,
                },
            )
            self._recovery_required = False
            self._runner_blocking_reasons = ()
            try:
                self._coordinator.start()
            except BaseException as exc:
                self._latch_exception("recovery_restart_failed", exc)
                raise
            return self._notify_status()

    def step(self, *, view_id: str | None = None) -> ExperimentStatusSnapshot:
        """Advance one non-motion state-machine action and then return.

        This method never executes a robot segment.  A map-ready step performs exactly
        one next-view selection and one short-segment preflight, stopping at approval.
        """

        with self._exclusive_operation():
            self._require_recovered()
            self._require_not_runner_blocked()
            checkpoint = self._coordinator.checkpoint
            try:
                if checkpoint.phase is StopScanPhase.IDLE:
                    self._coordinator.start()
                elif checkpoint.phase in {
                    StopScanPhase.BOOTSTRAP_MAP_REQUIRED,
                    StopScanPhase.AWAITING_CAPTURE,
                    StopScanPhase.MOTION_BLOCKED,
                }:
                    result = self._coordinator.capture_infer_update(view_id)
                    if type(result) is not PerceptionCycleResult:
                        if self._perception_callbacks:
                            raise SupervisedExperimentBlocked(
                                "Coordinator returned an untyped perception result"
                            )
                    else:
                        for callback in self._perception_callbacks:
                            callback(result)
                elif checkpoint.phase in {
                    StopScanPhase.BOOTSTRAP_MOTION_READY,
                    StopScanPhase.MAP_READY,
                }:
                    prepared = self._coordinator.prepare_next_segment()
                    if prepared is not None and type(prepared) is not PreparedSegment:
                        if self._prepared_segment_callbacks:
                            raise SupervisedExperimentBlocked(
                                "Coordinator returned an untyped prepared segment"
                            )
                    else:
                        for callback in self._prepared_segment_callbacks:
                            callback(prepared)
                elif checkpoint.phase in {
                    StopScanPhase.WAITING_APPROVAL,
                    StopScanPhase.COMPLETE,
                    StopScanPhase.ABORTED,
                    StopScanPhase.FAILED,
                }:
                    return self._notify_status()
                else:
                    raise SupervisedExperimentBlocked(
                        f"Coordinator is busy in phase {checkpoint.phase.value}"
                    )
            except StopScanBlocked as exc:
                if self.status.disposition is not ExperimentDisposition.BLOCKED:
                    return self._block(
                        f"coordinator_blocked:{exc}",
                        event_type="coordinator_operation_blocked",
                    )
                return self._notify_status()
            except BaseException as exc:
                self._latch_exception("step_failed", exc)
                return self._notify_status()
            return self._notify_status()

    def run_until_attention(
        self,
        *,
        bootstrap_view_provider: BootstrapViewProvider | None = None,
        maximum_steps: int = 32,
    ) -> ExperimentStatusSnapshot:
        """Advance without motion until capture input, approval, completion, or block.

        Post-segment captures already carry their exact expected view identity and need
        no provider.  Operator-guided bootstrap captures require the injected provider;
        absence of one is an attention point, not an inferred view name.
        """

        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        for _ in range(maximum_steps):
            current = self.status
            if current.disposition in {
                ExperimentDisposition.WAITING_APPROVAL,
                ExperimentDisposition.BLOCKED,
                ExperimentDisposition.COMPLETE,
            }:
                return current
            view_id: str | None = None
            if (
                current.disposition is ExperimentDisposition.NEEDS_CAPTURE
                and current.expected_capture_view_id is None
            ):
                if bootstrap_view_provider is None:
                    return current
                try:
                    view_id = bootstrap_view_provider(current)
                except BaseException as exc:
                    with self._exclusive_operation():
                        self._latch_exception("bootstrap_view_provider_failed", exc)
                        return self._notify_status()
                if view_id is None:
                    return current
                # Expected post-motion view identity is held by the coordinator.
            updated = self.step(view_id=view_id)
            if updated.disposition in {
                ExperimentDisposition.WAITING_APPROVAL,
                ExperimentDisposition.BLOCKED,
                ExperimentDisposition.COMPLETE,
            }:
                return updated
        return self._block(
            f"run_until_attention_step_limit:{maximum_steps}",
            event_type="supervised_step_limit_reached",
        )

    def execute_approved_segment(
        self,
        approval: OperatorApproval,
    ) -> ExperimentStatusSnapshot:
        """Delegate exactly one prepared segment through an injected adapter."""

        with self._exclusive_operation():
            self._require_recovered()
            self._require_not_runner_blocked()
            checkpoint = self._coordinator.checkpoint
            if checkpoint.phase is not StopScanPhase.WAITING_APPROVAL:
                raise SupervisedExperimentBlocked("No approval-eligible segment is prepared")
            if self._motion_executor is None:
                return self._block(
                    "motion_executor_not_injected",
                    event_type="external_execution_gate_blocked",
                )
            self._runner_blocking_reasons = ()
            try:
                self._motion_executor.execute_segment(
                    self._coordinator,  # type: ignore[arg-type]
                    approval,
                )
            except BaseException as exc:
                self._latch_exception("external_execution_failed", exc)
                return self._notify_status()
            phase = self._coordinator.checkpoint.phase
            if phase is not StopScanPhase.AWAITING_CAPTURE:
                return self._block(
                    f"executor_returned_without_awaiting_capture:{phase.value}",
                    event_type="external_execution_contract_failed",
                )
            return self._notify_status()

    def reject_prepared_segment(
        self,
        rejection: OperatorRejection,
    ) -> ExperimentStatusSnapshot:
        """Latch an operator refusal without invoking a motion or stop adapter."""

        with self._exclusive_operation():
            self._require_recovered()
            self._require_not_runner_blocked()
            if self._coordinator.checkpoint.phase is not StopScanPhase.WAITING_APPROVAL:
                raise SupervisedExperimentBlocked("No approval-eligible segment is prepared")
            return self._block(
                f"operator_rejected_segment:{rejection.reason}",
                event_type="external_segment_rejected",
                payload={"operator_id": rejection.operator_id},
            )

    def request_stop(self, reason: str) -> ExperimentStatusSnapshot:
        """Reach the coordinator's asynchronous stop path without runner serialization.

        Ordinary runner operations are intentionally single-flight, but an operator
        stop must interrupt capture, inference, preflight, or execution rather than
        being rejected because one of those operations owns ``_operation_lock``.
        The coordinator supplies the actual stop linearization.  Status callbacks
        are emitted here only when the runner lock can be acquired immediately;
        otherwise the in-flight operation publishes its terminal status itself.
        """

        try:
            self._coordinator.request_stop(reason)
        except StopScanError:
            raise
        except BaseException as exc:
            failure = f"stop_failed:{type(exc).__name__}:{exc}"
            if self._operation_lock.acquire(blocking=False):
                try:
                    self._latch_exception("stop_failed", exc)
                finally:
                    self._operation_lock.release()
            else:
                # Never race an immutable event append against the active
                # transaction.  The exception still propagates and the runner's
                # status fails closed immediately in memory.
                self._runner_blocking_reasons = (failure,)
            raise
        if self._operation_lock.acquire(blocking=False):
            try:
                return self._notify_status()
            finally:
                self._operation_lock.release()
        return self.status

    def approval_prompt(self) -> str:
        """Expose text only; this method cannot approve or execute the segment."""

        self._require_recovered()
        if self._runner_blocking_reasons:
            raise SupervisedExperimentBlocked(self._runner_blocking_reasons[0])
        return self._coordinator.approval_prompt()

    def _block(
        self,
        reason: str,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ExperimentStatusSnapshot:
        text = reason.strip()
        if not text:
            raise ValueError("Supervision block reason must be non-empty")
        self._runner_blocking_reasons = (text,)
        checkpoint = self._coordinator.checkpoint
        self._append_event(
            phase=checkpoint.phase.value,
            cycle_index=checkpoint.cycle_index,
            event_type=event_type,
            payload={
                **dict(payload or {}),
                "reason": text,
                "motion_authorized": False,
            },
        )
        return self._notify_status()

    def _latch_exception(self, label: str, exc: BaseException) -> None:
        reason = f"{label}:{type(exc).__name__}:{exc}"
        self._runner_blocking_reasons = (reason,)
        checkpoint = self._coordinator.checkpoint
        with suppress(BaseException):
            self._append_event(
                phase=checkpoint.phase.value,
                cycle_index=checkpoint.cycle_index,
                event_type="supervised_exception_blocked",
                payload={"reason": reason, "motion_authorized": False},
            )
        # The coordinator/run writer may already be terminal after an event-store
        # failure.  Never erase the original exception with a second write error.

    def _append_event(
        self,
        *,
        phase: str,
        cycle_index: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> StopScanRunEvent:
        return self._event_sink.append_event(
            phase=phase,
            cycle_index=cycle_index,
            event_type=event_type,
            payload=payload,
        )

    def _require_recovered(self) -> None:
        if self._recovery_required:
            raise SupervisedExperimentBlocked("Recovery confirmation is required")

    def _require_not_runner_blocked(self) -> None:
        if self._runner_blocking_reasons:
            raise SupervisedExperimentBlocked(self._runner_blocking_reasons[0])

    def _notify_status(self) -> ExperimentStatusSnapshot:
        snapshot = self.status
        try:
            for callback in self._status_callbacks:
                callback(snapshot)
        except BaseException as exc:
            self._latch_exception("status_callback_failed", exc)
            return self.status
        return snapshot

    def _exclusive_operation(self):
        return _RunnerOperation(self._operation_lock)


class _RunnerOperation:
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise SupervisedExperimentBlocked("Another supervised operation is active")

    def __exit__(self, *_: object) -> None:
        self._lock.release()


def _disposition(
    checkpoint: StopScanCheckpoint,
    *,
    recovery_required: bool,
    runner_blocking_reasons: tuple[str, ...],
) -> ExperimentDisposition:
    if recovery_required or runner_blocking_reasons or checkpoint.stop_requested:
        return ExperimentDisposition.BLOCKED
    if checkpoint.phase is StopScanPhase.WAITING_APPROVAL:
        return ExperimentDisposition.WAITING_APPROVAL
    if checkpoint.phase is StopScanPhase.COMPLETE:
        return ExperimentDisposition.COMPLETE
    if checkpoint.phase in {
        StopScanPhase.BOOTSTRAP_MAP_REQUIRED,
        StopScanPhase.AWAITING_CAPTURE,
    }:
        return ExperimentDisposition.NEEDS_CAPTURE
    if checkpoint.phase in {
        StopScanPhase.MOTION_BLOCKED,
        StopScanPhase.ABORTED,
        StopScanPhase.FAILED,
    }:
        return ExperimentDisposition.BLOCKED
    return ExperimentDisposition.READY
