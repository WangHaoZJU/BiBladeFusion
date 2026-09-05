from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from biblade_fusion.storage.stop_scan_run import StopScanRunWriter, read_stop_scan_run
from biblade_fusion.supervision.experiment import ExperimentDisposition
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturePurpose,
    StopScanBlocked,
    StopScanCheckpoint,
    StopScanPhase,
)
from biblade_fusion.workflows.supervised_experiment import (
    GuardedCoordinatorMotionExecutor,
    OperatorApproval,
    OperatorRejection,
    RecoveryConfirmation,
    SupervisedExperimentRunner,
)


class _ApprovalForwardingCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def execute_approved(self, *, operator_id: str, confirmation: str) -> object:
        self.calls.append((operator_id, confirmation))
        return object()


def _checkpoint(
    phase: StopScanPhase,
    *,
    cycle_index: int = 0,
    blocking_reasons: tuple[str, ...] = (),
) -> StopScanCheckpoint:
    return StopScanCheckpoint(
        phase=phase,
        cycle_index=cycle_index,
        current_view_id=None,
        proposed_view_id="fine_001" if phase is StopScanPhase.WAITING_APPROVAL else None,
        occupancy_binding=None,
        expected_capture_view_id=("fine_001" if phase is StopScanPhase.AWAITING_CAPTURE else None),
        expected_capture_purpose=(
            CapturePurpose.CANDIDATE if phase is StopScanPhase.AWAITING_CAPTURE else None
        ),
        blocking_reasons=blocking_reasons,
    )


class _FakeCoordinator:
    def __init__(
        self,
        writer: StopScanRunWriter,
        *,
        stale: bool = False,
        capture_error: BaseException | None = None,
    ) -> None:
        self._writer = writer
        self._checkpoint = _checkpoint(StopScanPhase.IDLE)
        self.stale = stale
        self.capture_error = capture_error
        self.capture_count = 0
        self.prepare_count = 0
        self.execute_count = 0

    @property
    def checkpoint(self) -> StopScanCheckpoint:
        return self._checkpoint

    def _event(self, event_type: str) -> None:
        self._writer.append_event(
            phase=self._checkpoint.phase.value,
            cycle_index=self._checkpoint.cycle_index,
            event_type=event_type,
            payload={},
        )

    def start(self) -> StopScanCheckpoint:
        self._checkpoint = _checkpoint(StopScanPhase.BOOTSTRAP_MAP_REQUIRED)
        self._event("run_started")
        return self._checkpoint

    def start_recovery(self) -> StopScanCheckpoint:
        self._checkpoint = _checkpoint(
            StopScanPhase.MOTION_BLOCKED,
            cycle_index=self._checkpoint.cycle_index,
            blocking_reasons=("recovery_requires_fresh_safety_refresh",),
        )
        self._event("run_recovery_started")
        return self._checkpoint

    def capture_infer_update(self, view_id: str | None = None) -> object:
        if self.capture_error is not None:
            raise self.capture_error
        if view_id is None:
            raise ValueError("fake bootstrap requires a view ID")
        self.capture_count += 1
        if self.stale:
            self._checkpoint = _checkpoint(
                StopScanPhase.MOTION_BLOCKED,
                blocking_reasons=("occupancy_not_fresh_map_ready:stale",),
            )
        else:
            self._checkpoint = _checkpoint(StopScanPhase.MAP_READY)
        self._event("perception_committed")
        return object()

    def prepare_next_segment(self) -> object:
        self.prepare_count += 1
        self._checkpoint = _checkpoint(StopScanPhase.WAITING_APPROVAL)
        self._event("single_segment_waiting_approval")
        return object()

    def approval_prompt(self) -> str:
        return "approve fine_001"

    def execute_approved(self, *, operator_id: str, confirmation: str) -> object:
        assert operator_id and confirmation
        self.execute_count += 1
        self._checkpoint = replace(
            _checkpoint(StopScanPhase.AWAITING_CAPTURE, cycle_index=1),
            current_view_id="bootstrap_00",
        )
        self._event("single_segment_complete")
        return object()

    def request_stop(self, reason: str) -> StopScanCheckpoint:
        assert reason
        self._checkpoint = _checkpoint(
            StopScanPhase.ABORTED,
            blocking_reasons=(reason,),
        )
        self._checkpoint = replace(
            self._checkpoint,
            stop_requested=True,
            stop_transport_acknowledged=True,
            stop_stationarity_verified=True,
        )
        self._event("operator_stop_observed")
        return self._checkpoint


class _InjectedExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute_segment(
        self,
        coordinator: _FakeCoordinator,
        approval: OperatorApproval,
    ) -> object:
        self.calls += 1
        return coordinator.execute_approved(
            operator_id=approval.operator_id,
            confirmation=approval.confirmation,
        )


def _runner(
    tmp_path: Path,
    *,
    stale: bool = False,
    capture_error: BaseException | None = None,
    executor: _InjectedExecutor | None = None,
    status_callbacks=(),
    perception_callbacks=(),
    prepared_segment_callbacks=(),
) -> tuple[SupervisedExperimentRunner, _FakeCoordinator]:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    coordinator = _FakeCoordinator(
        writer,
        stale=stale,
        capture_error=capture_error,
    )
    runner = SupervisedExperimentRunner(
        coordinator=coordinator,
        run_writer=writer,
        motion_executor=executor,  # type: ignore[arg-type]
        status_callbacks=status_callbacks,
        perception_callbacks=perception_callbacks,
        prepared_segment_callbacks=prepared_segment_callbacks,
    )
    return runner, coordinator


def test_untyped_perception_is_rejected_before_read_only_observer(tmp_path: Path) -> None:
    observed = []
    runner, coordinator = _runner(
        tmp_path,
        perception_callbacks=(observed.append,),
    )
    runner.start()

    status = runner.step(view_id="bootstrap_00")

    assert status.disposition is ExperimentDisposition.BLOCKED
    assert "untyped perception result" in status.blocking_reasons[0]
    assert coordinator.capture_count == 1
    assert observed == []


def test_untyped_prepared_segment_is_rejected_before_observer(tmp_path: Path) -> None:
    observed = []
    runner, coordinator = _runner(
        tmp_path,
        prepared_segment_callbacks=(observed.append,),
    )
    runner.start()
    runner.step(view_id="bootstrap_00")

    status = runner.step()

    assert status.disposition is ExperimentDisposition.BLOCKED
    assert "untyped prepared segment" in status.blocking_reasons[0]
    assert coordinator.prepare_count == 1
    assert observed == []


def test_run_until_attention_preflights_once_and_never_executes(tmp_path: Path) -> None:
    snapshots = []
    executor = _InjectedExecutor()
    runner, coordinator = _runner(
        tmp_path,
        executor=executor,
        status_callbacks=(snapshots.append,),
    )

    status = runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")

    assert status.disposition is ExperimentDisposition.WAITING_APPROVAL
    assert status.motion_command_capable is False
    assert coordinator.capture_count == 1
    assert coordinator.prepare_count == 1
    assert coordinator.execute_count == 0
    assert executor.calls == 0
    assert runner.approval_prompt() == "approve fine_001"
    assert snapshots[-1] == status

    executed = runner.execute_approved_segment(OperatorApproval("operator-1", "approve fine_001"))

    assert executed.disposition is ExperimentDisposition.NEEDS_CAPTURE
    assert coordinator.execute_count == 1
    assert executor.calls == 1


def test_guarded_executor_exactly_forwards_typed_approval() -> None:
    coordinator = _ApprovalForwardingCoordinator()
    adapter = GuardedCoordinatorMotionExecutor()
    approval = OperatorApproval("operator-7", "approve proposal sha256:abcd")

    result = adapter.execute_segment(  # type: ignore[arg-type]
        coordinator,
        approval,
    )

    assert type(result) is object
    assert coordinator.calls == [
        ("operator-7", "approve proposal sha256:abcd"),
    ]


def test_missing_executor_blocks_without_delegating_motion(tmp_path: Path) -> None:
    runner, coordinator = _runner(tmp_path)
    waiting = runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")
    assert waiting.disposition is ExperimentDisposition.WAITING_APPROVAL

    blocked = runner.execute_approved_segment(OperatorApproval("operator-1", "approve fine_001"))

    assert blocked.disposition is ExperimentDisposition.BLOCKED
    assert blocked.blocking_reasons == ("motion_executor_not_injected",)
    assert coordinator.execute_count == 0
    assert read_stop_scan_run(blocked.run_root).latest_event.event_type == (
        "external_execution_gate_blocked"
    )


def test_operator_rejection_is_durable_and_cannot_later_execute(tmp_path: Path) -> None:
    executor = _InjectedExecutor()
    runner, coordinator = _runner(tmp_path, executor=executor)
    runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")

    blocked = runner.reject_prepared_segment(
        OperatorRejection("operator-2", "clearance visually unacceptable")
    )

    assert blocked.disposition is ExperimentDisposition.BLOCKED
    assert blocked.blocking_reasons == (
        "operator_rejected_segment:clearance visually unacceptable",
    )
    assert coordinator.execute_count == 0
    event = read_stop_scan_run(blocked.run_root).latest_event
    assert event.event_type == "external_segment_rejected"
    assert event.payload["operator_id"] == "operator-2"
    try:
        runner.execute_approved_segment(OperatorApproval("operator-2", "changed mind"))
    except Exception as exc:
        assert "operator_rejected_segment" in str(exc)
    else:  # pragma: no cover - fail-safe assertion
        raise AssertionError("A rejected segment became executable")
    assert executor.calls == 0


def test_stale_occupancy_requests_safety_refresh_before_selection_or_preflight(
    tmp_path: Path,
) -> None:
    runner, coordinator = _runner(tmp_path, stale=True)

    status = runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")

    assert status.disposition is ExperimentDisposition.NEEDS_CAPTURE
    assert status.blocking_reasons == ("occupancy_not_fresh_map_ready:stale",)
    assert coordinator.prepare_count == 0
    assert coordinator.execute_count == 0


def test_planning_deadline_requires_restart_instead_of_safety_capture(
    tmp_path: Path,
) -> None:
    runner, coordinator = _runner(tmp_path)
    runner.start()
    runner.step(view_id="bootstrap_00")

    reason = (
        "planning_deadline_exceeded:PlanningDeadlineExceeded:"
        "planning/preflight exceeded its cooperative responsiveness budget"
    )

    def deadline_blocked() -> object:
        coordinator.prepare_count += 1
        coordinator._checkpoint = _checkpoint(  # noqa: SLF001
            StopScanPhase.MOTION_BLOCKED,
            blocking_reasons=(reason,),
        )
        coordinator._event("planning_deadline_exceeded")  # noqa: SLF001
        raise StopScanBlocked(reason)

    coordinator.prepare_next_segment = deadline_blocked  # type: ignore[method-assign]

    status = runner.step()

    assert status.phase == StopScanPhase.MOTION_BLOCKED.value
    assert status.disposition is ExperimentDisposition.BLOCKED
    assert status.blocking_reasons == (f"planning_restart_required:{reason}",)
    assert coordinator.prepare_count == 1
    assert coordinator.execute_count == 0
    events = read_stop_scan_run(status.run_root).events
    assert events[-2].event_type == "planning_deadline_exceeded"
    assert events[-1].event_type == "planning_restart_required"
    assert events[-1].payload["operator_capture_can_resolve"] is False
    assert events[-1].payload["pending_motion_restored"] is False
    assert events[-1].payload["resume_preserves_accepted_science"] is True
    assert all(event.event_type != "coordinator_operation_blocked" for event in events)


def test_non_planning_stop_scan_block_remains_terminal(tmp_path: Path) -> None:
    runner, coordinator = _runner(
        tmp_path,
        capture_error=StopScanBlocked("capture evidence identity changed"),
    )

    status = runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")

    assert status.disposition is ExperimentDisposition.BLOCKED
    assert status.blocking_reasons == (
        "coordinator_blocked:capture evidence identity changed",
    )
    assert coordinator.prepare_count == 0
    assert coordinator.execute_count == 0
    assert read_stop_scan_run(status.run_root).latest_event.event_type == (
        "coordinator_operation_blocked"
    )


def test_resume_requires_stop_confirmation_and_discards_motion_authority(
    tmp_path: Path,
) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    writer.append_event(
        phase="waiting_approval",
        cycle_index=4,
        event_type="single_segment_waiting_approval",
        payload={"proposal_id": "old"},
    )
    resumed_writer = StopScanRunWriter.resume(writer.root)
    coordinator = _FakeCoordinator(resumed_writer)
    runner = SupervisedExperimentRunner(
        coordinator=coordinator,
        run_writer=resumed_writer,
        recovery_required=True,
    )

    assert runner.status.disposition is ExperimentDisposition.BLOCKED
    assert runner.status.recovery_required is True
    rejected = runner.acknowledge_recovery(RecoveryConfirmation("operator-1", False, True))
    assert rejected.disposition is ExperimentDisposition.BLOCKED
    assert coordinator.execute_count == 0

    # A new runner represents a fresh process after the rejected recovery attempt.
    resumed_writer = StopScanRunWriter.resume(writer.root)
    coordinator = _FakeCoordinator(resumed_writer)
    runner = SupervisedExperimentRunner(
        coordinator=coordinator,
        run_writer=resumed_writer,
        recovery_required=True,
    )
    recovered = runner.acknowledge_recovery(RecoveryConfirmation("operator-1", True, True))

    assert recovered.disposition is ExperimentDisposition.NEEDS_CAPTURE
    assert recovered.cycle_index == 0
    assert recovered.phase == StopScanPhase.MOTION_BLOCKED.value
    assert coordinator.execute_count == 0
    events = read_stop_scan_run(writer.root).events
    assert events[-2].event_type == "recovery_acknowledged"
    assert events[-2].payload["pending_motion_restored"] is False
    assert events[-1].event_type == "run_recovery_started"


def test_capture_exception_latches_block_and_persists_reason(tmp_path: Path) -> None:
    runner, coordinator = _runner(
        tmp_path,
        capture_error=RuntimeError("gpu failed"),
    )

    status = runner.run_until_attention(bootstrap_view_provider=lambda _: "bootstrap_00")

    assert status.disposition is ExperimentDisposition.BLOCKED
    assert status.blocking_reasons == ("step_failed:RuntimeError:gpu failed",)
    assert coordinator.prepare_count == 0
    assert coordinator.execute_count == 0
    assert read_stop_scan_run(status.run_root).latest_event.event_type == (
        "supervised_exception_blocked"
    )


def test_request_stop_bypasses_active_runner_operation_lock(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="run-001")
    entered = threading.Event()
    release = threading.Event()

    class BlockingCoordinator(_FakeCoordinator):
        def capture_infer_update(self, view_id: str | None = None) -> object:
            assert view_id
            entered.set()
            assert release.wait(timeout=2.0)
            return object()

    coordinator = BlockingCoordinator(writer)
    runner = SupervisedExperimentRunner(
        coordinator=coordinator,
        run_writer=writer,
    )
    runner.start()
    worker = threading.Thread(target=lambda: runner.step(view_id="bootstrap-00"))
    worker.start()
    assert entered.wait(timeout=1.0)

    stopped = runner.request_stop("operator stop")

    assert stopped.disposition is ExperimentDisposition.BLOCKED
    assert stopped.stop_requested is True
    assert stopped.stop_transport_acknowledged is True
    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()


def test_bootstrap_provider_exception_latches_without_capture(tmp_path: Path) -> None:
    runner, coordinator = _runner(tmp_path)

    def fail_provider(_):
        raise RuntimeError("operator input unavailable")

    status = runner.run_until_attention(bootstrap_view_provider=fail_provider)

    assert status.disposition is ExperimentDisposition.BLOCKED
    assert status.blocking_reasons == (
        "bootstrap_view_provider_failed:RuntimeError:operator input unavailable",
    )
    assert coordinator.capture_count == 0
    assert coordinator.execute_count == 0
