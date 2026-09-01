from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.unknown_blade_runtime as runtime_module
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.planning import BladeSide
from biblade_fusion.robotics.stationarity import (
    BootstrapSafeStateEvidence,
    StationarityEvidence,
)
from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentStatusSnapshot,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    NextViewSelection,
    NextViewTarget,
)
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarsePhase,
    CoarsePhaseTransition,
)
from biblade_fusion.workflows.unknown_blade_runtime import (
    CoarseSessionNextViewAdapter,
    UnknownBladeResumePhase,
    UnknownBladeResumePlan,
    UnknownBladeRuntimeError,
    UnknownBladeRuntimePhase,
    UnknownBladeSupervisedRuntime,
    load_unknown_blade_resume_plan,
    open_production_unknown_blade_runtime,
    run_unknown_blade_operator_console,
    unknown_blade_runtime_readiness,
)


def _selection(
    view_id: str,
    *,
    reference: str = "b" * 64,
    policy: str = "c" * 64,
) -> NextViewSelection:
    target = NextViewTarget(
        view_id,
        (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )
    return NextViewSelection(
        target,
        "a" * 64,
        reference,
        policy,
        8,
        4,
        False,
        ("synthetic coarse target",),
    )


def _transition(
    tmp_path: Path,
    phase: CoarsePhase = CoarsePhase.COLLECTING,
) -> CoarsePhaseTransition:
    if phase is CoarsePhase.READY_FOR_FINE:
        return CoarsePhaseTransition(
            phase,
            ("all schema-5 gates passed",),
            tmp_path / "generation",
            tmp_path / "ready-generation",
            tmp_path / "schema5-reference",
        )
    return CoarsePhaseTransition(
        phase,
        ("more bilateral evidence is required",),
        tmp_path / "generation",
    )


class _FakeCoarseSession:
    def __init__(
        self,
        tmp_path: Path,
        *,
        selections: list[NextViewSelection] | None = None,
        transitions: list[CoarsePhaseTransition] | None = None,
    ) -> None:
        self.current_generation_path = (tmp_path / "generation").resolve()
        self.selections = list(selections or [_selection("coarse-000")])
        self.transitions = list(transitions or [_transition(tmp_path)])
        self.select_calls = 0
        self.stage_selected_calls = 0
        self.stage_operator_calls = 0
        self.accept_calls = 0
        self.reject_calls = 0

    def stage_operator_capture(self, *, seed=None, operator_side=None) -> None:
        del seed, operator_side
        self.stage_operator_calls += 1

    def stage_selected_capture(self, selection, *, seed=None) -> None:
        del selection, seed
        self.stage_selected_calls += 1

    def accept_cycle(self, result) -> Path:
        assert result.coarse_scan_view_path is not None
        self.accept_calls += 1
        return self.current_generation_path

    def reject_cycle(self) -> None:
        self.reject_calls += 1

    def select_next(self) -> NextViewSelection:
        value = self.selections[min(self.select_calls, len(self.selections) - 1)]
        self.select_calls += 1
        return value

    def evaluate_transition(self) -> CoarsePhaseTransition:
        transition = self.transitions[min(self.accept_calls - 1, len(self.transitions) - 1)]
        if transition.ready_generation_path is not None:
            self.current_generation_path = transition.ready_generation_path
        return transition


def test_coarse_target_is_stable_across_transit_capture(tmp_path: Path) -> None:
    session = _FakeCoarseSession(tmp_path)
    adapter = CoarseSessionNextViewAdapter(session)
    adapter.bind_checkpoint_sink(lambda _generation: None)

    first = adapter.select_next(SimpleNamespace(), SimpleNamespace())
    adapter.observe_perception(SimpleNamespace(coarse_scan_view_path=None))
    second = adapter.select_next(SimpleNamespace(), SimpleNamespace())

    assert first is second
    assert session.select_calls == 1
    assert session.stage_selected_calls == 1
    assert session.accept_calls == 0


def test_coarse_run_rejects_reference_hash_change(tmp_path: Path) -> None:
    session = _FakeCoarseSession(
        tmp_path,
        selections=[
            _selection("coarse-000", reference="b" * 64),
            _selection("coarse-001", reference="d" * 64),
        ],
        transitions=[_transition(tmp_path), _transition(tmp_path)],
    )
    adapter = CoarseSessionNextViewAdapter(session)
    adapter.bind_checkpoint_sink(lambda _generation: None)
    adapter.select_next(SimpleNamespace(), SimpleNamespace())
    adapter.observe_perception(SimpleNamespace(coarse_scan_view_path=tmp_path / "coarse-view-000"))

    with pytest.raises(UnknownBladeRuntimeError, match="changed within one run"):
        adapter.select_next(SimpleNamespace(), SimpleNamespace())

    assert session.stage_selected_calls == 1


def test_schema5_handoff_never_emits_changed_coarse_completion_binding(
    tmp_path: Path,
) -> None:
    session = _FakeCoarseSession(
        tmp_path,
        transitions=[_transition(tmp_path, CoarsePhase.READY_FOR_FINE)],
    )
    adapter = CoarseSessionNextViewAdapter(session)
    adapter.bind_checkpoint_sink(lambda _generation: None)
    adapter.observe_perception(SimpleNamespace(coarse_scan_view_path=tmp_path / "coarse-view"))
    assert adapter.last_transition is None
    adapter.promote_after_exact_map_ready(lambda _result: None)

    with pytest.raises(UnknownBladeRuntimeError, match="outer fine handoff"):
        adapter.select_next(SimpleNamespace(), SimpleNamespace())

    assert session.select_calls == 0


def test_schema5_promotion_without_a_new_science_result_does_not_replay_ready(
    tmp_path: Path,
) -> None:
    session = _FakeCoarseSession(
        tmp_path,
        transitions=[_transition(tmp_path, CoarsePhase.READY_FOR_FINE)],
    )
    adapter = CoarseSessionNextViewAdapter(session)
    adapter.bind_checkpoint_sink(lambda _generation: None)
    adapter.observe_perception(
        SimpleNamespace(coarse_scan_view_path=tmp_path / "coarse-view")
    )

    first = adapter.promote_after_exact_map_ready(lambda _result: None)
    repeated = adapter.promote_after_exact_map_ready(lambda _result: None)

    assert first is not None and first.phase is CoarsePhase.READY_FOR_FINE
    assert repeated is None


def test_rejected_staged_cycle_clears_session_and_cached_target(tmp_path: Path) -> None:
    session = _FakeCoarseSession(tmp_path)
    adapter = CoarseSessionNextViewAdapter(session)
    adapter.bind_checkpoint_sink(lambda _generation: None)
    first = adapter.select_next(SimpleNamespace(), SimpleNamespace())

    adapter.reject_staged_cycle()
    second = adapter.select_next(SimpleNamespace(), SimpleNamespace())

    assert first is second
    assert session.reject_calls == 1
    assert session.select_calls == 2
    assert session.stage_selected_calls == 2


def _status(
    tmp_path: Path,
    disposition: ExperimentDisposition,
    *,
    phase: str,
    cycle: int = 0,
    expected_view: str | None = None,
    blocking: tuple[str, ...] = (),
    stop_requested: bool = False,
    stop_acknowledged: bool = False,
) -> ExperimentStatusSnapshot:
    return ExperimentStatusSnapshot(
        run_id="runtime-test",
        run_root=tmp_path / "run",
        phase=phase,
        disposition=disposition,
        cycle_index=cycle,
        current_view_id=None,
        proposed_view_id=(
            "candidate-000" if disposition is ExperimentDisposition.WAITING_APPROVAL else None
        ),
        expected_capture_view_id=expected_view,
        expected_capture_purpose=("candidate" if expected_view is not None else None),
        blocking_reasons=blocking,
        event_count=cycle,
        latest_event_sha256=None,
        recovery_required=False,
        awaiting_external_approval=(disposition is ExperimentDisposition.WAITING_APPROVAL),
        stop_requested=stop_requested,
        stop_transport_acknowledged=stop_acknowledged,
        stop_stationarity_verified=stop_acknowledged,
    )


class _FakeRunner:
    def __init__(
        self,
        tmp_path: Path,
        observer: CoarseSessionNextViewAdapter | None,
        *,
        bootstrap_ready_after: int = 3,
    ) -> None:
        self.tmp_path = tmp_path
        self.observer = observer
        self.bootstrap_ready_after = bootstrap_ready_after
        self.capture_count = 0
        self.execute_count = 0
        self.plan_count = 0
        self.stop_count = 0
        self.start_count = 0
        self.recovery_confirmations = []
        self.next_capture_purpose = "candidate"
        self.last_step_view_id: str | None = "not-called"
        self._status = _status(
            tmp_path,
            ExperimentDisposition.READY,
            phase="idle",
        )

    @property
    def status(self) -> ExperimentStatusSnapshot:
        return self._status

    def start(self) -> ExperimentStatusSnapshot:
        self.start_count += 1
        self._status = _status(
            self.tmp_path,
            ExperimentDisposition.NEEDS_CAPTURE,
            phase="bootstrap_map_required",
        )
        return self._status

    def step(self, *, view_id: str | None = None) -> ExperimentStatusSnapshot:
        self.last_step_view_id = view_id
        self.capture_count += 1
        if self.observer is not None:
            self.observer.observe_perception(
                SimpleNamespace(
                    coarse_scan_view_path=(
                        None
                        if self._status.expected_capture_purpose == "transit"
                        else self.tmp_path / f"coarse-{self.capture_count}"
                    )
                )
            )
        disposition = (
            ExperimentDisposition.READY
            if self.capture_count >= self.bootstrap_ready_after
            else ExperimentDisposition.NEEDS_CAPTURE
        )
        self._status = _status(
            self.tmp_path,
            disposition,
            phase=(
                "map_ready"
                if disposition is ExperimentDisposition.READY
                else "bootstrap_map_required"
            ),
            cycle=self.capture_count,
        )
        return self._status

    def run_until_attention(self, *, maximum_steps: int = 32) -> ExperimentStatusSnapshot:
        assert maximum_steps > 0
        self.plan_count += 1
        if self.observer is not None:
            self.observer.select_next(SimpleNamespace(), SimpleNamespace())
        self._status = _status(
            self.tmp_path,
            ExperimentDisposition.WAITING_APPROVAL,
            phase="waiting_approval",
            cycle=self.capture_count,
        )
        return self._status

    def execute_approved_segment(self, approval) -> ExperimentStatusSnapshot:
        assert approval.confirmation == "EXECUTE abcdef123456"
        self.execute_count += 1
        self._status = _status(
            self.tmp_path,
            ExperimentDisposition.NEEDS_CAPTURE,
            phase="awaiting_capture",
            cycle=self.capture_count,
            expected_view="candidate-000",
        )
        self._status = replace(
            self._status,
            expected_capture_purpose=self.next_capture_purpose,
        )
        return self._status

    def request_stop(self, reason: str) -> ExperimentStatusSnapshot:
        assert reason
        self.stop_count += 1
        self._status = _status(
            self.tmp_path,
            ExperimentDisposition.BLOCKED,
            phase="aborted",
            cycle=self.capture_count,
            blocking=(reason,),
            stop_requested=True,
            stop_acknowledged=True,
        )
        return self._status

    def acknowledge_recovery(self, confirmation) -> ExperimentStatusSnapshot:
        self.recovery_confirmations.append(confirmation)
        self._status = _status(
            self.tmp_path,
            ExperimentDisposition.NEEDS_CAPTURE,
            phase="bootstrap_map_required",
            cycle=self.capture_count,
        )
        return self._status

    def approval_prompt(self) -> str:
        return "EXECUTE abcdef123456"


class _FakeExperimentHandoff:
    def __init__(
        self,
        *,
        fail_prepare: bool = False,
        fail_fine_started: bool = False,
        fail_fine_completed: bool = False,
    ) -> None:
        self.fail_prepare = fail_prepare
        self.fail_fine_started = fail_fine_started
        self.fail_fine_completed = fail_fine_completed
        self.prepared: tuple[Path, Path] | None = None
        self.fine_run_root: Path | None = None
        self.fine_start_candidates: list[Path] = []
        self.coarse_checkpoints: list[Path] = []
        self.fine_checkpoints: list[Path] = []
        self.fine_completed: list[tuple[Path, Path]] = []

    def append_coarse_checkpoint(self, *, coarse_generation):
        path = Path(coarse_generation).resolve()
        self.coarse_checkpoints.append(path)
        return SimpleNamespace(event_sha256="0" * 64)

    def prepare_handoff(self, *, schema5_generation, reference_coarse_model):
        if self.fail_prepare:
            raise RuntimeError("synthetic PREPARED persistence failure")
        self.prepared = (
            Path(schema5_generation).resolve(),
            Path(reference_coarse_model).resolve(),
        )
        return SimpleNamespace(event_sha256="a" * 64)

    def append_fine_start_candidate(self, *, fine_run_root):
        assert self.prepared is not None
        path = Path(fine_run_root).resolve()
        self.fine_start_candidates.append(path)
        return SimpleNamespace(event_sha256="b" * 64)

    def append_fine_started(self, *, timing_scope, budget_check):
        if self.fail_fine_started:
            raise RuntimeError("synthetic FINE_STARTED persistence failure")
        assert self.prepared is not None
        assert self.fine_start_candidates
        assert timing_scope in {"uninterrupted_total", "resume_fine_start"}
        budget_check()
        budget_check()
        self.fine_run_root = self.fine_start_candidates[-1]
        return SimpleNamespace(event_sha256="c" * 64)

    def append_unaccepted_fine_started(self, *, fine_run_root):
        assert self.prepared is not None
        path = Path(fine_run_root).resolve()
        self.fine_run_root = path
        return SimpleNamespace(event_sha256="c" * 64)

    def append_fine_checkpoint(self, *, accepted_surface_coverage_generation):
        path = Path(accepted_surface_coverage_generation).resolve()
        self.fine_checkpoints.append(path)
        return SimpleNamespace(event_sha256="d" * 64)

    def append_fine_completed(
        self,
        *,
        final_surface_coverage_generation,
        final_reconstruction_product,
    ):
        if self.fail_fine_completed:
            raise RuntimeError("synthetic FINE_COMPLETED persistence failure")
        value = (
            Path(final_surface_coverage_generation).resolve(),
            Path(final_reconstruction_product).resolve(),
        )
        self.fine_completed.append(value)
        return SimpleNamespace(event_sha256="e" * 64)


def _runtime(
    tmp_path: Path,
    *,
    transitions: list[CoarsePhaseTransition] | None = None,
    fine_factory=lambda _reference: (_ for _ in ()).throw(AssertionError("no fine")),
    map_ready_assertion=lambda _result: None,
    experiment_handoff: _FakeExperimentHandoff | None = None,
    resume_phase: UnknownBladeResumePhase | None = None,
    recovered_reference: Path | None = None,
    recovered_fine_runner: _FakeRunner | None = None,
    maximum_schema5_handoff_duration_s: float | None = None,
    experimental: bool = False,
    monotonic_clock=None,
) -> tuple[UnknownBladeSupervisedRuntime, _FakeRunner, _FakeCoarseSession]:
    session = _FakeCoarseSession(tmp_path, transitions=transitions)
    adapter = CoarseSessionNextViewAdapter(session)
    runner = _FakeRunner(tmp_path, adapter)
    state = RobotState(
        monotonic_time_ns=1_000_000_000,
        controller_time_s=1.0,
        joint_positions_rad=np.zeros(6),
        base_t_tcp=PoseSE3.from_rotation_translation(
            "base", "tcp", np.eye(3), np.zeros(3)
        ),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.0,
    )
    stationarity = StationarityEvidence(
        final_state=state,
        sample_count=3,
        duration_s=0.5,
        controller_duration_s=0.5,
        max_sample_gap_s=0.25,
        max_joint_delta_rad=0.0,
        max_tcp_translation_delta_m=0.0,
        max_tcp_rotation_delta_rad=0.0,
        goal_error_rad=0.0,
    )
    bootstrap_evidence = BootstrapSafeStateEvidence(
        stationarity=stationarity,
        stop_generation=1,
        runtime_state="STOPPED",
        robot_mode="IDLE",
        safety_status="NORMAL",
        max_actual_joint_velocity_rad_s=0.0,
        max_target_joint_velocity_rad_s=0.0,
        max_actual_tcp_linear_velocity_m_s=0.0,
        max_actual_tcp_angular_velocity_rad_s=0.0,
        max_target_tcp_linear_velocity_m_s=0.0,
        max_target_tcp_angular_velocity_rad_s=0.0,
    )
    runtime_kwargs = {}
    if monotonic_clock is not None:
        runtime_kwargs["monotonic_clock"] = monotonic_clock
    runtime = UnknownBladeSupervisedRuntime(
        coarse_runner=runner,
        coarse_adapter=adapter,
        create_fine_runner=fine_factory,
        operator_id="operator-1",
        minimum_operator_bootstrap_views=3,
        timeline_root=tmp_path / "timeline",
        establish_initial_stop=lambda: bootstrap_evidence,
        assert_exact_map_ready=map_ready_assertion,
        experiment_handoff=experiment_handoff or _FakeExperimentHandoff(),
        resume_phase=resume_phase,
        recovered_reference=recovered_reference,
        recovered_fine_runner=recovered_fine_runner,
        maximum_schema5_handoff_duration_s=maximum_schema5_handoff_duration_s,
        experimental=experimental,
        **runtime_kwargs,
    )
    return runtime, runner, session


def test_operator_must_trigger_each_initial_capture(tmp_path: Path) -> None:
    runtime, runner, session = _runtime(tmp_path)
    runtime.start()

    with pytest.raises(UnknownBladeRuntimeError, match="Press c"):
        runtime.advance_to_attention()

    for index in range(3):
        snapshot = runtime.capture_operator_view(view_id=f"manual-{index}")

    assert snapshot.phase is UnknownBladeRuntimePhase.COARSE_SCAN
    assert snapshot.operator_bootstrap_views == 3
    assert runner.capture_count == 3
    assert runner.execute_count == 0
    assert session.stage_operator_calls == 3


def test_operator_bootstrap_capture_can_directly_activate_ready_schema5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, coarse_runner, session = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
    )
    runtime.start()
    runtime.capture_operator_view()
    runtime.capture_operator_view()

    snapshot = runtime.capture_operator_view()

    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN
    assert snapshot.reference_coarse_model_path == (
        tmp_path / "schema5-reference"
    ).resolve()
    assert session.accept_calls == 3
    assert coarse_runner.stop_count == 1
    assert fine_runner.start_count == 1
    assert fine_runner.capture_count == 1


def test_experimental_runtime_enters_unaccepted_fine_without_release_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    handoff = _FakeExperimentHandoff()
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, coarse_runner, _session = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
        experiment_handoff=handoff,
        experimental=True,
    )
    runtime.start()
    runtime.capture_operator_view()
    runtime.capture_operator_view()

    snapshot = runtime.capture_operator_view()

    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN
    assert snapshot.reference_coarse_model_path == (
        tmp_path / "schema5-reference"
    ).resolve()
    assert coarse_runner.stop_count == 1
    assert fine_runner.start_count == 1
    assert fine_runner.capture_count == 1
    assert handoff.fine_run_root == (tmp_path / "fine" / "run").resolve()
    assert handoff.prepared == (
        (tmp_path / "ready-generation").resolve(),
        (tmp_path / "schema5-reference").resolve(),
    )


def test_schema_promotion_refuses_a_result_that_is_not_the_exact_map_ready_authority(
    tmp_path: Path,
) -> None:
    def reject(_result) -> None:
        raise UnknownBladeRuntimeError("published MAP_READY authority differs")

    runtime, _, session = _runtime(tmp_path, map_ready_assertion=reject)
    runtime.start()
    runtime.capture_operator_view()
    runtime.capture_operator_view()

    snapshot = runtime.capture_operator_view()

    assert session.accept_calls == 3
    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "MAP_READY authority" in str(snapshot.blocking_reason)
    assert runtime.snapshot.current_coarse_generation_path == (
        tmp_path / "generation"
    ).resolve()


def test_first_operator_capture_rejects_explicit_back_before_staging(
    tmp_path: Path,
) -> None:
    runtime, runner, session = _runtime(tmp_path)
    runtime.start()

    with pytest.raises(UnknownBladeRuntimeError, match="first operator capture"):
        runtime.capture_operator_view(operator_side=BladeSide.BACK)

    assert runner.capture_count == 0
    assert session.stage_operator_calls == 0


def test_exact_approval_executes_one_leg_then_auto_captures_and_preflights(
    tmp_path: Path,
) -> None:
    runtime, runner, _ = _runtime(tmp_path)
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    with pytest.raises(UnknownBladeRuntimeError, match="Approval mismatch"):
        runtime.execute_exact_approval("yes")
    assert runner.execute_count == 0

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.runner_status.disposition is ExperimentDisposition.WAITING_APPROVAL
    assert runner.execute_count == 1
    assert runner.capture_count == 4
    assert runner.last_step_view_id is None
    assert runner.plan_count == 2


def test_candidate_auto_capture_requires_one_accepted_coarse_cycle(
    tmp_path: Path,
) -> None:
    runtime, runner, session = _runtime(tmp_path)
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()
    runner.observer = None

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert session.reject_calls >= 1


def test_successful_transit_capture_retains_same_staged_coarse_target(
    tmp_path: Path,
) -> None:
    runtime, runner, session = _runtime(tmp_path)
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()
    runner.next_capture_purpose = "transit"

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.runner_status.disposition is ExperimentDisposition.WAITING_APPROVAL
    assert session.select_calls == 1
    assert session.reject_calls == 0


def test_operator_stop_remains_a_graceful_stopped_phase(tmp_path: Path) -> None:
    runtime, runner, _ = _runtime(tmp_path)
    runtime.start()

    snapshot = runtime.request_stop("operator requested stop")

    assert snapshot.phase is UnknownBladeRuntimePhase.STOPPED
    assert runner.stop_count == 1


def test_stop_failure_never_reports_stopped_and_retry_reaches_transport(
    tmp_path: Path,
) -> None:
    runtime, runner, _ = _runtime(tmp_path)
    runtime.start()
    original = runner.request_stop
    calls = 0

    def fail_once(reason: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transport rejected")
        return original(reason)

    runner.request_stop = fail_once  # type: ignore[method-assign]

    with pytest.raises(UnknownBladeRuntimeError, match="stop_failed"):
        runtime.request_stop("operator requested stop")
    assert runtime.snapshot.phase is UnknownBladeRuntimePhase.BLOCKED

    stopped = runtime.request_stop("operator retry stop")

    assert calls == 2
    assert stopped.phase is UnknownBladeRuntimePhase.STOPPED


def test_console_q_stops_without_capture_or_motion(tmp_path: Path) -> None:
    runtime, runner, _ = _runtime(tmp_path)
    output = []

    result = run_unknown_blade_operator_console(
        runtime,
        input_fn=lambda _prompt: "q",
        output_fn=output.append,
    )

    assert result == 0
    assert runner.capture_count == 0
    assert runner.execute_count == 0
    assert runner.stop_count == 1
    assert any("Read-only GUI" in line for line in output)


def test_cleanup_attempts_physical_stop_when_runtime_stop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FailingRuntime:
        snapshot = SimpleNamespace(phase=UnknownBladeRuntimePhase.COARSE_SCAN)

        def request_stop(self, reason: str):
            assert reason
            calls.append("runtime_stop")
            raise RuntimeError("coordinator stop failed")

    class Arm:
        def stop(self) -> None:
            calls.append("arm_stop")

    monkeypatch.setattr(
        runtime_module,
        "wait_until_settled",
        lambda *_args, **_kwargs: calls.append("stationarity"),
    )
    envelope = SimpleNamespace(
        maximum_feedback_interval_s=0.01,
        maximum_stopped_actual_joint_velocity_rad_s=0.001,
        maximum_stopped_target_joint_velocity_rad_s=0.001,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.001,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.001,
    )

    with pytest.raises(RuntimeError, match="coordinator stop failed"):
        runtime_module._finalize_production_runtime(
            FailingRuntime(),
            Arm(),
            load_settings("configs/default.yaml"),
            envelope,
        )

    assert calls == ["runtime_stop", "arm_stop", "stationarity"]


def test_cleanup_skips_invalid_runner_transition_after_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FailedRuntime:
        snapshot = SimpleNamespace(
            phase=UnknownBladeRuntimePhase.BLOCKED,
            runner_status=SimpleNamespace(phase="failed"),
        )

        def request_stop(self, _reason: str):
            calls.append("runtime_stop")

    class Arm:
        def stop(self) -> None:
            calls.append("arm_stop")

    monkeypatch.setattr(
        runtime_module,
        "wait_until_settled",
        lambda *_args, **_kwargs: calls.append("stationarity"),
    )
    envelope = SimpleNamespace(
        maximum_feedback_interval_s=0.01,
        maximum_stopped_actual_joint_velocity_rad_s=0.001,
        maximum_stopped_target_joint_velocity_rad_s=0.001,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.001,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.001,
    )

    runtime_module._finalize_production_runtime(
        FailedRuntime(),
        Arm(),
        load_settings("configs/default.yaml"),
        envelope,
    )

    assert calls == ["arm_stop", "stationarity"]


def test_cleanup_reports_each_stop_failure_without_hiding_details() -> None:
    class FailingRuntime:
        snapshot = SimpleNamespace(phase=UnknownBladeRuntimePhase.COARSE_SCAN)

        def request_stop(self, _reason: str):
            raise RuntimeError("coordinator stop failed")

    class FailingArm:
        def stop(self) -> None:
            raise RuntimeError("dashboard stop failed")

    envelope = SimpleNamespace(
        maximum_feedback_interval_s=0.01,
        maximum_stopped_actual_joint_velocity_rad_s=0.001,
        maximum_stopped_target_joint_velocity_rad_s=0.001,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.001,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.001,
    )

    with pytest.raises(UnknownBladeRuntimeError) as captured:
        runtime_module._finalize_production_runtime(
            FailingRuntime(),
            FailingArm(),
            load_settings("configs/default.yaml"),
            envelope,
        )

    message = str(captured.value)
    assert "coordinator stop failed" in message
    assert "dashboard stop failed" in message


def test_combined_runtime_failure_reports_primary_and_cleanup_errors() -> None:
    with pytest.raises(UnknownBladeRuntimeError) as captured:
        runtime_module._raise_combined_runtime_failure(
            RuntimeError("bootstrap failed"),
            RuntimeError("cleanup failed"),
        )

    message = str(captured.value)
    assert "bootstrap failed" in message
    assert "cleanup failed" in message


def test_ready_schema5_switches_to_fine_runner_without_coarse_completion_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    handoff = _FakeExperimentHandoff()
    runtime, coarse_runner, session = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda reference: (
            fine_runner
            if reference == (tmp_path / "schema5-reference").resolve()
            else (_ for _ in ()).throw(AssertionError("wrong reference"))
        ),
        experiment_handoff=handoff,
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN
    assert snapshot.reference_coarse_model_path == (tmp_path / "schema5-reference").resolve()
    assert fine_runner.capture_count == 1
    assert fine_runner.last_step_view_id == "fine_transition_bootstrap_000"
    assert fine_runner.plan_count == 1
    assert coarse_runner.execute_count == 1
    assert session.select_calls == 1
    assert handoff.prepared == (
        (tmp_path / "ready-generation").resolve(),
        (tmp_path / "schema5-reference").resolve(),
    )
    assert handoff.fine_run_root == (tmp_path / "fine" / "run").resolve()


def test_schema5_handoff_timing_overrun_stops_and_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    clock = SimpleNamespace(value=0.0)
    runtime, coarse_runner, session = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: (_ for _ in ()).throw(
            AssertionError("fine factory must not run after timing overrun")
        ),
        maximum_schema5_handoff_duration_s=0.5,
        monotonic_clock=lambda: clock.value,
    )
    original_evaluate = session.evaluate_transition

    def slow_schema5_evaluation():
        transition = original_evaluate()
        if transition.phase is CoarsePhase.READY_FOR_FINE:
            clock.value += 0.500001
        return transition

    session.evaluate_transition = slow_schema5_evaluation  # type: ignore[method-assign]
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "schema-5 handoff exceeded accepted timing budget" in str(
        snapshot.blocking_reason
    )
    assert coarse_runner.stop_count == 1


def test_prepared_write_failure_never_constructs_or_activates_fine_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, coarse_runner, _ = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
        experiment_handoff=_FakeExperimentHandoff(fail_prepare=True),
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "PREPARED persistence failure" in str(snapshot.blocking_reason)
    assert snapshot.reference_coarse_model_path is None
    assert fine_runner.start_count == 0
    assert fine_runner.stop_count == 0
    assert snapshot.runner_status is coarse_runner.status


def test_fine_started_write_failure_stops_orphan_and_keeps_coarse_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, coarse_runner, _ = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
        experiment_handoff=_FakeExperimentHandoff(fail_fine_started=True),
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "FINE_STARTED persistence failure" in str(snapshot.blocking_reason)
    assert snapshot.reference_coarse_model_path is None
    assert fine_runner.start_count == 1
    assert fine_runner.stop_count == 1
    assert snapshot.runner_status is coarse_runner.status


def test_fine_started_persistence_overrun_never_activates_fine_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    clock = SimpleNamespace(value=0.0)
    handoff = _FakeExperimentHandoff()
    original_append = handoff.append_fine_start_candidate

    def slow_append(*, fine_run_root):
        event = original_append(fine_run_root=fine_run_root)
        clock.value = 0.500001
        return event

    handoff.append_fine_start_candidate = slow_append  # type: ignore[method-assign]
    runtime, coarse_runner, _ = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
        experiment_handoff=handoff,
        maximum_schema5_handoff_duration_s=0.5,
        monotonic_clock=lambda: clock.value,
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "schema-5 handoff exceeded accepted timing budget" in str(
        snapshot.blocking_reason
    )
    assert snapshot.reference_coarse_model_path is None
    assert fine_runner.start_count == 1
    assert fine_runner.stop_count == 1
    assert handoff.fine_start_candidates == [(tmp_path / "fine" / "run").resolve()]
    assert handoff.fine_run_root is None
    assert snapshot.runner_status is coarse_runner.status


def test_fine_source_window_can_be_replenished_by_one_explicit_stopped_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=1)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, _, _ = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()
    runtime.execute_exact_approval("EXECUTE abcdef123456")
    fine_runner._status = _status(
        tmp_path / "fine",
        ExperimentDisposition.NEEDS_CAPTURE,
        phase="bootstrap_map_required",
        cycle=fine_runner.capture_count,
    )

    snapshot = runtime.capture_fine_source_replenishment(view_id="fine-refresh-000")

    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN
    assert snapshot.fine_source_replenishment_views == 1
    assert fine_runner.last_step_view_id == "fine-refresh-000"
    assert snapshot.runner_status.disposition is ExperimentDisposition.READY


def test_expired_coarse_window_enters_fine_replenishment_instead_of_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transitions = [
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path),
        _transition(tmp_path, CoarsePhase.READY_FOR_FINE),
    ]
    fine_runner = _FakeRunner(tmp_path / "fine", None, bootstrap_ready_after=3)
    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    runtime, _, _ = _runtime(
        tmp_path,
        transitions=transitions,
        fine_factory=lambda _reference: fine_runner,
    )
    runtime.start()
    for _ in range(3):
        runtime.capture_operator_view()
    runtime.advance_to_attention()

    snapshot = runtime.execute_exact_approval("EXECUTE abcdef123456")

    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN
    assert snapshot.runner_status.disposition is ExperimentDisposition.NEEDS_CAPTURE
    assert fine_runner.capture_count == 1

    snapshot = runtime.capture_fine_source_replenishment(view_id="fine-refresh-001")
    assert snapshot.runner_status.disposition is ExperimentDisposition.NEEDS_CAPTURE
    snapshot = runtime.capture_fine_source_replenishment(view_id="fine-refresh-002")
    assert snapshot.runner_status.disposition is ExperimentDisposition.READY
    assert snapshot.fine_source_replenishment_views == 2


@pytest.mark.parametrize(
    "resume_phase",
    [UnknownBladeResumePhase.COARSE, UnknownBladeResumePhase.FINE],
)
def test_resume_requires_fresh_stop_and_discards_pending_motion_authority(
    tmp_path: Path,
    resume_phase: UnknownBladeResumePhase,
) -> None:
    fine = _FakeRunner(tmp_path / "fine", None)
    recovered_fine = fine if resume_phase is UnknownBladeResumePhase.FINE else None
    runtime, coarse, _ = _runtime(
        tmp_path,
        resume_phase=resume_phase,
        recovered_reference=(
            tmp_path / "reference"
            if resume_phase is UnknownBladeResumePhase.FINE
            else None
        ),
        recovered_fine_runner=recovered_fine,
    )

    snapshot = runtime.start()
    active = fine if recovered_fine is not None else coarse

    assert active.start_count == 0
    assert len(active.recovery_confirmations) == 1
    confirmation = active.recovery_confirmations[0]
    assert confirmation.operator_id == "operator-1"
    assert confirmation.physical_stop_confirmed is True
    assert confirmation.discard_pending_motion is True
    assert snapshot.runner_status.disposition is ExperimentDisposition.NEEDS_CAPTURE
    assert active.execute_count == 0


def test_prepared_resume_starts_new_fine_run_then_binds_it(
    tmp_path: Path,
) -> None:
    fine = _FakeRunner(tmp_path / "fine-recovery", None)
    handoff = _FakeExperimentHandoff()
    handoff.prepared = (tmp_path / "schema5", tmp_path / "reference")
    runtime, _coarse, _ = _runtime(
        tmp_path,
        experiment_handoff=handoff,
        resume_phase=UnknownBladeResumePhase.PREPARED,
        recovered_reference=tmp_path / "reference",
        recovered_fine_runner=fine,
    )

    snapshot = runtime.start()

    assert fine.start_count == 1
    assert fine.recovery_confirmations == []
    assert handoff.fine_run_root == (tmp_path / "fine-recovery" / "run").resolve()
    assert snapshot.phase is UnknownBladeRuntimePhase.FINE_SCAN


def test_prepared_resume_timing_overrun_stops_and_blocks_before_binding(
    tmp_path: Path,
) -> None:
    fine = _FakeRunner(tmp_path / "fine-recovery", None)
    clock = SimpleNamespace(value=0.0)
    original_start = fine.start

    def slow_start():
        clock.value = 0.500001
        return original_start()

    fine.start = slow_start  # type: ignore[method-assign]
    handoff = _FakeExperimentHandoff()
    handoff.prepared = (tmp_path / "schema5", tmp_path / "reference")
    runtime, _coarse, _ = _runtime(
        tmp_path,
        experiment_handoff=handoff,
        resume_phase=UnknownBladeResumePhase.PREPARED,
        recovered_reference=tmp_path / "reference",
        recovered_fine_runner=fine,
        maximum_schema5_handoff_duration_s=0.5,
        monotonic_clock=lambda: clock.value,
    )

    snapshot = runtime.start()

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "schema-5 handoff exceeded accepted timing budget" in str(
        snapshot.blocking_reason
    )
    assert handoff.fine_run_root is None
    assert fine.stop_count == 1


def test_prepared_resume_persistence_overrun_blocks_after_durable_binding(
    tmp_path: Path,
) -> None:
    fine = _FakeRunner(tmp_path / "fine-recovery", None)
    clock = SimpleNamespace(value=0.0)
    handoff = _FakeExperimentHandoff()
    handoff.prepared = (tmp_path / "schema5", tmp_path / "reference")
    original_append = handoff.append_fine_start_candidate

    def slow_append(*, fine_run_root):
        event = original_append(fine_run_root=fine_run_root)
        clock.value = 0.500001
        return event

    handoff.append_fine_start_candidate = slow_append  # type: ignore[method-assign]
    runtime, _coarse, _ = _runtime(
        tmp_path,
        experiment_handoff=handoff,
        resume_phase=UnknownBladeResumePhase.PREPARED,
        recovered_reference=tmp_path / "reference",
        recovered_fine_runner=fine,
        maximum_schema5_handoff_duration_s=0.5,
        monotonic_clock=lambda: clock.value,
    )

    snapshot = runtime.start()

    assert snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert "schema-5 handoff exceeded accepted timing budget" in str(
        snapshot.blocking_reason
    )
    assert handoff.fine_start_candidates == [
        (tmp_path / "fine-recovery" / "run").resolve()
    ]
    assert handoff.fine_run_root is None
    assert fine.stop_count == 1


def test_fine_complete_requires_replay_and_outer_chain_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine = _FakeRunner(tmp_path / "fine", None)
    handoff = _FakeExperimentHandoff()
    runtime, _coarse, _ = _runtime(
        tmp_path,
        experiment_handoff=handoff,
        resume_phase=UnknownBladeResumePhase.FINE,
        recovered_reference=tmp_path / "reference",
        recovered_fine_runner=fine,
    )
    runtime.start()
    event_sha = "e" * 64
    coverage_root = (tmp_path / "final-coverage").resolve()
    reconstruction_root = (tmp_path / "final-reconstruction").resolve()
    terminal = SimpleNamespace(
        event_sha256=event_sha,
        event_type="coverage_complete",
        phase="complete",
        cycle_index=7,
        payload={
            "surface_generation_id": "coverage-final",
            "final_reconstruction": {
                "path": str(reconstruction_root),
                "artifact_id": "a" * 64,
                "metadata_sha256": "b" * 64,
            },
        },
    )
    complete = replace(
        _status(
            tmp_path / "fine",
            ExperimentDisposition.COMPLETE,
            phase="complete",
            cycle=7,
        ),
        latest_event_sha256=event_sha,
        event_count=8,
    )
    fine._status = complete
    monkeypatch.setattr(
        runtime_module,
        "read_stop_scan_run",
        lambda _path: SimpleNamespace(
            run_id=complete.run_id,
            events=(terminal,),
            latest_event=terminal,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "replay_final_fine_reconstruction",
        lambda _path: SimpleNamespace(
            root=reconstruction_root,
            artifact_id="a" * 64,
            metadata_sha256="b" * 64,
            result=SimpleNamespace(
                coverage=SimpleNamespace(
                    root=coverage_root,
                    generation_id="coverage-final",
                )
            ),
        ),
    )

    runtime._apply_terminal_status(complete)

    assert runtime.snapshot.phase is UnknownBladeRuntimePhase.COMPLETE
    assert handoff.fine_checkpoints == [coverage_root]
    assert handoff.fine_completed == [(coverage_root, reconstruction_root)]


def test_fine_complete_never_surfaces_before_outer_chain_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine = _FakeRunner(tmp_path / "fine", None)
    handoff = _FakeExperimentHandoff(fail_fine_completed=True)
    runtime, _coarse, _ = _runtime(
        tmp_path,
        experiment_handoff=handoff,
        resume_phase=UnknownBladeResumePhase.FINE,
        recovered_reference=tmp_path / "reference",
        recovered_fine_runner=fine,
    )
    runtime.start()
    event_sha = "e" * 64
    terminal = SimpleNamespace(
        event_sha256=event_sha,
        event_type="coverage_complete",
        phase="complete",
        payload={
            "surface_generation_id": "coverage-final",
            "final_reconstruction": {
                "path": str(tmp_path / "final-reconstruction"),
                "artifact_id": "a" * 64,
                "metadata_sha256": "b" * 64,
            },
        },
    )
    complete = replace(
        _status(
            tmp_path / "fine",
            ExperimentDisposition.COMPLETE,
            phase="complete",
        ),
        latest_event_sha256=event_sha,
    )
    fine._status = complete
    monkeypatch.setattr(
        runtime_module,
        "read_stop_scan_run",
        lambda _path: SimpleNamespace(
            run_id=complete.run_id,
            events=(terminal,),
            latest_event=terminal,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "replay_final_fine_reconstruction",
        lambda _path: SimpleNamespace(
            root=(tmp_path / "final-reconstruction").resolve(),
            artifact_id="a" * 64,
            metadata_sha256="b" * 64,
            result=SimpleNamespace(
                coverage=SimpleNamespace(
                    root=(tmp_path / "final-coverage").resolve(),
                    generation_id="coverage-final",
                )
            ),
        ),
    )

    runtime._apply_terminal_status(complete)

    assert runtime.snapshot.phase is UnknownBladeRuntimePhase.BLOCKED
    assert handoff.fine_completed == []


def test_resume_plan_is_derived_only_from_explicit_handoff_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root = tmp_path / "experiment"
    coarse_run = (experiment_root / "runs" / "coarse").resolve()
    coarse_generation = (experiment_root / "coarse-generation").resolve()
    schema5 = (experiment_root / "schema5").resolve()
    reference = (experiment_root / "reference").resolve()
    fine_run = (experiment_root / "runs" / "fine").resolve()
    coverage = (experiment_root / "coverage").resolve()
    reconstruction = (experiment_root / "reconstruction").resolve()

    def event(event_type: str, payload: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(event_type=event_type, payload=payload)

    initialized = event(
        "experiment_initialized",
        {"coarse_run_root": str(coarse_run)},
    )
    checkpoint = event(
        "coarse_checkpoint",
        {"coarse_generation": {"root": str(coarse_generation)}},
    )
    prepared = event(
        "handoff_prepared",
        {
            "schema5_generation": {"root": str(schema5)},
            "reference_coarse_model": {"root": str(reference)},
        },
    )
    candidate = event(
        "fine_start_candidate",
        {"fine_run_root": str(fine_run)},
    )
    started = event("fine_started", {"fine_run_root": str(fine_run)})
    fine_checkpoint = event(
        "fine_checkpoint",
        {
            "accepted_surface_coverage_generation": {"root": str(coverage)},
            "fine_event_count": 4,
            "fine_last_event_sha256": "f" * 64,
        },
    )
    completed = event(
        "fine_completed",
        {
            "final_surface_coverage_generation": {"root": str(coverage)},
            "final_reconstruction_product": {"root": str(reconstruction)},
        },
    )
    cases = (
        ((initialized,), UnknownBladeResumePhase.COARSE),
        ((initialized, checkpoint, prepared), UnknownBladeResumePhase.PREPARED),
        (
            (initialized, checkpoint, prepared, candidate),
            UnknownBladeResumePhase.PREPARED,
        ),
        (
            (
                initialized,
                checkpoint,
                prepared,
                candidate,
                started,
                fine_checkpoint,
            ),
            UnknownBladeResumePhase.FINE,
        ),
        (
            (
                initialized,
                checkpoint,
                prepared,
                candidate,
                started,
                fine_checkpoint,
                completed,
            ),
            UnknownBladeResumePhase.COMPLETE,
        ),
    )
    for events, expected_phase in cases:
        monkeypatch.setattr(
            runtime_module,
            "read_unknown_blade_experiment",
            lambda _path, values=events: SimpleNamespace(
                experiment_id="runtime-test",
                placement_id="blade-placement-runtime-test",
                events=values,
                fine_start_protocol=runtime_module.UNKNOWN_BLADE_FINE_START_PROTOCOL,
            ),
        )
        plan = load_unknown_blade_resume_plan(experiment_root)
        assert plan.phase is expected_phase
        assert plan.coarse_run_root == coarse_run
        assert plan.placement_id == "blade-placement-runtime-test"


def test_prepared_recovery_never_scans_or_reuses_an_orphan_fine_directory(
    tmp_path: Path,
) -> None:
    orphan = tmp_path / "runs" / "fine"
    orphan.mkdir(parents=True)
    (orphan / "unbound-event.json").write_text("orphan\n", encoding="utf-8")

    first = runtime_module._new_fine_recovery_run_root(tmp_path)
    second = runtime_module._new_fine_recovery_run_root(tmp_path)

    assert first.parent == orphan.parent
    assert first.name.startswith("fine_recovery_")
    assert first != orphan
    assert second != first
    assert not first.exists()
    assert (orphan / "unbound-event.json").read_text(encoding="utf-8") == "orphan\n"


def test_fine_checkpoint_records_run_advance_even_when_coverage_path_is_same(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _FakeExperimentHandoff()
    verify_calls: list[str] = []
    recorder = runtime_module._FineCheckpointRecorder(
        chain,
        verify_chain=lambda: verify_calls.append("verified"),
    )
    coverage = (tmp_path / "coverage").resolve()
    run_root = (tmp_path / "fine-run").resolve()
    reads = iter(
        (
            SimpleNamespace(
                events=(object(), object()),
                latest_event=SimpleNamespace(event_sha256="a" * 64),
            ),
            SimpleNamespace(
                events=(object(), object(), object()),
                latest_event=SimpleNamespace(event_sha256="b" * 64),
            ),
            SimpleNamespace(
                events=(object(), object(), object()),
                latest_event=SimpleNamespace(event_sha256="b" * 64),
            ),
        )
    )
    monkeypatch.setattr(
        runtime_module,
        "read_stop_scan_run",
        lambda _root: next(reads),
    )
    result = SimpleNamespace(coverage_path=coverage)

    assert recorder.record(result, fine_run_root=run_root) is True
    assert recorder.record(result, fine_run_root=run_root) is True
    assert recorder.record(result, fine_run_root=run_root) is False
    assert chain.fine_checkpoints == [coverage, coverage]
    assert verify_calls == ["verified", "verified"]


def test_experimental_open_keeps_science_authority_and_settings_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "occupancy": settings.occupancy.model_copy(
                update={
                    "workspace_bounds_min_m": (-1.0, -1.0, -1.0),
                    "workspace_bounds_max_m": (1.0, 1.0, 1.0),
                }
            ),
            "kinematics": settings.kinematics.model_copy(
                update={"model_path": tmp_path / "robot.urdf"}
            ),
        }
    )
    captured: dict[str, object] = {}

    class ExpectedEngineConstruction(Exception):
        pass

    class FakeArm:
        def __init__(self, _config) -> None:
            pass

        def connect(self, *, with_driver: bool) -> None:
            assert with_driver is False

        def release(self) -> None:
            pass

        def read_state(self):
            return SimpleNamespace(joint_positions_rad=(0.0,) * 6)

    class FakeCamera:
        def __init__(self, _config) -> None:
            pass

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

    hand_eye = SimpleNamespace(require_flange_primary=lambda: None)
    checker = SimpleNamespace(
        collision_model_id="collision-model",
        collision_model_hash="collision-hash",
        robot_geometry_hash="geometry-hash",
    )
    coarse_session = SimpleNamespace(prepare_engine_cycle=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        runtime_module,
        "require_unknown_blade_runtime_ready",
        lambda _settings, *, require_release_acceptance: (),
    )
    monkeypatch.setattr(runtime_module, "load_hand_eye_calibration", lambda _config: hand_eye)
    monkeypatch.setattr(
        runtime_module.Es68D435iCollisionResources,
        "packaged_template",
        lambda: object(),
    )
    monkeypatch.setattr(
        runtime_module.Cs68PinocchioCollisionChecker,
        "from_es68_resources",
        lambda *_args, **_kwargs: checker,
    )
    monkeypatch.setattr(
        runtime_module.LiveCollisionGeometry,
        "from_active_resources",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(runtime_module, "_load_motion_envelope", lambda *_args: (object(), "hash"))
    monkeypatch.setattr(
        runtime_module.Es68D435iRobotDepthRenderer,
        "from_active_resources",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runtime_module.Es68KinematicModel,
        "from_resources",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(runtime_module, "EliteArm", FakeArm)
    monkeypatch.setattr(runtime_module, "RealSenseD435i", FakeCamera)
    monkeypatch.setattr(runtime_module, "load_cs68_kinematics", lambda _path: object())
    monkeypatch.setattr(runtime_module, "EliteCs68IkChecker", lambda *_args: object())
    monkeypatch.setattr(runtime_module, "_bootstrap_foreground_config", lambda _settings: object())
    monkeypatch.setattr(runtime_module, "CoarseSciencePolicy", lambda **_kwargs: object())
    monkeypatch.setattr(runtime_module, "CoarseScienceSession", lambda **_kwargs: coarse_session)
    monkeypatch.setattr(runtime_module, "CoarseSessionNextViewAdapter", lambda _session: object())
    monkeypatch.setattr(runtime_module, "SynchronizedAcquirer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime_module,
        "FoundationStereoBackend",
        lambda _config: SimpleNamespace(prepare=lambda: None),
    )

    def capture_engine(**kwargs):
        captured.update(kwargs)
        raise ExpectedEngineConstruction

    monkeypatch.setattr(runtime_module, "FoundationStereoOccupancyCycleEngine", capture_engine)

    with (
        pytest.raises(ExpectedEngineConstruction),
        open_production_unknown_blade_runtime(
            settings,
            output_root=tmp_path / "experimental",
            operator_id="operator-1",
            experimental=True,
        ),
    ):
        raise AssertionError("engine construction should stop this focused test")

    assert captured["science_authority"] is None
    assert captured["science_authority_settings"] is None


def test_production_open_checks_readiness_before_hardware_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def refuse(_settings) -> None:
        calls.append("readiness")
        raise UnknownBladeRuntimeError("synthetic offline failure")

    class ForbiddenArm:
        def __init__(self, _config) -> None:  # pragma: no cover - safety assertion
            calls.append("hardware")

    monkeypatch.setattr(runtime_module, "require_unknown_blade_runtime_ready", refuse)
    monkeypatch.setattr(runtime_module, "EliteArm", ForbiddenArm)
    output = tmp_path / "must-not-exist"

    with (
        pytest.raises(UnknownBladeRuntimeError, match="offline failure"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=output,
            operator_id="operator-1",
        ),
    ):
        raise AssertionError("context must not open")

    assert calls == ["readiness"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("science_present", "timing_present"),
    ((False, False), (True, False), (False, True)),
)
def test_unaccepted_completed_resume_is_rejected_before_readiness_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    science_present: bool,
    timing_present: bool,
) -> None:
    root = tmp_path / "sealed"
    root.mkdir()
    plan = UnknownBladeResumePlan(
        experiment_root=root.resolve(),
        experiment_id="runtime-test",
        phase=UnknownBladeResumePhase.COMPLETE,
        coarse_run_root=(root / "runs" / "coarse").resolve(),
        coarse_generation_path=(root / "coarse-generation").resolve(),
        reference_coarse_model_path=(root / "reference").resolve(),
        fine_run_root=(root / "runs" / "fine").resolve(),
        accepted_fine_coverage_path=(root / "coverage").resolve(),
        final_reconstruction_path=(root / "reconstruction").resolve(),
        science_authority=(
            SimpleNamespace(identity="accepted-science") if science_present else None
        ),
        runtime_timing_authority=(
            SimpleNamespace(identity="accepted-timing") if timing_present else None
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_unknown_blade_resume_plan",
        lambda _root: plan,
    )
    monkeypatch.setattr(
        runtime_module,
        "require_unknown_blade_runtime_ready",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("completed resume must not run hardware readiness")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "EliteArm",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("completed resume must not construct hardware")
        ),
    )

    with (
        pytest.raises(UnknownBladeRuntimeError, match="requires science and runtime timing"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=root,
            operator_id="operator-1",
            resume=True,
        ),
    ):
        raise AssertionError("unaccepted COMPLETE must not open")


def test_science_bound_completed_resume_replays_without_constructing_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sealed-authority"
    root.mkdir()
    authority = SimpleNamespace(identity="accepted-science")
    timing_authority = SimpleNamespace(identity="accepted-timing")
    final_root = (root / "reconstruction").resolve()
    plan = UnknownBladeResumePlan(
        experiment_root=root.resolve(),
        experiment_id="runtime-test",
        phase=UnknownBladeResumePhase.COMPLETE,
        coarse_run_root=(root / "runs" / "coarse").resolve(),
        fine_run_root=(root / "runs" / "fine").resolve(),
        accepted_fine_coverage_path=(root / "coverage").resolve(),
        final_reconstruction_path=final_root,
        science_authority=authority,
        runtime_timing_authority=timing_authority,
    )
    terminal = SimpleNamespace(cycle_index=8, event_sha256="a" * 64)
    monkeypatch.setattr(runtime_module, "load_unknown_blade_resume_plan", lambda _root: plan)
    monkeypatch.setattr(
        runtime_module,
        "load_science_acceptance_authority",
        lambda _settings: authority,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_runtime_timing_acceptance_authority",
        lambda _settings: timing_authority,
    )
    monkeypatch.setattr(
        runtime_module,
        "replay_final_fine_reconstruction",
        lambda path, *, expected_science_authority: SimpleNamespace(
            root=Path(path).resolve(),
            science_authority=expected_science_authority,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "read_stop_scan_run",
        lambda _root: SimpleNamespace(
            run_id="runtime-test",
            root=plan.fine_run_root,
            events=(terminal,),
            latest_event=terminal,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "EliteArm",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("completed authority replay must not construct hardware")
        ),
    )

    with open_production_unknown_blade_runtime(
        load_settings("configs/default.yaml"),
        output_root=root,
        operator_id="operator-1",
        resume=True,
    ) as runtime:
        snapshot = runtime.snapshot

    assert snapshot.phase is UnknownBladeRuntimePhase.COMPLETE
    assert snapshot.final_reconstruction_path == final_root


def test_resume_missing_or_invalid_chain_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_module,
        "EliteArm",
        lambda _settings: calls.append("hardware"),
    )
    missing = tmp_path / "missing"

    with (
        pytest.raises(FileNotFoundError, match="resume root does not exist"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=missing,
            operator_id="operator-1",
            resume=True,
        ),
    ):
        raise AssertionError("context must not open")

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    monkeypatch.setattr(
        runtime_module,
        "load_unknown_blade_resume_plan",
        lambda _root: (_ for _ in ()).throw(ValueError("tampered handoff chain")),
    )
    with (
        pytest.raises(ValueError, match="tampered handoff chain"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=invalid,
            operator_id="operator-1",
            resume=True,
        ),
    ):
        raise AssertionError("context must not open")

    assert calls == []


def test_resume_rejects_a_different_physical_placement_before_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "placement-bound-run"
    root.mkdir()
    plan = UnknownBladeResumePlan(
        experiment_root=root.resolve(),
        experiment_id="runtime-test",
        placement_id="blade-placement-20260831-01",
        phase=UnknownBladeResumePhase.COARSE,
        coarse_run_root=(root / "runs" / "coarse").resolve(),
    )
    monkeypatch.setattr(runtime_module, "load_unknown_blade_resume_plan", lambda _root: plan)
    monkeypatch.setattr(
        runtime_module,
        "require_unknown_blade_runtime_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("placement mismatch must block before readiness")
        ),
    )

    with (
        pytest.raises(UnknownBladeRuntimeError, match="placement ID differs"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=root,
            operator_id="operator-1",
            run_id="runtime-test",
            placement_id="blade-placement-20260831-02",
            resume=True,
        ),
    ):
        raise AssertionError("mismatched placement must not open")


def test_existing_output_is_refused_before_hardware_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    output = tmp_path / "existing-run"
    output.mkdir()
    monkeypatch.setattr(
        runtime_module,
        "require_unknown_blade_runtime_ready",
        lambda _settings: (),
    )

    class ForbiddenArm:
        def __init__(self, _config) -> None:  # pragma: no cover - safety assertion
            calls.append("hardware")

    monkeypatch.setattr(runtime_module, "EliteArm", ForbiddenArm)

    with (
        pytest.raises(FileExistsError),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=output,
            operator_id="operator-1",
        ),
    ):
        raise AssertionError("context must not open")

    assert calls == []


@pytest.mark.parametrize(
    ("operator_id", "run_id"),
    (("bad operator", "valid-run"), ("operator-1", "../invalid-run")),
)
def test_invalid_identity_is_rejected_before_output_readiness_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator_id: str,
    run_id: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_module,
        "require_unknown_blade_runtime_ready",
        lambda _settings: calls.append("readiness"),
    )

    class ForbiddenArm:
        def __init__(self, _config) -> None:  # pragma: no cover - safety assertion
            calls.append("hardware")

    monkeypatch.setattr(runtime_module, "EliteArm", ForbiddenArm)
    output = tmp_path / "must-not-exist"

    with (
        pytest.raises(ValueError, match="run_id"),
        open_production_unknown_blade_runtime(
            load_settings("configs/default.yaml"),
            output_root=output,
            operator_id=operator_id,
            run_id=run_id,
        ),
    ):
        raise AssertionError("context must not open")

    assert calls == []
    assert not output.exists()


def test_unknown_runtime_refuses_configured_thermal_device_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "run_supervised_scan_readiness",
        lambda _settings, *, mode: (),
    )
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "proxy_model": settings.proxy_model.model_copy(update={"estimated_thickness_m": 0.003}),
            "blade_foreground": settings.blade_foreground.model_copy(update={"enabled": True}),
            "thermal": settings.thermal.model_copy(
                update={"enabled": True, "driver": "future-radiometric-adapter"}
            ),
        }
    )

    results = unknown_blade_runtime_readiness(settings)

    scope = next(item for item in results if item.name == "unknown_blade_coarse_to_fine")
    assert scope.level.value == "fail"
    assert "thermal must be disabled" in scope.details["missing"]
    assert scope.details["motion_authorized"] is False


def test_unknown_runtime_requires_bound_science_acceptance_and_schema5_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "run_supervised_scan_readiness",
        lambda _settings, *, mode: (),
    )
    settings = load_settings("configs/default.yaml")

    results = unknown_blade_runtime_readiness(settings)

    scope = next(item for item in results if item.name == "unknown_blade_coarse_to_fine")
    assert (
        "proxy_model.blade_envelope_min_m/max_m/minimum_envelope_retained_fraction"
        in scope.details["missing"]
    )
    assert "science_acceptance.path/id" in scope.details["missing"]
    assert (
        "stop_and_capture.maximum_schema5_handoff_duration_s"
        in scope.details["missing"]
    )
    assert scope.details["geometry_science_acceptance_eligible"] is False


def test_unknown_runtime_experimental_scope_skips_only_release_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "run_supervised_scan_readiness",
        lambda _settings, *, mode: (),
    )
    monkeypatch.setattr(
        runtime_module,
        "_load_motion_envelope",
        lambda _settings, _checker: (object(), "control-hash"),
    )
    monkeypatch.setattr(
        runtime_module.Cs68PinocchioCollisionChecker,
        "from_es68_resources",
        lambda *_args, **_kwargs: object(),
    )
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "proxy_model": settings.proxy_model.model_copy(
                update={
                    "estimated_thickness_m": 0.003,
                    "blade_envelope_min_m": (0.4, -0.2, 0.0),
                    "blade_envelope_max_m": (0.8, 0.2, 0.4),
                    "minimum_envelope_retained_fraction": 0.75,
                }
            ),
            "blade_foreground": settings.blade_foreground.model_copy(
                update={"enabled": True}
            ),
            "motion_preflight": settings.motion_preflight.model_copy(
                update={
                    "motion_envelope_acceptance_path": Path("/accepted/motion"),
                    "motion_envelope_acceptance_id": "a" * 64,
                }
            ),
        }
    )

    results = unknown_blade_runtime_readiness(
        settings,
        require_release_acceptance=False,
    )

    scope = next(item for item in results if item.name == "unknown_blade_coarse_to_fine")
    assert scope.level.value == "pass"
    assert scope.details["missing"] == []
    assert scope.details["release_scope"] == "experimental"
    assert scope.details["release_acceptance_bypassed"] is True
    assert scope.details["tracking_stop_envelope_motion_eligible"] is True


def test_unknown_runtime_science_gate_checks_contract_envelope_and_map_age_formula(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "run_supervised_scan_readiness",
        lambda _settings, *, mode: (),
    )
    calls: list[object] = []

    class Acceptance:
        def assert_matches(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        runtime_module,
        "read_science_acceptance",
        lambda path: calls.append(Path(path)) or Acceptance(),
    )
    monkeypatch.setattr(
        runtime_module,
        "science_runtime_contract_for_settings",
        lambda _settings: "c" * 64,
    )
    timing_calls: list[object] = []

    class TimingAcceptance:
        def assert_matches(self, **kwargs) -> None:
            timing_calls.append(kwargs)

    monkeypatch.setattr(
        runtime_module,
        "read_runtime_timing_acceptance",
        lambda path: timing_calls.append(Path(path)) or TimingAcceptance(),
    )
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "proxy_model": settings.proxy_model.model_copy(
                update={"estimated_thickness_m": 0.003}
            ),
            "blade_foreground": settings.blade_foreground.model_copy(
                update={"enabled": True}
            ),
            "science_acceptance": settings.science_acceptance.model_copy(
                update={
                    "path": tmp_path / "science",
                    "acceptance_id": "a" * 64,
                }
            ),
            "stop_and_capture": settings.stop_and_capture.model_copy(
                update={
                    "maximum_perception_cycle_duration_s": 2.0,
                    "maximum_operator_reposition_interval_s": 4.0,
                    "maximum_segment_execution_duration_s": 6.0,
                    "maximum_schema5_handoff_duration_s": 5.0,
                    "runtime_timing_acceptance_path": tmp_path / "timing",
                    "runtime_timing_acceptance_id": "d" * 64,
                }
            ),
            "occupancy": settings.occupancy.model_copy(
                update={"maximum_map_age_s": 27.0}
            ),
        }
    )

    results = unknown_blade_runtime_readiness(settings)

    scope = next(item for item in results if item.name == "unknown_blade_coarse_to_fine")
    assert scope.details["geometry_science_acceptance_eligible"] is True
    assert scope.details["runtime_timing_acceptance_eligible"] is True
    assert scope.details["required_map_age_exclusive_lower_bound_s"] == pytest.approx(7.0)
    assert Path(tmp_path / "science") in calls
    matched = next(item for item in calls if isinstance(item, dict))
    assert matched["acceptance_id"] == "a" * 64
    assert matched["runtime_contract_sha256"] == "c" * 64
    required = matched["required_test_envelope"]
    assert required.minimum_distance_m == settings.point_cloud.minimum_depth_m
    assert required.maximum_distance_m == settings.point_cloud.maximum_depth_m
    timing_matched = next(item for item in timing_calls if isinstance(item, dict))
    assert timing_matched["acceptance_id"] == "d" * 64
