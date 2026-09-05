from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.stop_scan_coordinator as stop_scan_module
import biblade_fusion.workflows.unknown_blade_runtime as runtime_module
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    MotionPreflightConfig,
    OccupancyConfig,
    RobotConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.mapping import OccupancyMapState
from biblade_fusion.perception.bootstrap_foreground import BootstrapSeed
from biblade_fusion.planning import BladeSide
from biblade_fusion.robotics.stationarity import (
    BootstrapSafeStateEvidence,
    StationarityEvidence,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturePurpose,
    NextViewSelection,
    NextViewTarget,
    OccupancyGenerationPublisher,
    StopScanCoordinator,
)
from biblade_fusion.workflows.supervised_experiment import (
    GuardedCoordinatorMotionExecutor,
    SupervisedExperimentRunner,
)
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarsePhase,
    CoarsePhaseTransition,
)
from biblade_fusion.workflows.unknown_blade_runtime import (
    CoarseSessionNextViewAdapter,
    UnknownBladeResumePhase,
    UnknownBladeRuntimePhase,
    UnknownBladeSupervisedRuntime,
    _BestEffortLiveSupervision,
    run_unknown_blade_operator_console,
)


def _load_test_support(name: str, relative_path: str):
    source = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load integration support from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stop_support = _load_test_support(
    "_bbf_stop_scan_test_support",
    "tests/unit/workflows/test_stop_scan_coordinator.py",
)
_runtime_support = _load_test_support(
    "_bbf_unknown_runtime_test_support",
    "tests/unit/workflows/test_unknown_blade_runtime.py",
)
FakePerception = _stop_support.FakePerception
FakeSafetyFactory = _stop_support.FakeSafetyFactory
FakeStateSource = _stop_support.FakeStateSource
_FAKE_OCCUPANCY_ASSETS = _stop_support._FAKE_OCCUPANCY_ASSETS
_FakeCoarseSession = _runtime_support._FakeCoarseSession
_FakeExperimentHandoff = _runtime_support._FakeExperimentHandoff


class _SciencePerception(FakePerception):
    """Lightweight engine double retaining the real coordinator transaction API."""

    def __init__(self, *args, science_phase: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.science_phase = science_phase

    def infer_and_update(self, captured):
        result = super().infer_and_update(captured)
        if self.science_phase == "coarse" and captured.purpose in {
            CapturePurpose.BOOTSTRAP,
            CapturePurpose.CANDIDATE,
        }:
            coarse = captured.cycle_root / "coarse_scan_view"
            coarse.mkdir()
            result = replace(result, coarse_scan_view_path=coarse)
        elif self.science_phase == "fine" and captured.purpose is CapturePurpose.BOOTSTRAP:
            coverage = captured.cycle_root / "surface_coverage"
            coverage.mkdir()
            result = replace(result, coverage_path=coverage)
        elif self.science_phase == "fine" and captured.purpose is CapturePurpose.CANDIDATE:
            foreground = captured.cycle_root / "blade_foreground"
            reconstructed = captured.cycle_root / "reconstructed_view"
            coverage = captured.cycle_root / "surface_coverage"
            foreground.mkdir()
            reconstructed.mkdir()
            coverage.mkdir()
            result = replace(
                result,
                blade_foreground_path=foreground,
                reconstructed_view_path=reconstructed,
                coverage_path=coverage,
            )
        self.pending = (captured, result)
        return result


class _SequenceSelector:
    def __init__(self, selections: list[NextViewSelection]) -> None:
        self._selections = list(selections)
        self.calls = 0

    def select_next(self, _observation, _generation) -> NextViewSelection:
        selection = self._selections[min(self.calls, len(self._selections) - 1)]
        self.calls += 1
        return selection


class _BrokenReadOnlyBridge:
    def __init__(self) -> None:
        self.calls = 0

    def observe_event(self, _event) -> None:
        self.calls += 1
        raise RuntimeError("read-only timeline unavailable")

    def __call__(self, _status) -> None:
        self.calls += 1
        raise RuntimeError("read-only timeline unavailable")

    def observe_perception(self, _result, **_kwargs) -> None:
        self.calls += 1
        raise RuntimeError("read-only timeline unavailable")

    def observe_prepared_segment(self, _prepared) -> None:
        self.calls += 1
        raise RuntimeError("read-only timeline unavailable")


def _target(view_id: str, joint_0: float) -> NextViewTarget:
    matrix = np.eye(4)
    matrix[0, 3] = joint_0
    return NextViewTarget(
        view_id,
        (joint_0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple(tuple(float(value) for value in row) for row in matrix),
    )


def _incomplete(view_id: str, joint_0: float, *, surface: str) -> NextViewSelection:
    return NextViewSelection(
        _target(view_id, joint_0),
        surface,
        "b" * 64,
        "c" * 64,
        4,
        1,
        False,
        ("science_rank=1",),
    )


def _configs(tmp_path: Path):
    robot = RobotConfig(
        model="es68",
        sdk_wheel=Path("/tmp/elite.whl"),
        settle_time_s=0.002,
        servoj_time_s=0.004,
        motion_enabled=True,
    )
    acquisition = AcquisitionConfig()
    motion = MotionPreflightConfig(servoj_dt_s=0.004)
    stop = StopAndCaptureConfig(
        enabled=True,
        maximum_segment_joint_delta_rad=0.05,
        settle_timeout_s=1.0,
        settle_poll_period_s=0.001,
        allow_single_view_bootstrap_motion=True,
    )
    occupancy = OccupancyConfig(
        enabled=True,
        workspace_bounds_min_m=(-0.2, -0.2, -0.2),
        workspace_bounds_max_m=(0.2, 0.2, 0.2),
        maximum_map_age_s=30.0,
        accepted_static_free_aabbs=(
            {
                "name": "offline_corridor",
                "minimum_m": (-0.2, -0.2, -0.2),
                "maximum_m": (0.2, 0.2, 0.2),
            },
        ),
        accepted_static_free_acceptance_id="a" * 64,
        accepted_static_free_acceptance_path=tmp_path / "accepted-static-free",
    )
    return robot, acquisition, motion, stop, occupancy


def _runner(
    tmp_path: Path,
    *,
    name: str,
    science_phase: str,
    mapping_counts: list[int],
    selector,
    observer: _BestEffortLiveSupervision,
    perception_callbacks=(),
    robot: FakeStateSource | None = None,
):
    robot_config, acquisition, motion, stop, occupancy = _configs(tmp_path)
    robot = robot or FakeStateSource(robot_config)
    publisher = OccupancyGenerationPublisher()
    perception = _SciencePerception(
        tmp_path / f"{name}-perception",
        robot,
        mapping_counts,
        acquisition_config=acquisition,
        occupancy_config=occupancy,
        coordinator_config=stop,
        science_phase=science_phase,
    )
    safety = FakeSafetyFactory(robot, publisher, motion, occupancy, stop)
    runner = SupervisedExperimentRunner.create(
        run_root=tmp_path / f"{name}-run",
        run_id=f"{name}-run",
        config=stop,
        acquisition_config=acquisition,
        robot_config=robot_config,
        motion_config=motion,
        occupancy_config=occupancy,
        robot=robot,
        perception=perception,
        selector=selector,
        safety_factory=safety,
        publisher=publisher,
        motion_executor=GuardedCoordinatorMotionExecutor(),
        status_callbacks=(observer,),
        event_callbacks=(observer.observe_event,),
        perception_callbacks=perception_callbacks,
        prepared_segment_callbacks=(observer.observe_prepared_segment,),
    )
    return runner, perception, publisher, safety, robot


def _bootstrap_evidence() -> BootstrapSafeStateEvidence:
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
    return BootstrapSafeStateEvidence(
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


@pytest.fixture(autouse=True)
def _semantic_readback_and_science_doubles(monkeypatch):
    _FAKE_OCCUPANCY_ASSETS.clear()
    monkeypatch.setattr(
        stop_scan_module,
        "read_occupancy_mapping",
        lambda path: _FAKE_OCCUPANCY_ASSETS[Path(path).resolve()],
    )
    monkeypatch.setattr(
        StopScanCoordinator,
        "_validate_science_assets",
        staticmethod(lambda _captured, _result: None),
    )
    monkeypatch.setattr(
        StopScanCoordinator,
        "_validate_coarse_science_asset",
        staticmethod(lambda _captured, _result: None),
    )
    yield
    _FAKE_OCCUPANCY_ASSETS.clear()


def test_single_roi_composed_runtime_reaches_fine_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _BrokenReadOnlyBridge()
    observer = _BestEffortLiveSupervision(bridge)  # type: ignore[arg-type]
    transitions = [
        CoarsePhaseTransition(CoarsePhase.COLLECTING, ("more views",), tmp_path / "generation"),
        CoarsePhaseTransition(CoarsePhase.COLLECTING, ("more views",), tmp_path / "generation"),
        CoarsePhaseTransition(
            CoarsePhase.READY_FOR_FINE,
            ("schema-5 ready",),
            tmp_path / "generation",
            tmp_path / "ready-generation",
            tmp_path / "schema5-reference",
        ),
    ]
    coarse_session = _FakeCoarseSession(
        tmp_path,
        selections=[
            _incomplete("coarse-auto-001", 0.01, surface="1" * 64),
            _incomplete("coarse-auto-002", 0.02, surface="2" * 64),
        ],
        transitions=transitions,
    )
    coarse_adapter = CoarseSessionNextViewAdapter(coarse_session)
    handoff = _FakeExperimentHandoff()
    coarse_runner, coarse_perception, coarse_publisher, coarse_safety, robot = _runner(
        tmp_path,
        name="coarse",
        science_phase="coarse",
        mapping_counts=[1, 2, 3],
        selector=coarse_adapter,
        observer=observer,
        perception_callbacks=(coarse_adapter.observe_perception,),
    )

    final_root = tmp_path / "final-reconstruction"
    final_root.mkdir()
    final_coverage = tmp_path / "final-coverage"
    final_coverage.mkdir()
    fine_selector = _SequenceSelector(
        [
            _incomplete("fine-auto-001", 0.03, surface="d" * 64),
            NextViewSelection(
                None,
                "d" * 64,
                "b" * 64,
                "c" * 64,
                4,
                0,
                True,
                ("coverage complete",),
                final_reconstruction_path=final_root,
                final_reconstruction_id="e" * 64,
                final_reconstruction_metadata_sha256="f" * 64,
            ),
        ]
    )
    fine_objects: dict[str, object] = {}

    def create_fine(_reference: Path):
        runner, perception, publisher, safety, _ = _runner(
            tmp_path,
            name="fine",
            science_phase="fine",
            mapping_counts=[3, 3],
            selector=fine_selector,
            observer=observer,
            robot=robot,
        )
        fine_objects.update(
            runner=runner,
            perception=perception,
            publisher=publisher,
            safety=safety,
        )
        return runner

    monkeypatch.setattr(
        runtime_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            previous_generation_path=(tmp_path / "generation").resolve(),
            coarse_model_path=(tmp_path / "schema5-reference").resolve(),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "replay_final_fine_reconstruction",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            artifact_id="e" * 64,
            metadata_sha256="f" * 64,
            result=SimpleNamespace(
                coverage=SimpleNamespace(
                    root=final_coverage.resolve(),
                    generation_id="d" * 64,
                )
            ),
        ),
    )
    runtime = UnknownBladeSupervisedRuntime(
        coarse_runner=coarse_runner,
        coarse_adapter=coarse_adapter,
        create_fine_runner=create_fine,
        operator_id="operator",
        minimum_operator_bootstrap_views=3,
        timeline_root=tmp_path / "timeline",
        establish_initial_stop=_bootstrap_evidence,
        assert_exact_map_ready=lambda result: (
            None
            if result.stored_occupancy.snapshot.map_state is OccupancyMapState.MAP_READY
            and coarse_publisher.current.artifact_path == result.occupancy_mapping_path
            else (_ for _ in ()).throw(AssertionError("coarse MAP_READY binding changed"))
        ),
        experiment_handoff=handoff,
        experimental=True,
    )
    console_prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        console_prompts.append(prompt)
        if prompt.startswith("Manually reposition"):
            return "c front"
        if prompt.startswith("Paste the exact token"):
            return runtime.approval_prompt()
        raise AssertionError(f"Unexpected operator prompt: {prompt}")

    with pytest.warns(RuntimeWarning, match="without interrupting"):
        exit_code = run_unknown_blade_operator_console(
            runtime,
            input_fn=input_fn,
            output_fn=lambda _line: None,
            initial_bootstrap_seed=BootstrapSeed.polygon(
                ((0, 0), (2, 0), (1, 2)),
                mode="hard_roi",
            ),
            initial_operator_side=BladeSide.FRONT,
        )
    completed = runtime.snapshot

    assert exit_code == 0
    assert completed.phase is UnknownBladeRuntimePhase.COMPLETE
    assert completed.final_reconstruction_path == final_root.resolve()
    assert coarse_session.stage_operator_calls == 1
    assert coarse_session.stage_selected_calls == 2
    assert [purpose for _, purpose in coarse_perception.capture_requests] == [
        CapturePurpose.BOOTSTRAP,
        CapturePurpose.CANDIDATE,
        CapturePurpose.CANDIDATE,
    ]
    fine_perception = fine_objects["perception"]
    assert [purpose for _, purpose in fine_perception.capture_requests] == [  # type: ignore[union-attr]
        CapturePurpose.BOOTSTRAP,
        CapturePurpose.CANDIDATE,
    ]
    assert coarse_safety.executor is not None
    assert coarse_safety.executor.execute_calls == 1
    assert fine_objects["safety"].executor.execute_calls == 1  # type: ignore[union-attr]
    assert bridge.calls == 1
    assert sum(prompt.startswith("Manually reposition") for prompt in console_prompts) == 1
    assert sum(prompt.startswith("Paste the exact token") for prompt in console_prompts) == 3


def test_resume_allocates_new_capture_identity_and_uses_safety_refresh(
    tmp_path: Path,
) -> None:
    bridge = _BrokenReadOnlyBridge()
    observer = _BestEffortLiveSupervision(bridge)  # type: ignore[arg-type]
    selector = _SequenceSelector([_incomplete("candidate", 0.01, surface="1" * 64)])
    runner, perception, _, _, _ = _runner(
        tmp_path,
        name="resume",
        science_phase="coarse",
        mapping_counts=[3],
        selector=selector,
        observer=observer,
    )
    with pytest.warns(RuntimeWarning, match="without interrupting"):
        runner.start()
    runner.step(view_id="operator_bootstrap_000000")
    previous_cycle = runner.status.cycle_index

    robot_config, acquisition, motion, stop, occupancy = _configs(tmp_path)
    robot = FakeStateSource(robot_config)
    publisher = OccupancyGenerationPublisher()
    recovered_session = _FakeCoarseSession(
        tmp_path,
        selections=[_incomplete("recovered-candidate", 0.01, surface="1" * 64)],
    )
    recovered_adapter = CoarseSessionNextViewAdapter(
        recovered_session,
        initial_accepted_cycle_count=1,
    )
    recovered_perception = _SciencePerception(
        tmp_path / "resume-perception",
        robot,
        [1],
        acquisition_config=acquisition,
        occupancy_config=occupancy,
        coordinator_config=stop,
        science_phase="coarse",
    )
    safety = FakeSafetyFactory(robot, publisher, motion, occupancy, stop)
    recovered = SupervisedExperimentRunner.resume(
        run_root=runner.status.run_root,
        config=stop,
        acquisition_config=acquisition,
        robot_config=robot_config,
        motion_config=motion,
        occupancy_config=occupancy,
        robot=robot,
        perception=recovered_perception,
        selector=recovered_adapter,
        safety_factory=safety,
        publisher=publisher,
        motion_executor=GuardedCoordinatorMotionExecutor(),
        perception_callbacks=(recovered_adapter.observe_perception,),
    )
    runtime = UnknownBladeSupervisedRuntime(
        coarse_runner=recovered,
        coarse_adapter=recovered_adapter,
        create_fine_runner=lambda _reference: (_ for _ in ()).throw(
            AssertionError("recovery must remain in coarse scan")
        ),
        operator_id="operator",
        minimum_operator_bootstrap_views=3,
        timeline_root=tmp_path / "resume-timeline",
        establish_initial_stop=_bootstrap_evidence,
        assert_exact_map_ready=lambda _result: None,
        experiment_handoff=_FakeExperimentHandoff(),
        resume_phase=UnknownBladeResumePhase.COARSE,
        initial_operator_bootstrap_views=1,
    )

    console_prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        console_prompts.append(prompt)
        if prompt.startswith("Keep the blade, fixture, camera mount"):
            return "c"
        if prompt.startswith("Paste the exact token"):
            return "q"
        raise AssertionError(f"Recovery unexpectedly requested another ROI: {prompt}")

    exit_code = run_unknown_blade_operator_console(
        runtime,
        input_fn=input_fn,
        output_fn=lambda _line: None,
    )

    assert exit_code == 0
    assert runtime.snapshot.phase is UnknownBladeRuntimePhase.STOPPED
    recovery_cycle = recovered_perception.committed[0][1]
    assert recovery_cycle > previous_cycle
    assert recovered_perception.capture_requests == [
        (
            f"coarse_safety_refresh_{recovery_cycle:06d}",
            CapturePurpose.SAFETY_REFRESH,
        )
    ]
    assert recovered_perception.committed[0] not in perception.committed
    assert recovered_session.stage_operator_calls == 0
    assert recovered_adapter.accepted_cycle_count == 1
    assert sum(prompt.startswith("Keep the blade") for prompt in console_prompts) == 1
    assert all("Manually reposition" not in prompt for prompt in console_prompts)
    assert all("hard_roi" not in prompt for prompt in console_prompts)
