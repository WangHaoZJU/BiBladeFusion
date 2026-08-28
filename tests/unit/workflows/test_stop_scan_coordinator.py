from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.stop_scan_coordinator as stop_scan_module
from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    MotionPreflightConfig,
    OccupancyConfig,
    RobotConfig,
    StopAndCaptureConfig,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.streaming import StreamServoJResult
from biblade_fusion.mapping import OccupancyMapState, OccupancySnapshot
from biblade_fusion.robotics.occupancy_collision import (
    _issue_occupancy_semantic_attestation,
)
from biblade_fusion.robotics.stationarity import validate_stationary_trace
from biblade_fusion.storage.inference_stationarity import (
    write_inference_stationarity,
)
from biblade_fusion.storage.occupancy_mapping import StoredOccupancyMapping
from biblade_fusion.storage.stop_scan_run import StopScanRunWriter, read_stop_scan_run
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    NextViewTarget,
    OccupancyGeneration,
    OccupancyGenerationPublisher,
    PerceptionCycleResult,
    StopScanAbortRequested,
    StopScanBlocked,
    StopScanCoordinator,
    StopScanError,
    StopScanPhase,
    _PreparedSegmentExecution,
)

NOW = datetime(2026, 8, 28, 1, 0, tzinfo=UTC)
_FAKE_OCCUPANCY_ASSETS: dict[Path, StoredOccupancyMapping] = {}


@pytest.fixture(autouse=True)
def _semantic_occupancy_readback_double(monkeypatch):
    """Keep coordinator tests focused while preserving mandatory disk readback."""

    _FAKE_OCCUPANCY_ASSETS.clear()

    def readback(path):
        resolved = Path(path).resolve()
        if resolved not in _FAKE_OCCUPANCY_ASSETS:
            raise ValueError(f"No synthetic semantic asset for {resolved}")
        return _FAKE_OCCUPANCY_ASSETS[resolved]

    monkeypatch.setattr(stop_scan_module, "read_occupancy_mapping", readback)
    yield
    _FAKE_OCCUPANCY_ASSETS.clear()


def _state(joints: np.ndarray, timestamp: int) -> RobotState:
    return RobotState(
        monotonic_time_ns=timestamp,
        controller_time_s=timestamp / 1e9,
        joint_positions_rad=joints,
        base_t_tcp=PoseSE3.from_rotation_translation(
            "base", "tcp", np.eye(3), (float(joints[0]), 0.0, 0.0)
        ),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.1,
    )


class FakeStateSource:
    def __init__(self, robot_config: RobotConfig) -> None:
        self.joints = np.zeros(6)
        self.timestamp = 1_000_000_000
        self.calls = 0
        self._robot_config = robot_config.model_copy(deep=True)

    @property
    def robot_config(self):
        return self._robot_config.model_copy(deep=True)

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read_state(self) -> RobotState:
        self.timestamp += 1_000_000
        return _state(self.joints.copy(), self.timestamp)

    def stop(self) -> None:
        self.calls += 1


def _bundle(view_id: str, sequence: int, state: RobotState) -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(4, 3, 100.0, 100.0, 2.0, 1.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), (-0.05, 0.0, 0.0)
        ),
        None,
    )
    image = np.zeros((3, 4), dtype=np.uint8)
    stereo = StereoFrame(
        state.monotonic_time_ns,
        sequence + 1,
        10.0,
        10.0,
        image,
        image,
        None,
        calibration,
    )
    return SynchronizedFrameBundle(
        view_id,
        sequence,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _stored_mapping(
    source_count: int,
    *,
    sequence_offset: int = 0,
    occupancy_metadata_sha256: str = "e" * 64,
) -> StoredOccupancyMapping:
    voxel = (0, 0, 0)
    state = (
        OccupancyMapState.MAP_READY
        if source_count >= 3
        else OccupancyMapState.MAPPING
    )
    snapshot = OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.1,
        origin_m=(-0.1, -0.1, -0.1),
        grid_shape=(2, 2, 2),
        free_indices=(frozenset({voxel}) if source_count >= 3 else frozenset()),
        free_observation_counts=((voxel, source_count),),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset(),
        sequence=source_count * 2 + sequence_offset,
        created_at_utc=NOW,
        source_view_ids=tuple(f"bootstrap-{index}" for index in range(source_count)),
        source_camera_centres_base_m=tuple(
            (index * 0.03, 0.0, 0.0) for index in range(source_count)
        ),
        source_camera_axes_base=((0.0, 0.0, 1.0),) * source_count,
        rebuild_started_at_utc=NOW,
        map_state=state,
        mapping_context_hash="d" * 64,
        parent_evidence_hash=("b" * 64 if source_count > 1 else None),
        quality_evidence_hash="c" * 64,
        state_reason="synthetic stop-scan mapping",
    )
    attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256=occupancy_metadata_sha256,
        snapshot=snapshot,
        robot_geometry_hash="2" * 64,
    )
    return StoredOccupancyMapping(
        snapshot=snapshot,
        mapping_context=None,  # coordinator consumes only the fully verified boundary
        frame_evidence=(),
        mapping_snapshots=(),
        result_snapshots=(),
        metadata={},
        semantic_attestation=attestation,
    )


class FakePerception:
    def __init__(
        self,
        root: Path,
        source: FakeStateSource,
        counts: list[int],
        *,
        acquisition_config: AcquisitionConfig,
        occupancy_config: OccupancyConfig,
        coordinator_config: StopAndCaptureConfig,
    ) -> None:
        self.root = root
        self.source = source
        self.counts = iter(counts)
        self.captures: list[str] = []
        self._acquisition_config = acquisition_config
        self._occupancy_config = occupancy_config
        self._coordinator_config = coordinator_config
        self.pending: tuple[CapturedStopScanView, PerceptionCycleResult] | None = None
        self.committed: list[tuple[str, int]] = []

    @property
    def robot_state_source(self):
        return self.source

    @property
    def acquisition_config(self):
        return self._acquisition_config

    @property
    def occupancy_config(self):
        return self._occupancy_config

    @property
    def coordinator_config(self):
        return self._coordinator_config

    def cancel_pending_capture(self, captured=None) -> None:
        if self.pending is not None and (
            captured is None or self.pending[0] is captured
        ):
            self.pending = None

    def commit_perception_cycle(self, captured, result) -> None:
        if (
            self.pending is None
            or self.pending[0] is not captured
            or self.pending[1].occupancy_mapping_path
            != result.occupancy_mapping_path
            or self.pending[1].inference_stationarity_sha256
            != result.inference_stationarity_sha256
        ):
            raise RuntimeError("synthetic perception commit mismatch")
        self.committed.append(
            (captured.bundle.view_id, captured.bundle.sequence_index)
        )
        self.pending = None

    def capture(self, view_id: str, sequence_index: int) -> CapturedStopScanView:
        self.captures.append(view_id)
        cycle = self.root / f"cycle-{sequence_index}-{view_id}"
        session = cycle / "raw-session"
        session.mkdir(parents=True)
        state = self.source.read_state()
        bundle = _bundle(view_id, sequence_index, state)
        (session / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "closed_at_utc": NOW.isoformat(),
                    "views": [
                        {"view_id": view_id, "sequence_index": sequence_index}
                    ],
                }
            ),
            encoding="utf-8",
        )
        return CapturedStopScanView(bundle, session, cycle, NOW)

    def infer_and_update(self, captured: CapturedStopScanView) -> PerceptionCycleResult:
        count = next(self.counts)
        occupancy_path = captured.cycle_root / "occupancy"
        occupancy_path.mkdir()
        metadata_path = occupancy_path / "metadata.json"
        metadata_path.write_text(
            json.dumps({"view_id": captured.bundle.view_id}),
            encoding="utf-8",
        )
        metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        stereo_path = captured.cycle_root / "stereo"
        stereo_path.mkdir()
        stereo_metadata_path = stereo_path / "metadata.json"
        stereo_metadata_path.write_text(
            json.dumps({"view_id": captured.bundle.view_id}),
            encoding="utf-8",
        )
        stereo_metadata_sha256 = hashlib.sha256(
            stereo_metadata_path.read_bytes()
        ).hexdigest()
        reference = captured.bundle.robot_state_before
        trace = (self.source.read_state(), self.source.read_state())
        stationarity = validate_stationary_trace(
            reference,
            trace,
            max_joint_delta_rad=self._acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                self._acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                self._acquisition_config.max_tcp_rotation_delta_rad
            ),
            maximum_robot_state_staleness_s=(
                self._coordinator_config.maximum_robot_state_staleness_s
            ),
        )
        stored_stationarity = write_inference_stationarity(
            captured.cycle_root / "inference_stationarity.json",
            view_id=captured.bundle.view_id,
            sequence_index=captured.bundle.sequence_index,
            reference=reference,
            trace=trace,
            evidence=stationarity,
            source_session_manifest=captured.raw_session_path / "manifest.json",
            max_joint_delta_rad=self._acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                self._acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                self._acquisition_config.max_tcp_rotation_delta_rad
            ),
            maximum_robot_state_staleness_s=(
                self._coordinator_config.maximum_robot_state_staleness_s
            ),
        )
        stored_mapping = _stored_mapping(
            count,
            occupancy_metadata_sha256=metadata_sha256,
        )
        evidence = SimpleNamespace(
            source_view_id=captured.bundle.view_id,
            source_sequence_index=captured.bundle.sequence_index,
            frame_number=captured.bundle.stereo.frame_number,
            source_session_manifest_sha256=hashlib.sha256(
                (captured.raw_session_path / "manifest.json").read_bytes()
            ).hexdigest(),
            source_stereo_metadata_sha256=stereo_metadata_sha256,
        )
        stored_mapping = replace(
            stored_mapping,
            frame_evidence=(evidence,),
            metadata={
                "frames": [
                    {
                        "sources": {
                            "session": {
                                "root": str(captured.raw_session_path),
                            },
                            "stereo_inference": {
                                "root": str(stereo_path),
                            },
                        }
                    }
                ]
            },
        )
        _FAKE_OCCUPANCY_ASSETS[occupancy_path.resolve()] = stored_mapping
        result = PerceptionCycleResult(
            bundle=captured.bundle,
            raw_session_path=captured.raw_session_path,
            stereo_inference_path=stereo_path,
            occupancy_mapping_path=occupancy_path,
            stored_occupancy=stored_mapping,
            stationarity_reference=reference,
            inference_robot_state_trace=trace,
            inference_stationarity=stationarity,
            inference_stationarity_path=stored_stationarity.path,
            inference_stationarity_sha256=stored_stationarity.file_sha256,
        )
        self.pending = (captured, result)
        return result


class FakeSelector:
    def __init__(self, target: NextViewTarget | None) -> None:
        self.target = target

    def select_next(self, observation, generation):
        del observation, generation
        return self.target


class FakeExecutor:
    def __init__(self, source: FakeStateSource, goal: tuple[float, ...]) -> None:
        self.source = source
        self.goal = np.asarray(goal)
        self.execute_calls = 0

    def approval_prompt(self, preflight) -> str:
        del preflight
        return "EXECUTE synthetic"

    def authorize(self, preflight, *, operator_id: str, confirmation: str):
        del preflight
        if operator_id != "operator" or confirmation != "EXECUTE synthetic":
            raise RuntimeError("approval mismatch")
        return object()

    def execute(
        self,
        preflight,
        permit,
        *,
        cancellation_requested=lambda: False,
    ) -> StreamServoJResult:
        del preflight, permit
        if cancellation_requested():
            raise StopScanAbortRequested("cancelled before fake execution")
        self.execute_calls += 1
        self.source.joints = self.goal.copy()
        self.source.stop()
        return StreamServoJResult(ok=True, commands_sent=5)


class FakeSafetyFactory:
    def __init__(
        self,
        source: FakeStateSource,
        publisher: OccupancyGenerationPublisher,
        motion_config: MotionPreflightConfig,
        occupancy_config: OccupancyConfig,
        coordinator_config: StopAndCaptureConfig,
        *,
        clear: bool = True,
    ) -> None:
        self.source = source
        self._publisher = publisher
        self._motion_config = motion_config
        self._occupancy_config = occupancy_config
        self._coordinator_config = coordinator_config
        self.clear = clear
        self.executor: FakeExecutor | None = None
        self.authoritative_preflight = None

    @property
    def motion_robot(self):
        return self.source

    @property
    def occupancy_publisher(self):
        return self._publisher

    @property
    def occupancy_config(self):
        return self._occupancy_config

    @property
    def motion_config(self):
        return self._motion_config

    @property
    def coordinator_config(self):
        return self._coordinator_config

    def prepare(self, proposal, generation) -> _PreparedSegmentExecution:
        evidence = SimpleNamespace(binding=generation.binding.tuple)
        occupancy = SimpleNamespace(evidence=evidence)
        preflight = SimpleNamespace(
            start_joint_positions_rad=proposal.start_joint_positions_rad,
            goal_joint_positions_rad=proposal.goal_joint_positions_rad,
            occupancy=occupancy,
            ready_for_approval=self.clear,
            blocking_reasons=(
                () if self.clear else ("continuous_swept_mesh_unavailable",)
            ),
            diagnostics={
                "stop_scan_occupancy_generation_id": generation.generation_id,
                "inference_stationarity_sha256": (
                    generation.inference_stationarity_sha256
                ),
            },
        )
        if self.clear:
            self.executor = FakeExecutor(self.source, proposal.goal_joint_positions_rad)
        self.authoritative_preflight = preflight
        return _PreparedSegmentExecution(proposal, preflight, self.executor)


def _coordinator(
    tmp_path: Path,
    *,
    mapping_counts: list[int],
    target: NextViewTarget | None,
    clear: bool = True,
    event_sink=None,
    safety_motion_config: MotionPreflightConfig | None = None,
    robot_model: str = "es68",
    robot_servoj_time_s: float = 0.004,
    motion_servoj_dt_s: float = 0.004,
    robot_motion_enabled: bool = True,
):
    robot_config = RobotConfig(
        model=robot_model,
        sdk_wheel=Path("/tmp/elite.whl"),
        settle_time_s=0.002,
        servoj_time_s=robot_servoj_time_s,
        motion_enabled=robot_motion_enabled,
    )
    motion_config = MotionPreflightConfig(servoj_dt_s=motion_servoj_dt_s)
    source = FakeStateSource(robot_config)
    publisher = OccupancyGenerationPublisher()
    coordinator_config = StopAndCaptureConfig(
        enabled=True,
        maximum_segment_joint_delta_rad=0.05,
        settle_timeout_s=1.0,
        settle_poll_period_s=0.001,
    )
    acquisition_config = AcquisitionConfig()
    occupancy_config = OccupancyConfig(
        enabled=True,
        workspace_bounds_min_m=(-0.2, -0.2, -0.2),
        workspace_bounds_max_m=(0.2, 0.2, 0.2),
        maximum_map_age_s=30.0,
    )
    perception = FakePerception(
        tmp_path,
        source,
        mapping_counts,
        acquisition_config=acquisition_config,
        occupancy_config=occupancy_config,
        coordinator_config=coordinator_config,
    )
    safety = FakeSafetyFactory(
        source,
        publisher,
        safety_motion_config or motion_config,
        occupancy_config,
        coordinator_config,
        clear=clear,
    )
    coordinator = StopScanCoordinator(
        config=coordinator_config,
        acquisition_config=acquisition_config,
        robot_config=robot_config,
        motion_config=motion_config,
        occupancy_config=occupancy_config,
        robot=source,
        perception=perception,
        selector=FakeSelector(target),
        safety_factory=safety,
        publisher=publisher,
        event_sink=event_sink,
        utc_clock=lambda: NOW,
    )
    return coordinator, source, source, perception, safety, publisher


def _target() -> NextViewTarget:
    return NextViewTarget(
        "fine-front-001",
        (0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )


def test_bootstrap_then_one_short_segment_requires_new_capture(tmp_path: Path) -> None:
    coordinator, _, stop, perception, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[1, 2, 3, 3],
        target=_target(),
    )
    assert coordinator.start().phase is StopScanPhase.BOOTSTRAP_MAP_REQUIRED
    coordinator.capture_infer_update("bootstrap-0")
    assert coordinator.checkpoint.phase is StopScanPhase.BOOTSTRAP_MAP_REQUIRED
    coordinator.capture_infer_update("bootstrap-1")
    assert coordinator.checkpoint.phase is StopScanPhase.BOOTSTRAP_MAP_REQUIRED
    coordinator.capture_infer_update("bootstrap-2")
    assert coordinator.checkpoint.phase is StopScanPhase.MAP_READY
    assert len(perception.committed) == 3

    prepared = coordinator.prepare_next_segment()

    assert prepared is not None and prepared.ready_for_approval
    assert not hasattr(prepared, "executor")
    assert prepared.motion_authorized is False
    assert prepared.preflight is not safety.authoritative_preflight
    prepared.preflight.diagnostics["caller_annotation"] = "detached"
    assert "caller_annotation" not in safety.authoritative_preflight.diagnostics
    assert prepared.proposal.final_target is False
    assert prepared.proposal.goal_joint_positions_rad[0] == pytest.approx(0.05)
    assert coordinator.approval_prompt() == "EXECUTE synthetic"
    coordinator.execute_approved(
        operator_id="operator",
        confirmation="EXECUTE synthetic",
    )
    assert coordinator.checkpoint.phase is StopScanPhase.AWAITING_CAPTURE
    assert safety.executor is not None and safety.executor.execute_calls == 1
    assert stop.calls == 4

    expected_capture = coordinator.checkpoint.expected_capture_view_id
    assert expected_capture == "transit_fine-front-001_cycle_0003"
    coordinator.capture_infer_update()
    assert perception.captures[-1] == expected_capture
    assert coordinator.checkpoint.phase is StopScanPhase.MAP_READY
    assert len(perception.committed) == 4


def test_missing_continuous_proof_blocks_without_executor(tmp_path: Path) -> None:
    coordinator, _, stop, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        clear=False,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    prepared = coordinator.prepare_next_segment()

    assert prepared is not None and prepared.ready_for_approval is False
    assert coordinator.checkpoint.phase is StopScanPhase.MOTION_BLOCKED
    assert "continuous_swept_mesh_unavailable" in coordinator.checkpoint.blocking_reasons
    assert safety.executor is None
    assert stop.calls == 1


def test_map_change_after_preflight_aborts_before_execute(tmp_path: Path) -> None:
    coordinator, _, stop, _, safety, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    changed_path = tmp_path / "changed"
    changed_path.mkdir()
    changed_metadata = changed_path / "metadata.json"
    changed_metadata.write_text("{}", encoding="utf-8")
    changed = _stored_mapping(
        3,
        sequence_offset=10,
        occupancy_metadata_sha256=hashlib.sha256(
            changed_metadata.read_bytes()
        ).hexdigest(),
    )
    current = publisher.current
    publisher.publish(
        OccupancyGeneration.verified(
            changed_path,
            changed,
            inference_stationarity_path=current.inference_stationarity_path,
            inference_stationarity_sha256=current.inference_stationarity_sha256,
        )
    )

    with pytest.raises(StopScanBlocked, match="changed before freeze"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED
    assert safety.executor is not None and safety.executor.execute_calls == 0
    assert stop.calls == 2


def test_multi_view_raw_session_is_rejected_before_inference(tmp_path: Path) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original_capture = perception.capture

    def invalid_capture(view_id: str, sequence_index: int):
        captured = original_capture(view_id, sequence_index)
        manifest_path = captured.raw_session_path / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["views"].append(dict(payload["views"][0]))
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return captured

    perception.capture = invalid_capture  # type: ignore[method-assign]
    coordinator.start()

    with pytest.raises(StopScanBlocked, match="one immutable raw session"):
        coordinator.capture_infer_update("bootstrap")

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED


def test_coordinator_writes_a_verifiable_hash_chained_event_run(tmp_path: Path) -> None:
    writer = StopScanRunWriter.create(tmp_path / "run", run_id="stop-scan-test")
    coordinator, *_ = _coordinator(
        tmp_path / "cycles",
        mapping_counts=[3],
        target=None,
        event_sink=writer,
    )

    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    stored = read_stop_scan_run(tmp_path / "run")

    assert stored.events[0].event_type == "run_started"
    assert stored.events[-1].phase == StopScanPhase.COMPLETE.value
    assert all(event.event_sha256 for event in stored.events)


def test_next_view_target_rejects_non_rigid_tcp_matrix() -> None:
    reflected = np.eye(4)
    reflected[0, 0] = -1.0

    with pytest.raises(ValueError, match="determinant must be \\+1"):
        NextViewTarget(
            "invalid-reflection",
            (0.0,) * 6,
            tuple(tuple(float(value) for value in row) for row in reflected),
        )


def test_already_reached_target_still_requires_full_preflight(tmp_path: Path) -> None:
    target = NextViewTarget(
        "current-pose",
        (0.0,) * 6,
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )
    coordinator, *_ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=target,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    prepared = coordinator.prepare_next_segment()

    assert prepared is not None
    assert prepared.proposal.final_target is True
    assert prepared.proposal.start_joint_positions_rad == (
        prepared.proposal.goal_joint_positions_rad
    )
    assert coordinator.checkpoint.phase is StopScanPhase.WAITING_APPROVAL


def test_request_stop_interrupts_execution_without_transaction_lock(
    tmp_path: Path,
) -> None:
    coordinator, _, stop, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    assert safety.executor is not None
    entered = threading.Event()
    release = threading.Event()

    def blocking_execute(preflight, permit, *, cancellation_requested):
        del preflight, permit
        entered.set()
        assert release.wait(timeout=2.0)
        if cancellation_requested():
            raise StopScanAbortRequested("operator emergency stop")
        return StreamServoJResult(ok=True, commands_sent=1)

    safety.executor.execute = blocking_execute  # type: ignore[method-assign]
    failures: list[BaseException] = []

    def run_execution() -> None:
        try:
            coordinator.execute_approved(
                operator_id="operator",
                confirmation="EXECUTE synthetic",
            )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_execution)
    worker.start()
    assert entered.wait(timeout=2.0)

    checkpoint = coordinator.request_stop("operator emergency stop")

    assert checkpoint.stop_requested is True
    assert stop.calls >= 1
    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert failures and isinstance(failures[0], StopScanAbortRequested)
    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED


def test_stop_latched_at_success_tail_cannot_return_motion_success(
    tmp_path: Path,
) -> None:
    class StopAtSegmentComplete:
        coordinator: StopScanCoordinator | None = None

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index, payload
            if event_type == "single_segment_complete":
                assert self.coordinator is not None
                self.coordinator.request_stop("tail-race stop")

    sink = StopAtSegmentComplete()
    coordinator, *_ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=sink,
    )
    sink.coordinator = coordinator
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()

    with pytest.raises(StopScanAbortRequested, match="tail-race stop"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED


def test_semantically_valid_but_unrelated_stationarity_asset_is_rejected(
    tmp_path: Path,
) -> None:
    coordinator, source, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original = perception.infer_and_update

    def unrelated_result(captured):
        result = original(captured)
        reference = source.read_state()
        trace = (source.read_state(), source.read_state())
        evidence = validate_stationary_trace(
            reference,
            trace,
            max_joint_delta_rad=perception.acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                perception.acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                perception.acquisition_config.max_tcp_rotation_delta_rad
            ),
            maximum_robot_state_staleness_s=(
                perception.coordinator_config.maximum_robot_state_staleness_s
            ),
        )
        unrelated = write_inference_stationarity(
            captured.cycle_root / "unrelated_stationarity.json",
            view_id=captured.bundle.view_id,
            sequence_index=captured.bundle.sequence_index,
            reference=reference,
            trace=trace,
            evidence=evidence,
            source_session_manifest=captured.raw_session_path / "manifest.json",
            max_joint_delta_rad=perception.acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(
                perception.acquisition_config.max_tcp_translation_delta_m
            ),
            max_tcp_rotation_delta_rad=(
                perception.acquisition_config.max_tcp_rotation_delta_rad
            ),
            maximum_robot_state_staleness_s=(
                perception.coordinator_config.maximum_robot_state_staleness_s
            ),
        )
        return replace(
            result,
            stationarity_reference=reference,
            inference_robot_state_trace=trace,
            inference_stationarity=evidence,
            inference_stationarity_path=unrelated.path,
            inference_stationarity_sha256=unrelated.file_sha256,
        )

    perception.infer_and_update = unrelated_result  # type: ignore[method-assign]
    coordinator.start()

    with pytest.raises(StopScanBlocked, match="does not cover the capture"):
        coordinator.capture_infer_update("bootstrap-ready")

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED


def test_stop_during_inference_rolls_back_uncommitted_source(tmp_path: Path) -> None:
    coordinator, _, _, perception, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original = perception.infer_and_update

    def stop_before_acceptance(captured):
        result = original(captured)
        coordinator.request_stop("operator stop during inference")
        return result

    perception.infer_and_update = stop_before_acceptance  # type: ignore[method-assign]
    coordinator.start()

    with pytest.raises(StopScanAbortRequested, match="during inference"):
        coordinator.capture_infer_update("bootstrap-ready")

    assert perception.pending is None
    assert perception.committed == []
    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED
    with pytest.raises(StopScanBlocked, match="No verified occupancy"):
        _ = publisher.current


def test_independent_readback_rejects_current_frame_source_mismatch(
    tmp_path: Path,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original = perception.infer_and_update

    def mismatched_mapping(captured):
        result = original(captured)
        stored = _FAKE_OCCUPANCY_ASSETS[result.occupancy_mapping_path]
        wrong = replace(
            stored.frame_evidence[-1],
            source_view_id="old-view",
        ) if hasattr(stored.frame_evidence[-1], "__dataclass_fields__") else SimpleNamespace(
            **{
                **vars(stored.frame_evidence[-1]),
                "source_view_id": "old-view",
            }
        )
        _FAKE_OCCUPANCY_ASSETS[result.occupancy_mapping_path] = replace(
            stored,
            frame_evidence=(wrong,),
        )
        return result

    perception.infer_and_update = mismatched_mapping  # type: ignore[method-assign]
    coordinator.start()

    with pytest.raises(StopScanBlocked, match="not the final frame"):
        coordinator.capture_infer_update("bootstrap-ready")

    assert perception.pending is None
    assert perception.committed == []


def test_external_map_replacement_breaks_observation_generation_binding(
    tmp_path: Path,
) -> None:
    coordinator, _, _, _, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    current = publisher.current
    changed_path = tmp_path / "external-generation"
    changed_path.mkdir()
    metadata_path = changed_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    changed_mapping = _stored_mapping(
        3,
        sequence_offset=20,
        occupancy_metadata_sha256=hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
    )
    publisher.publish(
        OccupancyGeneration.verified(
            changed_path,
            changed_mapping,
            inference_stationarity_path=current.inference_stationarity_path,
            inference_stationarity_sha256=current.inference_stationarity_sha256,
        )
    )

    with pytest.raises(StopScanBlocked, match="no longer matches"):
        coordinator.prepare_next_segment()


def test_mismatched_motion_preflight_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Preflight and coordination"):
        _coordinator(
            tmp_path,
            mapping_counts=[3],
            target=None,
            safety_motion_config=MotionPreflightConfig(speed_scaling=1.0),
        )


def test_stop_scan_rejects_cs68_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="robot.model='es68'"):
        _coordinator(
            tmp_path,
            mapping_counts=[3],
            target=None,
            robot_model="cs68",
        )


def test_stop_scan_rejects_split_servoj_period_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ServoJ time.*ServoJ dt"):
        _coordinator(
            tmp_path,
            mapping_counts=[3],
            target=None,
            robot_servoj_time_s=0.008,
        )


def test_stop_scan_rejects_disabled_motion_driver_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="robot.motion_enabled=true"):
        _coordinator(
            tmp_path,
            mapping_counts=[3],
            target=None,
            robot_motion_enabled=False,
        )


def test_event_sink_failure_cannot_leave_approval_executable(tmp_path: Path) -> None:
    class FailWaitingApproval:
        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index, payload
            if event_type == "single_segment_waiting_approval":
                raise OSError("event store unavailable")

    coordinator, _, _, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=FailWaitingApproval(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    with pytest.raises(StopScanError, match="event persistence failed"):
        coordinator.prepare_next_segment()

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert safety.executor is not None and safety.executor.execute_calls == 0
    with pytest.raises(StopScanError, match="No approval-eligible"):
        coordinator.approval_prompt()
    with pytest.raises(StopScanError, match="No approval-eligible"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )


@pytest.mark.parametrize(
    "failed_event",
    ["capture_started", "single_segment_preflight_started"],
)
def test_event_sink_failure_is_an_irreversible_terminal_latch(
    tmp_path: Path,
    failed_event: str,
) -> None:
    class FailOnce:
        failed = False

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index, payload
            if event_type == failed_event and not self.failed:
                self.failed = True
                raise OSError("single synthetic audit-store failure")

    coordinator, _, _, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=FailOnce(),
    )
    coordinator.start()

    if failed_event == "capture_started":
        with pytest.raises(StopScanError, match="already failed"):
            coordinator.capture_infer_update("bootstrap-ready")
    else:
        coordinator.capture_infer_update("bootstrap-ready")
        with pytest.raises(StopScanError, match="already failed"):
            coordinator.prepare_next_segment()

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert safety.executor is None
    with pytest.raises(StopScanError, match="Cannot capture from phase failed"):
        coordinator.capture_infer_update("recovery-is-forbidden")
    with pytest.raises(StopScanError, match="No approval-eligible"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )


def test_blocked_commit_never_delays_operator_physical_stop(tmp_path: Path) -> None:
    coordinator, _, stop, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original_commit = perception.commit_perception_cycle
    commit_entered = threading.Event()
    release_commit = threading.Event()

    def blocking_commit(captured, result):
        commit_entered.set()
        assert release_commit.wait(timeout=2.0)
        original_commit(captured, result)

    perception.commit_perception_cycle = blocking_commit  # type: ignore[method-assign]
    coordinator.start()
    failures: list[BaseException] = []

    def run_capture() -> None:
        try:
            coordinator.capture_infer_update("bootstrap-ready")
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_capture)
    worker.start()
    assert commit_entered.wait(timeout=2.0)

    checkpoint = coordinator.request_stop("stop while asset commit is blocked")

    assert checkpoint.stop_requested is True
    assert stop.calls >= 2  # pre-capture stop plus the asynchronous operator stop
    assert worker.is_alive()
    release_commit.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert failures and isinstance(failures[0], StopScanAbortRequested)
    # The request arrived after the explicit commit linearization point: the
    # immutable asset is accepted, while the run itself is terminal ABORTED.
    assert perception.committed == [("bootstrap-ready", 0)]
    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED


def test_failed_source_commit_never_exposes_candidate_generation(
    tmp_path: Path,
) -> None:
    coordinator, _, _, perception, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    commit_entered = threading.Event()
    release_commit = threading.Event()

    def failing_commit(captured, result):
        del captured, result
        commit_entered.set()
        assert release_commit.wait(timeout=2.0)
        raise RuntimeError("synthetic source-window commit failure")

    perception.commit_perception_cycle = failing_commit  # type: ignore[method-assign]
    coordinator.start()
    capture_failures: list[BaseException] = []

    def run_capture() -> None:
        try:
            coordinator.capture_infer_update("bootstrap-ready")
        except BaseException as exc:
            capture_failures.append(exc)

    capture_worker = threading.Thread(target=run_capture)
    capture_worker.start()
    assert commit_entered.wait(timeout=2.0)

    read_started = threading.Event()
    read_finished = threading.Event()
    read_failures: list[BaseException] = []

    def read_current() -> None:
        read_started.set()
        try:
            _ = publisher.current
        except BaseException as exc:
            read_failures.append(exc)
        finally:
            read_finished.set()

    reader = threading.Thread(target=read_current)
    reader.start()
    assert read_started.wait(timeout=2.0)
    # The reader is behind the publisher transaction lock; no staged candidate is
    # visible while the source-window acceptance decision is unresolved.
    assert not read_finished.wait(timeout=0.05)

    release_commit.set()
    capture_worker.join(timeout=2.0)
    reader.join(timeout=2.0)

    assert not capture_worker.is_alive()
    assert not reader.is_alive()
    assert capture_failures and isinstance(capture_failures[0], RuntimeError)
    assert read_failures and isinstance(read_failures[0], StopScanBlocked)
    assert perception.pending is None
    assert perception.committed == []
    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
