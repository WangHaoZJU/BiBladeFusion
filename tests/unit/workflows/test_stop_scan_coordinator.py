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
from biblade_fusion.robotics import MotionExecutionPermit
from biblade_fusion.robotics.guarded_execution import EmergencyStopUnconfirmedError
from biblade_fusion.robotics.occupancy_collision import (
    _issue_occupancy_semantic_attestation,
)
from biblade_fusion.robotics.stationarity import validate_stationary_trace
from biblade_fusion.storage.inference_stationarity import (
    write_inference_stationarity,
)
from biblade_fusion.storage.motion_envelope_acceptance import (
    StoredMotionEnvelopeAcceptance,
)
from biblade_fusion.storage.occupancy_mapping import StoredOccupancyMapping
from biblade_fusion.storage.stop_scan_run import StopScanRunWriter, read_stop_scan_run
from biblade_fusion.workflows.stop_scan_coordinator import (
    BladePlanningAssetError,
    CapturedStopScanView,
    CapturePurpose,
    GuardedSegmentSafetyFactory,
    NextViewSelection,
    NextViewTarget,
    NextViewUnavailable,
    OccupancyGeneration,
    OccupancyGenerationPublisher,
    PerceptionCycleResult,
    RankedNextViewCandidate,
    StopScanAbortRequested,
    StopScanBlocked,
    StopScanCoordinator,
    StopScanError,
    StopScanPhase,
    _PreparedSegmentExecution,
)

NOW = datetime(2026, 8, 28, 1, 0, tzinfo=UTC)
_FAKE_OCCUPANCY_ASSETS: dict[Path, StoredOccupancyMapping] = {}


def _motion_envelope(path: Path) -> StoredMotionEnvelopeAcceptance:
    return StoredMotionEnvelopeAcceptance(
        path=path,
        acceptance_id="6" * 64,
        workcell_id="test",
        operator_id="operator",
        accepted_at_utc=NOW,
        robot_geometry_hash="2" * 64,
        motion_model_contract_hash="3" * 64,
        motion_control_contract_hash="4" * 64,
        maximum_tracking_deviation_rad=(0.001,) * 6,
        maximum_stop_drift_rad=(0.001,) * 6,
        safety_margin_factor=1.0,
        maximum_feedback_interval_s=0.01,
        maximum_stop_acknowledgement_s=0.1,
        maximum_stopped_actual_joint_velocity_rad_s=0.002,
        maximum_stopped_target_joint_velocity_rad_s=0.002,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.002,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.002,
        trial_count=3,
        metadata_sha256="7" * 64,
    )


def test_safety_factory_binds_verified_static_free_acceptance(monkeypatch, tmp_path) -> None:
    accepted_path = tmp_path / "accepted-static-free"
    calls: dict[str, object] = {}

    class Acceptance:
        def assert_matches(self, **kwargs) -> None:
            calls.update(kwargs)

    monkeypatch.setattr(
        stop_scan_module,
        "read_static_free_acceptance",
        lambda path: Acceptance() if Path(path) == accepted_path else None,
    )
    region = {
        "name": "robot_staging",
        "minimum_m": (-0.2, -0.2, 0.0),
        "maximum_m": (0.2, 0.2, 0.5),
    }
    occupancy = OccupancyConfig(
        workspace_bounds_min_m=(-0.5, -0.5, 0.0),
        workspace_bounds_max_m=(0.5, 0.5, 1.0),
        accepted_static_free_aabbs=(region,),
        accepted_static_free_acceptance_id="a" * 64,
        accepted_static_free_acceptance_path=accepted_path,
    )
    checker = SimpleNamespace(
        robot_geometry_hash="2" * 64,
        motion_model_contract_hash="3" * 64,
    )
    motion_path = tmp_path / "motion-envelope"

    factory = GuardedSegmentSafetyFactory(
        SimpleNamespace(),
        checker,
        SimpleNamespace(),
        MotionPreflightConfig(
            motion_envelope_acceptance_path=motion_path,
            motion_envelope_acceptance_id="6" * 64,
        ),
        occupancy,
        StopAndCaptureConfig(),
        _motion_envelope(motion_path),
        "4" * 64,
    )

    assert factory.occupancy_config == occupancy
    assert calls["acceptance_id"] == "a" * 64
    assert calls["robot_geometry_hash"] == "2" * 64
    assert calls["workspace_minimum_m"] == (-0.5, -0.5, 0.0)
    assert calls["workspace_maximum_m"] == (0.5, 0.5, 1.0)
    assert tuple(item.name for item in calls["regions"]) == ("robot_staging",)


def test_coordinator_independently_verifies_coarse_science_binding(
    monkeypatch,
    tmp_path,
) -> None:
    import biblade_fusion.storage.coarse_scan as coarse_storage

    cycle = (tmp_path / "cycle").resolve()
    coarse = cycle / "coarse_scan_view"
    stereo = cycle / "stereo_inference"
    occupancy = cycle / "occupancy_mapping"
    coarse.mkdir(parents=True)
    bundle = SimpleNamespace(view_id="front_r00_c00", sequence_index=4)
    bundle.stereo = SimpleNamespace(frame_number=19)
    stored = SimpleNamespace(
        reconstructed=SimpleNamespace(
            view=SimpleNamespace(
                source_view_id=bundle.view_id,
                source_sequence_index=bundle.sequence_index,
                source_frame_number=bundle.stereo.frame_number,
            )
        ),
        metadata={
            "sources": {
                "stereo_inference": {"root": str(stereo)},
                "occupancy_mapping": {"root": str(occupancy)},
            }
        },
    )
    monkeypatch.setattr(coarse_storage, "read_coarse_scan_view", lambda path: stored)
    captured = SimpleNamespace(bundle=bundle, cycle_root=cycle)
    result = SimpleNamespace(
        coarse_scan_view_path=coarse,
        stereo_inference_path=stereo,
        occupancy_mapping_path=occupancy,
    )

    StopScanCoordinator._validate_coarse_science_asset(captured, result)

    stored.reconstructed.view.source_frame_number += 1
    with pytest.raises(StopScanBlocked, match="Coarse-science asset"):
        StopScanCoordinator._validate_coarse_science_asset(captured, result)


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
        runtime_state="STOPPED",
        actual_joint_velocity_rad_s=np.zeros(6),
        target_joint_velocity_rad_s=np.zeros(6),
        actual_tcp_velocity=np.zeros(6),
        target_tcp_velocity=np.zeros(6),
    )


class FakeStateSource:
    def __init__(self, robot_config: RobotConfig) -> None:
        self.joints = np.zeros(6)
        self.timestamp = 1_000_000_000
        self.calls = 0
        self._stop_generation = 0
        self._stopped = False
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
        self._stop_generation += 1
        self._stopped = True

    @property
    def stop_snapshot(self) -> tuple[int, bool]:
        return self._stop_generation, self._stopped


def _bundle(view_id: str, sequence: int, state: RobotState) -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(4, 3, 100.0, 100.0, 2.0, 1.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), (-0.05, 0.0, 0.0)),
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
    state = OccupancyMapState.MAP_READY if source_count >= 3 else OccupancyMapState.MAPPING
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
        self.capture_requests: list[tuple[str, CapturePurpose]] = []
        self.capture_purpose_override: CapturePurpose | None = None
        self.result_purpose_override: CapturePurpose | None = None
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
        if self.pending is not None and (captured is None or self.pending[0] is captured):
            self.pending = None

    def commit_perception_cycle(
        self,
        captured,
        result,
        *,
        before_commit=lambda _stage: None,
    ) -> None:
        before_commit("before_fake_pending_validation")
        if (
            self.pending is None
            or self.pending[0] is not captured
            or self.pending[1].occupancy_mapping_path != result.occupancy_mapping_path
            or self.pending[1].inference_stationarity_sha256 != result.inference_stationarity_sha256
        ):
            raise RuntimeError("synthetic perception commit mismatch")
        before_commit("before_fake_source_advance")
        self.committed.append((captured.bundle.view_id, captured.bundle.sequence_index))
        self.pending = None

    def capture(
        self,
        view_id: str,
        sequence_index: int,
        *,
        purpose: CapturePurpose,
    ) -> CapturedStopScanView:
        self.captures.append(view_id)
        self.capture_requests.append((view_id, purpose))
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
                    "views": [{"view_id": view_id, "sequence_index": sequence_index}],
                }
            ),
            encoding="utf-8",
        )
        return CapturedStopScanView(
            bundle,
            session,
            cycle,
            NOW,
            self.capture_purpose_override or purpose,
        )

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
        stereo_metadata_sha256 = hashlib.sha256(stereo_metadata_path.read_bytes()).hexdigest()
        reference = captured.bundle.robot_state_before
        trace = (self.source.read_state(), self.source.read_state())
        stationarity = validate_stationary_trace(
            reference,
            trace,
            max_joint_delta_rad=self._acquisition_config.max_joint_delta_rad,
            max_tcp_translation_delta_m=(self._acquisition_config.max_tcp_translation_delta_m),
            max_tcp_rotation_delta_rad=(self._acquisition_config.max_tcp_rotation_delta_rad),
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
            max_tcp_translation_delta_m=(self._acquisition_config.max_tcp_translation_delta_m),
            max_tcp_rotation_delta_rad=(self._acquisition_config.max_tcp_rotation_delta_rad),
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
            purpose=self.result_purpose_override or captured.purpose,
        )
        self.pending = (captured, result)
        return result


class FakeSelector:
    def __init__(self, target: NextViewTarget | None) -> None:
        self.target = target

    def select_next(self, observation, generation):
        del observation, generation
        return NextViewSelection(
            target=self.target,
            surface_generation_id="a" * 64,
            reference_model_sha256="b" * 64,
            selection_policy_sha256="c" * 64,
            required_patch_count=4,
            incomplete_patch_count=0 if self.target is None else 1,
            coverage_complete=self.target is None,
            diagnostics=("synthetic selector evidence",),
        )


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
        return MotionExecutionPermit(
            permit_id="fake-permit",
            preflight_fingerprint="1" * 64,
            operator_id=operator_id,
            collision_model_id="fake-model",
            collision_model_hash="2" * 64,
            robot_geometry_hash="3" * 64,
            motion_model_contract_hash="4" * 64,
            servoj_runtime_config_hash="5" * 64,
            motion_envelope_acceptance_id="6" * 64,
            motion_envelope_metadata_sha256="7" * 64,
            accepted_joint_uncertainty_rad=(0.001,) * 6,
            occupancy_sequence=3,
            occupancy_content_hash="6" * 64,
            occupancy_mapping_context_hash="7" * 64,
            occupancy_quality_evidence_hash="8" * 64,
            occupancy_metadata_sha256="9" * 64,
            occupancy_semantic_verifier_contract_hash="a" * 64,
            occupancy_semantic_attestation_hash="b" * 64,
            occupancy_policy_contract_hash="c" * 64,
            continuous_occupancy_sweep_verified=True,
            stop_generation=4,
            stop_latched=True,
            issued_monotonic_s=1.0,
            expires_monotonic_s=2.0,
        )

    def execute(
        self,
        preflight,
        permit,
        *,
        cancellation_requested=lambda: False,
        maximum_duration_s=None,
    ) -> StreamServoJResult:
        del preflight, permit
        assert maximum_duration_s is None or maximum_duration_s > 0.0
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
        self._motion_envelope = _motion_envelope(Path("/tmp/synthetic-motion-envelope"))

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

    @property
    def motion_envelope_acceptance(self):
        return self._motion_envelope

    def prepare(self, proposal, generation) -> _PreparedSegmentExecution:
        evidence = SimpleNamespace(binding=generation.binding.tuple)
        occupancy = SimpleNamespace(evidence=evidence)
        preflight = SimpleNamespace(
            start_joint_positions_rad=proposal.start_joint_positions_rad,
            goal_joint_positions_rad=proposal.goal_joint_positions_rad,
            occupancy=occupancy,
            ready_for_approval=self.clear,
            blocking_reasons=(() if self.clear else ("continuous_swept_mesh_unavailable",)),
            diagnostics={
                "stop_scan_occupancy_generation_id": generation.generation_id,
                "inference_stationarity_sha256": (generation.inference_stationarity_sha256),
                "surface_generation_id": proposal.surface_generation_id,
                "reference_model_sha256": proposal.reference_model_sha256,
                "selection_policy_sha256": proposal.selection_policy_sha256,
                "bootstrap_mapping_prefix": proposal.bootstrap_mapping_prefix,
                "planned_servoj_duration_s": 0.1,
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
    coordinator_updates: dict[str, object] | None = None,
    occupancy_updates: dict[str, object] | None = None,
    monotonic_clock=None,
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
    coordinator_values: dict[str, object] = {
        "enabled": True,
        "maximum_segment_joint_delta_rad": 0.05,
        "settle_timeout_s": 1.0,
        "settle_poll_period_s": 0.001,
    }
    coordinator_values.update(coordinator_updates or {})
    coordinator_config = StopAndCaptureConfig(**coordinator_values)
    acquisition_config = AcquisitionConfig()
    occupancy_values: dict[str, object] = {
        "enabled": True,
        "workspace_bounds_min_m": (-0.2, -0.2, -0.2),
        "workspace_bounds_max_m": (0.2, 0.2, 0.2),
        "maximum_map_age_s": 30.0,
    }
    occupancy_values.update(occupancy_updates or {})
    occupancy_config = OccupancyConfig(**occupancy_values)
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
    coordinator_kwargs = {}
    if monotonic_clock is not None:
        coordinator_kwargs["monotonic_clock"] = monotonic_clock
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
        **coordinator_kwargs,
    )
    return coordinator, source, source, perception, safety, publisher


def _target() -> NextViewTarget:
    return NextViewTarget(
        "fine-front-001",
        (0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )


def test_bootstrap_then_one_complete_viewpoint_motion_requires_new_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert [purpose for _, purpose in perception.capture_requests] == [
        CapturePurpose.BOOTSTRAP,
        CapturePurpose.BOOTSTRAP,
        CapturePurpose.BOOTSTRAP,
    ]

    prepared = coordinator.prepare_next_segment()

    assert prepared is not None and prepared.ready_for_approval
    assert not hasattr(prepared, "executor")
    assert prepared.motion_authorized is False
    assert prepared.preflight is not safety.authoritative_preflight
    prepared.preflight.diagnostics["caller_annotation"] = "detached"
    assert "caller_annotation" not in safety.authoritative_preflight.diagnostics
    assert prepared.proposal.final_target is True
    assert prepared.proposal.goal_joint_positions_rad[0] == pytest.approx(0.20)
    assert coordinator.approval_prompt() == "EXECUTE synthetic"
    coordinator.execute_approved(
        operator_id="operator",
        confirmation="EXECUTE synthetic",
    )

    assert coordinator.checkpoint.phase is StopScanPhase.AWAITING_CAPTURE
    assert safety.executor is not None and safety.executor.execute_calls == 1
    assert stop.calls == 4

    expected_capture = coordinator.checkpoint.expected_capture_view_id
    assert expected_capture == "fine-front-001"
    monkeypatch.setattr(
        StopScanCoordinator,
        "_validate_coarse_science_asset",
        staticmethod(lambda captured, result: None),
    )
    original_infer = perception.infer_and_update

    def infer_with_coarse_science(captured):
        coarse_path = captured.cycle_root / "coarse_scan_view"
        coarse_path.mkdir()
        result = replace(original_infer(captured), coarse_scan_view_path=coarse_path)
        perception.pending = (captured, result)
        return result

    perception.infer_and_update = infer_with_coarse_science  # type: ignore[method-assign]
    coordinator.capture_infer_update()
    assert perception.captures[-1] == expected_capture
    assert perception.capture_requests[-1] == (
        expected_capture,
        CapturePurpose.CANDIDATE,
    )
    assert coordinator.checkpoint.phase is StopScanPhase.MAP_READY
    assert len(perception.committed) == 4


def test_one_formal_coarse_view_enables_restricted_bootstrap_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_path = tmp_path / "static-free-acceptance"
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[1],
        target=_target(),
        occupancy_updates={
            "accepted_static_free_aabbs": (
                {
                    "name": "bootstrap_corridor",
                    "minimum_m": (-0.2, -0.2, -0.2),
                    "maximum_m": (0.0, 0.2, 0.2),
                },
            ),
            "accepted_static_free_acceptance_id": "a" * 64,
            "accepted_static_free_acceptance_path": accepted_path,
        },
        coordinator_updates={"allow_single_view_bootstrap_motion": True},
    )
    original_infer = perception.infer_and_update

    def infer_with_coarse_proxy(captured):
        result = original_infer(captured)
        coarse_path = captured.cycle_root / "coarse_scan_view"
        coarse_path.mkdir()
        result = replace(result, coarse_scan_view_path=coarse_path)
        perception.pending = (captured, result)
        return result

    perception.infer_and_update = infer_with_coarse_proxy  # type: ignore[method-assign]
    monkeypatch.setattr(
        StopScanCoordinator,
        "_validate_coarse_science_asset",
        staticmethod(lambda captured, result: None),
    )
    coordinator.start()

    coordinator.capture_infer_update("single-initial-view")

    assert coordinator.checkpoint.phase is StopScanPhase.BOOTSTRAP_MOTION_READY
    assert coordinator.checkpoint.occupancy_binding is not None
    prepared = coordinator.prepare_next_segment()
    assert prepared is not None and prepared.ready_for_approval
    assert prepared.proposal.bootstrap_mapping_prefix is True
    assert prepared.preflight.diagnostics["bootstrap_mapping_prefix"] is True


def test_single_view_without_static_free_policy_remains_operator_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[1],
        target=_target(),
    )
    original_infer = perception.infer_and_update

    def infer_with_coarse_proxy(captured):
        result = original_infer(captured)
        coarse_path = captured.cycle_root / "coarse_scan_view"
        coarse_path.mkdir()
        result = replace(result, coarse_scan_view_path=coarse_path)
        perception.pending = (captured, result)
        return result

    perception.infer_and_update = infer_with_coarse_proxy  # type: ignore[method-assign]
    monkeypatch.setattr(
        StopScanCoordinator,
        "_validate_coarse_science_asset",
        staticmethod(lambda captured, result: None),
    )
    coordinator.start()

    coordinator.capture_infer_update("single-initial-view")

    assert coordinator.checkpoint.phase is StopScanPhase.BOOTSTRAP_MAP_REQUIRED
    with pytest.raises(StopScanError, match="Cannot plan"):
        coordinator.prepare_next_segment()


def test_legacy_segment_bound_does_not_split_one_viewpoint_motion(
    tmp_path: Path,
) -> None:
    target = NextViewTarget(
        "fine-front-just-over-bound",
        (0.051, 0.0, 0.0, 0.0, 0.0, 0.0),
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
    assert prepared.proposal.goal_joint_positions_rad[0] == pytest.approx(0.051)


def test_exact_approval_is_persisted_once_before_motion(tmp_path: Path) -> None:
    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index
            self.events.append((event_type, dict(payload)))

    sink = RecordingSink()
    coordinator, _, _, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=sink,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()

    coordinator.execute_approved(
        operator_id="operator",
        confirmation="EXECUTE synthetic",
    )

    approved = [payload for kind, payload in sink.events if kind == "single_segment_approved"]
    assert len(approved) == 1
    assert approved[0]["operator_id"] == "operator"
    assert approved[0]["proposal_id"]
    assert approved[0]["permit_id"] == "fake-permit"
    assert approved[0]["permit_sha256"]
    assert approved[0]["preflight_fingerprint"] == "1" * 64
    assert approved[0]["stop_generation"] == 4
    assert approved[0]["stop_latched"] is True
    assert approved[0]["occupancy_content_hash"] == "6" * 64
    assert approved[0]["robot_geometry_hash"] == "3" * 64
    assert safety.executor is not None and safety.executor.execute_calls == 1


def test_perception_timing_budget_blocks_before_map_publish(tmp_path: Path) -> None:
    clock = iter((10.0, 11.000001))
    coordinator, _, _, perception, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        coordinator_updates={"maximum_perception_cycle_duration_s": 1.0},
        monotonic_clock=lambda: next(clock),
    )
    coordinator.start()

    with pytest.raises(StopScanBlocked, match="before publish"):
        coordinator.capture_infer_update("too-slow")

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert perception.committed == []
    assert publisher.current_if_available() is None


def test_perception_timing_budget_accepts_exact_boundary_and_records_duration(
    tmp_path: Path,
) -> None:
    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index
            self.events.append((event_type, dict(payload)))

    sink = RecordingSink()
    clock = iter((10.0, *([11.0] * 16)))
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        coordinator_updates={"maximum_perception_cycle_duration_s": 1.0},
        monotonic_clock=lambda: next(clock),
        event_sink=sink,
    )
    coordinator.start()

    coordinator.capture_infer_update("exact-boundary")

    assert len(perception.committed) == 1
    completed = next(
        payload
        for name, payload in sink.events
        if name == "foundation_stereo_completed"
    )
    assert completed["perception_cycle_duration_s"] == pytest.approx(1.0)
    assert completed["maximum_perception_cycle_duration_s"] == pytest.approx(1.0)


def test_perception_validation_time_is_checked_at_publish_boundary(
    tmp_path: Path,
) -> None:
    clock = SimpleNamespace(value=0.0)
    coordinator, _, _, perception, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        coordinator_updates={"maximum_perception_cycle_duration_s": 1.0},
        monotonic_clock=lambda: clock.value,
    )
    original_validate = coordinator._validate_perception_result  # noqa: SLF001

    def slow_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        clock.value = 1.000001
        return result

    coordinator._validate_perception_result = slow_validation  # type: ignore[method-assign]  # noqa: SLF001
    coordinator.start()

    with pytest.raises(StopScanBlocked, match="before publish"):
        coordinator.capture_infer_update("slow-validation")

    assert perception.committed == []
    assert publisher.current_if_available() is None


def test_planned_segment_timing_budget_blocks_before_approval(tmp_path: Path) -> None:
    coordinator, _, _, _, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        coordinator_updates={"maximum_segment_execution_duration_s": 0.05},
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    with pytest.raises(StopScanBlocked, match="planned segment exceeds"):
        coordinator.prepare_next_segment()

    assert coordinator.checkpoint.phase is StopScanPhase.MOTION_BLOCKED


def test_actual_segment_timing_overrun_stops_and_aborts(tmp_path: Path) -> None:
    # The perception transaction now rechecks its deadline at every commit gate.
    clock = iter((0.0, *([0.1] * 8), 10.0, 11.000001))
    coordinator, source, _, _, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        coordinator_updates={
            "maximum_perception_cycle_duration_s": 1.0,
            "maximum_segment_execution_duration_s": 1.0,
        },
        monotonic_clock=lambda: next(clock),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    stops_before = source.calls

    with pytest.raises(StopScanBlocked, match="segment execution exceeded"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED
    assert source.calls > stops_before


def test_unconfirmed_emergency_stop_is_persisted_as_terminal_failed(
    tmp_path: Path,
) -> None:
    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, dict[str, object]]] = []

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del cycle_index
            self.events.append((phase, event_type, dict(payload)))

    sink = RecordingSink()
    coordinator, source, _, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=sink,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    executor = safety.executor
    assert executor is not None
    operation_error = RuntimeError("ServoJ transport failed")
    executor_error = EmergencyStopUnconfirmedError(
        operation_error,
        (
            RuntimeError("watchdog Dashboard stop failed"),
            RuntimeError("executor boundary stop failed"),
        ),
    )

    def fail_execution(*args, **kwargs):
        del args, kwargs
        raise executor_error

    def fail_coordinator_stop() -> None:
        source.calls += 1
        raise RuntimeError("coordinator fallback stop failed")

    executor.execute = fail_execution  # type: ignore[method-assign]
    source.stop = fail_coordinator_stop  # type: ignore[method-assign]

    with pytest.raises(EmergencyStopUnconfirmedError) as raised:
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert raised.value.operation_error is operation_error
    assert len(raised.value.stop_errors) == 3
    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    terminal = [event for event in sink.events if event[1].endswith("stop_unconfirmed")]
    assert len(terminal) == 1
    phase, event_type, payload = terminal[0]
    assert phase == StopScanPhase.FAILED.value
    assert event_type == "single_segment_emergency_stop_unconfirmed"
    assert payload["error_code"] == "emergency_stop_unconfirmed"
    assert len(payload["stop_failures"]) == 3
    with pytest.raises(StopScanError, match="Cannot capture from phase failed"):
        coordinator.capture_infer_update("ordinary-recovery-forbidden")


def test_approval_event_failure_prevents_all_motion(tmp_path: Path) -> None:
    class FailApprovalEvent:
        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index, payload
            if event_type == "single_segment_approved":
                raise OSError("approval audit store unavailable")

    coordinator, _, _, _, safety, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=FailApprovalEvent(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()

    with pytest.raises(StopScanError, match="persistence"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert safety.executor is not None and safety.executor.execute_calls == 0


def test_final_target_capture_is_explicit_candidate_purpose(tmp_path: Path) -> None:
    target = NextViewTarget(
        "fine-front-near",
        (0.04, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3, 3],
        target=target,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    prepared = coordinator.prepare_next_segment()
    assert prepared is not None and prepared.proposal.final_target is True
    coordinator.execute_approved(
        operator_id="operator",
        confirmation="EXECUTE synthetic",
    )

    # The default fake produces only safety occupancy.  A candidate capture must
    # therefore be rejected even though its coordinator-assigned purpose is correct.
    with pytest.raises(StopScanBlocked):
        coordinator.capture_infer_update()

    assert perception.capture_requests[-1] == (
        target.view_id,
        CapturePurpose.CANDIDATE,
    )
    assert perception.committed == [("bootstrap-ready", 0)]


@pytest.mark.parametrize(
    "missing_asset",
    ["blade_foreground_path", "reconstructed_view_path", "coverage_path"],
)
def test_candidate_missing_any_science_asset_fails_before_commit_or_publish(
    tmp_path: Path,
    missing_asset: str,
) -> None:
    target = NextViewTarget(
        "fine-front-near",
        (0.04, 0.0, 0.0, 0.0, 0.0, 0.0),
        tuple(tuple(float(value) for value in row) for row in np.eye(4)),
    )
    coordinator, _, _, perception, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3, 3],
        target=target,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    accepted_generation_id = publisher.current.generation_id
    prepared = coordinator.prepare_next_segment()
    assert prepared is not None and prepared.proposal.final_target is True
    coordinator.execute_approved(
        operator_id="operator",
        confirmation="EXECUTE synthetic",
    )

    original_infer = perception.infer_and_update

    def infer_with_incomplete_science(captured):
        result = original_infer(captured)
        mask_path = captured.cycle_root / "blade-foreground"
        reconstructed_path = captured.cycle_root / "reconstructed-view"
        coverage_path = captured.cycle_root / "surface-coverage"
        for path in (mask_path, reconstructed_path, coverage_path):
            path.mkdir()
        paths = {
            "blade_foreground_path": mask_path,
            "reconstructed_view_path": reconstructed_path,
            "coverage_path": coverage_path,
        }
        paths[missing_asset] = None
        incomplete = replace(result, **paths)
        perception.pending = (captured, incomplete)
        return incomplete

    perception.infer_and_update = infer_with_incomplete_science  # type: ignore[method-assign]

    with pytest.raises(StopScanBlocked):
        coordinator.capture_infer_update()

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert perception.committed == [("bootstrap-ready", 0)]
    assert publisher.current.generation_id == accepted_generation_id


def test_motion_blocked_operator_capture_is_safety_refresh(tmp_path: Path) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3, 3],
        target=_target(),
        clear=False,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    assert coordinator.checkpoint.phase is StopScanPhase.MOTION_BLOCKED
    assert coordinator.checkpoint.expected_capture_view_id is None

    coordinator.capture_infer_update("safety-refresh-0")

    assert perception.capture_requests[-1] == (
        "safety-refresh-0",
        CapturePurpose.SAFETY_REFRESH,
    )


@pytest.mark.parametrize(
    ("tampered_boundary", "message"),
    [
        ("capture", "identity, sequence, or purpose changed"),
        ("result", "Capture purpose changed during perception"),
    ],
)
def test_coordinator_rejects_capture_purpose_drift(
    tmp_path: Path,
    tampered_boundary: str,
    message: str,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    if tampered_boundary == "capture":
        perception.capture_purpose_override = CapturePurpose.SAFETY_REFRESH
    else:
        perception.result_purpose_override = CapturePurpose.CANDIDATE
    coordinator.start()

    with pytest.raises(StopScanBlocked, match=message):
        coordinator.capture_infer_update("bootstrap-ready")

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED
    assert perception.committed == []


def test_awaiting_capture_without_expected_view_is_rejected(tmp_path: Path) -> None:
    coordinator, *_ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    coordinator._phase = StopScanPhase.AWAITING_CAPTURE
    coordinator._expected_capture_view_id = None

    with pytest.raises(StopScanError, match="has no expected post-motion view"):
        coordinator.capture_infer_update("operator-name-cannot-recover-purpose")


def test_transit_cycle_may_carry_prior_coverage_but_not_publish_external_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    captured = perception.capture(
        "transit_fine-front-001_cycle_0000",
        0,
        purpose=CapturePurpose.TRANSIT,
    )
    result = perception.infer_and_update(captured)
    prior_coverage = tmp_path / "persistent_surface_coverage"
    prior_coverage.mkdir()

    semantic_reads: list[Path] = []

    def read_prior_coverage(path: Path, **kwargs):
        assert kwargs == {"require_foreground_bound_science": True}
        semantic_reads.append(Path(path).resolve())
        return SimpleNamespace(
            root=prior_coverage.resolve(),
            ledger=SimpleNamespace(observation_ids=()),
            current_reconstructed_view_path=None,
        )

    monkeypatch.setattr(
        stop_scan_module,
        "read_surface_coverage_generation",
        read_prior_coverage,
    )
    carried = replace(result, coverage_path=prior_coverage)

    captured_candidate = replace(captured, purpose=CapturePurpose.CANDIDATE)
    carried_candidate = replace(carried, purpose=CapturePurpose.CANDIDATE)
    with pytest.raises(StopScanBlocked, match="allowed only for a bootstrap"):
        coordinator._validate_perception_result(
            captured_candidate,
            carried_candidate,
            CapturePurpose.CANDIDATE,
        )

    verified = coordinator._validate_perception_result(
        captured,
        carried,
        CapturePurpose.TRANSIT,
    )

    assert verified.snapshot == result.stored_occupancy.snapshot
    assert semantic_reads == [prior_coverage.resolve()]

    reconstruction = captured.cycle_root / "reconstructed_view"
    reconstruction.mkdir()
    invalid_successor = replace(
        carried,
        reconstructed_view_path=reconstruction,
    )
    with pytest.raises(StopScanBlocked, match="new surface coverage escaped"):
        coordinator._validate_perception_result(
            captured,
            invalid_successor,
            CapturePurpose.TRANSIT,
        )


def test_resume_bootstrap_may_carry_semantically_verified_prior_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    captured = perception.capture(
        "resume-bootstrap",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    result = perception.infer_and_update(captured)
    prior_coverage = tmp_path / "accepted_surface_coverage"
    prior_coverage.mkdir()
    semantic_reads: list[Path] = []

    def read_prior_coverage(path: Path, **kwargs):
        assert kwargs == {"require_foreground_bound_science": True}
        semantic_reads.append(Path(path).resolve())
        return SimpleNamespace(
            root=prior_coverage.resolve(),
            ledger=SimpleNamespace(observation_ids=()),
            current_reconstructed_view_path=None,
        )

    monkeypatch.setattr(
        stop_scan_module,
        "read_surface_coverage_generation",
        read_prior_coverage,
    )

    verified = coordinator._validate_perception_result(
        captured,
        replace(result, coverage_path=prior_coverage),
        CapturePurpose.BOOTSTRAP,
    )

    assert verified.snapshot == result.stored_occupancy.snapshot
    assert semantic_reads == [prior_coverage.resolve()]


def test_external_coverage_cannot_bypass_semantic_reader(
    tmp_path: Path,
) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    captured = perception.capture(
        "resume-bootstrap",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    result = perception.infer_and_update(captured)
    invalid_coverage = tmp_path / "not_a_surface_coverage_asset"
    invalid_coverage.mkdir()

    with pytest.raises(
        StopScanBlocked,
        match="Science asset failed independent semantic readback",
    ):
        coordinator._validate_perception_result(
            captured,
            replace(result, coverage_path=invalid_coverage),
            CapturePurpose.BOOTSTRAP,
        )


@pytest.mark.parametrize(
    "changed_field",
    ["reference_model_sha256", "selection_policy_sha256"],
)
def test_run_rejects_reference_or_selector_policy_drift(
    tmp_path: Path,
    changed_field: str,
) -> None:
    coordinator, _, _, _, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    first = NextViewSelection(
        target=_target(),
        surface_generation_id="a" * 64,
        reference_model_sha256="b" * 64,
        selection_policy_sha256="c" * 64,
        required_patch_count=4,
        incomplete_patch_count=1,
        coverage_complete=False,
    )
    coordinator._validate_selection_run_binding(first)
    changed = replace(first, **{changed_field: "d" * 64})

    with pytest.raises(BladePlanningAssetError, match="changed within one run"):
        coordinator._validate_selection_run_binding(changed)


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


def test_blocked_highest_gain_candidate_falls_back_to_next_safe_path(
    tmp_path: Path,
) -> None:
    class RankedSelector:
        def __init__(self) -> None:
            first = _target()
            second = NextViewTarget(
                "fine-back-002",
                (0.15, 0.0, 0.0, 0.0, 0.0, 0.0),
                tuple(tuple(float(value) for value in row) for row in np.eye(4)),
            )
            self.accepted: list[str] = []
            self.selection = NextViewSelection(
                target=first,
                surface_generation_id="a" * 64,
                reference_model_sha256="b" * 64,
                selection_policy_sha256="c" * 64,
                required_patch_count=4,
                incomplete_patch_count=2,
                coverage_complete=False,
                diagnostics=("science_rank=1", "expected_scientific_gain=0.8"),
                ranked_candidates=(
                    RankedNextViewCandidate(
                        first,
                        ("science_rank=1", "expected_scientific_gain=0.8"),
                    ),
                    RankedNextViewCandidate(
                        second,
                        ("science_rank=2", "expected_scientific_gain=0.7"),
                    ),
                ),
            )

        def select_next(self, observation, generation):
            del observation, generation
            return self.selection

        def accept_preflight_target(self, view_id: str) -> None:
            self.accepted.append(view_id)
            self.selection = self.selection.choose_ranked_candidate(view_id)

    class PathSelectiveSafety(FakeSafetyFactory):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attempted: list[str] = []

        def prepare(self, proposal, generation):
            self.attempted.append(proposal.target_view_id)
            self.clear = proposal.target_view_id == "fine-back-002"
            return super().prepare(proposal, generation)

    class RecordingSink:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def append_event(self, *, phase, cycle_index, event_type, payload):
            del phase, cycle_index
            self.events.append((event_type, dict(payload)))

    sink = RecordingSink()
    coordinator, source, _, _, original_safety, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
        event_sink=sink,
        coordinator_updates={"maximum_ranked_preflight_candidates": 1},
    )
    selector = RankedSelector()
    safety = PathSelectiveSafety(
        source,
        publisher,
        original_safety.motion_config,
        original_safety.occupancy_config,
        original_safety.coordinator_config,
    )
    coordinator._selector = selector  # type: ignore[assignment]  # noqa: SLF001
    coordinator._safety_factory = safety  # type: ignore[assignment]  # noqa: SLF001
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    prepared = coordinator.prepare_next_segment()

    assert prepared is not None and prepared.ready_for_approval
    assert prepared.proposal.target_view_id == "fine-back-002"
    assert coordinator.checkpoint.phase is StopScanPhase.WAITING_APPROVAL
    assert safety.attempted == ["fine-front-001", "fine-back-002"]
    assert selector.accepted == ["fine-back-002"]
    rejected = [
        payload
        for event_type, payload in sink.events
        if event_type == "ranked_candidate_preflight_rejected"
    ]
    assert rejected == [
        {
            "target_view_id": "fine-front-001",
            "preflight_attempt_index": 1,
            "science_rank": 1,
            "blocking_reasons": ["continuous_swept_mesh_unavailable"],
            "safety_thresholds_unchanged": True,
        }
    ]


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
        occupancy_metadata_sha256=hashlib.sha256(changed_metadata.read_bytes()).hexdigest(),
    )
    _FAKE_OCCUPANCY_ASSETS[changed_path.resolve()] = changed
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


def test_disk_tamper_after_preflight_blocks_at_freeze_before_motion(
    tmp_path: Path,
) -> None:
    coordinator, _, stop, _, safety, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    coordinator.prepare_next_segment()
    current = publisher.current
    metadata = current.artifact_path / "metadata.json"
    metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(StopScanBlocked, match="disk-authority readback before freeze"):
        coordinator.execute_approved(
            operator_id="operator",
            confirmation="EXECUTE synthetic",
        )

    assert coordinator.checkpoint.phase is StopScanPhase.ABORTED
    assert safety.executor is not None and safety.executor.execute_calls == 0
    assert stop.calls == 2


def test_publish_acceptance_timeout_keeps_previous_generation(
    tmp_path: Path,
) -> None:
    coordinator, _, _, _, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    current = publisher.current
    candidate_path = tmp_path / "candidate-generation"
    candidate_path.mkdir()
    candidate_metadata = candidate_path / "metadata.json"
    candidate_metadata.write_text("{}", encoding="utf-8")
    candidate_mapping = _stored_mapping(
        3,
        sequence_offset=30,
        occupancy_metadata_sha256=hashlib.sha256(candidate_metadata.read_bytes()).hexdigest(),
    )
    _FAKE_OCCUPANCY_ASSETS[candidate_path.resolve()] = candidate_mapping
    candidate = OccupancyGeneration.verified(
        candidate_path,
        candidate_mapping,
        inference_stationarity_path=current.inference_stationarity_path,
        inference_stationarity_sha256=current.inference_stationarity_sha256,
    )

    def timed_out_acceptance() -> None:
        raise TimeoutError("synthetic commit-boundary deadline")

    with pytest.raises(TimeoutError, match="commit-boundary deadline"):
        publisher.publish_after_acceptance(candidate, timed_out_acceptance)

    assert publisher.current.generation_id == current.generation_id


def test_publish_commit_and_freeze_are_serialized_by_one_authority_lock(
    tmp_path: Path,
) -> None:
    coordinator, _, _, _, _, publisher = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    original = publisher.current
    candidate_path = tmp_path / "serialized-generation"
    candidate_path.mkdir()
    candidate_metadata = candidate_path / "metadata.json"
    candidate_metadata.write_text("{}", encoding="utf-8")
    candidate_mapping = _stored_mapping(
        3,
        sequence_offset=40,
        occupancy_metadata_sha256=hashlib.sha256(candidate_metadata.read_bytes()).hexdigest(),
    )
    _FAKE_OCCUPANCY_ASSETS[candidate_path.resolve()] = candidate_mapping
    candidate = OccupancyGeneration.verified(
        candidate_path,
        candidate_mapping,
        inference_stationarity_path=original.inference_stationarity_path,
        inference_stationarity_sha256=original.inference_stationarity_sha256,
    )
    accept_entered = threading.Event()
    release_accept = threading.Event()
    publish_failures: list[BaseException] = []

    def accept() -> None:
        accept_entered.set()
        assert release_accept.wait(timeout=2.0)

    def publish() -> None:
        try:
            publisher.publish_after_acceptance(candidate, accept)
        except BaseException as exc:
            publish_failures.append(exc)

    worker = threading.Thread(target=publish)
    worker.start()
    assert accept_entered.wait(timeout=2.0)
    freeze_finished = threading.Event()
    freeze_failures: list[BaseException] = []

    def freeze_old() -> None:
        try:
            with publisher.freeze(
                expected_generation_id=original.generation_id,
                expected_binding=original.binding,
                expected_inference_stationarity_sha256=(original.inference_stationarity_sha256),
            ):
                pass
        except BaseException as exc:
            freeze_failures.append(exc)
        finally:
            freeze_finished.set()

    freezer = threading.Thread(target=freeze_old)
    freezer.start()
    assert not freeze_finished.wait(timeout=0.05)
    release_accept.set()
    worker.join(timeout=2.0)
    freezer.join(timeout=2.0)

    assert not worker.is_alive() and not freezer.is_alive()
    assert publish_failures == []
    assert len(freeze_failures) == 1
    assert isinstance(freeze_failures[0], StopScanBlocked)
    assert "changed before freeze" in str(freeze_failures[0])


def test_multi_view_raw_session_is_rejected_before_inference(tmp_path: Path) -> None:
    coordinator, _, _, perception, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    original_capture = perception.capture

    def invalid_capture(
        view_id: str,
        sequence_index: int,
        *,
        purpose: CapturePurpose,
    ):
        captured = original_capture(
            view_id,
            sequence_index,
            purpose=purpose,
        )
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
    assert stored.events[-1].payload["surface_generation_id"] == "a" * 64
    assert stored.events[-1].payload["reference_model_sha256"] == "b" * 64
    assert stored.events[-1].payload["selection_policy_sha256"] == "c" * 64
    assert stored.events[-1].payload["required_patch_count"] == 4
    assert all(event.event_sha256 for event in stored.events)


def test_completion_event_binds_terminal_reconstruction_evidence(tmp_path: Path) -> None:
    final_root = (tmp_path / "final_reconstruction").resolve()
    final_root.mkdir()

    class FinalSelector:
        def select_next(self, observation, generation):
            del observation, generation
            return NextViewSelection(
                target=None,
                surface_generation_id="a" * 64,
                reference_model_sha256="b" * 64,
                selection_policy_sha256="c" * 64,
                required_patch_count=4,
                incomplete_patch_count=0,
                coverage_complete=True,
                diagnostics=("terminal reconstruction replayed",),
                final_reconstruction_path=final_root,
                final_reconstruction_id="d" * 64,
                final_reconstruction_metadata_sha256="e" * 64,
            )

    writer = StopScanRunWriter.create(tmp_path / "run", run_id="final-evidence-test")
    coordinator, *_ = _coordinator(
        tmp_path / "cycles",
        mapping_counts=[3],
        target=None,
        event_sink=writer,
    )
    coordinator._selector = FinalSelector()
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")
    assert coordinator.prepare_next_segment() is None

    terminal = read_stop_scan_run(tmp_path / "run").events[-1]
    assert terminal.payload["final_reconstruction"] == {
        "path": str(final_root),
        "artifact_id": "d" * 64,
        "metadata_sha256": "e" * 64,
    }


def test_next_view_completion_requires_typed_zero_gap_evidence() -> None:
    with pytest.raises(ValueError, match="requires a concrete target"):
        NextViewSelection(
            target=None,
            surface_generation_id="a" * 64,
            reference_model_sha256="b" * 64,
            selection_policy_sha256="c" * 64,
            required_patch_count=4,
            incomplete_patch_count=0,
            coverage_complete=False,
        )


def test_untyped_selector_result_is_terminal_asset_failure(tmp_path: Path) -> None:
    class UntypedSelector:
        def select_next(self, observation, generation):
            del observation, generation
            return None

    coordinator, *_ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator._selector = UntypedSelector()  # type: ignore[assignment]
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    with pytest.raises(BladePlanningAssetError, match="untyped decision"):
        coordinator.prepare_next_segment()

    assert coordinator.checkpoint.phase is StopScanPhase.FAILED


def test_incomplete_coverage_without_reachable_view_is_motion_blocked(
    tmp_path: Path,
) -> None:
    class BlockedSelector:
        def select_next(self, observation, generation):
            del observation, generation
            raise NextViewUnavailable("no endpoint-feasible incomplete patch")

    coordinator, *_ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=None,
    )
    coordinator._selector = BlockedSelector()  # type: ignore[assignment]
    coordinator.start()
    coordinator.capture_infer_update("bootstrap-ready")

    with pytest.raises(NextViewUnavailable, match="no endpoint-feasible"):
        coordinator.prepare_next_segment()

    assert coordinator.checkpoint.phase is StopScanPhase.MOTION_BLOCKED


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

    def blocking_execute(
        preflight,
        permit,
        *,
        cancellation_requested,
        maximum_duration_s,
    ):
        del preflight, permit
        assert maximum_duration_s is None
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


def test_aborted_without_transport_ack_retries_physical_stop(tmp_path: Path) -> None:
    coordinator, _, stop, _, _, _ = _coordinator(
        tmp_path,
        mapping_counts=[3],
        target=_target(),
    )
    coordinator.start()
    stop_attempts = 0
    original_stop = stop.stop

    def fail_once_then_stop() -> None:
        nonlocal stop_attempts
        stop_attempts += 1
        if stop_attempts == 1:
            raise RuntimeError("synthetic stop transport failure")
        original_stop()

    stop.stop = fail_once_then_stop  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="stop transport failure"):
        coordinator.request_stop("operator stop")

    failed = coordinator.checkpoint
    assert failed.phase is StopScanPhase.ABORTED
    assert failed.stop_requested is True
    assert failed.stop_transport_acknowledged is False

    retried = coordinator.request_stop("operator stop retry")

    assert stop_attempts == 2
    assert retried.phase is StopScanPhase.ABORTED
    assert retried.stop_transport_acknowledged is True


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
            max_tcp_translation_delta_m=(perception.acquisition_config.max_tcp_translation_delta_m),
            max_tcp_rotation_delta_rad=(perception.acquisition_config.max_tcp_rotation_delta_rad),
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
            max_tcp_translation_delta_m=(perception.acquisition_config.max_tcp_translation_delta_m),
            max_tcp_rotation_delta_rad=(perception.acquisition_config.max_tcp_rotation_delta_rad),
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
        wrong = (
            replace(
                stored.frame_evidence[-1],
                source_view_id="old-view",
            )
            if hasattr(stored.frame_evidence[-1], "__dataclass_fields__")
            else SimpleNamespace(
                **{
                    **vars(stored.frame_evidence[-1]),
                    "source_view_id": "old-view",
                }
            )
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
        occupancy_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    )
    _FAKE_OCCUPANCY_ASSETS[changed_path.resolve()] = changed_mapping
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

    def blocking_commit(captured, result, *, before_commit=lambda _stage: None):
        commit_entered.set()
        assert release_commit.wait(timeout=2.0)
        original_commit(captured, result, before_commit=before_commit)

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

    def failing_commit(captured, result, *, before_commit=lambda _stage: None):
        del captured, result
        before_commit("before_synthetic_failure")
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
