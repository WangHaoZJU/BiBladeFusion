from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.foundation_stereo_cycle as cycle_module
from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    OccupancyConfig,
    StopAndCaptureConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.stereo import FoundationStereoBackend
from biblade_fusion.robotics import load_es68_flange_t_tcp
from biblade_fusion.workflows.foundation_stereo_cycle import (
    FoundationStereoCycleError,
    FoundationStereoOccupancyCycleEngine,
    _PendingPerceptionCommit,
)
from biblade_fusion.workflows.stop_scan_coordinator import CapturePurpose


def _state(timestamp: int) -> RobotState:
    return RobotState(
        monotonic_time_ns=timestamp,
        controller_time_s=timestamp / 1e9,
        joint_positions_rad=np.zeros(6),
        base_t_tcp=PoseSE3.identity("base", "tcp"),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.1,
    )


def _bundle(view_id: str, sequence_index: int) -> SynchronizedFrameBundle:
    state = _state(1_000_000_000 + sequence_index * 1_000_000)
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
        sequence_index + 1,
        1.0,
        1.0,
        image,
        image,
        None,
        calibration,
    )
    return SynchronizedFrameBundle(
        view_id,
        sequence_index,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
    )


class FakeAcquirer:
    def __init__(self, source: FakeStateSource) -> None:
        self.calls = 0
        self._source = source
        self._config = AcquisitionConfig()
        self.sampler_active_during_capture = False

    @property
    def robot_state_source(self):
        return self._source

    @property
    def acquisition_config(self):
        return self._config.model_copy(deep=True)

    def capture(self, view_id: str, sequence_index: int) -> SynchronizedFrameBundle:
        self.calls += 1
        self.sampler_active_during_capture = any(
            thread.name == "bbf-foundation-stereo-stationarity"
            and thread.is_alive()
            for thread in threading.enumerate()
        )
        return _bundle(view_id, sequence_index)


class FakeStateSource:
    def __init__(self) -> None:
        self.timestamp = 2_000_000_000

    def read_state(self) -> RobotState:
        self.timestamp += 1_000_000
        return _state(self.timestamp)


class UnusedRenderer:
    model_content_hash = "1" * 64
    joint_zero_offsets_rad = (0.0,) * 6

    def base_t_flange_matrix(self, joints):  # pragma: no cover
        del joints
        raise AssertionError

    def render_robot_depth(self, intrinsics, joints, pose):  # pragma: no cover
        del intrinsics, joints, pose
        raise AssertionError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_each_capture_is_a_separate_closed_single_view_session(tmp_path: Path) -> None:
    stereo_calibration = tmp_path / "stereo.yaml"
    hand_eye_path = tmp_path / "hand-eye.yaml"
    stereo_calibration.write_text("calibration: test\n", encoding="utf-8")
    hand_eye_path.write_text("hand_eye: test\n", encoding="utf-8")
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "realsense": settings.realsense.model_copy(
                update={"stereo_calibration_path": stereo_calibration}
            ),
            "occupancy": OccupancyConfig(
                enabled=True,
                workspace_bounds_min_m=(-0.2, -0.2, -0.2),
                workspace_bounds_max_m=(0.2, 0.2, 0.2),
            ),
            "stop_and_capture": StopAndCaptureConfig(
                enabled=True,
                maximum_segment_joint_delta_rad=0.05,
            ),
        }
    )
    flange_t_tcp = load_es68_flange_t_tcp()
    tcp_t_left_ir = PoseSE3.identity("tcp", "left_ir")
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "synthetic",
        20,
        0.001,
        0.1,
        hand_eye_path,
        flange_t_left_ir=flange_t_tcp.compose(tcp_t_left_ir),
    )
    source = FakeStateSource()
    acquirer = FakeAcquirer(source)
    backend = FoundationStereoBackend(settings.foundation_stereo)
    engine = FoundationStereoOccupancyCycleEngine(
        settings=settings,
        acquirer=acquirer,
        state_source=source,
        backend=backend,
        hand_eye=hand_eye,
        renderer=UnusedRenderer(),
        output_root=tmp_path / "run",
    )
    settings.occupancy.maximum_map_age_s = 999.0
    leaked_backend_config = backend.config
    leaked_backend_config.scale = 0.5

    assert engine.occupancy_config.maximum_map_age_s != 999.0
    assert backend.config.scale != 0.5

    first = engine.capture(
        "bootstrap-left",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    first_manifest = first.raw_session_path / "manifest.json"
    first_hash = _sha256(first_manifest)
    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    assert first.captured_at_utc == datetime.fromisoformat(
        first_payload["created_at_utc"]
    )
    engine.cancel_pending_capture(first)
    second = engine.capture(
        "bootstrap-right",
        1,
        purpose=CapturePurpose.BOOTSTRAP,
    )

    assert first.raw_session_path != second.raw_session_path
    assert _sha256(first_manifest) == first_hash
    for captured, expected_view in (
        (first, "bootstrap-left"),
        (second, "bootstrap-right"),
    ):
        manifest = json.loads(
            (captured.raw_session_path / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "completed"
        assert [item["view_id"] for item in manifest["views"]] == [expected_view]
    assert acquirer.calls == 2
    assert acquirer.sampler_active_during_capture is True
    engine.cancel_pending_capture(second)


def test_duplicate_cycle_identity_fails_before_recapture(tmp_path: Path) -> None:
    stereo_calibration = tmp_path / "stereo.yaml"
    hand_eye_path = tmp_path / "hand-eye.yaml"
    stereo_calibration.write_text("calibration: test\n", encoding="utf-8")
    hand_eye_path.write_text("hand_eye: test\n", encoding="utf-8")
    settings = load_settings("configs/default.yaml").model_copy(
        update={
            "realsense": load_settings("configs/default.yaml").realsense.model_copy(
                update={"stereo_calibration_path": stereo_calibration}
            ),
            "occupancy": OccupancyConfig(
                enabled=True,
                workspace_bounds_min_m=(-0.2, -0.2, -0.2),
                workspace_bounds_max_m=(0.2, 0.2, 0.2),
            ),
        }
    )
    tcp_t_left_ir = PoseSE3.identity("tcp", "left_ir")
    source = FakeStateSource()
    engine = FoundationStereoOccupancyCycleEngine(
        settings=settings,
        acquirer=FakeAcquirer(source),
        state_source=source,
        backend=FoundationStereoBackend(settings.foundation_stereo),
        hand_eye=HandEyeCalibration(
            tcp_t_left_ir,
            "synthetic",
            20,
            0.001,
            0.1,
            hand_eye_path,
            flange_t_left_ir=load_es68_flange_t_tcp().compose(tcp_t_left_ir),
        ),
        renderer=UnusedRenderer(),
        output_root=tmp_path / "run",
    )
    captured = engine.capture("same", 0, purpose=CapturePurpose.BOOTSTRAP)

    with pytest.raises(FoundationStereoCycleError, match="still awaiting inference"):
        engine.capture("same", 0, purpose=CapturePurpose.BOOTSTRAP)
    engine.cancel_pending_capture(captured)


def _engine_with_pending_transaction(
    tmp_path: Path,
    *,
    blade_foreground_path: Path | None = None,
    reconstructed_view_path: Path | None = None,
    coverage_path: Path | None = None,
    coverage_metadata_sha256: str | None = None,
) -> tuple[
    FoundationStereoOccupancyCycleEngine,
    SimpleNamespace,
    SimpleNamespace,
]:
    """Construct only the engine state involved in commit/cancel linearization."""

    engine = object.__new__(FoundationStereoOccupancyCycleEngine)
    engine._pending_lock = threading.Lock()
    engine._pending_key = None
    engine._pending_sampler = None
    engine._poisoned_reason = None
    engine._sources = ["accepted-source"]
    accepted = (tmp_path / "accepted-coverage").resolve()
    proposed = (tmp_path / "proposed-coverage").resolve()
    engine._accepted_coverage_path = accepted
    cycle_root = (tmp_path / "cycle").resolve()
    cycle_root.mkdir(exist_ok=True)
    key = ("candidate-001", 7)
    captured = SimpleNamespace(
        bundle=SimpleNamespace(view_id=key[0], sequence_index=key[1]),
        cycle_root=cycle_root,
    )
    occupancy_path = cycle_root / "occupancy"
    stationarity_sha256 = "a" * 64
    result = SimpleNamespace(
        occupancy_mapping_path=occupancy_path,
        inference_stationarity_sha256=stationarity_sha256,
        blade_foreground_path=blade_foreground_path,
        reconstructed_view_path=reconstructed_view_path,
        coverage_path=coverage_path,
    )
    engine._pending_commit = _PendingPerceptionCommit(
        key=key,
        cycle_root=cycle_root,
        occupancy_mapping_path=occupancy_path,
        inference_stationarity_sha256=stationarity_sha256,
        sources=("proposed-source",),
        blade_foreground_path=blade_foreground_path,
        reconstructed_view_path=reconstructed_view_path,
        coverage_path=coverage_path,
        coverage_metadata_sha256=coverage_metadata_sha256,
        accepted_coverage_path_after_commit=proposed,
    )
    return engine, captured, result


def test_cancel_pending_science_transaction_does_not_advance_coverage_or_sources(
    tmp_path: Path,
) -> None:
    proposed = (tmp_path / "candidate-coverage").resolve()
    engine, captured, _ = _engine_with_pending_transaction(
        tmp_path,
        coverage_path=proposed,
        coverage_metadata_sha256="b" * 64,
    )
    accepted_before = engine.accepted_coverage_path
    sources_before = list(engine._sources)

    engine.cancel_pending_capture(captured)

    assert engine._pending_commit is None
    assert engine.accepted_coverage_path == accepted_before
    assert engine._sources == sources_before


def test_tampered_pending_coverage_fails_without_advancing_engine_state(
    tmp_path: Path,
) -> None:
    coverage = (tmp_path / "candidate-coverage").resolve()
    coverage.mkdir()
    metadata = coverage / "coverage.json"
    metadata.write_text('{"generation":"original"}\n', encoding="utf-8")
    pinned_hash = _sha256(metadata)
    engine, captured, result = _engine_with_pending_transaction(
        tmp_path,
        coverage_path=coverage,
        coverage_metadata_sha256=pinned_hash,
    )
    accepted_before = engine.accepted_coverage_path
    sources_before = list(engine._sources)
    metadata.write_text('{"generation":"tampered"}\n', encoding="utf-8")

    with pytest.raises(FoundationStereoCycleError, match="metadata changed"):
        engine.commit_perception_cycle(captured, result)

    assert engine.accepted_coverage_path == accepted_before
    assert engine._sources == sources_before
    assert engine._pending_commit is not None


def test_failed_pending_science_readback_does_not_advance_engine_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask_path = (tmp_path / "blade-foreground").resolve()
    mask_path.mkdir()
    engine, captured, result = _engine_with_pending_transaction(
        tmp_path,
        blade_foreground_path=mask_path,
    )
    accepted_before = engine.accepted_coverage_path
    sources_before = list(engine._sources)
    monkeypatch.setattr(
        cycle_module,
        "read_blade_foreground_mask",
        lambda path: (_ for _ in ()).throw(ValueError("synthetic invalid mask")),
    )

    with pytest.raises(FoundationStereoCycleError, match="asset changed before commit"):
        engine.commit_perception_cycle(captured, result)

    assert engine.accepted_coverage_path == accepted_before
    assert engine._sources == sources_before
    assert engine._pending_commit is not None
