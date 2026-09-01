from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
from biblade_fusion.robotics import StationarityError, load_es68_flange_t_tcp
from biblade_fusion.storage.inference_stationarity import (
    read_inference_stationarity_trace,
)
from biblade_fusion.workflows.foundation_stereo_cycle import (
    FoundationStereoCycleError,
    FoundationStereoOccupancyCycleEngine,
    _PendingPerceptionCommit,
    _VerifiedSource,
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
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), (-0.05, 0.0, 0.0)),
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
            thread.name == "bbf-foundation-stereo-stationarity" and thread.is_alive()
            for thread in threading.enumerate()
        )
        return _bundle(view_id, sequence_index)


class FakeStateSource:
    def __init__(self) -> None:
        self.timestamp = 2_000_000_000

    def read_state(self) -> RobotState:
        self.timestamp += 1_000_000
        return _state(self.timestamp)


class SignallingStateSource(FakeStateSource):
    def __init__(self, required_reads: int = 3) -> None:
        super().__init__()
        self.required_reads = required_reads
        self.read_count = 0
        self.ready = threading.Event()

    def read_state(self) -> RobotState:
        self.read_count += 1
        state = super().read_state()
        if self.read_count >= self.required_reads:
            self.ready.set()
        return state


class UnusedRenderer:
    model_content_hash = "1" * 64
    self_mask_excluded_link_names: tuple[str, ...] = ()
    self_mask_render_backend = "test_unused:v1"
    joint_zero_offsets_rad = (0.0,) * 6

    def base_t_flange_matrix(self, joints):  # pragma: no cover
        del joints
        raise AssertionError

    def render_robot_depth(self, intrinsics, joints, pose):  # pragma: no cover
        del intrinsics, joints, pose
        raise AssertionError


def _capture_engine(
    tmp_path: Path,
    *,
    acquirer_type=FakeAcquirer,
) -> tuple[FoundationStereoOccupancyCycleEngine, FakeAcquirer]:
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
        }
    )
    source = FakeStateSource()
    acquirer = acquirer_type(source)
    tcp_t_left_ir = PoseSE3.identity("tcp", "left_ir")
    engine = FoundationStereoOccupancyCycleEngine(
        settings=settings,
        acquirer=acquirer,
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
    return engine, acquirer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_robot_state_sampler_uses_and_restores_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(cycle_module.os, "sched_get_priority_max", lambda _policy: 99)
    monkeypatch.setattr(cycle_module.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(
        cycle_module.os,
        "sched_getparam",
        lambda _pid: cycle_module.os.sched_param(0),
    )
    monkeypatch.setattr(
        cycle_module.os,
        "sched_setscheduler",
        lambda pid, policy, parameter: calls.append(
            (pid, policy, parameter.sched_priority)
        ),
    )
    source = SignallingStateSource()
    sampler = cycle_module._RobotStateSampler(source, 0.001, prefer_fifo=True)

    sampler.start()
    assert source.ready.wait(timeout=1.0)
    trace = sampler.finish()

    assert len(trace) >= 3
    assert calls == [
        (0, cycle_module.os.SCHED_FIFO, 10),
        (0, 0, 0),
    ]


def test_robot_state_sampler_retains_fail_closed_trace_when_fifo_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cycle_module.os, "sched_get_priority_max", lambda _policy: 99)
    monkeypatch.setattr(cycle_module.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(
        cycle_module.os,
        "sched_getparam",
        lambda _pid: cycle_module.os.sched_param(0),
    )

    def deny(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(cycle_module.os, "sched_setscheduler", deny)
    source = SignallingStateSource()
    sampler = cycle_module._RobotStateSampler(source, 0.001, prefer_fifo=True)

    sampler.start()
    assert source.ready.wait(timeout=1.0)
    trace = sampler.finish()

    assert len(trace) >= 3


def test_independent_sampler_trace_is_bound_to_camera_rtsi_states(
    tmp_path: Path,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    sampled = tuple(
        replace(
            _state(1_000_000_000 + index * 50_000_000),
            controller_time_s=1.0 + index * 0.05,
        )
        for index in range(3)
    )
    captured = tuple(
        replace(
            sampled[index],
            monotonic_time_ns=2_000_000_000 + index,
            controller_time_s=sampled[index].controller_time_s + 0.001,
        )
        for index in range(3)
    )

    engine._validate_capture_binding(sampled, captured)  # noqa: SLF001

    moved = replace(
        captured[1],
        joint_positions_rad=np.full(6, 0.1),
    )
    with pytest.raises(
        FoundationStereoCycleError,
        match="differs from camera bracket state 1",
    ):
        engine._validate_capture_binding(  # noqa: SLF001
            sampled,
            (captured[0], moved, captured[2]),
        )


def test_capture_uses_injected_process_sampler_boundary(tmp_path: Path) -> None:
    engine, _ = _capture_engine(tmp_path)

    class StubSampler:
        def __init__(self) -> None:
            self.started = False
            self.cancelled = False

        @property
        def is_alive(self) -> bool:
            return self.started and not self.cancelled

        @property
        def diagnostics(self) -> dict[str, object]:
            return {"sampler_kind": "stub"}

        def start(self) -> None:
            self.started = True

        def finish(self) -> tuple[RobotState, ...]:
            return (_state(1), _state(2), _state(3))

        def cancel(self) -> None:
            self.cancelled = True

    sampler = StubSampler()
    engine._robot_state_sampler_factory = lambda: sampler  # noqa: SLF001

    captured = engine.capture("process-boundary", 0, purpose=CapturePurpose.BOOTSTRAP)
    assert sampler.started is True
    engine.cancel_pending_capture(captured)
    assert sampler.cancelled is True


def _window_source(
    tmp_path: Path,
    *,
    sequence_index: int,
    monotonic_time_ns: int,
    captured_at_utc: datetime,
    camera_x_m: float | None = None,
) -> _VerifiedSource:
    bundle = _bundle(f"view-{sequence_index}", sequence_index)
    bundle = replace(
        bundle,
        stereo=replace(bundle.stereo, monotonic_time_ns=monotonic_time_ns),
    )
    captured = SimpleNamespace(bundle=bundle, captured_at_utc=captured_at_utc)
    return _VerifiedSource(
        captured=captured,
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / f"stereo-{sequence_index}",
        stereo_metadata_sha256="1" * 64,
        session_manifest_sha256="2" * 64,
        session_view_metadata_sha256="3" * 64,
        camera_center_base_m=(
            0.03 * sequence_index if camera_x_m is None else camera_x_m,
            0.0,
            0.0,
        ),
        camera_axis_base=(0.0, 0.0, 1.0),
    )


def test_source_window_uses_monotonic_capture_gap_not_wall_clock(tmp_path: Path) -> None:
    engine, _ = _capture_engine(tmp_path)
    stop = engine._settings.stop_and_capture.model_copy(  # noqa: SLF001
        update={"maximum_operator_reposition_interval_s": 2.0}
    )
    engine._settings = engine._settings.model_copy(  # noqa: SLF001
        update={"stop_and_capture": stop}
    )
    now = datetime(2026, 8, 29, tzinfo=UTC)
    engine._utc_clock = lambda: now  # noqa: SLF001
    first = _window_source(
        tmp_path,
        sequence_index=0,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now - timedelta(seconds=1),
    )
    # Deliberately jump the wall clock backwards while the capture clock advances.
    second = _window_source(
        tmp_path,
        sequence_index=1,
        monotonic_time_ns=2_000_000_000,
        captured_at_utc=now - timedelta(seconds=2),
    )
    engine._sources = [first]  # noqa: SLF001

    assert engine._fresh_rebuild_sources(second) == (first, second)  # noqa: SLF001


def test_source_window_retains_frames_older_than_motion_authorization_age(
    tmp_path: Path,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    now = datetime(2026, 8, 29, tzinfo=UTC)
    engine._utc_clock = lambda: now  # noqa: SLF001
    first = _window_source(
        tmp_path,
        sequence_index=0,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now - timedelta(minutes=10),
    )
    second = _window_source(
        tmp_path,
        sequence_index=1,
        monotonic_time_ns=2_000_000_000,
        captured_at_utc=now - timedelta(minutes=5),
    )
    engine._sources = [first]  # noqa: SLF001

    assert engine._fresh_rebuild_sources(second) == (first, second)  # noqa: SLF001


def test_source_window_is_bounded_to_latest_configured_views(tmp_path: Path) -> None:
    engine, _ = _capture_engine(tmp_path)
    now = datetime(2026, 8, 29, tzinfo=UTC)
    sources = tuple(
        _window_source(
            tmp_path,
            sequence_index=index,
            monotonic_time_ns=(index + 1) * 1_000_000_000,
            captured_at_utc=now + timedelta(seconds=index),
        )
        for index in range(4)
    )
    engine._sources = list(sources[:3])  # noqa: SLF001

    assert engine._fresh_rebuild_sources(sources[3]) == sources[1:]  # noqa: SLF001


def test_near_duplicate_capture_replaces_prior_view_without_free_vote(
    tmp_path: Path,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    now = datetime(2026, 8, 29, tzinfo=UTC)
    first = _window_source(
        tmp_path,
        sequence_index=0,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now,
        camera_x_m=0.0,
    )
    second = _window_source(
        tmp_path,
        sequence_index=1,
        monotonic_time_ns=2_000_000_000,
        captured_at_utc=now + timedelta(seconds=1),
        camera_x_m=0.03,
    )
    prior = _window_source(
        tmp_path,
        sequence_index=2,
        monotonic_time_ns=3_000_000_000,
        captured_at_utc=now + timedelta(seconds=2),
        camera_x_m=0.06,
    )
    replacement = _window_source(
        tmp_path,
        sequence_index=3,
        monotonic_time_ns=4_000_000_000,
        captured_at_utc=now + timedelta(seconds=3),
        camera_x_m=0.061,
    )
    engine._sources = [first, second, prior]  # noqa: SLF001

    assert engine._fresh_rebuild_sources(replacement) == (  # noqa: SLF001
        first,
        second,
        replacement,
    )


def test_source_window_discards_prefix_after_monotonic_gap(tmp_path: Path) -> None:
    engine, _ = _capture_engine(tmp_path)
    stop = engine._settings.stop_and_capture.model_copy(  # noqa: SLF001
        update={"maximum_operator_reposition_interval_s": 2.0}
    )
    engine._settings = engine._settings.model_copy(  # noqa: SLF001
        update={"stop_and_capture": stop}
    )
    now = datetime(2026, 8, 29, tzinfo=UTC)
    engine._utc_clock = lambda: now  # noqa: SLF001
    first = _window_source(
        tmp_path,
        sequence_index=0,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now - timedelta(seconds=3),
    )
    second = _window_source(
        tmp_path,
        sequence_index=1,
        monotonic_time_ns=4_000_000_001,
        captured_at_utc=now - timedelta(seconds=2),
    )
    third = _window_source(
        tmp_path,
        sequence_index=2,
        monotonic_time_ns=5_000_000_001,
        captured_at_utc=now - timedelta(seconds=1),
    )
    engine._sources = [first, second]  # noqa: SLF001

    assert engine._fresh_rebuild_sources(third) == (second, third)  # noqa: SLF001


def test_source_window_rejects_nonadvancing_monotonic_capture_clock(
    tmp_path: Path,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    stop = engine._settings.stop_and_capture.model_copy(  # noqa: SLF001
        update={"maximum_operator_reposition_interval_s": 2.0}
    )
    engine._settings = engine._settings.model_copy(  # noqa: SLF001
        update={"stop_and_capture": stop}
    )
    now = datetime(2026, 8, 29, tzinfo=UTC)
    engine._utc_clock = lambda: now  # noqa: SLF001
    first = _window_source(
        tmp_path,
        sequence_index=0,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now - timedelta(seconds=2),
    )
    second = _window_source(
        tmp_path,
        sequence_index=1,
        monotonic_time_ns=1_000_000_000,
        captured_at_utc=now - timedelta(seconds=1),
    )
    engine._sources = [first]  # noqa: SLF001

    with pytest.raises(FoundationStereoCycleError, match="did not advance"):
        engine._fresh_rebuild_sources(second)  # noqa: SLF001


def test_failed_physical_attempt_is_retained_and_same_logical_key_retries(
    tmp_path: Path,
) -> None:
    class FailOnceAcquirer(FakeAcquirer):
        def capture(self, view_id: str, sequence_index: int) -> SynchronizedFrameBundle:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic camera failure")
            return _bundle(view_id, sequence_index)

    engine, acquirer = _capture_engine(tmp_path, acquirer_type=FailOnceAcquirer)

    with pytest.raises(RuntimeError, match="camera failure"):
        engine.capture("same-logical-view", 0, purpose=CapturePurpose.BOOTSTRAP)

    failed_attempts = sorted((tmp_path / "run" / "cycles" / "000000_same-logical-view").iterdir())
    assert len(failed_attempts) == 1
    assert failed_attempts[0].name.startswith("attempt_")
    failed_manifests = tuple((failed_attempts[0] / "raw").glob("*/manifest.json"))
    assert len(failed_manifests) == 1
    failed_manifest = json.loads(failed_manifests[0].read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"

    captured = engine.capture(
        "same-logical-view",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )

    attempts = sorted(captured.cycle_root.parent.iterdir())
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert captured.cycle_root != failed_attempts[0]
    assert acquirer.calls == 2
    engine.cancel_pending_capture(captured)


def test_concurrent_capture_cannot_duplicate_one_physical_attempt(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingAcquirer(FakeAcquirer):
        def capture(self, view_id: str, sequence_index: int) -> SynchronizedFrameBundle:
            self.calls += 1
            entered.set()
            assert release.wait(timeout=2.0)
            return _bundle(view_id, sequence_index)

    engine, acquirer = _capture_engine(tmp_path, acquirer_type=BlockingAcquirer)
    captured: list[object] = []
    failures: list[BaseException] = []

    def first_capture() -> None:
        try:
            captured.append(engine.capture("concurrent", 0, purpose=CapturePurpose.BOOTSTRAP))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=first_capture)
    worker.start()
    assert entered.wait(timeout=2.0)
    with pytest.raises(FoundationStereoCycleError, match="still awaiting"):
        engine.capture("concurrent", 0, purpose=CapturePurpose.BOOTSTRAP)
    release.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert failures == []
    assert len(captured) == 1
    assert acquirer.calls == 1
    engine.cancel_pending_capture(captured[0])


class _DriftingScienceAuthority:
    def __init__(self, *, fail_on_current_call: int) -> None:
        self.fail_on_current_call = fail_on_current_call
        self.current_calls = 0

    def assert_current(self, _settings) -> None:
        self.current_calls += 1
        if self.current_calls == self.fail_on_current_call:
            raise ValueError("synthetic science runtime drift")

    def assert_inference_observation(self, _observation) -> None:
        return None


@pytest.mark.parametrize(
    ("fail_on_current_call", "expected_backend_calls"),
    [(1, 0), (2, 1)],
)
def test_science_authority_is_rechecked_immediately_before_and_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on_current_call: int,
    expected_backend_calls: int,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    captured = engine.capture("authority-window", 0, purpose=CapturePurpose.BOOTSTRAP)
    authority = _DriftingScienceAuthority(
        fail_on_current_call=fail_on_current_call,
    )
    engine._science_authority = authority
    engine._science_authority_settings = engine._settings.model_copy(deep=True)
    backend_calls: list[str] = []
    monkeypatch.setattr(
        cycle_module,
        "infer_rectified_stereo",
        lambda *_args, **_kwargs: backend_calls.append("infer") or SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="science runtime drift"):
        engine.infer_and_update(captured)

    assert backend_calls == ["infer"] * expected_backend_calls
    assert authority.current_calls == fail_on_current_call


def _stub_perception_pipeline(
    engine: FoundationStereoOccupancyCycleEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = SimpleNamespace()

    def write_stereo(path, *_args, **_kwargs) -> None:
        path.mkdir()
        (path / "metadata.json").write_text("{}\n", encoding="utf-8")

    def write_occupancy(path, *_args, **_kwargs) -> None:
        path.mkdir()
        (path / "metadata.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cycle_module, "infer_rectified_stereo", lambda *_args: observation)
    monkeypatch.setattr(cycle_module, "write_stereo_inference", write_stereo)
    monkeypatch.setattr(
        cycle_module,
        "read_stereo_inference",
        lambda _path: SimpleNamespace(observation=observation),
    )
    monkeypatch.setattr(
        cycle_module,
        "verify_stereo_inference_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(cycle_module, "write_occupancy_mapping", write_occupancy)
    monkeypatch.setattr(
        cycle_module,
        "read_occupancy_mapping",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cycle_module.OccupancyBinding,
        "from_mapping",
        classmethod(lambda cls, _mapping: SimpleNamespace(tuple=("synthetic",))),
    )
    monkeypatch.setattr(
        engine,
        "_camera_view_evidence",
        lambda *_args: {
            "camera_center_base_m": (0.0, 0.0, 0.0),
            "camera_axis_base": (0.0, 0.0, 1.0),
        },
    )
    monkeypatch.setattr(engine, "_fresh_rebuild_sources", lambda source: (source,))
    monkeypatch.setattr(
        engine,
        "_rebuild_updates",
        lambda _sources: (SimpleNamespace(),),
    )
    monkeypatch.setattr(
        engine,
        "_prepare_science_assets",
        lambda *_args: SimpleNamespace(
            blade_foreground_path=None,
            reconstructed_view_path=None,
            coverage_path=None,
            advances_coverage=False,
        ),
    )
    monkeypatch.setattr(
        engine,
        "_prepare_coarse_science_asset",
        lambda *_args: None,
    )


def test_rejected_stationarity_persists_sampler_trace_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _capture_engine(tmp_path)

    class DelayedSampler:
        def __init__(self) -> None:
            self.started = False

        @property
        def is_alive(self) -> bool:
            return self.started

        @property
        def diagnostics(self) -> dict[str, object]:
            return {
                "sampler_kind": "elite_rtsi_process",
                "maximum_raw_host_gap_s": 0.4,
                "scheduler": {"policy": "SCHED_FIFO", "priority": 10},
            }

        def start(self) -> None:
            self.started = True

        def finish(self) -> tuple[RobotState, ...]:
            self.started = False
            return tuple(
                RobotState(
                    monotonic_time_ns=round(host_s * 1e9),
                    controller_time_s=controller_s,
                    joint_positions_rad=np.zeros(6),
                    base_t_tcp=PoseSE3.identity("base", "tcp"),
                    robot_mode="IDLE",
                    safety_status="NORMAL",
                    speed_scaling=0.1,
                )
                for host_s, controller_s in (
                    (1.0, 1.0),
                    (1.1, 1.1),
                    (1.5, 1.2),
                )
            )

        def cancel(self) -> None:
            self.started = False

    sampler = DelayedSampler()
    engine._robot_state_sampler_factory = lambda: sampler  # noqa: SLF001
    captured = engine.capture(
        "rejected-trace",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    _stub_perception_pipeline(engine, monkeypatch)

    with pytest.raises(StationarityError, match="sample gap"):
        engine.infer_and_update(captured)

    diagnostic = read_inference_stationarity_trace(
        captured.cycle_root / "inference_stationarity_trace.json"
    )
    assert diagnostic.sampler_diagnostics["maximum_raw_host_gap_s"] == 0.4
    assert len(diagnostic.trace) == 3
    assert not (captured.cycle_root / "inference_stationarity.json").exists()


def test_coarse_foreground_preflight_failure_precedes_occupancy_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _capture_engine(tmp_path)
    captured = engine.capture(
        "preflight-first",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    _stub_perception_pipeline(engine, monkeypatch)
    prepared = SimpleNamespace()
    rebuild_calls = 0

    monkeypatch.setattr(
        cycle_module,
        "prepare_foundation_stereo_occupancy_frame",
        lambda *_args, **_kwargs: prepared,
    )

    def reject_preflight(*_args) -> None:
        raise ValueError("synthetic foreground rejection")

    def rebuild(*_args, **_kwargs):
        nonlocal rebuild_calls
        rebuild_calls += 1
        return (SimpleNamespace(),)

    engine._coarse_science_preflighter = reject_preflight  # noqa: SLF001
    monkeypatch.setattr(engine, "_rebuild_updates", rebuild)

    with pytest.raises(ValueError, match="synthetic foreground rejection"):
        engine.infer_and_update(captured)

    assert rebuild_calls == 0


def test_authoritative_stationarity_includes_exact_capture_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _capture_engine(tmp_path)

    class IndependentSampler:
        def __init__(self) -> None:
            self.started = False

        @property
        def is_alive(self) -> bool:
            return self.started

        @property
        def diagnostics(self) -> dict[str, object]:
            return {
                "sampler_kind": "elite_rtsi_process",
                "scheduler": {"policy": "SCHED_FIFO", "priority": 10},
            }

        def start(self) -> None:
            self.started = True

        def finish(self) -> tuple[RobotState, ...]:
            self.started = False
            return tuple(
                replace(
                    _state(host_ns),
                    controller_time_s=controller_s,
                )
                for host_ns, controller_s in (
                    # This packet arrived first but its controller timestamp is
                    # slightly ahead of the main RTSI capture bracket.
                    (999_000_000, 1.004),
                    (1_050_000_000, 1.05),
                    (1_100_000_000, 1.10),
                )
            )

        def cancel(self) -> None:
            self.started = False

    sampler = IndependentSampler()
    engine._robot_state_sampler_factory = lambda: sampler  # noqa: SLF001
    captured = engine.capture(
        "capture-contract",
        0,
        purpose=CapturePurpose.BOOTSTRAP,
    )
    _stub_perception_pipeline(engine, monkeypatch)

    result = engine.infer_and_update(captured)

    diagnostic = read_inference_stationarity_trace(
        captured.cycle_root / "inference_stationarity_trace.json"
    )
    authoritative = cycle_module.read_inference_stationarity(
        result.inference_stationarity_path
    )
    authoritative_trace = (authoritative.reference, *authoritative.trace)
    capture_states = (
        captured.bundle.robot_state_before,
        captured.bundle.selected_robot_state,
        captured.bundle.robot_state_after,
    )
    assert not any(
        cycle_module._robot_state_values_equal(capture_states[0], state)
        for state in diagnostic.trace
    )
    assert all(
        any(
            cycle_module._robot_state_values_equal(capture_state, sample)
            for sample in authoritative_trace
        )
        for capture_state in capture_states
    )
    assert all(state.controller_time_s != 1.004 for state in authoritative_trace)
    engine.cancel_pending_capture(captured)


def test_committed_logical_key_rejects_before_camera_attempt(tmp_path: Path) -> None:
    engine, acquirer = _capture_engine(tmp_path)
    logical_root = tmp_path / "run" / "cycles" / "000005_committed"
    logical_root.mkdir(parents=True)
    (logical_root / "committed.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FoundationStereoCycleError, match="already committed"):
        engine.capture("committed", 5, purpose=CapturePurpose.BOOTSTRAP)

    assert acquirer.calls == 0
    assert tuple(logical_root.iterdir()) == (logical_root / "committed.json",)


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
    assert first.captured_at_utc == datetime.fromisoformat(first_payload["created_at_utc"])
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


def test_pending_identity_blocks_but_cancelled_identity_can_retry(tmp_path: Path) -> None:
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
    retried = engine.capture("same", 0, purpose=CapturePurpose.BOOTSTRAP)
    assert retried.cycle_root != captured.cycle_root
    engine.cancel_pending_capture(retried)


def _engine_with_pending_transaction(
    tmp_path: Path,
    *,
    blade_foreground_path: Path | None = None,
    reconstructed_view_path: Path | None = None,
    coverage_path: Path | None = None,
    coverage_metadata_sha256: str | None = None,
    coarse_scan_view_path: Path | None = None,
    coarse_scan_metadata_sha256: str | None = None,
) -> tuple[
    FoundationStereoOccupancyCycleEngine,
    SimpleNamespace,
    SimpleNamespace,
]:
    """Construct only the engine state involved in commit/cancel linearization."""

    engine = object.__new__(FoundationStereoOccupancyCycleEngine)
    engine._pending_lock = threading.Lock()
    engine._pending_key = None
    engine._pending_attempt_root = None
    engine._pending_sampler = None
    engine._poisoned_reason = None
    engine._capture_roots = {}
    engine._sources = ["accepted-source"]
    accepted = (tmp_path / "accepted-coverage").resolve()
    proposed = (tmp_path / "proposed-coverage").resolve()
    engine._accepted_coverage_path = accepted
    cycle_root = (tmp_path / "cycle").resolve()
    cycle_root.mkdir(exist_ok=True)
    key = ("candidate-001", 7)
    occupancy_path = cycle_root / "occupancy"
    raw_session_path = cycle_root / "raw"
    stereo_path = cycle_root / "stereo"
    stationarity_path = cycle_root / "inference_stationarity.json"
    captured = SimpleNamespace(
        bundle=SimpleNamespace(
            view_id=key[0],
            sequence_index=key[1],
            stereo=SimpleNamespace(frame_number=19),
        ),
        cycle_root=cycle_root,
        raw_session_path=raw_session_path,
    )
    stationarity_sha256 = "a" * 64
    result = SimpleNamespace(
        raw_session_path=raw_session_path,
        stereo_inference_path=stereo_path,
        occupancy_mapping_path=occupancy_path,
        inference_stationarity_path=stationarity_path,
        inference_stationarity_sha256=stationarity_sha256,
        stored_occupancy=SimpleNamespace(),
        blade_foreground_path=blade_foreground_path,
        reconstructed_view_path=reconstructed_view_path,
        coverage_path=coverage_path,
        coarse_scan_view_path=coarse_scan_view_path,
    )
    engine._pending_commit = _PendingPerceptionCommit(
        key=key,
        cycle_root=cycle_root,
        raw_session_path=raw_session_path,
        raw_session_manifest_sha256="1" * 64,
        raw_session_view_metadata_sha256="2" * 64,
        stereo_inference_path=stereo_path,
        stereo_metadata_sha256="3" * 64,
        occupancy_mapping_path=occupancy_path,
        occupancy_metadata_sha256="4" * 64,
        occupancy_binding=("synthetic",),
        inference_stationarity_path=stationarity_path,
        inference_stationarity_sha256=stationarity_sha256,
        sources=("proposed-source",),
        blade_foreground_path=blade_foreground_path,
        reconstructed_view_path=reconstructed_view_path,
        coverage_path=coverage_path,
        coverage_metadata_sha256=coverage_metadata_sha256,
        accepted_coverage_path_after_commit=proposed,
        coarse_scan_view_path=coarse_scan_view_path,
        coarse_scan_metadata_sha256=coarse_scan_metadata_sha256,
    )
    return engine, captured, result


def _materialize_pinned_core_authority(
    engine: FoundationStereoOccupancyCycleEngine,
    captured: SimpleNamespace,
) -> dict[str, Path]:
    pending = engine._pending_commit
    assert pending is not None
    view_root = pending.raw_session_path / "views" / "0000_candidate"
    view_root.mkdir(parents=True)
    view_metadata = view_root / "metadata.json"
    view_metadata.write_text('{"raw":"view"}\n', encoding="utf-8")
    manifest = pending.raw_session_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "views": [
                    {
                        "view_id": captured.bundle.view_id,
                        "sequence_index": captured.bundle.sequence_index,
                        "path": "views/0000_candidate",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pending.stereo_inference_path.mkdir()
    stereo_metadata = pending.stereo_inference_path / "metadata.json"
    stereo_metadata.write_text('{"stereo":"authority"}\n', encoding="utf-8")
    pending.occupancy_mapping_path.mkdir()
    occupancy_metadata = pending.occupancy_mapping_path / "metadata.json"
    occupancy_metadata.write_text('{"occupancy":"authority"}\n', encoding="utf-8")
    pending.inference_stationarity_path.write_text(
        '{"stationarity":"authority"}\n',
        encoding="utf-8",
    )
    engine._pending_commit = replace(
        pending,
        raw_session_manifest_sha256=_sha256(manifest),
        raw_session_view_metadata_sha256=_sha256(view_metadata),
        stereo_metadata_sha256=_sha256(stereo_metadata),
        occupancy_metadata_sha256=_sha256(occupancy_metadata),
        inference_stationarity_sha256=_sha256(pending.inference_stationarity_path),
    )
    return {
        "raw": manifest,
        "view": view_metadata,
        "stereo": stereo_metadata,
        "stationarity": pending.inference_stationarity_path,
        "occupancy": occupancy_metadata,
    }


@pytest.mark.parametrize(
    "authority_name",
    ["raw", "view", "stereo", "stationarity", "occupancy"],
)
def test_each_core_disk_authority_is_rechecked_at_commit(
    tmp_path: Path,
    authority_name: str,
) -> None:
    engine, captured, result = _engine_with_pending_transaction(tmp_path)
    files = _materialize_pinned_core_authority(engine, captured)
    files[authority_name].write_text(
        files[authority_name].read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    result.inference_stationarity_sha256 = engine._pending_commit.inference_stationarity_sha256

    with pytest.raises(FoundationStereoCycleError, match="disk authority"):
        engine.commit_perception_cycle(captured, result)

    assert engine._pending_commit is not None
    assert engine._capture_roots == {}


def test_semantic_identity_is_rechecked_after_hashes_at_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, captured, result = _engine_with_pending_transaction(tmp_path)
    _materialize_pinned_core_authority(engine, captured)
    pending = engine._pending_commit
    assert pending is not None
    result.inference_stationarity_sha256 = pending.inference_stationarity_sha256
    calls: list[str] = []
    observation = SimpleNamespace(
        source_view_id="wrong-view",
        source_sequence_index=captured.bundle.sequence_index,
        rectified=SimpleNamespace(
            source_frame_number=captured.bundle.stereo.frame_number,
        ),
    )
    monkeypatch.setattr(
        cycle_module,
        "read_stereo_inference",
        lambda path: calls.append("stereo") or SimpleNamespace(observation=observation),
    )
    monkeypatch.setattr(
        cycle_module,
        "verify_stereo_inference_source",
        lambda stored, *, expected_session: calls.append("raw-source"),
    )
    monkeypatch.setattr(
        cycle_module,
        "read_inference_stationarity",
        lambda path: (
            calls.append("stationarity")
            or SimpleNamespace(
                view_id=captured.bundle.view_id,
                sequence_index=captured.bundle.sequence_index,
                source_session_manifest_path=(pending.raw_session_path / "manifest.json").resolve(),
                source_session_manifest_sha256=pending.raw_session_manifest_sha256,
            )
        ),
    )
    mapping = SimpleNamespace(
        frame_evidence=(
            SimpleNamespace(
                source_view_id=captured.bundle.view_id,
                source_sequence_index=captured.bundle.sequence_index,
                frame_number=captured.bundle.stereo.frame_number,
                source_stereo_metadata_sha256=pending.stereo_metadata_sha256,
                source_session_manifest_sha256=pending.raw_session_manifest_sha256,
                source_session_view_metadata_sha256=(pending.raw_session_view_metadata_sha256),
            ),
        )
    )
    monkeypatch.setattr(
        cycle_module,
        "read_occupancy_mapping",
        lambda path: calls.append("occupancy") or mapping,
    )
    monkeypatch.setattr(
        cycle_module.OccupancyBinding,
        "from_mapping",
        classmethod(lambda cls, value: SimpleNamespace(tuple=pending.occupancy_binding)),
    )

    with pytest.raises(FoundationStereoCycleError, match="identity or binding"):
        engine.commit_perception_cycle(captured, result)

    assert calls == ["stereo", "raw-source", "stationarity", "occupancy"]
    assert engine._capture_roots == {}


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


def test_only_successful_commit_occupies_the_logical_capture_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, captured, result = _engine_with_pending_transaction(tmp_path)
    key = (captured.bundle.view_id, captured.bundle.sequence_index)
    assert key not in engine._capture_roots
    monkeypatch.setattr(
        engine,
        "_reverify_pending_authority",
        lambda captured, result, pending: None,
    )

    engine.commit_perception_cycle(captured, result)

    assert engine._pending_commit is None
    assert engine._capture_roots == {key: captured.cycle_root}
    marker = captured.cycle_root.parent / "committed.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["logical_identity"] == {
        "view_id": key[0],
        "sequence_index": key[1],
    }
    assert payload["accepted_attempt"]["root"] == str(captured.cycle_root)


def test_deadline_at_marker_boundary_leaves_no_commit_or_source_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, captured, result = _engine_with_pending_transaction(tmp_path)
    sources_before = list(engine._sources)
    capture_roots_before = dict(engine._capture_roots)
    monkeypatch.setattr(
        engine,
        "_reverify_pending_authority",
        lambda captured, result, pending: None,
    )
    stages: list[str] = []

    def deadline_gate(stage: str) -> None:
        stages.append(stage)
        if stage == "before_logical_commit_marker_link":
            raise TimeoutError("synthetic perception deadline")

    with pytest.raises(TimeoutError, match="perception deadline"):
        engine.commit_perception_cycle(
            captured,
            result,
            before_commit=deadline_gate,
        )

    logical_root = captured.cycle_root.parent
    assert "before_logical_commit_temporary_write" in stages
    assert "before_logical_commit_marker_link" in stages
    assert not (logical_root / "committed.json").exists()
    assert list(logical_root.glob(".committed.*.partial")) == []
    assert engine._sources == sources_before
    assert engine._capture_roots == capture_roots_before
    assert engine._pending_commit is not None


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


def test_coarse_science_hook_is_stopped_transaction_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import biblade_fusion.storage.coarse_scan as coarse_storage

    cycle_root = (tmp_path / "cycle").resolve()
    coarse_path = cycle_root / "coarse_scan_view"
    coarse_path.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        coarse_storage,
        "read_coarse_scan_view",
        lambda path: calls.append(Path(path).resolve()),
    )
    engine = object.__new__(FoundationStereoOccupancyCycleEngine)
    engine._coarse_science_preparer = lambda *args: coarse_path
    captured = SimpleNamespace(
        purpose=CapturePurpose.CANDIDATE,
        cycle_root=cycle_root,
    )

    result = engine._prepare_coarse_science_asset(
        captured,
        SimpleNamespace(),
        cycle_root / "stereo",
        SimpleNamespace(),
        cycle_root / "occupancy",
    )

    assert result == coarse_path
    assert calls == [coarse_path]


def test_coarse_science_hook_cannot_escape_cycle_root(tmp_path: Path) -> None:
    cycle_root = (tmp_path / "cycle").resolve()
    cycle_root.mkdir()
    escaped = (tmp_path / "escaped").resolve()
    escaped.mkdir()
    engine = object.__new__(FoundationStereoOccupancyCycleEngine)
    engine._coarse_science_preparer = lambda *args: escaped
    captured = SimpleNamespace(
        purpose=CapturePurpose.BOOTSTRAP,
        cycle_root=cycle_root,
    )

    with pytest.raises(FoundationStereoCycleError, match="direct child"):
        engine._prepare_coarse_science_asset(
            captured,
            SimpleNamespace(),
            cycle_root / "stereo",
            SimpleNamespace(),
            cycle_root / "occupancy",
        )


def test_tampered_pending_coarse_asset_does_not_commit(tmp_path: Path) -> None:
    coarse = (tmp_path / "cycle" / "coarse_scan_view").resolve()
    coarse.mkdir(parents=True)
    metadata = coarse / "metadata.json"
    metadata.write_text('{"generation":"original"}\n', encoding="utf-8")
    engine, captured, result = _engine_with_pending_transaction(
        tmp_path,
        coarse_scan_view_path=coarse,
        coarse_scan_metadata_sha256=_sha256(metadata),
    )
    metadata.write_text('{"generation":"tampered"}\n', encoding="utf-8")

    with pytest.raises(FoundationStereoCycleError, match="coarse-scan metadata"):
        engine.commit_perception_cycle(captured, result)

    assert engine._sources == ["accepted-source"]
    assert engine._pending_commit is not None
