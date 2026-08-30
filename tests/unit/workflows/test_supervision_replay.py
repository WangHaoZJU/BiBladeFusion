from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    FoundationStereoConfig,
    OccupancyConfig,
    StereoRectificationConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.stereo import StereoRectifier, StereoResult
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import (
    SessionReader,
    SessionWriter,
    read_stereo_inference,
    write_reconstructed_view,
    write_stereo_inference,
)
from biblade_fusion.storage.occupancy_mapping import (
    OccupancyMappingValidationDependencies,
    VerifiedHandEyeSource,
    _write_occupancy_mapping_with_dependencies,
)
from biblade_fusion.supervision import load_snapshot_array
from biblade_fusion.workflows import (
    StereoInferenceObservation,
    integrate_foundation_stereo_occupancy,
    reconstruct_foundation_stereo_view,
)
from biblade_fusion.workflows.supervision_replay import (
    build_supervisory_replay_snapshot,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _Artifacts:
    occupancy_root: Path
    stereo_root: Path
    map_snapshot: object
    observation: StereoInferenceObservation
    captured: datetime
    session_root: Path
    bundle: SynchronizedFrameBundle
    hand_eye: HandEyeCalibration
    settings: object


class _NoRobotPixelsRenderer:
    model_content_hash = "a" * 64
    self_mask_excluded_link_names: tuple[str, ...] = ()
    self_mask_render_backend = "test_no_robot_pixels:v1"
    joint_zero_offsets_rad = (0.0,) * 6

    def base_t_flange_matrix(self, joint_positions_rad):
        return Es68KinematicModel.from_resources().base_t_flange(
            joint_positions_rad
        ).matrix

    def render_robot_depth(self, intrinsics, joint_positions_rad, base_t_camera):
        return np.full((intrinsics.height, intrinsics.width), np.inf)


def _bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(12, 10, 20.0, 20.0, 5.5, 4.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), (-0.05, 0.0, 0.0)
        ),
        None,
    )
    left = np.arange(120, dtype=np.uint8).reshape(10, 12)
    stereo = StereoFrame(1_000, 17, 1.0, 1.0, left, left[:, ::-1], None, calibration)
    joints = np.asarray((0.1, -0.2, 0.3, -0.1, 0.2, -0.3), dtype=np.float64)
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(joints)
    predicted_base_t_tcp = base_t_flange.compose(load_es68_flange_t_tcp())
    observed_matrix = predicted_base_t_tcp.matrix.copy()
    observed_matrix[0, 3] += 0.001
    base_t_tcp = PoseSE3("base", "tcp", observed_matrix)
    state = RobotState(
        1_000,
        1.0,
        joints,
        base_t_tcp,
        "IDLE",
        "NORMAL",
        0.2,
    )
    relative_rotation = state.base_t_tcp.rotation.T @ state.base_t_tcp.rotation
    rotation_delta_rad = float(
        np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    return SynchronizedFrameBundle(
        "view-a",
        0,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0.0, 0.0, 0.0, rotation_delta_rad, 0.0),
    )


def _observation(bundle: SynchronizedFrameBundle) -> StereoInferenceObservation:
    rectified = StereoRectifier(
        bundle.stereo.calibration, StereoRectificationConfig()
    ).rectify(bundle.stereo)
    valid = np.ones(rectified.left_ir.shape, dtype=np.bool_)
    confidence = np.full(rectified.left_ir.shape, 0.95, dtype=np.float32)
    result = StereoResult(
        np.full(rectified.left_ir.shape, 4.0, dtype=np.float32),
        valid,
        confidence,
        metadata={
            "backend": "foundation_stereo",
            "left_right_consistency_applied": True,
            "left_right_consistency_threshold_px": 1.0,
            "confidence_semantic": (
                "exp_negative_left_right_disparity_error_not_calibrated_probability"
            ),
        },
    )
    depth = np.full(rectified.left_ir.shape, 0.25, dtype=np.float64)
    return StereoInferenceObservation(
        bundle.view_id, bundle.sequence_index, rectified, result, depth
    )


def _real_artifacts(tmp_path: Path) -> _Artifacts:
    settings = load_settings(_REPOSITORY_ROOT / "configs/default.yaml")
    bundle = _bundle()
    session_writer = SessionWriter.create(tmp_path / "sessions", settings, label="source")
    session_writer.write_bundle(bundle)
    session_writer.close("completed")

    observation = _observation(bundle)
    stereo_root = write_stereo_inference(
        tmp_path / "stereo",
        observation,
        FoundationStereoConfig(device="cpu"),
        StereoRectificationConfig(),
        source_session=session_writer.path,
    )
    hand_eye_path = tmp_path / "hand_eye.yaml"
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(
        bundle.selected_robot_state.joint_positions_rad
    )
    flange_t_left_ir = PoseSE3(
        "flange",
        "left_ir",
        base_t_flange.inverse().matrix,
    )
    tcp_t_left_ir = load_es68_flange_t_tcp().inverse().compose(flange_t_left_ir)
    hand_eye_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "unit-test",
                "matrix": flange_t_left_ir.matrix.tolist(),
                "derived_runtime": {
                    "tcp_T_left_ir": tcp_t_left_ir.matrix.tolist(),
                },
                "quality": {
                    "sample_count": 20,
                    "translation_rmse_m": 0.001,
                    "rotation_rmse_deg": 0.1,
                    "rotation_span_deg": 45.0,
                    "translation_span_m": 0.1,
                    "rotation_axis_diversity": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "unit-test",
        20,
        0.001,
        0.1,
        hand_eye_path,
        flange_t_left_ir=flange_t_left_ir,
    )
    occupancy_config = OccupancyConfig(
        enabled=True,
        voxel_size_m=0.05,
        workspace_bounds_min_m=(-0.25, -0.25, 0.0),
        workspace_bounds_max_m=(0.25, 0.25, 0.5),
        integration_stride=1,
        minimum_valid_depth_fraction=0.1,
        minimum_source_views=3,
    )
    source_reader = SessionReader(session_writer.path)
    captured = datetime.fromisoformat(str(source_reader.manifest["created_at_utc"]))
    stereo_metadata = stereo_root / "metadata.json"
    session_manifest = session_writer.path / "manifest.json"
    session_view_metadata = (
        session_writer.path
        / source_reader.descriptor(bundle.view_id).relative_path
        / "metadata.json"
    )
    update = integrate_foundation_stereo_occupancy(
        None,
        bundle,
        observation,
        hand_eye,
        occupancy_config,
        settings.acquisition,
        _NoRobotPixelsRenderer(),
        captured_at_utc=captured,
        source_stereo_metadata_sha256=hashlib.sha256(
            stereo_metadata.read_bytes()
        ).hexdigest(),
        source_session_manifest_sha256=hashlib.sha256(
            session_manifest.read_bytes()
        ).hexdigest(),
        source_session_view_metadata_sha256=hashlib.sha256(
            session_view_metadata.read_bytes()
        ).hexdigest(),
    )
    validation_dependencies = OccupancyMappingValidationDependencies(
        stereo_reader=read_stereo_inference,
        stereo_source_verifier=lambda stored, session: None,
        session_reader_factory=SessionReader,
        hand_eye_reader=lambda path: VerifiedHandEyeSource(
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            flange_t_left_ir,
        ),
        renderer_factory=lambda offsets: _NoRobotPixelsRenderer(),
    )
    occupancy_root = _write_occupancy_mapping_with_dependencies(
        tmp_path / "occupancy",
        (update,),
        occupancy_config,
        settings.acquisition,
        source_stereo_inferences=(stereo_root,),
        source_sessions=(session_writer.path,),
        source_hand_eye=hand_eye_path,
        validation_dependencies=validation_dependencies,
    )
    return _Artifacts(
        occupancy_root,
        stereo_root,
        update.snapshot,
        observation,
        captured,
        session_writer.path,
        bundle,
        hand_eye,
        settings,
    )


def _array_record(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def test_real_assets_build_self_contained_blocked_replay_snapshot(tmp_path: Path) -> None:
    artifacts = _real_artifacts(tmp_path)

    stored = build_supervisory_replay_snapshot(
        tmp_path / "supervision",
        source_occupancy=artifacts.occupancy_root,
        source_stereo_inference=artifacts.stereo_root,
        created_at_utc=artifacts.captured + timedelta(seconds=2),
    )

    snapshot = stored.snapshot
    assert snapshot.safety.system_state == "BLOCKED"
    assert snapshot.safety.viewer_mode == "REPLAY"
    assert snapshot.safety.viewer_motion_command_capable is False
    assert "occupancy_replay_integrity_only_unverified_for_motion" in (
        snapshot.safety.blocking_reasons
    )
    assert snapshot.robot.actual_tcp_path_base_m is None
    assert snapshot.robot.planned_tcp_path_base_m is None
    assert snapshot.sensor.inference_latency_ms is None
    assert snapshot.sensor.dropped_frame_count is None
    assert snapshot.sensor.confidence is not None
    assert snapshot.occupancy.content_sha256 == artifacts.map_snapshot.content_hash
    assert snapshot.occupancy.state == "UNREADY"
    assert snapshot.occupancy.age_s == pytest.approx(2.0)

    occupied = load_snapshot_array(stored, snapshot.occupancy.occupied_centres_m)
    free = load_snapshot_array(stored, snapshot.occupancy.free_centres_m)
    expected_occupied = np.asarray(
        sorted(artifacts.map_snapshot.occupied_indices), dtype=np.float64
    ).reshape((-1, 3))
    expected_free = np.asarray(
        sorted(artifacts.map_snapshot.free_indices), dtype=np.float64
    ).reshape((-1, 3))
    expected_occupied = np.asarray(artifacts.map_snapshot.origin_m) + (
        expected_occupied + 0.5
    ) * artifacts.map_snapshot.voxel_size_m
    expected_free = np.asarray(artifacts.map_snapshot.origin_m) + (
        expected_free + 0.5
    ) * artifacts.map_snapshot.voxel_size_m
    np.testing.assert_allclose(occupied, expected_occupied)
    np.testing.assert_allclose(free, expected_free)

    depth = load_snapshot_array(stored, snapshot.sensor.depth_m)
    confidence = load_snapshot_array(stored, snapshot.sensor.confidence)
    np.testing.assert_allclose(depth, artifacts.observation.depth_m)
    np.testing.assert_allclose(confidence, artifacts.observation.result.confidence)
    assert snapshot.sensor.occupancy_quality_evidence_sha256 is not None
    assert snapshot.sensor.valid_depth_fraction == pytest.approx(1.0)
    assert snapshot.sensor.fk_tcp_translation_error_m == pytest.approx(0.001)
    assert snapshot.sensor.measured_valid_pixel_count == 120
    assert snapshot.sensor.retained_valid_pixel_count == 120
    assert snapshot.sensor.masked_valid_pixel_count == 0
    assert snapshot.robot.model_id.startswith("es68-d435i:UNVERIFIED:")
    assert any(
        reason.startswith("robot_visualization_")
        for reason in snapshot.safety.blocking_reasons
    )
    assert all(not Path(asset.path).is_absolute() for asset in snapshot.assets)
    assert all((stored.root / asset.path).is_file() for asset in snapshot.assets)
    assert not (stored.root / "snapshot.json").is_symlink()
    assert all(not path.is_symlink() for path in (stored.root / "arrays").iterdir())
    assert all(not path.is_symlink() for path in (stored.root / "assets").iterdir())


def test_replay_snapshot_writer_never_overwrites_existing_asset(tmp_path: Path) -> None:
    artifacts = _real_artifacts(tmp_path)
    output = tmp_path / "supervision"
    build_supervisory_replay_snapshot(
        output,
        source_occupancy=artifacts.occupancy_root,
        created_at_utc=artifacts.captured + timedelta(seconds=1),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        build_supervisory_replay_snapshot(
            output,
            source_occupancy=artifacts.occupancy_root,
            created_at_utc=artifacts.captured + timedelta(seconds=2),
        )


def test_replay_reclassifies_old_mapping_snapshot_as_stale(tmp_path: Path) -> None:
    artifacts = _real_artifacts(tmp_path)

    stored = build_supervisory_replay_snapshot(
        tmp_path / "stale-supervision",
        source_occupancy=artifacts.occupancy_root,
        created_at_utc=artifacts.captured + timedelta(seconds=6),
    )

    assert stored.snapshot.occupancy.state == "STALE"
    assert "occupancy_stale_by_active_maximum_age" in (
        stored.snapshot.safety.blocking_reasons
    )


def test_current_view_requires_exact_occupancy_acquisition_chain(tmp_path: Path) -> None:
    artifacts = _real_artifacts(tmp_path)
    view = reconstruct_foundation_stereo_view(
        artifacts.bundle,
        artifacts.observation,
        np.ones(artifacts.observation.depth_m.shape, dtype=np.bool_),
        artifacts.hand_eye,
        artifacts.settings.point_cloud,
        kinematics_config=artifacts.settings.kinematics,
        hand_eye_config=artifacts.settings.hand_eye,
    )
    view_root = write_reconstructed_view(
        tmp_path / "reconstructed-view",
        view,
        np.ones(artifacts.observation.depth_m.shape, dtype=np.bool_),
        artifacts.hand_eye,
        artifacts.settings.point_cloud,
        artifacts.settings.kinematics,
        artifacts.settings.hand_eye,
        source_session=artifacts.session_root,
        source_stereo_inference=artifacts.stereo_root,
    )

    stored = build_supervisory_replay_snapshot(
        tmp_path / "bound-current-view",
        source_occupancy=artifacts.occupancy_root,
        source_reconstructed_view=view_root,
        created_at_utc=artifacts.captured + timedelta(seconds=1),
    )
    assert stored.snapshot.reconstruction.provenance_status == "CURRENT_RUN_VERIFIED"
    assert stored.snapshot.reconstruction.current_points_m is not None

    metadata_path = view_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source"]["session"] = str(tmp_path / "unrelated-session")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="different acquisition session"):
        build_supervisory_replay_snapshot(
            tmp_path / "unbound-current-view",
            source_occupancy=artifacts.occupancy_root,
            source_reconstructed_view=view_root,
            created_at_utc=artifacts.captured + timedelta(seconds=1),
        )


def test_historical_preflight_uses_canonical_reader_and_exposes_endpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _real_artifacts(tmp_path)
    preflight_root = tmp_path / "preflight"
    preflight_root.mkdir()
    preflight_path = preflight_root / "motion_preflight.json"
    preflight_path.write_text("canonical-reader-owned-bytes\n", encoding="utf-8")
    occupancy_metadata = artifacts.occupancy_root / "metadata.json"
    calls: list[Path] = []
    transforms = []
    for translation in ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = translation
        transforms.append(matrix)
    legs = tuple(
        SimpleNamespace(
            preflight=SimpleNamespace(blocking_reasons=()),
            endpoint_consistency=SimpleNamespace(blocking_reasons=()),
            goal_base_t_tcp_matrix=matrix.tolist(),
        )
        for matrix in transforms
    )
    stored_preflight = SimpleNamespace(
        report=SimpleNamespace(
            ordered_view_ids=("view-a", "view-b"),
            legs=legs,
            ready_for_approval=True,
        ),
        metadata={
            "schema_version": 4,
            "evaluated_at_utc": artifacts.captured.isoformat(),
            "sources": {
                "occupancy": {
                    "root": str(artifacts.occupancy_root),
                    "file": "metadata.json",
                    "sha256": hashlib.sha256(occupancy_metadata.read_bytes()).hexdigest(),
                }
            },
        },
    )

    def canonical_reader(path: Path):
        calls.append(path.resolve())
        return stored_preflight

    monkeypatch.setattr(
        "biblade_fusion.workflows.supervision_replay.read_motion_preflight",
        canonical_reader,
    )
    stored = build_supervisory_replay_snapshot(
        tmp_path / "preflight-supervision",
        source_occupancy=artifacts.occupancy_root,
        source_motion_preflight=preflight_root,
        created_at_utc=artifacts.captured + timedelta(seconds=1),
    )

    assert calls == [preflight_root.resolve()]
    assert stored.snapshot.plan.state == "PREFLIGHT_FAILED"
    assert "historical_preflight_expired_requires_live_revalidation" in (
        stored.snapshot.plan.blocking_reasons
    )
    planned = load_snapshot_array(
        stored, stored.snapshot.robot.planned_tcp_path_base_m
    )
    np.testing.assert_allclose(planned, np.asarray(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))))
    assert stored.snapshot.robot.actual_tcp_path_base_m is None


def test_unbound_coarse_model_is_displayed_only_as_independent_reference(
    tmp_path: Path,
) -> None:
    artifacts = _real_artifacts(tmp_path)
    view = reconstruct_foundation_stereo_view(
        artifacts.bundle,
        artifacts.observation,
        np.ones(artifacts.observation.depth_m.shape, dtype=np.bool_),
        artifacts.hand_eye,
        artifacts.settings.point_cloud,
        kinematics_config=artifacts.settings.kinematics,
        hand_eye_config=artifacts.settings.hand_eye,
    )
    unrelated_view = write_reconstructed_view(
        tmp_path / "unrelated-view",
        view,
        np.ones(artifacts.observation.depth_m.shape, dtype=np.bool_),
        artifacts.hand_eye,
        artifacts.settings.point_cloud,
        artifacts.settings.kinematics,
        artifacts.settings.hand_eye,
        source_session=tmp_path / "different-session",
        source_stereo_inference=artifacts.stereo_root,
    )
    coarse_root = tmp_path / "coarse"
    coarse_root.mkdir()
    arrays = {
        "fused_points_m": view.base_cloud.points_m,
        "fused_normals": np.tile((0.0, 0.0, 1.0), (len(view.base_cloud.points_m), 1)),
        "fused_side_labels": np.ones(len(view.base_cloud.points_m), dtype=np.int8),
    }
    for name, array in arrays.items():
        np.save(coarse_root / f"{name}.npy", array, allow_pickle=False)
    unrelated_metadata = unrelated_view / "metadata.json"
    coarse_payload = {
        "schema_version": 4,
        "motion_authorized": False,
        "created_at_utc": artifacts.captured.isoformat(),
        "source_views": [
            {
                "path": str(unrelated_view),
                "metadata_sha256": hashlib.sha256(
                    unrelated_metadata.read_bytes()
                ).hexdigest(),
            }
        ],
        "files": {
            name: _array_record(coarse_root / f"{name}.npy") for name in arrays
        },
        "quality": {"patches": []},
        "fusion": {"refinements": []},
    }
    (coarse_root / "metadata.json").write_text(
        json.dumps(coarse_payload), encoding="utf-8"
    )

    stored = build_supervisory_replay_snapshot(
        tmp_path / "reference-supervision",
        source_occupancy=artifacts.occupancy_root,
        source_coarse_model=coarse_root,
        created_at_utc=artifacts.captured + timedelta(seconds=1),
    )

    reconstruction = stored.snapshot.reconstruction
    assert reconstruction.provenance_status == "INDEPENDENT_REFERENCE"
    assert reconstruction.model_version.startswith("reference-unbound:")
    assert reconstruction.provenance_reasons
    assert "coarse_model_not_bound_to_occupancy_chain" in (
        stored.snapshot.safety.blocking_reasons
    )
    assert any(
        event.severity == "WARNING" and event.category == "reconstruction"
        for event in stored.snapshot.events
    )
