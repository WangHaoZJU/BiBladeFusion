from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AcquisitionConfig,
    OccupancyConfig,
    StereoRectificationConfig,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.mapping import OccupancyMapState, OccupancyState
from biblade_fusion.perception.stereo import StereoRectifier, StereoResult
from biblade_fusion.robotics import load_es68_flange_t_tcp
from biblade_fusion.workflows import (
    OccupancyMappingError,
    StereoInferenceObservation,
    integrate_foundation_stereo_occupancy,
    mark_snapshot_stale_if_expired,
)
from biblade_fusion.workflows.occupancy_mapping import occupancy_physical_source_id

FOUNDATION_METADATA = {
    "backend": "foundation_stereo",
    "left_right_consistency_applied": True,
    "left_right_consistency_threshold_px": 1.0,
    "confidence_semantic": (
        "exp_negative_left_right_disparity_error_not_calibrated_probability"
    ),
}
SOURCE_SHA256 = "d" * 64


class EmptyRobotRenderer:
    model_content_hash = "a" * 64
    self_mask_excluded_link_names: tuple[str, ...] = ()
    self_mask_render_backend = "test_empty:v1"
    joint_zero_offsets_rad = (0.0,) * 6

    def base_t_flange_matrix(self, joint_positions_rad):
        matrix = load_es68_flange_t_tcp().inverse().matrix.copy()
        matrix[0, 3] += float(joint_positions_rad[0])
        return matrix

    def render_robot_depth(self, intrinsics, joint_positions_rad, base_t_camera):
        return np.full((intrinsics.height, intrinsics.width), np.inf)


class DifferentRobotRenderer(EmptyRobotRenderer):
    model_content_hash = "b" * 64


class InconsistentFkRenderer(EmptyRobotRenderer):
    def base_t_flange_matrix(self, joint_positions_rad):
        del joint_positions_rad
        matrix = load_es68_flange_t_tcp().inverse().matrix.copy()
        matrix[0, 3] = 0.01
        return matrix


def _bundle(
    metrics: CaptureMetrics | None = None,
    *,
    view_id: str = "view-007",
    sequence_index: int = 7,
    base_t_tcp: PoseSE3 | None = None,
    camera_offset_m: float = 0.0,
) -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(10, 10, 20.0, 20.0, 4.5, 4.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir",
            "left_ir",
            np.eye(3),
            [-0.05, 0, 0],
        ),
        None,
    )
    image = np.zeros((10, 10), dtype=np.uint8)
    stereo = StereoFrame(100, sequence_index, 1.0, 1.0, image, image, None, calibration)
    joints = np.zeros(6)
    joints[0] = camera_offset_m
    predicted_flange_matrix = load_es68_flange_t_tcp().inverse().matrix.copy()
    predicted_flange_matrix[0, 3] += camera_offset_m
    predicted_tcp = PoseSE3(
        "base",
        "flange",
        predicted_flange_matrix,
    ).compose(load_es68_flange_t_tcp())
    state = RobotState(
        100,
        1.0,
        joints,
        base_t_tcp or predicted_tcp,
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        view_id,
        sequence_index,
        state,
        state,
        state,
        stereo,
        None,
        metrics or CaptureMetrics(0, 0, 0, 0, 0),
    )


def _stereo(
    bundle: SynchronizedFrameBundle,
    *,
    confidence: float | None = 1.0,
    metadata: dict[str, object] | None = None,
) -> StereoInferenceObservation:
    rectified = StereoRectifier(
        bundle.stereo.calibration,
        StereoRectificationConfig(),
    ).rectify(bundle.stereo)
    confidence_array = (
        None
        if confidence is None
        else np.full((10, 10), confidence, dtype=np.float32)
    )
    result = StereoResult(
        np.full((10, 10), 2.0, dtype=np.float32),
        np.ones((10, 10), dtype=np.bool_),
        confidence_array,
        dict(FOUNDATION_METADATA if metadata is None else metadata),
    )
    return StereoInferenceObservation(
        bundle.view_id,
        bundle.sequence_index,
        rectified,
        result,
        np.full((10, 10), 0.5, dtype=np.float32),
    )


def _config() -> OccupancyConfig:
    return OccupancyConfig(
        enabled=True,
        voxel_size_m=0.01,
        workspace_bounds_min_m=(-0.3, -0.3, 0.0),
        workspace_bounds_max_m=(0.3, 0.3, 0.8),
        integration_stride=1,
        minimum_valid_depth_fraction=0.5,
    )


def _hand_eye(tmp_path: Path) -> HandEyeCalibration:
    source = tmp_path / "hand_eye.yaml"
    source.write_text("unit-test\n", encoding="utf-8")
    tcp_t_left_ir = PoseSE3.identity("tcp", "left_ir")
    return HandEyeCalibration(
        tcp_t_left_ir,
        "unit-test",
        20,
        0.001,
        0.2,
        source,
        flange_t_left_ir=load_es68_flange_t_tcp().compose(tcp_t_left_ir),
    )


def _integrate_views(
    tmp_path: Path,
    count: int,
    *,
    config: OccupancyConfig | None = None,
    first_capture: datetime | None = None,
):
    settings = config or _config()
    snapshot = None
    updates = []
    start = first_capture or datetime(2026, 8, 28, 6, 0, tzinfo=UTC)
    for index in range(count):
        bundle = _bundle(
            view_id=f"view-{index + 7:03d}",
            sequence_index=index + 7,
            camera_offset_m=0.03 * index,
        )
        update = integrate_foundation_stereo_occupancy(
            snapshot,
            bundle,
            _stereo(bundle),
            _hand_eye(tmp_path),
            settings,
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=start + timedelta(milliseconds=index),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )
        updates.append(update)
        snapshot = update.snapshot
    return updates


def test_one_view_is_evidence_bound_but_not_ready(tmp_path: Path) -> None:
    update = _integrate_views(tmp_path, 1)[0]

    assert update.snapshot.map_state is OccupancyMapState.MAPPING
    assert update.snapshot.source_view_ids == (update.evidence.physical_source_id,)
    assert update.evidence.source_view_id == "view-007"
    assert update.snapshot.quality_evidence_hash == update.evidence.quality_evidence_hash
    assert update.snapshot.state_at_point((0.0125, 0.0125, 0.5)) is OccupancyState.OCCUPIED
    assert update.evidence.robot_model_hash == "a" * 64
    assert update.evidence.fk_tcp_translation_error_m == pytest.approx(0.0)
    assert update.evidence.fk_tcp_rotation_error_deg == pytest.approx(0.0)
    np.testing.assert_allclose(
        update.evidence.base_t_flange_matrix,
        load_es68_flange_t_tcp().inverse().matrix,
    )
    np.testing.assert_allclose(
        update.evidence.predicted_base_t_tcp_matrix,
        np.eye(4),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        update.evidence.observed_base_t_tcp_matrix,
        np.eye(4),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        update.evidence.base_t_camera_matrix,
        np.eye(4),
        atol=1e-12,
    )
    assert update.evidence.valid_depth_fraction == 1.0
    assert update.mapping_snapshot.quality_evidence_hash is None
    assert update.mapping_context.to_payload()["robot"]["joint_zero_offsets_rad"] == [
        0.0
    ] * 6


def test_same_logical_view_can_be_retried_with_a_distinct_physical_capture(
    tmp_path: Path,
) -> None:
    settings = _config()
    hand_eye = _hand_eye(tmp_path)
    start = datetime(2026, 8, 28, 6, 0, tzinfo=UTC)
    first_bundle = _bundle(
        view_id="front-retry",
        sequence_index=7,
        camera_offset_m=0.0,
    )
    first = integrate_foundation_stereo_occupancy(
        None,
        first_bundle,
        _stereo(first_bundle),
        hand_eye,
        settings,
        AcquisitionConfig(),
        EmptyRobotRenderer(),
        captured_at_utc=start,
        source_stereo_metadata_sha256=SOURCE_SHA256,
        source_session_manifest_sha256=SOURCE_SHA256,
        source_session_view_metadata_sha256=SOURCE_SHA256,
    )
    second_bundle = _bundle(
        view_id="front-retry",
        sequence_index=7,
        camera_offset_m=0.03,
    )
    second = integrate_foundation_stereo_occupancy(
        first.snapshot,
        second_bundle,
        _stereo(second_bundle),
        hand_eye,
        settings,
        AcquisitionConfig(),
        EmptyRobotRenderer(),
        captured_at_utc=start + timedelta(milliseconds=1),
        source_stereo_metadata_sha256="e" * 64,
        source_session_manifest_sha256="e" * 64,
        source_session_view_metadata_sha256="e" * 64,
        previous_evidence_hash=first.evidence.quality_evidence_hash,
    )

    assert first.evidence.source_view_id == second.evidence.source_view_id == "front-retry"
    assert first.evidence.physical_source_id != second.evidence.physical_source_id
    assert second.snapshot.source_view_ids == (
        first.evidence.physical_source_id,
        second.evidence.physical_source_id,
    )


def test_same_physical_observation_is_rejected_even_if_reintegrated(
    tmp_path: Path,
) -> None:
    bundle = _bundle(view_id="front", sequence_index=7, camera_offset_m=0.0)
    settings = _config()
    hand_eye = _hand_eye(tmp_path)
    captured = datetime(2026, 8, 28, 6, 0, tzinfo=UTC)
    first = integrate_foundation_stereo_occupancy(
        None,
        bundle,
        _stereo(bundle),
        hand_eye,
        settings,
        AcquisitionConfig(),
        EmptyRobotRenderer(),
        captured_at_utc=captured,
        source_stereo_metadata_sha256=SOURCE_SHA256,
        source_session_manifest_sha256=SOURCE_SHA256,
        source_session_view_metadata_sha256=SOURCE_SHA256,
    )

    with pytest.raises(ValueError, match="already integrated"):
        integrate_foundation_stereo_occupancy(
            first.snapshot,
            bundle,
            _stereo(bundle),
            hand_eye,
            settings,
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=captured + timedelta(milliseconds=1),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
            previous_evidence_hash=first.evidence.quality_evidence_hash,
        )


def test_physical_source_identity_binds_all_raw_identity_fields() -> None:
    baseline = occupancy_physical_source_id(
        source_session_manifest_sha256="a" * 64,
        source_session_view_metadata_sha256="b" * 64,
        source_sequence_index=2,
        frame_number=3,
        source_view_id="front",
    )
    variants = (
        occupancy_physical_source_id(
            source_session_manifest_sha256="c" * 64,
            source_session_view_metadata_sha256="b" * 64,
            source_sequence_index=2,
            frame_number=3,
            source_view_id="front",
        ),
        occupancy_physical_source_id(
            source_session_manifest_sha256="a" * 64,
            source_session_view_metadata_sha256="c" * 64,
            source_sequence_index=2,
            frame_number=3,
            source_view_id="front",
        ),
        occupancy_physical_source_id(
            source_session_manifest_sha256="a" * 64,
            source_session_view_metadata_sha256="b" * 64,
            source_sequence_index=4,
            frame_number=3,
            source_view_id="front",
        ),
        occupancy_physical_source_id(
            source_session_manifest_sha256="a" * 64,
            source_session_view_metadata_sha256="b" * 64,
            source_sequence_index=2,
            frame_number=4,
            source_view_id="front",
        ),
        occupancy_physical_source_id(
            source_session_manifest_sha256="a" * 64,
            source_session_view_metadata_sha256="b" * 64,
            source_sequence_index=2,
            frame_number=3,
            source_view_id="back",
        ),
    )

    assert len(baseline) == 64
    assert baseline not in variants
    assert len(set(variants)) == len(variants)


def test_three_views_are_required_before_map_ready(tmp_path: Path) -> None:
    updates = _integrate_views(tmp_path, 3)

    assert [update.snapshot.map_state for update in updates] == [
        OccupancyMapState.MAPPING,
        OccupancyMapState.MAPPING,
        OccupancyMapState.MAP_READY,
    ]
    assert updates[-1].snapshot.source_view_ids == tuple(
        update.evidence.physical_source_id for update in updates
    )
    assert updates[1].evidence.previous_evidence_hash == updates[0].evidence.quality_evidence_hash
    assert updates[2].evidence.previous_evidence_hash == updates[1].evidence.quality_evidence_hash


def test_unsettled_frame_cannot_update_safety_map(tmp_path: Path) -> None:
    bundle = _bundle(CaptureMetrics(0, 0.02, 0, 0, 0))

    with pytest.raises(OccupancyMappingError, match="joint motion"):
        integrate_foundation_stereo_occupancy(
            None,
            bundle,
            _stereo(bundle),
            _hand_eye(tmp_path),
            _config(),
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=datetime.now(UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )


def test_expired_ready_map_materializes_stale_version(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    config = _config().model_copy(update={"maximum_map_age_s": 5.0})
    update = _integrate_views(
        tmp_path,
        3,
        config=config,
        first_capture=now - timedelta(seconds=10),
    )[-1]

    stale = mark_snapshot_stale_if_expired(
        update.snapshot,
        config,
        now_utc=now,
    )

    assert stale.map_state is OccupancyMapState.STALE
    assert stale.sequence == update.snapshot.sequence + 1
    assert stale.content_hash != update.snapshot.content_hash


def test_external_previous_hash_must_match_snapshot_chain(tmp_path: Path) -> None:
    first = _integrate_views(tmp_path, 1)[0]
    second_bundle = _bundle(
        view_id="view-008",
        sequence_index=8,
        camera_offset_m=0.03,
    )

    with pytest.raises(OccupancyMappingError, match="not bound"):
        integrate_foundation_stereo_occupancy(
            first.snapshot,
            second_bundle,
            _stereo(second_bundle),
            _hand_eye(tmp_path),
            _config(),
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=datetime(2026, 8, 28, 6, 0, 1, tzinfo=UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
            previous_evidence_hash="f" * 64,
        )


def test_mapping_context_is_locked_across_frames(tmp_path: Path) -> None:
    first = _integrate_views(tmp_path, 1)[0]
    second_bundle = _bundle(view_id="view-008", sequence_index=8)

    with pytest.raises(ValueError, match="mapping context"):
        integrate_foundation_stereo_occupancy(
            first.snapshot,
            second_bundle,
            _stereo(second_bundle),
            _hand_eye(tmp_path),
            _config(),
            AcquisitionConfig(),
            DifferentRobotRenderer(),
            captured_at_utc=datetime(2026, 8, 28, 6, 0, 1, tzinfo=UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )


def test_fk_must_agree_with_observed_controller_tcp(tmp_path: Path) -> None:
    bundle = _bundle()

    with pytest.raises(OccupancyMappingError, match="FK and observed controller TCP"):
        integrate_foundation_stereo_occupancy(
            None,
            bundle,
            _stereo(bundle),
            _hand_eye(tmp_path),
            _config(),
            AcquisitionConfig(),
            InconsistentFkRenderer(),
            captured_at_utc=datetime.now(UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )


def test_mapping_pose_uses_fk_flange_not_validation_only_controller_tcp(
    tmp_path: Path,
) -> None:
    observed_tcp = PoseSE3.from_rotation_translation(
        "base",
        "tcp",
        np.eye(3),
        (0.001, 0.0, 0.0),
    )
    bundle = _bundle(base_t_tcp=observed_tcp)

    update = integrate_foundation_stereo_occupancy(
        None,
        bundle,
        _stereo(bundle),
        _hand_eye(tmp_path),
        _config(),
        AcquisitionConfig(),
        EmptyRobotRenderer(),
        captured_at_utc=datetime.now(UTC),
        source_stereo_metadata_sha256=SOURCE_SHA256,
        source_session_manifest_sha256=SOURCE_SHA256,
        source_session_view_metadata_sha256=SOURCE_SHA256,
    )

    assert update.evidence.fk_tcp_translation_error_m == pytest.approx(0.001)
    np.testing.assert_allclose(update.evidence.observed_base_t_tcp_matrix, observed_tcp.matrix)
    np.testing.assert_allclose(update.evidence.base_t_camera_matrix, np.eye(4), atol=1e-12)


def test_legacy_tcp_primary_hand_eye_is_rejected_for_safety_mapping(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    source = tmp_path / "legacy_hand_eye.yaml"
    source.write_text("unit-test\n", encoding="utf-8")
    legacy = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "legacy-unit-test",
        20,
        0.001,
        0.2,
        source,
    )

    with pytest.raises(OccupancyMappingError, match="flange-primary"):
        integrate_foundation_stereo_occupancy(
            None,
            bundle,
            _stereo(bundle),
            legacy,
            _config(),
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=datetime.now(UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )


@pytest.mark.parametrize(
    ("confidence", "metadata", "message"),
    [
        (None, FOUNDATION_METADATA, "left-right confidence"),
        (
            1.0,
            {**FOUNDATION_METADATA, "left_right_consistency_applied": False},
            "consistency was not applied",
        ),
        (
            1.0,
            {**FOUNDATION_METADATA, "left_right_consistency_threshold_px": 2.0},
            "exceeds the occupancy quality contract",
        ),
        (0.2, FOUNDATION_METADATA, "valid depth fraction"),
    ],
)
def test_foundation_stereo_quality_gate_fails_closed(
    tmp_path: Path,
    confidence: float | None,
    metadata: dict[str, object],
    message: str,
) -> None:
    bundle = _bundle()

    with pytest.raises(OccupancyMappingError, match=message):
        integrate_foundation_stereo_occupancy(
            None,
            bundle,
            _stereo(bundle, confidence=confidence, metadata=metadata),
            _hand_eye(tmp_path),
            _config(),
            AcquisitionConfig(),
            EmptyRobotRenderer(),
            captured_at_utc=datetime.now(UTC),
            source_stereo_metadata_sha256=SOURCE_SHA256,
            source_session_manifest_sha256=SOURCE_SHA256,
            source_session_view_metadata_sha256=SOURCE_SHA256,
        )
