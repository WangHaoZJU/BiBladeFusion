from pathlib import Path

import numpy as np
import pytest
import yaml

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    DepthComparisonConfig,
    FoundationStereoConfig,
    PointCloudConfig,
    ProxyModelConfig,
    StereoRectificationConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot import RobotState
from biblade_fusion.perception.stereo import StereoRectifier, StereoResult
from biblade_fusion.storage import (
    SessionReader,
    SessionWriter,
    read_depth_aggregate,
    read_depth_comparison,
    write_depth_aggregate,
    write_depth_aggregate_manifest,
    write_depth_comparison,
    write_initialization,
    write_stereo_inference,
)
from biblade_fusion.workflows import (
    StereoInferenceObservation,
    compare_paired_depth,
    initialize_native_depth,
)


def _bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(20, 20, 100.0, 100.0, 9.5, 9.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]
        ),
        0.001,
        intrinsics,
        PoseSE3.identity("left_ir", "depth"),
    )
    image = np.zeros((20, 20), dtype=np.uint8)
    frame = StereoFrame(
        100,
        7,
        1.0,
        1.0,
        image,
        image,
        np.full((20, 20), 500, dtype=np.uint16),
        calibration,
    )
    state = RobotState(
        100,
        1.0,
        np.zeros(6),
        PoseSE3.identity("base", "tcp"),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        "seed", 0, state, state, state, frame, None, CaptureMetrics(0, 0, 0, 0, 0)
    )


def _sources(tmp_path: Path):
    bundle = _bundle()
    with SessionWriter.create(
        tmp_path / "sessions", load_settings("configs/default.yaml"), label="depth"
    ) as writer:
        writer.write_bundle(bundle)
    rectification = StereoRectificationConfig()
    rectified = StereoRectifier(bundle.stereo.calibration, rectification).rectify(
        bundle.stereo
    )
    result = StereoResult(
        np.full((20, 20), 10.0, dtype=np.float32),
        np.ones((20, 20), dtype=bool),
    )
    stereo_observation = StereoInferenceObservation(
        "seed", 0, rectified, result, result.depth_m(rectified.calibration)
    )
    stereo_path = write_stereo_inference(
        tmp_path / "stereo",
        stereo_observation,
        FoundationStereoConfig(device="cpu"),
        rectification,
        source_session=writer.path,
    )
    mask_path = tmp_path / "blade_mask.npy"
    np.save(mask_path, np.ones((20, 20), dtype=bool), allow_pickle=False)
    point_config = PointCloudConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_points=100,
    )
    comparison_config = DepthComparisonConfig(minimum_overlap_points=100)
    comparison = compare_paired_depth(
        bundle,
        stereo_observation,
        np.ones((20, 20), dtype=bool),
        point_config,
        comparison_config,
    )
    return (
        writer.path,
        stereo_path,
        mask_path,
        comparison,
        point_config,
        comparison_config,
    )


def test_depth_comparison_artifact_round_trip_and_checksum(tmp_path: Path) -> None:
    session, stereo, mask, comparison, point_config, comparison_config = _sources(
        tmp_path
    )
    output = write_depth_comparison(
        tmp_path / "comparison",
        comparison,
        point_config,
        comparison_config,
        source_session=session,
        source_stereo_inference=stereo,
        source_blade_mask=mask,
    )

    stored = read_depth_comparison(output)

    assert stored.comparison.metrics.overlap_pixel_count == 400
    assert stored.comparison.metrics.mean_absolute_error_m == pytest.approx(0.0)
    assert "not ground truth" in stored.metadata["interpretation"]

    np.save(
        output / "signed_error_m.npy",
        np.ones((20, 20), dtype=np.float32),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_depth_comparison(output)


def test_depth_aggregate_manifest_round_trip(tmp_path: Path) -> None:
    session, stereo, mask, comparison, point_config, comparison_config = _sources(
        tmp_path
    )
    comparison_path = write_depth_comparison(
        tmp_path / "comparison",
        comparison,
        point_config,
        comparison_config,
        source_session=session,
        source_stereo_inference=stereo,
        source_blade_mask=mask,
    )
    manifest = tmp_path / "aggregate.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "incidence_bin_edges_deg: [0, 15, 30, 90]\n"
        "comparisons:\n"
        f"  - artifact: {comparison_path}\n"
        "    side: front\n"
        "    incidence_angle_deg: 5\n",
        encoding="utf-8",
    )
    output = write_depth_aggregate(tmp_path / "aggregate", manifest)

    stored = read_depth_aggregate(output)

    groups = {group.group_id for group in stored.report.groups}
    assert groups == {"all", "side:front", "incidence:[0,15]deg"}
    assert "weight views equally" in stored.metadata["interpretation"]


def test_depth_manifest_labels_achieved_pose_from_initialization(tmp_path: Path) -> None:
    session, stereo, mask, comparison, point_config, comparison_config = _sources(
        tmp_path
    )
    comparison_path = write_depth_comparison(
        tmp_path / "comparison",
        comparison,
        point_config,
        comparison_config,
        source_session=session,
        source_stereo_inference=stereo,
        source_blade_mask=mask,
    )
    bundle = SessionReader(session).load_bundle("seed")
    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "test",
        20,
        0.001,
        0.2,
        tmp_path / "hand_eye.yaml",
    )
    proxy_config = ProxyModelConfig(
        voxel_size_m=0.0001,
        minimum_points=100,
        estimated_thickness_m=0.01,
    )
    observation = initialize_native_depth(
        bundle,
        np.ones((20, 20), dtype=bool),
        hand_eye,
        point_config,
        proxy_config,
    )
    initialization = write_initialization(
        tmp_path / "initialization",
        observation,
        np.ones((20, 20), dtype=bool),
        hand_eye,
        point_config,
        proxy_config,
        source_session=session,
    )

    manifest = write_depth_aggregate_manifest(
        tmp_path / "generated.yaml",
        (comparison_path,),
        initialization,
        comparison_config,
    )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    assert payload["labeling"]["method"] == "achieved_pose_against_fixed_proxy_v1"
    assert payload["comparisons"][0]["side"] == "front"
    assert payload["comparisons"][0]["incidence_angle_deg"] == pytest.approx(0.0)
    aggregate = write_depth_aggregate(tmp_path / "generated-aggregate", manifest)
    assert read_depth_aggregate(aggregate).report.groups[0].metrics.view_count == 1
