import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig, KinematicsConfig, PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import read_reconstructed_view, write_reconstructed_view
from biblade_fusion.workflows import AuthoritativeRobotPose, ReconstructedBladeView


def make_view(
    pose_authority: AuthoritativeRobotPose | None = None,
) -> ReconstructedBladeView:
    cloud = PointCloud(
        "base",
        np.array([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5], [0.0, 0.01, 0.5]]),
        np.array([[0, 0], [1, 0], [0, 1]]),
        (2, 2),
    )
    return ReconstructedBladeView(
        "front_r00_c00",
        3,
        17,
        CameraIntrinsics(2, 2, 100.0, 100.0, 0.5, 0.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        cloud,
        "native_realsense",
        pose_authority,
    )


def authoritative_inputs(
    tmp_path: Path,
) -> tuple[HandEyeCalibration, AuthoritativeRobotPose]:
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    flange_t_tcp = load_es68_flange_t_tcp()
    flange_t_left_ir = base_t_flange.inverse().compose(
        PoseSE3.identity("base", "left_ir")
    )
    source = tmp_path / "hand_eye.yaml"
    tcp_t_left_ir = flange_t_tcp.inverse().compose(flange_t_left_ir)
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "test",
                "matrix": flange_t_left_ir.matrix.tolist(),
                "derived_runtime": {
                    "tcp_T_left_ir": tcp_t_left_ir.matrix.tolist(),
                },
                "quality": {
                    "sample_count": 20,
                    "translation_rmse_m": 0.001,
                    "rotation_rmse_deg": 0.2,
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
        "test",
        20,
        0.001,
        0.2,
        source,
        flange_t_left_ir=flange_t_left_ir,
    )
    authority = AuthoritativeRobotPose(
        base_t_flange,
        base_t_flange.compose(flange_t_tcp),
        base_t_flange.compose(flange_t_tcp),
        0.0,
        0.0,
        0.002,
        0.3,
        (0.0,) * 6,
    )
    return hand_eye, authority


def test_reconstructed_view_round_trip_and_checksum(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_reconstructed_view(
        tmp_path / "view",
        make_view(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    stored = read_reconstructed_view(output)

    assert stored.view.source_view_id == "front_r00_c00"
    assert stored.view.source_sequence_index == 3
    assert stored.view.source_frame_number == 17
    assert stored.view.depth_source == "native_realsense"
    assert stored.view.pose_authority is not None
    np.testing.assert_allclose(stored.view.base_cloud.points_m, make_view().base_cloud.points_m)

    points = np.load(output / "base_points_m.npy", allow_pickle=False)
    points[0, 0] += 1.0
    np.save(output / "base_points_m.npy", points, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_reconstructed_view(output)


def test_authoritative_pose_rejects_tcp_not_derived_from_packaged_flange(
    tmp_path: Path,
) -> None:
    _, authority = authoritative_inputs(tmp_path)
    tampered = PoseSE3.from_rotation_translation(
        "base",
        "tcp",
        authority.predicted_base_t_tcp.rotation,
        authority.predicted_base_t_tcp.translation_m + np.array([0.001, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="packaged flange_T_tcp"):
        AuthoritativeRobotPose(
            authority.base_t_flange,
            tampered,
            tampered,
            0.0,
            0.0,
            0.002,
            0.3,
            authority.joint_zero_offsets_rad,
        )


def test_reconstructed_view_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    hand_eye, authority = authoritative_inputs(tmp_path)

    with pytest.raises(FileExistsError):
        write_reconstructed_view(
            output,
            make_view(authority),
            np.ones((2, 2), dtype=bool),
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=tmp_path / "session",
        )


def test_reconstructed_view_reader_rejects_tampered_projection_chain(
    tmp_path: Path,
) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_reconstructed_view(
        tmp_path / "view",
        make_view(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["transforms"]["base_T_projection_camera"][0][3] += 0.001
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="projection transform is inconsistent"):
        read_reconstructed_view(output)
