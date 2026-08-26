from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.storage import read_reconstructed_view, write_reconstructed_view
from biblade_fusion.workflows import ReconstructedBladeView


def make_view() -> ReconstructedBladeView:
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
    )


def hand_eye(tmp_path: Path) -> HandEyeCalibration:
    return HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "test",
        20,
        0.001,
        0.2,
        tmp_path / "hand_eye.yaml",
    )


def test_reconstructed_view_round_trip_and_checksum(tmp_path: Path) -> None:
    output = write_reconstructed_view(
        tmp_path / "view",
        make_view(),
        np.ones((2, 2), dtype=bool),
        hand_eye(tmp_path),
        PointCloudConfig(minimum_valid_points=3),
        source_session=tmp_path / "session",
    )
    stored = read_reconstructed_view(output)

    assert stored.view.source_view_id == "front_r00_c00"
    assert stored.view.source_sequence_index == 3
    assert stored.view.source_frame_number == 17
    assert stored.view.depth_source == "native_realsense"
    np.testing.assert_allclose(stored.view.base_cloud.points_m, make_view().base_cloud.points_m)

    points = np.load(output / "base_points_m.npy", allow_pickle=False)
    points[0, 0] += 1.0
    np.save(output / "base_points_m.npy", points, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_reconstructed_view(output)


def test_reconstructed_view_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_reconstructed_view(
            output,
            make_view(),
            np.ones((2, 2), dtype=bool),
            hand_eye(tmp_path),
            PointCloudConfig(minimum_valid_points=3),
            source_session=tmp_path / "session",
        )
