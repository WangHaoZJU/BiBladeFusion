import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig, ProxyModelConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.storage import read_initialization, write_initialization
from biblade_fusion.workflows import InitialObservation


def make_observation() -> InitialObservation:
    points = np.array([[0, 0, 0.5], [0.1, 0, 0.5], [0, 0.1, 0.5]])
    pixels = np.array([[0, 0], [1, 0], [0, 1]])
    cloud = PointCloud("base", points, pixels, (2, 2))
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.ones(3),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        3,
        3,
        3,
        1.0,
    )
    return InitialObservation(
        "seed",
        CameraIntrinsics(2, 2, 100, 100, 0.5, 0.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        cloud,
        proxy,
    )


def test_initialization_artifact_round_trip(tmp_path: Path) -> None:
    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "test",
        20,
        0.001,
        0.2,
        tmp_path / "he.yaml",
    )
    point_config = PointCloudConfig(minimum_valid_points=3)
    proxy_config = ProxyModelConfig(estimated_thickness_m=0.01)
    mask = np.array([[True, True], [True, False]])
    output = tmp_path / "initialization"

    write_initialization(
        output,
        make_observation(),
        mask,
        hand_eye,
        point_config,
        proxy_config,
        source_session=tmp_path / "session",
    )
    stored = read_initialization(output)

    assert stored.observation.source_view_id == "seed"
    assert stored.observation.left_intrinsics.fx == 100
    np.testing.assert_allclose(stored.observation.seed_joint_positions_rad, np.zeros(6))
    np.testing.assert_allclose(stored.hand_eye.tcp_t_left_ir.matrix, np.eye(4))
    np.testing.assert_allclose(
        stored.observation.base_cloud.points_m,
        make_observation().base_cloud.points_m,
    )
    np.testing.assert_array_equal(stored.blade_mask, mask)
    assert stored.metadata["processing"]["proxy_model"]["estimated_thickness_m"] == 0.01


def test_initialization_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "test",
        20,
        0.001,
        0.2,
        tmp_path / "he.yaml",
    )

    with pytest.raises(FileExistsError):
        write_initialization(
            output,
            make_observation(),
            np.ones((2, 2), dtype=bool),
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            ProxyModelConfig(estimated_thickness_m=0.01),
            source_session=tmp_path / "session",
        )


def test_initialization_reader_rejects_path_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "files": {
                    "base_points_m": "../outside.npy",
                    "pixel_uv": "pixels.npy",
                    "blade_mask": "mask.npy",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        read_initialization(artifact)
