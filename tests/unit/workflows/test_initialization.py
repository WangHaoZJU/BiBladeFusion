from pathlib import Path

import numpy as np

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig, ProxyModelConfig
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.workflows import initialize_native_depth


def make_bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(20, 20, 100.0, 100.0, 9.5, 9.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        0.001,
        intrinsics,
        PoseSE3.identity("left_ir", "depth"),
    )
    image = np.zeros((20, 20), dtype=np.uint8)
    stereo = StereoFrame(
        100,
        1,
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
        PoseSE3.from_rotation_translation("base", "tcp", np.eye(3), [1, 0, 0]),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        "seed",
        0,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0, 0, 0, 0, 0),
    )


def test_native_depth_initialization_reaches_base_frame_proxy() -> None:
    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "unit-test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
    )
    point_config = PointCloudConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_points=100,
    )
    proxy_config = ProxyModelConfig(
        voxel_size_m=0.0001,
        minimum_points=100,
        estimated_thickness_m=0.01,
    )

    result = initialize_native_depth(
        make_bundle(),
        np.ones((20, 20), dtype=bool),
        hand_eye,
        point_config,
        proxy_config,
    )

    assert result.base_cloud.frame == "base"
    assert result.left_intrinsics.width == 20
    np.testing.assert_allclose(result.seed_joint_positions_rad, np.zeros(6))
    assert result.base_cloud.points_m.shape == (400, 3)
    np.testing.assert_allclose(result.base_t_left_ir.translation_m, [1, 0, 0])
    np.testing.assert_allclose(result.base_cloud.points_m[:, 2], 0.5)
    assert result.proxy.frame_T_proxy.parent_frame == "base"
    assert result.proxy.contains(result.base_cloud.points_m).all()
