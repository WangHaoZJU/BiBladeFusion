from pathlib import Path

import numpy as np

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    PointCloudConfig,
    ProxyModelConfig,
    StereoRectificationConfig,
)
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.stereo import StereoRectifier, StereoResult
from biblade_fusion.workflows import (
    StereoInferenceObservation,
    initialize_foundation_stereo_depth,
    initialize_native_depth,
)


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
    assert result.planning_intrinsics.width == 20
    np.testing.assert_allclose(result.seed_joint_positions_rad, np.zeros(6))
    assert result.base_cloud.points_m.shape == (400, 3)
    np.testing.assert_allclose(result.base_t_left_ir.translation_m, [1, 0, 0])
    np.testing.assert_allclose(result.base_cloud.points_m[:, 2], 0.5)
    assert result.proxy.frame_T_proxy.parent_frame == "base"
    assert result.proxy.contains(result.base_cloud.points_m).all()


def test_foundation_stereo_initialization_uses_rectified_camera_frame() -> None:
    bundle = make_bundle()
    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "unit-test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
    )
    rectified = StereoRectifier(
        bundle.stereo.calibration,
        StereoRectificationConfig(),
    ).rectify(bundle.stereo)
    result = StereoResult(
        np.full((20, 20), 10.0, dtype=np.float32),
        np.ones((20, 20), dtype=bool),
    )
    stereo_observation = StereoInferenceObservation(
        "seed",
        0,
        rectified,
        result,
        result.depth_m(rectified.calibration),
    )

    observation = initialize_foundation_stereo_depth(
        bundle,
        stereo_observation,
        np.ones((20, 20), dtype=bool),
        hand_eye,
        PointCloudConfig(
            minimum_depth_m=0.1,
            maximum_depth_m=1.0,
            minimum_valid_points=100,
        ),
        ProxyModelConfig(
            voxel_size_m=0.0001,
            minimum_points=100,
            estimated_thickness_m=0.01,
        ),
    )

    assert observation.depth_source == "foundation_stereo"
    assert observation.base_t_projection_camera.child_frame == "left_rectified"
    assert observation.planning_intrinsics.distortion_model == "none"
    np.testing.assert_allclose(observation.base_cloud.points_m[:, 2], 0.5)
    np.testing.assert_allclose(observation.base_t_left_ir.translation_m, [1, 0, 0])
