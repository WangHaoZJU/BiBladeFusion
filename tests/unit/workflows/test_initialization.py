from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
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
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.workflows import (
    ReconstructionError,
    StereoInferenceObservation,
    initialize_foundation_stereo_depth,
    initialize_native_depth,
    reconstruct_native_depth_view,
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
    base_t_tcp = Es68KinematicModel.from_resources().base_t_flange(
        np.zeros(6)
    ).compose(load_es68_flange_t_tcp())
    state = RobotState(
        100,
        1.0,
        np.zeros(6),
        base_t_tcp,
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


def make_hand_eye() -> HandEyeCalibration:
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    desired = PoseSE3.from_rotation_translation(
        "base", "left_ir", np.eye(3), [1.0, 0.0, 0.0]
    )
    flange_t_left_ir = base_t_flange.inverse().compose(desired)
    return HandEyeCalibration(
        load_es68_flange_t_tcp().inverse().compose(flange_t_left_ir),
        "unit-test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
        flange_t_left_ir=flange_t_left_ir,
    )
def test_native_depth_initialization_reaches_base_frame_proxy() -> None:
    hand_eye = make_hand_eye()
    point_config = PointCloudConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_points=100,
    )
    proxy_config = ProxyModelConfig(
        voxel_size_m=0.0001,
        minimum_points=100,
        estimated_thickness_m=0.01,
        blade_envelope_min_m=(0.97, -0.06, 0.49),
        blade_envelope_max_m=(1.03, 0.06, 0.51),
        minimum_envelope_retained_fraction=0.5,
    )

    result = initialize_native_depth(
        make_bundle(),
        np.ones((20, 20), dtype=bool),
        hand_eye,
        point_config,
        proxy_config,
        kinematics_config=KinematicsConfig(),
        hand_eye_config=HandEyeConfig(),
    )

    assert result.base_cloud.frame == "base"
    assert result.planning_intrinsics.width == 20
    np.testing.assert_allclose(result.seed_joint_positions_rad, np.zeros(6))
    assert result.base_cloud.points_m.shape == (400, 3)
    np.testing.assert_allclose(result.base_t_left_ir.translation_m, [1, 0, 0])
    np.testing.assert_allclose(result.base_cloud.points_m[:, 2], 0.5)
    assert result.proxy.frame_T_proxy.parent_frame == "base"
    assert result.proxy.raw_point_count == 240
    assert np.count_nonzero(result.proxy_support_mask) == 240
    assert result.proxy.contains(result.proxy_support_points_m).all()


def test_native_depth_initialization_rejects_low_envelope_retention() -> None:
    with pytest.raises(ValueError, match="at least 75.000%"):
        initialize_native_depth(
            make_bundle(),
            np.ones((20, 20), dtype=bool),
            make_hand_eye(),
            PointCloudConfig(
                minimum_depth_m=0.1,
                maximum_depth_m=1.0,
                minimum_valid_points=100,
            ),
            ProxyModelConfig(
                voxel_size_m=0.0001,
                minimum_points=100,
                estimated_thickness_m=0.01,
                blade_envelope_min_m=(0.97, -0.06, 0.49),
                blade_envelope_max_m=(1.03, 0.06, 0.51),
                minimum_envelope_retained_fraction=0.75,
            ),
            kinematics_config=KinematicsConfig(),
            hand_eye_config=HandEyeConfig(),
        )


def test_foundation_stereo_initialization_uses_rectified_camera_frame() -> None:
    bundle = make_bundle()
    hand_eye = make_hand_eye()
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
        kinematics_config=KinematicsConfig(),
        hand_eye_config=HandEyeConfig(),
    )

    assert observation.depth_source == "foundation_stereo"
    assert observation.base_t_projection_camera.child_frame == "left_rectified"
    assert observation.planning_intrinsics.distortion_model == "none"
    np.testing.assert_allclose(observation.base_cloud.points_m[:, 2], 0.5)
    np.testing.assert_allclose(observation.base_t_left_ir.translation_m, [1, 0, 0])


def test_reconstruction_rejects_legacy_tcp_only_hand_eye() -> None:
    legacy = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "legacy",
        20,
        0.001,
        0.2,
        Path("legacy.yaml"),
    )

    with pytest.raises(ReconstructionError, match="flange-primary"):
        initialize_native_depth(
            make_bundle(),
            np.ones((20, 20), dtype=bool),
            legacy,
            PointCloudConfig(minimum_valid_points=100),
            ProxyModelConfig(
                voxel_size_m=0.0001,
                minimum_points=100,
                estimated_thickness_m=0.01,
            ),
            kinematics_config=KinematicsConfig(),
            hand_eye_config=HandEyeConfig(),
        )


def test_reconstruction_rejects_controller_tcp_outside_fk_gate() -> None:
    bundle = make_bundle()
    matrix = bundle.selected_robot_state.base_t_tcp.matrix.copy()
    matrix[0, 3] += 0.003
    state = replace(
        bundle.selected_robot_state,
        base_t_tcp=PoseSE3("base", "tcp", matrix),
    )
    mismatched = replace(
        bundle,
        robot_state_before=state,
        robot_state_after=state,
        selected_robot_state=state,
    )

    with pytest.raises(ReconstructionError, match="translation"):
        initialize_native_depth(
            mismatched,
            np.ones((20, 20), dtype=bool),
            make_hand_eye(),
            PointCloudConfig(minimum_valid_points=100),
            ProxyModelConfig(
                voxel_size_m=0.0001,
                minimum_points=100,
                estimated_thickness_m=0.01,
            ),
            kinematics_config=KinematicsConfig(),
            hand_eye_config=HandEyeConfig(),
        )


def test_reconstruction_uses_explicit_nonzero_joint_offsets() -> None:
    offsets = (0.01, -0.02, 0.015, 0.0, 0.0, 0.0)
    kinematics = KinematicsConfig(joint_zero_offsets_rad=offsets)
    bundle = make_bundle()
    base_t_flange = Es68KinematicModel.from_resources(
        joint_zero_offsets_rad=offsets
    ).base_t_flange(np.zeros(6))
    state = replace(
        bundle.selected_robot_state,
        base_t_tcp=base_t_flange.compose(load_es68_flange_t_tcp()),
    )
    consistent = replace(
        bundle,
        robot_state_before=state,
        robot_state_after=state,
        selected_robot_state=state,
    )

    result = reconstruct_native_depth_view(
        consistent,
        np.ones((20, 20), dtype=bool),
        make_hand_eye(),
        PointCloudConfig(minimum_valid_points=100),
        kinematics_config=kinematics,
        hand_eye_config=HandEyeConfig(),
    )

    assert result.pose_authority is not None
    assert result.pose_authority.joint_zero_offsets_rad == offsets
    np.testing.assert_allclose(
        result.base_t_left_ir.matrix,
        base_t_flange.compose(make_hand_eye().require_flange_primary()).matrix,
        atol=1e-12,
    )
