from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    DepthComparisonConfig,
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
    StereoRectificationConfig,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot import RobotState
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.perception.stereo import StereoRectifier, StereoResult
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.workflows import (
    DepthComparisonError,
    StereoInferenceObservation,
    classify_depth_view_geometry,
    compare_paired_depth,
)


def make_bundle() -> SynchronizedFrameBundle:
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
        PoseSE3.identity("base", "tcp"),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        "seed", 0, state, state, state, stereo, None, CaptureMetrics(0, 0, 0, 0, 0)
    )


def test_paired_depth_comparison_uses_shared_rectified_pixels() -> None:
    bundle = make_bundle()
    rectified = StereoRectifier(
        bundle.stereo.calibration, StereoRectificationConfig()
    ).rectify(bundle.stereo)
    result = StereoResult(
        np.full((20, 20), 10.0, dtype=np.float32),
        np.ones((20, 20), dtype=bool),
    )
    observation = StereoInferenceObservation(
        "seed", 0, rectified, result, result.depth_m(rectified.calibration)
    )

    comparison = compare_paired_depth(
        bundle,
        observation,
        np.ones((20, 20), dtype=bool),
        PointCloudConfig(
            minimum_depth_m=0.1,
            maximum_depth_m=1.0,
            minimum_valid_points=100,
        ),
        DepthComparisonConfig(minimum_overlap_points=100),
    )

    assert comparison.metrics.overlap_pixel_count == 400
    assert comparison.metrics.mean_absolute_error_m == pytest.approx(0.0)
    assert comparison.metrics.median_stereo_to_native_ratio == pytest.approx(1.0)
    assert comparison.metrics.agreement_fractions == (
        (0.005, 1.0),
        (0.01, 1.0),
        (0.02, 1.0),
    )


def test_paired_depth_comparison_rejects_mismatched_source() -> None:
    bundle = make_bundle()
    rectified = StereoRectifier(
        bundle.stereo.calibration, StereoRectificationConfig()
    ).rectify(bundle.stereo)
    result = StereoResult(
        np.full((20, 20), 10.0, dtype=np.float32),
        np.ones((20, 20), dtype=bool),
    )
    observation = StereoInferenceObservation(
        "another-view", 0, rectified, result, result.depth_m(rectified.calibration)
    )

    with pytest.raises(DepthComparisonError, match="does not match"):
        compare_paired_depth(
            bundle,
            observation,
            np.ones((20, 20), dtype=bool),
            PointCloudConfig(minimum_valid_points=100),
            DepthComparisonConfig(minimum_overlap_points=100),
        )


def test_depth_view_geometry_uses_achieved_pose_and_proxy_normal() -> None:
    bundle = make_bundle()
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    desired_base_t_left_ir = PoseSE3.from_rotation_translation(
        "base", "left_ir", np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.2]
    )
    state = replace(
        bundle.selected_robot_state,
        base_t_tcp=base_t_flange.compose(load_es68_flange_t_tcp()),
    )
    bundle = replace(
        bundle,
        robot_state_before=state,
        robot_state_after=state,
        selected_robot_state=state,
    )
    rectified = StereoRectifier(
        bundle.stereo.calibration, StereoRectificationConfig()
    ).rectify(bundle.stereo)
    result = StereoResult(
        np.full((20, 20), 10.0, dtype=np.float32),
        np.ones((20, 20), dtype=bool),
    )
    observation = StereoInferenceObservation(
        "seed", 0, rectified, result, result.depth_m(rectified.calibration)
    )
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.4, 0.2, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )
    flange_t_left_ir = base_t_flange.inverse().compose(desired_base_t_left_ir)
    hand_eye = HandEyeCalibration(
        load_es68_flange_t_tcp().inverse().compose(flange_t_left_ir),
        "test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
        flange_t_left_ir=flange_t_left_ir,
    )

    geometry = classify_depth_view_geometry(
        bundle,
        observation,
        proxy,
        hand_eye,
        0.02,
        kinematics_config=KinematicsConfig(),
        hand_eye_config=HandEyeConfig(),
    )

    assert geometry.side.value == "front"
    assert geometry.camera_side_offset_m == pytest.approx(0.2)
    assert geometry.incidence_angle_deg == pytest.approx(0.0)
