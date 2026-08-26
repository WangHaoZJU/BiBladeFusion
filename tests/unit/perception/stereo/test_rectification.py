import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import StereoRectificationConfig
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.conversions import rotation_vector_to_matrix
from biblade_fusion.perception.stereo import (
    RectifiedStereoCalibration,
    StereoRectifier,
    StereoResult,
    constrain_to_rectified_valid_regions,
)


def calibration(rotation=None, translation=(-0.05, 0.0, 0.0)):
    intrinsics = CameraIntrinsics(64, 48, 50.0, 50.0, 31.5, 23.5, "none", ())
    return StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir",
            "left_ir",
            np.eye(3) if rotation is None else rotation,
            translation,
        ),
        None,
    )


def frame(calibration_snapshot):
    image = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64).astype(np.uint8)
    return StereoFrame(100, 7, 1.0, 1.0, image, image, None, calibration_snapshot)


def test_horizontal_pair_rectifies_without_changing_geometry() -> None:
    source = calibration()
    rectifier = StereoRectifier(source, StereoRectificationConfig())

    result = rectifier.rectify(frame(source))

    np.testing.assert_array_equal(result.left_ir, result.right_ir)
    np.testing.assert_allclose(
        result.calibration.right_rectified_t_left_rectified.rotation,
        np.eye(3),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.calibration.right_rectified_t_left_rectified.translation_m,
        [-0.05, 0, 0],
        atol=1e-10,
    )
    assert result.calibration.baseline_m == 0.05
    assert result.calibration.left.distortion_model == "none"
    assert result.left_ir.flags.writeable is False


def test_rectified_frame_chain_preserves_original_stereo_transform() -> None:
    source = calibration(
        rotation_vector_to_matrix([0.0, 0.01, 0.0]),
        (-0.05, 0.001, 0.0),
    )
    rectified = StereoRectifier(source, StereoRectificationConfig()).calibration

    reconstructed = (
        rectified.right_rectified_t_right_ir.inverse()
        .compose(rectified.right_rectified_t_left_rectified)
        .compose(rectified.left_rectified_t_left_ir)
    )

    np.testing.assert_allclose(reconstructed.matrix, source.right_t_left.matrix, atol=1e-9)
    np.testing.assert_allclose(
        rectified.right_rectified_t_left_rectified.rotation,
        np.eye(3),
        atol=1e-9,
    )
    translation = rectified.right_rectified_t_left_rectified.translation_m
    assert abs(translation[1]) < 1e-9
    assert abs(translation[2]) < 1e-9


def test_stereo_validity_requires_both_rectified_regions() -> None:
    intrinsics = CameraIntrinsics(5, 3, 50.0, 50.0, 2.0, 1.0, "none", ())
    rectified_calibration = RectifiedStereoCalibration(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_rectified", "left_rectified", np.eye(3), [-0.05, 0, 0]
        ),
        PoseSE3.identity("left_rectified", "left_ir"),
        PoseSE3.identity("right_rectified", "right_ir"),
        np.eye(4),
        (1, 0, 4, 3),
        (1, 1, 3, 2),
    )
    result = StereoResult(
        np.ones((3, 5), dtype=np.float32),
        np.ones((3, 5), dtype=bool),
    )

    constrained = constrain_to_rectified_valid_regions(result, rectified_calibration)

    expected = np.zeros((3, 5), dtype=bool)
    expected[1:, 2:] = True
    np.testing.assert_array_equal(constrained.valid_mask, expected)
    assert constrained.metadata["rectified_valid_regions_applied"] is True
