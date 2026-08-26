import cv2
import numpy as np
import pytest

from biblade_fusion.calibration import CharucoDetectionError, CharucoTargetDetector
from biblade_fusion.core.settings import CharucoTargetConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics


def configured_target(**overrides) -> CharucoTargetConfig:
    values = {
        "squares_x": 7,
        "squares_y": 5,
        "square_length_m": 0.04,
        "marker_length_m": 0.03,
        "minimum_corners": 12,
        "maximum_reprojection_rmse_px": 1.0,
    }
    values.update(overrides)
    return CharucoTargetConfig.model_validate(values)


def test_detector_recovers_synthetic_perspective_board_pose() -> None:
    intrinsics = CameraIntrinsics(1280, 720, 900.0, 900.0, 639.5, 359.5, "none", ())
    config = configured_target()
    detector = CharucoTargetDetector(config, intrinsics)
    board_image = detector.board.generateImage((700, 500), marginSize=0, borderBits=1)
    rotation_vector = np.array([0.12, -0.16, 0.04], dtype=np.float64)
    translation = np.array([-0.14, -0.10, 0.70], dtype=np.float64)
    board_width = config.squares_x * config.square_length_m
    board_height = config.squares_y * config.square_length_m
    outer_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width, 0.0, 0.0],
            [board_width, board_height, 0.0],
            [0.0, board_height, 0.0],
        ],
        dtype=np.float32,
    )
    camera_matrix = np.array(
        [[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        outer_corners,
        rotation_vector,
        translation,
        camera_matrix,
        np.zeros(5),
    )
    source_corners = np.array(
        [[0.0, 0.0], [699.0, 0.0], [699.0, 499.0], [0.0, 499.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners, projected.reshape(4, 2))
    image = cv2.warpPerspective(
        board_image,
        homography,
        (intrinsics.width, intrinsics.height),
        borderValue=255,
    )

    detection = detector.detect(image)

    assert len(detection.charuco_ids) >= config.minimum_corners
    assert detection.reprojection_rmse_px < config.maximum_reprojection_rmse_px
    np.testing.assert_allclose(detection.left_ir_t_target.translation_m, translation, atol=0.003)
    expected_rotation, _ = cv2.Rodrigues(rotation_vector)
    np.testing.assert_allclose(
        detection.left_ir_t_target.rotation,
        expected_rotation,
        atol=0.01,
    )


def test_detector_requires_measured_board_dimensions() -> None:
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 319.5, 239.5, "none", ())

    with pytest.raises(CharucoDetectionError, match="must match the printed board"):
        CharucoTargetDetector(CharucoTargetConfig(), intrinsics)


def test_detector_rejects_image_without_board() -> None:
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 319.5, 239.5, "none", ())
    detector = CharucoTargetDetector(configured_target(), intrinsics)

    with pytest.raises(CharucoDetectionError, match="corners"):
        detector.detect(np.full((480, 640), 127, dtype=np.uint8))
