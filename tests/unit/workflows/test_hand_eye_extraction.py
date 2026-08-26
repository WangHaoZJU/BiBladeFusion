import json
from pathlib import Path

import cv2
import numpy as np

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import (
    CharucoTargetDetector,
    read_hand_eye_samples,
    write_hand_eye_samples,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CharucoTargetConfig
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.workflows import extract_hand_eye_samples


def target_config() -> CharucoTargetConfig:
    return CharucoTargetConfig(
        square_length_m=0.04,
        marker_length_m=0.03,
        minimum_corners=12,
        maximum_reprojection_rmse_px=1.0,
    )


def make_bundle(view_id: str, image: np.ndarray, sequence_index: int) -> SynchronizedFrameBundle:
    height, width = image.shape
    intrinsics = CameraIntrinsics(width, height, 900.0, 900.0, width / 2, height / 2, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        None,
    )
    stereo = StereoFrame(
        100 + sequence_index,
        sequence_index,
        1.0,
        1.0,
        image,
        image,
        None,
        calibration,
    )
    state = RobotState(
        100 + sequence_index,
        1.0,
        np.zeros(6),
        PoseSE3.from_rotation_translation("base", "tcp", np.eye(3), [0.1, 0.2, 0.3]),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        view_id,
        sequence_index,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0, 0, 0, 0, 0),
    )


def test_extracts_valid_views_and_records_rejections(tmp_path: Path) -> None:
    config = target_config()
    intrinsics = CameraIntrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0, "none", ())
    detector = CharucoTargetDetector(config, intrinsics)
    board = detector.board.generateImage((700, 500), marginSize=0, borderBits=1)
    object_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [config.squares_x * config.square_length_m, 0.0, 0.0],
            [
                config.squares_x * config.square_length_m,
                config.squares_y * config.square_length_m,
                0.0,
            ],
            [0.0, config.squares_y * config.square_length_m, 0.0],
        ],
        dtype=np.float32,
    )
    camera_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        object_corners,
        np.array([0.12, -0.16, 0.04]),
        np.array([-0.14, -0.10, 0.70]),
        camera_matrix,
        np.zeros(5),
    )
    source_corners = np.array(
        [[0.0, 0.0], [699.0, 0.0], [699.0, 499.0], [0.0, 499.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners, projected.reshape(4, 2))
    valid_image = cv2.warpPerspective(board, homography, (1280, 720), borderValue=255)
    blank_image = np.full_like(valid_image, 127)

    result = extract_hand_eye_samples(
        (
            (tmp_path / "session-a", make_bundle("good", valid_image, 0)),
            (tmp_path / "session-a", make_bundle("blank", blank_image, 1)),
        ),
        config,
    )

    assert len(result.samples) == 1
    assert len(result.rejected) == 1
    assert result.samples[0].sample_id.endswith(":0000:good")
    assert result.samples[0].charuco_corner_count >= config.minimum_corners
    assert result.rejected[0].sample_id.endswith(":0001:blank")
    destination = write_hand_eye_samples(
        tmp_path / "samples.yaml",
        result.samples,
        result.rejected,
    )
    loaded = read_hand_eye_samples(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded[0].source_session == str((tmp_path / "session-a").resolve())
    assert loaded[0].charuco_corner_count == result.samples[0].charuco_corner_count
    assert len(payload["rejected"]) == 1
