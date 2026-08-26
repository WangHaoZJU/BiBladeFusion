from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    FoundationStereoConfig,
    StereoRectificationConfig,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.stereo import StereoResult
from biblade_fusion.storage import read_stereo_inference, write_stereo_inference
from biblade_fusion.workflows import infer_rectified_stereo


class ConstantStereoBackend:
    def infer(self, left_rectified, right_rectified):
        assert left_rectified.shape == right_rectified.shape
        return StereoResult(
            np.full(left_rectified.shape, 5.0, dtype=np.float32),
            np.ones(left_rectified.shape, dtype=bool),
            metadata={"runtime": "constant-test"},
        )


def make_bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(20, 16, 50.0, 50.0, 9.5, 7.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        None,
    )
    image = np.arange(20 * 16, dtype=np.uint16).reshape(16, 20).astype(np.uint8)
    stereo = StereoFrame(1234, 7, 1.0, 1.0, image, image, None, calibration)
    state = RobotState(
        1200,
        1.0,
        np.zeros(6),
        PoseSE3.identity("base", "tcp"),
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


def test_stereo_workflow_and_artifact_round_trip(tmp_path: Path) -> None:
    rectification_config = StereoRectificationConfig()
    observation = infer_rectified_stereo(
        make_bundle(),
        ConstantStereoBackend(),
        rectification_config,
    )

    assert observation.result.valid_mask[:, :5].sum() == 0
    np.testing.assert_allclose(
        observation.depth_m[observation.result.valid_mask],
        0.5,
    )
    output = write_stereo_inference(
        tmp_path / "stereo",
        observation,
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=tmp_path / "session",
    )
    stored = read_stereo_inference(output)

    np.testing.assert_array_equal(
        stored.observation.result.disparity_px,
        observation.result.disparity_px,
    )
    np.testing.assert_array_equal(
        stored.observation.result.valid_mask,
        observation.result.valid_mask,
    )
    np.testing.assert_allclose(
        stored.observation.depth_m,
        observation.depth_m,
        equal_nan=True,
    )
    assert stored.metadata["files"]["depth_m"]["sha256"]
    assert stored.metadata["source"]["view_id"] == "seed"


def test_stereo_artifact_detects_array_tampering(tmp_path: Path) -> None:
    rectification_config = StereoRectificationConfig()
    observation = infer_rectified_stereo(
        make_bundle(),
        ConstantStereoBackend(),
        rectification_config,
    )
    output = write_stereo_inference(
        tmp_path / "stereo",
        observation,
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=tmp_path / "session",
    )
    np.save(output / "disparity_px.npy", np.zeros((16, 20), dtype=np.float32))

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_stereo_inference(output)
