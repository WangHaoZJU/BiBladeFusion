import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)


def make_calibration() -> StereoCalibrationSnapshot:
    intrinsics = CameraIntrinsics(4, 3, 100, 100, 2, 1.5, "none", ())
    return StereoCalibrationSnapshot(
        left=intrinsics,
        right=intrinsics,
        right_t_left=PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]
        ),
        native_depth_scale_m=0.001,
    )


def test_stereo_frame_copies_and_freezes_arrays() -> None:
    left = np.zeros((3, 4), dtype=np.uint8)
    frame = StereoFrame(1, 2, 3.0, 3.1, left, left, None, make_calibration())
    left[0, 0] = 255

    assert frame.left_ir[0, 0] == 0
    assert frame.left_ir.flags.writeable is False
    assert frame.calibration.baseline_m == pytest.approx(0.05)


def test_stereo_frame_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        StereoFrame(
            1,
            2,
            3.0,
            3.1,
            np.zeros((3, 4), dtype=np.uint8),
            np.zeros((2, 4), dtype=np.uint8),
            None,
            make_calibration(),
        )

