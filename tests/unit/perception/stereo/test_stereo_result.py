import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
)
from biblade_fusion.perception.stereo.base import StereoResult, disparity_to_depth_m


def make_calibration() -> StereoCalibrationSnapshot:
    intrinsics = CameraIntrinsics(3, 2, 100.0, 100.0, 1.5, 1.0, "none", ())
    return StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        None,
    )


def test_disparity_to_depth_uses_metric_baseline() -> None:
    disparity = np.array([[10.0, 5.0, 0.0], [np.inf, -1.0, 2.5]], dtype=np.float32)

    depth = disparity_to_depth_m(disparity, make_calibration())

    np.testing.assert_allclose(depth[0, :2], [0.5, 1.0])
    assert np.isnan(depth[0, 2])
    assert np.isnan(depth[1, 0])
    assert np.isnan(depth[1, 1])
    assert depth[1, 2] == 2.0


def test_stereo_result_applies_valid_mask() -> None:
    result = StereoResult(
        disparity_px=np.full((2, 3), 10.0, dtype=np.float32),
        valid_mask=np.array([[True, False, True], [True, True, True]]),
    )

    depth = result.depth_m(make_calibration())

    assert depth[0, 0] == 0.5
    assert np.isnan(depth[0, 1])
