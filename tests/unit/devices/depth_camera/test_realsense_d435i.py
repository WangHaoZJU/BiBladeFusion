from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from biblade_fusion.core.settings import RealSenseConfig
from biblade_fusion.devices.depth_camera.errors import DepthCameraConnectionError
from biblade_fusion.devices.depth_camera.realsense_d435i import RealSenseD435i


class FakeFrame:
    def __init__(self, data, frame_number=7, timestamp=12.5) -> None:
        self._data = data
        self._frame_number = frame_number
        self._timestamp = timestamp

    def __bool__(self) -> bool:
        return True

    def get_data(self):
        return self._data

    def get_frame_number(self):
        return self._frame_number

    def get_timestamp(self):
        return self._timestamp


class FakeFrames:
    def __init__(self) -> None:
        self.left = FakeFrame(np.ones((3, 4), dtype=np.uint8))
        self.right = FakeFrame(np.full((3, 4), 2, dtype=np.uint8), timestamp=12.6)
        self.depth = FakeFrame(np.full((3, 4), 1000, dtype=np.uint16))

    def get_infrared_frame(self, index: int):
        return self.left if index == 1 else self.right

    def get_depth_frame(self):
        return self.depth


class FakeVideoProfile:
    def __init__(self, index: int | str) -> None:
        self.index = index

    def as_video_stream_profile(self):
        return self

    def width(self):
        return 4

    def height(self):
        return 3

    def get_intrinsics(self):
        return SimpleNamespace(
            width=4,
            height=3,
            fx=100.0,
            fy=101.0,
            ppx=2.0,
            ppy=1.5,
            model="none",
            coeffs=[0, 0, 0, 0, 0],
        )

    def get_extrinsics_to(self, target):
        if self.index == "depth":
            assert target.index == 1
            return SimpleNamespace(
                rotation=np.eye(3).reshape(-1, order="F").tolist(),
                translation=[0.001, 0.0, 0.0],
            )
        assert self.index == 1 and target.index == 2
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        return SimpleNamespace(
            rotation=rotation.reshape(-1, order="F").tolist(),
            translation=[-0.05, 0.0, 0.0],
        )


class FakePipelineProfile:
    def get_stream(self, stream, index: int | None = None):
        if stream == "depth":
            return FakeVideoProfile("depth")
        assert stream == "infrared" and index is not None
        return FakeVideoProfile(index)

    def get_device(self):
        sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
        return SimpleNamespace(first_depth_sensor=lambda: sensor)


class FakePipeline:
    def __init__(self) -> None:
        self.stopped = False

    def start(self, config):
        assert len(config.enabled) == 3
        return FakePipelineProfile()

    def wait_for_frames(self, timeout_ms: int):
        assert timeout_ms == 5000
        return FakeFrames()

    def stop(self) -> None:
        self.stopped = True


class FakeConfig:
    def __init__(self) -> None:
        self.enabled = []

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args) -> None:
        self.enabled.append(args)


class FakeRs:
    stream = SimpleNamespace(infrared="infrared", depth="depth")
    format = SimpleNamespace(y8="y8", z16="z16")

    def __init__(self) -> None:
        self.pipeline_instance = FakePipeline()

    def pipeline(self):
        return self.pipeline_instance

    def config(self):
        return FakeConfig()


def _write_user_stereo_calibration(path: Path) -> Path:
    intrinsics = {
        "width": 4,
        "height": 3,
        "camera_matrix": [[100.0, 0.0, 2.0], [0.0, 101.0, 1.5], [0.0, 0.0, 1.0]],
        "opencv_distortion_model": "brown_conrady",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    payload = {
        "calibration_type": "d435i_ir_stereo_charuco",
        "factory_intrinsics_used": False,
        "left_ir": intrinsics,
        "right_ir": intrinsics,
        "right_ir_T_left_ir": [
            [1.0, 0.0, 0.0, -0.05],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_realsense_capture_returns_user_calibrated_stereo_bundle(tmp_path: Path) -> None:
    fake_rs = FakeRs()
    config = RealSenseConfig(
        infrared_width=4,
        infrared_height=3,
        warmup_frames=0,
        enable_native_depth=True,
        stereo_calibration_path=_write_user_stereo_calibration(tmp_path / "stereo.yaml"),
    )
    camera = RealSenseD435i(config, rs_module=fake_rs)

    with camera:
        frame = camera.capture()
        assert frame.frame_number == 7
        assert frame.left_ir.shape == (3, 4)
        assert frame.right_ir[0, 0] == 2
        assert frame.native_depth[0, 0] == 1000
        assert frame.calibration.native_depth_scale_m == 0.001
        assert frame.calibration.depth is not None
        assert frame.calibration.left_t_depth is not None
        np.testing.assert_allclose(frame.calibration.left_t_depth.translation_m, [0.001, 0, 0])
        assert frame.calibration.baseline_m == 0.05
        np.testing.assert_allclose(frame.calibration.right_t_left.rotation, np.eye(3))

    assert camera.is_open is False
    assert fake_rs.pipeline_instance.stopped is True


def test_realsense_refuses_factory_ir_calibration_fallback() -> None:
    fake_rs = FakeRs()
    config = RealSenseConfig(
        infrared_width=4,
        infrared_height=3,
        warmup_frames=0,
        enable_native_depth=True,
        stereo_calibration_path=None,
    )
    camera = RealSenseD435i(config, rs_module=fake_rs)

    with pytest.raises(DepthCameraConnectionError, match="factory IR"):
        camera.open()
    assert fake_rs.pipeline_instance.stopped is True
