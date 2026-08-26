from types import SimpleNamespace

import numpy as np

from biblade_fusion.core.settings import RealSenseConfig
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


def test_realsense_capture_returns_calibrated_stereo_bundle() -> None:
    fake_rs = FakeRs()
    config = RealSenseConfig(
        infrared_width=4,
        infrared_height=3,
        warmup_frames=0,
        enable_native_depth=True,
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
        np.testing.assert_allclose(
            frame.calibration.right_t_left.rotation,
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        )

    assert camera.is_open is False
    assert fake_rs.pipeline_instance.stopped is True
