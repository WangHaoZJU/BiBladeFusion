from dataclasses import replace

import numpy as np
import pytest

from biblade_fusion.acquisition.coordinator import SynchronizedAcquirer
from biblade_fusion.acquisition.errors import AcquisitionRejectedError
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AcquisitionConfig
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.thermal_camera import NullThermalCamera


def make_robot_state(time_ns: int, joint_offset: float = 0.0) -> RobotState:
    return RobotState(
        monotonic_time_ns=time_ns,
        controller_time_s=time_ns / 1e9,
        joint_positions_rad=np.full(6, joint_offset),
        base_t_tcp=PoseSE3.from_rotation_translation(
            "base", "tcp", np.eye(3), [joint_offset, 0, 0]
        ),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.2,
    )


def make_stereo_frame(time_ns: int) -> StereoFrame:
    intrinsics = CameraIntrinsics(4, 3, 100, 100, 2, 1.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        left=intrinsics,
        right=intrinsics,
        right_t_left=PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]
        ),
        native_depth_scale_m=None,
    )
    image = np.zeros((3, 4), dtype=np.uint8)
    return StereoFrame(time_ns, 1, 10.0, 10.1, image, image, None, calibration)


class FakeRobot:
    def __init__(self, states: list[RobotState]) -> None:
        self.states = iter(states)

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read_state(self) -> RobotState:
        return next(self.states)


class FakeStereoCamera:
    def __init__(self, frame: StereoFrame) -> None:
        self.frame = frame

    @property
    def is_open(self) -> bool:
        return True

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def capture(self) -> StereoFrame:
        return self.frame


def test_synchronized_capture_selects_nearest_robot_state() -> None:
    before = make_robot_state(1_000_000_000)
    after = make_robot_state(1_100_000_000)
    stereo = make_stereo_frame(1_080_000_000)
    acquirer = SynchronizedAcquirer(
        FakeRobot([before, after]),
        FakeStereoCamera(stereo),
        NullThermalCamera(),
        AcquisitionConfig(),
    )

    bundle = acquirer.capture("seed", 0)

    assert bundle.selected_robot_state is after
    assert bundle.metrics.bracket_ms == 100.0
    assert bundle.metrics.selected_robot_state_offset_ms == 20.0
    assert bundle.thermal is None


def test_capture_rejects_robot_motion() -> None:
    before = make_robot_state(1_000_000_000)
    after = make_robot_state(1_100_000_000, joint_offset=0.01)
    acquirer = SynchronizedAcquirer(
        FakeRobot([before, after]),
        FakeStereoCamera(make_stereo_frame(1_050_000_000)),
        NullThermalCamera(),
        AcquisitionConfig(),
    )

    with pytest.raises(AcquisitionRejectedError, match="joint motion"):
        acquirer.capture("seed", 0)


def test_capture_rejects_timestamp_outside_bracket() -> None:
    before = make_robot_state(1_000_000_000)
    after = make_robot_state(1_100_000_000)
    stereo = replace(make_stereo_frame(1_050_000_000), monotonic_time_ns=900_000_000)
    acquirer = SynchronizedAcquirer(
        FakeRobot([before, after]),
        FakeStereoCamera(stereo),
        NullThermalCamera(),
        AcquisitionConfig(),
    )

    with pytest.raises(AcquisitionRejectedError, match="outside robot-state bracket"):
        acquirer.capture("seed", 0)


def test_capture_can_require_thermal_observation() -> None:
    before = make_robot_state(1_000_000_000)
    after = make_robot_state(1_100_000_000)
    acquirer = SynchronizedAcquirer(
        FakeRobot([before, after]),
        FakeStereoCamera(make_stereo_frame(1_050_000_000)),
        NullThermalCamera(),
        AcquisitionConfig(),
        require_thermal=True,
    )

    with pytest.raises(AcquisitionRejectedError, match="thermal observation"):
        acquirer.capture("seed", 0)
