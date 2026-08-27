from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot.elite_readonly import EliteReadOnlyRobot
from biblade_fusion.devices.robot.errors import (
    RobotConfigurationError,
    RobotNotConnectedError,
)


class FakeRtsi:
    disconnected = False

    def __init__(self, output_recipe: str, input_recipe: str, frequency: float) -> None:
        assert Path(output_recipe).is_file()
        assert Path(input_recipe).is_file()
        assert frequency == 125.0

    def connect(self, ip: str) -> bool:
        return ip == "192.168.1.10"

    def disconnect(self) -> None:
        self.disconnected = True

    def getControllerVersion(self):
        return SimpleNamespace(major=2, minor=14, bugfix=2, build=1)

    def getActualJointPositions(self):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def getActualTCPPose(self):
        return [0.4, 0.1, 0.5, 0.0, 0.0, 0.0]

    def getTimestamp(self):
        return 12.5

    def getRobotMode(self):
        return SimpleNamespace(name="IDLE")

    def getSafetyStatus(self):
        return SimpleNamespace(name="NORMAL")

    def getActualSpeedScaling(self):
        return 0.25


def make_config(robot_ip: str | None) -> RobotConfig:
    return RobotConfig(
        robot_ip=robot_ip,
        sdk_wheel=Path("/tmp/elite.whl"),
    )


def test_elite_readonly_reads_state_and_disconnects() -> None:
    sdk = SimpleNamespace(RtsiIOInterface=FakeRtsi)
    robot = EliteReadOnlyRobot(make_config("192.168.1.10"), sdk_module=sdk)

    with robot:
        state = robot.read_state()
        assert robot.controller_version() == "2.14.2+1"
        assert state.controller_time_s == 12.5
        assert state.robot_mode == "IDLE"
        assert state.safety_status == "NORMAL"
        assert state.speed_scaling == 0.25
        np.testing.assert_allclose(state.joint_positions_rad, np.arange(6) / 10)
        np.testing.assert_allclose(state.base_t_tcp.translation_m, [0.4, 0.1, 0.5])

    assert robot.is_connected is False


def test_elite_readonly_requires_ip() -> None:
    robot = EliteReadOnlyRobot(make_config(None), sdk_module=SimpleNamespace())

    with pytest.raises(RobotConfigurationError, match="robot.robot_ip"):
        robot.connect()


def test_elite_readonly_rejects_read_before_connect() -> None:
    robot = EliteReadOnlyRobot(make_config("192.168.1.10"), sdk_module=SimpleNamespace())

    with pytest.raises(RobotNotConnectedError):
        robot.read_state()
