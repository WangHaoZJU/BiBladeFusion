"""HoloRobot-aligned EliteArm tests with an in-memory SDK double."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot import EliteArm, ServoJStream, ServoJStreamConfig
from biblade_fusion.devices.robot.conversions import rpy_xyz_to_matrix
from biblade_fusion.devices.robot.errors import (
    RobotCommandError,
    RobotHardwareFaultError,
    RobotMotionDisabledError,
)


class FakeDashboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def connect(self, ip: str, port: int) -> bool:
        self.calls.append(("connect", (ip, port)))
        return True

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))

    def powerOn(self) -> bool:
        self.calls.append(("powerOn", None))
        return True

    def brakeRelease(self) -> bool:
        self.calls.append(("brakeRelease", None))
        return True

    def powerOff(self) -> bool:
        self.calls.append(("powerOff", None))
        return True

    def setSpeedScaling(self, percent: int) -> bool:
        self.calls.append(("setSpeedScaling", percent))
        return True

    def playProgram(self) -> bool:
        self.calls.append(("playProgram", None))
        return True


class FakeRtsi:
    def __init__(
        self,
        *,
        output_recipe: list[str],
        input_recipe: list[str],
        frequency: float,
    ) -> None:
        self.output_recipe = output_recipe
        self.input_recipe = input_recipe
        self.frequency = frequency
        self.safety_status = 1
        self.disconnected = False

    def connect(self, ip: str) -> bool:
        return ip == "192.0.2.68"

    def disconnect(self) -> None:
        self.disconnected = True

    def getActualJointPositions(self) -> list[float]:
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def getActualTCPPose(self) -> list[float]:
        return [0.4, 0.1, 0.5, 0.3, -0.4, 0.5]

    def getTimestamp(self) -> float:
        return 12.5

    def getRobotMode(self) -> int:
        return 5

    def getSafetyStatus(self) -> int:
        return self.safety_status

    def getActualSpeedScaling(self) -> float:
        return 0.3


class FakeDriverConfig:
    pass


class FakeDriver:
    def __init__(self, config: FakeDriverConfig) -> None:
        self.config = config
        self.connected = True
        self.calls: list[tuple[str, object]] = []
        self.callback = None
        self.idle_result = True
        self.speedj_result = True
        self.servoj_result = True

    def isRobotConnected(self) -> bool:
        return self.connected

    def sendExternalControlScript(self) -> bool:
        self.calls.append(("sendExternalControlScript", None))
        self.connected = True
        return True

    def setTrajectoryResultCallback(self, callback) -> None:
        self.callback = callback
        self.calls.append(("setTrajectoryResultCallback", None))

    def writeTrajectoryControlAction(self, action, count: int, timeout_ms: int) -> bool:
        self.calls.append(("writeTrajectoryControlAction", (action, count, timeout_ms)))
        return True

    def writeTrajectoryPoint(
        self,
        position: list[float],
        trajectory_time_s: float,
        blend_radius_m: float,
        cartesian: bool,
    ) -> bool:
        self.calls.append(
            (
                "writeTrajectoryPoint",
                (position, trajectory_time_s, blend_radius_m, cartesian),
            )
        )
        self.callback(0)
        return True

    def writeIdle(self, mode: int) -> bool:
        self.calls.append(("writeIdle", mode))
        return self.idle_result

    def writeSpeedj(self, command: list[float], timeout_ms: int) -> bool:
        self.calls.append(("writeSpeedj", (command, timeout_ms)))
        return self.speedj_result

    def writeServoj(self, command: list[float], timeout_ms: int, *flags: bool) -> bool:
        self.calls.append(("writeServoj", (command, timeout_ms, flags)))
        return self.servoj_result

    def stopControl(self) -> None:
        self.calls.append(("stopControl", None))


class FakeSdk:
    class TrajectoryControlAction:
        START = "start"
        NOOP = "noop"

    def __init__(self) -> None:
        self.dashboard = FakeDashboard()
        self.rtsi: FakeRtsi | None = None
        self.driver: FakeDriver | None = None
        self.__file__ = __file__

    def DashboardClientInterface(self) -> FakeDashboard:
        return self.dashboard

    def RtsiIOInterface(self, **kwargs) -> FakeRtsi:
        self.rtsi = FakeRtsi(**kwargs)
        return self.rtsi

    def EliteDriverConfig(self) -> FakeDriverConfig:
        return FakeDriverConfig()

    def EliteDriver(self, config: FakeDriverConfig) -> FakeDriver:
        self.driver = FakeDriver(config)
        return self.driver


def config(*, motion_enabled: bool) -> RobotConfig:
    return RobotConfig(
        robot_ip="192.0.2.68",
        motion_enabled=motion_enabled,
        script_file_path=Path(__file__),
        sdk_wheel=Path("/tmp/elite.whl"),
    )


def connected_arm() -> tuple[EliteArm, FakeSdk]:
    sdk = FakeSdk()
    arm = EliteArm(config(motion_enabled=True), sdk_module=sdk, sleep_fn=lambda _: None)
    arm.connect()
    return arm, sdk


def enabled_arm() -> tuple[EliteArm, FakeSdk]:
    arm, sdk = connected_arm()
    arm.enable()
    return arm, sdk


def test_motion_driver_is_blocked_by_default_off_gate() -> None:
    arm = EliteArm(config(motion_enabled=False), sdk_module=FakeSdk())

    with pytest.raises(RobotMotionDisabledError, match="motion_enabled"):
        arm.connect()


def test_read_only_connection_does_not_construct_driver() -> None:
    sdk = FakeSdk()
    arm = EliteArm(config(motion_enabled=False), sdk_module=sdk)

    arm.connect(with_driver=False)

    assert arm.is_connected is True
    assert arm.has_motion_driver is False
    assert sdk.rtsi is not None
    assert sdk.rtsi.input_recipe == []


def test_connect_copies_holorobot_driver_configuration() -> None:
    arm, sdk = connected_arm()

    assert arm.has_motion_driver is True
    assert sdk.driver is not None
    assert sdk.driver.config.robot_ip == "192.0.2.68"
    assert sdk.driver.config.reverse_port == 50002
    assert sdk.driver.config.servoj_time == 0.004
    assert sdk.driver.config.servoj_lookahead_time == 0.1
    assert sdk.driver.config.servoj_gain == 2000


def test_enable_uses_holorobot_dashboard_sequence() -> None:
    arm, sdk = connected_arm()

    arm.enable()

    assert arm.is_enabled is True
    assert ("powerOn", None) in sdk.dashboard.calls
    assert ("brakeRelease", None) in sdk.dashboard.calls
    assert ("setSpeedScaling", 30) in sdk.dashboard.calls


def test_read_state_uses_holorobot_rpy_tcp_convention() -> None:
    arm, _ = connected_arm()

    state = arm.read_state()

    np.testing.assert_allclose(state.base_t_tcp.rotation, rpy_xyz_to_matrix([0.3, -0.4, 0.5]))
    np.testing.assert_allclose(state.joint_positions_rad, np.arange(6) / 10)


def test_move_joints_executes_holorobot_single_point_trajectory() -> None:
    arm, sdk = enabled_arm()
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

    arm.move_joints(target)

    point = next(value for name, value in sdk.driver.calls if name == "writeTrajectoryPoint")
    assert point == (target, 3.0, 0.0, False)
    names = [name for name, _ in sdk.driver.calls]
    assert names.index("writeTrajectoryControlAction") < names.index("writeTrajectoryPoint")
    assert names.index("writeTrajectoryPoint") < names.index("writeIdle")


def test_move_tcp_writes_xyz_rpy_cartesian_point() -> None:
    arm, sdk = enabled_arm()
    target = PoseSE3.from_rotation_translation(
        "base",
        "tcp",
        rpy_xyz_to_matrix([0.3, -0.4, 0.5]),
        [0.4, 0.1, 0.5],
    )

    arm.move_tcp(target)

    point = next(value for name, value in sdk.driver.calls if name == "writeTrajectoryPoint")
    np.testing.assert_allclose(point[0], [0.4, 0.1, 0.5, 0.3, -0.4, 0.5])
    assert point[3] is True


def test_unsafe_controller_state_fails_closed_before_motion() -> None:
    arm, sdk = enabled_arm()
    sdk.rtsi.safety_status = 3

    with pytest.raises(RobotHardwareFaultError, match="forbids motion"):
        arm.move_joints([0.0] * 6)

    assert not any(name == "writeTrajectoryPoint" for name, _ in sdk.driver.calls)


def test_stop_rejects_false_write_idle_result() -> None:
    arm, sdk = enabled_arm()
    sdk.driver.idle_result = False

    with pytest.raises(RobotCommandError, match="rejected"):
        arm.stop()


def test_write_joint_velocity_uses_holorobot_speedj_boundary() -> None:
    arm, sdk = enabled_arm()
    command = [0.1, 0.2, 0.0, -0.1, 0.0, 0.05]

    arm.write_joint_velocity(command, timeout_ms=240)

    assert ("writeSpeedj", (command, 240)) in sdk.driver.calls


def test_rejected_speedj_marks_arm_stopped() -> None:
    arm, sdk = enabled_arm()
    sdk.driver.speedj_result = False

    with pytest.raises(RobotCommandError, match="writeSpeedj rejected"):
        arm.write_joint_velocity([0.0] * 6, timeout_ms=240)

    with pytest.raises(RobotCommandError, match="stopped"):
        arm.write_joint_velocity([0.0] * 6, timeout_ms=240)


def test_stop_joint_velocity_sends_zero_without_write_idle() -> None:
    arm, sdk = enabled_arm()

    arm.stop_joint_velocity(timeout_ms=250)

    assert ("writeSpeedj", ([0.0] * 6, 250)) in sdk.driver.calls
    assert not any(name == "writeIdle" for name, _ in sdk.driver.calls)


def test_prepare_servoj_stream_primes_with_long_hold_timeout() -> None:
    arm, sdk = enabled_arm()

    arm.prepare_servoj_stream(dt_s=0.004)

    write = next(value for name, value in sdk.driver.calls if name == "writeServoj")
    assert write[1] == 3000


def test_stream_servoj_writes_every_command_with_short_timeout() -> None:
    arm, sdk = enabled_arm()
    stream = ServoJStream(
        commands=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.02, 0.1, 0.2, 0.3, 0.4, 0.5),
        ),
        dt_s=0.004,
    )

    result = arm.stream_servoj(
        stream,
        config=ServoJStreamConfig(
            dt_s=0.004,
            tracking_check_every_n_commands=99,
        ),
    )

    writes = [value for name, value in sdk.driver.calls if name == "writeServoj"]
    assert result.ok is True
    assert result.commands_sent == 3
    assert [value[1] for value in writes] == [500, 500, 500]
    assert result.timing_summary["planned_dt_s"] == 0.004


def test_stream_servoj_aborts_on_tracking_error() -> None:
    arm, _ = enabled_arm()
    stream = ServoJStream(
        commands=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.2, 0.1, 0.2, 0.3, 0.4, 0.5),
        ),
        dt_s=0.004,
    )

    result = arm.stream_servoj(
        stream,
        config=ServoJStreamConfig(
            dt_s=0.004,
            tracking_error_rad=0.01,
            tracking_check_every_n_commands=1,
            max_consecutive_tracking_violations=1,
        ),
    )

    assert result.ok is False
    assert result.abort_reason == "tracking_error_exceeded"


def test_stream_servoj_rejected_write_fails_closed() -> None:
    arm, sdk = enabled_arm()
    sdk.driver.servoj_result = False
    stream = ServoJStream(commands=((0.0, 0.1, 0.2, 0.3, 0.4, 0.5),), dt_s=0.004)

    result = arm.stream_servoj(stream, config=ServoJStreamConfig(dt_s=0.004))

    assert result.ok is False
    assert result.abort_reason == "driver_write_failed"
    assert result.commands_sent == 0
