"""HoloRobot-aligned EliteArm tests with an in-memory SDK double."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot import EliteArm, ServoJStream, ServoJStreamConfig
from biblade_fusion.devices.robot._motion_capability import (
    _GUARDED_MOTION_CAPABILITY,
)
from biblade_fusion.devices.robot.conversions import rpy_xyz_to_matrix
from biblade_fusion.devices.robot.errors import (
    RobotCommandError,
    RobotHardwareFaultError,
    RobotMotionDisabledError,
    RobotMotionInterruptedError,
    RobotNotEnabledError,
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

    def stopProgram(self) -> bool:
        self.calls.append(("stopProgram", None))
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
        self.timestamp_getter_entered: threading.Event | None = None
        self.timestamp_getter_release: threading.Event | None = None
        self._getter_counter_lock = threading.Lock()
        self._active_getters = 0
        self.maximum_concurrent_getters = 0

    def _enter_getter(self) -> None:
        with self._getter_counter_lock:
            self._active_getters += 1
            self.maximum_concurrent_getters = max(
                self.maximum_concurrent_getters,
                self._active_getters,
            )

    def _leave_getter(self) -> None:
        with self._getter_counter_lock:
            self._active_getters -= 1

    def connect(self, ip: str) -> bool:
        return ip == "192.0.2.68"

    def disconnect(self) -> None:
        self.disconnected = True

    def getActualJointPositions(self) -> list[float]:
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def getActualJointVelocity(self) -> list[float]:
        return [0.0] * 6

    def getTargetJointVelocity(self) -> list[float]:
        return [0.0] * 6

    def getActualTCPPose(self) -> list[float]:
        return [0.4, 0.1, 0.5, 0.3, -0.4, 0.5]

    def getActualTCPVelocity(self) -> list[float]:
        return [0.0] * 6

    def getTargetTCPVelocity(self) -> list[float]:
        return [0.0] * 6

    def getTimestamp(self) -> float:
        self._enter_getter()
        try:
            if self.timestamp_getter_entered is not None:
                self.timestamp_getter_entered.set()
            if self.timestamp_getter_release is not None and not self.timestamp_getter_release.wait(
                timeout=2.0
            ):
                raise TimeoutError("test timestamp getter was not released")
            return 12.5
        finally:
            self._leave_getter()

    def getRobotMode(self) -> int:
        return 5

    def getSafetyStatus(self) -> int:
        self._enter_getter()
        try:
            return self.safety_status
        finally:
            self._leave_getter()

    def getRuntimeState(self) -> int:
        return 3

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
        self.servoj_write_entered: threading.Event | None = None
        self.servoj_write_release: threading.Event | None = None

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
        if self.servoj_write_entered is not None:
            self.servoj_write_entered.set()
        if self.servoj_write_release is not None and not self.servoj_write_release.wait(
            timeout=2.0
        ):
            raise TimeoutError("test ServoJ write was not released")
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
    np.testing.assert_allclose(state.actual_joint_velocity_rad_s, np.zeros(6))
    np.testing.assert_allclose(state.target_joint_velocity_rad_s, np.zeros(6))
    np.testing.assert_allclose(state.actual_tcp_velocity, np.zeros(6))
    np.testing.assert_allclose(state.target_tcp_velocity, np.zeros(6))
    assert state.runtime_state == "3"


def test_bootstrap_controller_stop_is_dashboard_only_and_latches_generation() -> None:
    sdk = FakeSdk()
    arm = EliteArm(config(motion_enabled=True), sdk_module=sdk, sleep_fn=lambda _: None)
    arm.connect(with_driver=False)

    generation = arm.establish_bootstrap_controller_stop()

    assert generation == 1
    assert arm.stop_snapshot == (generation, True)
    assert arm.has_motion_driver is False
    assert ("stopProgram", None) in sdk.dashboard.calls


def test_bootstrap_controller_stop_accepts_already_stopped_dashboard_task() -> None:
    class AlreadyStoppedDashboard(FakeDashboard):
        def runningStatus(self) -> str:
            self.calls.append(("runningStatus", None))
            return "STOPPED"

        def stopProgram(self) -> bool:
            raise AssertionError("already-stopped task must not receive stopProgram")

    sdk = FakeSdk()
    sdk.dashboard = AlreadyStoppedDashboard()
    arm = EliteArm(config(motion_enabled=True), sdk_module=sdk, sleep_fn=lambda _: None)
    arm.connect(with_driver=False)

    generation = arm.establish_bootstrap_controller_stop()

    assert generation == 1
    assert arm.stop_snapshot == (generation, True)
    assert sdk.dashboard.calls[-1] == ("runningStatus", None)


def test_bootstrap_controller_stop_stops_and_rechecks_running_dashboard_task() -> None:
    class RunningDashboard(FakeDashboard):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = iter(("PLAYING", "STOPPED"))

        def runningStatus(self) -> str:
            self.calls.append(("runningStatus", None))
            return next(self.statuses)

    sdk = FakeSdk()
    sdk.dashboard = RunningDashboard()
    arm = EliteArm(config(motion_enabled=True), sdk_module=sdk, sleep_fn=lambda _: None)
    arm.connect(with_driver=False)

    arm.establish_bootstrap_controller_stop()

    assert sdk.dashboard.calls[-3:] == [
        ("runningStatus", None),
        ("stopProgram", None),
        ("runningStatus", None),
    ]


def test_bootstrap_controller_stop_rejects_failed_or_unconfirmed_dashboard_stop() -> None:
    class UnstoppableDashboard(FakeDashboard):
        def __init__(self, *, accepted: bool) -> None:
            super().__init__()
            self.accepted = accepted

        def runningStatus(self) -> str:
            self.calls.append(("runningStatus", None))
            return "PLAYING"

        def stopProgram(self) -> bool:
            self.calls.append(("stopProgram", None))
            return self.accepted

    for accepted, message in (
        (False, "rejected bootstrap controller stop"),
        (True, "did not reach STOPPED"),
    ):
        sdk = FakeSdk()
        sdk.dashboard = UnstoppableDashboard(accepted=accepted)
        arm = EliteArm(config(motion_enabled=True), sdk_module=sdk, sleep_fn=lambda _: None)
        arm.connect(with_driver=False)

        with pytest.raises(RobotCommandError, match=message):
            arm.establish_bootstrap_controller_stop()


def test_move_joints_executes_holorobot_single_point_trajectory() -> None:
    arm, sdk = enabled_arm()
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

    arm._guarded_move_joints(target, capability=_GUARDED_MOTION_CAPABILITY)

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

    arm._guarded_move_tcp(target, capability=_GUARDED_MOTION_CAPABILITY)

    point = next(value for name, value in sdk.driver.calls if name == "writeTrajectoryPoint")
    np.testing.assert_allclose(point[0], [0.4, 0.1, 0.5, 0.3, -0.4, 0.5])
    assert point[3] is True


def test_unsafe_controller_state_fails_closed_before_motion() -> None:
    arm, sdk = enabled_arm()
    sdk.rtsi.safety_status = 3

    with pytest.raises(RobotHardwareFaultError, match="forbids motion"):
        arm._guarded_move_joints(
            [0.0] * 6,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    assert not any(name == "writeTrajectoryPoint" for name, _ in sdk.driver.calls)


@pytest.mark.parametrize("safety_status", [0, 4, "not-an-enum", None])
def test_unknown_or_unparseable_safety_state_fails_closed(safety_status) -> None:
    arm, sdk = enabled_arm()
    sdk.rtsi.safety_status = safety_status

    with pytest.raises(RobotHardwareFaultError, match="motion|forbids"):
        arm._guarded_move_joints(
            [0.0] * 6,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    assert not any(name == "writeTrajectoryPoint" for name, _ in sdk.driver.calls)


@pytest.mark.parametrize(
    "config",
    [
        ServoJStreamConfig(dt_s=float("nan")),
        ServoJStreamConfig(warmup_duration_s=float("nan")),
        ServoJStreamConfig(tracking_error_rad=float("nan")),
        ServoJStreamConfig(timing_violation_factor=float("nan")),
    ],
)
def test_servoj_runtime_guards_reject_nonfinite_values(config) -> None:
    with pytest.raises(ValueError, match="finite"):
        config.validate()


def test_servoj_stream_rejects_nonfinite_period() -> None:
    stream = ServoJStream(commands=((0.0,) * 6,), dt_s=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        stream.validate()


def test_stop_rejects_false_write_idle_result() -> None:
    arm, sdk = enabled_arm()
    sdk.driver.idle_result = False

    with pytest.raises(RobotCommandError, match="rejected"):
        arm.stop()
    with pytest.raises(RobotMotionInterruptedError, match="stopped"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )


def test_unpowered_bootstrap_stop_uses_dashboard_without_power_or_reverse_start() -> None:
    arm, sdk = connected_arm()
    sdk.driver.connected = False

    arm.stop()

    assert arm.stop_snapshot[1] is True
    assert ("stopProgram", None) in sdk.dashboard.calls
    assert not any(
        name in {"powerOn", "brakeRelease", "playProgram"} for name, _ in sdk.dashboard.calls
    )
    assert not any(name == "writeIdle" for name, _ in sdk.driver.calls)


def test_public_enable_cannot_clear_stop_latch() -> None:
    arm, _ = enabled_arm()
    arm.stop()

    with pytest.raises(RobotMotionInterruptedError, match="one-shot approval"):
        arm.enable()


def test_guarded_servoj_recovery_requires_private_capability() -> None:
    arm, sdk = enabled_arm()
    arm.stop()

    with pytest.raises(PermissionError, match="guarded-executor capability"):
        arm._guarded_resume_servoj_control(
            expected_stop_generation=arm.stop_generation,
            capability=object(),
        )
    with pytest.raises(RobotMotionInterruptedError, match="stopped"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    assert not any(name == "writeServoj" for name, _ in sdk.driver.calls)


def test_guarded_enable_requires_capability_and_preserves_stop_latch() -> None:
    arm, sdk = connected_arm()
    arm.stop()
    generation = arm.stop_generation

    with pytest.raises(PermissionError, match="guarded-executor capability"):
        arm._guarded_enable_for_servoj_control(
            expected_stop_generation=generation,
            capability=object(),
        )
    assert arm.is_enabled is False

    arm._guarded_enable_for_servoj_control(
        expected_stop_generation=generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert arm.is_enabled is True
    assert arm.stop_generation == generation
    with pytest.raises(RobotMotionInterruptedError, match="stopped"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )
    assert [name for name, _ in sdk.driver.calls] == ["writeIdle"]


def test_guarded_enable_defers_reverse_session_until_approved_resume() -> None:
    arm, sdk = connected_arm()
    arm.stop()
    generation = arm.stop_generation
    sdk.driver.connected = False

    arm._guarded_enable_for_servoj_control(
        expected_stop_generation=generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert not any(name == "sendExternalControlScript" for name, _ in sdk.driver.calls)
    assert arm.stop_snapshot == (generation, True)

    arm._guarded_resume_servoj_control(
        expected_stop_generation=generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert [
        name for name, _ in sdk.driver.calls if name == "sendExternalControlScript"
    ] == ["sendExternalControlScript"]
    assert arm.stop_snapshot == (generation, False)


def test_stop_requests_reverse_idle_and_dashboard_program_stop() -> None:
    arm, sdk = enabled_arm()

    arm.stop()

    assert [name for name, _ in sdk.driver.calls] == ["writeIdle"]
    assert ("stopProgram", None) in sdk.dashboard.calls
    assert arm.stop_snapshot[1] is True


def test_stop_between_power_on_and_brake_release_forces_power_off() -> None:
    arm, sdk = connected_arm()
    arm.stop()
    approved_generation = arm.stop_generation
    original_power_on = sdk.dashboard.powerOn

    def power_on_then_stop() -> bool:
        accepted = original_power_on()
        arm.stop()
        return accepted

    sdk.dashboard.powerOn = power_on_then_stop

    with pytest.raises(RobotMotionInterruptedError, match="after powerOn"):
        arm._guarded_enable_for_servoj_control(
            expected_stop_generation=approved_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    assert arm.is_enabled is False
    assert ("powerOff", None) in sdk.dashboard.calls
    assert not any(name == "writeServoj" for name, _ in sdk.driver.calls)


def test_guarded_servoj_recovery_clears_stop_without_sending_motion() -> None:
    arm, sdk = enabled_arm()
    arm.stop()

    arm._guarded_resume_servoj_control(
        expected_stop_generation=arm.stop_generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert [name for name, _ in sdk.driver.calls] == ["writeIdle"]
    arm._guarded_prepare_servoj_stream(
        dt_s=0.004,
        expected_stop_generation=arm.stop_generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )
    assert [name for name, _ in sdk.driver.calls] == ["writeIdle", "writeServoj"]


def test_guarded_servoj_recovery_never_implicitly_enables_arm() -> None:
    arm, sdk = connected_arm()
    arm.stop()

    with pytest.raises(RobotNotEnabledError, match="must be enabled"):
        arm._guarded_resume_servoj_control(
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    assert not any(name in {"powerOn", "brakeRelease"} for name, _ in sdk.dashboard.calls)


def test_guarded_servoj_recovery_keeps_stop_latched_on_unsafe_state() -> None:
    arm, sdk = enabled_arm()
    arm.stop()
    sdk.rtsi.safety_status = 3

    with pytest.raises(RobotHardwareFaultError, match="forbids motion"):
        arm._guarded_resume_servoj_control(
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    sdk.rtsi.safety_status = 1
    with pytest.raises(RobotMotionInterruptedError, match="stopped"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )


def test_concurrent_stop_generation_wins_over_blocked_servoj_recovery() -> None:
    arm, _ = enabled_arm()
    arm.stop()
    approved_generation = arm.stop_generation
    recovery_entered = threading.Event()
    release_recovery = threading.Event()
    recovery_errors: list[BaseException] = []
    original_reverse_check = arm._ensure_driver_reverse_connected

    def blocked_reverse_check() -> None:
        recovery_entered.set()
        if not release_recovery.wait(timeout=2.0):
            raise TimeoutError("test recovery was not released")
        original_reverse_check()

    arm._ensure_driver_reverse_connected = blocked_reverse_check  # type: ignore[method-assign]

    def recover() -> None:
        try:
            arm._guarded_resume_servoj_control(
                expected_stop_generation=approved_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
        except BaseException as exc:
            recovery_errors.append(exc)

    recovery_thread = threading.Thread(target=recover)
    recovery_thread.start()
    assert recovery_entered.wait(timeout=1.0)

    # stop() must not wait for the recovery thread's long _motion_lock section.
    arm.stop()
    assert arm.stop_generation == approved_generation + 1
    assert recovery_thread.is_alive()

    release_recovery.set()
    recovery_thread.join(timeout=2.0)
    assert not recovery_thread.is_alive()
    assert len(recovery_errors) == 1
    assert isinstance(recovery_errors[0], RobotMotionInterruptedError)
    assert "stop latch changed" in str(recovery_errors[0])
    with pytest.raises(RobotMotionInterruptedError, match="stopped"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=arm.stop_generation,
            capability=_GUARDED_MOTION_CAPABILITY,
        )


def test_rtsi_state_and_motion_safety_getters_are_serialized() -> None:
    arm, sdk = enabled_arm()
    timestamp_entered = threading.Event()
    release_timestamp = threading.Event()
    sdk.rtsi.timestamp_getter_entered = timestamp_entered
    sdk.rtsi.timestamp_getter_release = release_timestamp
    errors: list[BaseException] = []

    def read_full_state() -> None:
        try:
            arm.read_state()
        except BaseException as exc:
            errors.append(exc)

    def read_motion_safety() -> None:
        try:
            arm._raise_for_safety()
        except BaseException as exc:
            errors.append(exc)

    state_thread = threading.Thread(target=read_full_state)
    safety_thread = threading.Thread(target=read_motion_safety)
    state_thread.start()
    assert timestamp_entered.wait(timeout=1.0)
    safety_thread.start()

    # The safety thread is blocked on _state_lock, not inside the vendor getter.
    assert safety_thread.is_alive()
    assert sdk.rtsi.maximum_concurrent_getters == 1

    release_timestamp.set()
    state_thread.join(timeout=2.0)
    safety_thread.join(timeout=2.0)
    assert not state_thread.is_alive()
    assert not safety_thread.is_alive()
    assert errors == []
    assert sdk.rtsi.maximum_concurrent_getters == 1


def test_long_servoj_write_does_not_hold_rtsi_state_lock() -> None:
    arm, sdk = enabled_arm()
    write_entered = threading.Event()
    release_write = threading.Event()
    sdk.driver.servoj_write_entered = write_entered
    sdk.driver.servoj_write_release = release_write
    stream = ServoJStream(
        commands=((0.0, 0.1, 0.2, 0.3, 0.4, 0.5),),
        dt_s=0.004,
    )
    stream_errors: list[BaseException] = []
    state_results: list[object] = []

    def run_stream() -> None:
        try:
            arm._guarded_stream_servoj(
                stream,
                config=ServoJStreamConfig(dt_s=0.004),
                expected_stop_generation=arm.stop_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
        except BaseException as exc:
            stream_errors.append(exc)

    def read_state() -> None:
        state_results.append(arm.read_state())

    stream_thread = threading.Thread(target=run_stream)
    stream_thread.start()
    assert write_entered.wait(timeout=1.0)
    state_thread = threading.Thread(target=read_state)
    state_thread.start()
    state_thread.join(timeout=1.0)

    assert not state_thread.is_alive()
    assert len(state_results) == 1
    assert stream_thread.is_alive()

    release_write.set()
    stream_thread.join(timeout=2.0)
    assert not stream_thread.is_alive()
    assert stream_errors == []


def test_stop_transport_gate_orders_idle_after_inflight_servoj() -> None:
    arm, sdk = enabled_arm()
    approved_generation = arm.stop_generation
    write_entered = threading.Event()
    release_write = threading.Event()
    stop_latched = threading.Event()
    sdk.driver.servoj_write_entered = write_entered
    sdk.driver.servoj_write_release = release_write
    original_latch_stop = arm._latch_stop
    stream_results: list[object] = []
    errors: list[BaseException] = []

    def latch_stop() -> int:
        generation = original_latch_stop()
        stop_latched.set()
        return generation

    arm._latch_stop = latch_stop  # type: ignore[method-assign]
    stream = ServoJStream(
        commands=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
        ),
        dt_s=0.004,
    )

    def run_stream() -> None:
        try:
            stream_results.append(
                arm._guarded_stream_servoj(
                    stream,
                    config=ServoJStreamConfig(dt_s=0.004),
                    expected_stop_generation=approved_generation,
                    capability=_GUARDED_MOTION_CAPABILITY,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    def request_stop() -> None:
        try:
            arm.stop()
        except BaseException as exc:
            errors.append(exc)

    stream_thread = threading.Thread(target=run_stream)
    stream_thread.start()
    assert write_entered.wait(timeout=1.0)
    stop_thread = threading.Thread(target=request_stop)
    stop_thread.start()
    assert stop_latched.wait(timeout=1.0)
    assert stop_thread.is_alive()

    release_write.set()
    stream_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)

    assert errors == []
    assert not stream_thread.is_alive()
    assert not stop_thread.is_alive()
    assert len(stream_results) == 1
    assert stream_results[0].ok is False
    assert stream_results[0].abort_reason == "operator_stop"
    transport_names = [name for name, _ in sdk.driver.calls if name in {"writeServoj", "writeIdle"}]
    assert transport_names == ["writeServoj", "writeIdle"]


def test_stop_latched_before_prepare_gate_prevents_servoj_write() -> None:
    arm, sdk = enabled_arm()
    approved_generation = arm.stop_generation
    prepare_waiting = threading.Event()
    stop_latched = threading.Event()
    release_gate = threading.Event()
    original_write = arm._write_servoj_joint
    original_latch_stop = arm._latch_stop
    errors: list[BaseException] = []

    def waiting_write(*args, **kwargs) -> None:
        prepare_waiting.set()
        original_write(*args, **kwargs)

    def latch_stop() -> int:
        generation = original_latch_stop()
        stop_latched.set()
        return generation

    arm._write_servoj_joint = waiting_write  # type: ignore[method-assign]
    arm._latch_stop = latch_stop  # type: ignore[method-assign]
    arm._command_io_lock.acquire()

    def prepare() -> None:
        try:
            arm._guarded_prepare_servoj_stream(
                dt_s=0.004,
                expected_stop_generation=approved_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
        except BaseException as exc:
            errors.append(exc)

    def request_stop() -> None:
        try:
            arm.stop()
        except BaseException as exc:
            errors.append(exc)
        finally:
            release_gate.set()

    prepare_thread = threading.Thread(target=prepare)
    stop_thread = threading.Thread(target=request_stop)
    prepare_thread.start()
    assert prepare_waiting.wait(timeout=1.0)
    stop_thread.start()
    assert stop_latched.wait(timeout=1.0)
    arm._command_io_lock.release()

    prepare_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)
    assert release_gate.is_set()
    assert not prepare_thread.is_alive()
    assert not stop_thread.is_alive()
    assert any(isinstance(exc, RobotMotionInterruptedError) for exc in errors)
    transport_names = [name for name, _ in sdk.driver.calls if name in {"writeServoj", "writeIdle"}]
    assert transport_names == ["writeIdle"]


def test_deadline_stop_uses_dashboard_without_waiting_for_servoj_io_lock() -> None:
    arm, sdk = enabled_arm()
    errors: list[BaseException] = []
    arm._command_io_lock.acquire()

    def request_deadline_stop() -> None:
        try:
            arm._guarded_deadline_stop(capability=_GUARDED_MOTION_CAPABILITY)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=request_deadline_stop)
    worker.start()
    worker.join(timeout=1.0)
    completed_while_servoj_lock_held = not worker.is_alive()
    arm._command_io_lock.release()
    worker.join(timeout=1.0)

    assert completed_while_servoj_lock_held
    assert not worker.is_alive()
    assert errors == []
    assert arm.stop_snapshot[1] is True
    assert ("stopProgram", None) in sdk.dashboard.calls


def test_public_motion_methods_are_permanently_guarded() -> None:
    arm, sdk = enabled_arm()

    with pytest.raises(RobotMotionDisabledError, match="GuardedEliteExecutor"):
        arm.move_joints([0.0] * 6)
    with pytest.raises(RobotMotionDisabledError, match="GuardedEliteExecutor"):
        arm.write_joint_velocity([0.0] * 6, timeout_ms=240)
    with pytest.raises(RobotMotionDisabledError, match="GuardedEliteExecutor"):
        arm.prepare_servoj_stream(dt_s=0.004)

    assert not any(
        name in {"writeTrajectoryPoint", "writeSpeedj", "writeServoj"}
        for name, _ in sdk.driver.calls
    )


def test_guarded_motion_primitive_rejects_missing_capability() -> None:
    arm, sdk = enabled_arm()

    with pytest.raises(PermissionError, match="guarded-executor capability"):
        arm._guarded_prepare_servoj_stream(
            dt_s=0.004,
            expected_stop_generation=arm.stop_generation,
            capability=object(),
        )

    assert not any(name == "writeServoj" for name, _ in sdk.driver.calls)


def test_write_joint_velocity_uses_holorobot_speedj_boundary() -> None:
    arm, sdk = enabled_arm()
    command = [0.1, 0.2, 0.0, -0.1, 0.0, 0.05]

    arm._guarded_write_joint_velocity(
        command,
        timeout_ms=240,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert ("writeSpeedj", (command, 240)) in sdk.driver.calls


def test_rejected_speedj_marks_arm_stopped() -> None:
    arm, sdk = enabled_arm()
    sdk.driver.speedj_result = False

    with pytest.raises(RobotCommandError, match="writeSpeedj rejected"):
        arm._guarded_write_joint_velocity(
            [0.0] * 6,
            timeout_ms=240,
            capability=_GUARDED_MOTION_CAPABILITY,
        )

    with pytest.raises(RobotCommandError, match="stopped"):
        arm._guarded_write_joint_velocity(
            [0.0] * 6,
            timeout_ms=240,
            capability=_GUARDED_MOTION_CAPABILITY,
        )


def test_stop_joint_velocity_sends_zero_without_write_idle() -> None:
    arm, sdk = enabled_arm()

    arm.stop_joint_velocity(timeout_ms=250)

    assert ("writeSpeedj", ([0.0] * 6, 250)) in sdk.driver.calls
    assert not any(name == "writeIdle" for name, _ in sdk.driver.calls)


def test_prepare_servoj_stream_primes_with_long_hold_timeout() -> None:
    arm, sdk = enabled_arm()

    arm._guarded_prepare_servoj_stream(
        dt_s=0.004,
        expected_stop_generation=arm.stop_generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    write = next(value for name, value in sdk.driver.calls if name == "writeServoj")
    assert write[1] == 3000


def test_guarded_endpoint_settle_reuses_holorobot_hold_before_stop() -> None:
    arm, sdk = enabled_arm()
    target = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

    evidence = arm._guarded_settle_servoj_endpoint(
        target,
        expected_stop_generation=arm.stop_generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert evidence["settled"] is True
    assert evidence["sample_count"] == 3
    assert evidence["final_tracking_error_rad"] == pytest.approx(0.0)
    hold_writes = [call for call in sdk.driver.calls if call[0] == "writeServoj"]
    assert len(hold_writes) == 3
    assert all(call[1][1] == 3000 for call in hold_writes)
    assert not any(call[0] == "writeIdle" for call in sdk.driver.calls)


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

    result = arm._guarded_stream_servoj(
        stream,
        capability=_GUARDED_MOTION_CAPABILITY,
        expected_stop_generation=arm.stop_generation,
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


def test_stream_servoj_checks_execution_deadline_inside_command_loop() -> None:
    arm, sdk = enabled_arm()
    stream = ServoJStream(
        commands=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
        ),
        dt_s=0.004,
    )
    checks = 0

    def deadline_exceeded() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    result = arm._guarded_stream_servoj(
        stream,
        capability=_GUARDED_MOTION_CAPABILITY,
        expected_stop_generation=arm.stop_generation,
        config=ServoJStreamConfig(
            dt_s=0.004,
            tracking_check_every_n_commands=99,
        ),
        deadline_exceeded=deadline_exceeded,
    )

    writes = [value for name, value in sdk.driver.calls if name == "writeServoj"]
    assert result.ok is False
    assert result.abort_reason == "execution_deadline_exceeded"
    assert result.commands_sent == 1
    assert len(writes) == 1
    assert arm.stop_snapshot[1] is True


def test_stream_servoj_aborts_on_tracking_error() -> None:
    arm, _ = enabled_arm()
    stream = ServoJStream(
        commands=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
            (0.2, 0.1, 0.2, 0.3, 0.4, 0.5),
        ),
        dt_s=0.004,
    )

    result = arm._guarded_stream_servoj(
        stream,
        capability=_GUARDED_MOTION_CAPABILITY,
        expected_stop_generation=arm.stop_generation,
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

    result = arm._guarded_stream_servoj(
        stream,
        config=ServoJStreamConfig(dt_s=0.004),
        expected_stop_generation=arm.stop_generation,
        capability=_GUARDED_MOTION_CAPABILITY,
    )

    assert result.ok is False
    assert result.abort_reason == "driver_write_failed"
    assert result.commands_sent == 0
