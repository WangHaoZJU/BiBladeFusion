from __future__ import annotations

import time
from enum import IntEnum
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.elite_rtsi_sampler import (
    EliteRtsiProcessSampler,
    EliteRtsiSamplerError,
    _elite_rtsi_sampler_worker,
    _require_fifo,
    _robot_state_from_recipe,
)


def _state(index: int) -> RobotState:
    return RobotState(
        monotonic_time_ns=1_000_000_000 + index * 50_000_000,
        controller_time_s=1.0 + index * 0.05,
        joint_positions_rad=np.zeros(6),
        base_t_tcp=PoseSE3.identity("base", "tcp"),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=1.0,
        runtime_state="3",
    )


def _successful_worker(
    _sdk_import_path,
    _robot_ip,
    _frequency,
    _period,
    stop_event,
    discard_event,
    connection,
) -> None:
    diagnostics = {"scheduler": {"policy": "SCHED_FIFO", "priority": 10}}
    connection.send(("ready", diagnostics))
    while not stop_event.is_set():
        time.sleep(0.001)
    connection.send(
        ("cancelled",)
        if discard_event.is_set()
        else (
            "result",
            (_state(0), _state(1), _state(2)),
            {**diagnostics, "packet_count": 3},
        )
    )
    connection.close()


def _failing_worker(
    _sdk_import_path,
    _robot_ip,
    _frequency,
    _period,
    stop_event,
    _discard_event,
    connection,
) -> None:
    connection.send(
        ("ready", {"scheduler": {"policy": "SCHED_FIFO", "priority": 10}})
    )
    while not stop_event.is_set():
        time.sleep(0.001)
    connection.send(("error", "SyntheticFailure", "packet stream failed"))
    connection.close()


def _startup_failing_worker(
    _sdk_import_path,
    _robot_ip,
    _frequency,
    _period,
    _stop_event,
    _discard_event,
    connection,
) -> None:
    connection.send(("error", "ConnectFailure", "second RTSI unavailable"))
    connection.close()


def _sampler(worker_target) -> EliteRtsiProcessSampler:
    robot = load_settings("configs/default.yaml").robot.model_copy(
        update={"robot_ip": "192.0.2.1"}
    )
    return EliteRtsiProcessSampler(
        robot,
        evidence_period_s=0.05,
        startup_timeout_s=2.0,
        shutdown_timeout_s=2.0,
        worker_target=worker_target,
    )


def test_process_sampler_returns_worker_trace() -> None:
    sampler = _sampler(_successful_worker)

    sampler.start()
    trace = sampler.finish()

    assert [state.controller_time_s for state in trace] == [1.0, 1.05, 1.1]
    assert sampler.diagnostics["packet_count"] == 3
    assert sampler.is_alive is False


def test_process_sampler_cancel_joins_without_accepting_evidence() -> None:
    sampler = _sampler(_successful_worker)

    sampler.start()
    sampler.cancel()

    assert sampler.is_alive is False


def test_process_sampler_propagates_worker_failure() -> None:
    sampler = _sampler(_failing_worker)
    sampler.start()

    with pytest.raises(
        EliteRtsiSamplerError,
        match="SyntheticFailure: packet stream failed",
    ):
        sampler.finish()
    sampler.cancel()

    assert sampler.is_alive is False


def test_process_sampler_startup_failure_can_be_cleanly_cancelled() -> None:
    sampler = _sampler(_startup_failing_worker)

    with pytest.raises(
        EliteRtsiSamplerError,
        match="ConnectFailure: second RTSI unavailable",
    ):
        sampler.start()
    sampler.cancel()

    assert sampler.is_alive is False


def test_raw_recipe_conversion_preserves_all_stationarity_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RobotMode(IntEnum):
        IDLE = 5

    class SafetyMode(IntEnum):
        NORMAL = 1

    values = {
        "timestamp": 12.5,
        "actual_joint_positions": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "actual_joint_speeds": [0.1] * 6,
        "target_joint_speeds": [0.2] * 6,
        "actual_TCP_pose": [0.4, -0.2, 0.5, 0.0, 0.0, 0.0],
        "actual_TCP_speed": [0.3] * 6,
        "target_TCP_speed": [0.4] * 6,
        "robot_mode": 5,
        "safety_status": 1,
        "speed_scaling": 0.8,
        "runtime_state": 3,
    }
    recipe = SimpleNamespace(getValue=lambda name: values[name])
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.time.monotonic_ns",
        lambda: 99,
    )

    state = _robot_state_from_recipe(
        SimpleNamespace(RobotMode=RobotMode, SafetyMode=SafetyMode),
        recipe,
    )

    assert state.monotonic_time_ns == 99
    assert state.controller_time_s == 12.5
    assert state.robot_mode == "IDLE"
    assert state.safety_status == "NORMAL"
    assert state.runtime_state == "3"
    assert np.array_equal(state.joint_positions_rad, np.arange(1.0, 7.0))
    assert np.array_equal(state.actual_joint_velocity_rad_s, np.full(6, 0.1))
    assert np.array_equal(state.target_joint_velocity_rad_s, np.full(6, 0.2))
    assert np.array_equal(state.actual_tcp_velocity, np.full(6, 0.3))
    assert np.array_equal(state.target_tcp_velocity, np.full(6, 0.4))


def test_worker_consumes_every_packet_and_retains_configured_evidence_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MutableEvent:
        def __init__(self) -> None:
            self.value = False

        def is_set(self) -> bool:
            return self.value

    class RecordingConnection:
        def __init__(self) -> None:
            self.messages = []

        def send(self, value) -> None:
            self.messages.append(value)

        def close(self) -> None:
            pass

    stop = MutableEvent()
    discard = MutableEvent()
    connection = RecordingConnection()
    recipe_values = {
        "timestamp": 0.0,
        "actual_joint_positions": [0.0] * 6,
        "actual_joint_speeds": [0.0] * 6,
        "target_joint_speeds": [0.0] * 6,
        "actual_joint_torques": [0.0] * 6,
        "actual_TCP_pose": [0.0] * 6,
        "actual_TCP_speed": [0.0] * 6,
        "target_TCP_speed": [0.0] * 6,
        "robot_mode": 5,
        "safety_status": 1,
        "speed_scaling": 1.0,
        "runtime_state": 3,
    }
    recipe = SimpleNamespace(getValue=lambda name: recipe_values[name])

    class Client:
        def __init__(self) -> None:
            self.receive_count = 0
            self.paused = False
            self.disconnected = False

        def connect(self, robot_ip) -> None:
            assert robot_ip == "192.0.2.1"

        def negotiateProtocolVersion(self) -> bool:
            return True

        def setupOutputRecipe(self, output_recipe, frequency):
            assert "timestamp" in output_recipe
            assert frequency == 125.0
            return recipe

        def start(self) -> bool:
            return True

        def receiveData(self, _recipe, _read_newest) -> bool:
            self.receive_count += 1
            recipe_values["timestamp"] = self.receive_count * 0.008
            if self.receive_count == 16:
                stop.value = True
            return True

        def pause(self) -> None:
            self.paused = True

        def disconnect(self) -> None:
            self.disconnected = True

    client = Client()

    class RobotMode(IntEnum):
        IDLE = 5

    class SafetyMode(IntEnum):
        NORMAL = 1

    sdk = SimpleNamespace(
        RtsiClientInterface=lambda: client,
        RobotMode=RobotMode,
        SafetyMode=SafetyMode,
    )
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.import_module",
        lambda _name: sdk,
    )
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler._require_fifo",
        lambda: {
            "policy": "SCHED_FIFO",
            "priority": 10,
            "cpu_affinity": [0],
        },
    )

    _elite_rtsi_sampler_worker(
        "elite_cs_sdk",
        "192.0.2.1",
        125.0,
        0.05,
        stop,
        discard,
        connection,
    )

    assert connection.messages[0][0] == "ready"
    assert connection.messages[1][0] == "result"
    trace = connection.messages[1][1]
    assert client.receive_count == 16
    assert [state.controller_time_s for state in trace] == pytest.approx(
        [0.008, 0.064, 0.12, 0.128]
    )
    assert client.paused is True
    assert client.disconnected is True


def test_fifo_setup_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.os.sched_get_priority_max",
        lambda _policy: 99,
    )
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.os.sched_setscheduler",
        lambda pid, policy, parameter: calls.append(
            (pid, policy, parameter.sched_priority)
        ),
    )
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler._scheduler_snapshot",
        lambda: {
            "policy": "SCHED_FIFO",
            "priority": 10,
            "cpu_affinity": [0, 1],
        },
    )

    result = _require_fifo()

    assert result["policy"] == "SCHED_FIFO"
    assert calls == [(0, 1, 10)]


def test_fifo_setup_failure_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.os.sched_get_priority_max",
        lambda _policy: 99,
    )

    def deny(*_args, **_kwargs) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(
        "biblade_fusion.devices.robot.elite_rtsi_sampler.os.sched_setscheduler",
        deny,
    )

    with pytest.raises(EliteRtsiSamplerError, match="LimitRTPRIO"):
        _require_fifo()
