"""Process-isolated read-only Elite RTSI sampling for long perception cycles."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable
from contextlib import suppress
from importlib import import_module
from multiprocessing.connection import Connection
from typing import Any, Protocol

import numpy as np

from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.conversions import elite_tcp_pose_to_se3
from biblade_fusion.devices.robot.elite_arm import _OUTPUT_RECIPE
from biblade_fusion.devices.robot.elite_readonly import _enum_label

_SAMPLER_FIFO_PRIORITY = 10
_DEFAULT_STARTUP_TIMEOUT_S = 10.0
_DEFAULT_SHUTDOWN_TIMEOUT_S = 60.0


class EliteRtsiSamplerError(RuntimeError):
    """The independent read-only telemetry trace could not be established."""


class _EventLike(Protocol):
    def is_set(self) -> bool: ...


class _SendConnection(Protocol):
    def send(self, value: object) -> None: ...

    def close(self) -> None: ...


_WorkerTarget = Callable[
    [str, str, float, float, _EventLike, _EventLike, _SendConnection],
    None,
]


def _enum_value_label(sdk: Any, enum_name: str, value: object) -> str:
    enum_type = getattr(sdk, enum_name, None)
    if enum_type is not None:
        with suppress(TypeError, ValueError):
            return _enum_label(enum_type(value))
    return _enum_label(value)


def _robot_state_from_recipe(
    sdk: Any,
    recipe: Any,
    *,
    monotonic_time_ns: int | None = None,
    controller_time_s: float | None = None,
) -> RobotState:
    return RobotState(
        monotonic_time_ns=(
            time.monotonic_ns()
            if monotonic_time_ns is None
            else monotonic_time_ns
        ),
        controller_time_s=(
            float(recipe.getValue("timestamp"))
            if controller_time_s is None
            else controller_time_s
        ),
        joint_positions_rad=np.asarray(
            recipe.getValue("actual_joint_positions"), dtype=np.float64
        ),
        base_t_tcp=elite_tcp_pose_to_se3(recipe.getValue("actual_TCP_pose")),
        robot_mode=_enum_value_label(sdk, "RobotMode", recipe.getValue("robot_mode")),
        safety_status=_enum_value_label(
            sdk,
            "SafetyMode",
            recipe.getValue("safety_status"),
        ),
        speed_scaling=float(recipe.getValue("speed_scaling")),
        runtime_state=_enum_label(recipe.getValue("runtime_state")),
        actual_joint_velocity_rad_s=np.asarray(
            recipe.getValue("actual_joint_speeds"), dtype=np.float64
        ),
        target_joint_velocity_rad_s=np.asarray(
            recipe.getValue("target_joint_speeds"), dtype=np.float64
        ),
        actual_tcp_velocity=np.asarray(
            recipe.getValue("actual_TCP_speed"), dtype=np.float64
        ),
        target_tcp_velocity=np.asarray(
            recipe.getValue("target_TCP_speed"), dtype=np.float64
        ),
    )


def _scheduler_snapshot() -> dict[str, object]:
    policy = os.sched_getscheduler(0)
    names = {
        getattr(os, "SCHED_OTHER", -1): "SCHED_OTHER",
        getattr(os, "SCHED_BATCH", -2): "SCHED_BATCH",
        getattr(os, "SCHED_IDLE", -3): "SCHED_IDLE",
        getattr(os, "SCHED_FIFO", -4): "SCHED_FIFO",
        getattr(os, "SCHED_RR", -5): "SCHED_RR",
    }
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    return {
        "policy": names.get(policy, str(policy)),
        "priority": int(os.sched_getparam(0).sched_priority),
        "cpu_affinity": affinity,
    }


def _require_fifo() -> dict[str, object]:
    """Enter and verify the scheduler policy required by live scan sampling."""

    try:
        maximum = os.sched_get_priority_max(os.SCHED_FIFO)
        if not 1 <= _SAMPLER_FIFO_PRIORITY <= maximum:
            raise EliteRtsiSamplerError(
                "configured sampler FIFO priority is unavailable"
            )
        os.sched_setscheduler(
            0,
            os.SCHED_FIFO,
            os.sched_param(_SAMPLER_FIFO_PRIORITY),
        )
        snapshot = _scheduler_snapshot()
    except (AttributeError, OSError) as exc:
        raise EliteRtsiSamplerError(
            "sampler SCHED_FIFO setup failed; run with LimitRTPRIO"
        ) from exc
    if (
        snapshot["policy"] != "SCHED_FIFO"
        or snapshot["priority"] != _SAMPLER_FIFO_PRIORITY
    ):
        raise EliteRtsiSamplerError(
            f"sampler scheduler verification failed: {snapshot}"
        )
    return snapshot


def _elite_rtsi_sampler_worker(
    sdk_import_path: str,
    robot_ip: str,
    rtsi_frequency_hz: float,
    evidence_period_s: float,
    stop_event: _EventLike,
    discard_event: _EventLike,
    connection: _SendConnection,
) -> None:
    """Consume every RTSI packet and retain a bounded-rate auditable trace."""

    client: Any | None = None
    started = False
    ready_sent = False
    trace: list[RobotState] = []
    diagnostics: dict[str, object] = {}
    outcome: tuple[object, ...]
    try:
        diagnostics.update(
            {
                "sampler_kind": "elite_rtsi_process",
                "rtsi_frequency_hz": rtsi_frequency_hz,
                "evidence_period_s": evidence_period_s,
                "scheduler": _require_fifo(),
            }
        )
        sdk = import_module(sdk_import_path)
        client = sdk.RtsiClientInterface()
        client.connect(robot_ip)
        if not client.negotiateProtocolVersion():
            raise EliteRtsiSamplerError("RTSI protocol negotiation failed")
        recipe = client.setupOutputRecipe(list(_OUTPUT_RECIPE), rtsi_frequency_hz)
        if recipe is None:
            raise EliteRtsiSamplerError("RTSI output recipe setup failed")
        if not client.start():
            raise EliteRtsiSamplerError("RTSI synchronization start failed")
        started = True
        if not client.receiveData(recipe, False):
            raise EliteRtsiSamplerError("initial RTSI packet receive failed")
        first_received_ns = time.monotonic_ns()
        first_controller_time_s = float(recipe.getValue("timestamp"))
        first = _robot_state_from_recipe(
            sdk,
            recipe,
            monotonic_time_ns=first_received_ns,
            controller_time_s=first_controller_time_s,
        )
        trace.append(first)
        last_retained_controller_time_s = first.controller_time_s
        last_received_controller_time_s = first.controller_time_s
        last_received_ns = first.monotonic_time_ns
        last_received = first
        packet_count = 1
        maximum_raw_host_gap_s = 0.0
        maximum_raw_controller_gap_s = 0.0
        maximum_controller_lag_s = 0.0
        connection.send(("ready", dict(diagnostics)))
        ready_sent = True

        while not stop_event.is_set():
            if not client.receiveData(recipe, False):
                raise EliteRtsiSamplerError("RTSI packet receive failed")
            received_ns = time.monotonic_ns()
            controller_time_s = float(recipe.getValue("timestamp"))
            raw_controller_gap_s = (
                controller_time_s - last_received_controller_time_s
            )
            if raw_controller_gap_s < 0.0:
                raise EliteRtsiSamplerError("RTSI controller timestamp moved backwards")
            raw_host_gap_s = (received_ns - last_received_ns) / 1e9
            packet_count += 1
            maximum_raw_host_gap_s = max(
                maximum_raw_host_gap_s,
                raw_host_gap_s,
            )
            maximum_raw_controller_gap_s = max(
                maximum_raw_controller_gap_s,
                raw_controller_gap_s,
            )
            host_elapsed_s = (received_ns - first.monotonic_time_ns) / 1e9
            controller_elapsed_s = controller_time_s - first.controller_time_s
            maximum_controller_lag_s = max(
                maximum_controller_lag_s,
                host_elapsed_s - controller_elapsed_s,
            )
            controller_step_s = (
                controller_time_s - last_retained_controller_time_s
            )
            if controller_step_s >= evidence_period_s:
                last_received = _robot_state_from_recipe(
                    sdk,
                    recipe,
                    monotonic_time_ns=received_ns,
                    controller_time_s=controller_time_s,
                )
                trace.append(last_received)
                last_retained_controller_time_s = last_received.controller_time_s
            last_received_controller_time_s = controller_time_s
            last_received_ns = received_ns

        if last_received_controller_time_s > trace[-1].controller_time_s:
            last_received = _robot_state_from_recipe(
                sdk,
                recipe,
                monotonic_time_ns=last_received_ns,
                controller_time_s=last_received_controller_time_s,
            )
            trace.append(last_received)
        diagnostics.update(
            {
                "packet_count": packet_count,
                "retained_sample_count": len(trace),
                "maximum_raw_host_gap_s": maximum_raw_host_gap_s,
                "maximum_raw_controller_gap_s": maximum_raw_controller_gap_s,
                "maximum_controller_lag_s": maximum_controller_lag_s,
                "final_controller_lag_s": (
                    (last_received_ns - first.monotonic_time_ns) / 1e9
                    - (
                        last_received_controller_time_s
                        - first.controller_time_s
                    )
                ),
            }
        )
        outcome = (
            ("cancelled",)
            if discard_event.is_set()
            else ("result", tuple(trace), dict(diagnostics))
        )
    except BaseException as exc:
        outcome = (
            "error",
            type(exc).__name__,
            str(exc),
            dict(diagnostics),
        )
    finally:
        if client is not None:
            if started:
                with suppress(BaseException):
                    client.pause()
            with suppress(BaseException):
                client.disconnect()
        try:
            connection.send(outcome)
        except BaseException:
            if not ready_sent:
                pass
        finally:
            connection.close()


class EliteRtsiProcessSampler:
    """One-shot, process-isolated read-only trace spanning a perception cycle."""

    def __init__(
        self,
        config: RobotConfig,
        *,
        evidence_period_s: float,
        startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
        shutdown_timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
        context: Any | None = None,
        worker_target: _WorkerTarget = _elite_rtsi_sampler_worker,
    ) -> None:
        if config.robot_ip is None:
            raise ValueError("robot.robot_ip must be configured for RTSI sampling")
        if evidence_period_s <= 0.0:
            raise ValueError("evidence_period_s must be positive")
        if startup_timeout_s <= 0.0 or shutdown_timeout_s <= 0.0:
            raise ValueError("sampler timeouts must be positive")
        self._config = config.model_copy(deep=True)
        self._evidence_period_s = float(evidence_period_s)
        self._startup_timeout_s = float(startup_timeout_s)
        self._shutdown_timeout_s = float(shutdown_timeout_s)
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._receive_connection: Connection | None = None
        self._process: Any | None = None
        self._stop_event: Any | None = None
        self._discard_event: Any | None = None
        self._started = False
        self._finished = False
        self._diagnostics: dict[str, object] | None = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def diagnostics(self) -> dict[str, object]:
        if self._diagnostics is None:
            raise EliteRtsiSamplerError("RTSI sampler diagnostics are unavailable")
        return dict(self._diagnostics)

    def start(self) -> None:
        if self._started:
            raise EliteRtsiSamplerError("RTSI sampler was already started")
        self._started = True
        receive_connection, send_connection = self._context.Pipe(duplex=False)
        self._receive_connection = receive_connection
        self._stop_event = self._context.Event()
        self._discard_event = self._context.Event()
        self._process = self._context.Process(
            target=self._worker_target,
            args=(
                self._config.sdk_import_path,
                self._config.robot_ip,
                self._config.rtsi_frequency_hz,
                self._evidence_period_s,
                self._stop_event,
                self._discard_event,
                send_connection,
            ),
            name="bbf-elite-rtsi-stationarity",
            daemon=True,
        )
        self._process.start()
        send_connection.close()
        message = self._receive_message(
            timeout_s=self._startup_timeout_s,
            phase="startup",
        )
        if message[0] != "ready" or len(message) != 2 or not isinstance(
            message[1], dict
        ):
            self._terminate_worker()
            self._raise_message_error(message, phase="startup")
        self._diagnostics = dict(message[1])

    def finish(self) -> tuple[RobotState, ...]:
        if self._finished:
            raise EliteRtsiSamplerError("RTSI sampler was already finished")
        if not self._started:
            raise EliteRtsiSamplerError("RTSI sampler was never started")
        self._finished = True
        self._stop_event.set()
        message = self._receive_message(
            timeout_s=self._shutdown_timeout_s,
            phase="shutdown",
        )
        self._join_worker()
        if message[0] != "result":
            self._raise_message_error(message, phase="shutdown")
        if len(message) != 3 or not isinstance(message[2], dict):
            raise EliteRtsiSamplerError(
                "RTSI sampler returned invalid diagnostics"
            )
        trace = message[1]
        if not isinstance(trace, tuple) or not all(
            isinstance(state, RobotState) for state in trace
        ):
            raise EliteRtsiSamplerError("RTSI sampler returned an invalid trace payload")
        if len(trace) < 3:
            raise EliteRtsiSamplerError(
                "RTSI sampler returned fewer than three telemetry samples"
            )
        self._diagnostics = dict(message[2])
        return trace

    def cancel(self) -> None:
        if self._process is not None and not self._process.is_alive():
            self._finished = True
            if self._receive_connection is not None:
                self._receive_connection.close()
            return
        if self._finished and self._process is None:
            return
        self._finished = True
        if not self._started or self._process is None:
            return
        self._discard_event.set()
        self._stop_event.set()
        message = self._receive_message(
            timeout_s=self._shutdown_timeout_s,
            phase="cancellation",
        )
        self._join_worker()
        if message[0] not in {"cancelled", "result"}:
            self._raise_message_error(message, phase="cancellation")

    def _receive_message(self, *, timeout_s: float, phase: str) -> tuple[object, ...]:
        connection = self._receive_connection
        if connection is None:
            raise EliteRtsiSamplerError(f"RTSI sampler {phase} has no IPC connection")
        if not connection.poll(timeout_s):
            self._terminate_worker()
            raise EliteRtsiSamplerError(f"RTSI sampler {phase} timed out")
        try:
            message = connection.recv()
        except (EOFError, OSError) as exc:
            self._terminate_worker()
            raise EliteRtsiSamplerError(
                f"RTSI sampler {phase} IPC failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(message, tuple) or not message:
            self._terminate_worker()
            raise EliteRtsiSamplerError(f"RTSI sampler {phase} returned an invalid message")
        return message

    def _join_worker(self) -> None:
        process = self._process
        if process is None:
            raise EliteRtsiSamplerError("RTSI sampler has no worker process")
        process.join(timeout=self._shutdown_timeout_s)
        if process.is_alive():
            self._terminate_worker()
            raise EliteRtsiSamplerError("RTSI sampler worker did not terminate")
        if process.exitcode != 0:
            raise EliteRtsiSamplerError(
                f"RTSI sampler worker exited with status {process.exitcode}"
            )
        if self._receive_connection is not None:
            self._receive_connection.close()

    def _terminate_worker(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if self._receive_connection is not None:
            self._receive_connection.close()

    @staticmethod
    def _raise_message_error(message: tuple[object, ...], *, phase: str) -> None:
        if message[0] == "error" and len(message) >= 3:
            raise EliteRtsiSamplerError(
                f"RTSI sampler {phase} failed: {message[1]}: {message[2]}"
            )
        raise EliteRtsiSamplerError(
            f"RTSI sampler {phase} returned unexpected outcome {message[0]!r}"
        )
