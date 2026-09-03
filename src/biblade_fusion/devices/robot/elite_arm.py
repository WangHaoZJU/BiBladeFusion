"""HoloRobot-aligned Elite CS68 state and motion backend.

This is a scoped adaptation of HoloRobot's ``backends/arm/elite/arm.py`` at the
commit recorded by :mod:`biblade_fusion.robotics.provenance`. Motion is additionally
guarded by BiBladeFusion's default-off ``robot.motion_enabled`` setting.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from importlib import import_module
from math import ceil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import RobotConfig
from biblade_fusion.devices.robot._motion_capability import (
    require_guarded_motion_capability,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.conversions import (
    elite_tcp_pose_to_se3,
    se3_to_elite_tcp_pose,
)
from biblade_fusion.devices.robot.elite_readonly import _enum_label
from biblade_fusion.devices.robot.errors import (
    RobotCommandError,
    RobotConfigurationError,
    RobotConnectionError,
    RobotHardwareFaultError,
    RobotMotionDisabledError,
    RobotMotionInterruptedError,
    RobotMotionTimeoutError,
    RobotNotConnectedError,
    RobotNotEnabledError,
    RobotReleasedError,
)
from biblade_fusion.devices.robot.streaming import (
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)

_TRAJECTORY_RESULT_SUCCESS = 0
_SERVOJ_STREAM_TIMEOUT_MS = 500
_SERVOJ_HOLD_TIMEOUT_MS = 3000
_MOTION_SAFE_SAFETY_MODES = {1, 2}  # NORMAL, REDUCED

_OUTPUT_RECIPE = [
    "actual_joint_positions",
    "actual_joint_speeds",
    "target_joint_speeds",
    "actual_joint_torques",
    "actual_TCP_pose",
    "actual_TCP_speed",
    "target_TCP_speed",
    "robot_mode",
    "safety_status",
    "speed_scaling",
    "runtime_state",
    "timestamp",
]
_INPUT_RECIPE = ["speed_slider_fraction"]


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(ceil(0.95 * len(ordered))) - 1)
    return ordered[max(index, 0)]


def _timing_summary(
    *,
    dt_s: float,
    tick_periods: Sequence[float],
    write_durations: Sequence[float],
    loop_durations: Sequence[float],
) -> dict[str, float | int]:
    overruns = [max(0.0, value - dt_s) for value in loop_durations]
    return {
        "planned_dt_s": dt_s,
        "tick_period_count": len(tick_periods),
        "avg_tick_period_s": _average(tick_periods),
        "max_tick_period_s": max(tick_periods, default=0.0),
        "p95_tick_period_s": _p95(tick_periods),
        "avg_write_s": _average(write_durations),
        "max_write_s": max(write_durations, default=0.0),
        "p95_write_s": _p95(write_durations),
        "avg_loop_body_s": _average(loop_durations),
        "max_loop_body_s": max(loop_durations, default=0.0),
        "avg_loop_body_overrun_s": _average(overruns),
        "max_loop_body_overrun_s": max(overruns, default=0.0),
    }


def _package_install_dir(sdk_module: Any) -> Path | None:
    candidate = getattr(sdk_module, "__file__", None)
    if candidate:
        return Path(str(candidate)).resolve().parent
    installed = sys.modules.get("elite_cs_sdk")
    candidate = getattr(installed, "__file__", None)
    return Path(str(candidate)).resolve().parent if candidate else None


def _resolve_script_path(config: RobotConfig, sdk_module: Any) -> str:
    if config.script_file_path is not None:
        configured = config.script_file_path.expanduser().resolve()
        if not configured.is_file():
            raise RobotConfigurationError(
                f"Elite external-control script does not exist: {configured}"
            )
        return str(configured)
    package_dir = _package_install_dir(sdk_module)
    candidate = package_dir / "external_control.script" if package_dir else None
    if candidate is None or not candidate.is_file():
        raise RobotConfigurationError(
            "Cannot locate external_control.script; configure robot.script_file_path"
        )
    return str(candidate)


class EliteArm:
    """Elite CS68 backend copied from HoloRobot's lifecycle and trajectory semantics."""

    def __init__(
        self,
        config: RobotConfig,
        *,
        sdk_module: ModuleType | Any | None = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
        stream_time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        # Keep the effective driver policy independent from later mutations of
        # the caller's Pydantic model.
        self._config = config.model_copy(deep=True)
        self._sdk_module = sdk_module
        self._time = time_fn
        self._sleep = sleep_fn
        self._stream_time = stream_time_fn
        self._sdk: Any | None = None
        self._dashboard: Any | None = None
        self._rtsi: Any | None = None
        self._driver: Any | None = None
        self._connected = False
        self._enabled = False
        self._released = False
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._stop_generation = 0
        self._state_lock = threading.Lock()
        self._command_io_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._trajectory_done: threading.Event | None = None
        self._trajectory_result: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def has_motion_driver(self) -> bool:
        return self._driver is not None

    @property
    def stop_generation(self) -> int:
        """Return the monotonic generation of the latest local stop latch."""

        with self._stop_lock:
            return self._stop_generation

    @property
    def stop_snapshot(self) -> tuple[int, bool]:
        """Return one atomic, read-only ``(generation, stopped)`` snapshot."""

        return self._stop_snapshot()

    @property
    def robot_config(self) -> RobotConfig:
        """Return a defensive copy of the configuration used by this driver."""

        return self._config.model_copy(deep=True)

    def __enter__(self) -> EliteArm:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def connect(self, *, with_driver: bool = True) -> None:
        """Connect Dashboard and RTSI, and optionally the external-control driver."""

        self._ensure_not_released()
        if with_driver:
            self._ensure_motion_configured()
        if self._connected:
            if with_driver and self._driver is None:
                self._connect_driver()
            return
        if self._config.robot_ip is None:
            raise RobotConfigurationError("robot.robot_ip must be configured")

        sdk = self._resolve_sdk()
        self._sdk = sdk
        dashboard = sdk.DashboardClientInterface()
        if not dashboard.connect(self._config.robot_ip, 29999):
            raise RobotConnectionError(
                f"failed to connect Elite Dashboard at {self._config.robot_ip}:29999"
            )
        self._dashboard = dashboard
        try:
            rtsi = sdk.RtsiIOInterface(
                output_recipe=list(_OUTPUT_RECIPE),
                input_recipe=list(_INPUT_RECIPE) if with_driver else [],
                frequency=self._config.rtsi_frequency_hz,
            )
            if not rtsi.connect(self._config.robot_ip):
                raise RobotConnectionError(
                    f"failed to connect Elite RTSI at {self._config.robot_ip}:30004"
                )
            self._rtsi = rtsi
            if with_driver:
                self._connect_driver()
        except Exception:
            self._cleanup_connections()
            raise
        with self._stop_lock:
            self._stopped = False
            # Publish connectivity only after the initial stop state is ready;
            # stop() can therefore never latch and then be overwritten here.
            self._connected = True

    def _connect_driver(self) -> None:
        self._ensure_motion_configured()
        if self._driver is not None:
            return
        sdk = self._sdk or self._resolve_sdk()
        driver_config = sdk.EliteDriverConfig()
        driver_config.robot_ip = self._config.robot_ip
        driver_config.local_ip = self._config.local_ip or ""
        driver_config.headless_mode = self._config.headless_mode
        driver_config.script_file_path = _resolve_script_path(self._config, sdk)
        driver_config.reverse_port = self._config.reverse_port
        driver_config.script_sender_port = self._config.script_sender_port
        driver_config.trajectory_port = self._config.trajectory_port
        driver_config.script_command_port = self._config.script_command_port
        driver_config.servoj_time = self._config.servoj_time_s
        driver_config.servoj_lookahead_time = self._config.servoj_lookahead_time_s
        driver_config.servoj_gain = self._config.servoj_gain
        driver_config.stopj_acc = self._config.stopj_acceleration_rad_s2
        try:
            self._driver = sdk.EliteDriver(driver_config)
            self._sleep(1.0)
        except Exception as exc:
            self._driver = None
            raise RobotConnectionError("failed to create EliteDriver") from exc

    def enable(self) -> None:
        """Power on, release brakes, and establish HoloRobot external control."""

        self._ensure_motion_configured()
        self._ensure_connected()
        self._connect_driver()
        if self._enabled:
            if self._stop_snapshot()[1]:
                raise RobotMotionInterruptedError(
                    "Elite arm stop latch may only be cleared by "
                    "GuardedEliteExecutor after a new one-shot approval"
                )
            self._ensure_driver_reverse_connected()
            return
        try:
            if not self._dashboard.powerOn():
                raise RobotNotEnabledError("Dashboard powerOn() failed")
            if not self._dashboard.brakeRelease():
                raise RobotNotEnabledError("Dashboard brakeRelease() failed")
            scaling = min(
                self._config.default_speed_scaling,
                self._config.maximum_speed_scaling,
            )
            self._dashboard.setSpeedScaling(max(1, min(100, int(scaling * 100))))
            self._wait_driver_connected()
        except RobotNotEnabledError:
            raise
        except Exception as exc:
            raise RobotNotEnabledError("failed to enable Elite arm") from exc
        with self._stop_lock:
            # An asynchronous stop issued during the dashboard sequence wins.
            if self._stopped:
                raise RobotMotionInterruptedError(
                    "Elite arm was stopped while enable was in progress"
                )
            self._enabled = True

    def disable(self) -> None:
        self._ensure_connected()
        try:
            self._dashboard.powerOff()
        finally:
            self._enabled = False

    def release(self) -> None:
        if self._released:
            return
        self._latch_stop()
        self._cleanup_connections()
        self._connected = False
        self._enabled = False
        self._released = True

    def read_state(self) -> RobotState:
        self._ensure_connected()
        with self._state_lock:
            rtsi = self._rtsi
            controller_time_s = float(rtsi.getTimestamp())
            joint_positions_rad = np.asarray(rtsi.getActualJointPositions(), dtype=np.float64)
            actual_joint_velocity = np.asarray(
                rtsi.getActualJointVelocity(),
                dtype=np.float64,
            )
            target_joint_velocity = np.asarray(
                rtsi.getTargetJointVelocity(),
                dtype=np.float64,
            )
            base_t_tcp = elite_tcp_pose_to_se3(rtsi.getActualTCPPose())
            actual_tcp_velocity = np.asarray(rtsi.getActualTCPVelocity(), dtype=np.float64)
            target_tcp_velocity = np.asarray(rtsi.getTargetTCPVelocity(), dtype=np.float64)
            robot_mode = _enum_label(rtsi.getRobotMode())
            safety_status = _enum_label(rtsi.getSafetyStatus())
            runtime_state = _enum_label(rtsi.getRuntimeState())
            speed_scaling = float(rtsi.getActualSpeedScaling())
            # Timestamp the completed serialized RTSI snapshot, not the beginning
            # of a sequence of vendor getters whose latency is otherwise hidden.
            observed_monotonic_ns = time.monotonic_ns()
        return RobotState(
            monotonic_time_ns=observed_monotonic_ns,
            controller_time_s=controller_time_s,
            joint_positions_rad=joint_positions_rad,
            base_t_tcp=base_t_tcp,
            robot_mode=robot_mode,
            safety_status=safety_status,
            speed_scaling=speed_scaling,
            runtime_state=runtime_state,
            actual_joint_velocity_rad_s=actual_joint_velocity,
            target_joint_velocity_rad_s=target_joint_velocity,
            actual_tcp_velocity=actual_tcp_velocity,
            target_tcp_velocity=target_tcp_velocity,
        )

    def move_joints(
        self,
        joint_positions_rad: Sequence[float],
        *,
        timeout_s: float | None = None,
    ) -> None:
        del joint_positions_rad, timeout_s
        raise RobotMotionDisabledError("Direct joint motion is disabled; use GuardedEliteExecutor")

    def _guarded_move_joints(
        self,
        joint_positions_rad: Sequence[float],
        *,
        timeout_s: float | None = None,
        capability: object,
    ) -> None:
        require_guarded_motion_capability(capability)
        joints = self._validated_joint_vector(joint_positions_rad)
        with self._motion_lock:
            self._ensure_ready_for_motion()
            self._execute_trajectory_point(
                joints,
                self._normalize_timeout(timeout_s),
                cartesian=False,
            )

    def move_tcp(self, target: PoseSE3, *, timeout_s: float | None = None) -> None:
        del target, timeout_s
        raise RobotMotionDisabledError(
            "Direct Cartesian motion is disabled; use GuardedEliteExecutor"
        )

    def _guarded_move_tcp(
        self,
        target: PoseSE3,
        *,
        timeout_s: float | None = None,
        capability: object,
    ) -> None:
        require_guarded_motion_capability(capability)
        if target.parent_frame not in {"base", "elite_b_base"}:
            raise RobotCommandError("Elite TCP target must be expressed in base or elite_b_base")
        pose = se3_to_elite_tcp_pose(target).tolist()
        with self._motion_lock:
            self._ensure_ready_for_motion()
            self._execute_trajectory_point(
                pose,
                self._normalize_timeout(timeout_s),
                cartesian=True,
            )

    def stop(self) -> None:
        self._ensure_connected()
        # Latch software motion off before touching the transport.  A rejected or
        # failed writeIdle leaves physical state uncertain and must never leave the
        # backend locally eligible for another unreviewed command.
        self._latch_stop()
        try:
            # Serialize only the current reverse-transport call, never the full
            # motion stream.  writeIdle ends ServoJ, while Dashboard stopProgram
            # transitions the controller task from PLAYING to STOPPED.  Both are
            # required before the caller can prove stationarity through RTSI.
            with self._command_io_lock:
                if self._driver is not None and self._driver.isRobotConnected():
                    accepted = self._driver.writeIdle(0)
                    stop_transport = "writeIdle"
                    self._request_dashboard_task_stop(context="guarded stop")
                else:
                    # Bootstrap is intentionally unpowered and may have no reverse
                    # EliteDriver session yet.  An already-STOPPED Dashboard task
                    # satisfies the controller postcondition; stopProgram itself is
                    # not idempotent on the deployed controller and returns False in
                    # that state.  The caller must still verify sampled RTSI
                    # stationarity before treating a capture as stopped.
                    self._ensure_dashboard_task_stopped(context="bootstrap stop")
                    accepted = True
                    stop_transport = "Dashboard task STOPPED"
        except RobotCommandError:
            raise
        except Exception as exc:
            raise RobotCommandError("failed to stop Elite arm") from exc
        if accepted is False:
            raise RobotCommandError(f"{stop_transport} rejected stop command")

    def _request_dashboard_task_stop(self, *, context: str) -> None:
        """Request a task stop without claiming the asynchronous RTSI postcondition."""

        status_fn = getattr(self._dashboard, "runningStatus", None)
        stop_program = getattr(self._dashboard, "stopProgram", None)
        if stop_program is None:
            raise RobotCommandError(f"Dashboard does not expose stopProgram for {context}")
        if status_fn is not None:
            try:
                before = _enum_label(status_fn()).strip().upper()
            except Exception as exc:
                raise RobotCommandError(
                    f"Dashboard task-status query failed for {context}"
                ) from exc
            if before in {"STOPPED", "3"}:
                return
        try:
            accepted = stop_program()
        except Exception as exc:
            raise RobotCommandError(f"Dashboard {context} command failed") from exc
        if accepted is False:
            raise RobotCommandError(f"Dashboard rejected {context}")

    def _guarded_deadline_stop(self, *, capability: object) -> None:
        """Latch motion off and request Dashboard stop outside ServoJ's I/O lock.

        ``writeServoj`` is a vendor call made while ``_command_io_lock`` is held.
        A deadline watchdog must not queue behind that same lock, so this private,
        capability-gated path uses the already connected Dashboard channel
        directly.  Dashboard acceptance is still a transport acknowledgement,
        not proof of controller stationarity; the coordinator retains its normal
        boundary stop and stationarity checks.
        """

        require_guarded_motion_capability(capability)
        self._ensure_connected()
        self._latch_stop()
        dashboard = self._dashboard
        stop_program = getattr(dashboard, "stopProgram", None)
        if stop_program is None:
            raise RobotCommandError(
                "Dashboard does not expose independent deadline stopProgram"
            )
        try:
            accepted = stop_program()
        except Exception as exc:
            raise RobotCommandError("independent deadline stopProgram failed") from exc
        if accepted is False:
            raise RobotCommandError("Dashboard rejected independent deadline stopProgram")

    def establish_bootstrap_controller_stop(self) -> int:
        """Establish the unpowered startup stop contract via Dashboard only.

        This is deliberately distinct from a segment-boundary ``writeIdle``.  It
        never creates reverse control, powers the arm, releases brakes, or clears a
        stop latch.  The caller must still prove multi-channel RTSI stationarity.
        """

        self._ensure_connected()
        if self._enabled or self._driver is not None:
            raise RobotCommandError(
                "Bootstrap controller stop requires an unpowered, driver-free connection"
            )
        generation = self._latch_stop()
        try:
            with self._command_io_lock:
                self._ensure_dashboard_task_stopped(context="bootstrap controller stop")
        except Exception as exc:
            if isinstance(exc, RobotCommandError):
                raise
            raise RobotCommandError("Dashboard bootstrap stopProgram failed") from exc
        return generation

    def _ensure_dashboard_task_stopped(self, *, context: str) -> None:
        """Make Dashboard task STOPPED without rejecting the already-stopped state."""

        status_fn = getattr(self._dashboard, "runningStatus", None)
        stop_program = getattr(self._dashboard, "stopProgram", None)
        if stop_program is None:
            raise RobotCommandError(f"Dashboard does not expose stopProgram for {context}")
        if status_fn is None:
            if stop_program() is False:
                raise RobotCommandError(f"Dashboard rejected {context}")
            return
        try:
            before = _enum_label(status_fn()).strip().upper()
        except Exception as exc:
            raise RobotCommandError(f"Dashboard task-status query failed for {context}") from exc
        if before in {"STOPPED", "3"}:
            return
        try:
            accepted = stop_program()
        except Exception as exc:
            raise RobotCommandError(f"Dashboard {context} command failed") from exc
        if accepted is False:
            raise RobotCommandError(
                f"Dashboard rejected {context} from task status {before}"
            )
        try:
            after = _enum_label(status_fn()).strip().upper()
        except Exception as exc:
            raise RobotCommandError(
                f"Dashboard post-stop task-status query failed for {context}"
            ) from exc
        if after not in {"STOPPED", "3"}:
            raise RobotCommandError(
                f"Dashboard {context} did not reach STOPPED (status={after})"
            )

    def write_joint_velocity(
        self,
        joint_velocity_rad_s: Sequence[float],
        *,
        timeout_ms: int,
    ) -> None:
        del joint_velocity_rad_s, timeout_ms
        raise RobotMotionDisabledError("Direct SpeedJ motion is disabled; use GuardedEliteExecutor")

    def _guarded_write_joint_velocity(
        self,
        joint_velocity_rad_s: Sequence[float],
        *,
        timeout_ms: int,
        capability: object,
    ) -> None:
        """Write one HoloRobot-style SpeedJ command; scheduling stays above this layer."""

        require_guarded_motion_capability(capability)
        values = self._validated_joint_vector(joint_velocity_rad_s)
        with self._motion_lock:
            self._ensure_ready_for_motion()
            try:
                self._write_speedj(values, timeout_ms=timeout_ms)
            except Exception:
                self._latch_stop()
                raise

    def prepare_joint_velocity_stream(self, *, timeout_ms: int) -> None:
        del timeout_ms
        raise RobotMotionDisabledError(
            "Direct SpeedJ preparation is disabled; use GuardedEliteExecutor"
        )

    def _guarded_prepare_joint_velocity_stream(
        self,
        *,
        timeout_ms: int,
        capability: object,
    ) -> None:
        """Acquire reverse control and validate it with a zero SpeedJ handshake."""

        require_guarded_motion_capability(capability)
        if timeout_ms <= 0:
            raise ValueError("SpeedJ timeout_ms must be positive")
        self.enable()
        with self._motion_lock:
            try:
                self._ensure_ready_for_motion()
                self._ensure_driver_reverse_connected()
                self._write_speedj([0.0] * 6, timeout_ms=timeout_ms)
            except Exception:
                self._latch_stop()
                raise

    def stop_joint_velocity(self, *, timeout_ms: int) -> None:
        """Stop SpeedJ with a zero-velocity write, matching HoloRobot semantics."""

        self._ensure_connected()
        with self._motion_lock:
            try:
                self._write_speedj([0.0] * 6, timeout_ms=timeout_ms)
            finally:
                self._latch_stop()

    def prepare_servoj_stream(
        self,
        *,
        dt_s: float,
        warmup_duration_s: float = 0.0,
    ) -> None:
        del dt_s, warmup_duration_s
        raise RobotMotionDisabledError(
            "Direct ServoJ preparation is disabled; use GuardedEliteExecutor"
        )

    def _guarded_resume_servoj_control(
        self,
        *,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded: Callable[[], bool] | None = None,
    ) -> None:
        """Clear a prior stop latch without sending a motion command.

        This primitive is deliberately private and capability-gated.  The guarded
        executor calls it only after consuming a fresh permit and revalidating the
        exact segment.  A stopped arm must already be powered and brake-released;
        recovery never performs an implicit power-on or brake release.
        """

        require_guarded_motion_capability(capability)
        self._raise_if_execution_deadline(
            deadline_exceeded,
            stage="before approved ServoJ recovery",
        )
        if type(expected_stop_generation) is not int or expected_stop_generation < 0:
            raise ValueError("expected_stop_generation must be a non-negative integer")
        self._ensure_motion_configured()
        self._ensure_connected()
        generation, stopped = self._stop_snapshot()
        if generation != expected_stop_generation or not stopped:
            raise RobotMotionInterruptedError(
                "Elite stop latch changed before approved ServoJ recovery"
            )
        with self._motion_lock:
            if not self._enabled:
                raise RobotNotEnabledError(
                    "Elite arm must be enabled before approved ServoJ recovery"
                )
            self._raise_for_safety()
            if self._driver is None:
                raise RobotNotEnabledError("EliteDriver is unavailable")
            self._ensure_driver_reverse_connected()
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="during approved ServoJ recovery",
            )
            # Compare-and-clear only after every non-motion prerequisite.  stop()
            # never takes _motion_lock, so a newer stop can win while recovery is
            # waiting on reverse-control setup.
            with self._stop_lock:
                if self._stop_generation != expected_stop_generation or not self._stopped:
                    raise RobotMotionInterruptedError(
                        "Elite stop latch changed during approved ServoJ recovery"
                    )
                # The generation comparison and latch clear are one atomic state
                # transition with respect to stop().  Clearing outside this lock
                # could otherwise overwrite a concurrently latched newer stop.
                self._stopped = False

    def _guarded_enable_for_servoj_control(
        self,
        *,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded: Callable[[], bool] | None = None,
    ) -> None:
        """Power/brake the arm after permit consumption without clearing its stop.

        Unknown-space bootstrap deliberately keeps the controller stopped and does
        not power the arm merely because the runtime was opened.  Consequently the
        first approved segment may need hardware preparation.  This private,
        capability-gated primitive performs only Dashboard/reverse-control setup;
        the stop generation and stop latch must remain unchanged.  The existing
        guarded resume primitive is the sole operation that may compare-and-clear
        that latch before ServoJ.
        """

        require_guarded_motion_capability(capability)
        self._raise_if_execution_deadline(
            deadline_exceeded,
            stage="before guarded enable",
        )
        if type(expected_stop_generation) is not int or expected_stop_generation < 0:
            raise ValueError("expected_stop_generation must be a non-negative integer")
        self._ensure_motion_configured()
        self._ensure_connected()
        generation, stopped = self._stop_snapshot()
        if generation != expected_stop_generation or not stopped:
            raise RobotMotionInterruptedError(
                "Guarded enable requires the exact approved stop latch"
            )
        if self._enabled:
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="during guarded enable",
            )
            return
        self._connect_driver()

        def require_exact_latch(stage: str) -> None:
            current_generation, current_stopped = self._stop_snapshot()
            if current_generation != expected_stop_generation or not current_stopped:
                raise RobotMotionInterruptedError(
                    f"Elite arm stop latch changed {stage} during guarded enable"
                )

        try:
            if not self._dashboard.powerOn():
                raise RobotNotEnabledError("Dashboard powerOn() failed")
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="after powerOn",
            )
            require_exact_latch("after powerOn")
            if not self._dashboard.brakeRelease():
                raise RobotNotEnabledError("Dashboard brakeRelease() failed")
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="after brakeRelease",
            )
            require_exact_latch("after brakeRelease")
            scaling = min(
                self._config.default_speed_scaling,
                self._config.maximum_speed_scaling,
            )
            self._dashboard.setSpeedScaling(max(1, min(100, int(scaling * 100))))
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="after speed scaling",
            )
            require_exact_latch("after speed scaling")
            self._wait_driver_connected()
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="after reverse connection",
            )
            require_exact_latch("after reverse connection")
            with self._stop_lock:
                if self._stop_generation != expected_stop_generation or not self._stopped:
                    raise RobotMotionInterruptedError(
                        "Elite arm stop latch changed during guarded enable"
                    )
                self._enabled = True
        except BaseException as exc:
            # Power/brake may already have partially succeeded.  Never leave that
            # partial preparation standing after a newer stop or setup failure.
            self._enabled = False
            with suppress(BaseException):
                if self._driver is not None and self._driver.isRobotConnected():
                    with self._command_io_lock:
                        self._driver.writeIdle(0)
                else:
                    stop_program = getattr(self._dashboard, "stopProgram", None)
                    if stop_program is not None:
                        stop_program()
            with suppress(BaseException):
                self._dashboard.powerOff()
            if isinstance(
                exc,
                (RobotMotionInterruptedError, RobotNotEnabledError),
            ):
                raise
            raise RobotNotEnabledError("failed to guarded-enable Elite arm") from exc

    def _guarded_prepare_servoj_stream(
        self,
        *,
        dt_s: float,
        warmup_duration_s: float = 0.0,
        expected_stop_generation: int,
        capability: object,
        deadline_exceeded: Callable[[], bool] | None = None,
    ) -> None:
        """Prime the reverse socket with the current joint position before ServoJ."""

        require_guarded_motion_capability(capability)
        self._raise_if_execution_deadline(
            deadline_exceeded,
            stage="before ServoJ preparation",
        )
        if dt_s <= 0.0 or warmup_duration_s < 0.0:
            raise ValueError("ServoJ timing values are invalid")
        self._ensure_ready_for_motion()
        self._ensure_driver_reverse_connected()
        positions = self._validated_joint_vector(self._read_actual_joint_positions())
        with self._motion_lock:
            if warmup_duration_s == 0.0:
                self._raise_if_execution_deadline(
                    deadline_exceeded,
                    stage="before ServoJ preparation write",
                )
                self._write_servoj_joint(
                    positions,
                    timeout_ms=_SERVOJ_HOLD_TIMEOUT_MS,
                    expected_stop_generation=expected_stop_generation,
                )
                self._raise_if_execution_deadline(
                    deadline_exceeded,
                    stage="after ServoJ preparation write",
                )
                return
            deadline = self._stream_time() + warmup_duration_s
            while self._stream_time() < deadline:
                self._raise_if_execution_deadline(
                    deadline_exceeded,
                    stage="during ServoJ warmup",
                )
                tick = self._stream_time()
                self._write_servoj_joint(
                    positions,
                    timeout_ms=_SERVOJ_HOLD_TIMEOUT_MS,
                    expected_stop_generation=expected_stop_generation,
                )
                remaining = min(
                    max(0.0, dt_s - (self._stream_time() - tick)),
                    max(0.0, deadline - self._stream_time()),
                )
                if remaining:
                    self._sleep(remaining)
            self._raise_if_execution_deadline(
                deadline_exceeded,
                stage="after ServoJ warmup",
            )

    def write_servoj_hold(
        self,
        joint_positions_rad: Sequence[float],
        *,
        timeout_ms: int = _SERVOJ_HOLD_TIMEOUT_MS,
    ) -> None:
        del joint_positions_rad, timeout_ms
        raise RobotMotionDisabledError("Direct ServoJ hold is disabled; use GuardedEliteExecutor")

    def _guarded_write_servoj_hold(
        self,
        joint_positions_rad: Sequence[float],
        *,
        timeout_ms: int = _SERVOJ_HOLD_TIMEOUT_MS,
        expected_stop_generation: int,
        capability: object,
    ) -> None:
        """Send one long-timeout ServoJ hold for acquisition gaps."""

        require_guarded_motion_capability(capability)
        values = self._validated_joint_vector(joint_positions_rad)
        with self._motion_lock:
            self._ensure_ready_for_motion()
            self._ensure_driver_reverse_connected()
            self._write_servoj_joint(
                values,
                timeout_ms=timeout_ms,
                expected_stop_generation=expected_stop_generation,
            )

    def stream_servoj(
        self,
        stream: ServoJStream,
        *,
        config: ServoJStreamConfig,
        tracking_samples: list[dict[str, Any]] | None = None,
    ) -> StreamServoJResult:
        del stream, config, tracking_samples
        raise RobotMotionDisabledError(
            "Direct ServoJ streaming is disabled; use GuardedEliteExecutor"
        )

    def _guarded_stream_servoj(
        self,
        stream: ServoJStream,
        *,
        config: ServoJStreamConfig,
        expected_stop_generation: int,
        capability: object,
        tracking_samples: list[dict[str, Any]] | None = None,
        deadline_exceeded: Callable[[], bool] | None = None,
    ) -> StreamServoJResult:
        """Send a fixed-rate ServoJ stream with HoloRobot timing/tracking aborts."""

        require_guarded_motion_capability(capability)
        self._ensure_ready_for_motion()
        stream.validate()
        config.validate()
        if not np.isclose(stream.dt_s, config.dt_s):
            raise ValueError("ServoJ stream and runtime dt_s must match")
        if self._driver is None:
            return StreamServoJResult(ok=False, abort_reason="driver_unavailable")
        self._ensure_driver_reverse_connected()
        with self._motion_lock:
            return self._stream_servoj_locked(
                stream,
                config=config,
                expected_stop_generation=expected_stop_generation,
                tracking_samples=tracking_samples,
                deadline_exceeded=deadline_exceeded,
            )

    def _stream_servoj_locked(
        self,
        stream: ServoJStream,
        *,
        config: ServoJStreamConfig,
        expected_stop_generation: int,
        tracking_samples: list[dict[str, Any]] | None,
        deadline_exceeded: Callable[[], bool] | None,
    ) -> StreamServoJResult:
        now = self._stream_time
        start = now()
        commands_sent = 0
        last_index: int | None = None
        maximum_tracking_error = 0.0
        consecutive_tracking = 0
        consecutive_timing = 0
        previous_tick: float | None = None
        tick_periods: list[float] = []
        write_durations: list[float] = []
        loop_durations: list[float] = []

        def result(ok: bool, reason: str | None) -> StreamServoJResult:
            return StreamServoJResult(
                ok=ok,
                commands_sent=commands_sent,
                duration_s=now() - start,
                max_tracking_error_rad=maximum_tracking_error,
                abort_reason=reason,
                last_command_index=last_index,
                timing_summary=_timing_summary(
                    dt_s=stream.dt_s,
                    tick_periods=tick_periods,
                    write_durations=write_durations,
                    loop_durations=loop_durations,
                ),
            )

        for index, command in enumerate(stream.commands):
            tick = now()
            if previous_tick is not None:
                tick_periods.append(tick - previous_tick)
            previous_tick = tick
            if self._is_stopped():
                return result(False, "operator_stop")
            if self._execution_deadline_exceeded(deadline_exceeded):
                self._latch_stop()
                return result(False, "execution_deadline_exceeded")
            write_start = now()
            try:
                self._write_servoj_joint(
                    list(command),
                    timeout_ms=_SERVOJ_STREAM_TIMEOUT_MS,
                    expected_stop_generation=expected_stop_generation,
                )
            except RobotMotionInterruptedError:
                return result(False, "operator_stop")
            except RobotCommandError:
                self._latch_stop()
                return result(False, "driver_write_failed")
            write_end = now()
            write_durations.append(write_end - write_start)
            commands_sent = index + 1
            last_index = index
            if self._execution_deadline_exceeded(deadline_exceeded):
                self._latch_stop()
                return result(False, "execution_deadline_exceeded")

            tracking_check = index > 0 and index % config.tracking_check_every_n_commands == 0
            actual: list[float] | None = None
            if tracking_check or tracking_samples is not None:
                try:
                    actual = self._validated_joint_vector(self._read_actual_joint_positions())
                except Exception:
                    self._latch_stop()
                    return result(False, "feedback_stale")
            loop_end = now()
            loop_duration = loop_end - tick
            loop_durations.append(loop_duration)
            if self._execution_deadline_exceeded(deadline_exceeded):
                self._latch_stop()
                return result(False, "execution_deadline_exceeded")

            if actual is not None:
                error = max(
                    abs(target - measured) for target, measured in zip(command, actual, strict=True)
                )
                maximum_tracking_error = max(maximum_tracking_error, error)
                if tracking_samples is not None:
                    tracking_samples.append(
                        {
                            "index": index,
                            "planned_t_s": round(index * stream.dt_s, 6),
                            "send_t_s": round(write_start - start, 6),
                            "write_duration_ms": round((write_end - write_start) * 1000.0, 3),
                            "q_cmd": list(command),
                            "q_actual": actual,
                            "max_error_rad": error,
                            "tracking_check": tracking_check,
                        }
                    )
                if tracking_check:
                    consecutive_tracking = (
                        consecutive_tracking + 1 if error > config.tracking_error_rad else 0
                    )
                    if consecutive_tracking >= config.max_consecutive_tracking_violations:
                        self._latch_stop()
                        return result(False, "tracking_error_exceeded")

            if index > 0 and loop_duration > config.timing_violation_factor * stream.dt_s:
                consecutive_timing += 1
            else:
                consecutive_timing = 0
            if consecutive_timing >= config.max_consecutive_timing_violations:
                self._latch_stop()
                return result(False, "timing_violation")

            if index < len(stream.commands) - 1:
                deadline = tick + stream.dt_s
                while now() < deadline:
                    if self._is_stopped():
                        return result(False, "operator_stop")
                    if self._execution_deadline_exceeded(deadline_exceeded):
                        self._latch_stop()
                        return result(False, "execution_deadline_exceeded")
                    self._sleep(min(0.01, deadline - now()))
        return result(True, None)

    def _write_speedj(self, values: Sequence[float], *, timeout_ms: int) -> None:
        if self._driver is None:
            raise RobotNotEnabledError("EliteDriver is unavailable")
        command = self._validated_joint_vector(values)
        writer = getattr(self._driver, "writeSpeedj", None)
        if writer is None:
            raise RobotCommandError("EliteDriver does not expose writeSpeedj")
        if writer(command, max(1, int(timeout_ms))) is False:
            raise RobotCommandError("writeSpeedj rejected command")

    def _write_servoj_joint(
        self,
        positions: list[float],
        *,
        timeout_ms: int,
        expected_stop_generation: int,
    ) -> None:
        if self._driver is None:
            raise RobotNotEnabledError("EliteDriver is unavailable")
        command = self._validated_joint_vector(positions)
        with self._command_io_lock:
            self._require_active_stop_generation(expected_stop_generation)
            writer = getattr(self._driver, "writeServoj", None)
            timeout = max(1, int(timeout_ms))
            if writer is not None:
                try:
                    accepted = writer(command, timeout, False, False)
                except TypeError:
                    try:
                        accepted = writer(command, timeout, False)
                    except TypeError:
                        accepted = writer(command, timeout)
            else:
                writer = getattr(self._driver, "writeServoJ", None)
                if writer is None:
                    raise RobotCommandError("EliteDriver does not expose writeServoj")
                try:
                    accepted = writer(
                        command,
                        self._config.servoj_time_s,
                        self._config.servoj_lookahead_time_s,
                        self._config.servoj_gain,
                    )
                except TypeError:
                    accepted = writer(command, self._config.servoj_time_s)
        if accepted is False:
            raise RobotCommandError("writeServoj rejected command")

    def _execute_trajectory_point(
        self,
        positions: list[float],
        timeout_s: float,
        *,
        cartesian: bool,
    ) -> None:
        self._ensure_ready_for_motion()
        sdk = self._sdk
        driver = self._driver
        self._trajectory_done = threading.Event()
        self._trajectory_result = None

        def on_result(result: Any) -> None:
            try:
                self._trajectory_result = int(result)
            except (TypeError, ValueError):
                self._trajectory_result = -1
            if self._trajectory_done is not None:
                self._trajectory_done.set()

        try:
            driver.setTrajectoryResultCallback(on_result)
            start_action = sdk.TrajectoryControlAction.START
            if not driver.writeTrajectoryControlAction(start_action, 1, 200):
                raise RobotCommandError("Trajectory START failed")
            if not driver.writeTrajectoryPoint(
                positions,
                self._config.default_trajectory_time_s,
                self._config.default_blend_radius_m,
                cartesian,
            ):
                raise RobotCommandError("writeTrajectoryPoint failed")
            noop = sdk.TrajectoryControlAction.NOOP
            deadline = self._time() + timeout_s
            while not self._trajectory_done.is_set():
                if self._is_stopped():
                    raise RobotMotionInterruptedError("motion interrupted by stop flag")
                if self._time() > deadline:
                    raise RobotMotionTimeoutError("motion timeout exceeded")
                with suppress(Exception):
                    driver.writeTrajectoryControlAction(noop, 0, 200)
                self._sleep(self._config.motion_poll_period_s)
            result = self._trajectory_result
        except (RobotCommandError, RobotMotionInterruptedError):
            self._latch_stop()
            raise
        except Exception as exc:
            self._latch_stop()
            raise RobotCommandError("Elite trajectory motion failed") from exc
        finally:
            self._trajectory_done = None
            self._trajectory_result = None
            with suppress(Exception):
                driver.writeIdle(0)
        if result != _TRAJECTORY_RESULT_SUCCESS:
            self._latch_stop()
            raise RobotMotionInterruptedError(
                f"trajectory motion did not complete successfully (result={result})"
            )

    def _raise_for_safety(self) -> None:
        try:
            safety = int(self._read_safety_status())
        except Exception as exc:
            raise RobotHardwareFaultError(
                "failed to read or parse robot safety status; motion is blocked"
            ) from exc
        if safety not in _MOTION_SAFE_SAFETY_MODES:
            raise RobotHardwareFaultError(f"robot safety mode {safety} forbids motion")

    def _wait_driver_connected(self) -> None:
        if self._driver is None:
            raise RobotConnectionError("EliteDriver is not initialized")
        if not self._driver.isRobotConnected():
            if self._config.headless_mode:
                if not self._driver.sendExternalControlScript():
                    raise RobotNotEnabledError("sendExternalControlScript() failed")
            elif not self._dashboard.playProgram():
                raise RobotNotEnabledError("Dashboard playProgram() failed")
        deadline = self._time() + 10.0
        while not self._driver.isRobotConnected():
            if self._time() > deadline:
                raise RobotConnectionError(
                    "robot did not connect to EliteDriver reverse ports within timeout"
                )
            self._sleep(0.01)

    def _ensure_driver_reverse_connected(self) -> None:
        if self._driver is None:
            raise RobotNotEnabledError("EliteDriver is unavailable")
        if not self._driver.isRobotConnected():
            self._wait_driver_connected()

    def _ensure_motion_configured(self) -> None:
        if not self._config.motion_enabled:
            raise RobotMotionDisabledError(
                "robot.motion_enabled is false; refusing to initialize motion control"
            )

    def _ensure_not_released(self) -> None:
        if self._released:
            raise RobotReleasedError("Elite arm has already been released")

    def _ensure_connected(self) -> None:
        self._ensure_not_released()
        if not self._connected or self._rtsi is None:
            raise RobotNotConnectedError("Elite arm is not connected")

    def _ensure_ready_for_motion(self) -> None:
        self._ensure_motion_configured()
        self._ensure_connected()
        if not self._enabled:
            raise RobotNotEnabledError("Elite arm is not enabled")
        if self._is_stopped():
            raise RobotMotionInterruptedError("Elite arm is stopped; call enable() before motion")
        self._raise_for_safety()

    def _stop_snapshot(self) -> tuple[int, bool]:
        with self._stop_lock:
            return self._stop_generation, self._stopped

    def _is_stopped(self) -> bool:
        return self._stop_snapshot()[1]

    @staticmethod
    def _execution_deadline_exceeded(
        callback: Callable[[], bool] | None,
    ) -> bool:
        if callback is None:
            return False
        value = callback()
        if type(value) is not bool:
            raise RobotCommandError("execution deadline callback must return bool")
        return value

    def _raise_if_execution_deadline(
        self,
        callback: Callable[[], bool] | None,
        *,
        stage: str,
    ) -> None:
        if self._execution_deadline_exceeded(callback):
            self._latch_stop()
            raise RobotMotionInterruptedError(
                f"segment execution deadline exceeded {stage}"
            )

    def _latch_stop(self) -> int:
        """Atomically publish a new stop before any potentially blocking I/O."""

        with self._stop_lock:
            self._stop_generation += 1
            self._stopped = True
            return self._stop_generation

    def _require_active_stop_generation(self, expected: int) -> None:
        if type(expected) is not int or expected < 0:
            raise ValueError("expected stop generation must be a non-negative integer")
        with self._stop_lock:
            if self._stop_generation != expected or self._stopped:
                raise RobotMotionInterruptedError(
                    "Elite stop generation changed before ServoJ transport write"
                )

    def _read_actual_joint_positions(self) -> Any:
        """Serialize one short RTSI joint getter with GUI/state sampling."""

        self._ensure_connected()
        with self._state_lock:
            return self._rtsi.getActualJointPositions()

    def _read_safety_status(self) -> Any:
        """Serialize one short RTSI safety getter with GUI/state sampling."""

        self._ensure_connected()
        with self._state_lock:
            return self._rtsi.getSafetyStatus()

    def _normalize_timeout(self, timeout_s: float | None) -> float:
        timeout = self._config.default_motion_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0.0 or timeout > self._config.maximum_motion_timeout_s:
            raise ValueError(
                "motion timeout must be positive and not exceed "
                f"{self._config.maximum_motion_timeout_s} seconds"
            )
        return timeout

    @staticmethod
    def _validated_joint_vector(values: Sequence[float]) -> list[float]:
        joints = np.asarray(values, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RobotCommandError("joint target must be a finite six-vector")
        return joints.tolist()

    def _resolve_sdk(self) -> Any:
        if self._sdk_module is not None:
            return self._sdk_module
        try:
            return import_module(self._config.sdk_import_path)
        except Exception as exc:
            raise RobotConfigurationError(
                f"failed to import Elite SDK module {self._config.sdk_import_path!r}"
            ) from exc

    def _cleanup_connections(self) -> None:
        if self._driver is not None:
            with suppress(Exception):
                self._driver.stopControl()
        if self._rtsi is not None:
            with suppress(Exception):
                self._rtsi.disconnect()
        if self._dashboard is not None:
            with suppress(Exception):
                self._dashboard.disconnect()
        self._driver = None
        self._rtsi = None
        self._dashboard = None
