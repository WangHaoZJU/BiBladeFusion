"""One-shot operator approval and live revalidation for Elite ServoJ execution."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import RobotCommandError
from biblade_fusion.devices.robot.streaming import (
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)
from biblade_fusion.robotics.motion_preflight import JointMotionPreflight
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
)


class GuardedArm(Protocol):
    def read_state(self) -> RobotState: ...

    def prepare_servoj_stream(
        self, *, dt_s: float, warmup_duration_s: float = 0.0
    ) -> None: ...

    def stream_servoj(
        self,
        stream: ServoJStream,
        *,
        config: ServoJStreamConfig,
        tracking_samples: list[dict[str, object]] | None = None,
    ) -> StreamServoJResult: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MotionExecutionPermit:
    """Opaque, expiring, one-shot permit bound to one exact preflight."""

    permit_id: str
    preflight_fingerprint: str
    operator_id: str
    issued_monotonic_s: float
    expires_monotonic_s: float


class GuardedEliteExecutor:
    """Execute only a clear preflight approved and revalidated in this process."""

    def __init__(
        self,
        arm: GuardedArm,
        collision_checker: Cs68PinocchioCollisionChecker,
        *,
        permit_lifetime_s: float = 30.0,
        live_start_tolerance_rad: float = 0.01,
        clock=time.monotonic,
    ) -> None:
        if permit_lifetime_s <= 0.0 or live_start_tolerance_rad <= 0.0:
            raise ValueError("Execution permit limits must be positive")
        self._arm = arm
        self._collision_checker = collision_checker
        self._permit_lifetime_s = float(permit_lifetime_s)
        self._live_start_tolerance_rad = float(live_start_tolerance_rad)
        self._clock = clock
        self._active_permits: set[str] = set()

    @staticmethod
    def preflight_fingerprint(preflight: JointMotionPreflight) -> str:
        stream = preflight.servoj_stream
        payload = {
            "status": preflight.status.value,
            "start": preflight.start_joint_positions_rad,
            "goal": preflight.goal_joint_positions_rad,
            "planning_waypoints": preflight.planning_waypoints,
            "stream_dt_s": stream.dt_s if stream is not None else None,
            "stream_commands": stream.commands if stream is not None else None,
            "blocking_reasons": preflight.blocking_reasons,
            "diagnostics": preflight.diagnostics,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def approval_prompt(self, preflight: JointMotionPreflight) -> str:
        fingerprint = self.preflight_fingerprint(preflight)
        return f"EXECUTE {fingerprint[:12]}"

    def authorize(
        self,
        preflight: JointMotionPreflight,
        *,
        operator_id: str,
        confirmation: str,
    ) -> MotionExecutionPermit:
        """Mint a permit only after an exact, preflight-bound operator confirmation."""

        if not preflight.ready_for_approval:
            raise RobotCommandError("motion preflight is not ready for approval")
        operator = operator_id.strip()
        if not operator:
            raise RobotCommandError("operator_id must be non-empty")
        expected = self.approval_prompt(preflight)
        if confirmation.strip() != expected:
            raise RobotCommandError(
                f"operator confirmation mismatch; expected exactly {expected!r}"
            )
        issued = float(self._clock())
        permit = MotionExecutionPermit(
            permit_id=secrets.token_hex(16),
            preflight_fingerprint=self.preflight_fingerprint(preflight),
            operator_id=operator,
            issued_monotonic_s=issued,
            expires_monotonic_s=issued + self._permit_lifetime_s,
        )
        self._active_permits.add(permit.permit_id)
        return permit

    def execute(
        self,
        preflight: JointMotionPreflight,
        permit: MotionExecutionPermit,
        *,
        stream_config: ServoJStreamConfig | None = None,
    ) -> StreamServoJResult:
        """Consume a permit, recheck live start/collision, then execute exact ServoJ."""

        self._consume_permit(preflight, permit)
        stream = preflight.servoj_stream
        if not preflight.ready_for_approval or stream is None:
            raise RobotCommandError("motion preflight is no longer executable")
        state = self._arm.read_state()
        maximum_start_error = float(
            np.max(
                np.abs(
                    state.joint_positions_rad
                    - np.asarray(preflight.start_joint_positions_rad)
                )
            )
        )
        if maximum_start_error > self._live_start_tolerance_rad:
            raise RobotCommandError(
                "live robot state no longer matches preflight start "
                f"({maximum_start_error:.6f} rad)"
            )
        maximum_step = float(preflight.diagnostics["maximum_joint_step_rad"])
        live_collision = self._collision_checker.check_path(
            state.joint_positions_rad,
            preflight.goal_joint_positions_rad,
            maximum_joint_step_rad=maximum_step,
        )
        if live_collision.status is not CollisionCheckStatus.CLEAR:
            reasons = ", ".join(live_collision.result.blocking_reasons)
            raise RobotCommandError(
                f"live collision revalidation blocked motion: {reasons}"
            )
        config = stream_config or ServoJStreamConfig(
            dt_s=stream.dt_s,
            tracking_check_every_n_commands=2,
        )
        if not np.isclose(config.dt_s, stream.dt_s):
            raise RobotCommandError("execution and preflight ServoJ dt_s differ")
        self._arm.prepare_servoj_stream(dt_s=stream.dt_s)
        result = self._arm.stream_servoj(stream, config=config)
        if not result.ok:
            with suppress(Exception):
                self._arm.stop()
            raise RobotCommandError(
                f"ServoJ execution aborted: {result.abort_reason or 'unknown'}"
            )
        return result

    def _consume_permit(
        self,
        preflight: JointMotionPreflight,
        permit: MotionExecutionPermit,
    ) -> None:
        if permit.permit_id not in self._active_permits:
            raise RobotCommandError("motion execution permit is unknown or already consumed")
        self._active_permits.remove(permit.permit_id)
        if float(self._clock()) > permit.expires_monotonic_s:
            raise RobotCommandError("motion execution permit has expired")
        expected = self.preflight_fingerprint(preflight)
        if permit.preflight_fingerprint != expected:
            raise RobotCommandError("motion execution permit belongs to another preflight")
