"""One-shot operator approval and live revalidation for Elite ServoJ execution."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from typing import Protocol

import numpy as np

from biblade_fusion.devices.robot._motion_capability import (
    _GUARDED_MOTION_CAPABILITY,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import RobotCommandError
from biblade_fusion.devices.robot.streaming import (
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)
from biblade_fusion.robotics.motion_preflight import (
    JointMotionPreflight,
    validate_preflight_servoj_contract,
)
from biblade_fusion.robotics.occupancy_collision import (
    OccupancyEvidenceError,
    OccupancyMapEvidence,
    OccupancyRobotCollisionChecker,
    OccupancySemanticAttestation,
)
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    Es68PinocchioCollisionChecker,
)


def _is_sha256_digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


class GuardedArm(Protocol):
    @property
    def stop_generation(self) -> int: ...

    def read_state(self) -> RobotState: ...

    def _guarded_resume_servoj_control(
        self,
        *,
        expected_stop_generation: int,
        capability: object,
    ) -> None: ...

    def _guarded_prepare_servoj_stream(
        self,
        *,
        dt_s: float,
        warmup_duration_s: float = 0.0,
        expected_stop_generation: int,
        capability: object,
    ) -> None: ...

    def _guarded_stream_servoj(
        self,
        stream: ServoJStream,
        *,
        config: ServoJStreamConfig,
        expected_stop_generation: int,
        capability: object,
        tracking_samples: list[dict[str, object]] | None = None,
    ) -> StreamServoJResult: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MotionExecutionPermit:
    """Opaque, expiring, one-shot permit bound to one exact preflight."""

    permit_id: str
    preflight_fingerprint: str
    operator_id: str
    collision_model_id: str | None
    collision_model_hash: str | None
    robot_geometry_hash: str
    motion_model_contract_hash: str
    servoj_runtime_config_hash: str
    occupancy_sequence: int
    occupancy_content_hash: str
    occupancy_mapping_context_hash: str
    occupancy_quality_evidence_hash: str
    occupancy_metadata_sha256: str
    occupancy_semantic_verifier_contract_hash: str
    occupancy_semantic_attestation_hash: str
    occupancy_policy_contract_hash: str
    continuous_occupancy_sweep_verified: bool
    stop_generation: int
    issued_monotonic_s: float
    expires_monotonic_s: float


class GuardedEliteExecutor:
    """Execute only a clear preflight approved and revalidated in this process."""

    def __init__(
        self,
        arm: GuardedArm,
        collision_checker: Cs68PinocchioCollisionChecker,
        occupancy_checker: OccupancyRobotCollisionChecker,
        *,
        permit_lifetime_s: float = 30.0,
        live_start_tolerance_rad: float = 0.01,
        execution_freshness_margin_s: float = 1.0,
        clock=time.monotonic,
    ) -> None:
        if (
            not math.isfinite(permit_lifetime_s)
            or permit_lifetime_s <= 0.0
            or not math.isfinite(live_start_tolerance_rad)
            or live_start_tolerance_rad <= 0.0
        ):
            raise ValueError("Execution permit limits must be finite and positive")
        if not isinstance(occupancy_checker, OccupancyRobotCollisionChecker):
            raise ValueError(
                "Guarded execution requires a concrete occupancy collision checker"
            )
        if occupancy_checker.robot_checker is not collision_checker:
            raise ValueError(
                "mesh and occupancy collision checks must share one checker instance"
            )
        if occupancy_checker.ignored_geometry_names:
            raise ValueError(
                "Guarded execution does not permit occupancy-geometry exemptions"
            )
        if not occupancy_checker.motion_semantic_attestation_valid:
            raise ValueError(
                "Guarded execution requires full semantic occupancy attestation"
            )
        if (
            not occupancy_checker.continuous_swept_volume_supported
            or occupancy_checker.verified_robot_geometry_hash
            != collision_checker.robot_geometry_hash
        ):
            raise ValueError(
                "Guarded execution requires a geometry-bound continuous occupancy "
                "sweep checker"
            )
        freshness_margin = float(execution_freshness_margin_s)
        if not math.isfinite(freshness_margin) or freshness_margin < 0.0:
            raise ValueError(
                "execution_freshness_margin_s must be finite and non-negative"
            )
        if (
            not isinstance(collision_checker, Es68PinocchioCollisionChecker)
            or collision_checker.model_name != "es68"
            or not collision_checker.collision_model_id
            or not _is_sha256_digest(collision_checker.collision_model_hash)
            or not _is_sha256_digest(collision_checker.robot_geometry_hash)
            or not _is_sha256_digest(collision_checker.motion_model_contract_hash)
            or not collision_checker.continuous_swept_volume_supported
        ):
            raise ValueError(
                "Guarded execution requires a fully hash-bound ES68 collision checker"
            )
        self._arm = arm
        self._collision_checker = collision_checker
        self._occupancy_checker = occupancy_checker
        self._permit_lifetime_s = float(permit_lifetime_s)
        self._live_start_tolerance_rad = float(live_start_tolerance_rad)
        self._execution_freshness_margin_s = freshness_margin
        self._clock = clock
        # Keep the authoritative permit object inside the executor.  The caller
        # receives a copy of the value contract, but cannot extend its lifetime or
        # alter any binding by returning a dataclasses.replace()-modified instance.
        self._active_permits: dict[str, MotionExecutionPermit] = {}

    @staticmethod
    def preflight_fingerprint(preflight: JointMotionPreflight) -> str:
        payload = asdict(preflight)
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
        self._require_matching_collision_model(preflight)
        self._require_exact_servoj_contract(preflight)
        runtime_config = self._require_preflight_runtime_config(preflight)
        self._require_matching_freshness_margin(preflight)
        evidence = self._require_matching_current_occupancy(preflight)
        operator = operator_id.strip()
        if not operator:
            raise RobotCommandError("operator_id must be non-empty")
        expected = self.approval_prompt(preflight)
        if confirmation.strip() != expected:
            raise RobotCommandError(
                f"operator confirmation mismatch; expected exactly {expected!r}"
            )
        issued = self._finite_clock_value(label="permit issue time")
        expires = issued + self._permit_lifetime_s
        if not math.isfinite(expires):
            raise RobotCommandError("motion execution permit expiry is not finite")
        stop_generation = self._require_arm_stop_generation()
        permit = MotionExecutionPermit(
            permit_id=secrets.token_hex(16),
            preflight_fingerprint=self.preflight_fingerprint(preflight),
            operator_id=operator,
            collision_model_id=self._collision_checker.collision_model_id,
            collision_model_hash=self._collision_checker.collision_model_hash,
            robot_geometry_hash=self._collision_checker.robot_geometry_hash,
            motion_model_contract_hash=(
                self._collision_checker.motion_model_contract_hash
            ),
            servoj_runtime_config_hash=self._servoj_runtime_config_hash(
                runtime_config
            ),
            occupancy_sequence=evidence.sequence,
            occupancy_content_hash=evidence.content_hash,
            occupancy_mapping_context_hash=evidence.mapping_context_hash,
            occupancy_quality_evidence_hash=evidence.quality_evidence_hash,
            occupancy_metadata_sha256=str(evidence.occupancy_metadata_sha256),
            occupancy_semantic_verifier_contract_hash=str(
                evidence.semantic_verifier_contract_hash
            ),
            occupancy_semantic_attestation_hash=str(
                evidence.semantic_attestation_hash
            ),
            occupancy_policy_contract_hash=(
                self._occupancy_checker.policy_contract_hash
            ),
            continuous_occupancy_sweep_verified=True,
            stop_generation=stop_generation,
            issued_monotonic_s=issued,
            expires_monotonic_s=expires,
        )
        # A frozen dataclass prevents ordinary assignment, but it is not a
        # security boundary: object.__setattr__ can still mutate an instance.
        # Keep a distinct authoritative value so a caller-side mutation cannot
        # rewrite the executor's expiry or any other approval binding in place.
        self._active_permits[permit.permit_id] = replace(permit)
        return permit

    def execute(
        self,
        preflight: JointMotionPreflight,
        permit: MotionExecutionPermit,
        *,
        stream_config: ServoJStreamConfig | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> StreamServoJResult:
        """Consume a permit, recheck live start/collision, then execute exact ServoJ."""

        self._consume_permit(preflight, permit)
        stream = preflight.servoj_stream
        if not preflight.ready_for_approval or stream is None:
            raise RobotCommandError("motion preflight is no longer executable")
        self._require_matching_collision_model(preflight)
        self._require_exact_servoj_contract(preflight)
        approved_runtime_config = self._require_preflight_runtime_config(preflight)
        self._require_matching_freshness_margin(preflight)
        if (
            permit.collision_model_id != self._collision_checker.collision_model_id
            or permit.collision_model_hash
            != self._collision_checker.collision_model_hash
            or permit.robot_geometry_hash
            != self._collision_checker.robot_geometry_hash
            or permit.motion_model_contract_hash
            != self._collision_checker.motion_model_contract_hash
        ):
            raise RobotCommandError("motion execution permit collision-model binding mismatch")
        expected_occupancy = self._require_matching_current_occupancy(preflight)
        if (
            permit.occupancy_sequence != expected_occupancy.sequence
            or permit.occupancy_content_hash != expected_occupancy.content_hash
            or permit.occupancy_mapping_context_hash
            != expected_occupancy.mapping_context_hash
            or permit.occupancy_quality_evidence_hash
            != expected_occupancy.quality_evidence_hash
            or permit.occupancy_metadata_sha256
            != expected_occupancy.occupancy_metadata_sha256
            or permit.occupancy_semantic_verifier_contract_hash
            != expected_occupancy.semantic_verifier_contract_hash
            or permit.occupancy_semantic_attestation_hash
            != expected_occupancy.semantic_attestation_hash
            or permit.occupancy_policy_contract_hash
            != self._occupancy_checker.policy_contract_hash
            or permit.continuous_occupancy_sweep_verified is not True
        ):
            raise RobotCommandError("motion execution permit occupancy binding mismatch")
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
        self._revalidate_exact_stream_path(
            state.joint_positions_rad,
            stream,
            preflight,
            expected_occupancy,
        )
        try:
            self._occupancy_checker.assert_current_evidence(
                expected_occupancy,
                required_freshness_horizon_s=self._required_freshness_horizon_s(
                    preflight
                ),
            )
        except OccupancyEvidenceError as exc:
            raise RobotCommandError(
                f"occupancy snapshot changed before execution: {exc}"
            ) from exc
        config = stream_config or approved_runtime_config
        if config != approved_runtime_config:
            raise RobotCommandError(
                "execution ServoJ runtime config differs from the approved preflight"
            )
        if (
            permit.servoj_runtime_config_hash
            != self._servoj_runtime_config_hash(approved_runtime_config)
        ):
            raise RobotCommandError("motion execution permit ServoJ-config binding mismatch")
        try:
            self._require_unchanged_stop_generation(
                permit.stop_generation,
                stage="before_control_recovery",
            )
            self._raise_if_cancellation_requested(
                cancellation_requested,
                stage="before_control_recovery",
            )
            self._arm._guarded_resume_servoj_control(
                expected_stop_generation=permit.stop_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
            self._raise_if_cancellation_requested(
                cancellation_requested,
                stage="after_control_recovery",
            )
            self._require_unchanged_stop_generation(
                permit.stop_generation,
                stage="after_control_recovery",
            )
            if self._finite_clock_value(label="post-recovery execution time") > (
                permit.expires_monotonic_s
            ):
                raise RobotCommandError(
                    "motion execution permit expired during control recovery"
                )
            recovered_state = self._arm.read_state()
            recovered_start_error = float(
                np.max(
                    np.abs(
                        recovered_state.joint_positions_rad
                        - np.asarray(preflight.start_joint_positions_rad)
                    )
                )
            )
            if recovered_start_error > self._live_start_tolerance_rad:
                raise RobotCommandError(
                    "live robot state changed during control recovery "
                    f"({recovered_start_error:.6f} rad)"
                )
            self._revalidate_exact_stream_path(
                recovered_state.joint_positions_rad,
                stream,
                preflight,
                expected_occupancy,
            )
            try:
                self._occupancy_checker.assert_current_evidence(
                    expected_occupancy,
                    required_freshness_horizon_s=(
                        self._required_freshness_horizon_s(preflight)
                    ),
                )
            except OccupancyEvidenceError as exc:
                raise RobotCommandError(
                    f"occupancy snapshot changed during control recovery: {exc}"
                ) from exc
            if self._finite_clock_value(
                label="post-revalidation execution time"
            ) > permit.expires_monotonic_s:
                raise RobotCommandError(
                    "motion execution permit expired during post-recovery revalidation"
                )
            self._require_unchanged_stop_generation(
                permit.stop_generation,
                stage="before_servoj_prepare",
            )
            self._arm._guarded_prepare_servoj_stream(
                dt_s=stream.dt_s,
                expected_stop_generation=permit.stop_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
            self._raise_if_cancellation_requested(
                cancellation_requested,
                stage="after_servoj_prepare",
            )
            self._require_unchanged_stop_generation(
                permit.stop_generation,
                stage="after_servoj_prepare",
            )
            if self._finite_clock_value(
                label="post-prepare execution time"
            ) > permit.expires_monotonic_s:
                raise RobotCommandError(
                    "motion execution permit expired during ServoJ preparation"
                )
            result = self._arm._guarded_stream_servoj(
                stream,
                config=config,
                expected_stop_generation=permit.stop_generation,
                capability=_GUARDED_MOTION_CAPABILITY,
            )
        except BaseException:
            # Driver preparation may already have primed ServoJ before failing,
            # while a streaming backend may raise instead of returning an abort
            # result.  In either case, make a best-effort transition to idle and
            # preserve the original exception for the caller.
            with suppress(BaseException):
                self._arm.stop()
            raise
        # A successful execution is not complete until the exact same arm has
        # acknowledged the segment-boundary stop.  Stop failures propagate and
        # therefore cannot be reported as successful motion.
        self._arm.stop()
        if not result.ok:
            raise RobotCommandError(
                f"ServoJ execution aborted: {result.abort_reason or 'unknown'}"
            )
        return result

    @staticmethod
    def _raise_if_cancellation_requested(
        callback: Callable[[], bool] | None,
        *,
        stage: str,
    ) -> None:
        if callback is None:
            return
        requested = callback()
        if type(requested) is not bool:
            raise RobotCommandError(
                "motion cancellation callback returned a non-boolean value"
            )
        if requested:
            raise RobotCommandError(f"motion execution cancelled at {stage}")

    def _require_arm_stop_generation(self) -> int:
        generation = self._arm.stop_generation
        if type(generation) is not int or generation < 0:
            raise RobotCommandError(
                "Elite arm stop generation must be a non-negative integer"
            )
        return generation

    def _require_unchanged_stop_generation(
        self,
        expected: int,
        *,
        stage: str,
    ) -> None:
        current = self._require_arm_stop_generation()
        if current != expected:
            raise RobotCommandError(
                "Elite arm stop generation changed at "
                f"{stage} (approved={expected}, current={current})"
            )

    def _consume_permit(
        self,
        preflight: JointMotionPreflight,
        permit: MotionExecutionPermit,
    ) -> None:
        authoritative = self._active_permits.pop(permit.permit_id, None)
        if authoritative is None:
            raise RobotCommandError("motion execution permit is unknown or already consumed")
        if permit != authoritative:
            raise RobotCommandError("motion execution permit payload was modified")
        if (
            not math.isfinite(authoritative.issued_monotonic_s)
            or not math.isfinite(authoritative.expires_monotonic_s)
            or authoritative.expires_monotonic_s
            <= authoritative.issued_monotonic_s
        ):
            raise RobotCommandError("motion execution permit timestamps are invalid")
        if self._finite_clock_value(label="permit consumption time") > (
            authoritative.expires_monotonic_s
        ):
            raise RobotCommandError("motion execution permit has expired")
        expected = self.preflight_fingerprint(preflight)
        if authoritative.preflight_fingerprint != expected:
            raise RobotCommandError("motion execution permit belongs to another preflight")

    def _require_matching_current_occupancy(
        self, preflight: JointMotionPreflight
    ) -> OccupancyMapEvidence:
        report = preflight.occupancy
        if (
            not preflight.occupancy_required
            or report is None
            or report.status is not CollisionCheckStatus.CLEAR
            or report.evidence is None
            or not report.evidence.semantic_attestation_valid
            or not preflight.continuous_occupancy_sweep_required
            or not report.continuous_swept_volume_evidence_valid
        ):
            raise RobotCommandError(
                "motion preflight lacks continuous occupancy-sweep evidence"
            )
        expected = report.evidence
        checker_attestation = self._occupancy_checker.semantic_attestation
        if (
            type(checker_attestation) is not OccupancySemanticAttestation
            or expected.occupancy_metadata_sha256
            != checker_attestation.occupancy_metadata_sha256
            or expected.semantic_verifier_contract_hash
            != checker_attestation.semantic_verifier_contract_hash
            or expected.semantic_attestation_hash
            != checker_attestation.attestation_hash
        ):
            raise RobotCommandError(
                "occupancy semantic attestation differs from executor"
            )
        if (
            expected.robot_geometry_hash
            != self._occupancy_checker.verified_robot_geometry_hash
            or expected.robot_geometry_hash
            != self._collision_checker.robot_geometry_hash
        ):
            raise RobotCommandError(
                "occupancy evidence robot geometry differs from executor"
            )
        recorded_policy_hash = report.result.diagnostics.get(
            "occupancy_policy_contract_hash"
        )
        if (
            not _is_sha256_digest(recorded_policy_hash)
            or recorded_policy_hash != self._occupancy_checker.policy_contract_hash
        ):
            raise RobotCommandError(
                "occupancy collision policy differs from the approved preflight"
            )
        try:
            self._occupancy_checker.assert_current_evidence(
                expected,
                required_freshness_horizon_s=self._required_freshness_horizon_s(
                    preflight
                ),
            )
        except OccupancyEvidenceError as exc:
            raise RobotCommandError(
                f"current occupancy snapshot does not match preflight: {exc}"
            ) from exc
        return expected

    def _require_matching_collision_model(
        self, preflight: JointMotionPreflight
    ) -> None:
        collision = preflight.collision
        if collision is None or collision.status is not CollisionCheckStatus.CLEAR:
            raise RobotCommandError("motion preflight lacks clear mesh-collision evidence")
        if (
            not preflight.swept_mesh_required
            or not collision.continuous_swept_volume_verified
        ):
            raise RobotCommandError(
                "motion preflight lacks continuous swept-mesh evidence"
            )
        diagnostics = collision.result.diagnostics
        binding = (
            diagnostics.get("model"),
            diagnostics.get("collision_model_id"),
            diagnostics.get("robot_geometry_hash"),
            diagnostics.get("motion_model_contract_hash"),
        )
        if binding != self._collision_checker.model_binding:
            raise RobotCommandError(
                "motion preflight collision-model evidence does not match executor"
            )

    def _require_exact_servoj_contract(
        self,
        preflight: JointMotionPreflight,
    ) -> None:
        try:
            validate_preflight_servoj_contract(preflight, self._collision_checker)
        except (TypeError, ValueError) as exc:
            raise RobotCommandError(
                f"motion preflight ServoJ contract is not reproducible: {exc}"
            ) from exc

    def _revalidate_exact_stream_path(
        self,
        live_start: np.ndarray,
        stream: ServoJStream,
        preflight: JointMotionPreflight,
        expected_occupancy: OccupancyMapEvidence,
    ) -> None:
        """Continuously recheck every exact command segment before driver prepare."""

        try:
            maximum_step = float(preflight.diagnostics["maximum_joint_step_rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotCommandError(
                "motion preflight maximum joint step is invalid"
            ) from exc
        if not math.isfinite(maximum_step) or maximum_step <= 0.0:
            raise RobotCommandError("motion preflight maximum joint step is invalid")
        chain = (
            tuple(float(value) for value in np.asarray(live_start, dtype=np.float64)),
            *stream.commands,
        )
        freshness_horizon = self._required_freshness_horizon_s(preflight)
        for segment_index, (start, goal) in enumerate(
            zip(chain[:-1], chain[1:], strict=True)
        ):
            live_collision = self._collision_checker.check_path(
                start,
                goal,
                maximum_joint_step_rad=maximum_step,
            )
            if (
                live_collision.status is not CollisionCheckStatus.CLEAR
                or not live_collision.continuous_swept_volume_verified
            ):
                reasons = ", ".join(live_collision.result.blocking_reasons)
                if not reasons:
                    reasons = "continuous_swept_mesh_unavailable"
                raise RobotCommandError(
                    "live collision revalidation blocked exact ServoJ segment "
                    f"{segment_index}: {reasons}"
                )
            live_occupancy = self._occupancy_checker.check_path(
                start,
                goal,
                maximum_joint_step_rad=maximum_step,
                expected_evidence=expected_occupancy,
                required_freshness_horizon_s=freshness_horizon,
            )
            if (
                live_occupancy.status is not CollisionCheckStatus.CLEAR
                or not live_occupancy.continuous_swept_volume_evidence_valid
            ):
                reasons = ", ".join(live_occupancy.result.blocking_reasons) or (
                    "continuous_swept_occupancy_unavailable"
                )
                raise RobotCommandError(
                    "live occupancy revalidation blocked exact ServoJ segment "
                    f"{segment_index}: {reasons}"
                )

    @staticmethod
    def _stream_duration_s(preflight: JointMotionPreflight) -> float:
        stream = preflight.servoj_stream
        if stream is None:
            raise RobotCommandError("motion preflight lacks a ServoJ stream")
        return max(0, len(stream.commands) - 1) * stream.dt_s

    def _required_freshness_horizon_s(
        self, preflight: JointMotionPreflight
    ) -> float:
        return self._stream_duration_s(preflight) + self._execution_freshness_margin_s

    def _require_matching_freshness_margin(
        self, preflight: JointMotionPreflight
    ) -> None:
        try:
            planned = float(preflight.diagnostics["execution_freshness_margin_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotCommandError(
                "motion preflight lacks a valid execution freshness margin"
            ) from exc
        if not math.isfinite(planned) or not math.isclose(
            planned,
            self._execution_freshness_margin_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RobotCommandError(
                "executor freshness margin differs from the approved preflight"
            )

    @staticmethod
    def _require_preflight_runtime_config(
        preflight: JointMotionPreflight,
    ) -> ServoJStreamConfig:
        config = preflight.servoj_runtime_config
        if config is None:
            raise RobotCommandError("motion preflight lacks a ServoJ runtime config")
        try:
            config.validate()
        except ValueError as exc:
            raise RobotCommandError(
                f"motion preflight ServoJ runtime config is invalid: {exc}"
            ) from exc
        stream = preflight.servoj_stream
        if stream is None or config.dt_s != stream.dt_s:
            raise RobotCommandError(
                "motion preflight ServoJ runtime config does not match its stream"
            )
        return config

    @staticmethod
    def _servoj_runtime_config_hash(config: ServoJStreamConfig) -> str:
        encoded = json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _finite_clock_value(self, *, label: str) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise RobotCommandError(f"{label} is unavailable") from exc
        if not math.isfinite(value):
            raise RobotCommandError(f"{label} must be finite")
        return value
