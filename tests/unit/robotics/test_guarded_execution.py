from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot import (
    ServoJStream,
    ServoJStreamConfig,
    StreamServoJResult,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.devices.robot.errors import RobotCommandError
from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    Es68PinocchioCollisionChecker,
    GuardedEliteExecutor,
    OccupancyRobotCollisionChecker,
    preflight_linear_joint_motion,
)
from biblade_fusion.robotics.occupancy_collision import (
    _issue_occupancy_semantic_attestation,
)


class _SyntheticSweptEs68Checker(Es68PinocchioCollisionChecker):
    def check_path(self, *args, **kwargs):
        return replace(
            super().check_path(*args, **kwargs),
            continuous_swept_volume_verified=True,
        )


class _SyntheticContinuousOccupancyChecker(OccupancyRobotCollisionChecker):
    @property
    def continuous_swept_volume_supported(self) -> bool:
        return True

    def check_path(self, *args, **kwargs):
        report = super().check_path(*args, **kwargs)
        if report.status is not CollisionCheckStatus.CLEAR:
            return report
        return replace(
            report,
            continuous_swept_volume_verified=True,
            result=replace(
                report.result,
                diagnostics={
                    **report.result.diagnostics,
                    "continuous_swept_volume_verified": True,
                    "continuous_sweep_backend": "synthetic_test_only",
                },
            ),
        )


def _attested_occupancy_checker(
    checker,
    snapshot,
    provider,
    **kwargs,
) -> _SyntheticContinuousOccupancyChecker:
    attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256="e" * 64,
        snapshot=snapshot,
        robot_geometry_hash=checker.robot_geometry_hash,
    )
    return _SyntheticContinuousOccupancyChecker(
        checker,
        provider,
        semantic_attestation=attestation,
        **kwargs,
    )


def _changed_snapshot(snapshot):
    last_centre = snapshot.source_camera_centres_base_m[-1]
    return replace(
        snapshot,
        sequence=snapshot.sequence + 1,
        created_at_utc=snapshot.created_at_utc + timedelta(milliseconds=1),
        source_view_ids=(*snapshot.source_view_ids, "changed-view"),
        source_camera_centres_base_m=(
            *snapshot.source_camera_centres_base_m,
            (last_centre[0] + 0.03, last_centre[1], last_centre[2]),
        ),
        source_camera_axes_base=(
            *snapshot.source_camera_axes_base,
            snapshot.source_camera_axes_base[-1],
        ),
        content_hash="",
    )


@dataclass
class FakeGuardedArm:
    joint_positions_rad: np.ndarray = field(default_factory=lambda: np.zeros(6))
    prepared: bool = False
    streamed: bool = False
    stopped: bool = False

    def read_state(self) -> RobotState:
        return RobotState(
            monotonic_time_ns=1,
            controller_time_s=1.0,
            joint_positions_rad=self.joint_positions_rad,
            base_t_tcp=PoseSE3.identity("base", "tcp"),
            robot_mode="IDLE",
            safety_status="NORMAL",
            speed_scaling=0.3,
        )

    def _guarded_prepare_servoj_stream(
        self,
        *,
        dt_s: float,
        warmup_duration_s: float = 0.0,
        capability: object,
    ) -> None:
        assert capability is not None
        assert dt_s == 0.004
        assert warmup_duration_s == 0.0
        self.prepared = True

    def _guarded_stream_servoj(
        self,
        stream,
        *,
        config,
        capability,
        tracking_samples=None,
    ):
        assert capability is not None
        assert config.dt_s == stream.dt_s
        assert tracking_samples is None
        self.streamed = True
        return StreamServoJResult(ok=True, commands_sent=len(stream.commands))

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(scope="module")
def checker() -> Cs68PinocchioCollisionChecker:
    base = Cs68PinocchioCollisionChecker.from_resources()
    payload = {field.name: getattr(base, field.name) for field in fields(base)}
    payload.update(
        model_name="es68",
        collision_model_id="test-es68-d435i",
        collision_model_hash="1" * 64,
        robot_geometry_hash="2" * 64,
        motion_model_contract_hash="3" * 64,
        continuous_swept_volume_supported=True,
    )
    return _SyntheticSweptEs68Checker(**payload)


@pytest.fixture(scope="module")
def clear_preflight(checker, occupancy_checker):
    return preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.03, -0.02, 0.01, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
    )


def test_authorization_requires_exact_preflight_bound_confirmation(
    checker, occupancy_checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)

    with pytest.raises(RobotCommandError, match="confirmation mismatch"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation="EXECUTE",
        )

    prompt = executor.approval_prompt(clear_preflight)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=prompt,
    )
    assert permit.preflight_fingerprint.startswith(prompt.removeprefix("EXECUTE "))


def test_execute_revalidates_and_consumes_one_shot_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    result = executor.execute(clear_preflight, permit)

    assert result.ok is True
    assert arm.prepared is True
    assert arm.streamed is True
    with pytest.raises(RobotCommandError, match="already consumed"):
        executor.execute(clear_preflight, permit)


def test_live_start_mismatch_blocks_before_driver_prepare(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm(joint_positions_rad=np.full(6, 0.02))
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        live_start_tolerance_rad=0.01,
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="no longer matches"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_expired_permit_is_consumed_without_motion(
    checker, occupancy_checker, clear_preflight
) -> None:
    clock = {"now": 10.0}
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        permit_lifetime_s=1.0,
        clock=lambda: clock["now"],
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    clock["now"] = 12.0

    with pytest.raises(RobotCommandError, match="expired"):
        executor.execute(clear_preflight, permit)

    assert arm.prepared is False


def test_caller_cannot_extend_expired_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    clock = {"now": 10.0}
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(
        arm,
        checker,
        occupancy_checker,
        permit_lifetime_s=1.0,
        clock=lambda: clock["now"],
    )
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    clock["now"] = 12.0

    with pytest.raises(RobotCommandError, match="payload was modified"):
        executor.execute(
            clear_preflight,
            replace(permit, expires_monotonic_s=1_000_000.0),
        )

    assert arm.prepared is False


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("permit_lifetime_s", float("nan")),
        ("live_start_tolerance_rad", float("nan")),
    ],
)
def test_executor_rejects_nonfinite_limits(
    checker, occupancy_checker, keyword, value
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GuardedEliteExecutor(
            FakeGuardedArm(),
            checker,
            occupancy_checker,
            **{keyword: value},
        )


def test_executor_rejects_nonfinite_clock(
    checker, occupancy_checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(
        FakeGuardedArm(),
        checker,
        occupancy_checker,
        clock=lambda: float("nan"),
    )

    with pytest.raises(RobotCommandError, match="issue time must be finite"):
        executor.authorize(
            clear_preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(clear_preflight),
        )


def test_authorization_rejects_tampered_servoj_detour(
    checker, occupancy_checker, clear_preflight
) -> None:
    original = clear_preflight.servoj_stream
    assert original is not None and len(original.commands) > 2
    commands = list(original.commands)
    midpoint = list(commands[len(commands) // 2])
    midpoint[1] += 0.001
    commands[len(commands) // 2] = tuple(midpoint)
    tampered = replace(
        clear_preflight,
        servoj_stream=ServoJStream(tuple(commands), original.dt_s),
    )
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)

    with pytest.raises(RobotCommandError, match="stream does not reproduce"):
        executor.authorize(
            tampered,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(tampered),
        )


def test_authorization_rejects_changed_occupancy_snapshot(
    checker, occupancy_snapshot
) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    holder["snapshot"] = _changed_snapshot(occupancy_snapshot)
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_authorization_rejects_mutable_snapshot_provider_result(
    checker, occupancy_snapshot
) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )

    class MutableSnapshotLookalike:
        frame_id = "base"
        map_state = "map_ready"
        sequence = occupancy_snapshot.sequence
        content_hash = occupancy_snapshot.content_hash

    holder["snapshot"] = MutableSnapshotLookalike()
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)

    with pytest.raises(RobotCommandError, match="concrete_immutable_snapshot"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_execute_rejects_snapshot_change_after_permit(
    checker, occupancy_snapshot
) -> None:
    holder = {"snapshot": occupancy_snapshot}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: holder["snapshot"],
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy)
    permit = executor.authorize(
        preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(preflight),
    )
    holder["snapshot"] = _changed_snapshot(occupancy_snapshot)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.execute(preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_permit_carries_explicit_occupancy_binding(
    checker, occupancy_checker, clear_preflight
) -> None:
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    evidence = clear_preflight.occupancy.evidence

    assert evidence is not None
    assert permit.occupancy_sequence == evidence.sequence
    assert permit.occupancy_content_hash == evidence.content_hash
    assert permit.occupancy_mapping_context_hash == evidence.mapping_context_hash
    assert permit.occupancy_quality_evidence_hash == evidence.quality_evidence_hash
    assert permit.occupancy_metadata_sha256 == evidence.occupancy_metadata_sha256
    assert (
        permit.occupancy_semantic_verifier_contract_hash
        == evidence.semantic_verifier_contract_hash
    )
    assert (
        permit.occupancy_semantic_attestation_hash
        == evidence.semantic_attestation_hash
    )
    assert permit.continuous_occupancy_sweep_verified is True
    assert permit.collision_model_id == checker.collision_model_id
    assert permit.collision_model_hash == checker.collision_model_hash
    assert permit.robot_geometry_hash == checker.robot_geometry_hash
    assert permit.motion_model_contract_hash == checker.motion_model_contract_hash
    assert len(permit.servoj_runtime_config_hash) == 64
    assert (
        permit.occupancy_policy_contract_hash
        == occupancy_checker.policy_contract_hash
    )


def test_execute_rejects_runtime_guard_relaxation(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="runtime config differs"):
        executor.execute(
            clear_preflight,
            permit,
            stream_config=ServoJStreamConfig(
                dt_s=0.004,
                tracking_error_rad=100.0,
                tracking_check_every_n_commands=99,
            ),
        )

    assert arm.prepared is False
    assert arm.streamed is False


def test_execute_rejects_occupancy_policy_change_after_permit(
    checker, occupancy_checker, clear_preflight
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )
    original = occupancy_checker.additional_clearance_m
    occupancy_checker.additional_clearance_m = original + 0.001
    try:
        with pytest.raises(RobotCommandError, match="policy differs"):
            executor.execute(clear_preflight, permit)
    finally:
        occupancy_checker.additional_clearance_m = original

    assert arm.prepared is False
    assert arm.streamed is False


def test_executor_rejects_distinct_mesh_checker_for_occupancy(checker) -> None:
    other_checker = Cs68PinocchioCollisionChecker.from_resources()
    occupancy = _SyntheticContinuousOccupancyChecker(
        other_checker,
        lambda: None,
        verified_robot_geometry_hash="8" * 64,
    )

    with pytest.raises(ValueError, match="share one checker instance"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)


def test_authorization_requires_freshness_for_whole_planned_stream(
    checker, occupancy_snapshot
) -> None:
    clock = {"utc": datetime(2026, 8, 28, 0, 0, 0, 100000, tzinfo=UTC)}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: occupancy_snapshot,
        maximum_map_age_s=5.0,
        utc_clock=lambda: clock["utc"],
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    executor = GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)
    clock["utc"] = datetime(2026, 8, 28, 0, 0, 4, 900000, tzinfo=UTC)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.authorize(
            preflight,
            operator_id="operator-a",
            confirmation=executor.approval_prompt(preflight),
        )


def test_execute_requires_freshness_for_whole_remaining_stream(
    checker, occupancy_snapshot
) -> None:
    clock = {"utc": datetime(2026, 8, 28, 0, 0, 0, 100000, tzinfo=UTC)}
    occupancy = _attested_occupancy_checker(
        checker,
        occupancy_snapshot,
        lambda: occupancy_snapshot,
        maximum_map_age_s=5.0,
        utc_clock=lambda: clock["utc"],
    )
    preflight = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy)
    permit = executor.authorize(
        preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(preflight),
    )
    clock["utc"] = datetime(2026, 8, 28, 0, 0, 4, 900000, tzinfo=UTC)

    with pytest.raises(RobotCommandError, match="does not match preflight"):
        executor.execute(preflight, permit)

    assert arm.prepared is False
    assert arm.streamed is False


def test_executor_rejects_current_discrete_occupancy_checker(checker) -> None:
    occupancy = OccupancyRobotCollisionChecker(checker, lambda: None)

    with pytest.raises(ValueError, match="semantic occupancy attestation"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, occupancy)


def test_executor_rejects_protocol_only_occupancy_checker(checker) -> None:
    class FakeOccupancyChecker:
        robot_checker = checker
        ignored_geometry_names = ()
        motion_semantic_attestation_valid = True
        continuous_swept_volume_supported = True
        verified_robot_geometry_hash = checker.robot_geometry_hash

    with pytest.raises(ValueError, match="concrete occupancy collision checker"):
        GuardedEliteExecutor(FakeGuardedArm(), checker, FakeOccupancyChecker())


@pytest.mark.parametrize(
    "permit_update",
    [
        {"occupancy_mapping_context_hash": "e" * 64},
        {"occupancy_quality_evidence_hash": "f" * 64},
        {"occupancy_metadata_sha256": "a" * 64},
        {"occupancy_semantic_verifier_contract_hash": "b" * 64},
        {"occupancy_semantic_attestation_hash": "c" * 64},
        {"continuous_occupancy_sweep_verified": False},
    ],
)
def test_execute_rejects_relaxed_or_rebound_occupancy_permit(
    checker, occupancy_checker, clear_preflight, permit_update
) -> None:
    arm = FakeGuardedArm()
    executor = GuardedEliteExecutor(arm, checker, occupancy_checker)
    permit = executor.authorize(
        clear_preflight,
        operator_id="operator-a",
        confirmation=executor.approval_prompt(clear_preflight),
    )

    with pytest.raises(RobotCommandError, match="permit payload was modified"):
        executor.execute(clear_preflight, replace(permit, **permit_update))

    assert arm.prepared is False
    assert arm.streamed is False
