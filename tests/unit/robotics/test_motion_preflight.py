from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime

import numpy as np
import pytest

from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68KinematicModel,
    Cs68PinocchioCollisionChecker,
    Es68PinocchioCollisionChecker,
    MotionPreflightStatus,
    OccupancyRobotCollisionChecker,
    preflight_linear_joint_motion,
)
from biblade_fusion.robotics import motion_preflight as motion_preflight_module
from biblade_fusion.robotics import occupancy_collision as occupancy_collision_module
from biblade_fusion.robotics.holorobot_joint_planner import (
    HoloRobotJointPlan,
    HoloRobotJointPlanStatus,
    resample_joint_path,
)
from biblade_fusion.robotics.motion_preflight import (
    HOLOROBOT_SAMPLED_VALIDATION,
    validate_preflight_servoj_contract,
)


class _SyntheticSweptEs68Checker(Es68PinocchioCollisionChecker):
    def check_path(self, *args, **kwargs):
        return replace(
            super().check_path(*args, **kwargs),
            continuous_swept_volume_verified=True,
        )


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


def test_preflight_fails_closed_without_collision_checker() -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=None,
    )

    assert report.status is MotionPreflightStatus.CHECKER_UNAVAILABLE
    assert report.blocking_reasons == ("checker_unavailable",)
    assert report.servoj_stream is None
    assert report.motion_authorized is False


def test_preflight_fails_closed_without_required_occupancy(checker) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
    )

    assert report.status is MotionPreflightStatus.CHECKER_UNAVAILABLE
    assert report.blocking_reasons == ("occupancy_checker_unavailable",)
    assert report.servoj_stream is None
    assert report.ready_for_approval is False


def test_real_mesh_checker_produces_swept_volume_evidence() -> None:
    checker = Cs68PinocchioCollisionChecker.from_resources()
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        require_occupancy=False,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.collision is not None
    assert report.collision.continuous_swept_volume_evidence_valid is True
    assert report.servoj_stream is not None
    assert report.ready_for_approval is False


def test_clear_preflight_builds_holorobot_dynamically_limited_servoj_stream(
    checker, occupancy_checker
) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.05, -0.04, 0.03, -0.02, 0.01, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        servoj_dt_s=0.004,
        speed_scaling=0.08,
        velocity_margin=0.8,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.ready_for_approval is True
    assert report.motion_authorized is False
    assert report.servoj_stream is not None
    assert report.occupancy is not None
    assert report.occupancy.continuous_swept_volume_verified is True
    assert report.continuous_occupancy_sweep_required is True
    assert report.occupancy.evidence is not None
    assert report.occupancy.evidence.sequence == 7
    assert report.occupancy.evidence.semantic_attestation_valid is True
    assert report.servoj_stream.commands[0] == (0.0,) * 6
    np.testing.assert_allclose(
        report.servoj_stream.commands[-1],
        [0.05, -0.04, 0.03, -0.02, 0.01, 0.0],
    )
    maximum_velocity = np.asarray(
        Cs68KinematicModel.from_resources().joint_velocity_limits_rad_s()
    )
    commands = np.asarray(report.servoj_stream.commands)
    observed_velocity = np.max(
        np.abs(np.diff(commands, axis=0)) / report.servoj_stream.dt_s,
        axis=0,
    )
    assert np.all(observed_velocity <= maximum_velocity * 0.08 * 0.8 + 1e-12)
    minimum_acceleration_duration_s = 2.0 * np.sqrt(0.05 / (4.0 * 0.08 * 0.8))
    actual_duration_s = (
        len(report.servoj_stream.commands) - 1
    ) * report.servoj_stream.dt_s
    assert actual_duration_s >= minimum_acceleration_duration_s
    assert actual_duration_s < minimum_acceleration_duration_s + 0.004 + 1e-12
    assert (
        report.diagnostics["trajectory_generator"]
        == "holorobot_velocity_acceleration_limited_servoj_v2"
    )
    assert report.diagnostics["servoj_path_knot_count"] == 2
    assert report.diagnostics["limiting_constraint"] == "acceleration"
    assert "acceleration_limits_unavailable" not in report.warnings


def test_online_holorobot_preflight_samples_segments_without_continuous_proof(
    checker,
    occupancy_checker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def continuous_proof_must_not_run(*_args, **_kwargs):
        pytest.fail("online HoloRobot mode must not run recursive continuous proof")

    monkeypatch.setattr(checker, "check_path", continuous_proof_must_not_run)
    monkeypatch.setattr(
        occupancy_checker,
        "check_path",
        continuous_proof_must_not_run,
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.05, -0.04, 0.03, -0.02, 0.01, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        path_validation_mode=HOLOROBOT_SAMPLED_VALIDATION,
    )

    assert report.ready_for_approval is True
    assert report.path_validation_mode == HOLOROBOT_SAMPLED_VALIDATION
    assert report.swept_mesh_required is False
    assert report.continuous_occupancy_sweep_required is False
    assert report.collision is not None
    assert report.occupancy is not None
    assert report.collision.sample_count == 4
    assert report.occupancy.sample_count == 4
    assert report.collision.proof_evidence is None
    assert report.occupancy.proof_evidence is None
    assert report.path_validation_evidence_sha256 is not None

    tampered = replace(report, path_validation_evidence_sha256="0" * 64)
    assert tampered.ready_for_approval is False


def test_online_holorobot_preflight_hashes_occupancy_once_at_each_boundary(
    checker,
    occupancy_checker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = occupancy_collision_module.compute_content_hash
    calls = 0

    def counted(snapshot):
        nonlocal calls
        calls += 1
        return original(snapshot)

    monkeypatch.setattr(occupancy_collision_module, "compute_content_hash", counted)

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.05, -0.04, 0.03, -0.02, 0.01, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        path_validation_mode=HOLOROBOT_SAMPLED_VALIDATION,
    )

    assert report.ready_for_approval is True
    assert report.occupancy is not None
    assert report.occupancy.sample_count == 4
    assert calls == 2


def test_online_holorobot_preflight_uses_bounded_ompl_detour(
    checker,
    occupancy_checker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_check = checker.check

    def corridor_check(configuration):
        result = original_check(configuration)
        joints = np.asarray(configuration, dtype=np.float64)
        if 0.035 <= joints[0] <= 0.075 and joints[2] < 0.025:
            return replace(
                result,
                status=CollisionCheckStatus.BLOCKED,
                blocking_reasons=("synthetic_joint_corridor",),
            )
        return result

    def fake_rrtconnect(start, goal, **_kwargs):
        waypoints = resample_joint_path(
            (
                start,
                (0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
                (0.1, 0.0, 0.05, 0.0, 0.0, 0.0),
                goal,
            ),
            maximum_joint_step_rad=0.02,
        )
        return HoloRobotJointPlan(
            HoloRobotJointPlanStatus.CLEAR,
            waypoints=waypoints,
            diagnostics={"planner": "synthetic_rrtconnect"},
        )

    monkeypatch.setattr(checker, "check", corridor_check)
    monkeypatch.setattr(motion_preflight_module, "ompl_available", lambda: True)
    monkeypatch.setattr(
        motion_preflight_module,
        "plan_holorobot_rrtconnect",
        fake_rrtconnect,
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        path_validation_mode=HOLOROBOT_SAMPLED_VALIDATION,
        enable_ompl_fallback=True,
    )

    assert report.ready_for_approval is True
    assert report.diagnostics["planner"] == "holorobot_composite_ompl_rrtconnect"
    assert report.diagnostics["fallback_used"] is True
    assert max(item[2] for item in report.planning_waypoints) == pytest.approx(0.05)
    assert validate_preflight_servoj_contract(report, checker) == report.servoj_stream
    tampered = replace(
        report,
        planning_waypoints=report.planning_waypoints[:-1],
    )
    assert tampered.ready_for_approval is False


def test_online_holorobot_preflight_does_not_send_unknown_state_to_ompl(
    checker,
    occupancy_checker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_check = checker.check

    def unknown_mid_path(configuration):
        result = original_check(configuration)
        joint_0 = float(configuration[0])
        if 0.035 <= joint_0 <= 0.075:
            return replace(
                result,
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=("synthetic_checker_evidence_unknown",),
            )
        return result

    def ompl_must_not_run(*_args, **_kwargs):
        pytest.fail("UNKNOWN evidence is not a HoloRobot PATH_BLOCKED result")

    monkeypatch.setattr(checker, "check", unknown_mid_path)
    monkeypatch.setattr(motion_preflight_module, "ompl_available", lambda: True)
    monkeypatch.setattr(
        motion_preflight_module,
        "plan_holorobot_rrtconnect",
        ompl_must_not_run,
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        path_validation_mode=HOLOROBOT_SAMPLED_VALIDATION,
        enable_ompl_fallback=True,
    )

    assert report.status is MotionPreflightStatus.BLOCKED
    assert report.collision is not None
    assert report.collision.status is CollisionCheckStatus.UNKNOWN
    assert report.diagnostics["fallback_used"] is False
    assert report.diagnostics["fallback_reason"] == "not_an_interior_path_block"


def test_online_holorobot_preflight_rejects_malformed_fallback_without_raising(
    checker,
    occupancy_checker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_check = checker.check

    def corridor_check(configuration):
        result = original_check(configuration)
        joint_0 = float(np.asarray(configuration, dtype=np.float64)[0])
        if 0.035 <= joint_0 <= 0.075:
            return replace(
                result,
                status=CollisionCheckStatus.BLOCKED,
                blocking_reasons=("synthetic_joint_corridor",),
            )
        return result

    def malformed_rrtconnect(start, goal, **_kwargs):
        return HoloRobotJointPlan(
            HoloRobotJointPlanStatus.CLEAR,
            waypoints=(
                tuple(value + 1e-12 for value in start),
                goal,
            ),
            diagnostics={"planner": "malformed_rrtconnect"},
        )

    monkeypatch.setattr(checker, "check", corridor_check)
    monkeypatch.setattr(motion_preflight_module, "ompl_available", lambda: True)
    monkeypatch.setattr(
        motion_preflight_module,
        "plan_holorobot_rrtconnect",
        malformed_rrtconnect,
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy_checker,
        maximum_joint_step_rad=0.02,
        path_validation_mode=HOLOROBOT_SAMPLED_VALIDATION,
        enable_ompl_fallback=True,
    )

    assert report.status is MotionPreflightStatus.BLOCKED
    assert "ompl_fallback:ompl_path_endpoint_contract_mismatch" in (
        report.blocking_reasons
    )


def test_folded_goal_is_blocked_before_trajectory_generation(checker) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.0, -3.0, 3.0, -3.0, 0.0, 0.0),
        collision_checker=checker,
        require_occupancy=False,
        require_swept_mesh=False,
        maximum_joint_step_rad=0.1,
    )

    assert report.status is MotionPreflightStatus.BLOCKED
    assert report.ready_for_approval is False
    assert report.servoj_stream is None
    assert any(reason.startswith("self_collision:") for reason in report.blocking_reasons)


def test_invalid_joint_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="six-vector"):
        preflight_linear_joint_motion(
            (0.0,) * 5,
            (0.0,) * 6,
            collision_checker=None,
        )


def test_mesh_only_diagnostic_cannot_be_approved(checker) -> None:
    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        require_occupancy=False,
        require_swept_mesh=False,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.servoj_stream is not None
    assert report.occupancy_required is False
    assert report.continuous_occupancy_sweep_required is True
    assert report.ready_for_approval is False
    assert "occupancy_disabled_offline_diagnostic_only" in report.warnings
    assert "continuous_swept_mesh_disabled_offline_diagnostic_only" in report.warnings


def test_real_occupancy_checker_produces_continuous_sweep_proof(
    checker, occupancy_snapshot
) -> None:
    occupancy = OccupancyRobotCollisionChecker(
        checker,
        lambda: occupancy_snapshot,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.blocking_reasons == ()
    assert report.servoj_stream is not None
    assert report.occupancy is not None
    assert report.occupancy.status.value == "clear"
    assert report.occupancy.continuous_swept_volume_evidence_valid is True
    assert report.ready_for_approval is False


def test_disabling_continuous_occupancy_sweep_is_diagnostic_only(
    checker, occupancy_snapshot
) -> None:
    occupancy = OccupancyRobotCollisionChecker(
        checker,
        lambda: occupancy_snapshot,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
        require_continuous_occupancy_sweep=False,
    )

    assert report.status is MotionPreflightStatus.CLEAR
    assert report.servoj_stream is not None
    assert report.continuous_occupancy_sweep_required is False
    assert report.ready_for_approval is False
    assert (
        "continuous_swept_occupancy_disabled_offline_diagnostic_only"
        in report.warnings
    )
    assert (
        "occupancy_semantic_attestation_unavailable_diagnostic_only"
        in report.warnings
    )


def test_preflight_requires_map_freshness_for_full_stream_duration(
    checker, occupancy_snapshot
) -> None:
    occupancy = OccupancyRobotCollisionChecker(
        checker,
        lambda: occupancy_snapshot,
        maximum_map_age_s=5.0,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 4, 900000, tzinfo=UTC),
    )
    assert occupancy.current_evidence().sequence == occupancy_snapshot.sequence

    report = preflight_linear_joint_motion(
        (0.0,) * 6,
        (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        collision_checker=checker,
        occupancy_checker=occupancy,
    )

    assert report.status is MotionPreflightStatus.BLOCKED
    assert report.servoj_stream is None
    assert "occupancy_map_stale_or_unusable" in report.blocking_reasons[0]
    assert report.diagnostics["planned_servoj_duration_s"] > 0.1
