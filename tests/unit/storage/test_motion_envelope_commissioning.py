import json
import math
from types import SimpleNamespace

import pytest

import biblade_fusion.storage.motion_envelope_commissioning as commissioning
from biblade_fusion.core.settings import CollisionConfig, MotionPreflightConfig
from biblade_fusion.storage.motion_envelope_commissioning import (
    CommissioningTrialCandidate,
    _nearest_equivalent_target,
    read_commissioning_trial_candidate,
    write_commissioning_trial_candidate,
)


def _candidate() -> CommissioningTrialCandidate:
    return CommissioningTrialCandidate(
        candidate_id="a" * 64,
        target_view_id="front_r00_c00",
        start_view_id="initial_occupancy_view_003",
        start_joint_positions_rad=(0.0,) * 6,
        raw_target_joint_positions_rad=(0.1,) * 6,
        normalized_target_joint_positions_rad=(0.1,) * 6,
        target_joint_turn_offsets=(0,) * 6,
        goal_joint_positions_rad=(0.02,) * 6,
        direction_scale=0.2,
        maximum_candidate_joint_delta_rad=0.02,
        maximum_remaining_target_joint_delta_rad=0.1,
        mesh_status="clear",
        mesh_continuous_swept_volume_verified=True,
        mesh_minimum_certificate_margin_m=0.01,
        estimated_servoj_duration_s=0.2,
        servoj_command_count=51,
        blocking_reasons=(),
        warnings=("occupancy_disabled_offline_diagnostic_only",),
    )


def test_nearest_equivalent_target_avoids_unnecessary_full_wrist_turn() -> None:
    start = (0.0, 0.0, 0.0, 0.0, 0.0, -0.9)
    target = (0.0, 0.0, 0.0, 0.0, 0.0, -5.4)
    limits = ((-2.0 * math.pi, 2.0 * math.pi),) * 6

    normalized, turns = _nearest_equivalent_target(start, target, limits)

    assert normalized[-1] == pytest.approx(-5.4 + 2.0 * math.pi)
    assert turns == (0, 0, 0, 0, 0, 1)


def test_commissioning_candidate_is_never_execution_capable() -> None:
    candidate = _candidate()

    assert candidate.motion_authorized is False
    assert candidate.execution_capable is False


def test_initialization_without_authorization_field_is_non_authorizing() -> None:
    commissioning._validate_initialization_safety_boundary(
        SimpleNamespace(metadata={"schema_version": 7})
    )

    with pytest.raises(ValueError, match="unexpectedly authorizes motion"):
        commissioning._validate_initialization_safety_boundary(
            SimpleNamespace(metadata={"motion_authorized": True})
        )


def test_commissioning_candidate_round_trip_preserves_fail_closed_boundary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan"
    initialization = tmp_path / "initialization"
    session = tmp_path / "session"
    occupancy = tmp_path / "occupancy"
    view = session / "views" / "0000_start"
    for directory in (plan, initialization, view, occupancy):
        directory.mkdir(parents=True)
    (plan / "view_plan.json").write_text("{}", encoding="utf-8")
    (initialization / "metadata.json").write_text("{}", encoding="utf-8")
    (session / "manifest.json").write_text("{}", encoding="utf-8")
    (view / "metadata.json").write_text("{}", encoding="utf-8")
    (occupancy / "metadata.json").write_text("{}", encoding="utf-8")
    candidate = _candidate()

    class FakeReader:
        def __init__(self, _root) -> None:
            pass

        def descriptor(self, _view_id):
            return SimpleNamespace(relative_path="views/0000_start")

    monkeypatch.setattr(commissioning, "SessionReader", FakeReader)
    monkeypatch.setattr(commissioning, "_derive_candidate", lambda **_kwargs: candidate)

    stored = write_commissioning_trial_candidate(
        tmp_path / "candidate",
        plan=plan,
        initialization=initialization,
        start_session=session,
        start_view_id="initial_occupancy_view_003",
        occupancy=occupancy,
        target_view_id="front_r00_c00",
        maximum_candidate_joint_delta_rad=0.02,
        motion_config=MotionPreflightConfig(),
        collision_config=CollisionConfig(),
    )
    verified = read_commissioning_trial_candidate(stored.path)

    assert verified.candidate == candidate
    assert verified.motion_authorized is False
    assert verified.execution_capable is False
    assert "servoj_stream" not in verified.metadata
    assert verified.metadata["safety_boundary"]["servoj_commands_persisted"] is False

    payload_path = stored.path / "candidate.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["motion_authorized"] = True
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safety boundary"):
        read_commissioning_trial_candidate(stored.path)


def test_commissioning_candidate_rejects_more_than_point_zero_two_rad() -> None:
    with pytest.raises(ValueError, match=r"\(0, 0\.02\]"):
        commissioning._derive_candidate(
            plan=None,
            initialization=None,
            start_session=None,
            start_view_id="start",
            occupancy=None,
            target_view_id="target",
            maximum_candidate_joint_delta_rad=0.020001,
            motion_config=MotionPreflightConfig(),
            collision_config=CollisionConfig(),
            joint_zero_offsets_rad=(0.0,) * 6,
        )
