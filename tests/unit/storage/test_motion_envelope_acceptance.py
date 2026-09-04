from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from biblade_fusion.core.settings import load_settings
from biblade_fusion.storage.motion_envelope_acceptance import (
    motion_control_contract_for_settings,
    motion_control_contract_sha256,
    read_motion_envelope_acceptance,
    write_motion_envelope_acceptance,
)


def _write(path: Path):
    return write_motion_envelope_acceptance(
        path,
        workcell_id="cell-a",
        operator_id="operator-a",
        accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        robot_geometry_hash="1" * 64,
        motion_model_contract_hash="2" * 64,
        motion_control_contract_hash="3" * 64,
        maximum_tracking_deviation_rad=(0.01, 0.02, 0.01, 0.02, 0.01, 0.02),
        maximum_stop_drift_rad=(0.001, 0.002, 0.001, 0.002, 0.001, 0.002),
        safety_margin_factor=1.5,
        maximum_feedback_interval_s=0.01,
        maximum_stop_acknowledgement_s=0.2,
        maximum_stopped_actual_joint_velocity_rad_s=0.002,
        maximum_stopped_target_joint_velocity_rad_s=0.002,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.002,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.002,
        trial_count=12,
        checklist={
            "final_collision_assembly_verified": True,
            "final_servoj_configuration_verified": True,
            "representative_workspace_paths_verified": True,
            "intentional_tracking_fault_stop_verified": True,
            "bootstrap_multichannel_stop_verified": True,
            "segment_boundary_stop_verified": True,
            "emergency_stop_verified": True,
        },
    )


def test_motion_envelope_acceptance_round_trips_and_binds_contracts(tmp_path: Path) -> None:
    stored = _write(tmp_path / "accepted")
    reread = read_motion_envelope_acceptance(stored.path)

    assert reread.acceptance_id == stored.acceptance_id
    assert reread.accepted_joint_uncertainty_rad == pytest.approx(
        (0.0165, 0.033, 0.0165, 0.033, 0.0165, 0.033)
    )
    reread.assert_matches(
        acceptance_id=stored.acceptance_id,
        robot_geometry_hash="1" * 64,
        motion_model_contract_hash="2" * 64,
        motion_control_contract_hash="3" * 64,
    )
    with pytest.raises(ValueError, match="ServoJ control contract"):
        reread.assert_matches(
            acceptance_id=stored.acceptance_id,
            robot_geometry_hash="1" * 64,
            motion_model_contract_hash="2" * 64,
            motion_control_contract_hash="4" * 64,
        )


def test_motion_envelope_acceptance_is_non_overwriting_and_tamper_evident(
    tmp_path: Path,
) -> None:
    stored = _write(tmp_path / "accepted")
    with pytest.raises(FileExistsError):
        _write(stored.path)

    metadata_path = stored.path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["measurements"]["maximum_stop_drift_rad"][0] = 0.5
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        read_motion_envelope_acceptance(stored.path)


def test_motion_control_contract_is_canonical_and_sensitive() -> None:
    first = motion_control_contract_sha256(
        robot_control={"servoj_gain": 2000, "dt_s": 0.004},
        motion_preflight={"speed_scaling": 0.08},
        servoj_runtime={"tracking_error_rad": 0.03},
    )
    reordered = motion_control_contract_sha256(
        robot_control={"dt_s": 0.004, "servoj_gain": 2000},
        motion_preflight={"speed_scaling": 0.08},
        servoj_runtime={"tracking_error_rad": 0.03},
    )
    changed = motion_control_contract_sha256(
        robot_control={"dt_s": 0.004, "servoj_gain": 2001},
        motion_preflight={"speed_scaling": 0.08},
        servoj_runtime={"tracking_error_rad": 0.03},
    )

    assert first == reordered
    assert first != changed


def test_motion_control_contract_excludes_path_search_policy() -> None:
    settings = load_settings("configs/default.yaml")
    baseline = motion_control_contract_for_settings(settings)
    changed = settings.model_copy(
        update={
            "motion_preflight": settings.motion_preflight.model_copy(
                update={
                    "enable_ompl_fallback": not (
                        settings.motion_preflight.enable_ompl_fallback
                    ),
                    "ompl_plan_timeout_s": 0.5,
                    "ompl_rrt_range_rad": 0.1,
                    "ompl_simplify_path": not (
                        settings.motion_preflight.ompl_simplify_path
                    ),
                }
            )
        }
    )

    assert motion_control_contract_for_settings(changed) == baseline


def test_motion_envelope_acceptance_rejects_unmeasured_or_unchecked_values(
    tmp_path: Path,
) -> None:
    values = dict(
        workcell_id="cell-a",
        operator_id="operator-a",
        accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        robot_geometry_hash="1" * 64,
        motion_model_contract_hash="2" * 64,
        motion_control_contract_hash="3" * 64,
        maximum_tracking_deviation_rad=(0.0,) * 6,
        maximum_stop_drift_rad=(0.0,) * 6,
        safety_margin_factor=1.0,
        maximum_feedback_interval_s=0.01,
        maximum_stop_acknowledgement_s=0.2,
        maximum_stopped_actual_joint_velocity_rad_s=0.002,
        maximum_stopped_target_joint_velocity_rad_s=0.002,
        maximum_stopped_actual_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_actual_tcp_angular_velocity_rad_s=0.002,
        maximum_stopped_target_tcp_linear_velocity_m_s=0.001,
        maximum_stopped_target_tcp_angular_velocity_rad_s=0.002,
        trial_count=3,
        checklist={name: True for name in (
            "final_collision_assembly_verified",
            "final_servoj_configuration_verified",
            "representative_workspace_paths_verified",
            "intentional_tracking_fault_stop_verified",
            "bootstrap_multichannel_stop_verified",
            "segment_boundary_stop_verified",
            "emergency_stop_verified",
        )},
    )
    with pytest.raises(ValueError, match="positive six-vector"):
        write_motion_envelope_acceptance(tmp_path / "invalid", **values)
