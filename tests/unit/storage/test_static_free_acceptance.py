from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from biblade_fusion.robotics import AcceptedStaticFreeAabb
from biblade_fusion.storage.static_free_acceptance import (
    read_static_free_acceptance,
    write_static_free_acceptance,
)

CHECKLIST = {
    "final_collision_assembly_verified": True,
    "workcell_cleared_verified": True,
    "emergency_stop_verified": True,
    "low_speed_known_safe_path_verified": True,
}


def _write(tmp_path):
    region = AcceptedStaticFreeAabb(
        "robot_staging",
        (-0.4, -0.4, 0.0),
        (0.4, 0.4, 0.8),
    )
    stored = write_static_free_acceptance(
        tmp_path / "acceptance",
        workcell_id="blade-cell-01",
        operator_id="operator-a",
        accepted_at_utc=datetime(2026, 8, 29, 8, 0, tzinfo=UTC),
        robot_geometry_hash="a" * 64,
        workspace_minimum_m=(-1.0, -1.0, 0.0),
        workspace_maximum_m=(1.0, 1.0, 2.0),
        regions=(region,),
        checklist=dict(CHECKLIST),
    )
    return stored, region


def test_static_free_acceptance_round_trip_and_exact_runtime_binding(tmp_path) -> None:
    stored, region = _write(tmp_path)

    reread = read_static_free_acceptance(stored.path)
    reread.assert_matches(
        acceptance_id=stored.acceptance_id,
        robot_geometry_hash="a" * 64,
        workspace_minimum_m=(-1.0, -1.0, 0.0),
        workspace_maximum_m=(1.0, 1.0, 2.0),
        regions=(region,),
    )
    assert reread.operator_id == "operator-a"
    assert len(reread.metadata_sha256) == 64

    with pytest.raises(ValueError, match="robot geometry"):
        reread.assert_matches(
            acceptance_id=stored.acceptance_id,
            robot_geometry_hash="b" * 64,
            workspace_minimum_m=(-1.0, -1.0, 0.0),
            workspace_maximum_m=(1.0, 1.0, 2.0),
            regions=(region,),
        )


def test_static_free_acceptance_rejects_incomplete_physical_checklist(tmp_path) -> None:
    checklist = dict(CHECKLIST)
    checklist["emergency_stop_verified"] = False

    with pytest.raises(ValueError, match="all physical"):
        write_static_free_acceptance(
            tmp_path / "acceptance",
            workcell_id="blade-cell-01",
            operator_id="operator-a",
            accepted_at_utc=datetime(2026, 8, 29, 8, 0, tzinfo=UTC),
            robot_geometry_hash="a" * 64,
            workspace_minimum_m=(-1.0, -1.0, 0.0),
            workspace_maximum_m=(1.0, 1.0, 2.0),
            regions=(
                AcceptedStaticFreeAabb(
                    "robot_staging", (-0.4, -0.4, 0.0), (0.4, 0.4, 0.8)
                ),
            ),
            checklist=checklist,
        )


def test_static_free_acceptance_tampering_is_detected(tmp_path) -> None:
    stored, _ = _write(tmp_path)
    metadata = stored.path / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["accepted_static_free_aabbs"][0]["maximum_m"][0] = 0.9
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        read_static_free_acceptance(stored.path)


def test_static_free_acceptance_is_write_once(tmp_path) -> None:
    stored, _ = _write(tmp_path)

    with pytest.raises(FileExistsError):
        write_static_free_acceptance(
            stored.path,
            workcell_id="blade-cell-01",
            operator_id="operator-a",
            accepted_at_utc=datetime(2026, 8, 29, 8, 0, tzinfo=UTC),
            robot_geometry_hash="a" * 64,
            workspace_minimum_m=(-1.0, -1.0, 0.0),
            workspace_maximum_m=(1.0, 1.0, 2.0),
            regions=(
                AcceptedStaticFreeAabb(
                    "robot_staging", (-0.4, -0.4, 0.0), (0.4, 0.4, 0.8)
                ),
            ),
            checklist=dict(CHECKLIST),
        )
