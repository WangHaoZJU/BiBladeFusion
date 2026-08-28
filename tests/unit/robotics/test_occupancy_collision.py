from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from biblade_fusion.mapping import OccupancyMapState, OccupancySnapshot
from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    OccupancyEvidenceError,
    OccupancyQueryState,
    OccupancyRobotCollisionChecker,
    occupancy_evidence_from_snapshot,
)
from biblade_fusion.robotics.occupancy_collision import (
    OccupancySemanticAttestation,
    _issue_occupancy_semantic_attestation,
)


@pytest.fixture(scope="module")
def checker() -> Cs68PinocchioCollisionChecker:
    return Cs68PinocchioCollisionChecker.from_resources()


def _checker(checker, snapshot) -> OccupancyRobotCollisionChecker:
    return OccupancyRobotCollisionChecker(
        checker,
        lambda: snapshot,
        maximum_map_age_s=30.0,
        additional_clearance_m=0.01,
        verified_robot_geometry_hash="9" * 64,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )


def test_robot_aabb_spheres_clear_on_known_free_map(
    checker, occupancy_snapshot
) -> None:
    result = _checker(checker, occupancy_snapshot).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.CLEAR
    assert result.evidence is not None
    assert result.evidence.sequence == occupancy_snapshot.sequence
    assert result.checked_sphere_count == 8
    assert len(result.diagnostics["queries"]) == 8


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (OccupancyQueryState.OCCUPIED, "environment_occupancy_occupied:"),
        (OccupancyQueryState.UNKNOWN, "environment_occupancy_unknown:"),
    ],
)
def test_occupied_and_unknown_voxels_block(
    checker, occupancy_snapshot, state, reason
) -> None:
    occupied = (
        occupancy_snapshot.free_indices
        if state is OccupancyQueryState.OCCUPIED
        else frozenset({(0, 0, 0)})
    )
    snapshot = replace(
        occupancy_snapshot,
        free_indices=frozenset(),
        free_observation_counts=(),
        occupied_indices=occupied,
        content_hash="",
    )
    result = _checker(checker, snapshot).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.BLOCKED
    assert result.blocking_reasons
    assert any(item.startswith(reason) for item in result.blocking_reasons)


@pytest.mark.parametrize("map_state", [OccupancyMapState.MAPPING, OccupancyMapState.STALE])
def test_non_ready_snapshot_fails_closed(
    checker, occupancy_snapshot, map_state
) -> None:
    snapshot = replace(
        occupancy_snapshot,
        map_state=map_state,
        sequence=occupancy_snapshot.sequence + 1,
        state_reason=f"synthetic {map_state.value}",
        content_hash="",
    )
    result = _checker(checker, snapshot).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert result.evidence is None
    assert result.blocking_reasons[0].startswith("occupancy_checker_error:")


def test_protocol_lookalike_snapshot_fails_closed(checker) -> None:
    class MutableSnapshotLookalike:
        frame_id = "base"
        map_state = "map_ready"
        sequence = 1
        content_hash = "a" * 64

    result = _checker(checker, MutableSnapshotLookalike()).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert result.evidence is None
    assert "concrete_immutable_snapshot" in result.blocking_reasons[0]


def test_recomputed_content_hash_catches_in_memory_tampering(
    checker, occupancy_snapshot
) -> None:
    snapshot = replace(occupancy_snapshot)
    object.__setattr__(snapshot, "content_hash", "a" * 64)
    result = _checker(checker, snapshot).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert "content_hash_mismatch" in result.blocking_reasons[0]


def test_map_change_during_query_fails_closed(checker, occupancy_snapshot) -> None:
    last_centre = occupancy_snapshot.source_camera_centres_base_m[-1]
    changed = replace(
        occupancy_snapshot,
        sequence=occupancy_snapshot.sequence + 1,
        source_view_ids=(*occupancy_snapshot.source_view_ids, "changed-view"),
        source_camera_centres_base_m=(
            *occupancy_snapshot.source_camera_centres_base_m,
            (last_centre[0] + 0.03, last_centre[1], last_centre[2]),
        ),
        source_camera_axes_base=(
            *occupancy_snapshot.source_camera_axes_base,
            occupancy_snapshot.source_camera_axes_base[-1],
        ),
        content_hash="",
    )
    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        return occupancy_snapshot if calls["count"] == 1 else changed

    occupancy = OccupancyRobotCollisionChecker(
        checker,
        provider,
        verified_robot_geometry_hash="9" * 64,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
    result = occupancy.check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert "occupancy_snapshot_changed" in result.blocking_reasons[0]


def test_path_report_binds_one_exact_snapshot(checker, occupancy_snapshot) -> None:
    result = _checker(checker, occupancy_snapshot).check_path(
        (0.0,) * 6,
        (0.03, 0.0, 0.0, 0.0, 0.0, 0.0),
        maximum_joint_step_rad=0.02,
    )

    assert result.status is CollisionCheckStatus.CLEAR
    assert result.evidence is not None
    assert result.evidence.binding == (
        "base",
        occupancy_snapshot.sequence,
        occupancy_snapshot.content_hash,
        occupancy_snapshot.mapping_context_hash,
        occupancy_snapshot.quality_evidence_hash,
        "9" * 64,
        None,
        None,
        None,
    )
    assert result.sample_count == 3
    assert result.continuous_swept_volume_verified is False
    assert (
        result.result.diagnostics["path_semantic"]
        == "discrete_joint_samples_only"
    )


def test_adapter_accepts_real_immutable_mapping_snapshot(checker) -> None:
    shape = (20, 20, 20)
    free_voxels = frozenset(
        (x, y, z)
        for x in range(shape[0])
        for y in range(shape[1])
        for z in range(shape[2])
    )
    snapshot = OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.5,
        origin_m=(-5.0, -5.0, -5.0),
        grid_shape=shape,
        free_indices=free_voxels,
        free_observation_counts=tuple((index, 3) for index in sorted(free_voxels)),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset(),
        sequence=3,
        created_at_utc=datetime(2026, 8, 28, 0, 0, tzinfo=UTC),
        source_view_ids=(
            "depth-view-0001",
            "depth-view-0002",
            "depth-view-0003",
        ),
        source_camera_centres_base_m=(
            (0.0, 0.0, 0.0),
            (0.03, 0.0, 0.0),
            (0.06, 0.0, 0.0),
        ),
        source_camera_axes_base=((0.0, 0.0, 1.0),) * 3,
        rebuild_started_at_utc=datetime(2026, 8, 28, 0, 0, tzinfo=UTC),
        map_state=OccupancyMapState.MAP_READY,
        mapping_context_hash="d" * 64,
        parent_evidence_hash="b" * 64,
        quality_evidence_hash="c" * 64,
        state_reason="synthetic fully observed free workspace",
    )
    result = _checker(checker, snapshot).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.CLEAR
    assert result.evidence is not None
    assert result.evidence.content_hash == snapshot.content_hash
    assert result.evidence.mapping_context_hash == snapshot.mapping_context_hash
    assert result.evidence.quality_evidence_hash == snapshot.quality_evidence_hash
    assert result.evidence.robot_geometry_hash == "9" * 64
    assert result.evidence.semantic_attestation_valid is False


@pytest.mark.parametrize("changed_field", ["mapping_context_hash", "quality_evidence_hash"])
def test_evidence_chain_hash_change_during_query_fails_closed(
    checker, occupancy_snapshot, changed_field
) -> None:
    changed = replace(
        occupancy_snapshot,
        **{changed_field: "e" * 64, "content_hash": ""},
    )
    calls = {"count": 0}

    def provider():
        calls["count"] += 1
        return occupancy_snapshot if calls["count"] == 1 else changed

    occupancy = OccupancyRobotCollisionChecker(
        checker,
        provider,
        verified_robot_geometry_hash="9" * 64,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )

    result = occupancy.check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert "occupancy_snapshot_changed" in result.blocking_reasons[0]


def test_checker_requires_hash_bound_robot_geometry(checker, occupancy_snapshot) -> None:
    with pytest.raises(ValueError, match="hash-bound robot geometry"):
        OccupancyRobotCollisionChecker(checker, lambda: occupancy_snapshot)

    with pytest.raises(
        OccupancyEvidenceError, match="robot_geometry_hash_must_be_verified"
    ):
        occupancy_evidence_from_snapshot(
            occupancy_snapshot,
            now_utc=datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
            max_age_s=30.0,
        )


def test_explicit_robot_identity_cannot_override_checker_hash(
    checker, occupancy_snapshot
) -> None:
    bound_checker = replace(checker, robot_geometry_hash="1" * 64)
    with pytest.raises(ValueError, match="differs from robot checker"):
        OccupancyRobotCollisionChecker(
            bound_checker,
            lambda: occupancy_snapshot,
            verified_robot_geometry_hash="2" * 64,
        )


def test_full_semantic_attestation_binds_concrete_snapshot_and_geometry(
    checker, occupancy_snapshot
) -> None:
    attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256="e" * 64,
        snapshot=occupancy_snapshot,
        robot_geometry_hash="9" * 64,
    )
    occupancy = OccupancyRobotCollisionChecker(
        checker,
        lambda: occupancy_snapshot,
        verified_robot_geometry_hash="9" * 64,
        semantic_attestation=attestation,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )

    evidence = occupancy.current_evidence()

    assert evidence.semantic_attestation_valid is True
    assert evidence.occupancy_metadata_sha256 == "e" * 64
    assert evidence.semantic_attestation_hash == attestation.attestation_hash


def test_semantic_attestation_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="full semantic verification"):
        OccupancySemanticAttestation()
