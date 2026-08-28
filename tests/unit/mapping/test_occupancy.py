from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from biblade_fusion.mapping.occupancy import (
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancySnapshot,
    OccupancyState,
)

NOW = datetime(2026, 8, 28, 3, 0, tzinfo=UTC)
EVIDENCE_HASH = "a" * 64
CONTEXT_HASH = "d" * 64


def make_mapping_snapshot() -> OccupancySnapshot:
    return OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.1,
        origin_m=(0.0, 0.0, 0.0),
        grid_shape=(3, 3, 3),
        free_indices=frozenset({(0, 0, 0)}),
        free_observation_counts=(((0, 0, 0), 3),),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset({(1, 1, 1)}),
        sequence=3,
        created_at_utc=NOW,
        source_view_ids=("view-001", "view-002", "view-003"),
        source_camera_centres_base_m=(
            (0.0, 0.0, 0.0),
            (0.03, 0.0, 0.0),
            (0.06, 0.0, 0.0),
        ),
        source_camera_axes_base=((0.0, 0.0, 1.0),) * 3,
        rebuild_started_at_utc=NOW,
        map_state=OccupancyMapState.MAPPING,
        mapping_context_hash=CONTEXT_HASH,
        parent_evidence_hash="b" * 64,
        state_reason="awaiting validation",
    )


def test_empty_grid_and_out_of_bounds_are_fail_closed_unknown() -> None:
    snapshot = OccupancyGridSpec(0.1, (0, 0, 0), (2, 2, 2)).empty_snapshot(
        created_at_utc=NOW
    )

    assert snapshot.map_state is OccupancyMapState.UNMAPPED
    assert snapshot.state_at_point((0.05, 0.05, 0.05)) is OccupancyState.UNKNOWN
    assert snapshot.state_at_point((-0.01, 0.05, 0.05)) is OccupancyState.UNKNOWN
    result = snapshot.query_sphere((-0.05, 0.05, 0.05), 0.0)
    assert result.state is OccupancyState.UNKNOWN
    assert result.blocked is True
    assert result.unknown_count == 1


def test_state_queries_use_occupied_free_unknown_priority() -> None:
    snapshot = make_mapping_snapshot()

    assert snapshot.state_at_index((0, 0, 0)) is OccupancyState.FREE
    assert snapshot.state_at_index((1, 1, 1)) is OccupancyState.OCCUPIED
    assert snapshot.state_at_index((2, 2, 2)) is OccupancyState.UNKNOWN
    assert snapshot.query_sphere((0.15, 0.15, 0.15), 0.0).state is OccupancyState.OCCUPIED
    assert snapshot.query_sphere((0.05, 0.05, 0.05), 0.0).blocked is False


def test_snapshot_is_immutable_and_hash_is_canonical() -> None:
    first = make_mapping_snapshot()
    second = OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.1,
        origin_m=(0, 0, 0),
        grid_shape=(3, 3, 3),
        free_indices=frozenset(reversed(tuple(first.free_indices))),
        free_observation_counts=tuple(reversed(first.free_observation_counts)),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset(reversed(tuple(first.occupied_indices))),
        sequence=3,
        created_at_utc=NOW,
        source_view_ids=("view-001", "view-002", "view-003"),
        source_camera_centres_base_m=first.source_camera_centres_base_m,
        source_camera_axes_base=first.source_camera_axes_base,
        rebuild_started_at_utc=NOW,
        map_state=OccupancyMapState.MAPPING,
        mapping_context_hash=CONTEXT_HASH,
        parent_evidence_hash="b" * 64,
        state_reason="awaiting validation",
    )

    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64
    assert first.version.startswith("3:")
    with pytest.raises(FrozenInstanceError):
        first.sequence = 7  # type: ignore[misc]


def test_free_vote_counts_and_threshold_are_hash_bound() -> None:
    first = make_mapping_snapshot()
    below_threshold = OccupancySnapshot(
        **{
            **{
                field: getattr(first, field)
                for field in (
                    "frame_id",
                    "voxel_size_m",
                    "origin_m",
                    "grid_shape",
                    "occupied_indices",
                    "sequence",
                    "created_at_utc",
                    "source_view_ids",
                    "source_camera_centres_base_m",
                    "source_camera_axes_base",
                    "rebuild_started_at_utc",
                    "map_state",
                    "mapping_context_hash",
                    "parent_evidence_hash",
                    "quality_evidence_hash",
                    "state_reason",
                )
            },
            "free_indices": frozenset(),
            "free_observation_counts": (((0, 0, 0), 2),),
            "minimum_free_observations": 3,
            "minimum_free_view_translation_m": 0.02,
            "minimum_free_view_direction_deg": 5.0,
        }
    )

    assert below_threshold.state_at_index((0, 0, 0)) is OccupancyState.UNKNOWN
    assert below_threshold.content_hash != first.content_hash


def test_snapshot_rejects_renamed_view_with_near_duplicate_pose() -> None:
    snapshot = make_mapping_snapshot()
    with pytest.raises(ValueError, match="not geometrically independent"):
        replace(
            snapshot,
            source_camera_centres_base_m=(
                snapshot.source_camera_centres_base_m[0],
                snapshot.source_camera_centres_base_m[0],
                snapshot.source_camera_centres_base_m[2],
            ),
            content_hash="",
        )


def test_lifecycle_requires_explicit_quality_evidence_and_preserves_observation_age() -> None:
    mapping = make_mapping_snapshot()

    ready = mapping.promote_to_ready(EVIDENCE_HASH)
    assert ready.sequence == mapping.sequence + 1
    assert ready.map_state is OccupancyMapState.MAP_READY
    assert ready.quality_evidence_hash == EVIDENCE_HASH
    assert ready.created_at_utc == mapping.created_at_utc
    assert ready.content_hash != mapping.content_hash
    assert ready.is_usable_for_preflight(NOW + timedelta(seconds=2), 3.0)
    assert not ready.is_usable_for_preflight(NOW + timedelta(seconds=4), 3.0)

    stale = ready.mark_stale("blade fixture moved")
    assert stale.map_state is OccupancyMapState.STALE
    assert stale.is_stale(NOW, 100.0)
    assert not stale.is_usable_for_preflight(NOW, 100.0)


def test_incomplete_mapping_prefix_binds_evidence_without_becoming_ready() -> None:
    mapping = make_mapping_snapshot()

    bound = mapping.bind_mapping_evidence(EVIDENCE_HASH)

    assert bound.map_state is OccupancyMapState.MAPPING
    assert bound.sequence == mapping.sequence + 1
    assert bound.quality_evidence_hash == EVIDENCE_HASH
    assert not bound.is_usable_for_preflight(NOW, 100.0)


def test_invalid_ready_and_overlapping_states_are_rejected() -> None:
    values = dict(
        frame_id="base",
        voxel_size_m=0.1,
        origin_m=(0.0, 0.0, 0.0),
        grid_shape=(2, 2, 2),
        free_indices=frozenset({(0, 0, 0)}),
        free_observation_counts=(((0, 0, 0), 2),),
        minimum_free_observations=2,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset(),
        sequence=2,
        created_at_utc=NOW,
        source_view_ids=("view-1", "view-2"),
        source_camera_centres_base_m=((0.0, 0.0, 0.0), (0.03, 0.0, 0.0)),
        source_camera_axes_base=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        rebuild_started_at_utc=NOW,
        mapping_context_hash=CONTEXT_HASH,
        parent_evidence_hash="b" * 64,
        state_reason="test",
    )
    with pytest.raises(ValueError, match="map_ready requires"):
        OccupancySnapshot(**values, map_state=OccupancyMapState.MAP_READY)
    with pytest.raises(ValueError, match="must not overlap"):
        OccupancySnapshot(
            **{
                **values,
                "occupied_indices": frozenset({(0, 0, 0)}),
                "map_state": OccupancyMapState.MAPPING,
            }
        )


def test_future_dated_map_fails_closed_as_stale() -> None:
    ready = make_mapping_snapshot().promote_to_ready(EVIDENCE_HASH)

    assert ready.is_stale(NOW - timedelta(milliseconds=1), 10.0)
