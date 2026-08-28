from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.mapping.integrator import (
    DepthIntegrationConfig,
    DepthIntegrationError,
    DepthRayIntegrator,
)
from biblade_fusion.mapping.occupancy import (
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancyState,
)

NOW = datetime(2026, 8, 28, 3, 30, tzinfo=UTC)
EVIDENCE_HASH = "b" * 64
CONTEXT_HASH = "d" * 64


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(1, 1, 1.0, 1.0, 0.0, 0.0, "none", ())


def pose(
    *,
    x_offset_m: float = 0.0,
    direction_deg: float = 0.0,
) -> PoseSE3:
    angle = np.deg2rad(direction_deg)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return PoseSE3.from_rotation_translation(
        "base",
        "left_rectified",
        rotation,
        [0.25 + x_offset_m, 0.25, 0.25],
    )


def integrator() -> DepthRayIntegrator:
    return DepthRayIntegrator(
        OccupancyGridSpec(0.5, (0.0, 0.0, 0.0), (2, 2, 8)),
        DepthIntegrationConfig(
            minimum_depth_m=0.1,
            maximum_depth_m=3.0,
            pixel_stride=1,
            minimum_valid_rays=1,
            free_space_margin_m=0.0,
        ),
        mapping_context_hash=CONTEXT_HASH,
    )


def test_one_depth_view_keeps_traversed_segment_unknown() -> None:
    snapshot = integrator().integrate(
        None,
        np.array([[1.25]], dtype=np.float32),
        intrinsics(),
        pose(),
        source_view_id="front-001",
        observed_at_utc=NOW,
    )

    assert snapshot.map_state is OccupancyMapState.MAPPING
    assert snapshot.sequence == 1
    assert snapshot.source_view_ids == ("front-001",)
    assert snapshot.occupied_indices == frozenset({(0, 0, 3)})
    assert snapshot.free_indices == frozenset()
    assert snapshot.free_observation_counts == (
        ((0, 0, 0), 1),
        ((0, 0, 1), 1),
        ((0, 0, 2), 1),
    )
    assert snapshot.state_at_index((0, 0, 1)) is OccupancyState.UNKNOWN


def test_many_rays_in_one_frame_cast_only_one_vote_per_voxel() -> None:
    mapper = integrator()
    two_pixel_intrinsics = CameraIntrinsics(
        2,
        1,
        1000.0,
        1000.0,
        0.0,
        0.0,
        "none",
        (),
    )
    snapshot = mapper.integrate(
        None,
        [[1.25, 1.25]],
        two_pixel_intrinsics,
        pose(),
        source_view_id="dense-front-001",
        observed_at_utc=NOW,
    )

    assert set(dict(snapshot.free_observation_counts).values()) == {1}
    assert snapshot.free_indices == frozenset()


def test_three_independent_views_promote_traversed_voxels_to_free() -> None:
    mapper = integrator()
    snapshot = None
    for index in range(3):
        mapping = mapper.integrate(
            snapshot,
            [[1.25]],
            intrinsics(),
            pose(x_offset_m=0.03 * index),
            source_view_id=f"front-{index + 1:03d}",
            observed_at_utc=NOW,
        )
        snapshot = mapping.bind_mapping_evidence(chr(ord("b") + index) * 64)

    assert mapping.free_indices == frozenset(
        {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
    )
    assert dict(mapping.free_observation_counts)[(0, 0, 1)] == 3
    assert mapping.state_at_index((0, 0, 1)) is OccupancyState.FREE


def test_occupied_observation_overrides_thresholded_free_evidence() -> None:
    mapper = integrator()
    snapshot = None
    for index in range(3):
        mapping = mapper.integrate(
            snapshot,
            [[1.75]],
            intrinsics(),
            pose(x_offset_m=0.03 * index),
            source_view_id=f"long-{index + 1:03d}",
            observed_at_utc=NOW,
        )
        snapshot = mapping.bind_mapping_evidence(chr(ord("b") + index) * 64)

    assert mapping.state_at_index((0, 0, 3)) is OccupancyState.FREE
    occupied = mapper.integrate(
        snapshot,
        [[1.25]],
        intrinsics(),
        pose(x_offset_m=0.09),
        source_view_id="short-004",
        observed_at_utc=NOW,
    )

    assert occupied.state_at_index((0, 0, 3)) is OccupancyState.OCCUPIED
    assert (0, 0, 3) not in occupied.free_indices
    assert dict(occupied.free_observation_counts)[(0, 0, 3)] == 3


def test_occupied_wins_over_later_free_ray_and_ready_returns_to_mapping() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(x_offset_m=0.03),
        source_view_id="front-001",
        observed_at_utc=NOW,
    ).promote_to_ready(EVIDENCE_HASH)

    second = mapper.integrate(
        first,
        [[1.75]],
        intrinsics(),
        pose(),
        source_view_id="front-002",
        observed_at_utc=NOW,
    )

    assert second.map_state is OccupancyMapState.MAPPING
    assert second.quality_evidence_hash is None
    assert second.occupied_indices == frozenset({(0, 0, 3), (0, 0, 4)})
    assert (0, 0, 3) not in second.free_indices


def test_minimum_free_observations_must_prevent_single_view_clearing() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DepthIntegrationConfig(minimum_free_observations=1)


def test_same_pose_with_new_source_id_is_rejected_without_votes() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="first",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)
    before_counts = first.free_observation_counts

    with pytest.raises(DepthIntegrationError, match="not geometrically independent"):
        mapper.integrate(
            first,
            [[1.25]],
            intrinsics(),
            pose(),
            source_view_id="renamed-duplicate",
            observed_at_utc=NOW,
        )

    assert first.free_observation_counts == before_counts
    assert first.source_view_ids == ("first",)


def test_near_duplicate_pose_is_rejected_without_votes() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="first",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)

    with pytest.raises(DepthIntegrationError, match="not geometrically independent"):
        mapper.integrate(
            first,
            [[1.25]],
            intrinsics(),
            pose(x_offset_m=0.001, direction_deg=1.0),
            source_view_id="near-duplicate",
            observed_at_utc=NOW,
        )

    assert dict(first.free_observation_counts)[(0, 0, 1)] == 1
    assert first.source_view_ids == ("first",)


def test_sufficient_translation_accepts_new_support_view() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="first",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)

    second = mapper.integrate(
        first,
        [[1.25]],
        intrinsics(),
        pose(x_offset_m=0.03),
        source_view_id="translated",
        observed_at_utc=NOW,
    )

    assert second.source_view_ids == ("first", "translated")


def test_sufficient_optical_axis_angle_accepts_new_support_view() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="first",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)

    second = mapper.integrate(
        first,
        [[1.25]],
        intrinsics(),
        pose(direction_deg=6.0),
        source_view_id="angled",
        observed_at_utc=NOW,
    )

    assert second.source_view_ids == ("first", "angled")


def test_map_age_uses_full_rebuild_start_not_latest_frame() -> None:
    mapper = integrator()
    first = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="first",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)
    second = mapper.integrate(
        first,
        [[1.25]],
        intrinsics(),
        pose(x_offset_m=0.03),
        source_view_id="second",
        observed_at_utc=NOW + timedelta(seconds=4),
    ).promote_to_ready("c" * 64)

    assert second.created_at_utc == NOW + timedelta(seconds=4)
    assert second.rebuild_started_at_utc == NOW
    assert second.is_stale(NOW + timedelta(seconds=6), 5.0)


def test_invalid_and_self_masked_depth_never_clears_unknown_space() -> None:
    mapper = integrator()

    with pytest.raises(DepthIntegrationError, match="0 valid sampled rays"):
        mapper.integrate(
            None,
            [[np.nan]],
            intrinsics(),
            pose(),
            source_view_id="invalid",
            observed_at_utc=NOW,
        )
    with pytest.raises(DepthIntegrationError, match="0 valid sampled rays"):
        mapper.integrate(
            None,
            [[1.0]],
            intrinsics(),
            pose(),
            valid_mask=[[False]],
            source_view_id="self-masked",
            observed_at_utc=NOW,
        )


def test_integration_rejects_duplicate_views_and_disconnected_frames() -> None:
    mapper = integrator()
    snapshot = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="same",
        observed_at_utc=NOW,
    ).bind_mapping_evidence(EVIDENCE_HASH)

    with pytest.raises(DepthIntegrationError, match="already integrated"):
        mapper.integrate(
            snapshot,
            [[1.25]],
            intrinsics(),
            pose(),
            source_view_id="same",
            observed_at_utc=NOW,
        )
    bad_pose = PoseSE3.identity("world", "left_rectified")
    with pytest.raises(DepthIntegrationError, match="base_T_left_rectified"):
        mapper.integrate(
            snapshot,
            [[1.25]],
            intrinsics(),
            bad_pose,
            source_view_id="other",
            observed_at_utc=NOW,
        )
    wrong_camera_frame = PoseSE3.identity("base", "left_ir")
    with pytest.raises(DepthIntegrationError, match="base_T_left_rectified"):
        mapper.integrate(
            snapshot,
            [[1.25]],
            intrinsics(),
            wrong_camera_frame,
            source_view_id="other-camera-frame",
            observed_at_utc=NOW,
        )


def test_same_observation_produces_same_content_hash() -> None:
    mapper = integrator()
    kwargs = dict(
        snapshot=None,
        depth_m=[[1.25]],
        intrinsics=intrinsics(),
        base_t_camera=pose(),
        source_view_id="deterministic",
        observed_at_utc=NOW,
    )

    assert mapper.integrate(**kwargs).content_hash == mapper.integrate(**kwargs).content_hash


def test_context_mismatch_and_unbound_prefix_fail_closed() -> None:
    mapper = integrator()
    unbound = mapper.integrate(
        None,
        [[1.25]],
        intrinsics(),
        pose(),
        source_view_id="front-001",
        observed_at_utc=NOW,
    )

    with pytest.raises(DepthIntegrationError, match="no bound evidence chain"):
        mapper.integrate(
            unbound,
            [[1.25]],
            intrinsics(),
            pose(),
            source_view_id="front-002",
            observed_at_utc=NOW,
        )

    bound = unbound.bind_mapping_evidence(EVIDENCE_HASH)
    other = DepthRayIntegrator(
        mapper.grid,
        mapper.config,
        mapping_context_hash="e" * 64,
    )
    with pytest.raises(DepthIntegrationError, match="mapping context"):
        other.integrate(
            bound,
            [[1.25]],
            intrinsics(),
            pose(),
            source_view_id="front-002",
            observed_at_utc=NOW,
        )
