from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from biblade_fusion.mapping import OccupancyMapState, OccupancySnapshot
from biblade_fusion.robotics import (
    Cs68PinocchioCollisionChecker,
    OccupancyQueryState,
    OccupancyRobotCollisionChecker,
)
from biblade_fusion.robotics.occupancy_collision import (
    _issue_occupancy_semantic_attestation,
)


@dataclass(frozen=True)
class FakeSphereQuery:
    state: OccupancyQueryState = OccupancyQueryState.FREE
    blocked: bool = False
    occupied_count: int = 0
    unknown_count: int = 0
    free_count: int = 1
    queried_count: int = 1


@dataclass(frozen=True)
class FakeOccupancySnapshot:
    frame_id: str = "base"
    map_state: str = "map_ready"
    sequence: int = 7
    content_hash: str = "a" * 64
    mapping_context_hash: str | None = "d" * 64
    quality_evidence_hash: str | None = "c" * 64
    created_at_utc: datetime = datetime(2026, 8, 28, 0, 0, tzinfo=UTC)
    source_view_ids: tuple[str, ...] = ("view-0001",)
    query: FakeSphereQuery = FakeSphereQuery()
    stale: bool = False
    raises_on_query: bool = False

    def is_stale(self, now_utc: datetime, max_age_s: float) -> bool:
        age_s = (now_utc - self.created_at_utc).total_seconds()
        return self.stale or age_s < 0.0 or age_s > max_age_s

    def is_usable_for_preflight(self, now_utc: datetime, max_age_s: float) -> bool:
        return self.map_state == "map_ready" and not self.is_stale(
            now_utc, max_age_s
        )

    def query_sphere(
        self,
        center_m,
        radius_m: float,
        *,
        unknown_is_occupied: bool = True,
    ) -> FakeSphereQuery:
        del center_m, radius_m
        assert unknown_is_occupied is True
        if self.raises_on_query:
            raise RuntimeError("synthetic occupancy query failure")
        return self.query


class SyntheticContinuousOccupancyChecker(OccupancyRobotCollisionChecker):
    """Test-only backend that explicitly stands in for a future proven sweep."""

    @property
    def continuous_swept_volume_supported(self) -> bool:
        return True

    def check_path(self, *args, **kwargs):
        report = super().check_path(*args, **kwargs)
        if report.status.value != "clear":
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


@pytest.fixture(scope="module")
def occupancy_snapshot() -> OccupancySnapshot:
    shape = (20, 20, 20)
    free_voxels = frozenset(
        (x, y, z)
        for x in range(shape[0])
        for y in range(shape[1])
        for z in range(shape[2])
    )
    return OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.5,
        origin_m=(-5.0, -5.0, -5.0),
        grid_shape=shape,
        free_indices=free_voxels,
        free_observation_counts=tuple(
            (index, 3) for index in sorted(free_voxels)
        ),
        minimum_free_observations=3,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset(),
        sequence=7,
        created_at_utc=datetime(2026, 8, 28, 0, 0, tzinfo=UTC),
        source_view_ids=("view-0001", "view-0002", "view-0003"),
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


@pytest.fixture(scope="module")
def occupancy_checker(
    checker: Cs68PinocchioCollisionChecker,
    occupancy_snapshot: OccupancySnapshot,
) -> OccupancyRobotCollisionChecker:
    robot_geometry_hash = checker.robot_geometry_hash
    assert robot_geometry_hash is not None
    attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256="e" * 64,
        snapshot=occupancy_snapshot,
        robot_geometry_hash=robot_geometry_hash,
    )
    return SyntheticContinuousOccupancyChecker(
        checker,
        lambda: occupancy_snapshot,
        maximum_map_age_s=30.0,
        semantic_attestation=attestation,
        utc_clock=lambda: datetime(2026, 8, 28, 0, 0, 1, tzinfo=UTC),
    )
