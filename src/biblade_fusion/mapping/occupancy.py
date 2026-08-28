"""Immutable, fail-closed three-state voxel occupancy contracts.

The safety map is expressed in the robot ``base`` frame and in metres.  Unknown
space is implicit: every in-bounds cell that is neither explicitly free nor
explicitly occupied is unknown.  Points outside the configured grid are also
unknown, so incomplete maps fail closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

VoxelIndex = tuple[int, int, int]
FreeObservationCount = tuple[VoxelIndex, int]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class OccupancyState(StrEnum):
    """State of one voxel in the conservative environment map."""

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class OccupancyMapState(StrEnum):
    """Explicit map lifecycle used by motion preflight."""

    UNMAPPED = "unmapped"
    MAPPING = "mapping"
    MAP_READY = "map_ready"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class SphereQueryResult:
    """Conservative result of querying a spherical collision primitive."""

    state: OccupancyState
    blocked: bool
    occupied_count: int
    unknown_count: int
    free_count: int
    queried_count: int

    def __post_init__(self) -> None:
        counts = (self.occupied_count, self.unknown_count, self.free_count, self.queried_count)
        if any(value < 0 for value in counts):
            raise ValueError("Sphere query counts must be non-negative")
        if self.occupied_count + self.unknown_count + self.free_count != self.queried_count:
            raise ValueError("Sphere query state counts must sum to queried_count")


@dataclass(frozen=True, slots=True)
class OccupancyGridSpec:
    """Fixed metric geometry of an occupancy grid."""

    voxel_size_m: float
    origin_m: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    frame_id: str = "base"

    def __post_init__(self) -> None:
        voxel_size = float(self.voxel_size_m)
        origin = _metric_triplet(self.origin_m, name="origin_m")
        shape = _shape_triplet(self.grid_shape)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if self.frame_id != "base":
            raise ValueError("Safety occupancy grid frame_id must be 'base'")
        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "grid_shape", shape)

    @property
    def bounds_max_m(self) -> tuple[float, float, float]:
        return tuple(
            self.origin_m[axis] + self.grid_shape[axis] * self.voxel_size_m
            for axis in range(3)
        )

    def empty_snapshot(
        self,
        *,
        created_at_utc: datetime,
        minimum_free_observations: int = 3,
        minimum_free_view_translation_m: float = 0.02,
        minimum_free_view_direction_deg: float = 5.0,
        reason: str = "no depth observations integrated",
    ) -> OccupancySnapshot:
        """Create an all-unknown, explicitly unusable initial snapshot."""

        return OccupancySnapshot(
            frame_id=self.frame_id,
            voxel_size_m=self.voxel_size_m,
            origin_m=self.origin_m,
            grid_shape=self.grid_shape,
            free_indices=frozenset(),
            free_observation_counts=(),
            minimum_free_observations=minimum_free_observations,
            minimum_free_view_translation_m=minimum_free_view_translation_m,
            minimum_free_view_direction_deg=minimum_free_view_direction_deg,
            occupied_indices=frozenset(),
            sequence=0,
            created_at_utc=created_at_utc,
            source_view_ids=(),
            source_camera_centres_base_m=(),
            source_camera_axes_base=(),
            rebuild_started_at_utc=None,
            map_state=OccupancyMapState.UNMAPPED,
            state_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class OccupancySnapshot:
    """Immutable version of a conservative, sparse three-state voxel grid.

    ``free_observation_counts`` stores at most one free-space vote per source
    view and voxel. ``free_indices`` is the strictly derived subset whose vote
    count reaches ``minimum_free_observations`` and which is not occupied. All
    other cells, including out-of-bounds cells, are unknown. A snapshot becomes
    usable by motion preflight only after :meth:`promote_to_ready` records a
    quality-evidence hash. ``rebuild_started_at_utc`` is the safety-freshness
    reference for the entire rebuild cycle; adding a recent frame updates
    ``created_at_utc`` but deliberately cannot make older FREE evidence fresh.
    """

    frame_id: str
    voxel_size_m: float
    origin_m: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    free_indices: frozenset[VoxelIndex]
    free_observation_counts: tuple[FreeObservationCount, ...]
    minimum_free_observations: int
    minimum_free_view_translation_m: float
    minimum_free_view_direction_deg: float
    occupied_indices: frozenset[VoxelIndex]
    sequence: int
    created_at_utc: datetime
    source_view_ids: tuple[str, ...]
    source_camera_centres_base_m: tuple[tuple[float, float, float], ...]
    source_camera_axes_base: tuple[tuple[float, float, float], ...]
    rebuild_started_at_utc: datetime | None
    map_state: OccupancyMapState = OccupancyMapState.UNMAPPED
    mapping_context_hash: str | None = None
    parent_evidence_hash: str | None = None
    quality_evidence_hash: str | None = None
    state_reason: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.frame_id != "base":
            raise ValueError("Safety occupancy snapshot frame_id must be 'base'")
        voxel_size = float(self.voxel_size_m)
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        origin = _metric_triplet(self.origin_m, name="origin_m")
        shape = _shape_triplet(self.grid_shape)
        free = _normalise_indices(self.free_indices, grid_shape=shape, name="free_indices")
        minimum_free_observations = _minimum_free_observations(
            self.minimum_free_observations
        )
        minimum_view_translation = _positive_finite(
            self.minimum_free_view_translation_m,
            name="minimum_free_view_translation_m",
        )
        minimum_view_direction = _direction_threshold_deg(
            self.minimum_free_view_direction_deg
        )
        free_observation_counts = _normalise_free_observation_counts(
            self.free_observation_counts,
            grid_shape=shape,
        )
        occupied = _normalise_indices(
            self.occupied_indices,
            grid_shape=shape,
            name="occupied_indices",
        )
        if free & occupied:
            raise ValueError("free_indices and occupied_indices must not overlap")
        if isinstance(self.sequence, bool) or int(self.sequence) != self.sequence:
            raise ValueError("sequence must be a non-negative integer")
        sequence = int(self.sequence)
        if sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        created = _utc_datetime(self.created_at_utc)
        source_views = tuple(str(value).strip() for value in self.source_view_ids)
        if any(not value for value in source_views):
            raise ValueError("source_view_ids must contain non-empty identifiers")
        if len(set(source_views)) != len(source_views):
            raise ValueError("source_view_ids must not contain duplicates")
        camera_centres = tuple(
            _metric_triplet(value, name="source_camera_centres_base_m entry")
            for value in self.source_camera_centres_base_m
        )
        camera_axes = tuple(
            _unit_axis(value, name="source_camera_axes_base entry")
            for value in self.source_camera_axes_base
        )
        if not (
            len(source_views) == len(camera_centres) == len(camera_axes)
        ):
            raise ValueError(
                "source camera pose evidence must align one-to-one with source_view_ids"
            )
        _validate_geometrically_independent_views(
            camera_centres,
            camera_axes,
            minimum_translation_m=minimum_view_translation,
            minimum_direction_deg=minimum_view_direction,
        )
        rebuild_started = (
            None
            if self.rebuild_started_at_utc is None
            else _utc_datetime(self.rebuild_started_at_utc)
        )
        if rebuild_started is not None and rebuild_started > created:
            raise ValueError("rebuild_started_at_utc cannot follow created_at_utc")
        if any(count > len(source_views) for _, count in free_observation_counts):
            raise ValueError(
                "free observation count cannot exceed the number of source views"
            )
        expected_free = frozenset(
            index
            for index, count in free_observation_counts
            if count >= minimum_free_observations and index not in occupied
        )
        if free != expected_free:
            raise ValueError(
                "free_indices must exactly match thresholded free observation counts"
            )
        map_state = OccupancyMapState(self.map_state)
        context_hash = self.mapping_context_hash
        parent_evidence = self.parent_evidence_hash
        evidence = self.quality_evidence_hash
        for field_name, digest in (
            ("mapping_context_hash", context_hash),
            ("parent_evidence_hash", parent_evidence),
            ("quality_evidence_hash", evidence),
        ):
            if digest is not None and not _SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if map_state is not OccupancyMapState.UNMAPPED and context_hash is None:
            raise ValueError(f"{map_state.value} requires mapping_context_hash")
        if map_state is OccupancyMapState.UNMAPPED and (
            context_hash is not None or parent_evidence is not None or evidence is not None
        ):
            raise ValueError("UNMAPPED must not carry mapping or evidence hashes")
        if map_state in {OccupancyMapState.MAP_READY, OccupancyMapState.STALE} and evidence is None:
            raise ValueError(f"{map_state.value} requires quality_evidence_hash")
        known_count = len(free) + len(occupied)
        if map_state is OccupancyMapState.UNMAPPED and (
            known_count != 0
            or free_observation_counts
            or source_views
            or camera_centres
            or camera_axes
            or rebuild_started is not None
            or sequence != 0
        ):
            raise ValueError("UNMAPPED must be sequence zero with no observations")
        if map_state is not OccupancyMapState.UNMAPPED and (
            (not free_observation_counts and not occupied)
            or not source_views
            or rebuild_started is None
            or sequence == 0
        ):
            raise ValueError(f"{map_state.value} requires versioned occupancy observations")
        if len(source_views) <= 1 and parent_evidence is not None:
            raise ValueError("First occupancy observation must not carry parent_evidence_hash")
        if len(source_views) > 1 and parent_evidence is None:
            raise ValueError("Multi-view occupancy requires parent_evidence_hash")
        reason = str(self.state_reason).strip()
        if not reason:
            raise ValueError("state_reason must be non-empty")

        object.__setattr__(self, "voxel_size_m", voxel_size)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "free_indices", free)
        object.__setattr__(self, "free_observation_counts", free_observation_counts)
        object.__setattr__(
            self,
            "minimum_free_observations",
            minimum_free_observations,
        )
        object.__setattr__(
            self,
            "minimum_free_view_translation_m",
            minimum_view_translation,
        )
        object.__setattr__(
            self,
            "minimum_free_view_direction_deg",
            minimum_view_direction,
        )
        object.__setattr__(self, "occupied_indices", occupied)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "source_view_ids", source_views)
        object.__setattr__(self, "source_camera_centres_base_m", camera_centres)
        object.__setattr__(self, "source_camera_axes_base", camera_axes)
        object.__setattr__(self, "rebuild_started_at_utc", rebuild_started)
        object.__setattr__(self, "map_state", map_state)
        object.__setattr__(self, "state_reason", reason)

        expected_hash = compute_content_hash(self)
        if self.content_hash:
            if not _SHA256_PATTERN.fullmatch(self.content_hash):
                raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
            if self.content_hash != expected_hash:
                raise ValueError("Occupancy snapshot content_hash mismatch")
        else:
            object.__setattr__(self, "content_hash", expected_hash)

    @property
    def version(self) -> str:
        """Human-readable version while the full hash remains the authority."""

        return f"{self.sequence}:{self.content_hash[:12]}"

    @property
    def known_voxel_count(self) -> int:
        return len(self.free_indices) + len(self.occupied_indices)

    @property
    def observed_voxel_count(self) -> int:
        return len(
            {index for index, _ in self.free_observation_counts}
            | set(self.occupied_indices)
        )

    @property
    def total_voxel_count(self) -> int:
        return math.prod(self.grid_shape)

    @property
    def unknown_voxel_count(self) -> int:
        return self.total_voxel_count - self.known_voxel_count

    @property
    def bounds_max_m(self) -> tuple[float, float, float]:
        return tuple(
            self.origin_m[axis] + self.grid_shape[axis] * self.voxel_size_m
            for axis in range(3)
        )

    def geometry_spec(self) -> OccupancyGridSpec:
        return OccupancyGridSpec(
            frame_id=self.frame_id,
            voxel_size_m=self.voxel_size_m,
            origin_m=self.origin_m,
            grid_shape=self.grid_shape,
        )

    def index_in_bounds(self, index: Sequence[int]) -> bool:
        ix, iy, iz = _index_triplet(index)
        sx, sy, sz = self.grid_shape
        return 0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz

    def index_at_point(self, point_m: Sequence[float]) -> VoxelIndex | None:
        point = _metric_triplet(point_m, name="point_m")
        index = tuple(
            math.floor((point[axis] - self.origin_m[axis]) / self.voxel_size_m)
            for axis in range(3)
        )
        return index if self.index_in_bounds(index) else None

    def state_at_index(self, index: Sequence[int]) -> OccupancyState:
        normalised = _index_triplet(index)
        if not self.index_in_bounds(normalised):
            return OccupancyState.UNKNOWN
        if normalised in self.occupied_indices:
            return OccupancyState.OCCUPIED
        if normalised in self.free_indices:
            return OccupancyState.FREE
        return OccupancyState.UNKNOWN

    def state_at_point(self, point_m: Sequence[float]) -> OccupancyState:
        index = self.index_at_point(point_m)
        if index is None:
            return OccupancyState.UNKNOWN
        return self.state_at_index(index)

    def query_sphere(
        self,
        center_m: Sequence[float],
        radius_m: float,
        *,
        unknown_is_occupied: bool = True,
    ) -> SphereQueryResult:
        """Query every voxel intersected by a metric sphere.

        Out-of-grid cells count as unknown.  ``unknown_is_occupied`` defaults to
        ``True`` and must remain enabled for safety preflight; the opt-out exists
        only for inspection and visualisation.
        """

        center = _metric_triplet(center_m, name="center_m")
        radius = float(radius_m)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("radius_m must be finite and non-negative")
        occupied = 0
        unknown = 0
        free = 0
        for index in sphere_intersecting_indices(
            center_m=center,
            radius_m=radius,
            origin_m=self.origin_m,
            voxel_size_m=self.voxel_size_m,
        ):
            state = self.state_at_index(index)
            if state is OccupancyState.OCCUPIED:
                occupied += 1
            elif state is OccupancyState.UNKNOWN:
                unknown += 1
            else:
                free += 1
        state = (
            OccupancyState.OCCUPIED
            if occupied
            else OccupancyState.UNKNOWN
            if unknown
            else OccupancyState.FREE
        )
        return SphereQueryResult(
            state=state,
            blocked=bool(occupied or (unknown_is_occupied and unknown)),
            occupied_count=occupied,
            unknown_count=unknown,
            free_count=free,
            queried_count=occupied + unknown + free,
        )

    def is_stale(self, now_utc: datetime, max_age_s: float) -> bool:
        """Return whether state or full-rebuild-cycle age makes the map stale."""

        maximum_age = float(max_age_s)
        if not math.isfinite(maximum_age) or maximum_age < 0.0:
            raise ValueError("max_age_s must be finite and non-negative")
        now = _utc_datetime(now_utc)
        if self.rebuild_started_at_utc is None:
            return True
        age_s = (now - self.rebuild_started_at_utc).total_seconds()
        return self.map_state is OccupancyMapState.STALE or age_s < 0.0 or age_s > maximum_age

    def is_usable_for_preflight(self, now_utc: datetime, max_age_s: float) -> bool:
        """Only validated, fresh MAP_READY snapshots are safety evidence."""

        return self.map_state is OccupancyMapState.MAP_READY and not self.is_stale(
            now_utc,
            max_age_s,
        )

    def promote_to_ready(
        self,
        quality_evidence_hash: str,
        *,
        reason: str = "self-filter and depth quality gates passed",
    ) -> OccupancySnapshot:
        """Create a validated MAP_READY version without refreshing observation age."""

        if self.map_state is not OccupancyMapState.MAPPING:
            raise ValueError("Only MAPPING snapshots can be promoted to MAP_READY")
        if self.quality_evidence_hash is not None:
            raise ValueError("MAPPING snapshot already has bound quality evidence")
        if self.observed_voxel_count == 0 or not self.source_view_ids:
            raise ValueError("Cannot promote an occupancy snapshot without observations")
        return replace(
            self,
            sequence=self.sequence + 1,
            map_state=OccupancyMapState.MAP_READY,
            quality_evidence_hash=quality_evidence_hash,
            state_reason=reason,
            content_hash="",
        )

    def bind_mapping_evidence(
        self,
        quality_evidence_hash: str,
        *,
        reason: str = "evidence chain bound; awaiting minimum source views",
    ) -> OccupancySnapshot:
        """Bind verified evidence to an incomplete MAPPING snapshot.

        This makes a multi-frame mapping prefix independently auditable without
        making it usable for motion preflight.
        """

        if self.map_state is not OccupancyMapState.MAPPING:
            raise ValueError("Only MAPPING snapshots can bind mapping evidence")
        if self.quality_evidence_hash is not None:
            raise ValueError("MAPPING snapshot already has bound quality evidence")
        return replace(
            self,
            sequence=self.sequence + 1,
            quality_evidence_hash=quality_evidence_hash,
            state_reason=reason,
            content_hash="",
        )

    def mark_stale(self, reason: str) -> OccupancySnapshot:
        """Create an explicit STALE version; stale maps cannot be re-promoted."""

        if self.map_state is not OccupancyMapState.MAP_READY:
            raise ValueError("Only MAP_READY snapshots can transition to STALE")
        return replace(
            self,
            sequence=self.sequence + 1,
            map_state=OccupancyMapState.STALE,
            state_reason=reason,
            content_hash="",
        )


def compute_content_hash(snapshot: OccupancySnapshot) -> str:
    """Return a deterministic SHA-256 over all authoritative snapshot content."""

    payload = snapshot_hash_payload(snapshot)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def snapshot_hash_payload(snapshot: OccupancySnapshot) -> dict[str, Any]:
    """Build the canonical payload, deliberately excluding ``content_hash``."""

    return {
        "frame_id": snapshot.frame_id,
        "voxel_size_m": snapshot.voxel_size_m,
        "origin_m": list(snapshot.origin_m),
        "grid_shape": list(snapshot.grid_shape),
        "free_indices": [list(index) for index in sorted(snapshot.free_indices)],
        "free_observation_counts": [
            [*index, count] for index, count in snapshot.free_observation_counts
        ],
        "minimum_free_observations": snapshot.minimum_free_observations,
        "minimum_free_view_translation_m": (
            snapshot.minimum_free_view_translation_m
        ),
        "minimum_free_view_direction_deg": (
            snapshot.minimum_free_view_direction_deg
        ),
        "occupied_indices": [list(index) for index in sorted(snapshot.occupied_indices)],
        "sequence": snapshot.sequence,
        "created_at_utc": snapshot.created_at_utc.isoformat(),
        "source_view_ids": list(snapshot.source_view_ids),
        "source_camera_centres_base_m": [
            list(value) for value in snapshot.source_camera_centres_base_m
        ],
        "source_camera_axes_base": [
            list(value) for value in snapshot.source_camera_axes_base
        ],
        "rebuild_started_at_utc": (
            snapshot.rebuild_started_at_utc.isoformat()
            if snapshot.rebuild_started_at_utc is not None
            else None
        ),
        "map_state": snapshot.map_state.value,
        "mapping_context_hash": snapshot.mapping_context_hash,
        "parent_evidence_hash": snapshot.parent_evidence_hash,
        "quality_evidence_hash": snapshot.quality_evidence_hash,
        "state_reason": snapshot.state_reason,
    }


def sphere_intersecting_indices(
    *,
    center_m: Sequence[float],
    radius_m: float,
    origin_m: Sequence[float],
    voxel_size_m: float,
) -> Iterable[VoxelIndex]:
    """Yield all (including out-of-grid) voxel cubes intersecting a sphere."""

    center = _metric_triplet(center_m, name="center_m")
    origin = _metric_triplet(origin_m, name="origin_m")
    radius = float(radius_m)
    voxel_size = float(voxel_size_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be finite and non-negative")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    minimum = tuple(
        math.floor((center[axis] - radius - origin[axis]) / voxel_size)
        for axis in range(3)
    )
    maximum = tuple(
        math.floor((center[axis] + radius - origin[axis]) / voxel_size)
        for axis in range(3)
    )
    radius_squared = radius * radius
    for ix in range(minimum[0], maximum[0] + 1):
        for iy in range(minimum[1], maximum[1] + 1):
            for iz in range(minimum[2], maximum[2] + 1):
                index = (ix, iy, iz)
                lower = tuple(origin[axis] + index[axis] * voxel_size for axis in range(3))
                upper = tuple(value + voxel_size for value in lower)
                squared_distance = 0.0
                for value, low, high in zip(center, lower, upper, strict=True):
                    if value < low:
                        squared_distance += (low - value) ** 2
                    elif value > high:
                        squared_distance += (value - high) ** 2
                if squared_distance <= radius_squared:
                    yield index


def _metric_triplet(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _shape_triplet(values: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("grid_shape must contain exactly three values")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("grid_shape values must be positive integers")
        result.append(int(value))
    return (result[0], result[1], result[2])


def _index_triplet(values: Sequence[int]) -> VoxelIndex:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("Voxel index must contain exactly three values")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError("Voxel index values must be integers")
        result.append(int(value))
    return (result[0], result[1], result[2])


def _normalise_indices(
    values: Iterable[Sequence[int]],
    *,
    grid_shape: tuple[int, int, int],
    name: str,
) -> frozenset[VoxelIndex]:
    result = frozenset(_index_triplet(value) for value in values)
    sx, sy, sz = grid_shape
    if any(not (0 <= ix < sx and 0 <= iy < sy and 0 <= iz < sz) for ix, iy, iz in result):
        raise ValueError(f"{name} contains an out-of-bounds voxel")
    return result


def _minimum_free_observations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("minimum_free_observations must be an integer of at least two")
    return value


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _direction_threshold_deg(value: float) -> float:
    result = _positive_finite(value, name="minimum_free_view_direction_deg")
    if result > 180.0:
        raise ValueError("minimum_free_view_direction_deg must not exceed 180")
    return result


def _unit_axis(
    values: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float]:
    axis = _metric_triplet(values, name=name)
    norm = math.sqrt(sum(value * value for value in axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be a unit vector")
    return axis


def _validate_geometrically_independent_views(
    centres: tuple[tuple[float, float, float], ...],
    axes: tuple[tuple[float, float, float], ...],
    *,
    minimum_translation_m: float,
    minimum_direction_deg: float,
) -> None:
    for current_index in range(len(centres)):
        for previous_index in range(current_index):
            translation = math.sqrt(
                sum(
                    (centres[current_index][axis] - centres[previous_index][axis])
                    ** 2
                    for axis in range(3)
                )
            )
            cosine = max(
                -1.0,
                min(
                    1.0,
                    sum(
                        axes[current_index][axis] * axes[previous_index][axis]
                        for axis in range(3)
                    ),
                ),
            )
            direction = math.degrees(math.acos(cosine))
            if (
                translation < minimum_translation_m
                and direction < minimum_direction_deg
            ):
                raise ValueError(
                    "source camera poses are not geometrically independent"
                )


def _normalise_free_observation_counts(
    values: Iterable[tuple[Sequence[int], int]],
    *,
    grid_shape: tuple[int, int, int],
) -> tuple[FreeObservationCount, ...]:
    result: dict[VoxelIndex, int] = {}
    for raw_index, raw_count in values:
        index = _index_triplet(raw_index)
        if index in result:
            raise ValueError("free_observation_counts contains duplicate voxels")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise ValueError("free observation counts must be positive integers")
        result[index] = raw_count
    _normalise_indices(
        result,
        grid_shape=grid_shape,
        name="free_observation_counts",
    )
    return tuple(sorted(result.items()))


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at_utc/now_utc must be timezone-aware")
    return value.astimezone(UTC)
