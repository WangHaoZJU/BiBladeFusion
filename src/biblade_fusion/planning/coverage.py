"""Bilateral surface-coverage accounting from pose-registered blade point clouds."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CoverageConfig
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning.filtering import CandidateStatus, EvaluatedCandidate, FilteredViewPlan
from biblade_fusion.planning.views import BilateralViewPlan, BladeSide


class CoverageError(ValueError):
    """Coverage evidence is inconsistent, empty, or not registered in the base frame."""


def coverage_observation_id(
    source_session: str | Path,
    view_id: str,
    sequence_index: int,
    frame_number: int,
) -> str:
    """Build a stable identity so one physical frame cannot be counted twice."""

    session = str(Path(source_session).resolve())
    session_key = hashlib.sha256(session.encode("utf-8")).hexdigest()[:8]
    return f"{Path(session).name}-{session_key}:{sequence_index:04d}:{frame_number}:{view_id}"


@dataclass(frozen=True, slots=True)
class PatchCoverage:
    patch_id: str
    side: BladeSide
    row: int
    column: int
    bin_point_counts: NDArray[np.int64]
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = np.array(self.bin_point_counts, dtype=np.int64, copy=True)
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
            raise ValueError("Patch coverage bins must be a square matrix")
        if np.any(counts < 0):
            raise ValueError("Patch coverage bin counts must be non-negative")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("Patch coverage observation IDs must be unique")
        counts.setflags(write=False)
        object.__setattr__(self, "bin_point_counts", counts)

    @property
    def point_count(self) -> int:
        return int(self.bin_point_counts.sum())

    def occupied_fraction(self, minimum_points_per_bin: int) -> float:
        return float(np.mean(self.bin_point_counts >= minimum_points_per_bin))


@dataclass(frozen=True, slots=True)
class CoverageLedger:
    patches: tuple[PatchCoverage, ...]
    observation_ids: tuple[str, ...]
    config: CoverageConfig
    rows: int
    columns: int

    def __post_init__(self) -> None:
        expected = 2 * self.rows * self.columns
        if len(self.patches) != expected:
            raise ValueError(f"Expected {expected} bilateral patch coverage entries")
        if len({patch.patch_id for patch in self.patches}) != len(self.patches):
            raise ValueError("Coverage patch IDs must be unique")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("Coverage ledger observation IDs must be unique")

    def patch(self, patch_id: str) -> PatchCoverage:
        matches = tuple(patch for patch in self.patches if patch.patch_id == patch_id)
        if len(matches) != 1:
            raise KeyError(f"Expected one coverage patch {patch_id!r}, found {len(matches)}")
        return matches[0]

    def is_complete(self, patch_id: str) -> bool:
        return (
            self.patch(patch_id).occupied_fraction(self.config.minimum_points_per_bin)
            >= self.config.completed_fraction
        )

    @property
    def completed_patch_ids(self) -> tuple[str, ...]:
        return tuple(patch.patch_id for patch in self.patches if self.is_complete(patch.patch_id))

    def completion_fraction(self, side: BladeSide | None = None) -> float:
        selected = tuple(
            patch for patch in self.patches if side is None or patch.side is side
        )
        return float(sum(self.is_complete(patch.patch_id) for patch in selected) / len(selected))


@dataclass(frozen=True, slots=True)
class CoverageDrivenViewPlan:
    remaining: tuple[EvaluatedCandidate, ...]
    completed_patch_ids: tuple[str, ...]
    blocked_patch_ids: tuple[str, ...]
    sequence: CoverageSequenceProposal

    @property
    def motion_authorized(self) -> bool:
        return False


COARSE_COVERAGE_SEQUENCE_POLICY = (
    "proxy_coarse_incomplete_endpoint_feasible_same_side_serpentine_v1"
)


@dataclass(frozen=True, slots=True)
class CoverageSequenceEntry:
    """One auditable, endpoint-feasible item in a non-executable scan order."""

    view_id: str
    patch_id: str
    side: BladeSide
    row: int
    column: int
    snake_rank: int
    occupied_fraction: float
    endpoint_status: CandidateStatus

    def __post_init__(self) -> None:
        if not self.view_id or not self.patch_id:
            raise ValueError("Coverage sequence IDs must be non-empty")
        if self.row < 0 or self.column < 0 or self.snake_rank < 0:
            raise ValueError("Coverage sequence indices must be non-negative")
        if not 0.0 <= self.occupied_fraction <= 1.0:
            raise ValueError("Coverage sequence occupied fraction must lie in [0, 1]")
        if self.endpoint_status is not CandidateStatus.ENDPOINT_FEASIBLE:
            raise ValueError("Coverage sequence entries must be endpoint feasible")


@dataclass(frozen=True, slots=True)
class CoverageSequenceProposal:
    """Coverage-first coarse-view order; it is evidence, never motion permission."""

    entries: tuple[CoverageSequenceEntry, ...]
    deferred_unverified_view_ids: tuple[str, ...]
    start_side: BladeSide
    policy: str = COARSE_COVERAGE_SEQUENCE_POLICY

    def __post_init__(self) -> None:
        ordered = self.ordered_view_ids
        if len(set(ordered)) != len(ordered):
            raise ValueError("Ordered coverage view IDs must be unique")
        if len(set(self.deferred_unverified_view_ids)) != len(
            self.deferred_unverified_view_ids
        ):
            raise ValueError("Deferred coverage view IDs must be unique")
        if set(ordered) & set(self.deferred_unverified_view_ids):
            raise ValueError("Ordered and deferred coverage views must be disjoint")
        if not self.policy:
            raise ValueError("Coverage sequence policy must be non-empty")

    @property
    def ordered_view_ids(self) -> tuple[str, ...]:
        return tuple(entry.view_id for entry in self.entries)

    @property
    def motion_authorized(self) -> bool:
        return False


def create_coverage_ledger(
    plan: BilateralViewPlan,
    config: CoverageConfig,
) -> CoverageLedger:
    bins = (config.bins_per_axis, config.bins_per_axis)
    patches = tuple(
        PatchCoverage(
            candidate.patch.patch_id,
            candidate.patch.side,
            candidate.patch.row,
            candidate.patch.column,
            np.zeros(bins, dtype=np.int64),
        )
        for candidate in plan.candidates
    )
    return CoverageLedger(patches, (), config, plan.rows, plan.columns)


def update_coverage(
    ledger: CoverageLedger,
    plan: BilateralViewPlan,
    proxy: BilateralBladeProxy,
    cloud: PointCloud,
    base_t_camera: PoseSE3,
    observation_id: str,
) -> CoverageLedger:
    """Accumulate one registered view into the front or back surface ledger."""

    if not observation_id:
        raise CoverageError("Coverage observation ID must be non-empty")
    if observation_id in ledger.observation_ids:
        raise CoverageError(f"Coverage observation already recorded: {observation_id}")
    if cloud.frame != "base" or base_t_camera.parent_frame != "base":
        raise CoverageError("Coverage cloud and camera pose must be registered in base")
    plan_ids = tuple(candidate.patch.patch_id for candidate in plan.candidates)
    if plan_ids != tuple(patch.patch_id for patch in ledger.patches):
        raise CoverageError("Coverage ledger does not match the supplied view plan")

    proxy_t_base = proxy.frame_T_proxy.inverse()
    camera_local = proxy_t_base.transform_points(base_t_camera.translation_m)
    if abs(camera_local[2]) < ledger.config.minimum_camera_side_offset_m:
        raise CoverageError("Camera is too close to the proxy mid-plane to identify a side")
    side = BladeSide.FRONT if camera_local[2] > 0.0 else BladeSide.BACK
    side_sign = 1.0 if side is BladeSide.FRONT else -1.0
    points = proxy_t_base.transform_points(cloud.points_m)
    surface_z = side_sign * proxy.extents_m[2] / 2.0
    on_surface = np.abs(points[:, 2] - surface_z) <= ledger.config.maximum_surface_distance_m
    extent_x, extent_y = plan.effective_surface_extents_m
    inside = (
        (points[:, 0] >= -extent_x / 2.0)
        & (points[:, 0] < extent_x / 2.0)
        & (points[:, 1] >= -extent_y / 2.0)
        & (points[:, 1] < extent_y / 2.0)
    )
    points = points[on_surface & inside]
    if len(points) < ledger.config.minimum_surface_points_per_view:
        raise CoverageError(
            f"Coverage view has {len(points)} usable surface points; at least "
            f"{ledger.config.minimum_surface_points_per_view} are required"
        )

    cell_width = extent_x / plan.columns
    cell_height = extent_y / plan.rows
    columns = np.floor((points[:, 0] + extent_x / 2.0) / cell_width).astype(int)
    rows = np.floor((points[:, 1] + extent_y / 2.0) / cell_height).astype(int)
    local_x = (points[:, 0] + extent_x / 2.0) / cell_width - columns
    local_y = (points[:, 1] + extent_y / 2.0) / cell_height - rows
    bins = ledger.config.bins_per_axis
    bin_x = np.minimum(np.floor(local_x * bins).astype(int), bins - 1)
    bin_y = np.minimum(np.floor(local_y * bins).astype(int), bins - 1)

    updated: list[PatchCoverage] = []
    for patch in ledger.patches:
        counts = patch.bin_point_counts.copy()
        observations = patch.observation_ids
        if patch.side is side:
            selected = (rows == patch.row) & (columns == patch.column)
            if np.any(selected):
                np.add.at(counts, (bin_y[selected], bin_x[selected]), 1)
                observations = (*observations, observation_id)
        updated.append(
            PatchCoverage(
                patch.patch_id,
                patch.side,
                patch.row,
                patch.column,
                counts,
                observations,
            )
        )
    return CoverageLedger(
        tuple(updated),
        (*ledger.observation_ids, observation_id),
        ledger.config,
        ledger.rows,
        ledger.columns,
    )


def select_uncovered_candidates(
    filtered_plan: FilteredViewPlan,
    ledger: CoverageLedger,
    *,
    start_side: BladeSide = BladeSide.FRONT,
) -> CoverageDrivenViewPlan:
    """Return incomplete candidates plus a non-executable coarse-scan sequence.

    Coverage completion and endpoint feasibility are hard gates.  The ordering policy
    then keeps one proxy side together and traverses its original grid rows in a
    deterministic boustrophedon pattern.  Geometry-only candidates are retained in
    ``remaining`` for audit, but are never placed in ``ordered_view_ids``.
    """

    evaluations = {item.candidate.patch.patch_id: item for item in filtered_plan.candidates}
    if set(evaluations) != {patch.patch_id for patch in ledger.patches}:
        raise CoverageError("Filtered plan and coverage ledger patch IDs do not match")
    completed = ledger.completed_patch_ids
    incomplete = tuple(
        patch.patch_id for patch in ledger.patches if patch.patch_id not in completed
    )
    remaining = tuple(
        evaluations[patch_id]
        for patch_id in incomplete
        if evaluations[patch_id].status is not CandidateStatus.REJECTED
    )
    blocked = tuple(
        patch_id
        for patch_id in incomplete
        if evaluations[patch_id].status is CandidateStatus.REJECTED
    )
    side_order = (start_side, _opposite_side(start_side))

    def sequence_key(item: EvaluatedCandidate) -> tuple[int, int, int, str]:
        patch = item.candidate.patch
        if patch.row >= ledger.rows or patch.column >= ledger.columns:
            raise CoverageError(
                f"Patch {patch.patch_id!r} lies outside the coverage grid"
            )
        side_rank = side_order.index(patch.side)
        snake_column = patch.column if patch.row % 2 == 0 else -patch.column
        return side_rank, patch.row, snake_column, item.candidate.view_id

    ordered_candidates = tuple(
        sorted(
            (
                item
                for item in remaining
                if item.status is CandidateStatus.ENDPOINT_FEASIBLE
                and item.joint_positions_rad is not None
            ),
            key=sequence_key,
        )
    )
    deferred = tuple(
        item.candidate.view_id
        for item in sorted(
            (
                item
                for item in remaining
                if item.status is not CandidateStatus.ENDPOINT_FEASIBLE
                or item.joint_positions_rad is None
            ),
            key=sequence_key,
        )
    )
    sequence = CoverageSequenceProposal(
        tuple(
            CoverageSequenceEntry(
                view_id=item.candidate.view_id,
                patch_id=item.candidate.patch.patch_id,
                side=item.candidate.patch.side,
                row=item.candidate.patch.row,
                column=item.candidate.patch.column,
                snake_rank=(
                    item.candidate.patch.row * ledger.columns
                    + (
                        item.candidate.patch.column
                        if item.candidate.patch.row % 2 == 0
                        else ledger.columns - 1 - item.candidate.patch.column
                    )
                ),
                occupied_fraction=ledger.patch(
                    item.candidate.patch.patch_id
                ).occupied_fraction(ledger.config.minimum_points_per_bin),
                endpoint_status=item.status,
            )
            for item in ordered_candidates
        ),
        deferred,
        start_side,
    )
    return CoverageDrivenViewPlan(remaining, completed, blocked, sequence)


def _opposite_side(side: BladeSide) -> BladeSide:
    if side is BladeSide.FRONT:
        return BladeSide.BACK
    if side is BladeSide.BACK:
        return BladeSide.FRONT
    raise CoverageError(f"Unsupported blade side: {side!r}")
