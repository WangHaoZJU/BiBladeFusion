"""Coverage and quality feedback against measured curved-surface samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.settings import SurfaceQualityConfig
from biblade_fusion.perception.fusion import RegisteredCloudView, estimate_normals
from biblade_fusion.perception.surface import CurvedBladeSurface, SurfaceRegion
from biblade_fusion.perception.tsdf import TriangleMesh
from biblade_fusion.planning.views import BladeSide


class SurfaceCoverageError(ValueError):
    """Surface evidence cannot be associated with the coarse curved model."""


@dataclass(frozen=True, slots=True)
class SurfacePatchEvidence:
    patch_id: str
    minimum_distances_m: NDArray[np.float64]
    best_normal_cosines: NDArray[np.float64]
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        distances = np.array(self.minimum_distances_m, dtype=np.float64, copy=True)
        cosines = np.array(self.best_normal_cosines, dtype=np.float64, copy=True)
        if distances.ndim != 1 or cosines.shape != distances.shape:
            raise ValueError("Surface evidence arrays must be matching vectors")
        if np.isnan(distances).any() or np.isnan(cosines).any():
            raise ValueError("Surface evidence arrays cannot contain NaN")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("Surface evidence observation IDs must be unique")
        distances.setflags(write=False)
        cosines.setflags(write=False)
        object.__setattr__(self, "minimum_distances_m", distances)
        object.__setattr__(self, "best_normal_cosines", cosines)


@dataclass(frozen=True, slots=True)
class SurfaceCoverageLedger:
    evidence: tuple[SurfacePatchEvidence, ...]
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len({item.patch_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("Surface coverage patch IDs must be unique")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("Surface coverage observation IDs must be unique")


@dataclass(frozen=True, slots=True)
class SurfacePatchQuality:
    patch_id: str
    side: BladeSide
    region: SurfaceRegion
    reference_point_count: int
    observed_point_count: int
    coverage_fraction: float
    rmse_m: float
    normal_consistency: float
    curvature_deg: float
    complete: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurfaceQualityReport:
    patches: tuple[SurfacePatchQuality, ...]
    completion_fraction: float
    edge_completion: dict[SurfaceRegion, float]
    mesh_triangle_count: int
    mesh_boundary_edge_count: int
    mesh_boundary_loop_count: int
    mesh_watertight: bool

    @property
    def incomplete_patch_ids(self) -> tuple[str, ...]:
        return tuple(item.patch_id for item in self.patches if not item.complete)


def create_surface_coverage_ledger(surface: CurvedBladeSurface) -> SurfaceCoverageLedger:
    return SurfaceCoverageLedger(
        tuple(
            SurfacePatchEvidence(
                patch.patch_id,
                np.full(len(patch.points_m), np.inf),
                np.full(len(patch.points_m), -1.0),
            )
            for patch in surface.patches
        ),
        (),
    )


def _nearest(
    query: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.empty(len(query), dtype=np.int64)
    distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        chunk = query[start : start + 256]
        squared = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        local = np.argmin(squared, axis=1)
        indices[start : start + len(chunk)] = local
        distances[start : start + len(chunk)] = np.sqrt(squared[np.arange(len(chunk)), local])
    return indices, distances


def update_surface_coverage(
    ledger: SurfaceCoverageLedger,
    surface: CurvedBladeSurface,
    observation: RegisteredCloudView,
    observation_id: str,
    config: SurfaceQualityConfig,
) -> SurfaceCoverageLedger:
    """Accumulate measured-surface distance, incidence, and local-normal evidence."""

    if not observation_id or observation_id in ledger.observation_ids:
        raise SurfaceCoverageError("Surface observation identity is empty or duplicated")
    if tuple(item.patch_id for item in ledger.evidence) != tuple(
        patch.patch_id for patch in surface.patches
    ):
        raise SurfaceCoverageError("Surface ledger does not match the curved model")
    side_sign = (
        1 if (observation.camera_origin_m - surface.center_m) @ surface.axes[:, 2] >= 0 else -1
    )
    side = BladeSide.FRONT if side_sign == 1 else BladeSide.BACK
    points = observation.points_m
    if len(points) < 7:
        raise SurfaceCoverageError("Surface observation has too few points")
    neighbors = min(16, len(points) - 1)
    observed_normals = estimate_normals(
        points, neighbors, orientation_hint=side_sign * surface.axes[:, 2]
    )
    updated: list[SurfacePatchEvidence] = []
    for patch, evidence in zip(surface.patches, ledger.evidence, strict=True):
        distances = evidence.minimum_distances_m.copy()
        cosines = evidence.best_normal_cosines.copy()
        ids = evidence.observation_ids
        if patch.side is side:
            view_direction = observation.camera_origin_m - patch.obb_center_m
            view_direction /= np.linalg.norm(view_direction)
            incidence = float(view_direction @ patch.main_normal)
            if incidence >= config.minimum_incidence_cosine:
                indices, current_distances = _nearest(patch.points_m, points)
                current_cosines = np.abs(
                    np.einsum("ij,ij->i", patch.normals, observed_normals[indices])
                )
                improved = current_distances < distances
                distances[improved] = current_distances[improved]
                cosines[improved] = current_cosines[improved]
                ids = (*ids, observation_id)
        updated.append(SurfacePatchEvidence(patch.patch_id, distances, cosines, ids))
    return SurfaceCoverageLedger(tuple(updated), (*ledger.observation_ids, observation_id))


def _boundary_loop_count(mesh: TriangleMesh) -> int:
    counts: dict[tuple[int, int], int] = {}
    for triangle in mesh.triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    adjacency: dict[int, set[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            unseen = adjacency[current] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return components


def evaluate_surface_quality(
    ledger: SurfaceCoverageLedger,
    surface: CurvedBladeSurface,
    config: SurfaceQualityConfig,
    *,
    mesh: TriangleMesh | None = None,
) -> SurfaceQualityReport:
    """Evaluate real-surface gaps and optional TSDF mesh boundary evidence."""

    if tuple(item.patch_id for item in ledger.evidence) != tuple(
        patch.patch_id for patch in surface.patches
    ):
        raise SurfaceCoverageError("Surface ledger does not match the curved model")
    qualities: list[SurfacePatchQuality] = []
    for patch, evidence in zip(surface.patches, ledger.evidence, strict=True):
        observed = evidence.minimum_distances_m <= config.maximum_surface_distance_m
        count = int(np.count_nonzero(observed))
        coverage = float(np.mean(observed))
        rmse = (
            float(np.sqrt(np.mean(evidence.minimum_distances_m[observed] ** 2)))
            if count
            else float("inf")
        )
        consistency = float(np.mean(evidence.best_normal_cosines[observed])) if count else 0.0
        reasons: list[str] = []
        # Visibility and fin topology may deliberately create a patch with fewer
        # reference samples than the global absolute floor.  Requiring more measured
        # reference samples than actually exist would make completion impossible;
        # the independent coverage-fraction gate still scales with patch size.
        required_observed_points = min(
            config.minimum_observed_points,
            len(patch.points_m),
        )
        if count < required_observed_points:
            reasons.append("too few measured reference samples")
        if coverage < config.completed_fraction:
            reasons.append("measured curved-surface coverage is incomplete")
        if rmse > config.maximum_rmse_m:
            reasons.append("surface residual RMSE exceeds the quality gate")
        if consistency < config.minimum_normal_consistency:
            reasons.append("observed local normals are inconsistent with the coarse surface")
        qualities.append(
            SurfacePatchQuality(
                patch.patch_id,
                patch.side,
                patch.region,
                len(patch.points_m),
                count,
                coverage,
                rmse,
                consistency,
                patch.curvature_deg,
                not reasons,
                tuple(reasons),
            )
        )
    edge_regions = (
        SurfaceRegion.LEADING_EDGE,
        SurfaceRegion.TRAILING_EDGE,
        SurfaceRegion.ROOT,
        SurfaceRegion.TIP,
        SurfaceRegion.FIN_ROOT,
        SurfaceRegion.FIN_FREE_EDGE,
    )
    edges = tuple(
        region for region in edge_regions if any(item.region is region for item in qualities)
    )
    edge_completion = {
        region: (
            float(np.mean([item.complete for item in qualities if item.region is region]))
            if any(item.region is region for item in qualities)
            else 0.0
        )
        for region in edges
    }
    boundary_edges = mesh.boundary_edge_count if mesh is not None else 0
    boundary_loops = _boundary_loop_count(mesh) if mesh is not None else 0
    return SurfaceQualityReport(
        tuple(qualities),
        float(np.mean([item.complete for item in qualities])),
        edge_completion,
        len(mesh.triangles) if mesh is not None else 0,
        boundary_edges,
        boundary_loops,
        mesh is not None and bool(len(mesh.triangles)) and boundary_edges == 0,
    )
