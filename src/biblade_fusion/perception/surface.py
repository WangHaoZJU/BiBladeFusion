"""Paper-derived curved-surface partitioning and fine-scan view generation.

The implementation follows the blade-view-planning sequence: Angle Criterion boundary
evidence, coordinate-separated blade boundaries, geodesic section-line coordinates,
equal-footprint partitioning, OBB patch centers, dominant local normals, and viewpoint
placement along those normals.  Curvature subdivision and four independently labelled
edge regions are explicit extensions for a thin blade that must be scanned bilaterally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil, radians

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    ReacquisitionPerturbationConfig,
    SurfacePartitionConfig,
    ViewPlanningConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.boundary import (
    BladeBoundaryModel,
    BoundaryModelError,
    BoundaryName,
    boundary_driven_coordinates,
    fit_blade_boundary,
)
from biblade_fusion.perception.features import (
    FinComponent,
    FinSegmentationError,
    segment_single_fin,
)
from biblade_fusion.perception.fusion import FusedBladeCloud, estimate_normals
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch


class SurfacePartitionError(ValueError):
    """The bilateral coarse cloud cannot support the requested surface partition."""


class SurfaceRegion(StrEnum):
    SURFACE = "surface"
    LEADING_EDGE = "leading_edge"
    TRAILING_EDGE = "trailing_edge"
    ROOT = "root"
    TIP = "tip"
    FIN_FACE = "fin_face"
    FIN_ROOT = "fin_root"
    FIN_FREE_EDGE = "fin_free_edge"


def _readonly(value: ArrayLike, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CurvedSurfacePatch:
    patch_id: str
    side: BladeSide
    region: SurfaceRegion
    row: int
    column: int
    adaptive_depth: int
    points_m: NDArray[np.float64]
    normals: NDArray[np.float64]
    section_coordinates: NDArray[np.float64]
    obb_center_m: NDArray[np.float64]
    obb_axes: NDArray[np.float64]
    obb_extents_m: NDArray[np.float64]
    main_normal: NDArray[np.float64]
    curvature_deg: float
    boundary_fraction: float

    def __post_init__(self) -> None:
        if not self.patch_id or self.row < 0 or self.column < 0 or self.adaptive_depth < 0:
            raise ValueError("Curved patch identity is invalid")
        points = _readonly(self.points_m, None, "Patch points")
        normals = _readonly(self.normals, points.shape, "Patch normals")
        coordinates = _readonly(
            self.section_coordinates, (len(points), 2), "Patch section coordinates"
        )
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
            raise ValueError("Curved patch requires at least six 3D points")
        center = _readonly(self.obb_center_m, (3,), "Patch OBB center")
        axes = _readonly(self.obb_axes, (3, 3), "Patch OBB axes")
        extents = _readonly(self.obb_extents_m, (3,), "Patch OBB extents")
        normal = _readonly(self.main_normal, (3,), "Patch main normal")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6) or np.linalg.det(axes) < 0:
            raise ValueError("Patch OBB axes must be right-handed and orthonormal")
        if np.any(extents < 0.0) or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            raise ValueError("Patch OBB extents/normal are invalid")
        if self.curvature_deg < 0.0 or not 0.0 <= self.boundary_fraction <= 1.0:
            raise ValueError("Patch curvature or boundary fraction is invalid")
        for name, value in (
            ("points_m", points),
            ("normals", normals),
            ("section_coordinates", coordinates),
            ("obb_center_m", center),
            ("obb_axes", axes),
            ("obb_extents_m", extents),
            ("main_normal", normal),
        ):
            object.__setattr__(self, name, value)

    @property
    def planar_extents_m(self) -> tuple[float, float]:
        ordered = np.sort(self.obb_extents_m)[::-1]
        return max(float(ordered[0]), 1e-6), max(float(ordered[1]), 1e-6)


@dataclass(frozen=True, slots=True)
class CurvedBladeSurface:
    frame: str
    patches: tuple[CurvedSurfacePatch, ...]
    axes: NDArray[np.float64]
    center_m: NDArray[np.float64]
    section_arc_lengths_m: tuple[float, float]
    angle_boundary_counts: tuple[int, int]
    base_grid_counts: tuple[int, int]
    base_footprint_m: tuple[float, float]
    footprint_source: str
    fin_components: tuple[FinComponent, ...] = ()
    boundary_models: tuple[BladeBoundaryModel, ...] = ()
    parameterization_methods: tuple[str, str] = ("section_fallback", "section_fallback")
    boundary_fallback_reasons: tuple[str, str] = ("", "")

    def __post_init__(self) -> None:
        if self.frame != "base" or not self.patches:
            raise ValueError("Curved surface must be a non-empty base-frame model")
        if len({patch.patch_id for patch in self.patches}) != len(self.patches):
            raise ValueError("Curved patch IDs must be unique")
        if {patch.side for patch in self.patches} != {BladeSide.FRONT, BladeSide.BACK}:
            raise ValueError("Curved surface must contain both sides")
        object.__setattr__(self, "axes", _readonly(self.axes, (3, 3), "Surface axes"))
        object.__setattr__(self, "center_m", _readonly(self.center_m, (3,), "Surface center"))
        if min(self.section_arc_lengths_m) <= 0.0:
            raise ValueError("Surface section arc lengths must be positive")
        if len(self.base_grid_counts) != 2 or min(self.base_grid_counts) < 1:
            raise ValueError("Surface base-grid counts must contain two positive values")
        if len(self.base_footprint_m) != 2 or min(self.base_footprint_m) <= 0.0:
            raise ValueError("Surface base footprint must contain two positive values")
        if self.footprint_source not in {"calibrated_intrinsics", "configured_override"}:
            raise ValueError("Surface footprint source is invalid")
        if len(self.parameterization_methods) != 2 or len(self.boundary_fallback_reasons) != 2:
            raise ValueError("Surface parameterization status must contain front/back entries")
        if any(
            method not in {"boundary_curves", "section_fallback"}
            for method in self.parameterization_methods
        ):
            raise ValueError("Surface parameterization method is invalid")
        if len({model.side for model in self.boundary_models}) != len(self.boundary_models):
            raise ValueError("Surface boundary models must have unique sides")
        if len({component.side for component in self.fin_components}) != len(self.fin_components):
            raise ValueError("Surface fin components must have unique sides")

    def for_side(self, side: BladeSide) -> tuple[CurvedSurfacePatch, ...]:
        return tuple(patch for patch in self.patches if patch.side is side)

    def for_region(self, region: SurfaceRegion) -> tuple[CurvedSurfacePatch, ...]:
        return tuple(patch for patch in self.patches if patch.region is region)

    def boundary_model(self, side: BladeSide) -> BladeBoundaryModel | None:
        return next((model for model in self.boundary_models if model.side is side), None)

    def fin_component(self, side: BladeSide) -> FinComponent | None:
        return next((item for item in self.fin_components if item.side is side), None)


@dataclass(frozen=True, slots=True)
class CurvedViewPlan:
    surface: CurvedBladeSurface
    candidates: tuple[CandidateView, ...]
    candidate_base_t_left_rectified: tuple[PoseSE3, ...]
    left_rectified_t_left_ir: PoseSE3
    footprint_m: tuple[float, float]

    def __post_init__(self) -> None:
        patch_ids = tuple(patch.patch_id for patch in self.surface.patches)
        candidate_ids = tuple(candidate.patch.patch_id for candidate in self.candidates)
        if patch_ids != candidate_ids:
            raise ValueError("Curved view candidates must preserve surface patch order")
        if len(self.candidate_base_t_left_rectified) != len(self.candidates):
            raise ValueError("Curved view plan requires one rectified pose per candidate")
        if (
            self.left_rectified_t_left_ir.parent_frame,
            self.left_rectified_t_left_ir.child_frame,
        ) != ("left_rectified", "left_ir"):
            raise ValueError("Curved view plan requires left_rectified_T_left_ir")
        for base_t_left_rectified, candidate in zip(
            self.candidate_base_t_left_rectified,
            self.candidates,
            strict=True,
        ):
            if (
                base_t_left_rectified.parent_frame,
                base_t_left_rectified.child_frame,
            ) != ("base", "left_rectified"):
                raise ValueError("Curved view plan requires base_T_left_rectified poses")
            if (
                candidate.base_t_left_ir.parent_frame,
                candidate.base_t_left_ir.child_frame,
            ) != ("base", "left_ir"):
                raise ValueError("Curved view candidates require base_T_left_ir poses")
            expected_base_t_left_ir = base_t_left_rectified.compose(self.left_rectified_t_left_ir)
            if not np.allclose(
                expected_base_t_left_ir.matrix,
                candidate.base_t_left_ir.matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "Curved view raw and rectified candidate poses violate calibration"
                )

    @property
    def motion_authorized(self) -> bool:
        return False


def generate_reacquisition_view(
    candidate: CandidateView,
    base_t_left_rectified: PoseSE3,
    left_rectified_t_left_ir: PoseSE3,
    perturbation: ReacquisitionPerturbationConfig,
    *,
    view_id: str,
    minimum_standoff_distance_m: float,
    maximum_standoff_distance_m: float,
) -> tuple[CandidateView, PoseSE3]:
    """Generate one bounded target-centred retry without bypassing later gates.

    ``tilt_deg`` is the polar offset from the patch outward normal and
    ``azimuth_deg`` selects its direction in the nominal camera-X tangent basis.
    The returned raw left-IR pose remains exactly linked to the rectified pose by
    the pinned stereo calibration.  This function proves geometry only; workspace,
    IK/FK, occupancy, and swept collision checks remain downstream obligations.
    """

    identity = str(view_id).strip()
    if not identity or identity == candidate.view_id:
        raise ValueError("Reacquisition view ID must be non-empty and new")
    if (
        base_t_left_rectified.parent_frame,
        base_t_left_rectified.child_frame,
    ) != ("base", "left_rectified") or (
        left_rectified_t_left_ir.parent_frame,
        left_rectified_t_left_ir.child_frame,
    ) != ("left_rectified", "left_ir"):
        raise ValueError("Reacquisition requires calibrated rectified/raw camera frames")
    expected_raw = base_t_left_rectified.compose(left_rectified_t_left_ir)
    if not np.allclose(
        expected_raw.matrix,
        candidate.base_t_left_ir.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Nominal reacquisition pose violates stereo calibration")

    target = candidate.patch.target_m
    normal = candidate.patch.outward_normal
    nominal_offset = base_t_left_rectified.translation_m - target
    nominal_distance = float(np.linalg.norm(nominal_offset))
    if (
        nominal_distance <= 1e-9
        or abs(nominal_distance - candidate.standoff_distance_m) > 1e-6
        or float(nominal_offset @ normal) / nominal_distance < 1.0 - 1e-6
    ):
        raise ValueError("Nominal reacquisition view is not patch-normal-facing")
    distance = candidate.standoff_distance_m + perturbation.distance_offset_m
    lower = float(minimum_standoff_distance_m)
    upper = float(maximum_standoff_distance_m)
    if (
        not np.isfinite((lower, upper, distance)).all()
        or lower <= 0.0
        or upper < lower
        or not lower <= distance <= upper
    ):
        raise ValueError("Reacquisition distance leaves the calibrated planning interval")

    tangent_x = base_t_left_rectified.rotation[:, 0].copy()
    tangent_x -= normal * float(tangent_x @ normal)
    tangent_norm = float(np.linalg.norm(tangent_x))
    if tangent_norm <= 1e-9:
        raise ValueError("Nominal reacquisition view has no stable tangent basis")
    tangent_x /= tangent_norm
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    tilt = np.deg2rad(perturbation.tilt_deg)
    azimuth = np.deg2rad(perturbation.azimuth_deg)
    tangent_direction = np.cos(azimuth) * tangent_x + np.sin(azimuth) * tangent_y
    camera_direction = np.cos(tilt) * normal + np.sin(tilt) * tangent_direction
    camera_direction /= np.linalg.norm(camera_direction)
    camera_position = target + distance * camera_direction

    camera_z = -camera_direction
    camera_x = base_t_left_rectified.rotation[:, 0].copy()
    camera_x -= camera_z * float(camera_x @ camera_z)
    if np.linalg.norm(camera_x) <= 1e-9:
        camera_x = tangent_y - camera_z * float(tangent_y @ camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, camera_z)
    rotation = np.column_stack((camera_x, camera_y, camera_z))
    retry_rectified = PoseSE3.from_rotation_translation(
        "base",
        "left_rectified",
        rotation,
        camera_position,
    )
    retry_raw = retry_rectified.compose(left_rectified_t_left_ir)
    scale = distance / candidate.standoff_distance_m
    angular_factor = float(np.cos(tilt))
    retry = CandidateView(
        identity,
        candidate.patch,
        retry_raw,
        distance,
        tuple(float(value * scale) for value in candidate.footprint_m),
        float(np.clip(candidate.projection_fraction * angular_factor, 0.0, 1.0)),
        float(np.clip(candidate.visibility_fraction * angular_factor, 0.0, 1.0)),
        f"{candidate.distance_policy}+bounded_reacquisition_v1",
    )
    return retry, retry_rectified


def derive_usable_footprint(
    intrinsics: CameraIntrinsics,
    config: ViewPlanningConfig,
) -> tuple[float, float]:
    """Derive the conservative perpendicular-plane footprint at the baseline distance."""

    baseline = config.standoff_distance_m
    if baseline is None:
        raise SurfacePartitionError(
            "standoff_distance_m is required to derive the fine-scan footprint"
        )
    margin = config.image_edge_margin_px
    available_width = (intrinsics.width - 1) - 2 * margin
    available_height = (intrinsics.height - 1) - 2 * margin
    if available_width <= 0 or available_height <= 0:
        raise SurfacePartitionError("image_edge_margin_px leaves no usable calibrated image area")
    footprint = (
        baseline * available_width / intrinsics.fx * config.footprint_utilization,
        baseline * available_height / intrinsics.fy * config.footprint_utilization,
    )
    if not np.isfinite(footprint).all() or min(footprint) <= 0.0:
        raise SurfacePartitionError("Calibrated intrinsics produced an invalid footprint")
    return float(footprint[0]), float(footprint[1])


@dataclass(frozen=True, slots=True)
class _SideParameterization:
    side: BladeSide
    points_m: NDArray[np.float64]
    normals: NDArray[np.float64]
    boundary_mask: NDArray[np.bool_]
    coordinates: NDArray[np.float64]
    arc_lengths_m: tuple[float, float]


def _voxel_centroids(points: NDArray[np.float64], voxel_size: float) -> NDArray[np.float64]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    result = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(result, inverse, points)
    result /= counts[:, None]
    return result


def _limit_points(points: NDArray[np.float64], maximum: int) -> NDArray[np.float64]:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def _angle_criterion(
    coordinates: NDArray[np.float64], neighbors: int, threshold_deg: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return boundary evidence from the largest empty local-neighbour angle."""

    largest = np.empty(len(coordinates), dtype=np.float64)
    k = min(neighbors, len(coordinates) - 1)
    for start in range(0, len(coordinates), 192):
        chunk = coordinates[start : start + 192]
        distances = np.sum((chunk[:, None, :] - coordinates[None, :, :]) ** 2, axis=2)
        local = np.argpartition(distances, k, axis=1)[:, 1 : k + 1]
        vectors = coordinates[local] - chunk[:, None, :]
        angles = np.sort(np.arctan2(vectors[:, :, 1], vectors[:, :, 0]), axis=1)
        wrapped = np.concatenate((angles, angles[:, :1] + 2.0 * np.pi), axis=1)
        largest[start : start + len(chunk)] = np.max(np.diff(wrapped, axis=1), axis=1)
    return largest > radians(threshold_deg), largest


def _section_coordinate(
    points: NDArray[np.float64], scalar: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Map a PCA coordinate to cumulative 3D section-line arc length."""

    count = int(np.clip(np.sqrt(len(points)), 12, 80))
    edges = np.linspace(float(scalar.min()), float(scalar.max()), count + 1)
    centers: list[NDArray[np.float64]] = []
    values: list[float] = []
    for index in range(count):
        selected = (scalar >= edges[index]) & (
            scalar <= edges[index + 1] if index == count - 1 else scalar < edges[index + 1]
        )
        if np.any(selected):
            centers.append(np.median(points[selected], axis=0))
            values.append(float(np.median(scalar[selected])))
    if len(centers) < 2:
        raise SurfacePartitionError("Section-line slicing produced fewer than two intersections")
    section = np.asarray(centers)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(section, axis=0), axis=1)))
    )
    total = float(cumulative[-1])
    if total <= 1e-9:
        raise SurfacePartitionError("Section-line arc length is degenerate")
    normalized = np.interp(scalar, np.asarray(values), cumulative / total)
    return np.clip(normalized, 0.0, 1.0), total


def _obb(points: NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    covariance = np.cov(points - center, rowvar=False, bias=True)
    _, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    local = (points - center) @ axes
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    return center + axes @ ((lower + upper) / 2.0), axes, upper - lower


def _dominant_normal(
    normals: NDArray[np.float64], config: SurfacePartitionConfig
) -> NDArray[np.float64]:
    azimuth = (np.arctan2(normals[:, 1], normals[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    elevation = np.arcsin(np.clip(normals[:, 2], -1.0, 1.0)) + np.pi / 2.0
    az = np.minimum(
        (azimuth / (2.0 * np.pi) * config.normal_azimuth_bins).astype(int),
        config.normal_azimuth_bins - 1,
    )
    el = np.minimum(
        (elevation / np.pi * config.normal_elevation_bins).astype(int),
        config.normal_elevation_bins - 1,
    )
    flat = el * config.normal_azimuth_bins + az
    dominant = int(np.argmax(np.bincount(flat)))
    selected = flat == dominant
    normal = normals[selected].mean(axis=0)
    if np.linalg.norm(normal) < 1e-12:
        normal = normals.mean(axis=0)
    return normal / np.linalg.norm(normal)


def _curvature_deg(normals: NDArray[np.float64], main: NDArray[np.float64]) -> float:
    angles = np.degrees(np.arccos(np.clip(normals @ main, -1.0, 1.0)))
    return float(np.percentile(angles, 90.0))


def _grid_count(length: float, footprint: float, overlap: float) -> int:
    if length <= footprint:
        return 1
    return ceil((length - footprint) / (footprint * (1.0 - overlap))) + 1


def _region_labels(coordinates: NDArray[np.float64], band: float) -> NDArray[np.int8]:
    s, t = coordinates.T
    labels = np.zeros(len(coordinates), dtype=np.int8)
    labels[t <= band] = 1  # leading
    labels[t >= 1.0 - band] = 2  # trailing
    labels[s <= band] = 3  # root has corner precedence
    labels[s >= 1.0 - band] = 4  # tip has corner precedence
    return labels


_REGIONS = {
    0: SurfaceRegion.SURFACE,
    1: SurfaceRegion.LEADING_EDGE,
    2: SurfaceRegion.TRAILING_EDGE,
    3: SurfaceRegion.ROOT,
    4: SurfaceRegion.TIP,
}


def _initial_groups(
    coordinates: NDArray[np.float64],
    labels: NDArray[np.int8],
    arc_lengths: tuple[float, float],
    config: SurfacePartitionConfig,
    grid_counts: tuple[int, int] | None = None,
) -> list[tuple[SurfaceRegion, int, int, NDArray[np.int64]]]:
    groups: list[tuple[SurfaceRegion, int, int, NDArray[np.int64]]] = []
    if grid_counts is None:
        major_count = _grid_count(
            arc_lengths[0], config.usable_footprint_m[0], config.overlap_fraction
        )
        minor_count = _grid_count(
            arc_lengths[1], config.usable_footprint_m[1], config.overlap_fraction
        )
    else:
        major_count, minor_count = grid_counts
    for code, region in _REGIONS.items():
        region_indices = np.flatnonzero(labels == code)
        if not len(region_indices):
            continue
        if region is SurfaceRegion.SURFACE:
            rows = np.minimum((coordinates[:, 1] * minor_count).astype(int), minor_count - 1)
            columns = np.minimum((coordinates[:, 0] * major_count).astype(int), major_count - 1)
        elif region in {SurfaceRegion.LEADING_EDGE, SurfaceRegion.TRAILING_EDGE}:
            rows = np.zeros(len(coordinates), dtype=int)
            columns = np.minimum((coordinates[:, 0] * major_count).astype(int), major_count - 1)
        else:
            rows = np.minimum((coordinates[:, 1] * minor_count).astype(int), minor_count - 1)
            columns = np.zeros(len(coordinates), dtype=int)
        for row, column in sorted(
            set(zip(rows[region_indices], columns[region_indices], strict=True))
        ):
            selected = region_indices[
                (rows[region_indices] == row) & (columns[region_indices] == column)
            ]
            if len(selected) >= config.minimum_patch_points:
                groups.append((region, int(row), int(column), selected))
    return groups


def _adaptive_groups(
    group: tuple[SurfaceRegion, int, int, NDArray[np.int64]],
    coordinates: NDArray[np.float64],
    normals: NDArray[np.float64],
    config: SurfacePartitionConfig,
    depth: int = 0,
) -> list[tuple[SurfaceRegion, int, int, int, NDArray[np.int64]]]:
    region, row, column, indices = group
    main = _dominant_normal(normals[indices], config)
    curvature = _curvature_deg(normals[indices], main)
    if (
        depth >= config.maximum_adaptive_depth
        or curvature <= config.curvature_split_threshold_deg
        or len(indices) < 2 * config.minimum_patch_points
    ):
        return [(region, row, column, depth, indices)]
    spread = np.ptp(coordinates[indices], axis=0)
    axis = int(np.argmax(spread))
    threshold = float(np.median(coordinates[indices, axis]))
    first = indices[coordinates[indices, axis] <= threshold]
    second = indices[coordinates[indices, axis] > threshold]
    if min(len(first), len(second)) < config.minimum_patch_points:
        return [(region, row, column, depth, indices)]
    results: list[tuple[SurfaceRegion, int, int, int, NDArray[np.int64]]] = []
    for child in (first, second):
        results.extend(
            _adaptive_groups((region, row, column, child), coordinates, normals, config, depth + 1)
        )
    return results


def _fin_grid_groups(
    component: FinComponent,
    mask: NDArray[np.bool_],
    config: SurfacePartitionConfig,
) -> list[tuple[int, int, NDArray[np.int64]]]:
    major_count = _grid_count(
        float(component.obb_extents_m[0]),
        config.usable_footprint_m[0],
        config.overlap_fraction,
    )
    minor_count = _grid_count(
        float(component.obb_extents_m[1]),
        config.usable_footprint_m[1],
        config.overlap_fraction,
    )
    rows = np.minimum(
        (component.local_coordinates[:, 1] * minor_count).astype(int), minor_count - 1
    )
    columns = np.minimum(
        (component.local_coordinates[:, 0] * major_count).astype(int), major_count - 1
    )
    minimum = max(6, config.minimum_patch_points // 2)
    groups: list[tuple[int, int, NDArray[np.int64]]] = []
    selected_indices = np.flatnonzero(mask)
    for row, column in sorted(
        set(zip(rows[selected_indices], columns[selected_indices], strict=True))
    ):
        selected = selected_indices[
            (rows[selected_indices] == row) & (columns[selected_indices] == column)
        ]
        if len(selected) >= minimum:
            groups.append((int(row), int(column), selected))
    if not groups and len(selected_indices) >= 6:
        groups.append((0, 0, selected_indices))
    return groups


def _fin_face_masks(
    component: FinComponent,
    config: SurfacePartitionConfig,
) -> tuple[tuple[str, NDArray[np.bool_], NDArray[np.float64]], ...]:
    face = ~(component.root_mask | component.free_edge_mask)
    if np.count_nonzero(face) < 6:
        face = np.ones(len(component.points_m), dtype=np.bool_)
    if not component.two_faces_observed:
        return (
            ("negative", face, -component.normal_axis),
            ("positive", face, component.normal_axis),
        )
    projection = (component.points_m - component.obb_center_m) @ component.normal_axis
    centers = np.percentile(projection[face], [25.0, 75.0])
    labels = np.zeros(len(component.points_m), dtype=np.bool_)
    for _ in range(12):
        labels[face] = np.argmin(np.abs(projection[face, None] - centers[None, :]), axis=1).astype(
            np.bool_
        )
        if min(np.count_nonzero(face & labels), np.count_nonzero(face & ~labels)) < 6:
            return (
                ("negative", face, -component.normal_axis),
                ("positive", face, component.normal_axis),
            )
        updated = np.array([projection[face & ~labels].mean(), projection[face & labels].mean()])
        if np.allclose(updated, centers, atol=1e-9):
            break
        centers = updated
    return (
        ("negative", face & ~labels, -component.normal_axis),
        ("positive", face & labels, component.normal_axis),
    )


def _fin_curvature_deg(normals: NDArray[np.float64], normal_axis: NDArray[np.float64]) -> float:
    angles = np.degrees(np.arccos(np.clip(np.abs(normals @ normal_axis), 0.0, 1.0)))
    return float(np.percentile(angles, 90.0))


def _fin_patches(
    component: FinComponent,
    surface_axes: NDArray[np.float64],
    config: SurfacePartitionConfig,
) -> list[CurvedSurfacePatch]:
    patches: list[CurvedSurfacePatch] = []

    def append_groups(
        region: SurfaceRegion,
        label: str,
        mask: NDArray[np.bool_],
        normal: NDArray[np.float64],
    ) -> None:
        unit_normal = normal / np.linalg.norm(normal)
        for row, column, indices in _fin_grid_groups(component, mask, config):
            center, axes, extents = _obb(component.points_m[indices])
            patches.append(
                CurvedSurfacePatch(
                    f"{component.component_id}_{region.value}_{label}_r{row:02d}_c{column:02d}",
                    component.side,
                    region,
                    row,
                    column,
                    0,
                    component.points_m[indices],
                    component.normals[indices],
                    component.local_coordinates[indices],
                    center,
                    axes,
                    extents,
                    unit_normal,
                    _fin_curvature_deg(component.normals[indices], component.normal_axis),
                    0.0 if region is SurfaceRegion.FIN_FACE else 1.0,
                )
            )

    for label, mask, normal in _fin_face_masks(component, config):
        append_groups(SurfaceRegion.FIN_FACE, label, mask, normal)
    side_sign = 1.0 if component.side is BladeSide.FRONT else -1.0
    main_outward = side_sign * surface_axes[:, 2]
    for label, sign in (("negative", -1.0), ("positive", 1.0)):
        root_normal = sign * component.normal_axis + config.fin_root_view_main_weight * main_outward
        append_groups(SurfaceRegion.FIN_ROOT, label, component.root_mask, root_normal)
    append_groups(
        SurfaceRegion.FIN_FREE_EDGE,
        "outward",
        component.free_edge_mask,
        main_outward,
    )
    return patches


def partition_curved_blade(
    fused: FusedBladeCloud,
    config: SurfacePartitionConfig,
    *,
    usable_footprint_m: tuple[float, float] | None = None,
    footprint_source: str | None = None,
) -> CurvedBladeSurface:
    """Partition both measured surfaces and all four boundaries of a coarse blade."""

    resolved_footprint = usable_footprint_m or config.usable_footprint_m
    if resolved_footprint is None:
        raise SurfacePartitionError(
            "Fine-scan footprint is unresolved; derive it from calibrated intrinsics "
            "or provide an explicit controlled-test override"
        )
    if not np.isfinite(resolved_footprint).all() or min(resolved_footprint) <= 0.0:
        raise SurfacePartitionError("Fine-scan footprint must contain two positive values")
    source = footprint_source or "configured_override"
    if source not in {"calibrated_intrinsics", "configured_override"}:
        raise SurfacePartitionError("Fine-scan footprint source is invalid")
    # Internal grouping helpers consume the resolved value through the validated
    # configuration object.  The caller-provided configuration remains immutable.
    config = config.model_copy(update={"usable_footprint_m": tuple(resolved_footprint)})

    side_parameterizations: list[_SideParameterization] = []
    boundary_counts: list[int] = []
    all_arc_lengths: list[tuple[float, float]] = []
    boundary_models: list[BladeBoundaryModel] = []
    fin_components: list[FinComponent] = []
    parameterization_methods: list[str] = []
    fallback_reasons: list[str] = []
    for side_enum, side_sign in ((BladeSide.FRONT, 1), (BladeSide.BACK, -1)):
        points = _limit_points(
            _voxel_centroids(fused.points_for_side(side_sign), config.voxel_size_m),
            config.maximum_points_per_side,
        )
        if len(points) < config.minimum_points_per_side:
            raise SurfacePartitionError(
                f"{side_enum.value} side has {len(points)} points; "
                f"at least {config.minimum_points_per_side} are required"
            )
        normals = estimate_normals(
            points,
            min(config.normal_neighbors, len(points) - 1),
            orientation_hint=side_sign * fused.axes[:, 2],
        )
        try:
            segmentation = segment_single_fin(
                points, normals, fused.center_m, fused.axes, side_enum, config
            )
        except FinSegmentationError as exc:
            raise SurfacePartitionError(str(exc)) from exc
        if segmentation.component is not None:
            fin_components.append(segmentation.component)
        points = points[segmentation.main_mask]
        normals = normals[segmentation.main_mask]
        planar = (points - fused.center_m) @ fused.axes[:, :2]
        boundary, _ = _angle_criterion(
            planar, config.normal_neighbors, config.angle_criterion_threshold_deg
        )
        boundary_counts.append(int(np.count_nonzero(boundary)))
        boundary_model = None
        fallback_reason = ""
        if config.boundary_curve_enabled:
            try:
                boundary_model = fit_blade_boundary(points, planar, boundary, side_enum, config)
                coordinates = boundary_driven_coordinates(
                    boundary_model, points, fused.center_m, fused.axes[:, :2]
                )
                major_arc = float(
                    np.mean(
                        (
                            boundary_model.curve(BoundaryName.LEADING_EDGE).arc_length_m,
                            boundary_model.curve(BoundaryName.TRAILING_EDGE).arc_length_m,
                        )
                    )
                )
                minor_arc = float(
                    np.mean(
                        (
                            boundary_model.curve(BoundaryName.ROOT).arc_length_m,
                            boundary_model.curve(BoundaryName.TIP).arc_length_m,
                        )
                    )
                )
            except BoundaryModelError as exc:
                if not config.boundary_allow_fallback:
                    raise SurfacePartitionError(
                        f"{side_enum.value} boundary-curve parameterization failed: {exc}"
                    ) from exc
                fallback_reason = str(exc)
        if boundary_model is None:
            s, major_arc = _section_coordinate(points, planar[:, 0])
            t, minor_arc = _section_coordinate(points, planar[:, 1])
            coordinates = np.column_stack((s, t))
            parameterization_methods.append("section_fallback")
        else:
            boundary_models.append(boundary_model)
            parameterization_methods.append("boundary_curves")
        fallback_reasons.append(fallback_reason)
        all_arc_lengths.append((major_arc, minor_arc))
        side_parameterizations.append(
            _SideParameterization(
                side_enum,
                points,
                normals,
                boundary,
                coordinates,
                (major_arc, minor_arc),
            )
        )

    # Use the longer measured side in each direction so that the footprint constraint
    # is conservative and front/back base-grid row/column identities stay comparable.
    shared_arc_lengths = tuple(
        float(np.max(values)) for values in zip(*all_arc_lengths, strict=True)
    )
    shared_grid_counts = (
        _grid_count(
            shared_arc_lengths[0],
            config.usable_footprint_m[0],
            config.overlap_fraction,
        ),
        _grid_count(
            shared_arc_lengths[1],
            config.usable_footprint_m[1],
            config.overlap_fraction,
        ),
    )
    patches: list[CurvedSurfacePatch] = []
    for side_data in side_parameterizations:
        side_enum = side_data.side
        points = side_data.points_m
        normals = side_data.normals
        boundary = side_data.boundary_mask
        coordinates = side_data.coordinates
        labels = _region_labels(coordinates, config.boundary_band_fraction)
        groups = _initial_groups(
            coordinates,
            labels,
            side_data.arc_lengths_m,
            config,
            shared_grid_counts,
        )
        serials: dict[tuple[SurfaceRegion, int, int], int] = {}
        for group in groups:
            for region, row, column, depth, indices in _adaptive_groups(
                group, coordinates, normals, config
            ):
                key = (region, row, column)
                serial = serials.get(key, 0)
                serials[key] = serial + 1
                suffix = f"_s{serial:02d}" if depth else ""
                patch_id = f"{side_enum.value}_{region.value}_r{row:02d}_c{column:02d}{suffix}"
                center, axes, extents = _obb(points[indices])
                main = _dominant_normal(normals[indices], config)
                patches.append(
                    CurvedSurfacePatch(
                        patch_id,
                        side_enum,
                        region,
                        row,
                        column,
                        depth,
                        points[indices],
                        normals[indices],
                        coordinates[indices],
                        center,
                        axes,
                        extents,
                        main,
                        _curvature_deg(normals[indices], main),
                        float(np.mean(boundary[indices])),
                    )
                )
    for component in fin_components:
        patches.extend(_fin_patches(component, fused.axes, config))
    if not patches:
        raise SurfacePartitionError("Surface partition produced no populated patches")
    return CurvedBladeSurface(
        "base",
        tuple(patches),
        fused.axes,
        fused.center_m,
        shared_arc_lengths,
        tuple(boundary_counts),
        shared_grid_counts,
        tuple(resolved_footprint),
        source,
        tuple(fin_components),
        tuple(boundary_models),
        tuple(parameterization_methods),
        tuple(fallback_reasons),
    )


def _fine_camera_rotation(
    rich_patch: CurvedSurfacePatch,
    surface_axes: NDArray[np.float64],
) -> NDArray[np.float64]:
    camera_z = -rich_patch.main_normal
    global_major = surface_axes[:, 0]
    camera_x = global_major - camera_z * float(global_major @ camera_z)
    if np.linalg.norm(camera_x) < 1e-8:
        fallback = surface_axes[:, 1]
        camera_x = fallback - camera_z * float(fallback @ camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, camera_z)
    return np.column_stack((camera_x, camera_y, camera_z))


def _distance_samples(config: ViewPlanningConfig) -> tuple[float, ...]:
    baseline = config.standoff_distance_m
    if baseline is None:
        raise SurfacePartitionError("standoff_distance_m is required for curved views")
    if not config.adaptive_standoff_enabled:
        return (baseline,)
    lower = config.minimum_standoff_distance_m
    upper = config.maximum_standoff_distance_m
    if lower is None or upper is None:
        raise SurfacePartitionError(
            "Adaptive fine-view planning requires minimum_standoff_distance_m and "
            "maximum_standoff_distance_m"
        )
    count = int(np.floor((upper - lower) / config.distance_search_step_m)) + 1
    if count > 10_000:
        raise SurfacePartitionError(
            "Adaptive standoff interval produces more than 10000 distance samples"
        )
    values = [lower + index * config.distance_search_step_m for index in range(count)]
    values.extend((lower, baseline, upper))
    return tuple(sorted({round(float(value), 12) for value in values if lower <= value <= upper}))


def _allowed_image_bounds(
    intrinsics: CameraIntrinsics,
    config: ViewPlanningConfig,
) -> tuple[float, float, float, float]:
    margin = float(config.image_edge_margin_px)
    width = float(intrinsics.width - 1)
    height = float(intrinsics.height - 1)
    usable_width = width - 2.0 * margin
    usable_height = height - 2.0 * margin
    if usable_width <= 0.0 or usable_height <= 0.0:
        raise SurfacePartitionError("Configured image margin leaves no usable image area")
    utilization_x = usable_width * config.footprint_utilization
    utilization_y = usable_height * config.footprint_utilization
    return (
        margin + (usable_width - utilization_x) / 2.0,
        margin + (usable_height - utilization_y) / 2.0,
        margin + (usable_width + utilization_x) / 2.0,
        margin + (usable_height + utilization_y) / 2.0,
    )


def _project_points(
    points_m: NDArray[np.float64],
    camera_position_m: NDArray[np.float64],
    rotation: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    camera_points = (points_m - camera_position_m) @ rotation
    depth = camera_points[:, 2]
    safe_depth = np.where(depth > 1e-12, depth, 1.0)
    x = camera_points[:, 0] / safe_depth
    y = camera_points[:, 1] / safe_depth
    model = intrinsics.distortion_model.lower()
    if model not in {"none", "distortion.none"}:
        if model not in {
            "brown_conrady",
            "distortion.brown_conrady",
            "modified_brown_conrady",
            "distortion.modified_brown_conrady",
        }:
            raise SurfacePartitionError(
                f"Fine-view projection does not support distortion model "
                f"{intrinsics.distortion_model}"
            )
        coefficients = np.zeros(8, dtype=np.float64)
        count = min(len(intrinsics.distortion_coefficients), len(coefficients))
        coefficients[:count] = intrinsics.distortion_coefficients[:count]
        k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
        radius2 = x * x + y * y
        numerator = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        denominator = 1.0 + k4 * radius2 + k5 * radius2**2 + k6 * radius2**3
        safe_denominator = np.where(
            np.abs(denominator) < 1e-12,
            np.copysign(1e-12, denominator),
            denominator,
        )
        radial = numerator / safe_denominator
        distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        distorted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = distorted_x, distorted_y
    u = intrinsics.fx * x + intrinsics.cx
    v = intrinsics.fy * y + intrinsics.cy
    return u, v, depth


def _projection_visibility(
    rich_patch: CurvedSurfacePatch,
    all_surface_points_m: NDArray[np.float64],
    camera_position_m: NDArray[np.float64],
    rotation: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    config: ViewPlanningConfig,
) -> tuple[float, float]:
    u_min, v_min, u_max, v_max = _allowed_image_bounds(intrinsics, config)
    target_u, target_v, target_depth = _project_points(
        rich_patch.points_m, camera_position_m, rotation, intrinsics
    )
    lower_depth = config.minimum_standoff_distance_m if config.adaptive_standoff_enabled else 0.0
    upper_depth = config.maximum_standoff_distance_m if config.adaptive_standoff_enabled else np.inf
    assert lower_depth is not None and upper_depth is not None
    projected = (
        (target_depth >= lower_depth)
        & (target_depth <= upper_depth)
        & (target_u >= u_min)
        & (target_u <= u_max)
        & (target_v >= v_min)
        & (target_v <= v_max)
    )
    projection_fraction = float(np.mean(projected))
    if not np.any(projected):
        return projection_fraction, 0.0

    all_u, all_v, all_depth = _project_points(
        all_surface_points_m, camera_position_m, rotation, intrinsics
    )
    in_image = (
        (all_depth > 0.0)
        & (all_u >= 0.0)
        & (all_u < intrinsics.width)
        & (all_v >= 0.0)
        & (all_v < intrinsics.height)
    )
    pixel_count = intrinsics.width * intrinsics.height
    z_buffer = np.full(pixel_count, np.inf, dtype=np.float64)
    all_pixels = np.floor(all_v[in_image]).astype(np.int64) * intrinsics.width + np.floor(
        all_u[in_image]
    ).astype(np.int64)
    np.minimum.at(z_buffer, all_pixels, all_depth[in_image])
    target_pixels = np.clip(
        np.floor(target_v).astype(np.int64), 0, intrinsics.height - 1
    ) * intrinsics.width + np.clip(np.floor(target_u).astype(np.int64), 0, intrinsics.width - 1)
    visible = projected & (
        target_depth <= z_buffer[target_pixels] + config.occlusion_depth_tolerance_m
    )
    return projection_fraction, float(np.mean(visible))


def _split_patch_for_visibility(
    patch: CurvedSurfacePatch,
) -> tuple[CurvedSurfacePatch, CurvedSurfacePatch] | None:
    if len(patch.points_m) < 12:
        return None
    spread = np.ptp(patch.section_coordinates, axis=0)
    axis = int(np.argmax(spread))
    threshold = float(np.median(patch.section_coordinates[:, axis]))
    masks = (
        patch.section_coordinates[:, axis] <= threshold,
        patch.section_coordinates[:, axis] > threshold,
    )
    if min(np.count_nonzero(mask) for mask in masks) < 6:
        return None
    children: list[CurvedSurfacePatch] = []
    for index, mask in enumerate(masks):
        points = patch.points_m[mask]
        normals = patch.normals[mask]
        center, axes, extents = _obb(points)
        children.append(
            CurvedSurfacePatch(
                f"{patch.patch_id}_v{index}",
                patch.side,
                patch.region,
                patch.row,
                patch.column,
                patch.adaptive_depth + 1,
                points,
                normals,
                patch.section_coordinates[mask],
                center,
                axes,
                extents,
                patch.main_normal,
                _curvature_deg(normals, patch.main_normal),
                patch.boundary_fraction,
            )
        )
    return children[0], children[1]


def generate_curved_view_plan(
    surface: CurvedBladeSurface,
    intrinsics: CameraIntrinsics,
    config: ViewPlanningConfig,
    partition_config: SurfacePartitionConfig,
    *,
    left_rectified_t_left_ir: PoseSE3,
) -> CurvedViewPlan:
    """Search a bounded per-region standoff and fail closed on invisible patches."""

    if (
        left_rectified_t_left_ir.parent_frame,
        left_rectified_t_left_ir.child_frame,
    ) != ("left_rectified", "left_ir"):
        raise ValueError("Fine-view planning requires left_rectified_T_left_ir")
    baseline = config.standoff_distance_m
    if baseline is None:
        raise SurfacePartitionError("standoff_distance_m is required for curved views")
    distances = _distance_samples(config)
    all_surface_points = np.vstack([patch.points_m for patch in surface.patches])
    planned_patches: list[CurvedSurfacePatch] = []
    candidates: list[CandidateView] = []
    rectified_poses: list[PoseSE3] = []

    detail_regions = {
        SurfaceRegion.LEADING_EDGE,
        SurfaceRegion.TRAILING_EDGE,
        SurfaceRegion.ROOT,
        SurfaceRegion.TIP,
        SurfaceRegion.FIN_ROOT,
        SurfaceRegion.FIN_FREE_EDGE,
    }

    def plan_patch(rich_patch: CurvedSurfacePatch, split_depth: int = 0) -> None:
        rotation = _fine_camera_rotation(rich_patch, surface.axes)
        prefer_near = (
            rich_patch.region in detail_regions
            or rich_patch.curvature_deg > partition_config.curvature_split_threshold_deg
        )
        ordered_distances = sorted(
            distances,
            key=(lambda distance: distance)
            if prefer_near
            else (lambda distance: (abs(distance - baseline), distance)),
        )
        best_projection = 0.0
        best_visibility = 0.0
        for distance in ordered_distances:
            camera_position = rich_patch.obb_center_m + rich_patch.main_normal * distance
            projection, visibility = _projection_visibility(
                rich_patch,
                all_surface_points,
                camera_position,
                rotation,
                intrinsics,
                config,
            )
            best_projection = max(best_projection, projection)
            best_visibility = max(best_visibility, visibility)
            if (
                projection + 1e-12 < config.minimum_patch_projection_fraction
                or visibility + 1e-12 < config.minimum_patch_visibility_fraction
            ):
                continue
            patch = SurfacePatch(
                rich_patch.patch_id,
                rich_patch.side,
                rich_patch.row,
                rich_patch.column,
                rich_patch.obb_center_m,
                rich_patch.main_normal,
                rich_patch.planar_extents_m,
            )
            base_t_left_rectified = PoseSE3.from_rotation_translation(
                "base", "left_rectified", rotation, camera_position
            )
            base_t_left_ir = base_t_left_rectified.compose(left_rectified_t_left_ir)
            footprint = tuple(
                float(value * distance / baseline) for value in surface.base_footprint_m
            )
            policy = (
                "adaptive_near_detail"
                if config.adaptive_standoff_enabled and prefer_near
                else "adaptive_nearest_baseline"
                if config.adaptive_standoff_enabled
                else "fixed_baseline"
            )
            planned_patches.append(rich_patch)
            rectified_poses.append(base_t_left_rectified)
            candidates.append(
                CandidateView(
                    rich_patch.patch_id,
                    patch,
                    base_t_left_ir,
                    distance,
                    footprint,
                    projection,
                    visibility,
                    policy,
                )
            )
            return

        if split_depth < config.maximum_visibility_split_depth:
            children = _split_patch_for_visibility(rich_patch)
            if children is not None:
                for child in children:
                    plan_patch(child, split_depth + 1)
                return
        raise SurfacePartitionError(
            f"Patch {rich_patch.patch_id} has no feasible fine-view distance; "
            f"best projection={best_projection:.3f}, visibility={best_visibility:.3f}"
        )

    for rich_patch in surface.patches:
        plan_patch(rich_patch)
        if len(candidates) > config.maximum_candidates:
            raise SurfacePartitionError(
                f"Curved surface requires more than maximum_candidates={config.maximum_candidates}"
            )
    planned_surface = replace(surface, patches=tuple(planned_patches))
    return CurvedViewPlan(
        planned_surface,
        tuple(candidates),
        tuple(rectified_poses),
        left_rectified_t_left_ir,
        planned_surface.base_footprint_m,
    )
