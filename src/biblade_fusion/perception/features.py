"""Single-fin-per-side segmentation for the photographed blade specimen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import SurfacePartitionConfig
from biblade_fusion.planning.views import BladeSide


class FinSegmentationError(ValueError):
    """A blade half-space cannot support the configured single-fin model."""


def _readonly(
    value: ArrayLike,
    shape: tuple[int, ...] | None,
    name: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FinComponent:
    component_id: str
    side: BladeSide
    points_m: NDArray[np.float64]
    normals: NDArray[np.float64]
    local_coordinates: NDArray[np.float64]
    height_residual_m: NDArray[np.float64]
    root_mask: NDArray[np.bool_]
    free_edge_mask: NDArray[np.bool_]
    obb_center_m: NDArray[np.float64]
    obb_axes: NDArray[np.float64]
    obb_extents_m: NDArray[np.float64]
    normal_axis: NDArray[np.float64]
    main_height_rmse_m: float
    face_separation_m: float
    two_faces_observed: bool

    def __post_init__(self) -> None:
        points = _readonly(self.points_m, None, "Fin points", np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12:
            raise ValueError("Fin component requires at least twelve 3D points")
        normals = _readonly(self.normals, points.shape, "Fin normals", np.float64)
        coordinates = _readonly(
            self.local_coordinates, (len(points), 2), "Fin local coordinates", np.float64
        )
        residual = _readonly(
            self.height_residual_m, (len(points),), "Fin height residual", np.float64
        )
        root = _readonly(self.root_mask, (len(points),), "Fin root mask", np.bool_)
        free = _readonly(self.free_edge_mask, (len(points),), "Fin free-edge mask", np.bool_)
        center = _readonly(self.obb_center_m, (3,), "Fin OBB center", np.float64)
        axes = _readonly(self.obb_axes, (3, 3), "Fin OBB axes", np.float64)
        extents = _readonly(self.obb_extents_m, (3,), "Fin OBB extents", np.float64)
        normal = _readonly(self.normal_axis, (3,), "Fin normal axis", np.float64)
        if not self.component_id or np.any(root & free):
            raise ValueError("Fin component identity or semantic masks are invalid")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6) or np.linalg.det(axes) < 0:
            raise ValueError("Fin OBB axes must be right-handed and orthonormal")
        if np.any(extents <= 0.0) or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            raise ValueError("Fin OBB extents or normal axis are invalid")
        if self.main_height_rmse_m < 0.0 or self.face_separation_m < 0.0:
            raise ValueError("Fin fit metrics are invalid")
        for name, value in (
            ("points_m", points),
            ("normals", normals),
            ("local_coordinates", coordinates),
            ("height_residual_m", residual),
            ("root_mask", root),
            ("free_edge_mask", free),
            ("obb_center_m", center),
            ("obb_axes", axes),
            ("obb_extents_m", extents),
            ("normal_axis", normal),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SideFinSegmentation:
    main_mask: NDArray[np.bool_]
    predicted_height_m: NDArray[np.float64]
    signed_height_residual_m: NDArray[np.float64]
    component: FinComponent | None

    def __post_init__(self) -> None:
        main = np.array(self.main_mask, dtype=np.bool_, copy=True)
        predicted = np.array(self.predicted_height_m, dtype=np.float64, copy=True)
        residual = np.array(self.signed_height_residual_m, dtype=np.float64, copy=True)
        if main.ndim != 1 or predicted.shape != main.shape or residual.shape != main.shape:
            raise ValueError("Side-fin segmentation arrays must be matching vectors")
        if not np.isfinite(predicted).all() or not np.isfinite(residual).all():
            raise ValueError("Side-fin height model must be finite")
        for value in (main, predicted, residual):
            value.setflags(write=False)
        object.__setattr__(self, "main_mask", main)
        object.__setattr__(self, "predicted_height_m", predicted)
        object.__setattr__(self, "signed_height_residual_m", residual)


def _height_design(
    planar: NDArray[np.float64],
) -> tuple[NDArray[np.float64], np.ndarray, np.ndarray]:
    center = np.median(planar, axis=0)
    scale = np.maximum(np.ptp(planar, axis=0) / 2.0, 1e-6)
    x, y = ((planar - center) / scale).T
    design = np.column_stack((np.ones(len(planar)), x, y, x * x, x * y, y * y))
    return design, center, scale


def _fit_main_height(
    planar: NDArray[np.float64],
    height: NDArray[np.float64],
    normals: NDArray[np.float64],
    main_normal: NDArray[np.float64],
    config: SurfacePartitionConfig,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    alignment = np.abs(normals @ main_normal)
    seed = alignment >= config.fin_main_normal_min_cosine
    if np.count_nonzero(seed) < max(30, config.minimum_points_per_side // 2):
        raise FinSegmentationError("Too few main-surface normal seeds for fin separation")
    design, _, _ = _height_design(planar)
    selected_design = design[seed]
    selected_height = height[seed]
    weights = np.ones(len(selected_height), dtype=np.float64)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(config.fin_height_fit_iterations):
        weighted = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            selected_design * weighted[:, None], selected_height * weighted, rcond=None
        )
        residual = selected_design @ coefficients - selected_height
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-6)
        threshold = max(config.fin_height_huber_delta_m, 2.5 * scale)
        weights = np.ones(len(residual), dtype=np.float64)
        outlier = np.abs(residual) > threshold
        weights[outlier] = threshold / np.abs(residual[outlier])
    predicted = design @ coefficients
    seed_residual = predicted[seed] - height[seed]
    rmse = float(np.sqrt(np.mean(seed_residual**2)))
    return predicted, rmse, alignment


def _connected_seed_components(
    points: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    seeds: NDArray[np.bool_],
    radius: float,
) -> list[NDArray[np.int64]]:
    cell_size = radius
    cells = np.floor(points / cell_size).astype(np.int64)
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for index in np.flatnonzero(eligible):
        lookup.setdefault(tuple(cells[index]), []).append(int(index))
    seed_indices = set(map(int, np.flatnonzero(seeds)))
    visited: set[int] = set()
    components: list[NDArray[np.int64]] = []
    offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
    maximum_squared = radius * radius
    while seed_indices:
        start = seed_indices.pop()
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            cell = cells[current]
            for offset in offsets:
                key = tuple(cell + offset)
                for neighbor in lookup.get(key, ()):
                    if neighbor in visited:
                        continue
                    if float(np.sum((points[current] - points[neighbor]) ** 2)) <= maximum_squared:
                        visited.add(neighbor)
                        seed_indices.discard(neighbor)
                        queue.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    return components


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


def _face_separation(
    points: NDArray[np.float64], normal_axis: NDArray[np.float64], config: SurfacePartitionConfig
) -> tuple[float, bool]:
    projection = points @ normal_axis
    centers = np.percentile(projection, [25.0, 75.0])
    labels = np.zeros(len(points), dtype=np.bool_)
    for _ in range(12):
        distances = np.abs(projection[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1).astype(np.bool_)
        if min(np.count_nonzero(labels), np.count_nonzero(~labels)) < 6:
            return 0.0, False
        updated = np.array([projection[~labels].mean(), projection[labels].mean()])
        if np.allclose(updated, centers, atol=1e-9):
            centers = updated
            break
        centers = updated
    separation = float(abs(centers[1] - centers[0]))
    within = max(float(np.std(projection[labels])), float(np.std(projection[~labels])), 1e-6)
    observed = separation >= config.fin_face_min_separation_m and separation >= 2.5 * within
    return separation if observed else 0.0, observed


def segment_single_fin(
    points_m: NDArray[np.float64],
    normals: NDArray[np.float64],
    center_m: NDArray[np.float64],
    axes: NDArray[np.float64],
    side: BladeSide,
    config: SurfacePartitionConfig,
) -> SideFinSegmentation:
    """Separate one protruding fin from one main-blade half-space."""

    planar = (points_m - center_m) @ axes[:, :2]
    height = (points_m - center_m) @ axes[:, 2]
    if config.fin_mode == "disabled":
        return SideFinSegmentation(
            np.ones(len(points_m), dtype=np.bool_), height, np.zeros(len(points_m)), None
        )
    predicted, rmse, alignment = _fit_main_height(planar, height, normals, axes[:, 2], config)
    signed_residual = height - predicted
    absolute_residual = np.abs(signed_residual)
    seeds = (absolute_residual >= config.fin_seed_min_height_m) & (
        alignment <= config.fin_seed_max_normal_cosine
    )
    eligible = (absolute_residual >= config.fin_grow_min_height_m) | (
        alignment <= config.fin_seed_max_normal_cosine
    )
    components = sorted(
        _connected_seed_components(points_m, eligible, seeds, config.fin_connectivity_radius_m),
        key=len,
        reverse=True,
    )
    components = [item for item in components if len(item) >= config.fin_minimum_points]
    if not components:
        if config.fin_mode == "required_single_per_side":
            raise FinSegmentationError(f"{side.value} side has no valid protruding fin")
        return SideFinSegmentation(
            np.ones(len(points_m), dtype=np.bool_), predicted, signed_residual, None
        )
    if len(components) > 1 and len(components[1]) / len(components[0]) > (
        config.fin_maximum_secondary_fraction
    ):
        raise FinSegmentationError(
            f"{side.value} side contains multiple significant protruding components"
        )
    indices = components[0]
    fin_points = points_m[indices]
    fin_normals = normals[indices]
    obb_center, obb_axes, extents = _obb(fin_points)
    if extents[1] < config.fin_minimum_span_m:
        raise FinSegmentationError(f"{side.value} fin span is below the configured minimum")
    if extents[2] / extents[1] > config.fin_maximum_thickness_ratio:
        raise FinSegmentationError(f"{side.value} protrusion is not a thin fin")
    normal_axis = obb_axes[:, 2].copy()
    if normal_axis[np.argmax(np.abs(normal_axis))] < 0.0:
        normal_axis *= -1.0
        obb_axes[:, 2] *= -1.0
        obb_axes[:, 1] *= -1.0
    local = (fin_points - obb_center) @ obb_axes[:, :2]
    lower = local.min(axis=0)
    span = np.maximum(np.ptp(local, axis=0), 1e-9)
    coordinates = np.clip((local - lower) / span, 0.0, 1.0)
    residual = absolute_residual[indices]
    root_limit = max(config.fin_root_band_m, float(np.percentile(residual, 15.0)))
    free_limit = max(
        float(residual.max() - config.fin_free_edge_band_m),
        float(np.percentile(residual, 82.0)),
    )
    root_mask = residual <= root_limit
    free_mask = (residual >= free_limit) & ~root_mask
    separation, two_faces = _face_separation(fin_points, normal_axis, config)
    component = FinComponent(
        f"{side.value}_fin_000",
        side,
        fin_points,
        fin_normals,
        coordinates,
        residual,
        root_mask,
        free_mask,
        obb_center,
        obb_axes,
        extents,
        normal_axis,
        rmse,
        separation,
        two_faces,
    )
    main_mask = np.ones(len(points_m), dtype=np.bool_)
    main_mask[indices] = False
    if np.count_nonzero(main_mask) < config.minimum_points_per_side:
        raise FinSegmentationError(f"{side.value} main surface is too small after fin removal")
    return SideFinSegmentation(main_mask, predicted, signed_residual, component)
