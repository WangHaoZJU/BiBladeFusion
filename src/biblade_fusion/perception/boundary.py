"""Robust four-boundary model for an irregular blade surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import SurfacePartitionConfig
from biblade_fusion.planning.views import BladeSide


class BoundaryModelError(ValueError):
    """AC boundary evidence cannot support a trustworthy four-curve model."""


class BoundaryName(StrEnum):
    ROOT = "root"
    TRAILING_EDGE = "trailing_edge"
    TIP = "tip"
    LEADING_EDGE = "leading_edge"


def _readonly(value: ArrayLike, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BoundaryCurve:
    name: BoundaryName
    degree: int
    knots: NDArray[np.float64]
    control_points_m: NDArray[np.float64]
    source_point_count: int
    fit_rmse_m: float
    inlier_fraction: float
    arc_length_m: float

    def __post_init__(self) -> None:
        knots = _readonly(self.knots, None, "Boundary knots")
        control = _readonly(self.control_points_m, None, "Boundary controls")
        if knots.ndim != 1 or control.ndim != 2 or control.shape[1] != 3:
            raise ValueError("Boundary spline arrays have invalid dimensions")
        if len(knots) != len(control) + self.degree + 1 or self.degree < 1:
            raise ValueError("Boundary spline knot/control count is inconsistent")
        if self.source_point_count < 4 or self.fit_rmse_m < 0.0:
            raise ValueError("Boundary spline fit metrics are invalid")
        if not 0.0 <= self.inlier_fraction <= 1.0 or self.arc_length_m <= 0.0:
            raise ValueError("Boundary spline quality/length is invalid")
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "control_points_m", control)

    def evaluate(self, parameters: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(parameters, dtype=np.float64)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Boundary spline parameters must lie in [0, 1]")
        basis = _bspline_basis(
            values.reshape(-1), self.knots, self.degree, len(self.control_points_m)
        )
        return (basis @ self.control_points_m).reshape((*values.shape, 3))

    def sample_by_arc_length(self, count: int) -> NDArray[np.float64]:
        if count < 2:
            raise ValueError("Boundary arc sampling requires at least two points")
        dense_parameters = np.linspace(0.0, 1.0, max(512, count * 32))
        dense = self.evaluate(dense_parameters)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
        )
        targets = np.linspace(0.0, cumulative[-1], count)
        parameters = np.interp(targets, cumulative, dense_parameters)
        return self.evaluate(parameters)


@dataclass(frozen=True, slots=True)
class BladeBoundaryModel:
    side: BladeSide
    corners_m: NDArray[np.float64]
    curves: tuple[BoundaryCurve, BoundaryCurve, BoundaryCurve, BoundaryCurve]
    ordered_contour_m: NDArray[np.float64]
    ordered_contour_planar: NDArray[np.float64]
    source_boundary_count: int
    fit_rmse_m: float

    def __post_init__(self) -> None:
        corners = _readonly(self.corners_m, (4, 3), "Boundary corners")
        contour = _readonly(self.ordered_contour_m, None, "Ordered boundary contour")
        planar = _readonly(self.ordered_contour_planar, (len(contour), 2), "Ordered planar contour")
        expected = (
            BoundaryName.ROOT,
            BoundaryName.TRAILING_EDGE,
            BoundaryName.TIP,
            BoundaryName.LEADING_EDGE,
        )
        if tuple(curve.name for curve in self.curves) != expected:
            raise ValueError("Boundary curves must use canonical cyclic order")
        if contour.ndim != 2 or contour.shape[1] != 3 or len(contour) < 12:
            raise ValueError("Ordered boundary contour is too small")
        if self.source_boundary_count < len(contour) or self.fit_rmse_m < 0.0:
            raise ValueError("Boundary model metrics are invalid")
        object.__setattr__(self, "corners_m", corners)
        object.__setattr__(self, "ordered_contour_m", contour)
        object.__setattr__(self, "ordered_contour_planar", planar)

    def curve(self, name: BoundaryName) -> BoundaryCurve:
        return next(curve for curve in self.curves if curve.name is name)


def _open_uniform_knots(control_count: int, degree: int) -> NDArray[np.float64]:
    internal_count = control_count - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2)[1:-1] if internal_count > 0 else np.empty(0)
    )
    return np.concatenate((np.zeros(degree + 1), internal, np.ones(degree + 1)))


def _bspline_basis(
    parameters: NDArray[np.float64],
    knots: NDArray[np.float64],
    degree: int,
    control_count: int,
) -> NDArray[np.float64]:
    values = np.clip(np.asarray(parameters, dtype=np.float64), 0.0, 1.0)
    basis = np.zeros((len(values), len(knots) - 1), dtype=np.float64)
    for index in range(len(knots) - 1):
        basis[:, index] = (values >= knots[index]) & (values < knots[index + 1])
    basis[values == 1.0, control_count - 1] = 1.0
    for order in range(1, degree + 1):
        updated_count = len(knots) - order - 1
        updated = np.zeros((len(values), updated_count), dtype=np.float64)
        for index in range(updated_count):
            left_denominator = knots[index + order] - knots[index]
            if left_denominator > 0.0:
                updated[:, index] += (values - knots[index]) / left_denominator * basis[:, index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            if right_denominator > 0.0:
                updated[:, index] += (
                    (knots[index + order + 1] - values) / right_denominator * basis[:, index + 1]
                )
        basis = updated
    if basis.shape[1] != control_count:
        raise BoundaryModelError("B-spline basis/control count is inconsistent")
    return basis


def _chord_parameters(points: NDArray[np.float64]) -> NDArray[np.float64]:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-12:
        raise BoundaryModelError("Boundary curve source points are degenerate")
    return cumulative / cumulative[-1]


def _fit_curve(
    name: BoundaryName,
    points: NDArray[np.float64],
    start_m: NDArray[np.float64],
    end_m: NDArray[np.float64],
    config: SurfacePartitionConfig,
) -> BoundaryCurve:
    source = np.vstack((start_m, points, end_m))
    keep = np.concatenate(([True], np.linalg.norm(np.diff(source, axis=0), axis=1) > 1e-9))
    source = source[keep]
    if len(source) < config.boundary_min_points_per_curve:
        raise BoundaryModelError(
            f"{name.value} has {len(source)} points; at least "
            f"{config.boundary_min_points_per_curve} are required"
        )
    degree = min(3, len(source) - 1)
    control_count = min(
        config.boundary_control_points,
        max(degree + 1, len(source) // 2),
    )
    knots = _open_uniform_knots(control_count, degree)
    parameters = _chord_parameters(source)
    basis = _bspline_basis(parameters, knots, degree, control_count)
    second_difference = np.zeros((max(control_count - 2, 0), control_count))
    for index in range(control_count - 2):
        second_difference[index, index : index + 3] = (1.0, -2.0, 1.0)
    fixed_controls = np.vstack((start_m, end_m))
    fixed_prediction = basis[:, :1] * fixed_controls[0] + basis[:, -1:] * fixed_controls[1]
    inner_basis = basis[:, 1:-1]
    weights = np.ones(len(source), dtype=np.float64)
    controls = np.vstack((fixed_controls[0], np.zeros((control_count - 2, 3)), fixed_controls[1]))
    for _ in range(config.boundary_robust_iterations):
        design = inner_basis * np.sqrt(weights)[:, None]
        target = (source - fixed_prediction) * np.sqrt(weights)[:, None]
        if len(second_difference):
            regularizer = np.sqrt(config.boundary_smoothing_lambda) * second_difference[:, 1:-1]
            regularizer_target = -np.sqrt(config.boundary_smoothing_lambda) * (
                second_difference[:, :1] * fixed_controls[0]
                + second_difference[:, -1:] * fixed_controls[1]
            )
            design = np.vstack((design, regularizer))
            target = np.vstack((target, regularizer_target))
        controls[1:-1], *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = np.linalg.norm(basis @ controls - source, axis=1)
        scale = max(1.4826 * float(np.median(residuals[1:-1])), 1e-6)
        threshold = max(config.boundary_huber_delta_m, 1.5 * scale)
        weights = np.ones_like(residuals)
        outliers = residuals > threshold
        weights[outliers] = threshold / residuals[outliers]
    fitted = basis @ controls
    residuals = np.linalg.norm(fitted - source, axis=1)
    inner_residuals = residuals[1:-1]
    inliers = inner_residuals <= max(
        config.boundary_huber_delta_m, 2.5 * np.median(inner_residuals)
    )
    if np.count_nonzero(inliers) < config.boundary_min_points_per_curve - 2:
        raise BoundaryModelError(f"{name.value} has too few robust inliers")
    rmse = float(np.sqrt(np.mean(inner_residuals[inliers] ** 2)))
    dense_basis = _bspline_basis(np.linspace(0.0, 1.0, 1024), knots, degree, control_count)
    dense = dense_basis @ controls
    arc_length = float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
    curve = BoundaryCurve(
        name,
        degree,
        knots,
        controls,
        len(source),
        rmse,
        float(np.mean(inliers)),
        arc_length,
    )
    if curve.fit_rmse_m > config.boundary_max_fit_rmse_m:
        raise BoundaryModelError(
            f"{name.value} fit RMSE {curve.fit_rmse_m:.6f} m exceeds "
            f"{config.boundary_max_fit_rmse_m:.6f} m"
        )
    if curve.inlier_fraction < config.boundary_min_inlier_fraction:
        raise BoundaryModelError(
            f"{name.value} inlier fraction {curve.inlier_fraction:.3f} is below "
            f"{config.boundary_min_inlier_fraction:.3f}"
        )
    return curve


def _ordered_outer_contour(
    points_m: NDArray[np.float64],
    planar: NDArray[np.float64],
    boundary_mask: NDArray[np.bool_],
    angular_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    indices = np.flatnonzero(boundary_mask)
    if len(indices) < 12:
        raise BoundaryModelError("Angle Criterion produced too few boundary candidates")
    candidates = planar[indices]
    squared = np.sum((candidates[:, None, :] - candidates[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.min(squared, axis=1))
    support_limit = max(4.0 * float(np.median(nearest)), np.percentile(nearest, 90.0))
    supported = nearest <= support_limit
    indices = indices[supported]
    candidates = candidates[supported]
    if len(indices) < 12:
        raise BoundaryModelError("Boundary support filtering retained too few candidates")
    center = np.median(candidates, axis=0)
    scale = np.ptp(candidates, axis=0)
    if np.any(scale <= 1e-9):
        raise BoundaryModelError("Boundary candidates are planar-degenerate")
    normalized = (candidates - center) / scale
    angles = (np.arctan2(normalized[:, 1], normalized[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    radii = np.linalg.norm(normalized, axis=1)
    bins = np.minimum((angles / (2.0 * np.pi) * angular_bins).astype(int), angular_bins - 1)
    retained: list[int] = []
    for bin_index in range(angular_bins):
        local = np.flatnonzero(bins == bin_index)
        if len(local):
            retained.append(int(local[np.argmax(radii[local])]))
    if len(retained) < 12:
        retained = list(range(len(indices)))
    retained_array = np.asarray(retained, dtype=np.int64)
    order = np.argsort(angles[retained_array])
    selected = indices[retained_array[order]]
    return points_m[selected], planar[selected]


def fit_blade_boundary(
    points_m: NDArray[np.float64],
    planar: NDArray[np.float64],
    boundary_mask: NDArray[np.bool_],
    side: BladeSide,
    config: SurfacePartitionConfig,
) -> BladeBoundaryModel:
    """Order AC candidates and fit four endpoint-consistent robust 3D splines."""

    contour_m, contour_planar = _ordered_outer_contour(
        points_m, planar, boundary_mask, config.boundary_angular_bins
    )
    major_min = float(contour_planar[:, 0].min())
    major_max = float(contour_planar[:, 0].max())
    major_tolerance = 0.02 * (major_max - major_min)
    root_indices = np.flatnonzero(contour_planar[:, 0] <= major_min + major_tolerance)
    tip_indices = np.flatnonzero(contour_planar[:, 0] >= major_max - major_tolerance)
    if min(len(root_indices), len(tip_indices)) < 2:
        count = min(max(4, len(contour_planar) // 30), len(contour_planar) // 2)
        major_order = np.argsort(contour_planar[:, 0])
        root_indices = major_order[:count]
        tip_indices = major_order[-count:]
    root_leading = root_indices[np.argmin(contour_planar[root_indices, 1])]
    root_trailing = root_indices[np.argmax(contour_planar[root_indices, 1])]
    tip_trailing = tip_indices[np.argmax(contour_planar[tip_indices, 1])]
    tip_leading = tip_indices[np.argmin(contour_planar[tip_indices, 1])]
    corner_indices = (root_leading, root_trailing, tip_trailing, tip_leading)
    corners = contour_m[np.asarray(corner_indices)]

    def cyclic_path(start: int, end: int) -> NDArray[np.int64]:
        size = len(contour_m)
        forward = (
            np.arange(start, end + 1)
            if start <= end
            else np.r_[np.arange(start, size), np.arange(end + 1)]
        )
        backward = (
            np.arange(start, end - 1, -1)
            if start >= end
            else np.r_[np.arange(start, -1, -1), np.arange(size - 1, end - 1, -1)]
        )
        return np.asarray(forward if len(forward) <= len(backward) else backward, dtype=np.int64)

    paths = tuple(
        cyclic_path(corner_indices[index], corner_indices[(index + 1) % 4]) for index in range(4)
    )

    def supported_path(indices: NDArray[np.int64]) -> NDArray[np.int64]:
        coordinates = contour_planar[indices]
        chord = coordinates[-1] - coordinates[0]
        squared_length = float(chord @ chord)
        if squared_length <= 1e-12:
            raise BoundaryModelError("Boundary corner pair is degenerate")
        parameters = np.clip(((coordinates - coordinates[0]) @ chord) / squared_length, 0.0, 1.0)
        projections = coordinates[0] + parameters[:, None] * chord
        distances = np.linalg.norm(coordinates - projections, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        limit = max(
            config.boundary_huber_delta_m * 3.0,
            median + 4.0 * 1.4826 * mad,
        )
        retained = distances <= limit
        retained[[0, -1]] = True
        return indices[retained]

    paths = tuple(supported_path(path) for path in paths)
    if min(map(len, paths)) < config.boundary_min_points_per_curve:
        raise BoundaryModelError("A fitted boundary has too few ordered contour points")
    curves = (
        _fit_curve(BoundaryName.ROOT, contour_m[paths[0]], corners[0], corners[1], config),
        _fit_curve(BoundaryName.TRAILING_EDGE, contour_m[paths[1]], corners[1], corners[2], config),
        _fit_curve(BoundaryName.TIP, contour_m[paths[2]], corners[2], corners[3], config),
        _fit_curve(BoundaryName.LEADING_EDGE, contour_m[paths[3]], corners[3], corners[0], config),
    )
    return BladeBoundaryModel(
        side,
        corners,
        curves,
        contour_m,
        contour_planar,
        int(np.count_nonzero(boundary_mask)),
        float(np.mean([curve.fit_rmse_m for curve in curves])),
    )


def _inverse_coons_coordinates(
    query: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    resolution = grid.shape[0]
    flattened = grid.reshape(-1, 2)
    result = np.empty((len(query), 2), dtype=np.float64)
    for start in range(0, len(query), 256):
        chunk = query[start : start + 256]
        distances = np.sum((chunk[:, None, :] - flattened[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        result[start : start + len(chunk), 0] = nearest // resolution
        result[start : start + len(chunk), 1] = nearest % resolution
    result /= resolution - 1
    return result


def _nearest_samples(
    query: NDArray[np.float64], curve: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.empty(len(query), dtype=np.int64)
    squared_distances = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 512):
        chunk = query[start : start + 512]
        distances = np.sum((chunk[:, None, :] - curve[None, :, :]) ** 2, axis=2)
        local = np.argmin(distances, axis=1)
        indices[start : start + len(chunk)] = local
        squared_distances[start : start + len(chunk)] = distances[np.arange(len(chunk)), local]
    return indices, squared_distances


def boundary_driven_coordinates(
    model: BladeBoundaryModel,
    points_m: NDArray[np.float64],
    center_m: NDArray[np.float64],
    planar_axes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map an irregular four-curve outline into a normalized transfinite domain."""

    planar = (points_m - center_m) @ planar_axes
    resolution = 96
    root = (
        model.curve(BoundaryName.ROOT).sample_by_arc_length(resolution) - center_m
    ) @ planar_axes
    trailing = (
        model.curve(BoundaryName.TRAILING_EDGE).sample_by_arc_length(resolution) - center_m
    ) @ planar_axes
    tip = (
        model.curve(BoundaryName.TIP).sample_by_arc_length(resolution)[::-1] - center_m
    ) @ planar_axes
    leading = (
        model.curve(BoundaryName.LEADING_EDGE).sample_by_arc_length(resolution)[::-1] - center_m
    ) @ planar_axes

    progress = np.linspace(0.0, 1.0, resolution)
    major = progress[:, None, None]
    minor = progress[None, :, None]
    corner_00, corner_01, corner_11, corner_10 = (
        (corner - center_m) @ planar_axes for corner in model.corners_m
    )
    bilinear = (
        (1.0 - major) * (1.0 - minor) * corner_00
        + (1.0 - major) * minor * corner_01
        + major * minor * corner_11
        + major * (1.0 - minor) * corner_10
    )
    grid = (
        (1.0 - major) * root[None, :, :]
        + major * tip[None, :, :]
        + (1.0 - minor) * leading[:, None, :]
        + minor * trailing[:, None, :]
        - bilinear
    )
    delta_major = np.diff(grid, axis=0)[:, :-1]
    delta_minor = np.diff(grid, axis=1)[:-1]
    signed_area = (
        delta_major[:, :, 0] * delta_minor[:, :, 1] - delta_major[:, :, 1] * delta_minor[:, :, 0]
    )
    orientation = np.sign(float(np.median(signed_area)))
    if orientation == 0.0 or np.percentile(orientation * signed_area, 1.0) <= 1e-12:
        raise BoundaryModelError("Fitted boundary curves produce a folded surface domain")
    coordinates = _inverse_coons_coordinates(planar, grid)
    best_distance = np.full(len(planar), np.inf, dtype=np.float64)
    canonical = (
        (root, lambda value: np.column_stack((np.zeros(len(value)), value))),
        (trailing, lambda value: np.column_stack((value, np.ones(len(value))))),
        (tip, lambda value: np.column_stack((np.ones(len(value)), value))),
        (leading, lambda value: np.column_stack((value, np.zeros(len(value))))),
    )
    for curve, make_coordinates in canonical:
        nearest, squared_distance = _nearest_samples(planar, curve)
        spacing = float(np.median(np.linalg.norm(np.diff(curve, axis=0), axis=1)))
        selected = (squared_distance <= (0.75 * spacing) ** 2) & (squared_distance < best_distance)
        if np.any(selected):
            curve_progress = nearest[selected] / (resolution - 1)
            coordinates[selected] = make_coordinates(curve_progress)
            best_distance[selected] = squared_distance[selected]
    return coordinates
