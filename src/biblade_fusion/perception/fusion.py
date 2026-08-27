"""Robot-pose-prior multi-view fusion for a bilateral thin blade.

The residual registration deliberately operates within a blade side.  This prevents a
locally smooth front observation from converging onto the geometrically similar back
surface, a common thin-wall ICP failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import MultiViewFusionConfig


class MultiViewFusionError(ValueError):
    """Registered observations cannot form a trustworthy bilateral coarse model."""


def _points(value: ArrayLike, name: str, *, minimum: int = 3) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < minimum:
        raise ValueError(f"{name} must have shape (N, 3) with N >= {minimum}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-vector")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RegisteredCloudView:
    """One cloud already transformed into the robot base by calibrated kinematics."""

    view_id: str
    points_m: NDArray[np.float64]
    camera_origin_m: NDArray[np.float64]
    pixel_uv: NDArray[np.int32] | None = None
    source_image_shape: tuple[int, int] | None = None
    intrinsic_matrix: NDArray[np.float64] | None = None
    base_t_camera_matrix: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not self.view_id:
            raise ValueError("Registered view ID must be non-empty")
        object.__setattr__(self, "points_m", _points(self.points_m, "Registered points"))
        object.__setattr__(self, "camera_origin_m", _vector(self.camera_origin_m, "Camera origin"))
        projection_values = (
            self.pixel_uv,
            self.source_image_shape,
            self.intrinsic_matrix,
            self.base_t_camera_matrix,
        )
        if all(value is None for value in projection_values):
            return
        if any(value is None for value in projection_values):
            raise ValueError("Projection metadata must be supplied together")
        pixels = np.array(self.pixel_uv, dtype=np.int32, copy=True)
        intrinsics = np.array(self.intrinsic_matrix, dtype=np.float64, copy=True)
        pose = np.array(self.base_t_camera_matrix, dtype=np.float64, copy=True)
        if pixels.shape != (len(self.points_m), 2):
            raise ValueError("Registered projection pixels must match points")
        if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("Registered projection matrices have invalid shape")
        if not np.isfinite(intrinsics).all() or not np.isfinite(pose).all():
            raise ValueError("Registered projection matrices must be finite")
        assert self.source_image_shape is not None
        height, width = self.source_image_shape
        if height <= 0 or width <= 0:
            raise ValueError("Registered source image shape must be positive")
        pixels.setflags(write=False)
        intrinsics.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, "pixel_uv", pixels)
        object.__setattr__(self, "intrinsic_matrix", intrinsics)
        object.__setattr__(self, "base_t_camera_matrix", pose)


@dataclass(frozen=True, slots=True)
class PoseRefinement:
    view_id: str
    side: int
    correction_matrix: NDArray[np.float64]
    correspondence_count: int
    rmse_before_m: float
    rmse_after_m: float
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        matrix = np.array(self.correction_matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Pose-refinement correction must be a finite 4x4 matrix")
        matrix.setflags(write=False)
        object.__setattr__(self, "correction_matrix", matrix)
        if self.side not in {-1, 1}:
            raise ValueError("Pose-refinement side must be -1 or +1")


@dataclass(frozen=True, slots=True)
class FusedBladeCloud:
    """Voxel-fused coarse model with immutable front/back membership."""

    points_m: NDArray[np.float64]
    normals: NDArray[np.float64]
    side_labels: NDArray[np.int8]
    center_m: NDArray[np.float64]
    axes: NDArray[np.float64]
    median_thickness_m: float
    refinements: tuple[PoseRefinement, ...]

    def __post_init__(self) -> None:
        points = _points(self.points_m, "Fused points", minimum=6)
        normals = np.array(self.normals, dtype=np.float64, copy=True)
        labels = np.array(self.side_labels, dtype=np.int8, copy=True)
        axes = np.array(self.axes, dtype=np.float64, copy=True)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError("Fused normals must match points and be finite")
        if labels.shape != (len(points),) or not set(np.unique(labels)).issubset({-1, 1}):
            raise ValueError("Fused side labels must contain only -1 and +1")
        if set(np.unique(labels)) != {-1, 1}:
            raise ValueError("Fused cloud must contain both blade sides")
        if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
            raise ValueError("Fused blade axes must be orthonormal")
        if np.linalg.det(axes) < 0.0:
            raise ValueError("Fused blade axes must be right-handed")
        if not np.isfinite(self.median_thickness_m) or self.median_thickness_m <= 0.0:
            raise ValueError("Fused blade thickness must be finite and positive")
        for array in (normals, labels, axes):
            array.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "side_labels", labels)
        object.__setattr__(self, "center_m", _vector(self.center_m, "Fused center"))
        object.__setattr__(self, "axes", axes)

    def points_for_side(self, side: int) -> NDArray[np.float64]:
        if side not in {-1, 1}:
            raise ValueError("Side must be -1 or +1")
        return self.points_m[self.side_labels == side]


def _voxel_centroids(points: NDArray[np.float64], voxel_size_m: float) -> NDArray[np.float64]:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    result = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(result, inverse, points)
    result /= counts[:, None]
    return result


def _subsample(points: NDArray[np.float64], maximum: int) -> NDArray[np.float64]:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def _nearest(
    query: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.empty(len(query), dtype=np.int64)
    squared = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        chunk = query[start : start + 256]
        distances = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        local = np.argmin(distances, axis=1)
        indices[start : start + len(chunk)] = local
        squared[start : start + len(chunk)] = distances[np.arange(len(chunk)), local]
    return indices, squared


def estimate_normals(
    points: ArrayLike,
    neighbors: int = 16,
    *,
    orientation_hint: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Estimate PCA normals without requiring SciPy/Open3D."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < neighbors + 1:
        raise MultiViewFusionError("Normal estimation has too few points")
    normals = np.empty_like(cloud)
    for start in range(0, len(cloud), 128):
        chunk = cloud[start : start + 128]
        distances = np.sum((chunk[:, None, :] - cloud[None, :, :]) ** 2, axis=2)
        local = np.argpartition(distances, neighbors, axis=1)[:, 1 : neighbors + 1]
        neighborhoods = cloud[local]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
        _, vectors = np.linalg.eigh(covariance)
        normals[start : start + len(chunk)] = vectors[:, :, 0]
    if orientation_hint is not None:
        hint = np.asarray(orientation_hint, dtype=np.float64)
        hint /= np.linalg.norm(hint)
        normals[normals @ hint < 0.0] *= -1.0
    return normals


def _rotation_vector_matrix(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = np.array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )
        return np.eye(3) + skew
    axis = vector / angle
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _apply(matrix: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_angle_deg(rotation: NDArray[np.float64]) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _refine_to_map(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    target_normals: NDArray[np.float64],
    config: MultiViewFusionConfig,
) -> tuple[NDArray[np.float64], int, float, float, bool, str]:
    correction = np.eye(4)
    before = float("inf")
    count = 0
    maximum_squared = config.maximum_correspondence_distance_m**2
    for iteration in range(config.icp_iterations):
        moved = _apply(correction, source)
        indices, squared = _nearest(moved, target)
        valid = squared <= maximum_squared
        count = int(np.count_nonzero(valid))
        if count < config.minimum_correspondences:
            return correction, count, before, before, False, "insufficient same-side overlap"
        p = moved[valid]
        q = target[indices[valid]]
        n = target_normals[indices[valid]]
        residual = np.einsum("ij,ij->i", n, q - p)
        rmse = float(np.sqrt(np.mean(residual**2)))
        if iteration == 0:
            before = rmse
        median = float(np.median(np.abs(residual)))
        scale = max(1.4826 * median, 1e-6)
        robust = 1.0 / np.sqrt(1.0 + (residual / (2.5 * scale)) ** 2)
        design = np.column_stack((np.cross(p, n), n)) * robust[:, None]
        rhs = residual * robust
        if config.pose_prior_weight > 0.0:
            prior = np.sqrt(config.pose_prior_weight) * np.eye(6)
            design = np.vstack((design, prior))
            rhs = np.concatenate((rhs, np.zeros(6)))
        delta, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        if np.linalg.norm(delta) < 1e-8:
            break
        update = np.eye(4)
        update[:3, :3] = _rotation_vector_matrix(delta[:3])
        update[:3, 3] = delta[3:]
        correction = update @ correction
    moved = _apply(correction, source)
    indices, squared = _nearest(moved, target)
    valid = squared <= maximum_squared
    count = int(np.count_nonzero(valid))
    if count < config.minimum_correspondences:
        return correction, count, before, float("inf"), False, "overlap vanished after refinement"
    residual = np.einsum(
        "ij,ij->i", target_normals[indices[valid]], target[indices[valid]] - moved[valid]
    )
    after = float(np.sqrt(np.mean(residual**2)))
    translation = float(np.linalg.norm(correction[:3, 3]))
    rotation = _rotation_angle_deg(correction[:3, :3])
    if translation > config.maximum_translation_correction_m:
        return correction, count, before, after, False, "translation correction exceeds bound"
    if rotation > config.maximum_rotation_correction_deg:
        return correction, count, before, after, False, "rotation correction exceeds bound"
    if after > before * 1.01:
        return correction, count, before, after, False, "residual refinement did not improve fit"
    return correction, count, before, after, True, "accepted bounded same-side refinement"


def _pca_frame(
    views: tuple[RegisteredCloudView, ...], config: MultiViewFusionConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sampled = np.vstack(
        [_subsample(_voxel_centroids(view.points_m, config.voxel_size_m), 1500) for view in views]
    )
    center = sampled.mean(axis=0)
    covariance = np.cov(sampled - center, rowvar=False, bias=True)
    _, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, ::-1]
    major = axes[:, 0]
    if major[np.argmax(np.abs(major))] < 0.0:
        major *= -1.0
    normal = axes[:, 2]
    # The first coarse view is the explicit front-side anchor. Using the mean camera
    # centre is unstable for a well-balanced bilateral scan because it approaches the
    # blade mid-surface and leaves PCA's normal sign unresolved.
    if (views[0].camera_origin_m - center) @ normal < 0.0:
        normal *= -1.0
    minor = np.cross(normal, major)
    minor /= np.linalg.norm(minor)
    return center, np.column_stack((major, minor, normal))


def fuse_registered_views(
    views: tuple[RegisteredCloudView, ...],
    config: MultiViewFusionConfig,
) -> FusedBladeCloud:
    """Fuse pose-registered coarse scans with bounded, same-side point-to-plane ICP."""

    if len(views) < 2 or len({view.view_id for view in views}) != len(views):
        raise MultiViewFusionError("Fusion requires at least two uniquely identified views")
    center, axes = _pca_frame(views, config)
    normal_axis = axes[:, 2]
    sides = tuple(
        1 if (view.camera_origin_m - center) @ normal_axis >= 0.0 else -1 for view in views
    )
    if set(sides) != {-1, 1}:
        raise MultiViewFusionError(
            "Coarse scan does not contain camera observations from both sides"
        )

    side_maps: dict[int, NDArray[np.float64]] = {}
    refinements: list[PoseRefinement] = []
    for view, side in zip(views, sides, strict=True):
        source = _subsample(
            _voxel_centroids(view.points_m, config.voxel_size_m), config.maximum_icp_points
        )
        if side not in side_maps:
            side_maps[side] = source
            refinements.append(
                PoseRefinement(
                    view.view_id, side, np.eye(4), len(source), 0.0, 0.0, True, "side anchor"
                )
            )
            continue
        target = _subsample(side_maps[side], config.maximum_icp_points)
        target_normals = estimate_normals(
            target,
            min(config.normal_neighbors, len(target) - 1),
            orientation_hint=side * normal_axis,
        )
        correction, count, before, after, accepted, reason = _refine_to_map(
            source, target, target_normals, config
        )
        corrected = _apply(correction, source) if accepted else source
        side_maps[side] = _voxel_centroids(
            np.vstack((side_maps[side], corrected)), config.voxel_size_m
        )
        refinements.append(
            PoseRefinement(
                view.view_id,
                side,
                correction if accepted else np.eye(4),
                count,
                before,
                after,
                accepted,
                reason,
            )
        )

    front = side_maps[1]
    back = side_maps[-1]
    if min(len(front), len(back)) <= config.normal_neighbors:
        raise MultiViewFusionError("A fused blade side has too few points for local geometry")
    front_normals = estimate_normals(front, config.normal_neighbors, orientation_hint=normal_axis)
    back_normals = estimate_normals(back, config.normal_neighbors, orientation_hint=-normal_axis)
    front_level = float(np.median((front - center) @ normal_axis))
    back_level = float(np.median((back - center) @ normal_axis))
    thickness = abs(front_level - back_level)
    if thickness <= config.voxel_size_m * 0.25:
        raise MultiViewFusionError(
            "Front/back coarse surfaces are not separable at this voxel scale"
        )
    return FusedBladeCloud(
        np.vstack((front, back)),
        np.vstack((front_normals, back_normals)),
        np.concatenate((np.ones(len(front), dtype=np.int8), -np.ones(len(back), dtype=np.int8))),
        center,
        axes,
        thickness,
        tuple(refinements),
    )
