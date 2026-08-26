"""Construct a conservative bilateral proxy from a single visible blade face."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.perception.proxy.model import BilateralBladeProxy


class ProxyBuildError(ValueError):
    """The initial observation cannot support a reliable bilateral proxy."""


def _voxel_centroids(points: NDArray[np.float64], voxel_size_m: float) -> NDArray[np.float64]:
    indices = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    centroids = np.zeros((counts.size, 3), dtype=np.float64)
    np.add.at(centroids, inverse, points)
    centroids /= counts[:, None]
    return centroids


def _orient_major_axis(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Resolve PCA's arbitrary major-axis sign deterministically."""

    dominant_index = int(np.argmax(np.abs(axis)))
    return -axis if axis[dominant_index] < 0.0 else axis


def _expand_about_midpoint(
    lower: float, upper: float, minimum_extent: float | None
) -> tuple[float, float]:
    if minimum_extent is None or upper - lower >= minimum_extent:
        return lower, upper
    midpoint = (lower + upper) / 2.0
    return midpoint - minimum_extent / 2.0, midpoint + minimum_extent / 2.0


def build_bilateral_proxy(
    points_m: ArrayLike,
    frame_T_camera: PoseSE3,
    config: ProxyModelConfig,
    *,
    proxy_frame: str = "blade_proxy",
) -> BilateralBladeProxy:
    """Build an oriented proxy that explicitly includes the unseen blade side.

    Input points and the camera optical center must share ``frame_T_camera``'s parent
    frame. The estimated thickness is mandatory because a single depth observation
    cannot recover the hidden surface position.
    """

    if config.estimated_thickness_m is None:
        raise ProxyBuildError(
            "estimated_thickness_m is required before bilateral proxy construction"
        )

    raw_points = np.asarray(points_m, dtype=np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3:
        raise ProxyBuildError("Initial point cloud must have shape (N, 3)")
    raw_point_count = raw_points.shape[0]
    finite_points = raw_points[np.isfinite(raw_points).all(axis=1)]
    finite_point_count = finite_points.shape[0]
    if finite_point_count < config.minimum_points:
        raise ProxyBuildError(
            f"Initial point cloud has {finite_point_count} finite points; "
            f"at least {config.minimum_points} are required"
        )

    points = _voxel_centroids(finite_points, config.voxel_size_m)
    if points.shape[0] < 3:
        raise ProxyBuildError("Voxelized point cloud contains fewer than three points")

    surface_centroid = points.mean(axis=0)
    centered = points - surface_centroid
    covariance = centered.T @ centered / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    scale = max(float(eigenvalues[0]), np.finfo(np.float64).tiny)
    if eigenvalues[1] <= scale * 1e-10:
        raise ProxyBuildError("Initial point cloud is point-like or line-like")

    camera_vector = frame_T_camera.translation_m - surface_centroid
    camera_distance = float(np.linalg.norm(camera_vector))
    if camera_distance <= 1e-9:
        raise ProxyBuildError("Camera optical center coincides with the observed surface")
    camera_direction = camera_vector / camera_distance

    major_axis = _orient_major_axis(eigenvectors[:, 0])
    normal_axis = eigenvectors[:, 2]
    camera_normal_cosine = float(abs(normal_axis @ camera_direction))
    if camera_normal_cosine < config.minimum_camera_normal_cosine:
        raise ProxyBuildError(
            "Initial view is too grazing for a reliable surface normal: "
            f"cosine {camera_normal_cosine:.3f} < {config.minimum_camera_normal_cosine:.3f}"
        )
    if normal_axis @ camera_direction < 0.0:
        normal_axis = -normal_axis
    minor_axis = np.cross(normal_axis, major_axis)
    minor_axis /= np.linalg.norm(minor_axis)
    axes = np.column_stack((major_axis, minor_axis, normal_axis))

    local_points = centered @ axes
    lower = local_points.min(axis=0)
    upper = local_points.max(axis=0)
    lower[:2] -= config.tangential_margin_m
    upper[:2] += config.tangential_margin_m

    if config.estimated_planar_extents_m is not None:
        for axis_index, estimated_extent in enumerate(config.estimated_planar_extents_m):
            lower[axis_index], upper[axis_index] = _expand_about_midpoint(
                float(lower[axis_index]),
                float(upper[axis_index]),
                estimated_extent + 2.0 * config.tangential_margin_m,
            )

    # +normal is the visible/camera side; the unobserved face lies toward -normal.
    lower[2] -= config.estimated_thickness_m + config.hidden_side_margin_m
    upper[2] += config.visible_side_margin_m

    local_center = (lower + upper) / 2.0
    extents = upper - lower
    proxy_center = surface_centroid + axes @ local_center
    frame_T_proxy = PoseSE3.from_rotation_translation(
        frame_T_camera.parent_frame,
        proxy_frame,
        axes,
        proxy_center,
    )
    return BilateralBladeProxy(
        frame_T_proxy=frame_T_proxy,
        extents_m=extents,
        observed_surface_centroid_m=surface_centroid,
        pca_eigenvalues_m2=eigenvalues,
        raw_point_count=raw_point_count,
        finite_point_count=finite_point_count,
        voxel_point_count=points.shape[0],
        camera_normal_cosine=camera_normal_cosine,
    )
