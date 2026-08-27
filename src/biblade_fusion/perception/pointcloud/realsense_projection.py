"""Offline deprojection using the same distortion models as librealsense."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud.model import PointCloud
from biblade_fusion.perception.pointcloud.projection import (
    DepthProjectionError,
    _is_rectified,
    _select_depth_samples,
    depth_image_to_point_cloud,
)


def _librealsense_intrinsics(intrinsics: CameraIntrinsics, rs: Any) -> Any:
    model_name = intrinsics.distortion_model.lower().split(".")[-1]
    try:
        distortion_model = getattr(rs.distortion, model_name)
    except AttributeError as exc:
        raise DepthProjectionError(
            f"Unsupported RealSense distortion model: {intrinsics.distortion_model}"
        ) from exc
    coefficients = list(intrinsics.distortion_coefficients)
    if len(coefficients) > 5:
        raise DepthProjectionError("RealSense distortion calibration has more than five values")
    coefficients.extend([0.0] * (5 - len(coefficients)))

    native = rs.intrinsics()
    native.width = intrinsics.width
    native.height = intrinsics.height
    native.fx = intrinsics.fx
    native.fy = intrinsics.fy
    native.ppx = intrinsics.cx
    native.ppy = intrinsics.cy
    native.model = distortion_model
    native.coeffs = coefficients
    return native


def realsense_depth_image_to_point_cloud(
    depth_m: ArrayLike,
    intrinsics: CameraIntrinsics,
    config: PointCloudConfig,
    *,
    frame: str = "depth",
    valid_mask: ArrayLike | None = None,
    rs_module: ModuleType | Any | None = None,
) -> PointCloud:
    """Deproject native D435i depth, including calibrated lens distortion."""

    if _is_rectified(intrinsics):
        return depth_image_to_point_cloud(
            depth_m,
            intrinsics,
            config,
            frame=frame,
            valid_mask=valid_mask,
        )

    u, v, z = _select_depth_samples(depth_m, intrinsics, config, valid_mask)
    rs = rs_module or import_module("pyrealsense2")
    native_intrinsics = _librealsense_intrinsics(intrinsics, rs)
    try:
        points = np.asarray(
            [
                rs.rs2_deproject_pixel_to_point(
                    native_intrinsics,
                    [float(pixel_u), float(pixel_v)],
                    float(depth),
                )
                for pixel_u, pixel_v, depth in zip(u, v, z, strict=True)
            ],
            dtype=np.float64,
        )
    except Exception as exc:
        raise DepthProjectionError(f"librealsense deprojection failed: {exc}") from exc
    pixels = np.column_stack((u, v)).astype(np.int32)
    return PointCloud(frame, points, pixels, (intrinsics.height, intrinsics.width))


def realsense_project_points_to_pixels(
    points_m: ArrayLike,
    intrinsics: CameraIntrinsics,
    *,
    rs_module: ModuleType | Any | None = None,
) -> NDArray[np.float64]:
    """Project camera-frame points with the matching librealsense distortion model."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise DepthProjectionError("Projected points must be a finite (N, 3) array")
    if _is_rectified(intrinsics):
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        positive = points[:, 2] > 0.0
        pixels[positive, 0] = (
            intrinsics.fx * points[positive, 0] / points[positive, 2] + intrinsics.cx
        )
        pixels[positive, 1] = (
            intrinsics.fy * points[positive, 1] / points[positive, 2] + intrinsics.cy
        )
        return pixels

    rs = rs_module or import_module("pyrealsense2")
    native_intrinsics = _librealsense_intrinsics(intrinsics, rs)
    try:
        pixels = np.asarray(
            [
                rs.rs2_project_point_to_pixel(native_intrinsics, point.tolist())
                if point[2] > 0.0
                else [np.nan, np.nan]
                for point in points
            ],
            dtype=np.float64,
        )
    except Exception as exc:
        raise DepthProjectionError(f"librealsense projection failed: {exc}") from exc
    return pixels
