"""Metric depth conversion and calibrated pinhole back-projection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud.model import PointCloud


class DepthProjectionError(ValueError):
    """Depth data or calibration cannot safely produce a point cloud."""


def native_depth_to_meters(
    native_depth: ArrayLike,
    depth_scale_m: float,
) -> NDArray[np.float32]:
    """Convert unsigned sensor units to metres, representing zero as ``NaN``."""

    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise DepthProjectionError("Native depth scale must be finite and positive")
    native = np.asarray(native_depth)
    if native.ndim != 2 or not np.issubdtype(native.dtype, np.unsignedinteger):
        raise DepthProjectionError("Native depth must be a two-dimensional unsigned array")
    depth = native.astype(np.float32) * np.float32(depth_scale_m)
    depth[native == 0] = np.nan
    depth.setflags(write=False)
    return depth


def _is_rectified(intrinsics: CameraIntrinsics) -> bool:
    model = intrinsics.distortion_model.lower().split(".")[-1]
    return model in {"none", "distortion_none"} or not any(intrinsics.distortion_coefficients)


def depth_image_to_point_cloud(
    depth_m: ArrayLike,
    intrinsics: CameraIntrinsics,
    config: PointCloudConfig,
    *,
    frame: str,
    valid_mask: ArrayLike | None = None,
) -> PointCloud:
    """Back-project rectified axial depth into camera-frame metric points."""

    if config.maximum_depth_m <= config.minimum_depth_m:
        raise DepthProjectionError("maximum_depth_m must exceed minimum_depth_m")
    if not _is_rectified(intrinsics):
        raise DepthProjectionError(
            f"Pinhole projection requires rectified intrinsics; got {intrinsics.distortion_model}"
        )

    depth = np.asarray(depth_m, dtype=np.float64)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.shape != expected_shape:
        raise DepthProjectionError(f"Depth shape {depth.shape} does not match {expected_shape}")
    valid = (
        np.isfinite(depth) & (depth >= config.minimum_depth_m) & (depth <= config.maximum_depth_m)
    )
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=np.bool_)
        if supplied_mask.shape != expected_shape:
            raise DepthProjectionError("Depth valid mask must match the depth image")
        valid &= supplied_mask

    stride = config.pixel_stride
    sampled_valid = valid[::stride, ::stride]
    sampled_v, sampled_u = np.nonzero(sampled_valid)
    v = sampled_v * stride
    u = sampled_u * stride
    if u.size < config.minimum_valid_points:
        raise DepthProjectionError(
            f"Depth image has {u.size} usable points; "
            f"at least {config.minimum_valid_points} are required"
        )

    z = depth[v, u]
    x = (u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))
    pixels = np.column_stack((u, v)).astype(np.int32)
    return PointCloud(frame, points, pixels, expected_shape)
