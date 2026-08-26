"""Immutable, frame-aware point-cloud contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3


@dataclass(frozen=True, slots=True)
class PointCloud:
    frame: str
    points_m: NDArray[np.float64]
    pixel_uv: NDArray[np.int32]
    source_image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.frame:
            raise ValueError("Point-cloud frame must be non-empty")
        points = np.array(self.points_m, dtype=np.float64, copy=True)
        pixels = np.array(self.pixel_uv, dtype=np.int32, copy=True)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Point-cloud points must have shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("Point-cloud points must be finite")
        if pixels.shape != (points.shape[0], 2):
            raise ValueError("Point-cloud pixels must have shape (N, 2)")
        height, width = self.source_image_shape
        if height <= 0 or width <= 0:
            raise ValueError("Point-cloud source image shape must be positive")
        if pixels.size and (
            np.any(pixels[:, 0] < 0)
            or np.any(pixels[:, 0] >= width)
            or np.any(pixels[:, 1] < 0)
            or np.any(pixels[:, 1] >= height)
        ):
            raise ValueError("Point-cloud pixel coordinates lie outside the source image")
        points.setflags(write=False)
        pixels.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "pixel_uv", pixels)

    def transformed(self, parent_t_cloud: PoseSE3) -> PointCloud:
        """Transform this cloud into a named parent frame."""

        if parent_t_cloud.child_frame != self.frame:
            raise ValueError(
                f"Point cloud is in {self.frame}, but pose child is {parent_t_cloud.child_frame}"
            )
        return PointCloud(
            frame=parent_t_cloud.parent_frame,
            points_m=parent_t_cloud.transform_points(self.points_m),
            pixel_uv=self.pixel_uv,
            source_image_shape=self.source_image_shape,
        )
