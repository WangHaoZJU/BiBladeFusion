"""Frame-aware point-cloud generation and transformations."""

from biblade_fusion.perception.pointcloud.model import PointCloud
from biblade_fusion.perception.pointcloud.projection import (
    DepthProjectionError,
    depth_image_to_point_cloud,
    native_depth_to_meters,
)

__all__ = [
    "DepthProjectionError",
    "PointCloud",
    "depth_image_to_point_cloud",
    "native_depth_to_meters",
]
