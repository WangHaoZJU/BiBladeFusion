"""Frame-aware point-cloud generation and transformations."""

from biblade_fusion.perception.pointcloud.model import PointCloud
from biblade_fusion.perception.pointcloud.projection import (
    DepthProjectionError,
    depth_image_to_point_cloud,
    native_depth_to_meters,
)
from biblade_fusion.perception.pointcloud.realsense_projection import (
    realsense_depth_image_to_point_cloud,
)

__all__ = [
    "DepthProjectionError",
    "PointCloud",
    "depth_image_to_point_cloud",
    "native_depth_to_meters",
    "realsense_depth_image_to_point_cloud",
]
