"""Frame-aware point-cloud generation and transformations."""

from biblade_fusion.perception.pointcloud.model import PointCloud
from biblade_fusion.perception.pointcloud.projection import (
    DepthProjectionError,
    depth_image_to_point_cloud,
    native_depth_to_meters,
    point_cloud_to_depth_image,
)
from biblade_fusion.perception.pointcloud.realsense_projection import (
    realsense_depth_image_to_point_cloud,
    realsense_project_points_to_pixels,
)

__all__ = [
    "DepthProjectionError",
    "PointCloud",
    "depth_image_to_point_cloud",
    "native_depth_to_meters",
    "point_cloud_to_depth_image",
    "realsense_depth_image_to_point_cloud",
    "realsense_project_points_to_pixels",
]
