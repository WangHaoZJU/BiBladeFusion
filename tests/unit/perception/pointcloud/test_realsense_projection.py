from types import SimpleNamespace

import numpy as np

from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import (
    realsense_depth_image_to_point_cloud,
    realsense_project_points_to_pixels,
)


class FakeNativeIntrinsics:
    pass


class FakeRs:
    distortion = SimpleNamespace(brown_conrady="brown")

    @staticmethod
    def intrinsics():
        return FakeNativeIntrinsics()

    @staticmethod
    def rs2_deproject_pixel_to_point(native, pixel, depth):
        assert native.model == "brown"
        return [pixel[0] * depth, pixel[1] * depth, depth]

    @staticmethod
    def rs2_project_point_to_pixel(native, point):
        assert native.model == "brown"
        return [point[0] / point[2], point[1] / point[2]]


def test_realsense_projection_handles_distorted_native_depth() -> None:
    intrinsics = CameraIntrinsics(
        2,
        2,
        100.0,
        100.0,
        1.0,
        1.0,
        "distortion.brown_conrady",
        (0.1, 0.0, 0.0, 0.0, 0.0),
    )
    config = PointCloudConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=2.0,
        minimum_valid_points=3,
    )

    cloud = realsense_depth_image_to_point_cloud(
        np.ones((2, 2)), intrinsics, config, rs_module=FakeRs()
    )

    np.testing.assert_allclose(
        cloud.points_m,
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    )
    np.testing.assert_array_equal(cloud.pixel_uv, [[0, 0], [1, 0], [0, 1], [1, 1]])


def test_realsense_projection_handles_distorted_points() -> None:
    intrinsics = CameraIntrinsics(
        2,
        2,
        100.0,
        100.0,
        1.0,
        1.0,
        "distortion.brown_conrady",
        (0.1, 0.0, 0.0, 0.0, 0.0),
    )

    pixels = realsense_project_points_to_pixels(
        np.array([[2.0, 4.0, 2.0], [1.0, 1.0, -1.0]]),
        intrinsics,
        rs_module=FakeRs(),
    )

    np.testing.assert_allclose(pixels[0], [1.0, 2.0])
    assert np.isnan(pixels[1]).all()
