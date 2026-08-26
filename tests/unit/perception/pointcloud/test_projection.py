import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import (
    DepthProjectionError,
    depth_image_to_point_cloud,
    native_depth_to_meters,
)


def intrinsics(distortion_model: str = "none", coefficients: tuple[float, ...] = ()):
    return CameraIntrinsics(3, 2, 2.0, 2.0, 1.0, 0.5, distortion_model, coefficients)


def config(**overrides: object) -> PointCloudConfig:
    values = {
        "minimum_depth_m": 0.1,
        "maximum_depth_m": 2.0,
        "minimum_valid_points": 3,
    }
    values.update(overrides)
    return PointCloudConfig.model_validate(values)


def test_native_depth_conversion_preserves_missing_values() -> None:
    native = np.array([[1000, 0], [500, 2000]], dtype=np.uint16)

    depth = native_depth_to_meters(native, 0.001)

    np.testing.assert_allclose(depth[[0, 1, 1], [0, 0, 1]], [1.0, 0.5, 2.0])
    assert np.isnan(depth[0, 1])
    assert depth.flags.writeable is False


def test_rectified_depth_back_projects_and_keeps_pixel_provenance() -> None:
    depth = np.array([[1.0, 1.0, np.nan], [0.05, 2.0, 3.0]])

    cloud = depth_image_to_point_cloud(depth, intrinsics(), config(), frame="left_ir")

    np.testing.assert_array_equal(cloud.pixel_uv, [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_allclose(
        cloud.points_m,
        [[-0.5, -0.25, 1.0], [0.0, -0.25, 1.0], [0.0, 0.5, 2.0]],
    )
    transformed = cloud.transformed(
        PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [1, 2, 3])
    )
    np.testing.assert_allclose(transformed.points_m[0], [0.5, 1.75, 4.0])
    assert transformed.frame == "base"


def test_projection_applies_stride_and_mask() -> None:
    depth = np.ones((3, 3))
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])
    square_intrinsics = CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0, "none", ())

    cloud = depth_image_to_point_cloud(
        depth,
        square_intrinsics,
        config(pixel_stride=2),
        frame="depth",
        valid_mask=mask,
    )

    np.testing.assert_array_equal(cloud.pixel_uv, [[0, 0], [2, 0], [0, 2], [2, 2]])


def test_projection_rejects_uncorrected_distortion() -> None:
    with pytest.raises(DepthProjectionError, match="requires rectified"):
        depth_image_to_point_cloud(
            np.ones((2, 3)),
            intrinsics("brown_conrady", (0.1, 0, 0, 0, 0)),
            config(),
            frame="depth",
        )


def test_projection_rejects_too_few_valid_points() -> None:
    with pytest.raises(DepthProjectionError, match="usable points"):
        depth_image_to_point_cloud(
            np.full((2, 3), np.nan),
            intrinsics(),
            config(),
            frame="depth",
        )
