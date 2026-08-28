from __future__ import annotations

import numpy as np

from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping.robot_depth_renderer import (
    _clip_triangle_to_near_plane,
    _rasterize_triangle_depth,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(21, 21, 10.0, 10.0, 10.0, 10.0, "none", ())


def test_triangle_rasterizer_writes_projective_depth_inside_silhouette() -> None:
    depth = np.full((21, 21), np.inf)
    triangle = np.array([[-0.4, -0.4, 1.0], [0.4, -0.4, 1.0], [0.0, 0.4, 1.0]])

    _rasterize_triangle_depth(depth, triangle, _intrinsics())

    assert np.isclose(depth[10, 10], 1.0)
    assert np.isinf(depth[0, 0])


def test_triangle_rasterizer_retains_nearest_surface() -> None:
    depth = np.full((21, 21), np.inf)
    far = np.array([[-0.4, -0.4, 2.0], [0.4, -0.4, 2.0], [0.0, 0.4, 2.0]])
    near = far / np.array([1.0, 1.0, 2.0])

    _rasterize_triangle_depth(depth, far, _intrinsics())
    _rasterize_triangle_depth(depth, near, _intrinsics())

    assert np.isclose(depth[10, 10], 1.0)


def test_triangle_crossing_camera_plane_is_clipped_not_dropped() -> None:
    triangle = np.array(
        [[-0.01, -0.01, -0.1], [0.4, -0.4, 1.0], [0.0, 0.4, 1.0]],
        dtype=np.float64,
    )

    clipped = _clip_triangle_to_near_plane(triangle, near_z_m=0.01)

    assert len(clipped) == 2
    assert all(np.all(item[:, 2] >= 0.01) for item in clipped)


def test_triangle_fully_behind_camera_is_rejected() -> None:
    triangle = np.array(
        [[-0.1, -0.1, -1.0], [0.1, -0.1, -0.5], [0.0, 0.1, -0.2]],
        dtype=np.float64,
    )

    assert _clip_triangle_to_near_plane(triangle) == ()
