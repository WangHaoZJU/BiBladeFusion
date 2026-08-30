from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping.robot_depth_renderer import (
    _clip_triangle_to_near_plane,
    _load_template_meshes,
    _rasterize_triangle_depth,
    _raycast_triangle_meshes_depth,
)
from biblade_fusion.robotics.collision_template import Es68D435iCollisionResources


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


def test_self_mask_mesh_loader_can_exclude_camera_attachment() -> None:
    template = Es68D435iCollisionResources.packaged_template().load_active()

    meshes = _load_template_meshes(
        template,
        excluded_link_names=(template.attachment.link_name,),
    )

    assert {mesh.link_name for mesh in meshes} == {spec.link_name for spec in template.links}


def test_open3d_raycast_depth_uses_camera_z_parameter() -> None:
    pytest.importorskip("open3d")
    vertices = np.array([[-0.4, -0.4, 1.0], [0.4, -0.4, 1.0], [0.0, 0.4, 1.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    depth = _raycast_triangle_meshes_depth(((vertices, faces),), _intrinsics())

    assert np.isclose(depth[10, 10], 1.0, atol=1e-6)
    assert np.isinf(depth[0, 0])
