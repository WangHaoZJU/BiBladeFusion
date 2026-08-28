"""Render the validated ES68 collision model for depth-consistent self filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    Es68D435iCollisionTemplate,
    es68_d435i_robot_geometry_hash,
    write_es68_d435i_collision_urdf,
)
from biblade_fusion.robotics.pinocchio_collision import PinocchioCs68Model


@dataclass(frozen=True, slots=True)
class _TriangleMesh:
    link_name: str
    vertices_m: NDArray[np.float64]
    faces: NDArray[np.int64]
    link_t_mesh: NDArray[np.float64]


@dataclass(slots=True)
class Es68D435iRobotDepthRenderer:
    """Software z-buffer over the exact articulated collision-mesh asset."""

    template: Es68D435iCollisionTemplate
    pinocchio_model: PinocchioCs68Model
    meshes: tuple[_TriangleMesh, ...]
    model_content_hash: str
    joint_zero_offsets_rad: tuple[float, ...]
    _temporary_directory: TemporaryDirectory[str] = field(repr=False)

    @classmethod
    def from_active_resources(
        cls,
        resources: Es68D435iCollisionResources | None = None,
        *,
        joint_zero_offsets_rad: Sequence[float] = (),
    ) -> Es68D435iRobotDepthRenderer:
        resolved = resources or Es68D435iCollisionResources.packaged_template()
        template = resolved.load_active()
        temporary = TemporaryDirectory(prefix="biblade-es68-self-mask-")
        urdf = write_es68_d435i_collision_urdf(
            Path(temporary.name) / "es68_d435i.urdf",
            template,
        )
        offsets = tuple(float(value) for value in joint_zero_offsets_rad)
        if not offsets:
            offsets = (0.0,) * 6
        model = PinocchioCs68Model.from_urdf(
            urdf,
            joint_zero_offsets_rad=offsets,
        )
        meshes = tuple(_load_template_meshes(template))
        return cls(
            template,
            model,
            meshes,
            es68_d435i_robot_geometry_hash(
                template,
                joint_zero_offsets_rad=offsets,
            ),
            offsets,
            temporary,
        )

    def render_robot_depth(
        self,
        intrinsics: CameraIntrinsics,
        joint_positions_rad: tuple[float, ...] | NDArray[np.float64],
        base_t_camera: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return nearest collision-model z in the supplied camera frame."""

        transform = np.asarray(base_t_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("base_t_camera must be a finite 4x4 transform")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("base_t_camera is not homogeneous")
        import pinocchio as pin

        configuration = self.pinocchio_model._to_configuration(joint_positions_rad)
        pin.forwardKinematics(
            self.pinocchio_model.model,
            self.pinocchio_model.data,
            configuration,
        )
        pin.updateFramePlacements(self.pinocchio_model.model, self.pinocchio_model.data)
        camera_t_base = np.linalg.inv(transform)
        depth = np.full((intrinsics.height, intrinsics.width), np.inf, dtype=np.float64)
        for mesh in self.meshes:
            frame_id = int(self.pinocchio_model.model.getFrameId(mesh.link_name))
            if frame_id >= len(self.pinocchio_model.model.frames):
                raise ValueError(f"Generated ES68 URDF lacks frame {mesh.link_name!r}")
            base_t_link = np.asarray(
                self.pinocchio_model.data.oMf[frame_id].homogeneous,
                dtype=np.float64,
            )
            camera_t_mesh = camera_t_base @ base_t_link @ mesh.link_t_mesh
            vertices = _transform_points(mesh.vertices_m, camera_t_mesh)
            for face in mesh.faces:
                triangle = vertices[face]
                for visible_triangle in _clip_triangle_to_near_plane(triangle):
                    _rasterize_triangle_depth(depth, visible_triangle, intrinsics)
        depth.setflags(write=False)
        return depth

    def base_t_flange_matrix(
        self,
        joint_positions_rad: tuple[float, ...] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the exact ES68 FK used to place the rendered link meshes."""

        matrix = np.asarray(
            self.pinocchio_model.forward_kinematics(joint_positions_rad),
            dtype=np.float64,
        )
        matrix.setflags(write=False)
        return matrix


def _load_template_meshes(
    template: Es68D435iCollisionTemplate,
) -> list[_TriangleMesh]:
    import trimesh

    loaded: list[_TriangleMesh] = []
    for spec in (*template.links, template.attachment):
        raw = trimesh.load_mesh(spec.mesh_path, process=False)
        if isinstance(raw, trimesh.Scene):
            geometries = tuple(raw.geometry.values())
            if not geometries:
                raise ValueError(f"Empty collision mesh scene: {spec.mesh_path}")
            raw = trimesh.util.concatenate(geometries)
        vertices = np.asarray(raw.vertices, dtype=np.float64) * template.mesh_scale
        faces = np.asarray(raw.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
            raise ValueError(f"Invalid collision mesh geometry: {spec.mesh_path}")
        loaded.append(
            _TriangleMesh(
                spec.link_name,
                vertices,
                faces,
                _pose_from_xyz_rpy(spec.origin_xyz_m, spec.origin_rpy_rad),
            )
        )
    return loaded


def _pose_from_xyz_rpy(
    xyz_m: tuple[float, float, float],
    rpy_rad: tuple[float, float, float],
) -> NDArray[np.float64]:
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz_m, dtype=np.float64)
    return transform


def _transform_points(
    points_m: NDArray[np.float64], transform: NDArray[np.float64]
) -> NDArray[np.float64]:
    homogeneous = np.column_stack((points_m, np.ones(len(points_m), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3]


def _clip_triangle_to_near_plane(
    triangle_camera_m: NDArray[np.float64],
    *,
    near_z_m: float = 1e-5,
) -> tuple[NDArray[np.float64], ...]:
    """Clip one camera-frame triangle instead of dropping a partial silhouette.

    Skipping a triangle when only one vertex lies behind the optical centre can leave
    a hole in the rendered robot mask. The polygon is therefore clipped against the
    positive-depth half-space and triangulated as a fan.
    """

    triangle = np.asarray(triangle_camera_m, dtype=np.float64)
    if triangle.shape != (3, 3) or not np.isfinite(triangle).all():
        raise ValueError("triangle_camera_m must be a finite 3x3 array")
    near = float(near_z_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_z_m must be finite and positive")

    polygon = [vertex.copy() for vertex in triangle]
    clipped: list[NDArray[np.float64]] = []
    for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        start_inside = bool(start[2] >= near)
        end_inside = bool(end[2] >= near)
        if start_inside:
            clipped.append(start)
        if start_inside == end_inside:
            continue
        fraction = (near - float(start[2])) / float(end[2] - start[2])
        intersection = start + fraction * (end - start)
        intersection[2] = near
        clipped.append(intersection)
    if len(clipped) < 3:
        return ()
    anchor = clipped[0]
    return tuple(
        np.asarray((anchor, clipped[index], clipped[index + 1]), dtype=np.float64)
        for index in range(1, len(clipped) - 1)
    )


def _rasterize_triangle_depth(
    depth_m: NDArray[np.float64],
    triangle_camera_m: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> None:
    """Perspective-correct software z-buffer for one triangle."""

    triangle = np.asarray(triangle_camera_m, dtype=np.float64)
    z = triangle[:, 2]
    uv = np.column_stack(
        (
            intrinsics.fx * triangle[:, 0] / z + intrinsics.cx,
            intrinsics.fy * triangle[:, 1] / z + intrinsics.cy,
        )
    )
    height, width = depth_m.shape
    x_min = max(0, int(np.floor(np.min(uv[:, 0]))))
    x_max = min(width - 1, int(np.ceil(np.max(uv[:, 0]))))
    y_min = max(0, int(np.floor(np.min(uv[:, 1]))))
    y_max = min(height - 1, int(np.ceil(np.max(uv[:, 1]))))
    if x_max < x_min or y_max < y_min:
        return
    x = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    y = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(x, y)
    p0, p1, p2 = uv
    denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (
        p0[1] - p2[1]
    )
    if abs(float(denominator)) <= 1e-12:
        return
    w0 = (
        (p1[1] - p2[1]) * (grid_x - p2[0])
        + (p2[0] - p1[0]) * (grid_y - p2[1])
    ) / denominator
    w1 = (
        (p2[1] - p0[1]) * (grid_x - p2[0])
        + (p0[0] - p2[0]) * (grid_y - p2[1])
    ) / denominator
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-10) & (w1 >= -1e-10) & (w2 >= -1e-10)
    if not np.any(inside):
        return
    reciprocal_z = w0 / z[0] + w1 / z[1] + w2 / z[2]
    rendered = np.full(reciprocal_z.shape, np.inf, dtype=np.float64)
    positive = inside & (reciprocal_z > 0.0)
    rendered[positive] = 1.0 / reciprocal_z[positive]
    local = depth_m[y_min : y_max + 1, x_min : x_max + 1]
    update = rendered < local
    local[update] = rendered[update]
