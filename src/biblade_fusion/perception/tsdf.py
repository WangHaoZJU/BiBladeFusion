"""Sparse bilateral TSDF integration and dependency-free mesh extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.settings import TSDFConfig
from biblade_fusion.perception.fusion import FusedBladeCloud, RegisteredCloudView


class TSDFError(ValueError):
    """TSDF integration would violate a configured geometry or memory bound."""


@dataclass(frozen=True, slots=True)
class SparseTSDFVolume:
    side: int
    origin_m: NDArray[np.float64]
    voxel_size_m: float
    truncation_distance_m: float
    voxel_indices: NDArray[np.int32]
    tsdf: NDArray[np.float64]
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        origin = np.array(self.origin_m, dtype=np.float64, copy=True)
        indices = np.array(self.voxel_indices, dtype=np.int32, copy=True)
        tsdf = np.array(self.tsdf, dtype=np.float64, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if self.side not in {-1, 1} or origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Sparse TSDF side/origin is invalid")
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("Sparse TSDF indices must have shape (N, 3)")
        if tsdf.shape != (len(indices),) or weights.shape != tsdf.shape:
            raise ValueError("Sparse TSDF values must match voxel indices")
        if not np.isfinite(tsdf).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Sparse TSDF values/weights are invalid")
        if np.any(np.abs(tsdf) > 1.0 + 1e-9):
            raise ValueError("Sparse TSDF values must be normalized to [-1, 1]")
        for array in (origin, indices, tsdf, weights):
            array.setflags(write=False)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "voxel_indices", indices)
        object.__setattr__(self, "tsdf", tsdf)
        object.__setattr__(self, "weights", weights)

    @property
    def voxel_count(self) -> int:
        return len(self.voxel_indices)


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    vertices_m: NDArray[np.float64]
    triangles: NDArray[np.int32]
    triangle_sides: NDArray[np.int8]

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices_m, dtype=np.float64, copy=True)
        triangles = np.array(self.triangles, dtype=np.int32, copy=True)
        sides = np.array(self.triangle_sides, dtype=np.int8, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite and have shape (N, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Mesh triangles must have shape (M, 3)")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("Mesh triangle index lies outside the vertex array")
        if sides.shape != (len(triangles),) or not set(np.unique(sides)).issubset({-1, 1}):
            raise ValueError("Mesh triangle sides must contain only -1/+1")
        for array in (vertices, triangles, sides):
            array.setflags(write=False)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "triangle_sides", sides)

    @property
    def boundary_edge_count(self) -> int:
        counts: dict[tuple[int, int], int] = {}
        for triangle in self.triangles:
            for first, second in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                edge = tuple(sorted((int(first), int(second))))
                counts[edge] = counts.get(edge, 0) + 1
        return sum(count == 1 for count in counts.values())


@dataclass(frozen=True, slots=True)
class BilateralTSDFResult:
    front: SparseTSDFVolume
    back: SparseTSDFVolume
    mesh: TriangleMesh
    protected_truncation_distance_m: float
    backend: str = "numpy_sparse"
    feature_thicknesses_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.front.side != 1 or self.back.side != -1:
            raise ValueError("Bilateral TSDF front/back volume labels are invalid")
        if self.protected_truncation_distance_m <= 0.0:
            raise ValueError("Protected TSDF truncation distance must be positive")
        if any(value <= 0.0 or not np.isfinite(value) for value in self.feature_thicknesses_m):
            raise ValueError("Feature wall thicknesses must be finite and positive")


def _apply(matrix: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _integrate_side(
    side: int,
    view_points: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    origin: NDArray[np.float64],
    truncation: float,
    config: TSDFConfig,
) -> SparseTSDFVolume:
    index_chunks: list[NDArray[np.int64]] = []
    value_chunks: list[NDArray[np.float64]] = []
    sample_offsets = np.arange(
        -truncation, truncation + config.voxel_size_m * 0.5, config.voxel_size_m
    )
    for points, camera in view_points:
        rays = points - camera
        distances = np.linalg.norm(rays, axis=1)
        valid = distances > truncation
        rays = rays[valid] / distances[valid, None]
        surfaces = points[valid]
        if not len(surfaces):
            continue
        samples = surfaces[:, None, :] + rays[:, None, :] * sample_offsets[None, :, None]
        indices = np.floor((samples - origin) / config.voxel_size_m).astype(np.int64)
        values = np.broadcast_to(-sample_offsets[None, :] / truncation, indices.shape[:2])
        index_chunks.append(indices.reshape(-1, 3))
        value_chunks.append(np.clip(values.reshape(-1), -1.0, 1.0))
    if not index_chunks:
        raise TSDFError(f"No valid projective samples for side {side:+d}")
    indices = np.vstack(index_chunks)
    values = np.concatenate(value_chunks)
    unique, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    if len(unique) > config.maximum_voxels:
        raise TSDFError(
            f"Side {side:+d} TSDF requires {len(unique)} sparse voxels, exceeding "
            f"maximum_voxels={config.maximum_voxels}"
        )
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return SparseTSDFVolume(
        side,
        origin,
        config.voxel_size_m,
        truncation,
        unique.astype(np.int32),
        sums / counts,
        counts.astype(np.float64),
    )


_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int32,
)
_TETRAHEDRA = ((0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6))
_TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _volume_mesh(
    volume: SparseTSDFVolume, minimum_weight: float
) -> tuple[list[NDArray[np.float64]], list[tuple[int, int, int]], list[int]]:
    values = {
        tuple(index): (float(tsdf), float(weight))
        for index, tsdf, weight in zip(
            volume.voxel_indices, volume.tsdf, volume.weights, strict=True
        )
    }
    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []
    sides: list[int] = []
    vertex_lookup: dict[tuple[float, float, float], int] = {}
    bases: set[tuple[int, int, int]] = set()
    for index in values:
        base = np.asarray(index, dtype=np.int32)
        for corner in _CORNERS:
            bases.add(tuple(base - corner))

    def vertex_index(position: NDArray[np.float64]) -> int:
        key = tuple(np.round(position, 10))
        if key not in vertex_lookup:
            vertex_lookup[key] = len(vertices)
            vertices.append(position)
        return vertex_lookup[key]

    for base_tuple in bases:
        base = np.asarray(base_tuple, dtype=np.int32)
        corner_indices = base + _CORNERS
        samples = [values.get(tuple(index)) for index in corner_indices]
        if any(sample is None or sample[1] < minimum_weight for sample in samples):
            continue
        scalar = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        if np.all(scalar >= 0.0) or np.all(scalar < 0.0):
            continue
        positions = volume.origin_m + corner_indices * volume.voxel_size_m
        for tetra in _TETRAHEDRA:
            crossings: list[NDArray[np.float64]] = []
            for first, second in _TETRA_EDGES:
                a = tetra[first]
                b = tetra[second]
                va = scalar[a]
                vb = scalar[b]
                if (va < 0.0) == (vb < 0.0) or np.isclose(va, vb):
                    continue
                fraction = float(va / (va - vb))
                crossings.append(positions[a] + fraction * (positions[b] - positions[a]))
            if len(crossings) == 3:
                triangles.append(tuple(vertex_index(point) for point in crossings))
                sides.append(volume.side)
            elif len(crossings) == 4:
                ids = [vertex_index(point) for point in crossings]
                triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
                sides.extend((volume.side, volume.side))
    return vertices, triangles, sides


def extract_bilateral_mesh(
    front: SparseTSDFVolume,
    back: SparseTSDFVolume,
    minimum_weight: float = 1.0,
) -> TriangleMesh:
    """Extract front/back zero sets using marching tetrahedra."""

    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []
    sides: list[int] = []
    for volume in (front, back):
        local_vertices, local_triangles, local_sides = _volume_mesh(volume, minimum_weight)
        offset = len(vertices)
        vertices.extend(local_vertices)
        triangles.extend(
            tuple(index + offset for index in triangle) for triangle in local_triangles
        )
        sides.extend(local_sides)
    return TriangleMesh(
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
        np.asarray(sides, dtype=np.int8),
    )


def _open3d_mesh(
    fused: FusedBladeCloud,
    views: tuple[RegisteredCloudView, ...],
    corrections: dict[str, NDArray[np.float64]],
    truncation: float,
    config: TSDFConfig,
) -> TriangleMesh | None:
    """Use calibrated projective Open3D integration when metadata is available."""

    if not config.use_open3d_if_available or any(view.pixel_uv is None for view in views):
        return None
    try:
        import open3d as o3d
    except ImportError:
        return None
    volumes = {
        side: o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_size_m,
            sdf_trunc=truncation,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        for side in (-1, 1)
    }
    for view in views:
        if (
            view.pixel_uv is None
            or view.source_image_shape is None
            or view.intrinsic_matrix is None
            or view.base_t_camera_matrix is None
        ):
            return None
        height, width = view.source_image_shape
        camera_t_base = np.linalg.inv(view.base_t_camera_matrix)
        camera_points = _apply(camera_t_base, view.points_m)
        depth = np.zeros((height, width), dtype=np.float32)
        valid = camera_points[:, 2] > 0.0
        pixels = view.pixel_uv[valid]
        depths = camera_points[valid, 2].astype(np.float32)
        if not len(depths):
            continue
        depth[pixels[:, 1], pixels[:, 0]] = depths
        color = np.zeros((height, width, 3), dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=float(depths.max() + truncation),
            convert_rgb_to_intensity=False,
        )
        matrix = view.intrinsic_matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(matrix[0, 0]),
            float(matrix[1, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        )
        side = 1 if (view.camera_origin_m - fused.center_m) @ fused.axes[:, 2] >= 0.0 else -1
        corrected_base_t_camera = corrections.get(view.view_id, np.eye(4)) @ (
            view.base_t_camera_matrix
        )
        volumes[side].integrate(rgbd, intrinsic, np.linalg.inv(corrected_base_t_camera))
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    sides: list[np.ndarray] = []
    for side in (1, -1):
        mesh = volumes[side].extract_triangle_mesh()
        local_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        local_triangles = np.asarray(mesh.triangles, dtype=np.int32)
        if not len(local_triangles):
            continue
        offset = sum(len(item) for item in vertices)
        vertices.append(local_vertices)
        triangles.append(local_triangles + offset)
        sides.append(np.full(len(local_triangles), side, dtype=np.int8))
    if not triangles:
        return None
    return TriangleMesh(np.vstack(vertices), np.vstack(triangles), np.concatenate(sides))


def integrate_bilateral_tsdf(
    fused: FusedBladeCloud,
    views: tuple[RegisteredCloudView, ...],
    config: TSDFConfig,
    *,
    feature_thicknesses_m: tuple[float, ...] = (),
) -> BilateralTSDFResult:
    """Integrate side volumes with main-blade and observed-fin thin-wall protection."""

    if not views:
        raise TSDFError("TSDF integration requires registered views")
    thicknesses = (fused.median_thickness_m, *feature_thicknesses_m)
    protected = min(
        config.truncation_distance_m,
        min(thicknesses) * config.thin_wall_band_fraction,
    )
    if protected < config.voxel_size_m:
        raise TSDFError(
            "Thin-wall-protected truncation is below one voxel; reduce voxel size or "
            "provide a better-separated coarse model"
        )
    correction_by_id = {item.view_id: item.correction_matrix for item in fused.refinements}
    side_views: dict[int, list[tuple[NDArray[np.float64], NDArray[np.float64]]]] = {1: [], -1: []}
    for view in views:
        side = 1 if (view.camera_origin_m - fused.center_m) @ fused.axes[:, 2] >= 0.0 else -1
        correction = correction_by_id.get(view.view_id, np.eye(4))
        side_views[side].append(
            (_apply(correction, view.points_m), _apply(correction, view.camera_origin_m[None])[0])
        )
    if not side_views[1] or not side_views[-1]:
        raise TSDFError("TSDF requires at least one registered observation on each side")
    origin = fused.points_m.min(axis=0) - protected - config.voxel_size_m
    front = _integrate_side(1, side_views[1], origin, protected, config)
    back = _integrate_side(-1, side_views[-1], origin, protected, config)
    mesh = _open3d_mesh(fused, views, correction_by_id, protected, config)
    backend = "open3d_scalable" if mesh is not None else "numpy_sparse"
    if mesh is None:
        mesh = extract_bilateral_mesh(front, back, config.minimum_weight)
    return BilateralTSDFResult(
        front,
        back,
        mesh,
        protected,
        backend,
        tuple(float(value) for value in feature_thicknesses_m),
    )
