"""Deterministic pinhole-depth ray integration into sparse occupancy snapshots."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.diagnostics.performance_timing import (
    performance_span,
    performance_timed,
)
from biblade_fusion.mapping.occupancy import (
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancySnapshot,
    VoxelIndex,
)


class DepthIntegrationError(ValueError):
    """A depth observation cannot safely update the occupancy map."""


_COOPERATIVE_YIELD_RAY_BATCH = 32


@dataclass(frozen=True, slots=True)
class DepthIntegrationConfig:
    """Quality and sampling limits for one metric depth image."""

    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 3.0
    pixel_stride: int = 4
    minimum_valid_rays: int = 1
    free_space_margin_m: float = 0.01
    minimum_free_observations: int = 3
    minimum_free_view_translation_m: float = 0.02
    minimum_free_view_direction_deg: float = 5.0
    ray_integration_backend: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self) -> None:
        minimum = float(self.minimum_depth_m)
        maximum = float(self.maximum_depth_m)
        margin = float(self.free_space_margin_m)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_depth_m must be finite and non-negative")
        if not math.isfinite(maximum) or maximum <= minimum:
            raise ValueError("maximum_depth_m must be finite and exceed minimum_depth_m")
        if (
            isinstance(self.pixel_stride, bool)
            or not isinstance(self.pixel_stride, (int, np.integer))
            or self.pixel_stride < 1
        ):
            raise ValueError("pixel_stride must be a positive integer")
        if (
            isinstance(self.minimum_valid_rays, bool)
            or not isinstance(self.minimum_valid_rays, (int, np.integer))
            or self.minimum_valid_rays < 1
        ):
            raise ValueError("minimum_valid_rays must be a positive integer")
        if not math.isfinite(margin) or margin < 0.0 or margin >= maximum:
            raise ValueError(
                "free_space_margin_m must be finite, non-negative, and below max depth"
            )
        if (
            isinstance(self.minimum_free_observations, bool)
            or not isinstance(self.minimum_free_observations, (int, np.integer))
            or self.minimum_free_observations < 2
        ):
            raise ValueError(
                "minimum_free_observations must be an integer of at least two"
            )
        view_translation = float(self.minimum_free_view_translation_m)
        view_direction = float(self.minimum_free_view_direction_deg)
        backend = str(self.ray_integration_backend).strip().lower()
        if backend not in {"cpu", "cuda"}:
            raise ValueError("ray_integration_backend must be 'cpu' or 'cuda'")
        if not math.isfinite(view_translation) or view_translation <= 0.0:
            raise ValueError(
                "minimum_free_view_translation_m must be finite and positive"
            )
        if (
            not math.isfinite(view_direction)
            or view_direction <= 0.0
            or view_direction > 180.0
        ):
            raise ValueError(
                "minimum_free_view_direction_deg must be finite in (0, 180]"
            )
        object.__setattr__(self, "minimum_depth_m", minimum)
        object.__setattr__(self, "maximum_depth_m", maximum)
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))
        object.__setattr__(self, "minimum_valid_rays", int(self.minimum_valid_rays))
        object.__setattr__(self, "free_space_margin_m", margin)
        object.__setattr__(
            self,
            "minimum_free_observations",
            int(self.minimum_free_observations),
        )
        object.__setattr__(
            self,
            "minimum_free_view_translation_m",
            view_translation,
        )
        object.__setattr__(
            self,
            "minimum_free_view_direction_deg",
            view_direction,
        )
        object.__setattr__(self, "ray_integration_backend", backend)


class DepthRayIntegrator:
    """Fuse settled, rectified depth frames using occupied-wins semantics.

    Occupied observations are monotonic: a later free-space ray cannot erase an
    occupied cell.  This deliberately conservative rule avoids unsafe clearing
    from reflective dropouts or small pose errors.  Every update returns a new
    immutable ``MAPPING`` snapshot and invalidates any previous quality evidence.
    """

    def __init__(
        self,
        grid: OccupancyGridSpec,
        config: DepthIntegrationConfig | None = None,
        *,
        mapping_context_hash: str,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", mapping_context_hash) is None:
            raise ValueError("mapping_context_hash must be a lowercase SHA-256 hex digest")
        self.grid = grid
        self.config = config or DepthIntegrationConfig()
        self.mapping_context_hash = mapping_context_hash

    @performance_timed("occupancy.depth_ray_integrator")
    def integrate(
        self,
        snapshot: OccupancySnapshot | None,
        depth_m: ArrayLike,
        intrinsics: CameraIntrinsics,
        base_t_camera: PoseSE3,
        *,
        valid_mask: ArrayLike | None = None,
        source_view_id: str,
        observed_at_utc: datetime | None = None,
    ) -> OccupancySnapshot:
        """Integrate one axial-depth image expressed in metres.

        Invalid or masked pixels do not clear rays; they remain unknown.  Valid
        rays mark traversed cells free up to a configurable surface margin, then
        mark an in-grid hit cell occupied.
        """

        source_id = str(source_view_id).strip()
        if not source_id:
            raise DepthIntegrationError("source_view_id must be non-empty")
        previous = self._validate_snapshot(snapshot)
        if source_id in previous.source_view_ids:
            raise DepthIntegrationError(f"source_view_id already integrated: {source_id}")
        self._validate_pose(base_t_camera)
        camera_center, camera_axis = self._camera_view_evidence(base_t_camera)
        self._validate_geometric_view_independence(
            previous,
            camera_center,
            camera_axis,
        )
        depth, selected_v, selected_u = self._select_depth_rays(
            depth_m,
            intrinsics,
            valid_mask,
        )

        z = depth[selected_v, selected_u]
        x = (selected_u.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
        y = (selected_v.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
        camera_points = np.column_stack((x, y, z))
        base_points = base_t_camera.transform_points(camera_points)
        camera_origin = base_t_camera.translation_m

        if self.config.ray_integration_backend == "cuda":
            with performance_span("occupancy.cuda_ray_integration"):
                observed_indices, new_free, new_occupied = _integrate_rays_cuda(
                    base_points,
                    camera_origin,
                    self.grid,
                    self.config.free_space_margin_m,
                )
        else:
            with performance_span("occupancy.cpu_ray_integration"):
                observed_indices, new_free, new_occupied = _integrate_rays_cpu(
                    base_points,
                    camera_origin,
                    self.grid,
                    self.config.free_space_margin_m,
                )

        if not observed_indices:
            raise DepthIntegrationError("No valid depth ray intersects the occupancy grid")

        # Static-map safety rule: occupied evidence is never ray-cleared.  New
        # occupied endpoints also dominate any old or newly observed free state.
        occupied = previous.occupied_indices | frozenset(new_occupied)
        free_counts = dict(previous.free_observation_counts)
        for index in new_free - occupied:
            free_counts[index] = free_counts.get(index, 0) + 1
        free_observation_counts = tuple(sorted(free_counts.items()))
        free = frozenset(
            index
            for index, count in free_observation_counts
            if count >= self.config.minimum_free_observations and index not in occupied
        )
        observed = observed_at_utc or datetime.now(UTC)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise DepthIntegrationError("observed_at_utc must be timezone-aware")
        observed = observed.astimezone(UTC)
        if snapshot is not None and observed < previous.created_at_utc:
            raise DepthIntegrationError("Depth observations must be integrated in UTC order")
        return OccupancySnapshot(
            frame_id=self.grid.frame_id,
            voxel_size_m=self.grid.voxel_size_m,
            origin_m=self.grid.origin_m,
            grid_shape=self.grid.grid_shape,
            free_indices=free,
            free_observation_counts=free_observation_counts,
            minimum_free_observations=self.config.minimum_free_observations,
            minimum_free_view_translation_m=(
                self.config.minimum_free_view_translation_m
            ),
            minimum_free_view_direction_deg=(
                self.config.minimum_free_view_direction_deg
            ),
            occupied_indices=occupied,
            sequence=previous.sequence + 1,
            created_at_utc=observed,
            source_view_ids=(*previous.source_view_ids, source_id),
            source_camera_centres_base_m=(
                *previous.source_camera_centres_base_m,
                camera_center,
            ),
            source_camera_axes_base=(
                *previous.source_camera_axes_base,
                camera_axis,
            ),
            rebuild_started_at_utc=(
                observed
                if snapshot is None
                else previous.rebuild_started_at_utc
            ),
            map_state=OccupancyMapState.MAPPING,
            mapping_context_hash=self.mapping_context_hash,
            parent_evidence_hash=previous.quality_evidence_hash,
            quality_evidence_hash=None,
            state_reason=f"integrated {source_id}; awaiting self-mask and depth quality gates",
        )

    def _validate_snapshot(self, snapshot: OccupancySnapshot | None) -> OccupancySnapshot:
        if snapshot is None:
            return self.grid.empty_snapshot(
                created_at_utc=datetime.now(UTC),
                minimum_free_observations=self.config.minimum_free_observations,
                minimum_free_view_translation_m=(
                    self.config.minimum_free_view_translation_m
                ),
                minimum_free_view_direction_deg=(
                    self.config.minimum_free_view_direction_deg
                ),
            )
        if snapshot.geometry_spec() != self.grid:
            raise DepthIntegrationError(
                "Occupancy snapshot geometry does not match integrator grid"
            )
        if snapshot.mapping_context_hash != self.mapping_context_hash:
            raise DepthIntegrationError(
                "Occupancy snapshot mapping context does not match this integrator"
            )
        if (
            snapshot.minimum_free_observations
            != self.config.minimum_free_observations
        ):
            raise DepthIntegrationError(
                "Occupancy snapshot free-observation threshold does not match this integrator"
            )
        if not math.isclose(
            snapshot.minimum_free_view_translation_m,
            self.config.minimum_free_view_translation_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            snapshot.minimum_free_view_direction_deg,
            self.config.minimum_free_view_direction_deg,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DepthIntegrationError(
                "Occupancy snapshot view-independence thresholds do not match this integrator"
            )
        if snapshot.map_state not in {
            OccupancyMapState.MAPPING,
            OccupancyMapState.MAP_READY,
        }:
            raise DepthIntegrationError(
                f"Cannot continue integration from {snapshot.map_state.value}"
            )
        if snapshot.source_view_ids and snapshot.quality_evidence_hash is None:
            raise DepthIntegrationError(
                "Previous occupancy snapshot has no bound evidence chain"
            )
        return snapshot

    @staticmethod
    def _validate_pose(base_t_camera: PoseSE3) -> None:
        if (
            base_t_camera.parent_frame != "base"
            or base_t_camera.child_frame != "left_rectified"
        ):
            raise DepthIntegrationError(
                "Depth integration requires base_T_left_rectified"
            )

    @staticmethod
    def _camera_view_evidence(
        base_t_camera: PoseSE3,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        center = tuple(float(value) for value in base_t_camera.translation_m)
        axis = tuple(float(value) for value in base_t_camera.rotation[:, 2])
        return center, axis

    def _validate_geometric_view_independence(
        self,
        previous: OccupancySnapshot,
        camera_center: tuple[float, float, float],
        camera_axis: tuple[float, float, float],
    ) -> None:
        center = np.asarray(camera_center, dtype=np.float64)
        axis = np.asarray(camera_axis, dtype=np.float64)
        for source_id, old_center_raw, old_axis_raw in zip(
            previous.source_view_ids,
            previous.source_camera_centres_base_m,
            previous.source_camera_axes_base,
            strict=True,
        ):
            old_center = np.asarray(old_center_raw, dtype=np.float64)
            old_axis = np.asarray(old_axis_raw, dtype=np.float64)
            translation_m = float(np.linalg.norm(center - old_center))
            cosine = float(np.clip(np.dot(axis, old_axis), -1.0, 1.0))
            direction_deg = math.degrees(math.acos(cosine))
            if (
                translation_m < self.config.minimum_free_view_translation_m
                and direction_deg < self.config.minimum_free_view_direction_deg
            ):
                raise DepthIntegrationError(
                    "Depth view is not geometrically independent from accepted "
                    f"source {source_id}: translation={translation_m:.9f} m, "
                    f"direction={direction_deg:.9f} deg"
                )

    def _select_depth_rays(
        self,
        depth_m: ArrayLike,
        intrinsics: CameraIntrinsics,
        valid_mask: ArrayLike | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        depth = np.asarray(depth_m, dtype=np.float64)
        expected_shape = (intrinsics.height, intrinsics.width)
        if depth.shape != expected_shape:
            raise DepthIntegrationError(
                f"Depth shape {depth.shape} does not match intrinsics {expected_shape}"
            )
        model = intrinsics.distortion_model.lower().split(".")[-1]
        is_rectified = model in {"none", "distortion_none"} or not any(
            intrinsics.distortion_coefficients
        )
        if not is_rectified:
            raise DepthIntegrationError(
                "Depth ray integration requires rectified intrinsics; "
                f"got {intrinsics.distortion_model}"
            )
        valid = (
            np.isfinite(depth)
            & (depth >= self.config.minimum_depth_m)
            & (depth <= self.config.maximum_depth_m)
        )
        if valid_mask is not None:
            supplied = np.asarray(valid_mask)
            if supplied.shape != expected_shape:
                raise DepthIntegrationError("valid_mask must match the depth image")
            if supplied.dtype != np.bool_:
                raise DepthIntegrationError("valid_mask must contain boolean values")
            valid &= supplied

        stride = self.config.pixel_stride
        sampled_v, sampled_u = np.nonzero(valid[::stride, ::stride])
        selected_v = sampled_v.astype(np.int64) * stride
        selected_u = sampled_u.astype(np.int64) * stride
        if selected_u.size < self.config.minimum_valid_rays:
            raise DepthIntegrationError(
                f"Depth image has {selected_u.size} valid sampled rays; "
                f"at least {self.config.minimum_valid_rays} are required"
            )
        return depth, selected_v, selected_u


@dataclass(frozen=True, slots=True)
class _RayTraversalState:
    current: VoxelIndex
    target: VoxelIndex
    step: VoxelIndex
    t_max: tuple[float, float, float]
    t_delta: tuple[float, float, float]
    maximum_steps: int


def _ray_endpoint(
    hit_point: NDArray[np.float64],
    camera_origin: NDArray[np.float64],
    grid: OccupancyGridSpec,
    free_space_margin_m: float,
) -> tuple[VoxelIndex | None, NDArray[np.float64] | None]:
    """Return the occupied endpoint and the conservative free-segment endpoint."""

    hit_index = _world_to_index(hit_point, grid)
    direction = hit_point - camera_origin
    ray_length = float(np.linalg.norm(direction))
    if ray_length <= free_space_margin_m:
        return hit_index, None
    return (
        hit_index,
        hit_point - direction / ray_length * free_space_margin_m,
    )


def _integrate_rays_cpu(
    base_points: NDArray[np.float64],
    camera_origin: NDArray[np.float64],
    grid: OccupancyGridSpec,
    free_space_margin_m: float,
) -> tuple[set[VoxelIndex], set[VoxelIndex], set[VoxelIndex]]:
    """Preserve the original deterministic Python DDA implementation."""

    observed_indices: set[VoxelIndex] = set()
    new_free: set[VoxelIndex] = set()
    new_occupied: set[VoxelIndex] = set()
    for ray_number, hit_point in enumerate(base_points, start=1):
        if ray_number % _COOPERATIVE_YIELD_RAY_BATCH == 0:
            # The safety sampler shares this interpreter. Explicitly release the
            # GIL during the long deterministic CPU ray loop.
            time.sleep(0)
        hit_index, free_end = _ray_endpoint(
            hit_point,
            camera_origin,
            grid,
            free_space_margin_m,
        )
        if hit_index is not None:
            new_occupied.add(hit_index)
            observed_indices.add(hit_index)
        if free_end is None:
            continue
        for index in _ray_voxel_indices(camera_origin, free_end, grid):
            if not _index_in_bounds(index, grid.grid_shape):
                continue
            observed_indices.add(index)
            if index != hit_index:
                new_free.add(index)
    return observed_indices, new_free, new_occupied


def _torch_module() -> Any:
    try:
        return import_module("torch")
    except Exception as exc:
        raise DepthIntegrationError(
            "CUDA ray integration requires the pinned PyTorch runtime"
        ) from exc


def _integrate_rays_cuda(
    base_points: NDArray[np.float64],
    camera_origin: NDArray[np.float64],
    grid: OccupancyGridSpec,
    free_space_margin_m: float,
) -> tuple[set[VoxelIndex], set[VoxelIndex], set[VoxelIndex]]:
    """Traverse all free-space segments on CUDA with exact occupancy semantics.

    CPU preparation intentionally uses the same scalar endpoint and DDA-state
    construction as ``_ray_voxel_indices``. CUDA advances those states and performs
    idempotent writes into a per-source dense bitmap. The bitmap is converted back to
    a set before the unchanged occupied-wins merge.
    """

    torch = _torch_module()
    try:
        if not bool(torch.cuda.is_available()):
            raise DepthIntegrationError(
                "CUDA ray integration was requested but torch.cuda.is_available() is false"
            )
        device = torch.device("cuda")
        states: list[_RayTraversalState] = []
        traversal_hits: list[VoxelIndex | None] = []
        new_occupied: set[VoxelIndex] = set()
        for ray_number, hit_point in enumerate(base_points, start=1):
            if ray_number % _COOPERATIVE_YIELD_RAY_BATCH == 0:
                time.sleep(0)
            hit_index, free_end = _ray_endpoint(
                hit_point,
                camera_origin,
                grid,
                free_space_margin_m,
            )
            if hit_index is not None:
                new_occupied.add(hit_index)
            if free_end is None:
                continue
            states.append(_ray_traversal_state(camera_origin, free_end, grid))
            traversal_hits.append(hit_index)

        if not states:
            return set(new_occupied), set(), new_occupied

        current_host = np.asarray([state.current for state in states], dtype=np.int64)
        target_host = np.asarray([state.target for state in states], dtype=np.int64)
        step_host = np.asarray([state.step for state in states], dtype=np.int64)
        t_max_host = np.asarray([state.t_max for state in states], dtype=np.float64)
        t_delta_host = np.asarray([state.t_delta for state in states], dtype=np.float64)
        maximum_steps_host = np.asarray(
            [state.maximum_steps for state in states],
            dtype=np.int64,
        )
        no_hit = np.iinfo(np.int64).min
        hit_host = np.asarray(
            [
                hit_index if hit_index is not None else (no_hit, no_hit, no_hit)
                for hit_index in traversal_hits
            ],
            dtype=np.int64,
        )

        with torch.inference_mode():
            current = torch.from_numpy(current_host).to(device=device)
            target = torch.from_numpy(target_host).to(device=device)
            step = torch.from_numpy(step_host).to(device=device)
            t_max = torch.from_numpy(t_max_host).to(device=device)
            t_delta = torch.from_numpy(t_delta_host).to(device=device)
            maximum_steps = torch.from_numpy(maximum_steps_host).to(device=device)
            hit = torch.from_numpy(hit_host).to(device=device)
            shape = torch.tensor(grid.grid_shape, dtype=torch.int64, device=device)
            voxel_count = math.prod(grid.grid_shape)
            free_bitmap = torch.zeros(voxel_count, dtype=torch.bool, device=device)

            def mark_free(indices: Any, rows: Any) -> None:
                in_bounds = torch.all((indices >= 0) & (indices < shape), dim=1)
                differs_from_hit = torch.any(indices != hit, dim=1)
                selected = rows & in_bounds & differs_from_hit
                linear = (
                    (indices[:, 0] * grid.grid_shape[1] + indices[:, 1])
                    * grid.grid_shape[2]
                    + indices[:, 2]
                )
                free_bitmap.index_fill_(0, linear[selected], True)

            all_rows = torch.ones(current.shape[0], dtype=torch.bool, device=device)
            mark_free(current, all_rows)
            unreached = torch.any(current != target, dim=1)
            maximum_iterations = int(maximum_steps_host.max(initial=0))
            for iteration in range(maximum_iterations):
                stepping = unreached & (iteration < maximum_steps)
                next_t = torch.min(t_max, dim=1).values
                tolerance = 1e-12 * torch.maximum(
                    torch.ones_like(next_t),
                    torch.abs(next_t),
                )
                advance = stepping[:, None] & (
                    t_max <= next_t[:, None] + tolerance[:, None]
                )
                current = current + step * advance
                t_max = torch.where(advance, t_max + t_delta, t_max)
                mark_free(current, stepping)
                reached_now = torch.all(current == target, dim=1)
                unreached = unreached & ~reached_now

            torch.cuda.synchronize(device)
            if bool(torch.any(unreached).item()):
                raise DepthIntegrationError(
                    "CUDA voxel ray traversal failed to reach one or more endpoints"
                )
            free_linear = (
                torch.nonzero(free_bitmap, as_tuple=False)
                .flatten()
                .to(device="cpu")
                .numpy()
            )

        yz = grid.grid_shape[1] * grid.grid_shape[2]
        z_count = grid.grid_shape[2]
        new_free = {
            (int(linear // yz), int((linear % yz) // z_count), int(linear % z_count))
            for linear in free_linear
        }
        observed = new_free | new_occupied
        return observed, new_free, new_occupied
    except DepthIntegrationError:
        raise
    except Exception as exc:
        raise DepthIntegrationError(
            f"CUDA ray integration failed closed: {type(exc).__name__}: {exc}"
        ) from exc


def _world_to_index(point_m: Sequence[float], grid: OccupancyGridSpec) -> VoxelIndex | None:
    index = tuple(
        math.floor((float(point_m[axis]) - grid.origin_m[axis]) / grid.voxel_size_m)
        for axis in range(3)
    )
    return index if _index_in_bounds(index, grid.grid_shape) else None


def _index_in_bounds(index: Sequence[int], shape: Sequence[int]) -> bool:
    return all(0 <= int(index[axis]) < int(shape[axis]) for axis in range(3))


def _ray_voxel_indices(
    start_m: Sequence[float],
    end_m: Sequence[float],
    grid: OccupancyGridSpec,
) -> list[VoxelIndex]:
    """Traverse a finite segment with deterministic Amanatides-Woo DDA."""

    state = _ray_traversal_state(start_m, end_m, grid)
    current = list(state.current)
    target = list(state.target)
    result: list[VoxelIndex] = [(current[0], current[1], current[2])]
    if current == target:
        return result
    step = list(state.step)
    t_max = list(state.t_max)
    t_delta = list(state.t_delta)
    for _ in range(state.maximum_steps):
        next_t = min(t_max)
        tolerance = 1e-12 * max(1.0, abs(next_t))
        for axis in range(3):
            if t_max[axis] <= next_t + tolerance:
                current[axis] += step[axis]
                t_max[axis] += t_delta[axis]
        index = (current[0], current[1], current[2])
        result.append(index)
        if current == target:
            return result
    raise DepthIntegrationError("Voxel ray traversal failed to reach its endpoint")


def _ray_traversal_state(
    start_m: Sequence[float],
    end_m: Sequence[float],
    grid: OccupancyGridSpec,
) -> _RayTraversalState:
    """Construct the scalar float64 state shared by CPU and CUDA DDA."""

    start = np.asarray(start_m, dtype=np.float64)
    end = np.asarray(end_m, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or not np.isfinite((start, end)).all():
        raise DepthIntegrationError("Ray endpoints must be finite three-vectors")
    delta = end - start
    current = tuple(
        math.floor((start[axis] - grid.origin_m[axis]) / grid.voxel_size_m)
        for axis in range(3)
    )
    target = tuple(
        math.floor((end[axis] - grid.origin_m[axis]) / grid.voxel_size_m)
        for axis in range(3)
    )
    step = [0, 0, 0]
    t_max = [math.inf, math.inf, math.inf]
    t_delta = [math.inf, math.inf, math.inf]
    for axis in range(3):
        component = float(delta[axis])
        if component > 0.0:
            step[axis] = 1
            boundary = grid.origin_m[axis] + (current[axis] + 1) * grid.voxel_size_m
            t_max[axis] = (boundary - start[axis]) / component
            t_delta[axis] = grid.voxel_size_m / component
        elif component < 0.0:
            step[axis] = -1
            boundary = grid.origin_m[axis] + current[axis] * grid.voxel_size_m
            t_max[axis] = (boundary - start[axis]) / component
            t_delta[axis] = -grid.voxel_size_m / component

    return _RayTraversalState(
        current=current,
        target=target,
        step=(step[0], step[1], step[2]),
        t_max=(t_max[0], t_max[1], t_max[2]),
        t_delta=(t_delta[0], t_delta[1], t_delta[2]),
        maximum_steps=(
            sum(abs(target[axis] - current[axis]) for axis in range(3)) + 3
        ),
    )
