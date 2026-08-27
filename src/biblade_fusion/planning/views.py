"""Partition both proxy faces and generate normal-facing candidate views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import atan, ceil, tan

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ViewPlanningConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.proxy import BilateralBladeProxy


class ViewPlanningError(ValueError):
    """The proxy and camera model cannot produce a safe bounded view plan."""


class BladeSide(StrEnum):
    FRONT = "front"
    BACK = "back"


def _readonly_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class SurfacePatch:
    patch_id: str
    side: BladeSide
    row: int
    column: int
    target_m: NDArray[np.float64]
    outward_normal: NDArray[np.float64]
    planar_extents_m: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.patch_id:
            raise ValueError("Patch ID must be non-empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("Patch indices must be non-negative")
        if self.planar_extents_m[0] <= 0.0 or self.planar_extents_m[1] <= 0.0:
            raise ValueError("Patch planar extents must be positive")
        target = _readonly_vector(self.target_m, "Patch target")
        normal = _readonly_vector(self.outward_normal, "Patch normal")
        norm = np.linalg.norm(normal)
        if not np.isclose(norm, 1.0, atol=1e-7):
            raise ValueError("Patch normal must be a unit vector")
        object.__setattr__(self, "target_m", target)
        object.__setattr__(self, "outward_normal", normal)


@dataclass(frozen=True, slots=True)
class CandidateView:
    view_id: str
    patch: SurfacePatch
    base_t_left_ir: PoseSE3
    standoff_distance_m: float
    footprint_m: tuple[float, float]
    projection_fraction: float = 1.0
    visibility_fraction: float = 1.0
    distance_policy: str = "fixed_baseline"

    def __post_init__(self) -> None:
        if self.base_t_left_ir.parent_frame != "base":
            raise ValueError("Candidate camera pose parent frame must be base")
        if self.standoff_distance_m <= 0.0:
            raise ValueError("Candidate standoff must be positive")
        if self.footprint_m[0] <= 0.0 or self.footprint_m[1] <= 0.0:
            raise ValueError("Candidate footprint must be positive")
        if not 0.0 <= self.projection_fraction <= 1.0:
            raise ValueError("Candidate projection fraction must lie in [0, 1]")
        if not 0.0 <= self.visibility_fraction <= 1.0:
            raise ValueError("Candidate visibility fraction must lie in [0, 1]")
        if not self.distance_policy:
            raise ValueError("Candidate distance policy must be non-empty")

    @property
    def optical_axis(self) -> NDArray[np.float64]:
        """Camera +Z axis expressed in the base frame."""

        return self.base_t_left_ir.rotation[:, 2]


@dataclass(frozen=True, slots=True)
class BilateralViewPlan:
    candidates: tuple[CandidateView, ...]
    rows: int
    columns: int
    footprint_m: tuple[float, float]
    effective_surface_extents_m: tuple[float, float]

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("View-plan grid dimensions must be positive")
        expected = 2 * self.rows * self.columns
        if len(self.candidates) != expected:
            raise ValueError(f"Expected {expected} bilateral candidates")

    def for_side(self, side: BladeSide) -> tuple[CandidateView, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.patch.side is side)


def _field_of_view(width: int, principal: float, focal: float) -> float:
    """Full pinhole field of view allowing an off-center principal point."""

    negative_extent = principal
    positive_extent = (width - 1) - principal
    if negative_extent <= 0.0 or positive_extent <= 0.0:
        raise ViewPlanningError("Camera principal point must lie inside the image")
    return atan(negative_extent / focal) + atan(positive_extent / focal)


def _grid_count(surface_extent: float, footprint: float, overlap: float) -> int:
    tolerance = max(surface_extent, footprint) * 1e-12
    if surface_extent <= footprint + tolerance:
        return 1
    step = footprint * (1.0 - overlap)
    ratio = (surface_extent - footprint) / step
    return ceil(ratio - 1e-12) + 1


def _cell_centers(extent: float, count: int) -> NDArray[np.float64]:
    cell_extent = extent / count
    return np.linspace(-extent / 2.0 + cell_extent / 2.0, extent / 2.0 - cell_extent / 2.0, count)


def generate_bilateral_view_plan(
    proxy: BilateralBladeProxy,
    intrinsics: CameraIntrinsics,
    config: ViewPlanningConfig,
) -> BilateralViewPlan:
    """Generate one normal-facing camera candidate per patch on both proxy faces."""

    if config.standoff_distance_m is None:
        raise ViewPlanningError("standoff_distance_m must be configured before view planning")
    standoff = config.standoff_distance_m
    horizontal_fov = _field_of_view(intrinsics.width, intrinsics.cx, intrinsics.fx)
    vertical_fov = _field_of_view(intrinsics.height, intrinsics.cy, intrinsics.fy)
    footprint = (
        2.0 * standoff * tan(horizontal_fov / 2.0) * config.footprint_utilization,
        2.0 * standoff * tan(vertical_fov / 2.0) * config.footprint_utilization,
    )
    if not np.isfinite(footprint).all() or min(footprint) <= 0.0:
        raise ViewPlanningError("Camera model produced an invalid surface footprint")
    if any(config.edge_margin_m > axis_footprint / 2.0 for axis_footprint in footprint):
        raise ViewPlanningError(
            "edge_margin_m exceeds half the usable camera footprint; patch targets "
            "would leave the proxy face"
        )

    effective_extents = (
        float(proxy.extents_m[0] + 2.0 * config.edge_margin_m),
        float(proxy.extents_m[1] + 2.0 * config.edge_margin_m),
    )
    columns = _grid_count(effective_extents[0], footprint[0], config.overlap_fraction)
    rows = _grid_count(effective_extents[1], footprint[1], config.overlap_fraction)
    total_candidates = 2 * rows * columns
    if total_candidates > config.maximum_candidates:
        raise ViewPlanningError(
            f"Bilateral grid requires {total_candidates} candidates, exceeding "
            f"maximum_candidates={config.maximum_candidates}"
        )

    major_axis, minor_axis, front_normal = proxy.axes.T
    major_centers = _cell_centers(effective_extents[0], columns)
    minor_centers = _cell_centers(effective_extents[1], rows)
    cell_extents = (effective_extents[0] / columns, effective_extents[1] / rows)
    candidates: list[CandidateView] = []
    for side, sign in ((BladeSide.FRONT, 1.0), (BladeSide.BACK, -1.0)):
        outward_normal = sign * front_normal
        face_center = proxy.center_m + outward_normal * proxy.extents_m[2] / 2.0
        camera_z = -outward_normal
        camera_x = major_axis
        camera_y = np.cross(camera_z, camera_x)
        camera_y /= np.linalg.norm(camera_y)
        rotation = np.column_stack((camera_x, camera_y, camera_z))

        for row, minor_offset in enumerate(minor_centers):
            for column, major_offset in enumerate(major_centers):
                patch_id = f"{side.value}_r{row:02d}_c{column:02d}"
                target = face_center + major_axis * major_offset + minor_axis * minor_offset
                patch = SurfacePatch(
                    patch_id,
                    side,
                    row,
                    column,
                    target,
                    outward_normal,
                    cell_extents,
                )
                camera_position = target + outward_normal * standoff
                pose = PoseSE3.from_rotation_translation(
                    "base",
                    f"{patch_id}_left_ir",
                    rotation,
                    camera_position,
                )
                candidates.append(CandidateView(patch_id, patch, pose, standoff, footprint))
    return BilateralViewPlan(
        tuple(candidates),
        rows,
        columns,
        footprint,
        effective_extents,
    )
