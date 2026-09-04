"""Score, deduplicate, and endpoint-filter geometric candidate views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import acos, degrees
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.planning_deadline import (
    PlanningDeadlineExceeded,
    require_planning_time,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AxisAlignedBoxConfig, ViewFilterConfig
from biblade_fusion.diagnostics.performance_timing import (
    performance_span,
    performance_timed,
)
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning.views import CandidateView


class ReachabilityState(StrEnum):
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    state: ReachabilityState
    message: str
    joint_positions_rad: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.joint_positions_rad is None:
            if self.state is ReachabilityState.REACHABLE:
                raise ValueError(
                    "Reachable endpoint result requires a concrete joint solution"
                )
            return
        joints = np.array(self.joint_positions_rad, dtype=np.float64, copy=True)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Reachability joint solution must be a finite six-vector")
        if self.state is not ReachabilityState.REACHABLE:
            raise ValueError(
                "Only a reachable endpoint may carry a joint solution"
            )
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions_rad", joints)


@runtime_checkable
class ReachabilityChecker(Protocol):
    """Read-only endpoint IK boundary; implementations must not command motion."""

    def check(self, base_t_left_ir: PoseSE3) -> ReachabilityResult: ...


class CandidateStatus(StrEnum):
    REJECTED = "rejected"
    GEOMETRY_ONLY = "geometry_only"
    ENDPOINT_FEASIBLE = "endpoint_feasible"


@dataclass(frozen=True, slots=True)
class BladeClearanceEnvelope:
    """Conservative full-surface OBB used only for camera clearance."""

    frame_t_envelope: PoseSE3
    extents_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.frame_t_envelope.parent_frame != "base":
            raise ValueError("Blade clearance envelope must be expressed in base")
        extents = np.array(self.extents_m, dtype=np.float64, copy=True)
        if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(
            extents <= 0.0
        ):
            raise ValueError("Blade clearance envelope extents must be positive")
        extents.setflags(write=False)
        object.__setattr__(self, "extents_m", extents)


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    look_at_cosine: float
    incidence_cosine: float
    coverage_ratio: float
    view_distance_m: float
    standoff_error_m: float
    proxy_clearance_m: float
    geometric_score: float


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    candidate: CandidateView
    status: CandidateStatus
    metrics: CandidateMetrics
    reasons: tuple[str, ...]
    joint_positions_rad: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.joint_positions_rad is None:
            return
        joints = np.array(self.joint_positions_rad, dtype=np.float64, copy=True)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Evaluated candidate joints must be a finite six-vector")
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions_rad", joints)


@dataclass(frozen=True, slots=True)
class FilteredViewPlan:
    candidates: tuple[EvaluatedCandidate, ...]
    duplicate_view_ids: tuple[str, ...]

    @property
    def accepted(self) -> tuple[EvaluatedCandidate, ...]:
        """Candidates not rejected geometrically; this does not authorize motion."""

        return tuple(
            item for item in self.candidates if item.status is not CandidateStatus.REJECTED
        )

    @property
    def endpoint_feasible(self) -> tuple[EvaluatedCandidate, ...]:
        return tuple(
            item for item in self.candidates if item.status is CandidateStatus.ENDPOINT_FEASIBLE
        )

    @property
    def motion_authorized(self) -> bool:
        """Endpoint filtering never proves trajectory safety or authorizes motion."""

        return False


def _rotation_distance_deg(first: PoseSE3, second: PoseSE3) -> float:
    relative = first.rotation.T @ second.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _point_inside_workspace(position: np.ndarray, box: AxisAlignedBoxConfig, radius: float) -> bool:
    lower = np.asarray(box.minimum_m) + radius
    upper = np.asarray(box.maximum_m) - radius
    return bool(np.all(position >= lower) and np.all(position <= upper))


def _sphere_intersects_box(position: np.ndarray, box: AxisAlignedBoxConfig, radius: float) -> bool:
    lower = np.asarray(box.minimum_m)
    upper = np.asarray(box.maximum_m)
    closest = np.clip(position, lower, upper)
    return bool(np.linalg.norm(position - closest) <= radius)


def _proxy_clearance(
    position: np.ndarray,
    proxy: BilateralBladeProxy | BladeClearanceEnvelope,
) -> float:
    frame = (
        proxy.frame_T_proxy
        if isinstance(proxy, BilateralBladeProxy)
        else proxy.frame_t_envelope
    )
    local = frame.inverse().transform_points(position)
    outside = np.maximum(np.abs(local) - proxy.extents_m / 2.0, 0.0)
    return float(np.linalg.norm(outside))


def _metrics(
    candidate: CandidateView,
    proxy: BilateralBladeProxy | BladeClearanceEnvelope,
    projection_pose: PoseSE3 | None = None,
) -> CandidateMetrics:
    geometry_pose = projection_pose or candidate.base_t_left_ir
    camera_position = geometry_pose.translation_m
    view_vector = candidate.patch.target_m - camera_position
    view_distance = float(np.linalg.norm(view_vector))
    if view_distance <= 1e-12:
        look_at_cosine = -1.0
    else:
        look_at_cosine = float(
            geometry_pose.rotation[:, 2] @ (view_vector / view_distance)
        )
    incidence_cosine = float(
        (-geometry_pose.rotation[:, 2]) @ candidate.patch.outward_normal
    )
    coverage_ratio = float(
        min(1.0, candidate.footprint_m[0] / candidate.patch.planar_extents_m[0])
        * min(1.0, candidate.footprint_m[1] / candidate.patch.planar_extents_m[1])
    )
    # Clearance and workspace checks remain tied to the physical raw camera pose;
    # only the projection geometry may use the virtual rectified camera frame.
    proxy_clearance_m = _proxy_clearance(
        candidate.base_t_left_ir.translation_m,
        proxy,
    )
    standoff_error_m = abs(view_distance - candidate.standoff_distance_m)
    geometric_score = float(
        np.clip(0.4 * look_at_cosine + 0.4 * incidence_cosine + 0.2 * coverage_ratio, 0, 1)
    )
    return CandidateMetrics(
        look_at_cosine,
        incidence_cosine,
        coverage_ratio,
        view_distance,
        standoff_error_m,
        proxy_clearance_m,
        geometric_score,
    )


def _is_duplicate(
    candidate: CandidateView,
    retained: EvaluatedCandidate,
    config: ViewFilterConfig,
) -> bool:
    translation_distance = np.linalg.norm(
        candidate.base_t_left_ir.translation_m - retained.candidate.base_t_left_ir.translation_m
    )
    rotation_distance = _rotation_distance_deg(
        candidate.base_t_left_ir,
        retained.candidate.base_t_left_ir,
    )
    return bool(
        translation_distance <= config.duplicate_translation_tolerance_m
        and rotation_distance <= config.duplicate_rotation_tolerance_deg
    )


@performance_timed("planning.filter_candidate_views")
def filter_candidate_views(
    candidates: tuple[CandidateView, ...],
    proxy: BilateralBladeProxy | BladeClearanceEnvelope,
    config: ViewFilterConfig,
    reachability_checker: ReachabilityChecker | None = None,
    *,
    projection_poses: Mapping[str, PoseSE3] | None = None,
    deduplicate: bool = True,
    workspace_mode: Literal["required", "advisory", "disabled"] = "required",
) -> FilteredViewPlan:
    """Apply endpoint-only checks while retaining explicit uncertainty state.

    ``projection_poses`` supplies the virtual rectified-camera poses used for
    look-at, incidence, and standoff geometry.  Raw ``base_T_left_ir`` remains the
    sole pose passed to robot IK and physical workspace/clearance checks.
    """

    if workspace_mode not in {"required", "advisory", "disabled"}:
        raise ValueError("workspace_mode must be 'required', 'advisory', or 'disabled'")
    candidate_ids = tuple(candidate.view_id for candidate in candidates)
    if projection_poses is not None:
        if set(projection_poses) != set(candidate_ids):
            raise ValueError(
                "Projection-pose identities must exactly match candidate views"
            )
        for pose in projection_poses.values():
            if (pose.parent_frame, pose.child_frame) != (
                "base",
                "left_rectified",
            ):
                raise ValueError(
                    "Projection geometry requires base_T_left_rectified poses"
                )

    evaluated: list[EvaluatedCandidate] = []
    duplicates: list[str] = []
    for candidate in candidates:
        require_planning_time(f"before candidate filter {candidate.view_id}")
        projection_pose = (
            projection_poses[candidate.view_id]
            if projection_poses is not None
            else None
        )
        metrics = _metrics(candidate, proxy, projection_pose)
        reasons: list[str] = []
        rejected = False
        if metrics.look_at_cosine < config.minimum_look_at_cosine:
            rejected = True
            reasons.append("camera optical axis does not reliably intersect its patch target")
        if metrics.incidence_cosine < config.minimum_incidence_cosine:
            rejected = True
            reasons.append("surface incidence angle is too oblique")
        if metrics.standoff_error_m > config.maximum_standoff_error_m:
            rejected = True
            reasons.append("camera-to-target distance does not match planned standoff")
        if metrics.proxy_clearance_m < config.camera_clearance_radius_m:
            rejected = True
            reasons.append("camera clearance sphere intersects the blade proxy")

        position = candidate.base_t_left_ir.translation_m
        workspace_verified = workspace_mode != "required" or config.workspace is not None
        if workspace_mode != "disabled":
            if config.workspace is None:
                suffix = " (advisory)" if workspace_mode == "advisory" else ""
                reasons.append(f"workspace bounds are not configured{suffix}")
            elif not _point_inside_workspace(
                position,
                config.workspace,
                config.camera_clearance_radius_m,
            ):
                if workspace_mode == "required":
                    rejected = True
                    reasons.append(f"camera leaves workspace {config.workspace.name}")
                else:
                    reasons.append(
                        f"camera leaves advisory workspace {config.workspace.name}"
                    )
        for volume in config.forbidden_volumes:
            if _sphere_intersects_box(position, volume, config.camera_clearance_radius_m):
                rejected = True
                reasons.append(f"camera intersects forbidden volume {volume.name}")

        reachability_verified = False
        joint_solution = None
        if reachability_checker is None:
            reasons.append("robot endpoint reachability is not checked")
        else:
            try:
                with performance_span("planning.reachability_check"):
                    result = reachability_checker.check(candidate.base_t_left_ir)
                require_planning_time(
                    f"after candidate reachability check {candidate.view_id}"
                )
            except PlanningDeadlineExceeded:
                raise
            except Exception as exc:
                reasons.append(f"robot endpoint reachability check failed: {exc}")
            else:
                if result.state is ReachabilityState.UNREACHABLE:
                    rejected = True
                    reasons.append(result.message or "robot endpoint is unreachable")
                elif result.state is ReachabilityState.UNKNOWN:
                    reasons.append(result.message or "robot endpoint reachability is unknown")
                else:
                    reachability_verified = True
                    joint_solution = result.joint_positions_rad

        status = (
            CandidateStatus.REJECTED
            if rejected
            else (
                CandidateStatus.ENDPOINT_FEASIBLE
                if workspace_verified and reachability_verified
                else CandidateStatus.GEOMETRY_ONLY
            )
        )
        item = EvaluatedCandidate(
            candidate,
            status,
            metrics,
            tuple(reasons),
            joint_solution,
        )
        if deduplicate and item.status is not CandidateStatus.REJECTED:
            duplicate_index = next(
                (
                    index
                    for index, retained in enumerate(evaluated)
                    if retained.status is not CandidateStatus.REJECTED
                    and _is_duplicate(candidate, retained, config)
                ),
                None,
            )
            if duplicate_index is not None:
                retained = evaluated[duplicate_index]
                if item.metrics.geometric_score > retained.metrics.geometric_score:
                    duplicates.append(retained.candidate.view_id)
                    evaluated[duplicate_index] = item
                else:
                    duplicates.append(candidate.view_id)
                continue
        evaluated.append(item)

    return FilteredViewPlan(tuple(evaluated), tuple(duplicates))
