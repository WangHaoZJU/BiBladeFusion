"""Fail-closed geometric inspection of persisted fine-scan view plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import acos, degrees

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.settings import ViewFilterConfig
from biblade_fusion.storage.coarse_model import StoredCoarseModelSummary


class FinePlanInspectionError(ValueError):
    """A coarse-model artifact cannot support fine-plan inspection."""


REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "surface": (90, 155, 230),
    "leading_edge": (245, 166, 35),
    "trailing_edge": (226, 84, 84),
    "root": (140, 105, 205),
    "tip": (65, 185, 120),
    "fin_face": (30, 190, 195),
    "fin_root": (245, 95, 175),
    "fin_free_edge": (235, 215, 75),
}


@dataclass(frozen=True, slots=True)
class FineViewInspection:
    view_id: str
    patch_id: str
    side: str
    region: str
    standoff_distance_m: float
    footprint_m: tuple[float, float]
    projection_fraction: float
    visibility_fraction: float
    distance_policy: str
    base_t_left_ir: NDArray[np.float64]
    target_m: NDArray[np.float64]
    outward_normal: NDArray[np.float64]
    accepted: bool
    reasons: tuple[str, ...]

    @property
    def camera_position_m(self) -> NDArray[np.float64]:
        return self.base_t_left_ir[:3, 3]


@dataclass(frozen=True, slots=True)
class FinePlanInspection:
    source_root: str
    source_schema_version: int
    motion_authorized: bool
    geometry_passed: bool
    robot_feasibility: str
    global_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    views: tuple[FineViewInspection, ...]
    region_counts: dict[str, dict[str, int]]
    inspection_configuration: dict[str, object]
    scene_points_m: NDArray[np.float64]
    scene_colors_rgb: NDArray[np.uint8]


def _array(summary: StoredCoarseModelSummary, name: str) -> np.ndarray:
    try:
        relative = summary.metadata["files"][name]["path"]
    except KeyError as exc:
        raise FinePlanInspectionError(f"Coarse model is missing array {name}") from exc
    return np.load(summary.root / str(relative), allow_pickle=False)


def _rotation_distance_deg(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _unit(value: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)


def inspect_fine_plan(
    summary: StoredCoarseModelSummary,
    filter_config: ViewFilterConfig,
) -> FinePlanInspection:
    """Validate persisted per-patch camera geometry without claiming robot feasibility."""

    metadata = summary.metadata
    schema = int(metadata["schema_version"])
    if schema < 4 or "candidates" not in metadata.get("view_plan", {}):
        raise FinePlanInspectionError(
            "Fine-plan inspection requires a schema-4 coarse model; regenerate the artifact"
        )
    patch_payloads = metadata["surface"]["patches"]
    candidate_payloads = metadata["view_plan"]["candidates"]
    candidate_ids = metadata["view_plan"]["candidate_ids"]
    if len(patch_payloads) != len(candidate_payloads) or len(candidate_ids) != len(
        candidate_payloads
    ):
        raise FinePlanInspectionError("Patch and fine-view counts do not match")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise FinePlanInspectionError("Fine-view identifiers are not unique")

    patch_by_id = {str(item["patch_id"]): item for item in patch_payloads}
    if len(patch_by_id) != len(patch_payloads):
        raise FinePlanInspectionError("Patch identifiers are not unique")
    stored_raw_transforms = np.asarray(
        _array(summary, "candidate_base_T_left_ir"), dtype=np.float64
    )
    expected_transform_shape = (len(candidate_payloads), 4, 4)
    if stored_raw_transforms.shape != expected_transform_shape:
        raise FinePlanInspectionError("Persisted raw candidate transforms have invalid shape")
    stored_geometry_transforms = (
        np.asarray(_array(summary, "candidate_base_T_left_rectified"), dtype=np.float64)
        if schema >= 5
        else stored_raw_transforms
    )
    if stored_geometry_transforms.shape != expected_transform_shape:
        raise FinePlanInspectionError("Persisted rectified candidate transforms have invalid shape")

    planning = metadata["view_plan"]["configuration"]
    baseline = float(metadata["view_plan"]["baseline_standoff_distance_m"])
    adaptive = bool(planning["adaptive_standoff_enabled"])
    lower = planning.get("minimum_standoff_distance_m")
    upper = planning.get("maximum_standoff_distance_m")
    projection_gate = float(planning["minimum_patch_projection_fraction"])
    visibility_gate = float(planning["minimum_patch_visibility_fraction"])
    views: list[FineViewInspection] = []
    for index, candidate in enumerate(candidate_payloads):
        patch_id = str(candidate["patch_id"])
        if patch_id not in patch_by_id:
            raise FinePlanInspectionError(f"Fine view references unknown patch {patch_id}")
        patch = patch_by_id[patch_id]
        raw_matrix = np.asarray(candidate["base_T_left_ir"], dtype=np.float64)
        geometry_matrix = (
            np.asarray(candidate["base_T_left_rectified"], dtype=np.float64)
            if schema >= 5
            else raw_matrix
        )
        target = np.asarray(patch["obb_center_m"], dtype=np.float64)
        normal = np.asarray(patch["main_normal"], dtype=np.float64)
        distance = float(candidate["standoff_distance_m"])
        projection = float(candidate["projection_fraction"])
        visibility = float(candidate["visibility_fraction"])
        footprint = tuple(float(value) for value in candidate["footprint_m"])
        reasons: list[str] = []
        if raw_matrix.shape != (4, 4) or not np.isfinite(raw_matrix).all():
            reasons.append("raw camera transform is not a finite 4x4 matrix")
        else:
            raw_rotation = raw_matrix[:3, :3]
            if not np.allclose(
                raw_rotation.T @ raw_rotation, np.eye(3), atol=1e-6
            ) or not np.isclose(np.linalg.det(raw_rotation), 1.0, atol=1e-6):
                reasons.append("raw camera rotation is not right-handed and orthonormal")
            if not np.allclose(
                raw_matrix,
                stored_raw_transforms[index],
                rtol=0.0,
                atol=1e-10,
            ):
                reasons.append("metadata and checksummed raw camera transforms disagree")
        geometry_valid = geometry_matrix.shape == (4, 4) and np.isfinite(geometry_matrix).all()
        if not geometry_valid:
            reasons.append("rectified camera transform is not a finite 4x4 matrix")
        else:
            geometry_rotation = geometry_matrix[:3, :3]
            if not np.allclose(
                geometry_rotation.T @ geometry_rotation,
                np.eye(3),
                atol=1e-6,
            ) or not np.isclose(np.linalg.det(geometry_rotation), 1.0, atol=1e-6):
                reasons.append("rectified camera rotation is not right-handed and orthonormal")
            if not np.allclose(
                geometry_matrix,
                stored_geometry_transforms[index],
                rtol=0.0,
                atol=1e-10,
            ):
                reasons.append("metadata and checksummed rectified camera transforms disagree")
            view_vector = target - geometry_matrix[:3, 3]
            measured_distance = float(np.linalg.norm(view_vector))
            if abs(measured_distance - distance) > filter_config.maximum_standoff_error_m:
                reasons.append("camera-to-target distance does not match selected standoff")
            if (
                float(geometry_rotation[:, 2] @ _unit(view_vector))
                < filter_config.minimum_look_at_cosine
            ):
                reasons.append("camera optical axis does not look at the patch centre")
            if (
                float((-geometry_rotation[:, 2]) @ _unit(normal))
                < filter_config.minimum_incidence_cosine
            ):
                reasons.append("camera incidence does not match the planned outward normal")
        if projection + 1e-12 < projection_gate:
            reasons.append("patch projection fraction is below the configured gate")
        if visibility + 1e-12 < visibility_gate:
            reasons.append("patch visibility fraction is below the configured gate")
        if min(footprint) <= 0.0 or not np.isfinite(footprint).all():
            reasons.append("nominal footprint is invalid")
        if adaptive:
            if lower is None or upper is None or not float(lower) <= distance <= float(upper):
                reasons.append("selected standoff lies outside adaptive bounds")
        elif abs(distance - baseline) > 1e-12:
            reasons.append("fixed-distance plan does not use the baseline standoff")
        views.append(
            FineViewInspection(
                str(candidate["view_id"]),
                patch_id,
                str(patch["side"]),
                str(patch["region"]),
                distance,
                footprint,
                projection,
                visibility,
                str(candidate["distance_policy"]),
                raw_matrix,
                target,
                normal,
                not reasons,
                tuple(reasons),
            )
        )

    # A duplicate is a planning defect, so both members are marked rather than silently
    # discarding the later one.
    duplicate_indices: set[int] = set()
    for first in range(len(views)):
        for second in range(first + 1, len(views)):
            translation = float(
                np.linalg.norm(views[first].camera_position_m - views[second].camera_position_m)
            )
            rotation = _rotation_distance_deg(
                views[first].base_t_left_ir[:3, :3], views[second].base_t_left_ir[:3, :3]
            )
            if (
                translation <= filter_config.duplicate_translation_tolerance_m
                and rotation <= filter_config.duplicate_rotation_tolerance_deg
            ):
                duplicate_indices.update((first, second))
    for index in duplicate_indices:
        item = views[index]
        views[index] = replace(
            item,
            accepted=False,
            reasons=(*item.reasons, "candidate duplicates another camera pose"),
        )

    expected_regions = {"surface", "leading_edge", "trailing_edge", "root", "tip"}
    fin_mode = str(metadata["surface"]["configuration"].get("fin_mode", "disabled"))
    if fin_mode != "disabled":
        expected_regions.update(("fin_face", "fin_root", "fin_free_edge"))
    global_reasons: list[str] = []
    region_counts: dict[str, dict[str, int]] = {}
    for side in ("front", "back"):
        region_counts[side] = {}
        for region in sorted(expected_regions):
            matching = [item for item in views if item.side == side and item.region == region]
            accepted_count = sum(item.accepted for item in matching)
            region_counts[side][region] = accepted_count
            if not matching:
                global_reasons.append(f"{side}/{region} has no planned fine view")
            elif accepted_count == 0:
                global_reasons.append(f"{side}/{region} has no accepted fine view")

    patch_points = np.asarray(_array(summary, "patch_points_m"), dtype=np.float64)
    patch_offsets = np.asarray(_array(summary, "patch_offsets"), dtype=np.int64)
    if len(patch_offsets) != len(patch_payloads) + 1 or patch_offsets[-1] != len(patch_points):
        raise FinePlanInspectionError("Patch point offsets do not match the persisted points")
    colors = np.empty((len(patch_points), 3), dtype=np.uint8)
    for index, patch in enumerate(patch_payloads):
        color = REGION_COLORS.get(str(patch["region"]), (180, 180, 180))
        colors[patch_offsets[index] : patch_offsets[index + 1]] = color

    warnings = [
        "robot IK, robot/camera-body collision, and trajectory continuity are not evaluated",
        "inspection output is non-executable and cannot authorize motion",
    ]
    if filter_config.workspace is None:
        warnings.append("workspace bounds are not configured")
    geometry_passed = not global_reasons and all(item.accepted for item in views)
    return FinePlanInspection(
        str(summary.root),
        schema,
        False,
        geometry_passed,
        "unverified",
        tuple(global_reasons),
        tuple(warnings),
        tuple(views),
        region_counts,
        filter_config.model_dump(mode="json"),
        patch_points,
        colors,
    )
