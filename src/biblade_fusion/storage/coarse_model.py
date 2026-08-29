"""Immutable storage for the paper-derived coarse-model reconstruction result."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.workflows.coarse_model import CoarseModelResult

COARSE_MODEL_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class StoredCoarseModelSummary:
    root: Path
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def write_coarse_model(
    output_dir: str | Path,
    result: CoarseModelResult,
    settings: AppSettings,
    *,
    source_views: tuple[str | Path, ...],
) -> Path:
    """Persist fused geometry, partitions, TSDF, mesh, plan, and quality atomically."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Coarse-model output already exists: {output}")
    if not source_views:
        raise ValueError("Coarse-model artifact requires source reconstructed views")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    surface = result.surface
    offsets = [0]
    for patch in surface.patches:
        offsets.append(offsets[-1] + len(patch.points_m))
    fin_offsets = [0]
    for component in surface.fin_components:
        fin_offsets.append(fin_offsets[-1] + len(component.points_m))
    boundary_contour_offsets = [0]
    boundary_control_offsets = [0]
    boundary_knot_offsets = [0]
    boundary_curves = []
    for model in surface.boundary_models:
        boundary_contour_offsets.append(boundary_contour_offsets[-1] + len(model.ordered_contour_m))
        for curve in model.curves:
            boundary_curves.append(curve)
            boundary_control_offsets.append(
                boundary_control_offsets[-1] + len(curve.control_points_m)
            )
            boundary_knot_offsets.append(boundary_knot_offsets[-1] + len(curve.knots))
    arrays: dict[str, np.ndarray] = {
        "fused_points_m": result.fused_cloud.points_m,
        "fused_normals": result.fused_cloud.normals,
        "fused_side_labels": result.fused_cloud.side_labels,
        "patch_points_m": np.vstack([patch.points_m for patch in surface.patches]),
        "patch_normals": np.vstack([patch.normals for patch in surface.patches]),
        "patch_section_coordinates": np.vstack(
            [patch.section_coordinates for patch in surface.patches]
        ),
        "patch_offsets": np.asarray(offsets, dtype=np.int64),
        "fin_component_points_m": (
            np.vstack([component.points_m for component in surface.fin_components])
            if surface.fin_components
            else np.empty((0, 3), dtype=np.float64)
        ),
        "fin_component_normals": (
            np.vstack([component.normals for component in surface.fin_components])
            if surface.fin_components
            else np.empty((0, 3), dtype=np.float64)
        ),
        "fin_component_local_coordinates": (
            np.vstack([component.local_coordinates for component in surface.fin_components])
            if surface.fin_components
            else np.empty((0, 2), dtype=np.float64)
        ),
        "fin_component_height_residual_m": (
            np.concatenate([component.height_residual_m for component in surface.fin_components])
            if surface.fin_components
            else np.empty(0, dtype=np.float64)
        ),
        "fin_component_root_masks": (
            np.concatenate([component.root_mask for component in surface.fin_components])
            if surface.fin_components
            else np.empty(0, dtype=np.bool_)
        ),
        "fin_component_free_edge_masks": (
            np.concatenate([component.free_edge_mask for component in surface.fin_components])
            if surface.fin_components
            else np.empty(0, dtype=np.bool_)
        ),
        "fin_component_offsets": np.asarray(fin_offsets, dtype=np.int64),
        "boundary_corners_m": (
            np.stack([model.corners_m for model in surface.boundary_models])
            if surface.boundary_models
            else np.empty((0, 4, 3), dtype=np.float64)
        ),
        "boundary_contour_points_m": (
            np.vstack([model.ordered_contour_m for model in surface.boundary_models])
            if surface.boundary_models
            else np.empty((0, 3), dtype=np.float64)
        ),
        "boundary_contour_offsets": np.asarray(boundary_contour_offsets, dtype=np.int64),
        "boundary_control_points_m": (
            np.vstack([curve.control_points_m for curve in boundary_curves])
            if boundary_curves
            else np.empty((0, 3), dtype=np.float64)
        ),
        "boundary_control_offsets": np.asarray(boundary_control_offsets, dtype=np.int64),
        "boundary_knots": (
            np.concatenate([curve.knots for curve in boundary_curves])
            if boundary_curves
            else np.empty(0, dtype=np.float64)
        ),
        "boundary_knot_offsets": np.asarray(boundary_knot_offsets, dtype=np.int64),
        "candidate_base_T_left_ir": np.stack(
            [candidate.base_t_left_ir.matrix for candidate in result.view_plan.candidates]
        ),
        "candidate_base_T_left_rectified": np.stack(
            [pose.matrix for pose in result.view_plan.candidate_base_t_left_rectified]
        ),
        "front_tsdf_indices": result.tsdf.front.voxel_indices,
        "front_tsdf_values": result.tsdf.front.tsdf,
        "front_tsdf_weights": result.tsdf.front.weights,
        "back_tsdf_indices": result.tsdf.back.voxel_indices,
        "back_tsdf_values": result.tsdf.back.tsdf,
        "back_tsdf_weights": result.tsdf.back.weights,
        "mesh_vertices_m": result.tsdf.mesh.vertices_m,
        "mesh_triangles": result.tsdf.mesh.triangles,
        "mesh_triangle_sides": result.tsdf.mesh.triangle_sides,
        "coverage_minimum_distances_m": np.concatenate(
            [item.minimum_distances_m for item in result.coverage.evidence]
        ),
        "coverage_best_normal_cosines": np.concatenate(
            [item.best_normal_cosines for item in result.coverage.evidence]
        ),
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        source_records = []
        for source in source_views:
            root = Path(source).resolve()
            metadata_path = root / "metadata.json"
            if not metadata_path.is_file():
                raise ValueError(f"Reconstructed-view metadata does not exist: {metadata_path}")
            source_records.append({"path": str(root), "metadata_sha256": _sha256(metadata_path)})
        payload: dict[str, Any] = {
            "schema_version": COARSE_MODEL_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "source_views": source_records,
            "files": {name: _record(temporary / f"{name}.npy") for name in arrays},
            "fusion": {
                "center_m": result.fused_cloud.center_m.tolist(),
                "axes": result.fused_cloud.axes.tolist(),
                "median_thickness_m": result.fused_cloud.median_thickness_m,
                "configuration": settings.multi_view_fusion.model_dump(mode="json"),
                "refinements": [
                    {
                        "view_id": item.view_id,
                        "side": item.side,
                        "correction_matrix": item.correction_matrix.tolist(),
                        "correspondence_count": item.correspondence_count,
                        "rmse_before_m": _finite_or_none(item.rmse_before_m),
                        "rmse_after_m": _finite_or_none(item.rmse_after_m),
                        "accepted": item.accepted,
                        "reason": item.reason,
                    }
                    for item in result.fused_cloud.refinements
                ],
            },
            "surface": {
                "section_arc_lengths_m": list(surface.section_arc_lengths_m),
                "angle_boundary_counts": list(surface.angle_boundary_counts),
                "base_grid_counts": list(surface.base_grid_counts),
                "base_footprint_m": list(surface.base_footprint_m),
                "footprint_source": surface.footprint_source,
                "parameterization_methods": list(surface.parameterization_methods),
                "boundary_fallback_reasons": list(surface.boundary_fallback_reasons),
                "configuration": settings.surface_partition.model_dump(mode="json"),
                "fin_components": [
                    {
                        "component_id": component.component_id,
                        "side": component.side.value,
                        "point_range": [fin_offsets[index], fin_offsets[index + 1]],
                        "root_point_count": int(np.count_nonzero(component.root_mask)),
                        "free_edge_point_count": int(np.count_nonzero(component.free_edge_mask)),
                        "obb_center_m": component.obb_center_m.tolist(),
                        "obb_axes": component.obb_axes.tolist(),
                        "obb_extents_m": component.obb_extents_m.tolist(),
                        "normal_axis": component.normal_axis.tolist(),
                        "main_height_rmse_m": component.main_height_rmse_m,
                        "face_separation_m": component.face_separation_m,
                        "two_faces_observed": component.two_faces_observed,
                    }
                    for index, component in enumerate(surface.fin_components)
                ],
                "boundary_models": [
                    {
                        "side": model.side.value,
                        "source_boundary_count": model.source_boundary_count,
                        "fit_rmse_m": model.fit_rmse_m,
                        "contour_range": [
                            boundary_contour_offsets[model_index],
                            boundary_contour_offsets[model_index + 1],
                        ],
                        "curves": [
                            {
                                "name": curve.name.value,
                                "degree": curve.degree,
                                "source_point_count": curve.source_point_count,
                                "fit_rmse_m": curve.fit_rmse_m,
                                "inlier_fraction": curve.inlier_fraction,
                                "arc_length_m": curve.arc_length_m,
                                "control_range": [
                                    boundary_control_offsets[model_index * 4 + curve_index],
                                    boundary_control_offsets[model_index * 4 + curve_index + 1],
                                ],
                                "knot_range": [
                                    boundary_knot_offsets[model_index * 4 + curve_index],
                                    boundary_knot_offsets[model_index * 4 + curve_index + 1],
                                ],
                            }
                            for curve_index, curve in enumerate(model.curves)
                        ],
                    }
                    for model_index, model in enumerate(surface.boundary_models)
                ],
                "patches": [
                    {
                        "patch_id": patch.patch_id,
                        "side": patch.side.value,
                        "region": patch.region.value,
                        "row": patch.row,
                        "column": patch.column,
                        "adaptive_depth": patch.adaptive_depth,
                        "point_range": [offsets[index], offsets[index + 1]],
                        "obb_center_m": patch.obb_center_m.tolist(),
                        "obb_axes": patch.obb_axes.tolist(),
                        "obb_extents_m": patch.obb_extents_m.tolist(),
                        "main_normal": patch.main_normal.tolist(),
                        "curvature_deg": patch.curvature_deg,
                        "boundary_fraction": patch.boundary_fraction,
                    }
                    for index, patch in enumerate(surface.patches)
                ],
            },
            "view_plan": {
                "baseline_standoff_distance_m": settings.view_planning.standoff_distance_m,
                "baseline_footprint_m": list(result.view_plan.footprint_m),
                "left_rectified_T_left_ir": (
                    result.view_plan.left_rectified_t_left_ir.matrix.tolist()
                ),
                "configuration": settings.view_planning.model_dump(mode="json"),
                "candidate_ids": [candidate.view_id for candidate in result.view_plan.candidates],
                "candidates": [
                    {
                        "view_id": candidate.view_id,
                        "patch_id": candidate.patch.patch_id,
                        "standoff_distance_m": candidate.standoff_distance_m,
                        "footprint_m": list(candidate.footprint_m),
                        "projection_fraction": candidate.projection_fraction,
                        "visibility_fraction": candidate.visibility_fraction,
                        "distance_policy": candidate.distance_policy,
                        "base_T_left_rectified": base_t_left_rectified.matrix.tolist(),
                        "base_T_left_ir": candidate.base_t_left_ir.matrix.tolist(),
                    }
                    for candidate, base_t_left_rectified in zip(
                        result.view_plan.candidates,
                        result.view_plan.candidate_base_t_left_rectified,
                        strict=True,
                    )
                ],
            },
            "tsdf": {
                "backend": result.tsdf.backend,
                "configuration": settings.tsdf.model_dump(mode="json"),
                "protected_truncation_distance_m": (result.tsdf.protected_truncation_distance_m),
                "feature_thicknesses_m": list(result.tsdf.feature_thicknesses_m),
                "origin_m": result.tsdf.front.origin_m.tolist(),
                "mesh_boundary_edge_count": result.tsdf.mesh.boundary_edge_count,
            },
            "quality": {
                "configuration": settings.surface_quality.model_dump(mode="json"),
                "completion_fraction": result.quality.completion_fraction,
                "mesh_triangle_count": result.quality.mesh_triangle_count,
                "mesh_boundary_edge_count": result.quality.mesh_boundary_edge_count,
                "mesh_boundary_loop_count": result.quality.mesh_boundary_loop_count,
                "mesh_watertight": result.quality.mesh_watertight,
                "edge_completion": {
                    key.value: value for key, value in result.quality.edge_completion.items()
                },
                "patches": [
                    {
                        "patch_id": item.patch_id,
                        "side": item.side.value,
                        "region": item.region.value,
                        "reference_point_count": item.reference_point_count,
                        "observed_point_count": item.observed_point_count,
                        "coverage_fraction": item.coverage_fraction,
                        "rmse_m": _finite_or_none(item.rmse_m),
                        "normal_consistency": item.normal_consistency,
                        "curvature_deg": item.curvature_deg,
                        "complete": item.complete,
                        "reasons": list(item.reasons),
                    }
                    for item in result.quality.patches
                ],
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_coarse_model_summary(path: str | Path) -> StoredCoarseModelSummary:
    """Validate every array manifest and source binding without loading all arrays."""

    root = Path(path)
    try:
        payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        schema = int(payload["schema_version"])
        if schema not in {1, 2, 3, 4, COARSE_MODEL_SCHEMA_VERSION}:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("coarse-model artifact must explicitly forbid motion")
        for source in payload["source_views"]:
            metadata = Path(str(source["path"])) / "metadata.json"
            if _sha256(metadata) != str(source["metadata_sha256"]):
                raise ValueError(f"source reconstructed-view checksum mismatch: {metadata}")
        for record in payload["files"].values():
            relative = Path(str(record["path"]))
            file_path = (root.resolve() / relative).resolve()
            if relative.is_absolute() or not file_path.is_relative_to(root.resolve()):
                raise ValueError(f"coarse-model array path escapes output: {relative}")
            if _sha256(file_path) != str(record["sha256"]):
                raise ValueError(f"coarse-model array checksum mismatch: {relative}")
            array = np.load(file_path, mmap_mode="r", allow_pickle=False)
            try:
                if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
                    raise ValueError(f"coarse-model array manifest mismatch: {relative}")
            finally:
                del array
        if schema == COARSE_MODEL_SCHEMA_VERSION:
            view_plan = payload["view_plan"]
            candidates = view_plan["candidates"]
            candidate_ids = view_plan["candidate_ids"]
            if len(candidate_ids) != len(candidates) or len(set(candidate_ids)) != len(
                candidate_ids
            ):
                raise ValueError("coarse-model candidate identities are inconsistent")
            calibration = PoseSE3(
                "left_rectified",
                "left_ir",
                view_plan["left_rectified_T_left_ir"],
            )
            rectified_transforms = np.asarray(
                np.load(
                    root / str(payload["files"]["candidate_base_T_left_rectified"]["path"]),
                    allow_pickle=False,
                ),
                dtype=np.float64,
            )
            raw_transforms = np.asarray(
                np.load(
                    root / str(payload["files"]["candidate_base_T_left_ir"]["path"]),
                    allow_pickle=False,
                ),
                dtype=np.float64,
            )
            expected_shape = (len(candidates), 4, 4)
            if (
                rectified_transforms.shape != expected_shape
                or raw_transforms.shape != expected_shape
            ):
                raise ValueError("coarse-model candidate transforms have invalid shape")
            for index, candidate in enumerate(candidates):
                if str(candidate["view_id"]) != str(candidate_ids[index]):
                    raise ValueError("coarse-model candidate order is inconsistent")
                metadata_rectified = np.asarray(
                    candidate["base_T_left_rectified"], dtype=np.float64
                )
                metadata_raw = np.asarray(candidate["base_T_left_ir"], dtype=np.float64)
                if not np.allclose(
                    metadata_rectified,
                    rectified_transforms[index],
                    rtol=0.0,
                    atol=1e-10,
                ) or not np.allclose(
                    metadata_raw,
                    raw_transforms[index],
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        "coarse-model metadata and checksummed candidate transforms disagree"
                    )
                base_t_left_rectified = PoseSE3(
                    "base", "left_rectified", rectified_transforms[index]
                )
                base_t_left_ir = PoseSE3("base", "left_ir", raw_transforms[index])
                expected_raw = base_t_left_rectified.compose(calibration)
                if not np.allclose(
                    expected_raw.matrix,
                    base_t_left_ir.matrix,
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError(
                        "coarse-model raw and rectified candidate poses violate calibration"
                    )
        return StoredCoarseModelSummary(root.resolve(), payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coarse-model artifact {root}: {exc}") from exc
