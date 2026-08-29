"""Immutable generations of fine-scan evidence against a schema-5 coarse surface.

The coarse model is a geometric reference only.  Its acquisition views and its
coarse-workflow coverage ledger are deliberately not imported into this ledger.
Every non-initial generation is reproducible from exactly one previous generation
and one semantically validated reconstructed view.
"""

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
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    NextViewSelectionConfig,
    ReacquisitionPerturbationConfig,
    SurfacePartitionConfig,
    SurfaceQualityConfig,
    ViewPlanningConfig,
)
from biblade_fusion.perception.features import FinComponent
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    CurvedSurfacePatch,
    CurvedViewPlan,
    SurfaceRegion,
    generate_reacquisition_view,
)
from biblade_fusion.planning.surface_coverage import (
    SurfaceCoverageLedger,
    SurfacePatchEvidence,
    SurfaceQualityReport,
    create_surface_coverage_ledger,
    evaluate_surface_quality,
    update_surface_coverage,
)
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch
from biblade_fusion.storage.coarse_model import (
    StoredCoarseModelSummary,
    read_coarse_model_summary,
)
from biblade_fusion.storage.reconstructed_view import (
    RECONSTRUCTED_VIEW_SCHEMA_VERSION,
    SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION,
    StoredReconstructedBladeView,
    read_reconstructed_view,
)
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.workflows.coarse_model import registered_cloud_view

SURFACE_COVERAGE_SCHEMA_VERSION = 2
LEGACY_SURFACE_COVERAGE_SCHEMA_VERSION = 1
SURFACE_COVERAGE_COARSE_SCHEMA_VERSION = 5
REACQUISITION_VIEW_ID_SCHEMA = "fine_patch_reacquisition_v2"
PhysicalSourceIdentity = tuple[str, str, int, int]
_METADATA_NAME = "coverage.json"
_ARRAY_NAMES = (
    "minimum_distances_m",
    "best_normal_cosines",
    "patch_offsets",
)


@dataclass(frozen=True, slots=True)
class StoredSurfaceCoverageGeneration:
    """A verified fine-scan coverage generation and its fixed reference model."""

    root: Path
    generation_id: str
    metadata_sha256: str
    reference: StoredCoarseModelSummary
    surface: CurvedBladeSurface
    view_plan: CurvedViewPlan
    ledger: SurfaceCoverageLedger
    quality: SurfaceQualityReport
    quality_config: SurfaceQualityConfig
    required_patch_ids: tuple[str, ...]
    required_regions: tuple[SurfaceRegion, ...]
    previous_generation_path: Path | None
    current_reconstructed_view_path: Path | None
    metadata: dict[str, Any]
    current_reacquisition: FineReacquisitionProvenance | None = None
    physical_source_identities: tuple[PhysicalSourceIdentity, ...] = ()

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _ReferenceGeometry:
    summary: StoredCoarseModelSummary
    metadata_sha256: str
    surface: CurvedBladeSurface
    view_plan: CurvedViewPlan
    partition_config: SurfacePartitionConfig


@dataclass(frozen=True, slots=True)
class FineReacquisitionProvenance:
    """Typed authority for one bounded retry of a fixed nominal patch view."""

    view_id: str
    nominal_candidate_id: str
    patch_id: str
    attempt: int
    distance_offset_m: float
    tilt_deg: float
    azimuth_deg: float
    selection_policy_sha256: str
    reference_metadata_sha256: str

    def __post_init__(self) -> None:
        for name in ("view_id", "nominal_candidate_id", "patch_id"):
            raw = getattr(self, name)
            if type(raw) is not str:
                raise TypeError(f"Fine reacquisition {name} must be a string")
            value = raw.strip()
            if not value:
                raise ValueError(f"Fine reacquisition {name} must be non-empty")
            object.__setattr__(self, name, value)
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ValueError("Fine reacquisition attempt must be a positive integer")
        perturbations = (
            ("distance_offset_m", self.distance_offset_m),
            ("tilt_deg", self.tilt_deg),
            ("azimuth_deg", self.azimuth_deg),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for _, value in perturbations
        ):
            raise TypeError("Fine reacquisition perturbations must be numeric")
        values = np.asarray(tuple(value for _, value in perturbations), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Fine reacquisition perturbation must be finite")
        for (name, _), value in zip(perturbations, values, strict=True):
            object.__setattr__(self, name, float(value))
        for name in ("selection_policy_sha256", "reference_metadata_sha256"):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"Fine reacquisition {name} must be a string")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Fine reacquisition {name} must be a SHA-256 digest")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "bounded_reacquisition",
            "id_schema": REACQUISITION_VIEW_ID_SCHEMA,
            "view_id": self.view_id,
            "nominal_candidate_id": self.nominal_candidate_id,
            "patch_id": self.patch_id,
            "attempt": self.attempt,
            "distance_offset_m": self.distance_offset_m,
            "tilt_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "selection_policy_sha256": self.selection_policy_sha256,
            "reference_metadata_sha256": self.reference_metadata_sha256,
        }


def reacquisition_view_id(
    nominal_candidate_id: str,
    patch_id: str,
    attempt: int,
    selection_policy_sha256: str,
) -> str:
    """Derive the immutable capture ID that carries retry policy provenance."""

    if type(nominal_candidate_id) is not str or type(patch_id) is not str:
        raise TypeError("Reacquisition candidate and patch IDs must be strings")
    nominal = nominal_candidate_id.strip()
    patch = patch_id.strip()
    if (
        not nominal
        or not patch
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise ValueError("Reacquisition ID inputs are invalid")
    if type(selection_policy_sha256) is not str:
        raise TypeError("Reacquisition ID selection policy must be a string")
    policy = selection_policy_sha256
    if len(policy) != 64 or any(
        character not in "0123456789abcdef" for character in policy
    ):
        raise ValueError("Reacquisition ID selection policy must be a SHA-256")
    source = json.dumps(
        {
            "schema": REACQUISITION_VIEW_ID_SCHEMA,
            "nominal_candidate_id": nominal,
            "patch_id": patch,
            "attempt": int(attempt),
            "selection_policy_sha256": policy,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"fine_reacq_a{attempt:02d}_{hashlib.sha256(source).hexdigest()}"


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


def _load_array(
    root: Path,
    record: dict[str, Any],
    *,
    label: str,
    dtype: np.dtype[Any] | type[Any],
) -> np.ndarray:
    relative = Path(str(record["path"]))
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"{label} array path escapes its artifact: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"{label} array checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"{label} array manifest mismatch: {relative}")
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{label} array has unexpected dtype: {array.dtype}")
    return array


def _source_record(root: Path, metadata_name: str) -> dict[str, str]:
    resolved = root.resolve()
    metadata = resolved / metadata_name
    if not metadata.is_file():
        raise ValueError(f"Source metadata does not exist: {metadata}")
    return {"path": str(resolved), "metadata_sha256": _sha256(metadata)}


def _verify_source_record(record: dict[str, Any], metadata_name: str, *, label: str) -> Path:
    raw = Path(str(record["path"]))
    resolved = raw.resolve()
    if not raw.is_absolute() or raw != resolved:
        raise ValueError(f"{label} source path must be absolute and canonical: {raw}")
    metadata = resolved / metadata_name
    if _sha256(metadata) != str(record["metadata_sha256"]):
        raise ValueError(f"{label} source metadata checksum mismatch: {metadata}")
    return resolved


def _offsets(value: np.ndarray, *, count: int, total: int, label: str) -> NDArray[np.int64]:
    if value.shape != (count + 1,):
        raise ValueError(f"{label} offsets have invalid shape")
    result = np.asarray(value, dtype=np.int64)
    if result[0] != 0 or result[-1] != total or np.any(np.diff(result) < 0):
        raise ValueError(f"{label} offsets are not a complete monotonic partition")
    return result


def _range(data: dict[str, Any], expected: tuple[int, int], *, label: str) -> None:
    values = data["point_range"]
    if (
        not isinstance(values, list)
        or len(values) != 2
        or tuple(int(value) for value in values) != expected
    ):
        raise ValueError(f"{label} point range does not match its array offsets")


def _surface_from_reference(
    summary: StoredCoarseModelSummary,
) -> tuple[CurvedBladeSurface, SurfacePartitionConfig]:
    root = summary.root
    payload = summary.metadata
    files = payload["files"]
    surface_data = payload["surface"]
    patch_data = surface_data["patches"]
    points = _load_array(
        root, files["patch_points_m"], label="coarse patch points", dtype=np.float64
    )
    normals = _load_array(
        root, files["patch_normals"], label="coarse patch normals", dtype=np.float64
    )
    coordinates = _load_array(
        root,
        files["patch_section_coordinates"],
        label="coarse patch coordinates",
        dtype=np.float64,
    )
    offsets = _offsets(
        _load_array(root, files["patch_offsets"], label="coarse patch offsets", dtype=np.int64),
        count=len(patch_data),
        total=len(points),
        label="coarse patch",
    )
    if normals.shape != points.shape or coordinates.shape != (len(points), 2):
        raise ValueError("Coarse patch arrays have inconsistent shapes")
    patches: list[CurvedSurfacePatch] = []
    for index, data in enumerate(patch_data):
        first, last = int(offsets[index]), int(offsets[index + 1])
        _range(data, (first, last), label=f"coarse patch {index}")
        patches.append(
            CurvedSurfacePatch(
                str(data["patch_id"]),
                BladeSide(str(data["side"])),
                SurfaceRegion(str(data["region"])),
                int(data["row"]),
                int(data["column"]),
                int(data["adaptive_depth"]),
                points[first:last],
                normals[first:last],
                coordinates[first:last],
                data["obb_center_m"],
                data["obb_axes"],
                data["obb_extents_m"],
                data["main_normal"],
                float(data["curvature_deg"]),
                float(data["boundary_fraction"]),
            )
        )

    component_data = surface_data["fin_components"]
    component_points = _load_array(
        root,
        files["fin_component_points_m"],
        label="coarse fin points",
        dtype=np.float64,
    )
    component_normals = _load_array(
        root,
        files["fin_component_normals"],
        label="coarse fin normals",
        dtype=np.float64,
    )
    component_coordinates = _load_array(
        root,
        files["fin_component_local_coordinates"],
        label="coarse fin coordinates",
        dtype=np.float64,
    )
    component_residual = _load_array(
        root,
        files["fin_component_height_residual_m"],
        label="coarse fin residual",
        dtype=np.float64,
    )
    root_masks = _load_array(
        root,
        files["fin_component_root_masks"],
        label="coarse fin root masks",
        dtype=np.bool_,
    )
    free_masks = _load_array(
        root,
        files["fin_component_free_edge_masks"],
        label="coarse fin free-edge masks",
        dtype=np.bool_,
    )
    component_offsets = _offsets(
        _load_array(
            root,
            files["fin_component_offsets"],
            label="coarse fin offsets",
            dtype=np.int64,
        ),
        count=len(component_data),
        total=len(component_points),
        label="coarse fin",
    )
    total = len(component_points)
    if (
        component_normals.shape != component_points.shape
        or component_coordinates.shape != (total, 2)
        or component_residual.shape != (total,)
        or root_masks.shape != (total,)
        or free_masks.shape != (total,)
    ):
        raise ValueError("Coarse fin arrays have inconsistent shapes")
    components: list[FinComponent] = []
    for index, data in enumerate(component_data):
        first, last = int(component_offsets[index]), int(component_offsets[index + 1])
        _range(data, (first, last), label=f"coarse fin component {index}")
        two_faces = data["two_faces_observed"]
        if not isinstance(two_faces, bool):
            raise ValueError("Coarse fin two_faces_observed must be boolean")
        component = FinComponent(
            str(data["component_id"]),
            BladeSide(str(data["side"])),
            component_points[first:last],
            component_normals[first:last],
            component_coordinates[first:last],
            component_residual[first:last],
            root_masks[first:last],
            free_masks[first:last],
            data["obb_center_m"],
            data["obb_axes"],
            data["obb_extents_m"],
            data["normal_axis"],
            float(data["main_height_rmse_m"]),
            float(data["face_separation_m"]),
            two_faces,
        )
        if int(data["root_point_count"]) != int(np.count_nonzero(component.root_mask)):
            raise ValueError("Coarse fin root-point summary is inconsistent")
        if int(data["free_edge_point_count"]) != int(np.count_nonzero(component.free_edge_mask)):
            raise ValueError("Coarse fin free-edge summary is inconsistent")
        components.append(component)

    fusion = payload["fusion"]
    axes = np.asarray(fusion["axes"], dtype=np.float64)
    if (
        axes.shape != (3, 3)
        or not np.isfinite(axes).all()
        or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(axes), 1.0, atol=1e-6)
    ):
        raise ValueError("Coarse surface axes are not right-handed and orthonormal")
    section_lengths = tuple(float(value) for value in surface_data["section_arc_lengths_m"])
    angle_counts = tuple(int(value) for value in surface_data["angle_boundary_counts"])
    grid_counts = tuple(int(value) for value in surface_data["base_grid_counts"])
    footprint = tuple(float(value) for value in surface_data["base_footprint_m"])
    methods = tuple(str(value) for value in surface_data["parameterization_methods"])
    fallbacks = tuple(str(value) for value in surface_data["boundary_fallback_reasons"])
    if not all(
        len(value) == 2
        for value in (section_lengths, angle_counts, grid_counts, footprint, methods, fallbacks)
    ):
        raise ValueError("Coarse surface front/back metadata must contain exactly two values")
    surface = CurvedBladeSurface(
        "base",
        tuple(patches),
        axes,
        fusion["center_m"],
        section_lengths,
        angle_counts,
        grid_counts,
        footprint,
        str(surface_data["footprint_source"]),
        tuple(components),
        (),
        methods,
        fallbacks,
    )
    config = SurfacePartitionConfig.model_validate(surface_data["configuration"])
    _validate_required_fin(surface, config)
    return surface, config


def read_coarse_surface_reference(path: str | Path) -> CurvedBladeSurface:
    """Fully restore the fixed schema-5 surface used by online fine science.

    This public reader deliberately returns geometry only after the coarse-model
    reader has checked every bound array and source metadata checksum.  It lets a
    downstream scientific asset replay its projection without manufacturing an
    empty coverage generation merely to recover the reference surface.
    """

    summary = read_coarse_model_summary(path)
    if int(summary.metadata["schema_version"]) != SURFACE_COVERAGE_COARSE_SCHEMA_VERSION:
        raise ValueError("Fine-science reference requires an exact schema-5 coarse model")
    surface, _ = _surface_from_reference(summary)
    return surface


def _validate_required_fin(surface: CurvedBladeSurface, config: SurfacePartitionConfig) -> None:
    if config.fin_mode != "required_single_per_side":
        return
    by_side = {component.side: component for component in surface.fin_components}
    if set(by_side) != {BladeSide.FRONT, BladeSide.BACK}:
        raise ValueError("Required fin mode needs one coarse fin component on each side")
    required = {
        SurfaceRegion.FIN_FACE,
        SurfaceRegion.FIN_ROOT,
        SurfaceRegion.FIN_FREE_EDGE,
    }
    for side in (BladeSide.FRONT, BladeSide.BACK):
        if not by_side[side].two_faces_observed:
            raise ValueError(f"Required {side.value} fin does not have two observed faces")
        present = {patch.region for patch in surface.for_side(side)}
        missing = required - present
        if missing:
            names = ", ".join(sorted(region.value for region in missing))
            raise ValueError(f"Required {side.value} fin patch regions are missing: {names}")


def _view_plan_from_reference(
    summary: StoredCoarseModelSummary, surface: CurvedBladeSurface
) -> CurvedViewPlan:
    root = summary.root
    payload = summary.metadata
    data = payload["view_plan"]
    candidates_data = data["candidates"]
    raw_matrices = _load_array(
        root,
        payload["files"]["candidate_base_T_left_ir"],
        label="coarse raw candidate poses",
        dtype=np.float64,
    )
    rectified_matrices = _load_array(
        root,
        payload["files"]["candidate_base_T_left_rectified"],
        label="coarse rectified candidate poses",
        dtype=np.float64,
    )
    expected_shape = (len(surface.patches), 4, 4)
    if raw_matrices.shape != expected_shape or rectified_matrices.shape != expected_shape:
        raise ValueError("Coarse candidate pose arrays do not match surface patches")
    candidate_ids = tuple(str(value) for value in data["candidate_ids"])
    if len(candidates_data) != len(surface.patches) or candidate_ids != tuple(
        str(item["view_id"]) for item in candidates_data
    ):
        raise ValueError("Coarse candidate identity list is inconsistent")
    candidates: list[CandidateView] = []
    rectified_poses: list[PoseSE3] = []
    calibration = PoseSE3(
        "left_rectified",
        "left_ir",
        data["left_rectified_T_left_ir"],
    )
    for index, (patch, item) in enumerate(zip(surface.patches, candidates_data, strict=True)):
        if str(item["patch_id"]) != patch.patch_id:
            raise ValueError("Coarse candidate order does not match surface patch order")
        metadata_raw = np.asarray(item["base_T_left_ir"], dtype=np.float64)
        metadata_rectified = np.asarray(item["base_T_left_rectified"], dtype=np.float64)
        if not np.array_equal(metadata_raw, raw_matrices[index]) or not np.array_equal(
            metadata_rectified, rectified_matrices[index]
        ):
            raise ValueError("Coarse candidate pose metadata does not match its arrays")
        view_id = str(item["view_id"])
        target = SurfacePatch(
            patch.patch_id,
            patch.side,
            patch.row,
            patch.column,
            patch.obb_center_m,
            patch.main_normal,
            patch.planar_extents_m,
        )
        candidates.append(
            CandidateView(
                view_id,
                target,
                PoseSE3("base", "left_ir", raw_matrices[index]),
                float(item["standoff_distance_m"]),
                tuple(float(value) for value in item["footprint_m"]),
                float(item["projection_fraction"]),
                float(item["visibility_fraction"]),
                str(item["distance_policy"]),
            )
        )
        rectified_poses.append(PoseSE3("base", "left_rectified", rectified_matrices[index]))
    footprint = tuple(float(value) for value in data["baseline_footprint_m"])
    if len(footprint) != 2:
        raise ValueError("Coarse baseline footprint must contain two values")
    return CurvedViewPlan(
        surface,
        tuple(candidates),
        tuple(rectified_poses),
        calibration,
        footprint,
    )


def _read_reference(path: Path, expected_sha256: str | None = None) -> _ReferenceGeometry:
    summary = read_coarse_model_summary(path)
    if int(summary.metadata["schema_version"]) != SURFACE_COVERAGE_COARSE_SCHEMA_VERSION:
        raise ValueError("Surface coverage requires an exact schema-5 coarse-model reference")
    metadata_sha256 = _sha256(summary.root / "metadata.json")
    if expected_sha256 is not None and metadata_sha256 != expected_sha256:
        raise ValueError("Coarse-model reference metadata checksum mismatch")
    surface, partition_config = _surface_from_reference(summary)
    return _ReferenceGeometry(
        summary,
        metadata_sha256,
        surface,
        _view_plan_from_reference(summary, surface),
        partition_config,
    )


def _required_regions(surface: CurvedBladeSurface) -> tuple[SurfaceRegion, ...]:
    return tuple(dict.fromkeys(patch.region for patch in surface.patches))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_policy_record(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("Fine selection policy payload must be an object")
    canonical = json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return {
        "id_schema": REACQUISITION_VIEW_ID_SCHEMA,
        "selection_policy_sha256": _canonical_sha256(canonical),
        "selection_policy": canonical,
    }


def _validated_selection_policy(
    record: object,
    *,
    reference: _ReferenceGeometry,
    quality_config: SurfaceQualityConfig,
) -> tuple[NextViewSelectionConfig, dict[str, Any]] | None:
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {
        "id_schema",
        "selection_policy_sha256",
        "selection_policy",
    }:
        raise ValueError("Fine reacquisition policy record is malformed")
    if record["id_schema"] != REACQUISITION_VIEW_ID_SCHEMA:
        raise ValueError("Fine reacquisition ID schema changed")
    payload = record["selection_policy"]
    if not isinstance(payload, dict) or set(payload) != {
        "algorithm",
        "selection",
        "surface_quality",
        "view_filter",
        "kinematics",
        "motion_endpoint_gate",
        "expected_reference",
        "terminal_reconstruction",
        "flange_T_left_ir",
        "fk_implementation",
    }:
        raise ValueError("Fine selection-policy payload is incomplete")
    policy_sha256 = str(record["selection_policy_sha256"])
    if _canonical_sha256(payload) != policy_sha256:
        raise ValueError("Fine selection-policy SHA-256 does not match its payload")
    if payload["algorithm"] != "bilateral_single_fin_coverage_priority_v2":
        raise ValueError("Fine selection-policy algorithm changed")
    selection = NextViewSelectionConfig.model_validate(payload["selection"])
    if selection.model_dump(mode="json") != payload["selection"]:
        raise ValueError("Fine reacquisition configuration is not canonical")
    stored_quality = SurfaceQualityConfig.model_validate(payload["surface_quality"])
    if stored_quality.model_dump(mode="json") != quality_config.model_dump(mode="json"):
        raise ValueError("Fine selection and coverage quality policies disagree")
    expected_reference = payload["expected_reference"]
    if not isinstance(expected_reference, dict) or set(expected_reference) != {
        "root",
        "metadata_sha256",
    }:
        raise ValueError("Fine selection-policy reference binding is malformed")
    if (
        Path(str(expected_reference["root"])).resolve() != reference.summary.root
        or str(expected_reference["metadata_sha256"]) != reference.metadata_sha256
    ):
        raise ValueError("Fine selection policy is bound to another coarse reference")
    endpoint = payload["motion_endpoint_gate"]
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "maximum_translation_error_m",
        "maximum_rotation_error_deg",
    }:
        raise ValueError("Fine selection-policy endpoint gate is malformed")
    endpoint_values = np.asarray(tuple(float(value) for value in endpoint.values()))
    if not np.isfinite(endpoint_values).all() or np.any(endpoint_values <= 0.0):
        raise ValueError("Fine selection-policy endpoint tolerances are invalid")
    PoseSE3("flange", "left_ir", payload["flange_T_left_ir"])
    if not str(payload["fk_implementation"]).strip():
        raise ValueError("Fine selection-policy FK implementation is missing")
    return selection, payload


def _provenance_from_payload(payload: object) -> FineReacquisitionProvenance:
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "id_schema",
        "view_id",
        "nominal_candidate_id",
        "patch_id",
        "attempt",
        "distance_offset_m",
        "tilt_deg",
        "azimuth_deg",
        "selection_policy_sha256",
        "reference_metadata_sha256",
    }:
        raise ValueError("Fine reacquisition provenance is malformed")
    if payload["kind"] != "bounded_reacquisition":
        raise ValueError("Fine reacquisition provenance kind changed")
    if payload["id_schema"] != REACQUISITION_VIEW_ID_SCHEMA:
        raise ValueError("Fine reacquisition provenance ID schema changed")
    return FineReacquisitionProvenance(
        view_id=payload["view_id"],
        nominal_candidate_id=payload["nominal_candidate_id"],
        patch_id=payload["patch_id"],
        attempt=payload["attempt"],
        distance_offset_m=payload["distance_offset_m"],
        tilt_deg=payload["tilt_deg"],
        azimuth_deg=payload["azimuth_deg"],
        selection_policy_sha256=payload["selection_policy_sha256"],
        reference_metadata_sha256=payload["reference_metadata_sha256"],
    )


def _rotation_error_deg(expected: PoseSE3, actual: PoseSE3) -> float:
    relative = expected.rotation.T @ actual.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _validate_reacquisition_provenance(
    provenance: FineReacquisitionProvenance,
    current: StoredReconstructedBladeView,
    reference: _ReferenceGeometry,
    policy_record: dict[str, Any] | None,
    quality_config: SurfaceQualityConfig,
) -> None:
    validated = _validated_selection_policy(
        policy_record,
        reference=reference,
        quality_config=quality_config,
    )
    if validated is None:
        raise ValueError("A fine retry requires a pinned selection-policy payload")
    selection, policy_payload = validated
    policy_sha256 = str(policy_record["selection_policy_sha256"])
    if (
        provenance.view_id != current.view.source_view_id
        or provenance.selection_policy_sha256 != policy_sha256
        or provenance.reference_metadata_sha256 != reference.metadata_sha256
    ):
        raise ValueError("Fine retry identity, policy, or reference binding changed")
    hand_eye = current.metadata.get("hand_eye")
    if not isinstance(hand_eye, dict) or "flange_T_left_ir" not in hand_eye:
        raise ValueError("Fine retry reconstructed view lacks hand-eye provenance")
    expected_flange_t_left_ir = PoseSE3(
        "flange",
        "left_ir",
        policy_payload["flange_T_left_ir"],
    )
    actual_flange_t_left_ir = PoseSE3(
        "flange",
        "left_ir",
        hand_eye["flange_T_left_ir"],
    )
    if not np.allclose(
        actual_flange_t_left_ir.matrix,
        expected_flange_t_left_ir.matrix,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Fine retry hand-eye differs from its pinned selection policy")
    matches = tuple(
        (candidate, projection)
        for candidate, projection in zip(
            reference.view_plan.candidates,
            reference.view_plan.candidate_base_t_left_rectified,
            strict=True,
        )
        if candidate.view_id == provenance.nominal_candidate_id
    )
    if len(matches) != 1:
        raise ValueError("Fine retry nominal candidate is not unique in the reference")
    nominal, nominal_projection = matches[0]
    if nominal.patch.patch_id != provenance.patch_id:
        raise ValueError("Fine retry patch differs from its nominal candidate")
    if provenance.attempt > selection.maximum_reacquisition_attempts_per_patch:
        raise ValueError("Fine retry attempt exceeds the pinned bounded budget")
    perturbation = selection.reacquisition_perturbations[provenance.attempt - 1]
    expected_values = (
        perturbation.distance_offset_m,
        perturbation.tilt_deg,
        perturbation.azimuth_deg,
    )
    actual_values = (
        provenance.distance_offset_m,
        provenance.tilt_deg,
        provenance.azimuth_deg,
    )
    if actual_values != expected_values:
        raise ValueError("Fine retry perturbation differs from the pinned attempt")
    expected_id = reacquisition_view_id(
        nominal.view_id,
        nominal.patch.patch_id,
        provenance.attempt,
        policy_sha256,
    )
    if provenance.view_id != expected_id:
        raise ValueError("Fine retry view ID does not replay from its provenance")
    planning = ViewPlanningConfig.model_validate(
        reference.summary.metadata["view_plan"]["configuration"]
    )
    lower = planning.minimum_standoff_distance_m
    upper = planning.maximum_standoff_distance_m
    if lower is None or upper is None:
        raise ValueError("Fine retry reference has no bounded standoff interval")
    expected_candidate, expected_projection = generate_reacquisition_view(
        nominal,
        nominal_projection,
        reference.view_plan.left_rectified_t_left_ir,
        ReacquisitionPerturbationConfig.model_validate(perturbation),
        view_id=expected_id,
        minimum_standoff_distance_m=lower,
        maximum_standoff_distance_m=upper,
    )
    if expected_candidate.patch.patch_id != provenance.patch_id:
        raise ValueError("Fine retry replay changed its target patch")
    actual_projection = current.view.base_t_projection_camera
    translation_error = float(
        np.linalg.norm(actual_projection.translation_m - expected_projection.translation_m)
    )
    rotation_error = _rotation_error_deg(expected_projection, actual_projection)
    endpoint = policy_payload["motion_endpoint_gate"]
    if (
        translation_error > float(endpoint["maximum_translation_error_m"])
        or rotation_error > float(endpoint["maximum_rotation_error_deg"])
    ):
        raise ValueError("Fine retry capture is outside its pinned endpoint tolerance")


def _physical_source_identity(
    current: StoredReconstructedBladeView,
) -> PhysicalSourceIdentity | None:
    if int(current.metadata["schema_version"]) != SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION:
        # Schema-2 views remain readable for offline/legacy coverage.  They have no
        # foreground asset that strictly replays the raw stereo source, so they
        # cannot contribute a physical-frame identity to a production lineage.
        return None
    # The schema-3 reconstructed-view reader has already replayed its foreground
    # asset, which strictly verifies the raw session behind this stereo artifact.
    # Re-read the checksummed stereo metadata here to derive a path-independent ID.
    source = current.metadata["source"]
    stereo_value = source.get("stereo_inference")
    if stereo_value is None:
        raise ValueError("Science coverage observation has no stereo-inference source")
    stereo = read_stereo_inference(Path(str(stereo_value)).resolve())
    observation = stereo.observation
    if (
        observation.source_view_id != current.view.source_view_id
        or observation.source_sequence_index != current.view.source_sequence_index
        or observation.rectified.source_frame_number != current.view.source_frame_number
    ):
        raise ValueError("Science coverage stereo physical identity changed")
    integrity = stereo.metadata["source"].get("raw_session_integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "session_manifest_sha256",
        "view_metadata_sha256",
        "left_ir_npy_sha256",
        "right_ir_npy_sha256",
        "raw_calibration_content_hash",
    }:
        raise ValueError("Science coverage stereo lacks complete raw-session integrity")
    for label, key in (
        ("session manifest", "session_manifest_sha256"),
        ("view metadata", "view_metadata_sha256"),
        ("left IR array", "left_ir_npy_sha256"),
        ("right IR array", "right_ir_npy_sha256"),
        ("raw calibration", "raw_calibration_content_hash"),
    ):
        digest = integrity[key]
        if type(digest) is not str:
            raise TypeError(f"Science coverage {label} identity must be a string")
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError(f"Science coverage {label} identity is not a SHA-256")
    manifest_sha256 = integrity["session_manifest_sha256"]
    view_metadata_sha256 = integrity["view_metadata_sha256"]
    return (
        manifest_sha256,
        view_metadata_sha256,
        int(current.view.source_sequence_index),
        int(current.view.source_frame_number),
    )


def _validate_reacquisition_lineage(
    provenance: FineReacquisitionProvenance,
    previous: StoredSurfaceCoverageGeneration,
    reference: _ReferenceGeometry,
    policy_record: dict[str, Any] | None,
) -> None:
    validated = _validated_selection_policy(
        policy_record,
        reference=reference,
        quality_config=previous.quality_config,
    )
    if validated is None:
        raise ValueError("A fine retry lineage requires a pinned selection policy")
    if provenance.nominal_candidate_id not in previous.ledger.observation_ids:
        raise ValueError("Fine retry cannot precede its nominal candidate observation")
    quality = next(
        (item for item in previous.quality.patches if item.patch_id == provenance.patch_id),
        None,
    )
    if quality is None or quality.complete:
        raise ValueError("Fine retry requires an incomplete target patch")


def _validate_fine_reconstructed_view(
    current: StoredReconstructedBladeView,
    reference: _ReferenceGeometry,
    *,
    require_foreground_bound_science: bool,
    quality_config: SurfaceQualityConfig,
    policy_record: dict[str, Any] | None,
    reacquisition: FineReacquisitionProvenance | None,
) -> None:
    schema_version = int(current.metadata["schema_version"])
    if (
        schema_version
        not in {
            RECONSTRUCTED_VIEW_SCHEMA_VERSION,
            SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION,
        }
        or current.view.pose_authority is None
    ):
        raise ValueError(
            "Fine coverage requires a current authoritative schema-2/3 reconstructed view"
        )
    if (
        require_foreground_bound_science
        and schema_version != SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION
    ):
        raise ValueError(
            "Online fine coverage requires every observation in the lineage to use "
            "a foreground-bound schema-3 reconstructed view"
        )
    if current.view.depth_source != "foundation_stereo":
        raise ValueError("Fine coverage requires a FoundationStereo reconstructed view")
    candidate_ids = {candidate.view_id for candidate in reference.view_plan.candidates}
    if reacquisition is None:
        if current.view.source_view_id not in candidate_ids:
            raise ValueError("Fine reconstructed source view ID is not a reference candidate ID")
    else:
        if schema_version != SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION:
            raise ValueError(
                "A fine retry requires a foreground-bound schema-3 reconstructed view"
            )
        if current.view.source_view_id in candidate_ids:
            raise ValueError("A nominal fine candidate cannot cite retry provenance")
        _validate_reacquisition_provenance(
            reacquisition,
            current,
            reference,
            policy_record,
            quality_config,
        )
    expected_base_t_left_rectified = current.view.base_t_left_ir.compose(
        reference.view_plan.left_rectified_t_left_ir.inverse()
    )
    if not np.allclose(
        current.view.base_t_projection_camera.matrix,
        expected_base_t_left_rectified.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Fine reconstructed raw/rectified pose chain differs from its coarse reference"
        )


def _validate_ledger(ledger: SurfaceCoverageLedger, surface: CurvedBladeSurface) -> None:
    patch_ids = tuple(patch.patch_id for patch in surface.patches)
    if tuple(item.patch_id for item in ledger.evidence) != patch_ids:
        raise ValueError("Fine coverage ledger patch IDs/order do not match the reference")
    if any(not item for item in ledger.observation_ids):
        raise ValueError("Fine coverage observation IDs must be non-empty")
    order = {observation_id: index for index, observation_id in enumerate(ledger.observation_ids)}
    for patch, evidence in zip(surface.patches, ledger.evidence, strict=True):
        if len(evidence.minimum_distances_m) != len(patch.points_m):
            raise ValueError(f"Fine evidence length does not match patch {patch.patch_id}")
        distances = evidence.minimum_distances_m
        cosines = evidence.best_normal_cosines
        if np.any(distances < 0.0) or not np.isfinite(cosines).all():
            raise ValueError(f"Fine evidence values are invalid for patch {patch.patch_id}")
        if np.any(cosines < -1.0) or np.any(cosines > 1.0):
            raise ValueError(f"Fine normal evidence is outside [-1, 1] for {patch.patch_id}")
        if any(observation_id not in order for observation_id in evidence.observation_ids):
            raise ValueError(f"Patch {patch.patch_id} cites an unknown observation")
        indices = tuple(order[item] for item in evidence.observation_ids)
        if indices != tuple(sorted(indices)):
            raise ValueError(f"Patch {patch.patch_id} observation lineage is out of order")


def _assert_same_ledger(
    actual: SurfaceCoverageLedger,
    expected: SurfaceCoverageLedger,
    *,
    label: str,
) -> None:
    if actual.observation_ids != expected.observation_ids or len(actual.evidence) != len(
        expected.evidence
    ):
        raise ValueError(f"{label} observation lineage does not match")
    for first, second in zip(actual.evidence, expected.evidence, strict=True):
        if (
            first.patch_id != second.patch_id
            or first.observation_ids != second.observation_ids
            or not np.array_equal(first.minimum_distances_m, second.minimum_distances_m)
            or not np.array_equal(first.best_normal_cosines, second.best_normal_cosines)
        ):
            raise ValueError(f"{label} evidence does not match patch {second.patch_id}")


def _quality_payload(report: SurfaceQualityReport) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "completion_fraction": report.completion_fraction,
        "mesh_triangle_count": report.mesh_triangle_count,
        "mesh_boundary_edge_count": report.mesh_boundary_edge_count,
        "mesh_boundary_loop_count": report.mesh_boundary_loop_count,
        "mesh_watertight": report.mesh_watertight,
        "edge_completion": {
            region.value: value for region, value in report.edge_completion.items()
        },
        "patches": [
            {
                "patch_id": item.patch_id,
                "side": item.side.value,
                "region": item.region.value,
                "reference_point_count": item.reference_point_count,
                "observed_point_count": item.observed_point_count,
                "coverage_fraction": item.coverage_fraction,
                "rmse_m": finite_or_none(item.rmse_m),
                "normal_consistency": item.normal_consistency,
                "curvature_deg": item.curvature_deg,
                "complete": item.complete,
                "reasons": list(item.reasons),
            }
            for item in report.patches
        ],
    }


def _quality_summary(
    report: SurfaceQualityReport, required_patch_ids: tuple[str, ...]
) -> dict[str, Any]:
    by_id = {item.patch_id: item for item in report.patches}
    incomplete = tuple(patch_id for patch_id in required_patch_ids if not by_id[patch_id].complete)
    return {
        "complete": not incomplete,
        "required_patch_count": len(required_patch_ids),
        "complete_patch_count": len(required_patch_ids) - len(incomplete),
        "incomplete_patch_ids": list(incomplete),
    }


def _generation_id(payload: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "generation_id"}
    }
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_payload(ledger: SurfaceCoverageLedger, offsets: list[int]) -> dict[str, Any]:
    return {
        "observation_ids": list(ledger.observation_ids),
        "patches": [
            {
                "patch_id": item.patch_id,
                "point_range": [offsets[index], offsets[index + 1]],
                "observation_ids": list(item.observation_ids),
            }
            for index, item in enumerate(ledger.evidence)
        ],
    }


def _ledger_from_payload(
    root: Path,
    payload: dict[str, Any],
    surface: CurvedBladeSurface,
) -> SurfaceCoverageLedger:
    files = payload["files"]
    if set(files) != set(_ARRAY_NAMES):
        raise ValueError("Surface-coverage array manifest has unexpected entries")
    distances = _load_array(
        root,
        files["minimum_distances_m"],
        label="surface-coverage distances",
        dtype=np.float64,
    )
    cosines = _load_array(
        root,
        files["best_normal_cosines"],
        label="surface-coverage normal evidence",
        dtype=np.float64,
    )
    ledger_data = payload["ledger"]
    patches_data = ledger_data["patches"]
    if distances.ndim != 1 or cosines.shape != distances.shape:
        raise ValueError("Surface-coverage evidence arrays have inconsistent shapes")
    if len(patches_data) != len(surface.patches):
        raise ValueError("Surface-coverage patch metadata does not match its reference")
    offsets = _offsets(
        _load_array(
            root,
            files["patch_offsets"],
            label="surface-coverage offsets",
            dtype=np.int64,
        ),
        count=len(surface.patches),
        total=len(distances),
        label="surface-coverage patch",
    )
    evidence: list[SurfacePatchEvidence] = []
    for index, (patch, data) in enumerate(zip(surface.patches, patches_data, strict=True)):
        first, last = int(offsets[index]), int(offsets[index + 1])
        _range(data, (first, last), label=f"surface-coverage patch {index}")
        if str(data["patch_id"]) != patch.patch_id or last - first != len(patch.points_m):
            raise ValueError("Surface-coverage patch identity/shape drifted from its reference")
        evidence.append(
            SurfacePatchEvidence(
                patch.patch_id,
                distances[first:last],
                cosines[first:last],
                tuple(str(value) for value in data["observation_ids"]),
            )
        )
    ledger = SurfaceCoverageLedger(
        tuple(evidence), tuple(str(value) for value in ledger_data["observation_ids"])
    )
    _validate_ledger(ledger, surface)
    return ledger


def write_surface_coverage_generation(
    output_dir: str | Path,
    *,
    reference_coarse_model: str | Path,
    quality_config: SurfaceQualityConfig,
    previous_generation: str | Path | None = None,
    current_reconstructed_view: str | Path | None = None,
    observation_id: str | None = None,
    ledger: SurfaceCoverageLedger | None = None,
    selection_policy_payload: dict[str, Any] | None = None,
    current_reacquisition: FineReacquisitionProvenance | None = None,
) -> Path:
    """Write an initial empty ledger or append exactly one fine observation.

    When ``ledger`` is omitted, the writer constructs the only admissible ledger.
    When supplied, it is treated as an assertion and must match that construction
    element-for-element.
    """

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Surface-coverage output already exists: {output}")
    config = SurfaceQualityConfig.model_validate(quality_config)
    reference = _read_reference(Path(reference_coarse_model).resolve())
    required_patch_ids = tuple(patch.patch_id for patch in reference.surface.patches)
    required_regions = _required_regions(reference.surface)
    reference_record = {
        "kind": "coarse_model_schema_5",
        **_source_record(reference.summary.root, "metadata.json"),
    }
    requested_policy = _selection_policy_record(selection_policy_payload)
    if requested_policy is not None:
        _validated_selection_policy(
            requested_policy,
            reference=reference,
            quality_config=config,
        )

    previous_record: dict[str, Any] | None = None
    current_record: dict[str, Any] | None = None
    policy_record = requested_policy
    if previous_generation is None:
        if (
            current_reconstructed_view is not None
            or observation_id is not None
            or current_reacquisition is not None
        ):
            raise ValueError(
                "Initial fine coverage generation cannot contain a reconstructed observation"
            )
        expected = create_surface_coverage_ledger(reference.surface)
    else:
        if current_reconstructed_view is None or observation_id is None or not observation_id:
            raise ValueError(
                "A successor generation requires one reconstructed view and observation ID"
            )
        previous = read_surface_coverage_generation(previous_generation)
        if (
            previous.reference.root != reference.summary.root
            or previous.metadata["reference"]["metadata_sha256"] != reference.metadata_sha256
        ):
            raise ValueError("Previous generation uses a different coarse-model reference")
        if previous.quality_config.model_dump(mode="json") != config.model_dump(mode="json"):
            raise ValueError("Surface quality configuration cannot drift between generations")
        if (
            previous.required_patch_ids != required_patch_ids
            or previous.required_regions != required_regions
        ):
            raise ValueError("Required surface identities cannot drift between generations")
        inherited_policy = previous.metadata.get("reacquisition_policy")
        if requested_policy is not None and requested_policy != inherited_policy:
            raise ValueError("Fine selection policy cannot drift between generations")
        policy_record = inherited_policy
        _validated_selection_policy(
            policy_record,
            reference=reference,
            quality_config=config,
        )
        current_path = Path(current_reconstructed_view).resolve()
        current = read_reconstructed_view(current_path)
        _validate_fine_reconstructed_view(
            current,
            reference,
            require_foreground_bound_science=False,
            quality_config=config,
            policy_record=policy_record,
            reacquisition=current_reacquisition,
        )
        if current_reacquisition is not None:
            _validate_reacquisition_lineage(
                current_reacquisition,
                previous,
                reference,
                policy_record,
            )
        if observation_id != current.view.source_view_id:
            raise ValueError("Fine observation ID must equal the reconstructed source view ID")
        if observation_id in previous.ledger.observation_ids:
            raise ValueError("Fine observation ID was already committed")
        physical_identity = _physical_source_identity(current)
        if (
            physical_identity is not None
            and physical_identity in previous.physical_source_identities
        ):
            raise ValueError("Fine coverage cannot reuse one physical camera frame")
        expected = update_surface_coverage(
            previous.ledger,
            reference.surface,
            registered_cloud_view(current.view),
            observation_id,
            config,
        )
        previous_record = {
            **_source_record(previous.root, _METADATA_NAME),
            "generation_id": previous.generation_id,
        }
        current_record = {
            **_source_record(current_path, "metadata.json"),
            "observation_id": observation_id,
            "source_view_id": current.view.source_view_id,
            "source_sequence_index": current.view.source_sequence_index,
            "source_frame_number": current.view.source_frame_number,
            "view_authority": (
                {
                    "kind": "nominal_candidate",
                    "view_id": current.view.source_view_id,
                    "patch_id": next(
                        candidate.patch.patch_id
                        for candidate in reference.view_plan.candidates
                        if candidate.view_id == current.view.source_view_id
                    ),
                }
                if current_reacquisition is None
                else current_reacquisition.to_payload()
            ),
        }
    if ledger is not None:
        _validate_ledger(ledger, reference.surface)
        _assert_same_ledger(ledger, expected, label="Supplied fine ledger")
    ledger = expected
    # The coarse mesh is a fixed targeting reference, not accumulated fine
    # reconstruction evidence.  Reporting it here would falsely make every fine
    # generation inherit coarse mesh quality, so mesh diagnostics remain unknown.
    quality = evaluate_surface_quality(ledger, reference.surface, config)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    offsets = [0]
    for item in ledger.evidence:
        offsets.append(offsets[-1] + len(item.minimum_distances_m))
    arrays = {
        "minimum_distances_m": np.concatenate(
            [item.minimum_distances_m for item in ledger.evidence]
        ),
        "best_normal_cosines": np.concatenate(
            [item.best_normal_cosines for item in ledger.evidence]
        ),
        "patch_offsets": np.asarray(offsets, dtype=np.int64),
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        payload: dict[str, Any] = {
            "schema_version": SURFACE_COVERAGE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "reference": reference_record,
            "reacquisition_policy": policy_record,
            "previous_generation": previous_record,
            "current_observation": current_record,
            "files": {name: _record(temporary / f"{name}.npy") for name in arrays},
            "quality_configuration": config.model_dump(mode="json"),
            "required_patch_ids": list(required_patch_ids),
            "required_regions": [region.value for region in required_regions],
            "ledger": _ledger_payload(ledger, offsets),
            "quality": _quality_payload(quality),
            "summary": _quality_summary(quality, required_patch_ids),
        }
        payload["generation_id"] = _generation_id(payload)
        (temporary / _METADATA_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _read_generation(
    root: Path,
    *,
    active: set[Path],
    references: dict[tuple[Path, str], _ReferenceGeometry],
    require_foreground_bound_science: bool,
) -> StoredSurfaceCoverageGeneration:
    resolved = root.resolve()
    if resolved in active:
        raise ValueError(f"Surface-coverage lineage contains a cycle at {resolved}")
    active.add(resolved)
    try:
        metadata_path = resolved / _METADATA_NAME
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        schema_version = int(payload["schema_version"])
        if schema_version not in {
            LEGACY_SURFACE_COVERAGE_SCHEMA_VERSION,
            SURFACE_COVERAGE_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("Surface-coverage generations must explicitly forbid motion")
        reference_record = payload["reference"]
        if reference_record.get("kind") != "coarse_model_schema_5":
            raise ValueError("Surface-coverage reference kind is unsupported")
        reference_root = _verify_source_record(
            reference_record, "metadata.json", label="coarse-model reference"
        )
        reference_sha256 = str(reference_record["metadata_sha256"])
        reference_key = (reference_root, reference_sha256)
        reference = references.get(reference_key)
        if reference is None:
            reference = _read_reference(reference_root, reference_sha256)
            references[reference_key] = reference
        config = SurfaceQualityConfig.model_validate(payload["quality_configuration"])
        if config.model_dump(mode="json") != payload["quality_configuration"]:
            raise ValueError("Surface quality configuration is not canonical")
        policy_record = (
            payload.get("reacquisition_policy")
            if schema_version == SURFACE_COVERAGE_SCHEMA_VERSION
            else None
        )
        _validated_selection_policy(
            policy_record,
            reference=reference,
            quality_config=config,
        )
        required_patch_ids = tuple(str(value) for value in payload["required_patch_ids"])
        expected_patch_ids = tuple(patch.patch_id for patch in reference.surface.patches)
        if required_patch_ids != expected_patch_ids:
            raise ValueError("Required patch IDs drifted from the coarse reference")
        required_regions = tuple(SurfaceRegion(str(value)) for value in payload["required_regions"])
        if required_regions != _required_regions(reference.surface):
            raise ValueError("Required regions drifted from the coarse reference")
        ledger = _ledger_from_payload(resolved, payload, reference.surface)

        previous_path: Path | None = None
        current_path: Path | None = None
        current_reacquisition: FineReacquisitionProvenance | None = None
        physical_source_identities: tuple[PhysicalSourceIdentity, ...] = ()
        previous_record = payload["previous_generation"]
        current_record = payload["current_observation"]
        if previous_record is None:
            if current_record is not None:
                raise ValueError("Initial fine coverage generation contains an observation")
            _assert_same_ledger(
                ledger,
                create_surface_coverage_ledger(reference.surface),
                label="Initial fine ledger",
            )
        else:
            if current_record is None:
                raise ValueError("Successor generation is missing its current observation")
            previous_path = _verify_source_record(
                previous_record, _METADATA_NAME, label="previous surface-coverage generation"
            )
            previous = _read_generation(
                previous_path,
                active=active,
                references=references,
                require_foreground_bound_science=require_foreground_bound_science,
            )
            if str(previous_record["generation_id"]) != previous.generation_id:
                raise ValueError("Previous surface-coverage generation ID changed")
            if (
                previous.reference.root != reference.summary.root
                or previous.metadata["reference"]["metadata_sha256"] != reference.metadata_sha256
            ):
                raise ValueError("Surface-coverage coarse reference drifted across lineage")
            if previous.quality_config.model_dump(mode="json") != config.model_dump(mode="json"):
                raise ValueError("Surface quality configuration drifted across lineage")
            if previous.metadata.get("reacquisition_policy") != policy_record:
                raise ValueError("Fine reacquisition policy drifted across lineage")
            if (
                previous.required_patch_ids != required_patch_ids
                or previous.required_regions != required_regions
            ):
                raise ValueError("Required surface identities drifted across lineage")
            current_path = _verify_source_record(
                current_record, "metadata.json", label="current reconstructed view"
            )
            current = read_reconstructed_view(current_path)
            if schema_version == SURFACE_COVERAGE_SCHEMA_VERSION:
                authority = current_record.get("view_authority")
                if not isinstance(authority, dict):
                    raise ValueError("Current fine observation lacks typed view authority")
                kind = authority.get("kind")
                if kind == "nominal_candidate":
                    if set(authority) != {"kind", "view_id", "patch_id"}:
                        raise ValueError("Nominal fine-view authority is malformed")
                    matches = tuple(
                        candidate
                        for candidate in reference.view_plan.candidates
                        if candidate.view_id == str(authority["view_id"])
                    )
                    if (
                        len(matches) != 1
                        or matches[0].patch.patch_id != str(authority["patch_id"])
                        or str(authority["view_id"]) != current.view.source_view_id
                    ):
                        raise ValueError("Nominal fine-view authority changed")
                elif kind == "bounded_reacquisition":
                    current_reacquisition = _provenance_from_payload(authority)
                else:
                    raise ValueError("Current fine-view authority kind is unsupported")
            _validate_fine_reconstructed_view(
                current,
                reference,
                require_foreground_bound_science=require_foreground_bound_science,
                quality_config=config,
                policy_record=policy_record,
                reacquisition=current_reacquisition,
            )
            if current_reacquisition is not None:
                _validate_reacquisition_lineage(
                    current_reacquisition,
                    previous,
                    reference,
                    policy_record,
                )
            if (
                str(current_record["source_view_id"]) != current.view.source_view_id
                or int(current_record["source_sequence_index"])
                != current.view.source_sequence_index
                or int(current_record["source_frame_number"]) != current.view.source_frame_number
            ):
                raise ValueError("Current reconstructed-view identity summary changed")
            observation_id = str(current_record["observation_id"])
            if observation_id != current.view.source_view_id:
                raise ValueError(
                    "Current fine observation ID does not equal reconstructed source view ID"
                )
            if ledger.observation_ids != (*previous.ledger.observation_ids, observation_id):
                raise ValueError("Successor must append exactly its one current observation")
            physical_identity = _physical_source_identity(current)
            if (
                physical_identity is not None
                and physical_identity in previous.physical_source_identities
            ):
                raise ValueError("Fine coverage lineage reuses one physical camera frame")
            physical_source_identities = previous.physical_source_identities
            if physical_identity is not None:
                physical_source_identities = (*physical_source_identities, physical_identity)
            expected = update_surface_coverage(
                previous.ledger,
                reference.surface,
                registered_cloud_view(current.view),
                observation_id,
                config,
            )
            _assert_same_ledger(ledger, expected, label="Replayed successor ledger")

        quality = evaluate_surface_quality(ledger, reference.surface, config)
        if payload["quality"] != _quality_payload(quality):
            raise ValueError("Stored surface quality does not match independent evaluation")
        if payload["summary"] != _quality_summary(quality, required_patch_ids):
            raise ValueError("Stored surface-coverage completion summary is false")
        generation_id = str(payload["generation_id"])
        if not generation_id or generation_id != _generation_id(payload):
            raise ValueError("Surface-coverage generation ID does not match its contents")
        return StoredSurfaceCoverageGeneration(
            resolved,
            generation_id,
            _sha256(metadata_path),
            reference.summary,
            reference.surface,
            reference.view_plan,
            ledger,
            quality,
            config,
            required_patch_ids,
            required_regions,
            previous_path,
            current_path,
            payload,
            current_reacquisition,
            physical_source_identities,
        )
    finally:
        active.remove(resolved)


def read_surface_coverage_generation(
    path: str | Path,
    *,
    require_foreground_bound_science: bool = False,
) -> StoredSurfaceCoverageGeneration:
    """Validate provenance, replay lineage, and independently recompute quality.

    The generic reader retains schema-2 compatibility for offline/manual assets.  An
    online scientific run sets ``require_foreground_bound_science`` so every non-empty
    generation in the recursively replayed lineage must point to a schema-3 view whose
    foreground/source bindings have themselves passed semantic readback.
    """

    root = Path(path)
    try:
        if type(require_foreground_bound_science) is not bool:
            raise TypeError("require_foreground_bound_science must be a bool")
        return _read_generation(
            root,
            active=set(),
            references={},
            require_foreground_bound_science=require_foreground_bound_science,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid surface-coverage generation {root}: {exc}") from exc
