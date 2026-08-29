"""Immutable terminal fine reconstruction and strict replay verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.settings import (
    FineFinalizationConfig,
    MultiViewFusionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
)
from biblade_fusion.perception.fusion import FusedBladeCloud, PoseRefinement
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.perception.tsdf import (
    BilateralTSDFResult,
    SparseTSDFVolume,
    TriangleMesh,
)
from biblade_fusion.planning.surface_coverage import (
    SurfacePatchQuality,
    SurfaceQualityReport,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.surface_coverage import read_surface_coverage_generation
from biblade_fusion.workflows.coarse_model import registered_cloud_view
from biblade_fusion.workflows.fine_finalization import (
    FinalFineReconstruction,
    FineFinalizationGateReport,
    build_final_fine_reconstruction,
)

FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION = 2
_LEGACY_FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION = 1
_METADATA_NAME = "final_reconstruction.json"
_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "fused_points_m": np.dtype(np.float64),
    "fused_normals": np.dtype(np.float64),
    "fused_side_labels": np.dtype(np.int8),
    "front_tsdf_indices": np.dtype(np.int32),
    "front_tsdf_values": np.dtype(np.float64),
    "front_tsdf_weights": np.dtype(np.float64),
    "back_tsdf_indices": np.dtype(np.int32),
    "back_tsdf_values": np.dtype(np.float64),
    "back_tsdf_weights": np.dtype(np.float64),
    "mesh_vertices_m": np.dtype(np.float64),
    "mesh_triangles": np.dtype(np.int32),
    "mesh_triangle_sides": np.dtype(np.int8),
}


@dataclass(frozen=True, slots=True)
class StoredFinalFineReconstruction:
    root: Path
    artifact_id: str
    metadata_sha256: str
    result: FinalFineReconstruction
    fusion_config: MultiViewFusionConfig
    tsdf_config: TSDFConfig
    surface_quality_config: SurfaceQualityConfig
    finalization_config: FineFinalizationConfig
    science_authority: ScienceAcceptanceAuthority | None
    metadata: dict[str, Any]

    @property
    def motion_authorized(self) -> bool:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def _source_record(root: Path, metadata_name: str) -> dict[str, Any]:
    resolved = root.resolve()
    metadata = resolved / metadata_name
    if root != resolved or not resolved.is_dir() or not metadata.is_file():
        raise ValueError(f"Final reconstruction source is not canonical: {root}")
    return {
        "root": str(resolved),
        "metadata_name": metadata_name,
        "metadata_sha256": _sha256(metadata),
        "metadata_size_bytes": metadata.stat().st_size,
    }


def _verify_source(record: dict[str, Any], *, label: str) -> Path:
    if set(record) != {
        "root",
        "metadata_name",
        "metadata_sha256",
        "metadata_size_bytes",
    }:
        raise ValueError(f"{label} source record has unexpected fields")
    root = Path(str(record["root"]))
    resolved = root.resolve()
    if not root.is_absolute() or root != resolved:
        raise ValueError(f"{label} source root is not absolute and canonical")
    metadata_name = str(record["metadata_name"])
    if Path(metadata_name).name != metadata_name:
        raise ValueError(f"{label} metadata name is unsafe")
    metadata = resolved / metadata_name
    if (
        not resolved.is_dir()
        or not metadata.is_file()
        or metadata.stat().st_size != int(record["metadata_size_bytes"])
        or _sha256(metadata) != str(record["metadata_sha256"])
    ):
        raise ValueError(f"{label} immutable source changed: {resolved}")
    return resolved


def _quality_payload(report: SurfaceQualityReport) -> dict[str, Any]:
    return {
        "patches": [
            {
                "patch_id": item.patch_id,
                "side": item.side.value,
                "region": item.region.value,
                "reference_point_count": item.reference_point_count,
                "observed_point_count": item.observed_point_count,
                "coverage_fraction": item.coverage_fraction,
                "rmse_m": item.rmse_m,
                "normal_consistency": item.normal_consistency,
                "curvature_deg": item.curvature_deg,
                "complete": item.complete,
                "reasons": list(item.reasons),
            }
            for item in report.patches
        ],
        "completion_fraction": report.completion_fraction,
        "edge_completion": {
            region.value: value for region, value in report.edge_completion.items()
        },
        "mesh_triangle_count": report.mesh_triangle_count,
        "mesh_boundary_edge_count": report.mesh_boundary_edge_count,
        "mesh_boundary_loop_count": report.mesh_boundary_loop_count,
        "mesh_watertight": report.mesh_watertight,
    }


def _quality(payload: dict[str, Any]) -> SurfaceQualityReport:
    if set(payload) != {
        "patches",
        "completion_fraction",
        "edge_completion",
        "mesh_triangle_count",
        "mesh_boundary_edge_count",
        "mesh_boundary_loop_count",
        "mesh_watertight",
    }:
        raise ValueError("Stored terminal surface-quality fields changed")
    expected_patch_fields = {
        "patch_id",
        "side",
        "region",
        "reference_point_count",
        "observed_point_count",
        "coverage_fraction",
        "rmse_m",
        "normal_consistency",
        "curvature_deg",
        "complete",
        "reasons",
    }
    if not isinstance(payload["patches"], list) or any(
        not isinstance(item, dict) or set(item) != expected_patch_fields
        for item in payload["patches"]
    ):
        raise ValueError("Stored terminal patch-quality fields changed")
    patches = tuple(
        SurfacePatchQuality(
            str(item["patch_id"]),
            BladeSide(str(item["side"])),
            SurfaceRegion(str(item["region"])),
            int(item["reference_point_count"]),
            int(item["observed_point_count"]),
            float(item["coverage_fraction"]),
            float(item["rmse_m"]),
            float(item["normal_consistency"]),
            float(item["curvature_deg"]),
            bool(item["complete"]),
            tuple(str(value) for value in item["reasons"]),
        )
        for item in payload["patches"]
    )
    return SurfaceQualityReport(
        patches,
        float(payload["completion_fraction"]),
        {
            SurfaceRegion(name): float(value)
            for name, value in payload["edge_completion"].items()
        },
        int(payload["mesh_triangle_count"]),
        int(payload["mesh_boundary_edge_count"]),
        int(payload["mesh_boundary_loop_count"]),
        bool(payload["mesh_watertight"]),
    )


def _gate_payload(report: FineFinalizationGateReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["violations"] = list(report.violations)
    payload["passed"] = report.passed
    return payload


def _gates(payload: dict[str, Any]) -> FineFinalizationGateReport:
    if set(payload) != {
        "required_patch_count",
        "complete_patch_count",
        "front_source_view_count",
        "back_source_view_count",
        "front_mesh_triangle_count",
        "back_mesh_triangle_count",
        "front_fin_count",
        "back_fin_count",
        "mesh_boundary_edge_count",
        "mesh_boundary_loop_count",
        "mesh_watertight",
        "violations",
        "passed",
    }:
        raise ValueError("Stored terminal gate fields changed")
    if payload.get("passed") is not True or payload.get("violations") != []:
        raise ValueError("Stored terminal reconstruction does not contain passing gates")
    return FineFinalizationGateReport(
        int(payload["required_patch_count"]),
        int(payload["complete_patch_count"]),
        int(payload["front_source_view_count"]),
        int(payload["back_source_view_count"]),
        int(payload["front_mesh_triangle_count"]),
        int(payload["back_mesh_triangle_count"]),
        int(payload["front_fin_count"]),
        int(payload["back_fin_count"]),
        int(payload["mesh_boundary_edge_count"]),
        int(payload["mesh_boundary_loop_count"]),
        bool(payload["mesh_watertight"]),
        tuple(str(value) for value in payload["violations"]),
    )


def _load_array(root: Path, record: dict[str, Any], name: str) -> np.ndarray:
    if set(record) != {"path", "sha256", "size_bytes", "dtype", "shape"}:
        raise ValueError(f"Final reconstruction {name} record has unexpected fields")
    relative = Path(str(record["path"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path.parent != root:
        raise ValueError(f"Final reconstruction {name} path escapes its asset")
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or _sha256(path) != str(record["sha256"])
    ):
        raise ValueError(f"Final reconstruction {name} checksum mismatch")
    array = np.load(path, allow_pickle=False)
    expected_dtype = _ARRAY_DTYPES[name]
    if (
        array.dtype != expected_dtype
        or str(array.dtype) != str(record["dtype"])
        or list(array.shape) != record["shape"]
    ):
        raise ValueError(f"Final reconstruction {name} dtype/shape mismatch")
    return array


def _refinements(payload: list[dict[str, Any]]) -> tuple[PoseRefinement, ...]:
    expected = {
        "view_id",
        "side",
        "correction_matrix",
        "correspondence_count",
        "rmse_before_m",
        "rmse_after_m",
        "accepted",
        "reason",
    }
    if any(not isinstance(item, dict) or set(item) != expected for item in payload):
        raise ValueError("Stored terminal pose-refinement fields changed")
    return tuple(
        PoseRefinement(
            str(item["view_id"]),
            int(item["side"]),
            np.asarray(item["correction_matrix"], dtype=np.float64),
            int(item["correspondence_count"]),
            float("inf") if item["rmse_before_m"] is None else float(item["rmse_before_m"]),
            float("inf") if item["rmse_after_m"] is None else float(item["rmse_after_m"]),
            bool(item["accepted"]),
            str(item["reason"]),
        )
        for item in payload
    )


def write_final_fine_reconstruction(
    output_dir: str | Path,
    result: FinalFineReconstruction,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
    science_authority: ScienceAcceptanceAuthority,
) -> Path:
    """Persist a production terminal reconstruction under an exact science authority."""

    if science_authority is None:
        raise ValueError(
            "Production final reconstruction requires a science acceptance authority"
        )
    return _write_final_fine_reconstruction(
        output_dir,
        result,
        fusion_config=fusion_config,
        tsdf_config=tsdf_config,
        surface_quality_config=surface_quality_config,
        finalization_config=finalization_config,
        science_authority=science_authority,
    )


def write_unaccepted_legacy_fine_reconstruction(
    output_dir: str | Path,
    result: FinalFineReconstruction,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
) -> Path:
    """Persist schema-1 evidence for offline audit only.

    This deliberately named API cannot be used by the production finalizer or the
    production experiment writer.  It exists only so historical schema-1 assets remain
    reproducible in migration/audit tests.
    """

    return _write_final_fine_reconstruction(
        output_dir,
        result,
        fusion_config=fusion_config,
        tsdf_config=tsdf_config,
        surface_quality_config=surface_quality_config,
        finalization_config=finalization_config,
        science_authority=None,
    )


def _write_final_fine_reconstruction(
    output_dir: str | Path,
    result: FinalFineReconstruction,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
    science_authority: ScienceAcceptanceAuthority | None,
) -> Path:
    """Atomically persist one production or explicitly unaccepted legacy asset."""

    if not result.gates.passed:
        raise ValueError("Cannot persist a final reconstruction whose terminal gates failed")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Final reconstruction output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    arrays = {
        "fused_points_m": result.fused_cloud.points_m,
        "fused_normals": result.fused_cloud.normals,
        "fused_side_labels": result.fused_cloud.side_labels,
        "front_tsdf_indices": result.tsdf.front.voxel_indices,
        "front_tsdf_values": result.tsdf.front.tsdf,
        "front_tsdf_weights": result.tsdf.front.weights,
        "back_tsdf_indices": result.tsdf.back.voxel_indices,
        "back_tsdf_values": result.tsdf.back.tsdf,
        "back_tsdf_weights": result.tsdf.back.weights,
        "mesh_vertices_m": result.tsdf.mesh.vertices_m,
        "mesh_triangles": result.tsdf.mesh.triangles,
        "mesh_triangle_sides": result.tsdf.mesh.triangle_sides,
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        coverage_record = _source_record(result.coverage.root, "coverage.json")
        coverage_record["generation_id"] = result.coverage.generation_id
        reference_record = _source_record(result.coverage.reference.root, "metadata.json")
        sources = []
        for root, view in zip(
            result.source_view_roots,
            result.registered_views,
            strict=True,
        ):
            record = _source_record(root, "metadata.json")
            record["view_id"] = view.view_id
            sources.append(record)
        refinements = []
        for item in result.fused_cloud.refinements:
            refinements.append(
                {
                    "view_id": item.view_id,
                    "side": item.side,
                    "correction_matrix": item.correction_matrix.tolist(),
                    "correspondence_count": item.correspondence_count,
                    "rmse_before_m": (
                        item.rmse_before_m if np.isfinite(item.rmse_before_m) else None
                    ),
                    "rmse_after_m": (
                        item.rmse_after_m if np.isfinite(item.rmse_after_m) else None
                    ),
                    "accepted": item.accepted,
                    "reason": item.reason,
                }
            )
        payload: dict[str, Any] = {
            "schema_version": (
                FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION
                if science_authority is not None
                else _LEGACY_FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "sources": {
                "terminal_coverage": coverage_record,
                "coarse_reference": reference_record,
                "reconstructed_views": sources,
            },
            "files": {
                name: _array_record(temporary / f"{name}.npy") for name in arrays
            },
            "configuration": {
                "fusion": fusion_config.model_dump(mode="json"),
                "tsdf": tsdf_config.model_dump(mode="json"),
                "surface_quality": surface_quality_config.model_dump(mode="json"),
                "finalization": finalization_config.model_dump(mode="json"),
            },
            "fusion": {
                "center_m": result.fused_cloud.center_m.tolist(),
                "axes": result.fused_cloud.axes.tolist(),
                "median_thickness_m": result.fused_cloud.median_thickness_m,
                "refinements": refinements,
            },
            "tsdf": {
                "front_origin_m": result.tsdf.front.origin_m.tolist(),
                "back_origin_m": result.tsdf.back.origin_m.tolist(),
                "voxel_size_m": result.tsdf.front.voxel_size_m,
                "protected_truncation_distance_m": (
                    result.tsdf.protected_truncation_distance_m
                ),
                "backend": result.tsdf.backend,
                "feature_thicknesses_m": list(result.tsdf.feature_thicknesses_m),
            },
            "surface_quality": _quality_payload(result.quality),
            "terminal_gates": _gate_payload(result.gates),
        }
        if science_authority is not None:
            science_authority.assert_acceptance_asset_current()
            payload["science_acceptance_authority"] = science_authority.to_payload()
        identity_payload = {
            key: value for key, value in payload.items() if key != "created_at_utc"
        }
        payload["artifact_id"] = _canonical_sha256(identity_payload)
        (temporary / _METADATA_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        read_final_fine_reconstruction(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_final_fine_reconstruction(
    path: str | Path,
) -> StoredFinalFineReconstruction:
    """Strictly verify immutable arrays and every cited source asset."""

    root = Path(path).resolve()
    metadata_path = root / _METADATA_NAME
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", -1))
        expected_fields = {
            "schema_version",
            "created_at_utc",
            "motion_authorized",
            "sources",
            "files",
            "configuration",
            "fusion",
            "tsdf",
            "surface_quality",
            "terminal_gates",
            "artifact_id",
        }
        if schema_version == FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION:
            expected_fields.add("science_acceptance_authority")
        elif schema_version != _LEGACY_FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION:
            raise ValueError("terminal schema is invalid")
        if set(payload) != expected_fields:
            raise ValueError("terminal metadata has unexpected fields")
        if (
            payload["motion_authorized"] is not False
        ):
            raise ValueError("terminal schema or motion boundary is invalid")
        science_authority = (
            ScienceAcceptanceAuthority.from_payload(
                payload["science_acceptance_authority"]
            )
            if schema_version == FINAL_FINE_RECONSTRUCTION_SCHEMA_VERSION
            else None
        )
        if science_authority is not None:
            science_authority.assert_acceptance_asset_current()
        identity_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"created_at_utc", "artifact_id"}
        }
        if _canonical_sha256(identity_payload) != str(payload["artifact_id"]):
            raise ValueError("terminal artifact identity mismatch")
        if set(payload["files"]) != set(_ARRAY_DTYPES):
            raise ValueError("terminal array set is incomplete")
        arrays = {
            name: _load_array(root, payload["files"][name], name)
            for name in _ARRAY_DTYPES
        }

        sources = payload["sources"]
        if set(sources) != {
            "terminal_coverage",
            "coarse_reference",
            "reconstructed_views",
        }:
            raise ValueError("terminal source set is invalid")
        coverage_record = dict(sources["terminal_coverage"])
        generation_id = str(coverage_record.pop("generation_id"))
        coverage_root = _verify_source(coverage_record, label="terminal coverage")
        coverage = read_surface_coverage_generation(
            coverage_root,
            require_foreground_bound_science=True,
        )
        if coverage.generation_id != generation_id:
            raise ValueError("terminal coverage generation identity changed")
        reference_root = _verify_source(
            sources["coarse_reference"], label="coarse reference"
        )
        if coverage.reference.root != reference_root:
            raise ValueError("terminal coverage and coarse reference disagree")

        if not isinstance(sources["reconstructed_views"], list):
            raise ValueError("terminal reconstructed-view sources must be a list")
        source_roots: list[Path] = []
        registered = []
        for index, raw_record in enumerate(sources["reconstructed_views"]):
            record = dict(raw_record)
            expected_view_id = str(record.pop("view_id"))
            source_root = _verify_source(record, label=f"reconstructed view {index}")
            stored = read_reconstructed_view(source_root)
            if stored.view.source_view_id != expected_view_id:
                raise ValueError("terminal reconstructed-view identity changed")
            source_roots.append(source_root)
            registered.append(registered_cloud_view(stored.view))
        if tuple(item.view_id for item in registered) != coverage.ledger.observation_ids:
            raise ValueError("terminal source sequence differs from the coverage ledger")

        configuration = payload["configuration"]
        if set(configuration) != {
            "fusion",
            "tsdf",
            "surface_quality",
            "finalization",
        }:
            raise ValueError("terminal configuration set is invalid")
        fusion_config = MultiViewFusionConfig.model_validate(configuration["fusion"])
        tsdf_config = TSDFConfig.model_validate(configuration["tsdf"])
        quality_config = SurfaceQualityConfig.model_validate(
            configuration["surface_quality"]
        )
        finalization_config = FineFinalizationConfig.model_validate(
            configuration["finalization"]
        )
        if coverage.quality_config.model_dump(mode="json") != quality_config.model_dump(
            mode="json"
        ):
            raise ValueError("terminal quality configuration differs from coverage")

        fusion = payload["fusion"]
        if not isinstance(fusion, dict) or set(fusion) != {
            "center_m",
            "axes",
            "median_thickness_m",
            "refinements",
        }:
            raise ValueError("terminal fusion fields are invalid")
        fused = FusedBladeCloud(
            arrays["fused_points_m"],
            arrays["fused_normals"],
            arrays["fused_side_labels"],
            np.asarray(fusion["center_m"], dtype=np.float64),
            np.asarray(fusion["axes"], dtype=np.float64),
            float(fusion["median_thickness_m"]),
            _refinements(fusion["refinements"]),
        )
        tsdf_payload = payload["tsdf"]
        if not isinstance(tsdf_payload, dict) or set(tsdf_payload) != {
            "front_origin_m",
            "back_origin_m",
            "voxel_size_m",
            "protected_truncation_distance_m",
            "backend",
            "feature_thicknesses_m",
        }:
            raise ValueError("terminal TSDF fields are invalid")
        voxel_size = float(tsdf_payload["voxel_size_m"])
        truncation = float(tsdf_payload["protected_truncation_distance_m"])
        front = SparseTSDFVolume(
            1,
            np.asarray(tsdf_payload["front_origin_m"], dtype=np.float64),
            voxel_size,
            truncation,
            arrays["front_tsdf_indices"],
            arrays["front_tsdf_values"],
            arrays["front_tsdf_weights"],
        )
        back = SparseTSDFVolume(
            -1,
            np.asarray(tsdf_payload["back_origin_m"], dtype=np.float64),
            voxel_size,
            truncation,
            arrays["back_tsdf_indices"],
            arrays["back_tsdf_values"],
            arrays["back_tsdf_weights"],
        )
        mesh = TriangleMesh(
            arrays["mesh_vertices_m"],
            arrays["mesh_triangles"],
            arrays["mesh_triangle_sides"],
        )
        tsdf = BilateralTSDFResult(
            front,
            back,
            mesh,
            truncation,
            str(tsdf_payload["backend"]),
            tuple(float(value) for value in tsdf_payload["feature_thicknesses_m"]),
        )
        quality = _quality(payload["surface_quality"])
        gates = _gates(payload["terminal_gates"])
        result = FinalFineReconstruction(
            coverage,
            tuple(source_roots),
            tuple(registered),
            fused,
            tsdf,
            quality,
            gates,
        )
        return StoredFinalFineReconstruction(
            root,
            str(payload["artifact_id"]),
            _sha256(metadata_path),
            result,
            fusion_config,
            tsdf_config,
            quality_config,
            finalization_config,
            science_authority,
            payload,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid final fine reconstruction {root}: {exc}") from exc


def replay_final_fine_reconstruction(
    path: str | Path,
    *,
    expected_science_authority: ScienceAcceptanceAuthority | None = None,
) -> StoredFinalFineReconstruction:
    """Recompute fusion, TSDF, quality and gates from immutable cited sources."""

    stored = read_final_fine_reconstruction(path)
    if (
        expected_science_authority is not None
        and stored.science_authority != expected_science_authority
    ):
        raise ValueError("Final reconstruction science authority changed")
    replayed = build_final_fine_reconstruction(
        stored.result.coverage.root,
        fusion_config=stored.fusion_config,
        tsdf_config=stored.tsdf_config,
        surface_quality_config=stored.surface_quality_config,
        finalization_config=stored.finalization_config,
    )
    comparisons = (
        ("fused points", stored.result.fused_cloud.points_m, replayed.fused_cloud.points_m),
        ("fused normals", stored.result.fused_cloud.normals, replayed.fused_cloud.normals),
        ("fused sides", stored.result.fused_cloud.side_labels, replayed.fused_cloud.side_labels),
        (
            "front TSDF indices",
            stored.result.tsdf.front.voxel_indices,
            replayed.tsdf.front.voxel_indices,
        ),
        ("front TSDF values", stored.result.tsdf.front.tsdf, replayed.tsdf.front.tsdf),
        ("front TSDF weights", stored.result.tsdf.front.weights, replayed.tsdf.front.weights),
        (
            "back TSDF indices",
            stored.result.tsdf.back.voxel_indices,
            replayed.tsdf.back.voxel_indices,
        ),
        ("back TSDF values", stored.result.tsdf.back.tsdf, replayed.tsdf.back.tsdf),
        ("back TSDF weights", stored.result.tsdf.back.weights, replayed.tsdf.back.weights),
        ("mesh vertices", stored.result.tsdf.mesh.vertices_m, replayed.tsdf.mesh.vertices_m),
        ("mesh triangles", stored.result.tsdf.mesh.triangles, replayed.tsdf.mesh.triangles),
        ("mesh sides", stored.result.tsdf.mesh.triangle_sides, replayed.tsdf.mesh.triangle_sides),
    )
    for label, expected, actual in comparisons:
        if not np.array_equal(expected, actual):
            raise ValueError(f"Final fine replay changed {label}")
    stored_fused = stored.result.fused_cloud
    replayed_fused = replayed.fused_cloud
    fused_semantics = (
        np.array_equal(stored_fused.center_m, replayed_fused.center_m)
        and np.array_equal(stored_fused.axes, replayed_fused.axes)
        and stored_fused.median_thickness_m == replayed_fused.median_thickness_m
        and _refinement_payload(stored_fused.refinements)
        == _refinement_payload(replayed_fused.refinements)
    )
    if not fused_semantics:
        raise ValueError("Final fine replay changed fusion semantics")
    stored_tsdf = stored.result.tsdf
    replayed_tsdf = replayed.tsdf
    tsdf_semantics = (
        np.array_equal(stored_tsdf.front.origin_m, replayed_tsdf.front.origin_m)
        and np.array_equal(stored_tsdf.back.origin_m, replayed_tsdf.back.origin_m)
        and stored_tsdf.front.voxel_size_m == replayed_tsdf.front.voxel_size_m
        and stored_tsdf.back.voxel_size_m == replayed_tsdf.back.voxel_size_m
        and stored_tsdf.front.truncation_distance_m
        == replayed_tsdf.front.truncation_distance_m
        and stored_tsdf.back.truncation_distance_m
        == replayed_tsdf.back.truncation_distance_m
        and stored_tsdf.protected_truncation_distance_m
        == replayed_tsdf.protected_truncation_distance_m
        and stored_tsdf.backend == replayed_tsdf.backend
        and stored_tsdf.feature_thicknesses_m == replayed_tsdf.feature_thicknesses_m
    )
    if not tsdf_semantics:
        raise ValueError("Final fine replay changed TSDF semantics")
    if (
        _quality_payload(stored.result.quality) != _quality_payload(replayed.quality)
        or _gate_payload(stored.result.gates) != _gate_payload(replayed.gates)
    ):
        raise ValueError("Final fine replay changed quality or terminal gates")
    return stored


def _refinement_payload(
    refinements: tuple[PoseRefinement, ...],
) -> tuple[tuple[object, ...], ...]:
    """Canonical comparison payload for non-scalar pose-refinement records."""

    return tuple(
        (
            item.view_id,
            item.side,
            item.correction_matrix.tobytes(),
            item.correspondence_count,
            item.rmse_before_m,
            item.rmse_after_m,
            item.accepted,
            item.reason,
        )
        for item in refinements
    )
