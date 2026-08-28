"""Immutable JSON storage for offline bilateral view plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.planning import (
    BilateralViewPlan,
    BladeSide,
    CandidateMetrics,
    CandidateStatus,
    CandidateView,
    EvaluatedCandidate,
    FilteredViewPlan,
    SurfacePatch,
)
from biblade_fusion.workflows import OfflineViewPlanningResult

VIEW_PLAN_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class StoredViewPlan:
    result: OfflineViewPlanningResult
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_payload(candidate: CandidateView) -> dict[str, Any]:
    patch = candidate.patch
    return {
        "view_id": candidate.view_id,
        "side": patch.side.value,
        "row": patch.row,
        "column": patch.column,
        "target_m": patch.target_m.tolist(),
        "outward_normal": patch.outward_normal.tolist(),
        "patch_planar_extents_m": list(patch.planar_extents_m),
        "base_T_left_ir": candidate.base_t_left_ir.matrix.tolist(),
        "standoff_distance_m": candidate.standoff_distance_m,
        "footprint_m": list(candidate.footprint_m),
    }


def _evaluation_payload(item: EvaluatedCandidate) -> dict[str, Any]:
    return {
        "view_id": item.candidate.view_id,
        "status": item.status.value,
        "reasons": list(item.reasons),
        "joint_positions_rad": (
            item.joint_positions_rad.tolist() if item.joint_positions_rad is not None else None
        ),
        "metrics": {
            "look_at_cosine": item.metrics.look_at_cosine,
            "incidence_cosine": item.metrics.incidence_cosine,
            "coverage_ratio": item.metrics.coverage_ratio,
            "view_distance_m": item.metrics.view_distance_m,
            "standoff_error_m": item.metrics.standoff_error_m,
            "proxy_clearance_m": item.metrics.proxy_clearance_m,
            "geometric_score": item.metrics.geometric_score,
        },
    }


def write_view_plan(
    output_dir: str | Path,
    result: OfflineViewPlanningResult,
    planning_config: ViewPlanningConfig,
    filter_config: ViewFilterConfig,
    *,
    source_initialization: str | Path,
    source_kinematics: str | Path | None = None,
    joint_zero_offsets_rad: tuple[float, float, float, float, float, float] | None = None,
) -> Path:
    """Write an offline-only plan without overwriting an existing artifact."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"View-plan output already exists: {output}")
    geometric = result.geometric_plan
    filtered = result.filtered_plan
    has_endpoint_solutions = bool(filtered.endpoint_feasible)
    if has_endpoint_solutions and source_kinematics is None:
        raise ValueError(
            "Endpoint-feasible plans must record their controller kinematics artifact"
        )
    if has_endpoint_solutions and joint_zero_offsets_rad is None:
        raise ValueError(
            "Endpoint-feasible plans must record the applied joint-zero offsets"
        )
    offsets = None
    if joint_zero_offsets_rad is not None:
        offset_array = np.asarray(joint_zero_offsets_rad, dtype=np.float64)
        if offset_array.shape != (6,) or not np.isfinite(offset_array).all():
            raise ValueError("joint_zero_offsets_rad must be a finite six-vector")
        offsets = [float(value) for value in offset_array]
    kinematics_record = None
    if source_kinematics is not None:
        kinematics_path = Path(source_kinematics).resolve()
        if not kinematics_path.is_file():
            raise ValueError(f"Kinematics source does not exist: {kinematics_path}")
        kinematics_record = {
            "path": str(kinematics_path),
            "sha256": _sha256(kinematics_path),
            "joint_zero_offsets_rad": offsets,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    payload: dict[str, Any] = {
        "schema_version": VIEW_PLAN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_initialization": str(Path(source_initialization).resolve()),
        "source_kinematics": kinematics_record,
        "motion_authorized": False,
        "grid": {
            "rows": geometric.rows,
            "columns": geometric.columns,
            "footprint_m": list(geometric.footprint_m),
            "effective_surface_extents_m": list(geometric.effective_surface_extents_m),
        },
        "configuration": {
            "view_planning": planning_config.model_dump(mode="json"),
            "view_filter": filter_config.model_dump(mode="json"),
        },
        "candidates": [_candidate_payload(candidate) for candidate in geometric.candidates],
        "evaluations": [_evaluation_payload(item) for item in filtered.candidates],
        "duplicate_view_ids": list(filtered.duplicate_view_ids),
        "summary": {
            "geometric_candidates": len(geometric.candidates),
            "accepted_candidates": len(filtered.accepted),
            "endpoint_feasible_candidates": len(filtered.endpoint_feasible),
            "rejected_candidates": sum(
                item.status is CandidateStatus.REJECTED for item in filtered.candidates
            ),
        },
    }
    try:
        path = temporary / "view_plan.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(output)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _candidate_from_payload(payload: dict[str, Any]) -> CandidateView:
    view_id = str(payload["view_id"])
    patch = SurfacePatch(
        patch_id=view_id,
        side=BladeSide(str(payload["side"])),
        row=int(payload["row"]),
        column=int(payload["column"]),
        target_m=payload["target_m"],
        outward_normal=payload["outward_normal"],
        planar_extents_m=tuple(float(value) for value in payload["patch_planar_extents_m"]),
    )
    return CandidateView(
        view_id,
        patch,
        PoseSE3("base", f"{view_id}_left_ir", payload["base_T_left_ir"]),
        float(payload["standoff_distance_m"]),
        tuple(float(value) for value in payload["footprint_m"]),
    )


def read_view_plan(path: str | Path) -> StoredViewPlan:
    """Reconstruct the typed, explicitly non-authorized offline view plan."""

    root = Path(path)
    try:
        payload = json.loads((root / "view_plan.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("view plan root must be an object")
        schema_version = int(payload["schema_version"])
        if schema_version not in {1, 2, VIEW_PLAN_SCHEMA_VERSION}:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("stored view plan must explicitly forbid motion")
        candidates = tuple(_candidate_from_payload(item) for item in payload["candidates"])
        grid = payload["grid"]
        geometric = BilateralViewPlan(
            candidates,
            int(grid["rows"]),
            int(grid["columns"]),
            tuple(float(value) for value in grid["footprint_m"]),
            tuple(float(value) for value in grid["effective_surface_extents_m"]),
        )
        candidate_by_id = {candidate.view_id: candidate for candidate in candidates}
        evaluations = []
        for item in payload["evaluations"]:
            metrics = item["metrics"]
            evaluations.append(
                EvaluatedCandidate(
                    candidate_by_id[str(item["view_id"])],
                    CandidateStatus(str(item["status"])),
                    CandidateMetrics(
                        float(metrics["look_at_cosine"]),
                        float(metrics["incidence_cosine"]),
                        float(metrics["coverage_ratio"]),
                        float(metrics["view_distance_m"]),
                        float(metrics["standoff_error_m"]),
                        float(metrics["proxy_clearance_m"]),
                        float(metrics["geometric_score"]),
                    ),
                    tuple(str(reason) for reason in item["reasons"]),
                    item.get("joint_positions_rad"),
                )
            )
        filtered = FilteredViewPlan(
            tuple(evaluations),
            tuple(str(view_id) for view_id in payload["duplicate_view_ids"]),
        )
        kinematics_record = payload.get("source_kinematics")
        if kinematics_record is not None:
            kinematics_path = Path(str(kinematics_record["path"])).resolve()
            if _sha256(kinematics_path) != str(kinematics_record["sha256"]):
                raise ValueError("view-plan kinematics checksum mismatch")
            offsets = kinematics_record.get("joint_zero_offsets_rad")
            if offsets is not None:
                offset_array = np.asarray(offsets, dtype=np.float64)
                if offset_array.shape != (6,) or not np.isfinite(offset_array).all():
                    raise ValueError("view-plan joint-zero offsets are invalid")
        if (
            filtered.endpoint_feasible
            and kinematics_record is None
            and schema_version == VIEW_PLAN_SCHEMA_VERSION
        ):
            raise ValueError("endpoint-feasible plan lacks kinematics provenance")
        if (
            filtered.endpoint_feasible
            and schema_version == VIEW_PLAN_SCHEMA_VERSION
            and kinematics_record is not None
            and kinematics_record.get("joint_zero_offsets_rad") is None
        ):
            raise ValueError("endpoint-feasible plan lacks joint-zero-offset provenance")
        return StoredViewPlan(OfflineViewPlanningResult(geometric, filtered), payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid view-plan artifact {root}: {exc}") from exc
