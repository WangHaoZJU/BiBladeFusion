"""Immutable, non-executable plans reduced by measured bilateral coverage."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from biblade_fusion.planning import CoverageDrivenViewPlan, select_uncovered_candidates
from biblade_fusion.storage.coverage import read_coverage_ledger
from biblade_fusion.storage.view_plan import read_view_plan

COVERAGE_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredCoverageDrivenPlan:
    plan: CoverageDrivenViewPlan
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: str | Path, filename: str) -> dict[str, str]:
    resolved_root = Path(root).resolve()
    source_file = resolved_root / filename
    if not source_file.is_file():
        raise ValueError(f"Required source artifact file does not exist: {source_file}")
    return {
        "root": str(resolved_root),
        "file": filename,
        "sha256": _sha256(source_file),
    }


def _resolve_source(record: dict[str, Any]) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    source_file = (root / relative).resolve()
    if relative.is_absolute() or not source_file.is_relative_to(root):
        raise ValueError(f"coverage-plan source path escapes artifact: {relative}")
    if _sha256(source_file) != str(record["sha256"]):
        raise ValueError(f"coverage-plan source checksum mismatch: {source_file}")
    return root


def _derive_plan(source_plan: Path, source_coverage: Path) -> CoverageDrivenViewPlan:
    stored_view_plan = read_view_plan(source_plan)
    stored_coverage = read_coverage_ledger(source_coverage)
    ledger_plan = Path(str(stored_coverage.metadata["source_plan"])).resolve()
    if ledger_plan != source_plan.resolve():
        raise ValueError("Coverage ledger does not belong to the source view plan")
    return select_uncovered_candidates(
        stored_view_plan.result.filtered_plan,
        stored_coverage.ledger,
    )


def write_coverage_driven_plan(
    output_dir: str | Path,
    *,
    source_plan: str | Path,
    source_coverage: str | Path,
) -> Path:
    """Derive and persist the next offline view set from immutable source artifacts."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Coverage-plan output already exists: {output}")
    view_plan_root = Path(source_plan).resolve()
    coverage_root = Path(source_coverage).resolve()
    reduced = _derive_plan(view_plan_root, coverage_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload: dict[str, Any] = {
            "schema_version": COVERAGE_PLAN_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "sources": {
                "view_plan": _source_record(view_plan_root, "view_plan.json"),
                "coverage": _source_record(coverage_root, "coverage.json"),
            },
            "completed_patch_ids": list(reduced.completed_patch_ids),
            "remaining_view_ids": [
                item.candidate.view_id for item in reduced.remaining
            ],
            "blocked_patch_ids": list(reduced.blocked_patch_ids),
            "summary": {
                "completed_patches": len(reduced.completed_patch_ids),
                "remaining_views": len(reduced.remaining),
                "blocked_patches": len(reduced.blocked_patch_ids),
            },
        }
        (temporary / "coverage_plan.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_coverage_driven_plan(path: str | Path) -> StoredCoverageDrivenPlan:
    """Validate provenance and reconstruct a coverage-reduced offline plan."""

    root = Path(path)
    try:
        payload = json.loads((root / "coverage_plan.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("coverage-driven plan root must be an object")
        if int(payload["schema_version"]) != COVERAGE_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("stored coverage-driven plan must explicitly forbid motion")
        sources = payload["sources"]
        view_plan_root = _resolve_source(sources["view_plan"])
        coverage_root = _resolve_source(sources["coverage"])
        reduced = _derive_plan(view_plan_root, coverage_root)
        expected = {
            "completed_patch_ids": list(reduced.completed_patch_ids),
            "remaining_view_ids": [
                item.candidate.view_id for item in reduced.remaining
            ],
            "blocked_patch_ids": list(reduced.blocked_patch_ids),
        }
        for key, values in expected.items():
            if payload[key] != values:
                raise ValueError(f"coverage-driven plan {key} does not match its sources")
        summary = payload["summary"]
        if summary != {
            "completed_patches": len(reduced.completed_patch_ids),
            "remaining_views": len(reduced.remaining),
            "blocked_patches": len(reduced.blocked_patch_ids),
        }:
            raise ValueError("coverage-driven plan summary does not match its sources")
        return StoredCoverageDrivenPlan(reduced, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coverage-driven plan artifact {root}: {exc}") from exc
