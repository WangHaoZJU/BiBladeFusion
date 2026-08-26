"""Atomic, checksummed persistence for bilateral surface-coverage ledgers."""

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

from biblade_fusion.core.settings import CoverageConfig
from biblade_fusion.planning import BladeSide, CoverageLedger, PatchCoverage

COVERAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredCoverageLedger:
    ledger: CoverageLedger
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_coverage_ledger(
    output_dir: str | Path,
    ledger: CoverageLedger,
    *,
    source_plan: str | Path,
    source_initialization: str | Path,
    previous_ledger: str | Path | None = None,
) -> Path:
    """Write one immutable coverage state; updates must use a new output path."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Coverage output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        counts = np.stack([patch.bin_point_counts for patch in ledger.patches])
        counts_path = temporary / "bin_point_counts.npy"
        np.save(counts_path, counts, allow_pickle=False)
        payload: dict[str, Any] = {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_plan": str(Path(source_plan).resolve()),
            "source_initialization": str(Path(source_initialization).resolve()),
            "previous_ledger": (
                str(Path(previous_ledger).resolve()) if previous_ledger is not None else None
            ),
            "motion_authorized": False,
            "configuration": ledger.config.model_dump(mode="json"),
            "grid": {"rows": ledger.rows, "columns": ledger.columns},
            "observation_ids": list(ledger.observation_ids),
            "patches": [
                {
                    "patch_id": patch.patch_id,
                    "side": patch.side.value,
                    "row": patch.row,
                    "column": patch.column,
                    "observation_ids": list(patch.observation_ids),
                    "point_count": patch.point_count,
                    "occupied_fraction": patch.occupied_fraction(
                        ledger.config.minimum_points_per_bin
                    ),
                    "complete": ledger.is_complete(patch.patch_id),
                }
                for patch in ledger.patches
            ],
            "counts_file": {
                "path": counts_path.name,
                "sha256": _sha256(counts_path),
                "dtype": str(counts.dtype),
                "shape": list(counts.shape),
            },
            "summary": {
                "overall_completion_fraction": ledger.completion_fraction(),
                "front_completion_fraction": ledger.completion_fraction(BladeSide.FRONT),
                "back_completion_fraction": ledger.completion_fraction(BladeSide.BACK),
                "completed_patch_ids": list(ledger.completed_patch_ids),
            },
        }
        metadata_path = temporary / "coverage.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_coverage_ledger(path: str | Path) -> StoredCoverageLedger:
    """Verify array integrity and reconstruct a typed coverage ledger."""

    root = Path(path)
    try:
        payload = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("coverage root must be an object")
        if int(payload["schema_version"]) != COVERAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("stored coverage must explicitly forbid motion")
        record = payload["counts_file"]
        relative = Path(str(record["path"]))
        resolved_root = root.resolve()
        counts_path = (resolved_root / relative).resolve()
        if relative.is_absolute() or not counts_path.is_relative_to(resolved_root):
            raise ValueError(f"coverage counts path escapes output directory: {relative}")
        if _sha256(counts_path) != str(record["sha256"]):
            raise ValueError("coverage counts checksum mismatch")
        counts = np.load(counts_path, allow_pickle=False)
        if str(counts.dtype) != str(record["dtype"]) or list(counts.shape) != record["shape"]:
            raise ValueError("coverage counts manifest mismatch")
        patch_data = payload["patches"]
        if len(patch_data) != len(counts):
            raise ValueError("coverage patch metadata and count arrays differ")
        patches = tuple(
            PatchCoverage(
                str(item["patch_id"]),
                BladeSide(str(item["side"])),
                int(item["row"]),
                int(item["column"]),
                counts[index],
                tuple(str(value) for value in item["observation_ids"]),
            )
            for index, item in enumerate(patch_data)
        )
        grid = payload["grid"]
        ledger = CoverageLedger(
            patches,
            tuple(str(value) for value in payload["observation_ids"]),
            CoverageConfig.model_validate(payload["configuration"]),
            int(grid["rows"]),
            int(grid["columns"]),
        )
        if set(payload["summary"]["completed_patch_ids"]) != set(ledger.completed_patch_ids):
            raise ValueError("coverage summary does not match reconstructed counts")
        return StoredCoverageLedger(ledger, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coverage artifact {root}: {exc}") from exc
