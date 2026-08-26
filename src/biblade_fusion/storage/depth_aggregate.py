"""Reproducible stratified aggregate artifacts for paired depth experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from biblade_fusion.planning import BladeSide
from biblade_fusion.storage.depth_comparison import read_depth_comparison
from biblade_fusion.workflows import (
    DepthAggregateReport,
    LabeledDepthComparison,
    aggregate_depth_comparisons,
)

DEPTH_AGGREGATE_SCHEMA_VERSION = 1
DEPTH_AGGREGATE_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredDepthAggregate:
    report: DepthAggregateReport
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Depth aggregate source file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _verify(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"])).resolve()
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"depth aggregate source checksum mismatch: {path}")
    return path


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _load_manifest(
    manifest_path: Path,
) -> tuple[tuple[LabeledDepthComparison, ...], tuple[float, ...], list[dict[str, Any]]]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Depth aggregate manifest root must be a mapping")
    if int(payload["schema_version"]) != DEPTH_AGGREGATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema {payload['schema_version']}")
    edges = tuple(float(value) for value in payload["incidence_bin_edges_deg"])
    entries = payload["comparisons"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Depth aggregate manifest comparisons must be a non-empty list")
    labeled: list[LabeledDepthComparison] = []
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "artifact",
            "side",
            "incidence_angle_deg",
        }:
            raise ValueError(
                "Each comparison requires only artifact, side, and incidence_angle_deg"
            )
        supplied = Path(str(entry["artifact"]))
        artifact = (
            supplied.resolve()
            if supplied.is_absolute()
            else (manifest_path.parent / supplied).resolve()
        )
        stored = read_depth_comparison(artifact)
        side = BladeSide(str(entry["side"]))
        incidence = float(entry["incidence_angle_deg"])
        labeled.append(LabeledDepthComparison(stored.comparison, side, incidence))
        records.append(
            {
                "root": str(artifact),
                "metadata": _record(artifact / "depth_comparison.json"),
                "side": side.value,
                "incidence_angle_deg": incidence,
            }
        )
    return tuple(labeled), edges, records


def write_depth_aggregate(
    output_dir: str | Path,
    manifest: str | Path,
) -> Path:
    """Create one immutable stratified report from a versioned experiment manifest."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Depth aggregate output already exists: {output}")
    manifest_path = Path(manifest).resolve()
    labeled, edges, records = _load_manifest(manifest_path)
    report = aggregate_depth_comparisons(labeled, edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload: dict[str, Any] = {
            "schema_version": DEPTH_AGGREGATE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_manifest": _record(manifest_path),
            "comparisons": records,
            "report": _json_value(asdict(report)),
            "interpretation": (
                "view_mean metrics weight views equally; pooled metrics weight shared "
                "pixels equally; neither native nor stereo depth is ground truth"
            ),
        }
        (temporary / "depth_aggregate.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_depth_aggregate(path: str | Path) -> StoredDepthAggregate:
    """Re-read every comparison and reject a stale or edited aggregate report."""

    root = Path(path)
    try:
        payload = json.loads((root / "depth_aggregate.json").read_text(encoding="utf-8"))
        if int(payload["schema_version"]) != DEPTH_AGGREGATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        _verify(payload["source_manifest"])
        labeled = []
        for entry in payload["comparisons"]:
            _verify(entry["metadata"])
            stored = read_depth_comparison(entry["root"])
            labeled.append(
                LabeledDepthComparison(
                    stored.comparison,
                    BladeSide(str(entry["side"])),
                    float(entry["incidence_angle_deg"]),
                )
            )
        edges = tuple(float(value) for value in payload["report"]["incidence_bin_edges_deg"])
        report = aggregate_depth_comparisons(tuple(labeled), edges)
        if payload["report"] != _json_value(asdict(report)):
            raise ValueError("depth aggregate report does not match its sources")
        return StoredDepthAggregate(report, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid depth aggregate artifact {root}: {exc}") from exc
