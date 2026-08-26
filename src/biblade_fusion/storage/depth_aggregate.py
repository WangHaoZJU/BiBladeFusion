"""Reproducible stratified aggregate artifacts for paired depth experiments."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from biblade_fusion.core.settings import DepthComparisonConfig
from biblade_fusion.planning import BladeSide
from biblade_fusion.storage.depth_comparison import read_depth_comparison
from biblade_fusion.storage.initialization import read_initialization
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.workflows import (
    DepthAggregateReport,
    LabeledDepthComparison,
    aggregate_depth_comparisons,
    classify_depth_view_geometry,
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
    labeling = payload.get("labeling")
    if labeling is not None:
        if labeling.get("method") != "achieved_pose_against_fixed_proxy_v1":
            raise ValueError("Unsupported depth manifest labeling method")
        initialization_metadata = (
            Path(str(labeling["source_initialization"])).resolve() / "metadata.json"
        )
        if _sha256(initialization_metadata) != str(
            labeling["source_initialization_metadata_sha256"]
        ):
            raise ValueError("Depth manifest initialization checksum mismatch")
    edges = tuple(float(value) for value in payload["incidence_bin_edges_deg"])
    entries = payload["comparisons"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Depth aggregate manifest comparisons must be a non-empty list")
    labeled: list[LabeledDepthComparison] = []
    records: list[dict[str, Any]] = []
    for entry in entries:
        required = {"artifact", "side", "incidence_angle_deg"}
        optional = {"camera_side_offset_m", "incidence_cosine"}
        if (
            not isinstance(entry, dict)
            or not required.issubset(entry)
            or not set(entry).issubset(required | optional)
        ):
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
        if bool(optional & set(entry)) != optional.issubset(entry):
            raise ValueError("Depth manifest geometry evidence must be complete")
        if optional.issubset(entry):
            offset = float(entry["camera_side_offset_m"])
            cosine = float(entry["incidence_cosine"])
            if (offset > 0.0) != (side is BladeSide.FRONT):
                raise ValueError("Camera side offset sign does not match blade side")
            if abs(cosine - math.cos(math.radians(incidence))) > 1e-9:
                raise ValueError("Incidence cosine does not match incidence angle")
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


def write_depth_aggregate_manifest(
    output_path: str | Path,
    comparisons: tuple[str | Path, ...],
    initialization: str | Path,
    config: DepthComparisonConfig,
) -> Path:
    """Label comparison artifacts from achieved poses and a fixed proxy."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Depth aggregate manifest already exists: {output}")
    if not comparisons:
        raise ValueError("At least one depth comparison artifact is required")
    initialization_path = Path(initialization).resolve()
    stored_initialization = read_initialization(initialization_path)
    entries = []
    for comparison_path in comparisons:
        artifact = Path(comparison_path).resolve()
        stored = read_depth_comparison(artifact)
        source = stored.metadata["source"]
        bundle = SessionReader(source["session"]["root"]).load_bundle(
            stored.comparison.source_view_id
        )
        stereo = read_stereo_inference(source["stereo_inference"]["root"]).observation
        geometry = classify_depth_view_geometry(
            bundle,
            stereo,
            stored_initialization.observation.proxy,
            stored_initialization.hand_eye,
            config.minimum_camera_side_offset_m,
        )
        entries.append(
            {
                "artifact": str(artifact),
                "side": geometry.side.value,
                "incidence_angle_deg": geometry.incidence_angle_deg,
                "camera_side_offset_m": geometry.camera_side_offset_m,
                "incidence_cosine": geometry.incidence_cosine,
            }
        )
    initialization_metadata = initialization_path / "metadata.json"
    payload = {
        "schema_version": DEPTH_AGGREGATE_MANIFEST_SCHEMA_VERSION,
        "labeling": {
            "method": "achieved_pose_against_fixed_proxy_v1",
            "source_initialization": str(initialization_path),
            "source_initialization_metadata_sha256": _sha256(initialization_metadata),
            "minimum_camera_side_offset_m": config.minimum_camera_side_offset_m,
        },
        "incidence_bin_edges_deg": list(config.incidence_bin_edges_deg),
        "comparisons": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


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
