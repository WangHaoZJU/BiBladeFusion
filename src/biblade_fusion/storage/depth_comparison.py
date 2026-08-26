"""Checksummed artifacts for paired native/stereo depth experiments."""

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

from biblade_fusion.core.settings import DepthComparisonConfig, PointCloudConfig
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.workflows import (
    DepthComparisonMetrics,
    PairedDepthComparison,
    compare_paired_depth,
)

DEPTH_COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredDepthComparison:
    comparison: PairedDepthComparison
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Depth-comparison source file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _array_record(path: Path) -> dict[str, Any]:
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


def _session_source_records(
    source_session: str | Path,
    view_id: str,
    sequence_index: int,
    frame_number: int,
) -> dict[str, Any]:
    reader = SessionReader(source_session)
    descriptor = reader.descriptor(view_id)
    view_root = (reader.path / descriptor.relative_path).resolve()
    metadata_path = view_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        descriptor.sequence_index != sequence_index
        or int(metadata["stereo"]["frame_number"]) != frame_number
    ):
        raise ValueError("Depth comparison does not match its source session view")
    native_filename = metadata["stereo"].get("native_depth_file")
    if native_filename is None:
        raise ValueError("Depth-comparison source view has no native depth file")
    return {
        "root": str(reader.path),
        "view_metadata": _source_record(metadata_path),
        "native_depth": _source_record(view_root / str(native_filename)),
    }


def write_depth_comparison(
    output_dir: str | Path,
    comparison: PairedDepthComparison,
    point_cloud_config: PointCloudConfig,
    comparison_config: DepthComparisonConfig,
    *,
    source_session: str | Path,
    source_stereo_inference: str | Path,
    source_blade_mask: str | Path,
) -> Path:
    """Persist one paired-depth experiment without overwriting its sources."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Depth-comparison output already exists: {output}")
    session_records = _session_source_records(
        source_session,
        comparison.source_view_id,
        comparison.source_sequence_index,
        comparison.source_frame_number,
    )
    stereo_metadata = Path(source_stereo_inference).resolve() / "metadata.json"
    stereo_payload = json.loads(stereo_metadata.read_text(encoding="utf-8"))
    stereo_source = stereo_payload["source"]
    if (
        str(stereo_source["view_id"]) != comparison.source_view_id
        or int(stereo_source["sequence_index"]) != comparison.source_sequence_index
        or int(stereo_source["frame_number"]) != comparison.source_frame_number
    ):
        raise ValueError("Depth comparison does not match its stereo inference source")
    mask_path = Path(source_blade_mask).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    arrays = {
        "native_depth_left_rectified_m": comparison.native_depth_left_rectified_m,
        "comparison_mask": comparison.comparison_mask,
        "signed_error_m": comparison.signed_error_m,
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        payload: dict[str, Any] = {
            "schema_version": DEPTH_COMPARISON_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "view_id": comparison.source_view_id,
                "sequence_index": comparison.source_sequence_index,
                "frame_number": comparison.source_frame_number,
                "session": session_records,
                "stereo_inference": {
                    "root": str(Path(source_stereo_inference).resolve()),
                    "metadata": _source_record(stereo_metadata),
                },
                "blade_mask": _source_record(mask_path),
            },
            "files": {
                name: _array_record(temporary / f"{name}.npy") for name in arrays
            },
            "metrics": asdict(comparison.metrics),
            "processing": {
                "point_cloud": point_cloud_config.model_dump(mode="json"),
                "depth_comparison": comparison_config.model_dump(mode="json"),
            },
            "interpretation": (
                "signed_error_m is FoundationStereo minus native RealSense axial depth; "
                "native RealSense is a comparison reference, not ground truth"
            ),
        }
        (temporary / "depth_comparison.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _verify_source(record: dict[str, Any]) -> None:
    path = Path(str(record["path"])).resolve()
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"depth-comparison source checksum mismatch: {path}")


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(str(record["path"]))
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"depth-comparison path escapes output: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"depth-comparison checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"depth-comparison array manifest mismatch: {relative}")
    return array


def read_depth_comparison(path: str | Path) -> StoredDepthComparison:
    """Verify source provenance, arrays, metrics, and reconstruct the experiment result."""

    root = Path(path)
    try:
        payload = json.loads(
            (root / "depth_comparison.json").read_text(encoding="utf-8")
        )
        if int(payload["schema_version"]) != DEPTH_COMPARISON_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        source = payload["source"]
        for record in (
            source["session"]["view_metadata"],
            source["session"]["native_depth"],
            source["stereo_inference"]["metadata"],
            source["blade_mask"],
        ):
            _verify_source(record)
        arrays = {
            name: _load_array(root, record) for name, record in payload["files"].items()
        }
        metric_data = payload["metrics"]
        metrics = DepthComparisonMetrics(
            blade_pixel_count=int(metric_data["blade_pixel_count"]),
            native_valid_pixel_count=int(metric_data["native_valid_pixel_count"]),
            stereo_valid_pixel_count=int(metric_data["stereo_valid_pixel_count"]),
            overlap_pixel_count=int(metric_data["overlap_pixel_count"]),
            native_coverage_fraction=float(metric_data["native_coverage_fraction"]),
            stereo_coverage_fraction=float(metric_data["stereo_coverage_fraction"]),
            overlap_fraction=float(metric_data["overlap_fraction"]),
            signed_mean_error_m=float(metric_data["signed_mean_error_m"]),
            signed_median_error_m=float(metric_data["signed_median_error_m"]),
            mean_absolute_error_m=float(metric_data["mean_absolute_error_m"]),
            root_mean_square_error_m=float(
                metric_data["root_mean_square_error_m"]
            ),
            p95_absolute_error_m=float(metric_data["p95_absolute_error_m"]),
            median_stereo_to_native_ratio=float(
                metric_data["median_stereo_to_native_ratio"]
            ),
            agreement_fractions=tuple(
                (float(item[0]), float(item[1]))
                for item in metric_data["agreement_fractions"]
            ),
        )
        comparison = PairedDepthComparison(
            str(source["view_id"]),
            int(source["sequence_index"]),
            int(source["frame_number"]),
            arrays["native_depth_left_rectified_m"],
            arrays["comparison_mask"],
            arrays["signed_error_m"],
            metrics,
        )
        bundle = SessionReader(source["session"]["root"]).load_bundle(
            comparison.source_view_id
        )
        stereo = read_stereo_inference(
            source["stereo_inference"]["root"]
        ).observation
        blade_mask = np.load(source["blade_mask"]["path"], allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Depth-comparison blade mask must be one .npy array")
        processing = payload["processing"]
        expected = compare_paired_depth(
            bundle,
            stereo,
            blade_mask,
            PointCloudConfig.model_validate(processing["point_cloud"]),
            DepthComparisonConfig.model_validate(processing["depth_comparison"]),
        )
        if not np.array_equal(comparison.comparison_mask, expected.comparison_mask):
            raise ValueError("depth-comparison mask does not match its sources")
        for name in ("native_depth_left_rectified_m", "signed_error_m"):
            if not np.allclose(
                getattr(comparison, name),
                getattr(expected, name),
                equal_nan=True,
            ):
                raise ValueError(f"depth-comparison {name} does not match its sources")
        if asdict(comparison.metrics) != asdict(expected.metrics):
            raise ValueError("depth-comparison metrics do not match its sources")
        return StoredDepthComparison(comparison, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid depth-comparison artifact {root}: {exc}") from exc
