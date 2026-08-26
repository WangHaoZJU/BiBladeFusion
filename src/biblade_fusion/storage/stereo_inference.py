"""Atomic, checksummed persistence for calibrated stereo inference results."""

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
from biblade_fusion.core.settings import FoundationStereoConfig, StereoRectificationConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.stereo import (
    RectifiedStereoCalibration,
    RectifiedStereoFrame,
    StereoResult,
)
from biblade_fusion.workflows import StereoInferenceObservation

STEREO_INFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredStereoInference:
    observation: StereoInferenceObservation
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion_model": intrinsics.distortion_model,
        "distortion_coefficients": list(intrinsics.distortion_coefficients),
    }


def _intrinsics_from_payload(payload: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        int(payload["width"]),
        int(payload["height"]),
        float(payload["fx"]),
        float(payload["fy"]),
        float(payload["cx"]),
        float(payload["cy"]),
        str(payload["distortion_model"]),
        tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _file_record(path: Path) -> dict[str, Any]:
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


def write_stereo_inference(
    output_dir: str | Path,
    observation: StereoInferenceObservation,
    foundation_config: FoundationStereoConfig,
    rectification_config: StereoRectificationConfig,
    *,
    source_session: str | Path,
) -> Path:
    """Write a derived stereo artifact once, without modifying its raw session."""

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"Stereo inference output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    arrays = {
        "left_rectified": observation.rectified.left_ir,
        "right_rectified": observation.rectified.right_ir,
        "disparity_px": observation.result.disparity_px,
        "valid_mask": observation.result.valid_mask,
        "depth_m": observation.depth_m,
        "disparity_to_depth_q": observation.rectified.calibration.disparity_to_depth_q,
    }
    try:
        for filename, array in arrays.items():
            np.save(temporary / f"{filename}.npy", array, allow_pickle=False)
        files = {
            name: _file_record(temporary / f"{name}.npy")
            for name in arrays
        }
        calibration = observation.rectified.calibration
        metadata: dict[str, Any] = {
            "schema_version": STEREO_INFERENCE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "session": str(Path(source_session).resolve()),
                "view_id": observation.source_view_id,
                "sequence_index": observation.source_sequence_index,
                "frame_number": observation.rectified.source_frame_number,
                "monotonic_time_ns": observation.rectified.source_monotonic_time_ns,
            },
            "files": files,
            "calibration": {
                "left": _intrinsics_payload(calibration.left),
                "right": _intrinsics_payload(calibration.right),
                "right_rectified_T_left_rectified": (
                    calibration.right_rectified_t_left_rectified.matrix.tolist()
                ),
                "left_rectified_T_left_ir": calibration.left_rectified_t_left_ir.matrix.tolist(),
                "right_rectified_T_right_ir": (
                    calibration.right_rectified_t_right_ir.matrix.tolist()
                ),
                "left_valid_roi": list(calibration.left_valid_roi),
                "right_valid_roi": list(calibration.right_valid_roi),
                "baseline_m": calibration.baseline_m,
            },
            "inference": observation.result.metadata,
            "processing": {
                "foundation_stereo": foundation_config.model_dump(mode="json"),
                "rectification": rectification_config.model_dump(mode="json"),
            },
        }
        _atomic_json(temporary / "metadata.json", metadata)
        temporary.replace(output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(str(record["path"]))
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"artifact path escapes output directory: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"artifact checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"artifact array manifest mismatch: {relative}")
    return array


def read_stereo_inference(path: str | Path) -> StoredStereoInference:
    """Validate checksums and reconstruct an immutable stereo observation."""

    root = Path(path)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata["schema_version"]) != STEREO_INFERENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {metadata['schema_version']}")
        files = metadata["files"]
        arrays = {name: _load_array(root, record) for name, record in files.items()}
        calibration_data = metadata["calibration"]
        calibration = RectifiedStereoCalibration(
            _intrinsics_from_payload(calibration_data["left"]),
            _intrinsics_from_payload(calibration_data["right"]),
            PoseSE3(
                "right_rectified",
                "left_rectified",
                calibration_data["right_rectified_T_left_rectified"],
            ),
            PoseSE3(
                "left_rectified",
                "left_ir",
                calibration_data["left_rectified_T_left_ir"],
            ),
            PoseSE3(
                "right_rectified",
                "right_ir",
                calibration_data["right_rectified_T_right_ir"],
            ),
            arrays["disparity_to_depth_q"],
            tuple(int(value) for value in calibration_data["left_valid_roi"]),
            tuple(int(value) for value in calibration_data["right_valid_roi"]),
        )
        source = metadata["source"]
        rectified = RectifiedStereoFrame(
            arrays["left_rectified"],
            arrays["right_rectified"],
            calibration,
            int(source["monotonic_time_ns"]),
            int(source["frame_number"]),
        )
        result = StereoResult(
            arrays["disparity_px"],
            arrays["valid_mask"],
            metadata=metadata["inference"],
        )
        observation = StereoInferenceObservation(
            str(source["view_id"]),
            int(source["sequence_index"]),
            rectified,
            result,
            arrays["depth_m"],
        )
        return StoredStereoInference(observation, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid stereo inference artifact {root}: {exc}") from exc
