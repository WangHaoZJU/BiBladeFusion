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
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
)
from biblade_fusion.perception.stereo import (
    RectifiedStereoCalibration,
    RectifiedStereoFrame,
    StereoResult,
)
from biblade_fusion.workflows import StereoInferenceObservation

STEREO_INFERENCE_SCHEMA_VERSION = 2
_RECTIFICATION_IMPLEMENTATION = (
    "biblade_fusion.perception.stereo.StereoRectifier:opencv_stereoRectify:v1"
)


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


def _raw_calibration_payload(
    calibration: StereoCalibrationSnapshot,
) -> dict[str, Any]:
    return {
        "left": _intrinsics_payload(calibration.left),
        "right": _intrinsics_payload(calibration.right),
        "right_T_left": calibration.right_t_left.matrix.tolist(),
    }


def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rectified_calibration_payload(
    calibration: RectifiedStereoCalibration,
) -> dict[str, Any]:
    return {
        "left": _intrinsics_payload(calibration.left),
        "right": _intrinsics_payload(calibration.right),
        "right_rectified_T_left_rectified": (
            calibration.right_rectified_t_left_rectified.matrix.tolist()
        ),
        "left_rectified_T_left_ir": (
            calibration.left_rectified_t_left_ir.matrix.tolist()
        ),
        "right_rectified_T_right_ir": (
            calibration.right_rectified_t_right_ir.matrix.tolist()
        ),
        "disparity_to_depth_q": calibration.disparity_to_depth_q.tolist(),
        "left_valid_roi": list(calibration.left_valid_roi),
        "right_valid_roi": list(calibration.right_valid_roi),
    }


def _source_integrity_record(
    source_session: Path,
    observation: StereoInferenceObservation,
) -> tuple[dict[str, Any] | None, StereoCalibrationSnapshot | None]:
    """Bind the derived result to exact raw NPY files when a session exists."""

    manifest = source_session / "manifest.json"
    if not manifest.is_file():
        return None, None
    from biblade_fusion.storage.reader import SessionReader

    reader = SessionReader(source_session)
    bundle = reader.load_bundle(observation.source_view_id)
    if (
        bundle.sequence_index != observation.source_sequence_index
        or bundle.stereo.frame_number != observation.rectified.source_frame_number
        or bundle.stereo.monotonic_time_ns
        != observation.rectified.source_monotonic_time_ns
    ):
        raise ValueError("Stereo inference does not match its declared raw session view")
    descriptor = reader.descriptor(observation.source_view_id)
    view_root = (reader.path / descriptor.relative_path).resolve()
    metadata_path = view_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stereo_metadata = metadata["stereo"]
    left_path = (view_root / str(stereo_metadata["left_file"])).resolve()
    right_path = (view_root / str(stereo_metadata["right_file"])).resolve()
    if not left_path.is_relative_to(view_root) or not right_path.is_relative_to(view_root):
        raise ValueError("Raw stereo array path escapes its source view")
    record = {
        "session_manifest_sha256": _sha256(manifest),
        "view_metadata_sha256": _sha256(metadata_path),
        "left_ir_npy_sha256": _sha256(left_path),
        "right_ir_npy_sha256": _sha256(right_path),
        "raw_calibration_content_hash": _canonical_payload_hash(
            _raw_calibration_payload(bundle.stereo.calibration)
        ),
    }
    return record, bundle.stereo.calibration


def write_stereo_inference(
    output_dir: str | Path,
    observation: StereoInferenceObservation,
    foundation_config: FoundationStereoConfig,
    rectification_config: StereoRectificationConfig,
    *,
    source_session: str | Path,
    source_stereo_calibration: str | Path | None = None,
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
    if observation.result.confidence is not None:
        arrays["confidence"] = observation.result.confidence
    try:
        for filename, array in arrays.items():
            np.save(temporary / f"{filename}.npy", array, allow_pickle=False)
        files = {
            name: _file_record(temporary / f"{name}.npy")
            for name in arrays
        }
        calibration = observation.rectified.calibration
        resolved_session = Path(source_session).resolve()
        source_integrity, raw_calibration = _source_integrity_record(
            resolved_session,
            observation,
        )
        calibration_asset = None
        if source_stereo_calibration is not None:
            from biblade_fusion.calibration import load_stereo_calibration

            calibration_path = Path(source_stereo_calibration).resolve()
            if not calibration_path.is_file():
                raise ValueError(
                    f"Stereo calibration source does not exist: {calibration_path}"
                )
            loaded = load_stereo_calibration(calibration_path)
            if raw_calibration is None:
                raise ValueError(
                    "A bound stereo calibration requires a readable raw source session"
                )
            if _raw_calibration_payload(loaded) != _raw_calibration_payload(
                raw_calibration
            ):
                raise ValueError(
                    "Bound stereo calibration asset differs from the raw session snapshot"
                )
            calibration_asset = {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
            }
        metadata: dict[str, Any] = {
            "schema_version": STEREO_INFERENCE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "session": str(resolved_session),
                "view_id": observation.source_view_id,
                "sequence_index": observation.source_sequence_index,
                "frame_number": observation.rectified.source_frame_number,
                "monotonic_time_ns": observation.rectified.source_monotonic_time_ns,
                "raw_session_integrity": source_integrity,
                "stereo_calibration_asset": calibration_asset,
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
                "rectification_implementation": _RECTIFICATION_IMPLEMENTATION,
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
        schema_version = int(metadata["schema_version"])
        if schema_version not in {1, STEREO_INFERENCE_SCHEMA_VERSION}:
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
            arrays.get("confidence"),
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


def verify_stereo_inference_source(
    stored: StoredStereoInference,
    *,
    expected_session: str | Path | None = None,
) -> Any:
    """Reproduce rectification from the exact bound raw session and calibration.

    The regular reader validates the derived artifact itself.  This stricter
    verifier additionally follows every external source binding and is required
    before a stereo result may contribute FREE-space evidence.
    """

    try:
        metadata = stored.metadata
        if int(metadata["schema_version"]) != STEREO_INFERENCE_SCHEMA_VERSION:
            raise ValueError(
                "safety source verification requires schema-2 stereo inference"
            )
        source = metadata["source"]
        session_root = Path(str(source["session"])).resolve()
        if expected_session is not None and session_root != Path(
            expected_session
        ).resolve():
            raise ValueError("stereo source session differs from the expected session")
        expected_integrity = source.get("raw_session_integrity")
        if not isinstance(expected_integrity, dict):
            raise ValueError("stereo artifact lacks raw-session integrity evidence")
        actual_integrity, raw_calibration = _source_integrity_record(
            session_root,
            stored.observation,
        )
        if actual_integrity != expected_integrity or raw_calibration is None:
            raise ValueError("stereo raw-session integrity evidence does not reproduce")

        calibration_asset = source.get("stereo_calibration_asset")
        if not isinstance(calibration_asset, dict):
            raise ValueError("stereo artifact lacks a bound calibration asset")
        calibration_path = Path(str(calibration_asset["path"])).resolve()
        if (
            not calibration_path.is_file()
            or _sha256(calibration_path) != str(calibration_asset["sha256"])
        ):
            raise ValueError("bound stereo calibration checksum mismatch")
        from biblade_fusion.calibration import load_stereo_calibration
        from biblade_fusion.perception.stereo import StereoRectifier
        from biblade_fusion.storage.reader import SessionReader

        calibrated = load_stereo_calibration(calibration_path)
        if _raw_calibration_payload(calibrated) != _raw_calibration_payload(
            raw_calibration
        ):
            raise ValueError(
                "bound stereo calibration differs from the raw-session snapshot"
            )
        processing = metadata["processing"]
        if processing.get("rectification_implementation") != (
            _RECTIFICATION_IMPLEMENTATION
        ):
            raise ValueError("stereo rectification implementation identity is invalid")
        rectification_config = StereoRectificationConfig.model_validate(
            processing["rectification"]
        )
        inference = metadata["inference"]
        if (
            inference.get("backend") != "foundation_stereo"
            or inference.get("runtime") != "official_nvidia_foundation_stereo"
        ):
            raise ValueError(
                "safety stereo source requires the official FoundationStereo runtime"
            )
        runtime_sources = (
            (
                Path(str(inference["repository_path"])).resolve()
                / "core/foundation_stereo.py",
                "source_sha256",
            ),
            (Path(str(inference["checkpoint_path"])).resolve(), "checkpoint_sha256"),
            (
                Path(str(inference["model_config_path"])).resolve(),
                "model_config_sha256",
            ),
        )
        for runtime_path, hash_key in runtime_sources:
            if (
                not runtime_path.is_file()
                or _sha256(runtime_path) != str(inference[hash_key])
            ):
                raise ValueError(
                    f"FoundationStereo runtime source checksum mismatch: {runtime_path}"
                )
        bundle = SessionReader(session_root).load_bundle(str(source["view_id"]))
        reproduced = StereoRectifier(
            bundle.stereo.calibration,
            rectification_config,
        ).rectify(bundle.stereo)
        observation = stored.observation
        if not np.array_equal(reproduced.left_ir, observation.rectified.left_ir):
            raise ValueError("left rectified image does not reproduce from raw input")
        if not np.array_equal(reproduced.right_ir, observation.rectified.right_ir):
            raise ValueError("right rectified image does not reproduce from raw input")
        if _rectified_calibration_payload(
            reproduced.calibration
        ) != _rectified_calibration_payload(observation.rectified.calibration):
            raise ValueError(
                "rectified calibration/Q does not reproduce from the bound calibration"
            )
        return bundle
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid stereo semantic source chain: {exc}") from exc
