"""Immutable, source-bound blade-foreground mask assets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import BladeForegroundConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.blade_foreground import (
    BladeForegroundDiagnostics,
    BladeForegroundMaskResult,
    reference_guided_blade_mask,
)
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash

BLADE_FOREGROUND_SCHEMA_VERSION = 1

_ARTIFACT_KIND = "biblade_fusion.blade_foreground_mask"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARRAY_NAMES = (
    "mask",
    "reference_depth_m",
    "target_reference_depth_m",
    "eligible_mask",
)


@dataclass(frozen=True, slots=True)
class StoredBladeForegroundMask:
    """One fully integrity-checked foreground decision and its provenance."""

    root: Path
    result: BladeForegroundMaskResult
    metadata: dict[str, Any]


def write_blade_foreground_mask(
    output_dir: str | Path,
    result: BladeForegroundMaskResult,
    *,
    view_id: str,
    sequence_index: int,
    frame_number: int,
    base_t_left_rectified: PoseSE3,
    intrinsics: CameraIntrinsics,
    source_session: str | Path,
    source_stereo_inference: str | Path,
    source_occupancy_mapping: str | Path,
    reference_coarse_model: str | Path,
    source_integration_valid_mask_hash: str,
    target_patch_id: str,
) -> Path:
    """Atomically persist a mask and every source needed to audit its decision."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Blade-foreground output already exists: {output}")
    if not view_id:
        raise ValueError("Blade-foreground view_id must be non-empty")
    if sequence_index < 0 or frame_number < 0:
        raise ValueError("Blade-foreground sequence and frame numbers must be non-negative")
    if not target_patch_id or target_patch_id != result.diagnostics.target_patch_id:
        raise ValueError("Blade-foreground target patch identity is inconsistent")
    if (base_t_left_rectified.parent_frame, base_t_left_rectified.child_frame) != (
        "base",
        "left_rectified",
    ):
        raise ValueError("Blade-foreground pose must be base_T_left_rectified")
    _sha256_digest(
        source_integration_valid_mask_hash,
        label="source integration-valid-mask content hash",
    )
    arrays = _result_arrays(result)
    _validate_result(result, arrays, intrinsics=intrinsics)

    sources = {
        "session": _directory_source_record(Path(source_session), "manifest.json"),
        "stereo_inference": _directory_source_record(
            Path(source_stereo_inference), "metadata.json"
        ),
        "occupancy_mapping": _directory_source_record(
            Path(source_occupancy_mapping), "metadata.json"
        ),
        "reference_coarse_model": _directory_source_record(
            Path(reference_coarse_model), "metadata.json"
        ),
    }
    _validate_source_chain(
        sources,
        view_id=view_id,
        sequence_index=sequence_index,
        frame_number=frame_number,
        base_t_left_rectified=base_t_left_rectified,
        intrinsics=intrinsics,
        integration_valid_mask_hash=source_integration_valid_mask_hash,
        eligible_mask=arrays["eligible_mask"],
        target_patch_id=target_patch_id,
    )
    replayed = _replay_source_result(
        sources,
        view_id=view_id,
        sequence_index=sequence_index,
        frame_number=frame_number,
        base_t_left_rectified=base_t_left_rectified,
        intrinsics=intrinsics,
        eligible_mask=arrays["eligible_mask"],
        target_patch_id=target_patch_id,
        config=result.config,
    )
    _assert_same_result(result, replayed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        metadata: dict[str, Any] = {
            "schema_version": BLADE_FOREGROUND_SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "identity": {
                "view_id": view_id,
                "sequence_index": sequence_index,
                "frame_number": frame_number,
                "target_patch_id": target_patch_id,
            },
            "camera": {
                "base_T_left_rectified": base_t_left_rectified.matrix.tolist(),
                "intrinsics": _intrinsics_payload(intrinsics),
            },
            "processing": {
                "algorithm": result.algorithm,
                "config": _object_payload(result.config),
                "policy_sha256": result.policy_sha256,
            },
            "diagnostics": _object_payload(result.diagnostics),
            "files": {name: _array_record(temporary / f"{name}.npy") for name in _ARRAY_NAMES},
            "sources": sources,
            "source_integration_valid_mask_content_hash": (source_integration_valid_mask_hash),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_blade_foreground_mask(path: str | Path) -> StoredBladeForegroundMask:
    """Read a mask only after checking arrays, invariants and all bound sources."""

    root = Path(path).resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata["schema_version"]) != BLADE_FOREGROUND_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {metadata['schema_version']}")
        if metadata.get("artifact_kind") != _ARTIFACT_KIND:
            raise ValueError("unexpected blade-foreground artifact kind")
        if metadata.get("motion_authorized") is not False:
            raise ValueError("blade-foreground artifact must explicitly forbid motion")

        files = _mapping(metadata, "files")
        if set(files) != set(_ARRAY_NAMES):
            raise ValueError("blade-foreground file set is incomplete or unexpected")
        arrays = {name: _load_array(root, _mapping(files, name)) for name in _ARRAY_NAMES}
        processing = _mapping(metadata, "processing")
        config = _config_from_payload(_mapping(processing, "config"))
        diagnostics = BladeForegroundDiagnostics(**dict(_mapping(metadata, "diagnostics")))
        result = BladeForegroundMaskResult(
            mask=arrays["mask"],
            reference_depth_m=arrays["reference_depth_m"],
            target_reference_depth_m=arrays["target_reference_depth_m"],
            eligible_mask=arrays["eligible_mask"],
            diagnostics=diagnostics,
            config=config,
            algorithm=str(processing["algorithm"]),
            policy_sha256=str(processing["policy_sha256"]),
        )
        camera = _mapping(metadata, "camera")
        intrinsics = _intrinsics_from_payload(_mapping(camera, "intrinsics"))
        _validate_result(result, arrays, intrinsics=intrinsics)
        identity = _mapping(metadata, "identity")
        if str(identity["target_patch_id"]) != diagnostics.target_patch_id:
            raise ValueError("blade-foreground target patch identity changed")
        pose = PoseSE3("base", "left_rectified", camera["base_T_left_rectified"])
        integration_hash = str(metadata["source_integration_valid_mask_content_hash"])
        _sha256_digest(integration_hash, label="integration-valid-mask content hash")
        sources = _mapping(metadata, "sources")
        if set(sources) != {
            "session",
            "stereo_inference",
            "occupancy_mapping",
            "reference_coarse_model",
        }:
            raise ValueError("blade-foreground source set is incomplete or unexpected")
        _validate_source_chain(
            sources,
            view_id=str(identity["view_id"]),
            sequence_index=int(identity["sequence_index"]),
            frame_number=int(identity["frame_number"]),
            base_t_left_rectified=pose,
            intrinsics=intrinsics,
            integration_valid_mask_hash=integration_hash,
            eligible_mask=result.eligible_mask,
            target_patch_id=diagnostics.target_patch_id,
        )
        replayed = _replay_source_result(
            sources,
            view_id=str(identity["view_id"]),
            sequence_index=int(identity["sequence_index"]),
            frame_number=int(identity["frame_number"]),
            base_t_left_rectified=pose,
            intrinsics=intrinsics,
            eligible_mask=result.eligible_mask,
            target_patch_id=diagnostics.target_patch_id,
            config=config,
        )
        _assert_same_result(result, replayed)
        return StoredBladeForegroundMask(root, result, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid blade-foreground artifact {root}: {exc}") from exc


def _result_arrays(result: BladeForegroundMaskResult) -> dict[str, np.ndarray]:
    return {
        "mask": np.asarray(result.mask),
        "reference_depth_m": np.asarray(result.reference_depth_m),
        "target_reference_depth_m": np.asarray(result.target_reference_depth_m),
        "eligible_mask": np.asarray(result.eligible_mask),
    }


def _validate_result(
    result: BladeForegroundMaskResult,
    arrays: Mapping[str, np.ndarray],
    *,
    intrinsics: CameraIntrinsics,
) -> None:
    shape = (intrinsics.height, intrinsics.width)
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("blade-foreground arrays do not match camera image shape")
    if arrays["mask"].dtype != np.bool_ or arrays["eligible_mask"].dtype != np.bool_:
        raise ValueError("blade-foreground mask arrays must have bool dtype")
    for name in ("reference_depth_m", "target_reference_depth_m"):
        if arrays[name].dtype != np.float64:
            raise ValueError(f"blade-foreground {name} must have float64 dtype")
        finite = np.isfinite(arrays[name])
        if np.any(arrays[name][finite] <= 0.0):
            raise ValueError(f"blade-foreground {name} contains non-positive depth")
    if np.any(arrays["mask"] & ~arrays["eligible_mask"]):
        raise ValueError("blade-foreground mask must be a subset of eligible_mask")
    diagnostics = result.diagnostics
    if (
        not np.isfinite(diagnostics.target_incidence_cosine)
        or diagnostics.target_incidence_cosine < result.config.minimum_target_incidence_cosine
        or diagnostics.target_incidence_cosine > 1.0 + 1e-12
    ):
        raise ValueError("blade-foreground target incidence violates its policy")
    expected = {
        "image_pixel_count": int(np.prod(shape)),
        "eligible_pixel_count": int(np.count_nonzero(arrays["eligible_mask"])),
        "reference_pixel_count": int(np.count_nonzero(np.isfinite(arrays["reference_depth_m"]))),
        "eligible_reference_pixel_count": int(
            np.count_nonzero(arrays["eligible_mask"] & np.isfinite(arrays["reference_depth_m"]))
        ),
        "target_reference_pixel_count": int(
            np.count_nonzero(np.isfinite(arrays["target_reference_depth_m"]))
        ),
        "eligible_target_reference_pixel_count": int(
            np.count_nonzero(
                arrays["eligible_mask"] & np.isfinite(arrays["target_reference_depth_m"])
            )
        ),
        "mask_pixel_count": int(np.count_nonzero(arrays["mask"])),
    }
    for field, value in expected.items():
        if int(getattr(diagnostics, field)) != value:
            raise ValueError(f"blade-foreground diagnostic {field} does not reproduce")
    if (
        diagnostics.reference_pixel_count < result.config.minimum_reference_pixels
        or diagnostics.eligible_reference_pixel_count < result.config.minimum_reference_pixels
        or diagnostics.target_reference_pixel_count < result.config.minimum_target_reference_pixels
        or diagnostics.eligible_target_reference_pixel_count
        < result.config.minimum_target_reference_pixels
        or diagnostics.mask_pixel_count < result.config.minimum_mask_pixels
        or diagnostics.target_mask_pixel_count < result.config.minimum_target_mask_pixels
    ):
        raise ValueError("blade-foreground support counts violate their policy")
    if (
        expected["eligible_reference_pixel_count"] <= 0
        or expected["eligible_target_reference_pixel_count"] <= 0
    ):
        raise ValueError("blade-foreground eligible reference support must be non-empty")
    if not 0 <= diagnostics.valid_eligible_depth_pixel_count <= expected["eligible_pixel_count"]:
        raise ValueError("blade-foreground valid eligible-depth count is outside its bounds")
    if diagnostics.mask_pixel_count > diagnostics.valid_eligible_depth_pixel_count:
        raise ValueError("blade-foreground mask exceeds valid eligible-depth support")
    if (
        not 0
        <= diagnostics.target_mask_pixel_count
        <= min(
            diagnostics.mask_pixel_count,
            diagnostics.eligible_target_reference_pixel_count,
        )
    ):
        raise ValueError("blade-foreground target mask count is outside its bounds")
    ratios = {
        "mask_fraction": diagnostics.mask_pixel_count / diagnostics.image_pixel_count,
        "reference_match_fraction": (
            diagnostics.mask_pixel_count / diagnostics.eligible_reference_pixel_count
        ),
        "target_match_fraction": (
            diagnostics.target_mask_pixel_count / diagnostics.eligible_target_reference_pixel_count
        ),
    }
    for field, expected_ratio in ratios.items():
        actual_ratio = float(getattr(diagnostics, field))
        if not np.isfinite(actual_ratio) or actual_ratio != expected_ratio:
            raise ValueError(f"blade-foreground diagnostic {field} does not reproduce")
    if (
        diagnostics.reference_match_fraction < result.config.minimum_reference_match_fraction
        or diagnostics.target_match_fraction < result.config.minimum_target_match_fraction
        or not result.config.minimum_mask_fraction
        <= diagnostics.mask_fraction
        <= result.config.maximum_mask_fraction
    ):
        raise ValueError("blade-foreground match or area ratios violate their policy")
    if not result.algorithm:
        raise ValueError("blade-foreground algorithm identity must be non-empty")
    if not result.config.enabled or result.config.method != "reference_projected":
        raise ValueError("blade-foreground configuration is not active reference projection")
    _sha256_digest(result.policy_sha256, label="blade-foreground policy SHA-256")
    policy_payload = {
        "algorithm": result.algorithm,
        "configuration": _object_payload(result.config),
    }
    canonical_policy = json.dumps(
        policy_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical_policy).hexdigest() != result.policy_sha256:
        raise ValueError("blade-foreground policy SHA-256 does not reproduce")


def _validate_source_chain(
    sources: Mapping[str, Any],
    *,
    view_id: str,
    sequence_index: int,
    frame_number: int,
    base_t_left_rectified: PoseSE3,
    intrinsics: CameraIntrinsics,
    integration_valid_mask_hash: str,
    eligible_mask: np.ndarray,
    target_patch_id: str,
) -> None:
    session_root = _verify_directory_source_record(
        _mapping(sources, "session"), "manifest.json", label="session"
    )
    stereo_root = _verify_directory_source_record(
        _mapping(sources, "stereo_inference"),
        "metadata.json",
        label="stereo inference",
    )
    occupancy_root = _verify_directory_source_record(
        _mapping(sources, "occupancy_mapping"),
        "metadata.json",
        label="occupancy mapping",
    )
    _verify_directory_source_record(
        _mapping(sources, "reference_coarse_model"),
        "metadata.json",
        label="reference coarse model",
    )

    stereo = _read_json(stereo_root / "metadata.json")
    if int(stereo["schema_version"]) != 2:
        raise ValueError("blade-foreground requires schema-2 stereo inference")
    stereo_source = _mapping(stereo, "source")
    if (
        str(stereo_source["view_id"]) != view_id
        or int(stereo_source["sequence_index"]) != sequence_index
        or int(stereo_source["frame_number"]) != frame_number
    ):
        raise ValueError("blade-foreground identity differs from stereo inference")
    stereo_session = Path(str(stereo_source["session"])).resolve()
    if stereo_session != session_root:
        raise ValueError("blade-foreground session differs from stereo source")
    calibration = _mapping(stereo, "calibration")
    if _intrinsics_from_payload(_mapping(calibration, "left")) != intrinsics:
        raise ValueError("blade-foreground intrinsics differ from stereo inference")

    session = _read_json(session_root / "manifest.json")
    matching_views = [
        item
        for item in session["views"]
        if str(item["view_id"]) == view_id and int(item["sequence_index"]) == sequence_index
    ]
    if len(matching_views) != 1:
        raise ValueError("blade-foreground source session view is not unique")
    view_relative = Path(str(matching_views[0]["path"]))
    view_metadata_path = _contained(session_root, view_relative / "metadata.json")
    view_metadata = _read_json(view_metadata_path)
    if int(_mapping(view_metadata, "stereo")["frame_number"]) != frame_number:
        raise ValueError("blade-foreground frame number differs from raw session")

    occupancy = _read_json(occupancy_root / "metadata.json")
    if (
        int(occupancy["schema_version"]) != 6
        or occupancy.get("artifact_kind") != "biblade_fusion.occupancy_mapping"
        or occupancy.get("motion_authorized") is not False
    ):
        raise ValueError("blade-foreground requires a schema-6 occupancy mapping")
    matching_frames = []
    for frame in occupancy["frames"]:
        evidence = _mapping(frame, "evidence")
        if (
            str(evidence["source_view_id"]) == view_id
            and int(evidence["source_sequence_index"]) == sequence_index
            and int(evidence["frame_number"]) == frame_number
        ):
            matching_frames.append(frame)
    if len(matching_frames) != 1:
        raise ValueError("blade-foreground occupancy frame is not unique")
    frame = matching_frames[0]
    evidence = _mapping(frame, "evidence")
    if _sha256(stereo_root / "metadata.json") != str(evidence["source_stereo_metadata_sha256"]):
        raise ValueError("occupancy evidence does not bind the stereo inference")
    if _sha256(session_root / "manifest.json") != str(evidence["source_session_manifest_sha256"]):
        raise ValueError("occupancy evidence does not bind the raw session manifest")
    if _sha256(view_metadata_path) != str(evidence["source_session_view_metadata_sha256"]):
        raise ValueError("occupancy evidence does not bind the raw view metadata")
    frame_sources = _mapping(frame, "sources")
    if (
        _occupancy_source_root(_mapping(frame_sources, "stereo_inference"), "metadata.json")
        != stereo_root
        or _occupancy_source_root(_mapping(frame_sources, "session"), "manifest.json")
        != session_root
    ):
        raise ValueError("blade-foreground sources differ from occupancy frame sources")
    if not np.allclose(
        np.asarray(evidence["base_t_camera_matrix"], dtype=np.float64),
        base_t_left_rectified.matrix,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("blade-foreground pose differs from occupancy evidence")
    integration_record = _mapping(_mapping(frame, "files"), "integration_valid_mask")
    integration_path = _contained(occupancy_root, str(integration_record["path"]))
    if _sha256(integration_path) != str(integration_record["sha256"]):
        raise ValueError("occupancy integration-valid-mask file checksum mismatch")
    integration = np.load(integration_path, allow_pickle=False)
    if integration.dtype != np.bool_ or integration.shape != eligible_mask.shape:
        raise ValueError("occupancy integration-valid mask contract changed")
    actual_hash = occupancy_array_content_hash(integration)
    if actual_hash != integration_valid_mask_hash:
        raise ValueError("occupancy integration-valid-mask content hash changed")
    if str(evidence["integration_valid_mask_content_hash"]) != actual_hash:
        raise ValueError("occupancy evidence integration-valid-mask hash changed")
    if not np.array_equal(eligible_mask, integration):
        raise ValueError("blade-foreground eligible mask differs from occupancy integration mask")

    coarse = _read_json(
        _verify_directory_source_record(
            _mapping(sources, "reference_coarse_model"),
            "metadata.json",
            label="reference coarse model",
        )
        / "metadata.json"
    )
    if int(coarse["schema_version"]) != 5 or coarse.get("motion_authorized") is not False:
        raise ValueError("blade-foreground requires a schema-5 reference coarse model")
    patches = _mapping(coarse, "surface")["patches"]
    matching_patch_ids = [
        str(patch["patch_id"]) for patch in patches if str(patch["patch_id"]) == target_patch_id
    ]
    if len(matching_patch_ids) != 1:
        raise ValueError(
            "blade-foreground target patch is not unique in the reference coarse model"
        )


def _replay_source_result(
    sources: Mapping[str, Any],
    *,
    view_id: str,
    sequence_index: int,
    frame_number: int,
    base_t_left_rectified: PoseSE3,
    intrinsics: CameraIntrinsics,
    eligible_mask: np.ndarray,
    target_patch_id: str,
    config: BladeForegroundConfig,
) -> BladeForegroundMaskResult:
    """Re-run the scientific decision from fully read source arrays."""

    # Local imports avoid a storage-package import cycle: surface coverage reads
    # reconstructed views, whose schema-3 reader in turn validates this asset.
    from biblade_fusion.storage.stereo_inference import (
        read_stereo_inference,
        verify_stereo_inference_source,
    )
    from biblade_fusion.storage.surface_coverage import (
        read_coarse_surface_reference,
    )

    session_root = _verify_directory_source_record(
        _mapping(sources, "session"), "manifest.json", label="session"
    )
    stereo_root = _verify_directory_source_record(
        _mapping(sources, "stereo_inference"),
        "metadata.json",
        label="stereo inference",
    )
    coarse_root = _verify_directory_source_record(
        _mapping(sources, "reference_coarse_model"),
        "metadata.json",
        label="reference coarse model",
    )
    stored_stereo = read_stereo_inference(stereo_root)
    verify_stereo_inference_source(stored_stereo, expected_session=session_root)
    observation = stored_stereo.observation
    if (
        observation.source_view_id != view_id
        or observation.source_sequence_index != sequence_index
        or observation.rectified.source_frame_number != frame_number
        or observation.rectified.calibration.left != intrinsics
    ):
        raise ValueError("blade-foreground replay stereo identity changed")
    surface = read_coarse_surface_reference(coarse_root)
    return reference_guided_blade_mask(
        observation.depth_m,
        np.asarray(eligible_mask, dtype=np.bool_),
        intrinsics,
        base_t_left_rectified,
        surface,
        target_patch_id,
        config,
    )


def _assert_same_result(
    stored: BladeForegroundMaskResult,
    replayed: BladeForegroundMaskResult,
) -> None:
    if (
        stored.algorithm != replayed.algorithm
        or stored.policy_sha256 != replayed.policy_sha256
        or stored.config != replayed.config
        or stored.diagnostics != replayed.diagnostics
        or not np.array_equal(stored.mask, replayed.mask)
        or not np.array_equal(
            stored.reference_depth_m,
            replayed.reference_depth_m,
            equal_nan=True,
        )
        or not np.array_equal(
            stored.target_reference_depth_m,
            replayed.target_reference_depth_m,
            equal_nan=True,
        )
        or not np.array_equal(stored.eligible_mask, replayed.eligible_mask)
    ):
        raise ValueError(
            "blade-foreground arrays or diagnostics do not reproduce from bound sources"
        )


def _occupancy_source_root(record: Mapping[str, Any], expected_filename: str) -> Path:
    raw_root = Path(str(record["root"]))
    root = raw_root.resolve()
    if not raw_root.is_absolute() or raw_root != root:
        raise ValueError("occupancy frame source root must be absolute and canonical")
    if str(record["file"]) != expected_filename:
        raise ValueError("occupancy source metadata filename changed")
    source = root / expected_filename
    if not source.is_file() or _sha256(source) != str(record["sha256"]):
        raise ValueError("occupancy frame source checksum mismatch")
    return root


def _array_record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "content_hash": occupancy_array_content_hash(array),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def _load_array(root: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = _contained(root, str(record["path"]))
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"blade-foreground array checksum mismatch: {path.name}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != list(record["shape"]):
        raise ValueError(f"blade-foreground array manifest mismatch: {path.name}")
    if occupancy_array_content_hash(array) != str(record["content_hash"]):
        raise ValueError(f"blade-foreground array content hash mismatch: {path.name}")
    return array


def _directory_source_record(root: Path, metadata_name: str) -> dict[str, Any]:
    resolved = root.resolve()
    metadata = resolved / metadata_name
    if not resolved.is_dir() or not metadata.is_file():
        raise ValueError(f"Blade-foreground source does not exist: {resolved}")
    return {
        "root": str(resolved),
        "metadata_file": metadata_name,
        "metadata_sha256": _sha256(metadata),
        "metadata_size_bytes": metadata.stat().st_size,
    }


def _verify_directory_source_record(
    record: Mapping[str, Any],
    expected_metadata_name: str,
    *,
    label: str,
) -> Path:
    raw_root = Path(str(record["root"]))
    root = raw_root.resolve()
    if not raw_root.is_absolute() or raw_root != root:
        raise ValueError(f"blade-foreground {label} root must be absolute and canonical")
    if str(record["metadata_file"]) != expected_metadata_name:
        raise ValueError(f"blade-foreground {label} metadata filename changed")
    metadata = root / expected_metadata_name
    if (
        not root.is_dir()
        or not metadata.is_file()
        or _sha256(metadata) != str(record["metadata_sha256"])
        or metadata.stat().st_size != int(record["metadata_size_bytes"])
    ):
        raise ValueError(f"blade-foreground {label} source changed: {root}")
    return root


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


def _intrinsics_from_payload(payload: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        distortion_model=str(payload["distortion_model"]),
        distortion_coefficients=tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _config_from_payload(payload: Mapping[str, Any]) -> BladeForegroundConfig:
    validator = getattr(BladeForegroundConfig, "model_validate", None)
    if validator is not None:
        return validator(dict(payload))
    return BladeForegroundConfig(**dict(payload))


def _object_payload(value: Any) -> dict[str, Any]:
    dumper = getattr(value, "model_dump", None)
    if dumper is not None:
        payload = dumper(mode="json")
    elif is_dataclass(value):
        payload = asdict(value)
    else:
        raise TypeError(f"Cannot serialize blade-foreground value {type(value).__name__}")
    if not isinstance(payload, dict):
        raise TypeError("Blade-foreground serialized value must be an object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"blade-foreground {key} must be an object")
    return value


def _contained(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    resolved_root = root.resolve()
    path = (resolved_root / value).resolve()
    if value.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError("blade-foreground path escapes its artifact root")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: str, *, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value
