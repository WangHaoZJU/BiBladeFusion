"""Immutable, replay-verified assets for unknown-blade bootstrap masks."""

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

from biblade_fusion.perception.bootstrap_foreground import (
    BOOTSTRAP_FOREGROUND_ALGORITHM,
    BootstrapForegroundConfig,
    BootstrapForegroundResult,
    BootstrapSeed,
    bootstrap_policy_sha256,
    bootstrap_seed_payload,
)
from biblade_fusion.storage.stereo_inference import (
    read_stereo_inference,
    verify_stereo_inference_source,
)
from biblade_fusion.workflows.bootstrap_foreground import (
    BootstrapForegroundObservation,
    bootstrap_foundation_stereo_foreground,
)

BOOTSTRAP_FOREGROUND_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "biblade_fusion.bootstrap_foreground"
_ARRAY_NAMES = ("mask", "seed_mask")


@dataclass(frozen=True, slots=True)
class StoredBootstrapForeground:
    root: Path
    observation: BootstrapForegroundObservation
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(str(record["path"]))
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"Bootstrap foreground array escapes its asset: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Bootstrap foreground array checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != list(record["shape"]):
        raise ValueError(f"Bootstrap foreground array manifest mismatch: {relative}")
    return array


def _source_record(source: Path) -> dict[str, Any]:
    root = source.resolve()
    metadata = root / "metadata.json"
    if not root.is_dir() or not metadata.is_file():
        raise ValueError(f"Bootstrap stereo source is missing: {root}")
    return {
        "root": str(root),
        "metadata_sha256": _sha256(metadata),
        "metadata_size_bytes": metadata.stat().st_size,
    }


def _verify_source_record(record: dict[str, Any]) -> Path:
    raw = Path(str(record["root"]))
    root = raw.resolve()
    metadata = root / "metadata.json"
    if not raw.is_absolute() or raw != root:
        raise ValueError("Bootstrap stereo source root must be absolute and canonical")
    if (
        not root.is_dir()
        or not metadata.is_file()
        or _sha256(metadata) != str(record["metadata_sha256"])
        or metadata.stat().st_size != int(record["metadata_size_bytes"])
    ):
        raise ValueError("Bootstrap stereo source changed")
    return root


def _seed_from_payload(payload: Any) -> BootstrapSeed | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Bootstrap seed payload must be an object or null")
    return BootstrapSeed(
        kind=str(payload["kind"]),  # type: ignore[arg-type]
        mode=str(payload["mode"]),  # type: ignore[arg-type]
        vertices_uv=tuple(
            (float(vertex[0]), float(vertex[1])) for vertex in payload["vertices_uv"]
        ),
    )


def _assert_same_result(
    expected: BootstrapForegroundResult,
    actual: BootstrapForegroundResult,
) -> None:
    scalar_fields = (
        "diagnostics",
        "config",
        "seed",
        "algorithm",
        "policy_sha256",
        "left_image_content_sha256",
        "depth_content_sha256",
        "valid_mask_content_sha256",
    )
    if any(getattr(expected, field) != getattr(actual, field) for field in scalar_fields):
        raise ValueError("Bootstrap foreground result does not replay from its stereo source")
    if not np.array_equal(expected.mask, actual.mask) or not np.array_equal(
        expected.seed_mask, actual.seed_mask
    ):
        raise ValueError("Bootstrap foreground arrays do not replay from their stereo source")


def _replay(
    source_root: Path,
    config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
) -> BootstrapForegroundObservation:
    stored_stereo = read_stereo_inference(source_root)
    source_session = Path(str(stored_stereo.metadata["source"]["session"])).resolve()
    verify_stereo_inference_source(
        stored_stereo,
        expected_session=source_session,
    )
    return bootstrap_foundation_stereo_foreground(
        stored_stereo.observation,
        config,
        seed,
    )


def write_bootstrap_foreground(
    output_dir: str | Path,
    observation: BootstrapForegroundObservation,
    *,
    source_stereo_inference: str | Path,
) -> Path:
    """Persist once only after deterministic replay from the bound stereo asset."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Bootstrap foreground output already exists: {output}")
    source_record = _source_record(Path(source_stereo_inference))
    source_root = Path(str(source_record["root"]))
    replayed = _replay(
        source_root,
        observation.result.config,
        observation.result.seed,
    )
    if (
        replayed.source_view_id != observation.source_view_id
        or replayed.source_sequence_index != observation.source_sequence_index
        or replayed.source_frame_number != observation.source_frame_number
    ):
        raise ValueError("Bootstrap foreground identity differs from its stereo source")
    _assert_same_result(observation.result, replayed.result)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        np.save(temporary / "mask.npy", observation.result.mask, allow_pickle=False)
        np.save(
            temporary / "seed_mask.npy",
            observation.result.seed_mask,
            allow_pickle=False,
        )
        result = observation.result
        metadata: dict[str, Any] = {
            "schema_version": BOOTSTRAP_FOREGROUND_SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "identity": {
                "view_id": observation.source_view_id,
                "sequence_index": observation.source_sequence_index,
                "frame_number": observation.source_frame_number,
            },
            "processing": {
                "algorithm": result.algorithm,
                "config": asdict(result.config),
                "seed": bootstrap_seed_payload(result.seed),
                "policy_sha256": result.policy_sha256,
            },
            "input_content_sha256": {
                "left_rectified": result.left_image_content_sha256,
                "depth_m": result.depth_content_sha256,
                "valid_mask": result.valid_mask_content_sha256,
            },
            "diagnostics": asdict(result.diagnostics),
            "files": {name: _array_record(temporary / f"{name}.npy") for name in _ARRAY_NAMES},
            "sources": {"stereo_inference": source_record},
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


def read_bootstrap_foreground(path: str | Path) -> StoredBootstrapForeground:
    """Verify hashes, source identity and algorithm replay before returning a mask."""

    root = Path(path).resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata["schema_version"]) != BOOTSTRAP_FOREGROUND_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {metadata['schema_version']}")
        if metadata.get("artifact_kind") != _ARTIFACT_KIND:
            raise ValueError("unexpected bootstrap foreground artifact kind")
        if metadata.get("motion_authorized") is not False:
            raise ValueError("bootstrap foreground artifact must forbid motion")
        files = metadata["files"]
        if set(files) != set(_ARRAY_NAMES):
            raise ValueError("bootstrap foreground file set is incomplete or unexpected")
        arrays = {name: _load_array(root, files[name]) for name in _ARRAY_NAMES}

        processing = metadata["processing"]
        if processing["algorithm"] != BOOTSTRAP_FOREGROUND_ALGORITHM:
            raise ValueError("bootstrap foreground algorithm identity changed")
        config = BootstrapForegroundConfig(**processing["config"])
        seed = _seed_from_payload(processing["seed"])
        expected_policy = bootstrap_policy_sha256(config, seed)
        if processing["policy_sha256"] != expected_policy:
            raise ValueError("bootstrap foreground policy hash changed")
        source_record = metadata["sources"]["stereo_inference"]
        source_root = _verify_source_record(source_record)
        replayed = _replay(source_root, config, seed)

        identity = metadata["identity"]
        if (
            identity["view_id"] != replayed.source_view_id
            or int(identity["sequence_index"]) != replayed.source_sequence_index
            or int(identity["frame_number"]) != replayed.source_frame_number
        ):
            raise ValueError("bootstrap foreground source identity changed")
        result = replayed.result
        if metadata["diagnostics"] != asdict(result.diagnostics):
            raise ValueError("bootstrap foreground diagnostics do not replay")
        expected_inputs = {
            "left_rectified": result.left_image_content_sha256,
            "depth_m": result.depth_content_sha256,
            "valid_mask": result.valid_mask_content_sha256,
        }
        if metadata["input_content_sha256"] != expected_inputs:
            raise ValueError("bootstrap foreground input identities do not replay")
        if not np.array_equal(arrays["mask"], result.mask) or not np.array_equal(
            arrays["seed_mask"], result.seed_mask
        ):
            raise ValueError("bootstrap foreground mask does not replay")
        return StoredBootstrapForeground(root, replayed, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid bootstrap foreground artifact {root}: {exc}") from exc
