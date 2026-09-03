"""Immutable digital asset for depth-derived unknown-environment occupancy."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AcquisitionConfig, OccupancyConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping import (
    DepthIntegrationConfig,
    DepthRayIntegrator,
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancySnapshot,
    load_occupancy_snapshot,
    save_occupancy_snapshot,
)
from biblade_fusion.mapping.robot_depth_renderer import Es68D435iRobotDepthRenderer
from biblade_fusion.mapping.self_mask import RobotSelfMaskReport
from biblade_fusion.robotics import (
    Es68KinematicModel,
    Es68ModelResources,
    load_es68_flange_t_tcp,
)
from biblade_fusion.robotics.occupancy_collision import (
    OccupancySemanticAttestation,
    _issue_occupancy_semantic_attestation,
)
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.storage.stereo_inference import (
    read_stereo_inference,
    verify_stereo_inference_source,
)
from biblade_fusion.workflows.occupancy_mapping import (
    OccupancyFrameEvidence,
    OccupancyFrameUpdate,
    OccupancyMappingContext,
    RobotDepthRenderer,
    occupancy_array_content_hash,
    occupancy_physical_source_id,
)

OCCUPANCY_MAPPING_SCHEMA_VERSION = 7
LEGACY_OCCUPANCY_MAPPING_SCHEMA_VERSION = 6

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_POSE_TRANSLATION_ATOL_M = 1e-8
_POSE_ROTATION_ATOL_DEG = 1e-5
_SCALAR_TRANSLATION_ATOL_M = 1e-10
_SCALAR_ROTATION_ATOL_DEG = 1e-7


@dataclass(frozen=True, slots=True)
class StoredOccupancyMapping:
    """Fully verified occupancy asset and its reproducible evidence chain."""

    snapshot: OccupancySnapshot
    mapping_context: OccupancyMappingContext
    frame_evidence: tuple[OccupancyFrameEvidence, ...]
    mapping_snapshots: tuple[OccupancySnapshot, ...]
    result_snapshots: tuple[OccupancySnapshot, ...]
    metadata: dict[str, Any]
    semantic_attestation: OccupancySemanticAttestation
    verification_status: str = "full_semantic_verified_for_motion_preflight"
    motion_eligible: bool = True


@dataclass(frozen=True, slots=True)
class ReplayOccupancyMapping:
    """Integrity-checked replay data that is never motion-safety evidence.

    This deliberately distinct type is used on visualization hosts that do not
    possess the active ES68+D435i collision assets.  It cannot be supplied where
    :class:`StoredOccupancyMapping` is required by motion preflight.
    """

    snapshot: OccupancySnapshot
    mapping_context: OccupancyMappingContext
    frame_evidence: tuple[OccupancyFrameEvidence, ...]
    mapping_snapshots: tuple[OccupancySnapshot, ...]
    result_snapshots: tuple[OccupancySnapshot, ...]
    metadata: dict[str, Any]
    verification_status: str = "integrity_only_unverified_for_motion"
    motion_eligible: bool = False


@dataclass(frozen=True, slots=True)
class LegacyReplayOccupancyMapping:
    """Schema-6 integrity replay that is permanently ineligible for motion."""

    snapshot: OccupancySnapshot
    metadata: dict[str, Any]
    legacy_schema_version: int = LEGACY_OCCUPANCY_MAPPING_SCHEMA_VERSION
    verification_status: str = "legacy_integrity_only_unverified_for_motion"
    motion_eligible: bool = False


@dataclass(frozen=True, slots=True)
class VerifiedHandEyeSource:
    """Primary transform independently re-read from one schema-2 hand-eye asset."""

    artifact_sha256: str
    flange_t_left_ir: PoseSE3

    def __post_init__(self) -> None:
        _sha256_digest(self.artifact_sha256, label="hand-eye source SHA-256")
        if (
            self.flange_t_left_ir.parent_frame,
            self.flange_t_left_ir.child_frame,
        ) != ("flange", "left_ir"):
            raise ValueError("Verified hand-eye source must contain flange_T_left_ir")


@dataclass(frozen=True, slots=True)
class OccupancyMappingValidationDependencies:
    """Explicit readers/render factory used by full safety verification.

    Production callers omit this object and receive the strict real-asset
    implementation.  Unit tests may inject deterministic in-memory readers while
    still exercising the same semantic comparison code.
    """

    stereo_reader: Callable[[str | Path], Any]
    stereo_source_verifier: Callable[[Any, Path], Any]
    session_reader_factory: Callable[[str | Path], Any]
    hand_eye_reader: Callable[[Path], VerifiedHandEyeSource]
    renderer_factory: Callable[[Sequence[float]], RobotDepthRenderer]


@dataclass(frozen=True, slots=True)
class _DecodedOccupancyMapping:
    snapshot: OccupancySnapshot
    context: OccupancyMappingContext
    updates: tuple[OccupancyFrameUpdate, ...]
    metadata: dict[str, Any]
    metadata_sha256: str
    stereo_roots: tuple[Path, ...]
    session_roots: tuple[Path, ...]
    hand_eye_path: Path


@dataclass(frozen=True, slots=True)
class _PoseReplayContract:
    robot_model_hash: str
    hand_eye_hash: str
    joint_zero_offsets_rad: tuple[float, float, float, float, float, float]
    flange_t_tcp: PoseSE3
    flange_t_left_ir: PoseSE3
    left_rectified_t_left_ir: PoseSE3


@dataclass(frozen=True, slots=True)
class _LiveVerificationCacheEntry:
    """Process-local authority for a mapping just produced by the live integrator.

    The expensive mathematical replay remains mandatory when an artifact enters a
    process from disk.  Inside the process that just computed and semantically
    validated the updates, repeating every ray cast on each read adds no independent
    evidence.  File identity snapshots invalidate this shortcut if any bound file is
    replaced or edited before publication/execution.
    """

    stored: StoredOccupancyMapping
    file_identities: tuple[tuple[str, int, int, int], ...]


_LIVE_CACHE_CAPACITY = 8
_LIVE_VERIFICATION_CACHE: OrderedDict[Path, _LiveVerificationCacheEntry] = OrderedDict()
_LIVE_VERIFICATION_CACHE_LOCK = threading.Lock()


def write_occupancy_mapping(
    output_dir: str | Path,
    updates: Sequence[OccupancyFrameUpdate],
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    *,
    source_stereo_inferences: Sequence[str | Path],
    source_sessions: Sequence[str | Path],
    source_hand_eye: str | Path,
) -> Path:
    """Persist snapshots, masks, sources and a recomputable evidence chain."""

    return _write_occupancy_mapping_with_dependencies(
        output_dir,
        updates,
        occupancy_config,
        acquisition_config,
        source_stereo_inferences=source_stereo_inferences,
        source_sessions=source_sessions,
        source_hand_eye=source_hand_eye,
        validation_dependencies=_production_validation_dependencies(),
    )


def write_live_occupancy_mapping(
    output_dir: str | Path,
    updates: Sequence[OccupancyFrameUpdate],
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    *,
    source_stereo_inferences: Sequence[str | Path],
    source_sessions: Sequence[str | Path],
    source_hand_eye: str | Path,
) -> tuple[Path, StoredOccupancyMapping]:
    """Persist one live result and retain its already-proven in-process authority.

    This is deliberately narrower than :func:`write_occupancy_mapping`: callers must
    supply the exact updates returned by the active integrator.  Structural, pose,
    source and robot-render semantics are still validated before publication, while
    the depth rays themselves are not recomputed.  A later process has no cache entry
    and therefore performs the original full mathematical replay.
    """

    dependencies = _production_validation_dependencies()
    destination = _write_occupancy_mapping_with_dependencies(
        output_dir,
        updates,
        occupancy_config,
        acquisition_config,
        source_stereo_inferences=source_stereo_inferences,
        source_sessions=source_sessions,
        source_hand_eye=source_hand_eye,
        validation_dependencies=dependencies,
        replay_depth_rays=False,
    )
    root = destination.resolve()
    metadata_bytes = (root / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    final = updates[-1]
    context_robot = _mapping(final.mapping_context.to_payload(), "robot")
    robot_geometry_hash = _sha256_digest(
        context_robot["model_content_hash"],
        label="robot.model_content_hash",
    )
    stored = StoredOccupancyMapping(
        snapshot=final.snapshot,
        mapping_context=final.mapping_context,
        frame_evidence=tuple(update.evidence for update in updates),
        mapping_snapshots=tuple(update.mapping_snapshot for update in updates),
        result_snapshots=tuple(update.snapshot for update in updates),
        metadata=dict(metadata),
        semantic_attestation=_issue_occupancy_semantic_attestation(
            occupancy_metadata_sha256=metadata_sha256,
            snapshot=final.snapshot,
            robot_geometry_hash=robot_geometry_hash,
        ),
    )
    bound_roots = (
        root,
        *(Path(path).resolve() for path in source_stereo_inferences),
        *(Path(path).resolve() for path in source_sessions),
    )
    identities = _file_identities((*bound_roots, Path(source_hand_eye).resolve()))
    with _LIVE_VERIFICATION_CACHE_LOCK:
        _LIVE_VERIFICATION_CACHE[root] = _LiveVerificationCacheEntry(
            replace(stored, metadata=deepcopy(stored.metadata)),
            identities,
        )
        _LIVE_VERIFICATION_CACHE.move_to_end(root)
        while len(_LIVE_VERIFICATION_CACHE) > _LIVE_CACHE_CAPACITY:
            _LIVE_VERIFICATION_CACHE.popitem(last=False)
    return root, stored


def _write_occupancy_mapping_with_dependencies(
    output_dir: str | Path,
    updates: Sequence[OccupancyFrameUpdate],
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    *,
    source_stereo_inferences: Sequence[str | Path],
    source_sessions: Sequence[str | Path],
    source_hand_eye: str | Path,
    validation_dependencies: OccupancyMappingValidationDependencies,
    replay_depth_rays: bool = True,
) -> Path:

    if not updates:
        raise ValueError("At least one occupancy update is required")
    if not (len(updates) == len(source_stereo_inferences) == len(source_sessions)):
        raise ValueError("Occupancy updates and source lists must have equal length")
    _validate_update_chain(
        updates,
        occupancy_config,
        acquisition_config,
        replay_depth_rays=replay_depth_rays,
    )
    stereo_roots = tuple(Path(path).resolve() for path in source_stereo_inferences)
    session_roots = tuple(Path(path).resolve() for path in source_sessions)
    hand_eye_path = Path(source_hand_eye).resolve()
    if not hand_eye_path.is_file():
        raise ValueError(f"Occupancy hand-eye source does not exist: {hand_eye_path}")
    _validate_semantic_source_chain(
        updates,
        stereo_roots,
        session_roots,
        hand_eye_path,
        validation_dependencies,
    )

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Occupancy output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        final_snapshot = updates[-1].snapshot
        final_path = save_occupancy_snapshot(temporary / "occupancy.json", final_snapshot)
        frames: list[dict[str, Any]] = []
        for index, (update, stereo_root, session_root) in enumerate(
            zip(updates, source_stereo_inferences, source_sessions, strict=True)
        ):
            prefix = f"{index:04d}_{_safe_name(update.evidence.source_view_id)}"
            files: dict[str, dict[str, Any]] = {}
            for name, array in (
                ("source_depth_m", update.source_depth_m),
                ("stereo_valid_mask", update.stereo_valid_mask),
                ("stereo_confidence", update.stereo_confidence),
                ("predicted_robot_depth_m", update.predicted_robot_depth_m),
                ("robot_mask", update.robot_mask),
                ("integration_valid_mask", update.integration_valid_mask),
            ):
                path = temporary / f"{prefix}_{name}.npy"
                np.save(path, array, allow_pickle=False)
                files[name] = _array_record(path)

            mapping_path = save_occupancy_snapshot(
                temporary / f"{prefix}_mapping_snapshot.json",
                update.mapping_snapshot,
            )
            result_path = save_occupancy_snapshot(
                temporary / f"{prefix}_result_snapshot.json",
                update.snapshot,
            )
            stereo_path = Path(stereo_root).resolve()
            session_path = Path(session_root).resolve()
            frames.append(
                {
                    "evidence": asdict(update.evidence),
                    "mapping_snapshot": _snapshot_record(mapping_path, update.mapping_snapshot),
                    "result_snapshot": _snapshot_record(result_path, update.snapshot),
                    "files": files,
                    "sources": {
                        "stereo_inference": _source_record(stereo_path, "metadata.json"),
                        "session": _source_record(session_path, "manifest.json"),
                    },
                }
            )

        context = updates[0].mapping_context
        metadata = {
            "schema_version": OCCUPANCY_MAPPING_SCHEMA_VERSION,
            "artifact_kind": "biblade_fusion.occupancy_mapping",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "mapping_context": {
                "payload": context.to_payload(),
                "content_hash": context.content_hash,
            },
            "snapshot": _snapshot_record(final_path, final_snapshot),
            "configuration": {
                "occupancy": occupancy_config.model_dump(mode="json"),
                "acquisition": acquisition_config.model_dump(mode="json"),
            },
            "sources": {
                "hand_eye": _source_record(
                    hand_eye_path.parent,
                    hand_eye_path.name,
                ),
            },
            "frames": frames,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_occupancy_mapping(path: str | Path) -> StoredOccupancyMapping:
    """Fully verify an occupancy asset for possible motion-preflight use.

    A fresh process re-reads semantic source artifacts and re-renders robot self-depth
    with the active ES68+D435i STL bundle. The live writer may supply an equivalent
    in-process authority only while every bound file identity is unchanged. Missing
    active assets remain a hard failure at this safety boundary.
    """

    root = Path(path).resolve()
    cached = _read_live_verification_cache(root)
    if cached is not None:
        return cached
    return _read_occupancy_mapping_with_dependencies(
        root,
        validation_dependencies=_production_validation_dependencies(),
    )


def _read_occupancy_mapping_with_dependencies(
    path: str | Path,
    *,
    validation_dependencies: OccupancyMappingValidationDependencies,
) -> StoredOccupancyMapping:
    root = Path(path).resolve()
    decoded = _read_occupancy_mapping_integrity(root)
    _validate_semantic_source_chain(
        decoded.updates,
        decoded.stereo_roots,
        decoded.session_roots,
        decoded.hand_eye_path,
        validation_dependencies,
    )
    context_robot = _mapping(decoded.context.to_payload(), "robot")
    robot_geometry_hash = _sha256_digest(
        context_robot["model_content_hash"],
        label="robot.model_content_hash",
    )
    if _sha256(root / "metadata.json") != decoded.metadata_sha256:
        raise ValueError("Occupancy metadata changed during full semantic verification")
    semantic_attestation = _issue_occupancy_semantic_attestation(
        occupancy_metadata_sha256=decoded.metadata_sha256,
        snapshot=decoded.snapshot,
        robot_geometry_hash=robot_geometry_hash,
    )
    return StoredOccupancyMapping(
        snapshot=decoded.snapshot,
        mapping_context=decoded.context,
        frame_evidence=tuple(update.evidence for update in decoded.updates),
        mapping_snapshots=tuple(update.mapping_snapshot for update in decoded.updates),
        result_snapshots=tuple(update.snapshot for update in decoded.updates),
        metadata=decoded.metadata,
        semantic_attestation=semantic_attestation,
    )


def _file_identities(
    roots: Sequence[Path],
) -> tuple[tuple[str, int, int, int], ...]:
    """Snapshot ordinary-file identity without re-reading large immutable arrays."""

    paths: set[Path] = set()
    for raw in roots:
        root = Path(raw).resolve()
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*") if path.is_file())
        else:
            raise ValueError(f"Bound live occupancy source is missing: {root}")
    identities: list[tuple[str, int, int, int]] = []
    for path in sorted(paths, key=str):
        stat = path.stat()
        identities.append(
            (str(path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
        )
    return tuple(identities)


def _read_live_verification_cache(root: Path) -> StoredOccupancyMapping | None:
    with _LIVE_VERIFICATION_CACHE_LOCK:
        entry = _LIVE_VERIFICATION_CACHE.get(root)
    if entry is None:
        return None
    try:
        current = tuple(
            (
                path_text,
                int((stat := Path(path_text).stat()).st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
            for path_text, _size, _mtime, _ctime in entry.file_identities
        )
    except OSError:
        current = ()
    if current != entry.file_identities:
        with _LIVE_VERIFICATION_CACHE_LOCK:
            _LIVE_VERIFICATION_CACHE.pop(root, None)
        return None
    with _LIVE_VERIFICATION_CACHE_LOCK:
        if root in _LIVE_VERIFICATION_CACHE:
            _LIVE_VERIFICATION_CACHE.move_to_end(root)
    # The dataclass is frozen, but its JSON metadata is an ordinary dictionary.
    # Never let a caller mutate the cached authority observed by a later read.
    return replace(entry.stored, metadata=deepcopy(entry.stored.metadata))


def read_occupancy_mapping_for_replay(path: str | Path) -> ReplayOccupancyMapping:
    """Read structural integrity only for permanently blocked visualization replay.

    This named API intentionally returns a different type and never claims motion
    eligibility.  It is suitable when the visualization host lacks active robot
    collision meshes; motion preflight must call :func:`read_occupancy_mapping`.
    """

    decoded = _read_occupancy_mapping_integrity(path)
    return ReplayOccupancyMapping(
        decoded.snapshot,
        decoded.context,
        tuple(update.evidence for update in decoded.updates),
        tuple(update.mapping_snapshot for update in decoded.updates),
        tuple(update.snapshot for update in decoded.updates),
        decoded.metadata,
    )


def read_legacy_occupancy_mapping_for_replay(
    path: str | Path,
) -> LegacyReplayOccupancyMapping:
    """Read a schema-6 artifact only through an explicitly non-motion API.

    Legacy frame identities used a logical view label and therefore cannot prove
    physical-observation uniqueness.  This reader checks file/source integrity for
    visualization and archival migration, but deliberately does not construct a
    semantic attestation or a motion-eligible ``StoredOccupancyMapping``.
    """

    root = Path(path).resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError("occupancy metadata root must be an object")
        if int(metadata["schema_version"]) != LEGACY_OCCUPANCY_MAPPING_SCHEMA_VERSION:
            raise ValueError("explicit legacy reader accepts only schema 6")
        if metadata.get("artifact_kind") != "biblade_fusion.occupancy_mapping":
            raise ValueError("unexpected occupancy artifact kind")
        if metadata.get("motion_authorized") is not False:
            raise ValueError("Legacy occupancy artifact must explicitly forbid motion")
        snapshot = _load_snapshot_record(root, _mapping(metadata, "snapshot"))
        top_sources = _mapping(metadata, "sources")
        _require_exact_keys(top_sources, {"hand_eye"}, label="artifact sources")
        _verify_source(_mapping(top_sources, "hand_eye"), relocation_root=root)
        frames = _sequence(metadata, "frames")
        if not frames:
            raise ValueError("legacy occupancy artifact contains no frame evidence")
        last_result: OccupancySnapshot | None = None
        for raw_frame in frames:
            if not isinstance(raw_frame, Mapping):
                raise ValueError("legacy occupancy frame record must be an object")
            evidence = _mapping(raw_frame, "evidence")
            if "physical_source_id" in evidence:
                raise ValueError("schema-6 evidence unexpectedly claims a physical source ID")
            sources = _mapping(raw_frame, "sources")
            _require_exact_keys(
                sources,
                {"stereo_inference", "session"},
                label="legacy frame sources",
            )
            stereo_metadata = _verify_source(
                _mapping(sources, "stereo_inference"),
                expected_filename="metadata.json",
                relocation_root=root,
            )
            session_manifest = _verify_source(
                _mapping(sources, "session"),
                expected_filename="manifest.json",
                relocation_root=root,
            )
            if _sha256(stereo_metadata) != str(evidence["source_stereo_metadata_sha256"]):
                raise ValueError("legacy stereo metadata SHA-256 differs from evidence")
            if _sha256(session_manifest) != str(evidence["source_session_manifest_sha256"]):
                raise ValueError("legacy session manifest SHA-256 differs from evidence")
            files = _mapping(raw_frame, "files")
            source_depth = _load_array(root, _mapping(files, "source_depth_m"))
            stereo_valid = _load_array(root, _mapping(files, "stereo_valid_mask"))
            stereo_confidence = _load_array(root, _mapping(files, "stereo_confidence"))
            predicted_depth = _load_array(root, _mapping(files, "predicted_robot_depth_m"))
            robot_mask = _load_array(root, _mapping(files, "robot_mask"))
            integration_mask = _load_array(root, _mapping(files, "integration_valid_mask"))
            _validate_array_dtypes(
                source_depth=source_depth,
                stereo_valid=stereo_valid,
                stereo_confidence=stereo_confidence,
                predicted_depth=predicted_depth,
                robot_mask=robot_mask,
                integration_mask=integration_mask,
            )
            mapping_snapshot = _load_snapshot_record(
                root,
                _mapping(raw_frame, "mapping_snapshot"),
            )
            last_result = _load_snapshot_record(
                root,
                _mapping(raw_frame, "result_snapshot"),
            )
            logical_id = str(evidence["source_view_id"]).strip()
            mapping_ids = tuple(str(value).strip() for value in evidence["mapping_source_view_ids"])
            if not logical_id or not mapping_ids or mapping_ids[-1] != logical_id:
                raise ValueError("legacy logical source identity is inconsistent")
            if mapping_snapshot.source_view_ids != mapping_ids:
                raise ValueError("legacy snapshot logical source identities differ")
            expected_array_hashes = {
                "source_depth_content_hash": occupancy_array_content_hash(source_depth),
                "stereo_valid_mask_content_hash": occupancy_array_content_hash(stereo_valid),
                "stereo_confidence_content_hash": occupancy_array_content_hash(
                    stereo_confidence
                ),
                "predicted_robot_depth_content_hash": occupancy_array_content_hash(
                    predicted_depth
                ),
                "robot_mask_content_hash": occupancy_array_content_hash(robot_mask),
                "integration_valid_mask_content_hash": occupancy_array_content_hash(
                    integration_mask
                ),
            }
            if any(
                str(evidence[name]) != expected
                for name, expected in expected_array_hashes.items()
            ):
                raise ValueError("legacy occupancy array content hash differs from evidence")
            if (
                str(evidence["mapping_snapshot_content_hash"])
                != mapping_snapshot.content_hash
                or int(evidence["mapping_snapshot_sequence"]) != mapping_snapshot.sequence
            ):
                raise ValueError("legacy mapping snapshot identity differs from evidence")
            raw_hash_payload = dict(evidence)
            claimed_evidence_hash = str(raw_hash_payload.pop("quality_evidence_hash"))
            reproduced_evidence_hash = hashlib.sha256(
                json.dumps(
                    raw_hash_payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if claimed_evidence_hash != reproduced_evidence_hash:
                raise ValueError("legacy occupancy evidence hash mismatch")
            if last_result.quality_evidence_hash != claimed_evidence_hash:
                raise ValueError("legacy result snapshot is not bound to frame evidence")
        if last_result is None or snapshot != last_result:
            raise ValueError("legacy final occupancy snapshot differs from its final frame")
        return LegacyReplayOccupancyMapping(snapshot, dict(metadata))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid legacy occupancy-mapping artifact {root}: {exc}") from exc


def _read_occupancy_mapping_integrity(path: str | Path) -> _DecodedOccupancyMapping:
    root = Path(path).resolve()
    try:
        metadata_bytes = (root / "metadata.json").read_bytes()
        metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, Mapping):
            raise ValueError("occupancy metadata root must be an object")
        if int(metadata["schema_version"]) != OCCUPANCY_MAPPING_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {metadata['schema_version']}")
        if metadata.get("artifact_kind") != "biblade_fusion.occupancy_mapping":
            raise ValueError("unexpected occupancy artifact kind")
        if metadata.get("motion_authorized") is not False:
            raise ValueError("Occupancy mapping artifact must explicitly forbid motion")

        configuration = _mapping(metadata, "configuration")
        occupancy_config = OccupancyConfig.model_validate(_mapping(configuration, "occupancy"))
        acquisition_config = AcquisitionConfig.model_validate(
            _mapping(configuration, "acquisition")
        )
        context_record = _mapping(metadata, "mapping_context")
        context = OccupancyMappingContext(
            json.dumps(
                _mapping(context_record, "payload"),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            str(context_record["content_hash"]),
        )
        _validate_context_configuration(context, occupancy_config, acquisition_config)

        top_sources = _mapping(metadata, "sources")
        _require_exact_keys(top_sources, {"hand_eye"}, label="artifact sources")
        hand_eye_record = _mapping(top_sources, "hand_eye")
        hand_eye_path = _verify_source(hand_eye_record, relocation_root=root)

        raw_frames = _sequence(metadata, "frames")
        if not raw_frames:
            raise ValueError("occupancy artifact contains no frame evidence")
        updates: list[OccupancyFrameUpdate] = []
        stereo_roots: list[Path] = []
        session_roots: list[Path] = []
        for raw_frame in raw_frames:
            if not isinstance(raw_frame, Mapping):
                raise ValueError("occupancy frame record must be an object")
            sources = _mapping(raw_frame, "sources")
            _require_exact_keys(
                sources,
                {"stereo_inference", "session"},
                label="frame sources",
            )
            stereo_path = _verify_source(
                _mapping(sources, "stereo_inference"),
                expected_filename="metadata.json",
                relocation_root=root,
            )
            session_path = _verify_source(
                _mapping(sources, "session"),
                expected_filename="manifest.json",
                relocation_root=root,
            )
            stereo_roots.append(stereo_path.parent)
            session_roots.append(session_path.parent)
            files = _mapping(raw_frame, "files")
            predicted_depth = _load_array(
                root,
                _mapping(files, "predicted_robot_depth_m"),
            )
            source_depth = _load_array(root, _mapping(files, "source_depth_m"))
            stereo_valid = _load_array(root, _mapping(files, "stereo_valid_mask"))
            stereo_confidence = _load_array(
                root,
                _mapping(files, "stereo_confidence"),
            )
            robot_mask = _load_array(root, _mapping(files, "robot_mask"))
            integration_mask = _load_array(
                root,
                _mapping(files, "integration_valid_mask"),
            )
            _validate_array_dtypes(
                source_depth=source_depth,
                stereo_valid=stereo_valid,
                stereo_confidence=stereo_confidence,
                predicted_depth=predicted_depth,
                robot_mask=robot_mask,
                integration_mask=integration_mask,
            )
            mapping_snapshot = _load_snapshot_record(
                root,
                _mapping(raw_frame, "mapping_snapshot"),
            )
            result_snapshot = _load_snapshot_record(
                root,
                _mapping(raw_frame, "result_snapshot"),
            )
            evidence = _evidence_from_payload(_mapping(raw_frame, "evidence"))
            updates.append(
                OccupancyFrameUpdate(
                    result_snapshot,
                    mapping_snapshot,
                    context,
                    source_depth,
                    stereo_valid,
                    stereo_confidence,
                    predicted_depth,
                    robot_mask,
                    integration_mask,
                    evidence,
                )
            )

        _validate_update_chain(updates, occupancy_config, acquisition_config)
        final_snapshot = _load_snapshot_record(root, _mapping(metadata, "snapshot"))
        if final_snapshot != updates[-1].snapshot:
            raise ValueError("final occupancy snapshot does not match the last frame result")
        return _DecodedOccupancyMapping(
            snapshot=final_snapshot,
            context=context,
            updates=tuple(updates),
            metadata=dict(metadata),
            metadata_sha256=metadata_sha256,
            stereo_roots=tuple(stereo_roots),
            session_roots=tuple(session_roots),
            hand_eye_path=hand_eye_path,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid occupancy-mapping artifact {root}: {exc}") from exc


def _production_validation_dependencies() -> OccupancyMappingValidationDependencies:
    return OccupancyMappingValidationDependencies(
        stereo_reader=read_stereo_inference,
        stereo_source_verifier=lambda stored, session: verify_stereo_inference_source(
            stored, expected_session=session
        ),
        session_reader_factory=SessionReader,
        hand_eye_reader=_read_verified_hand_eye_source,
        renderer_factory=lambda offsets: Es68D435iRobotDepthRenderer.from_active_resources(
            joint_zero_offsets_rad=offsets,
        ),
    )


def _read_verified_hand_eye_source(path: Path) -> VerifiedHandEyeSource:
    resolved = path.resolve()
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("hand-eye root must be an object")
        if int(payload["schema_version"]) != 2:
            raise ValueError("safety occupancy requires schema-2 hand-eye calibration")
        if (str(payload["parent_frame"]), str(payload["child_frame"])) != (
            "flange",
            "left_ir",
        ):
            raise ValueError("hand-eye frames must be flange and left_ir")
        primary = PoseSE3("flange", "left_ir", payload["matrix"])
        explicit = payload.get("flange_T_left_ir")
        if explicit is not None:
            _require_pose_close(
                PoseSE3("flange", "left_ir", explicit),
                primary,
                label="Hand-eye primary flange_T_left_ir",
            )
        derived = payload.get("derived_runtime")
        if not isinstance(derived, Mapping):
            raise ValueError("schema-2 hand-eye lacks derived_runtime validation")
        packaged_flange_t_tcp = load_es68_flange_t_tcp()
        recorded_flange_t_tcp = PoseSE3(
            "flange",
            "tcp",
            derived["flange_T_tcp_validation"],
        )
        _require_pose_close(
            recorded_flange_t_tcp,
            packaged_flange_t_tcp,
            label="Hand-eye flange_T_tcp validation asset",
        )
        expected_tcp_t_left_ir = packaged_flange_t_tcp.inverse().compose(primary)
        recorded_tcp_t_left_ir = PoseSE3(
            "tcp",
            "left_ir",
            derived["tcp_T_left_ir"],
        )
        _require_pose_close(
            recorded_tcp_t_left_ir,
            expected_tcp_t_left_ir,
            label="Hand-eye derived tcp_T_left_ir",
        )
        return VerifiedHandEyeSource(_sha256(resolved), primary)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid bound hand-eye source {resolved}: {exc}") from exc


def _validate_semantic_source_chain(
    updates: Sequence[OccupancyFrameUpdate],
    stereo_roots: Sequence[Path],
    session_roots: Sequence[Path],
    hand_eye_path: Path,
    dependencies: OccupancyMappingValidationDependencies,
) -> None:
    if not updates or not (len(updates) == len(stereo_roots) == len(session_roots)):
        raise ValueError("Occupancy semantic source lists must match frame updates")
    context = updates[0].mapping_context
    payload = context.to_payload()
    robot = _mapping(payload, "robot")
    hand_eye = _mapping(payload, "hand_eye")
    offsets = _finite_six_vector(
        robot["joint_zero_offsets_rad"],
        label="robot.joint_zero_offsets_rad",
    )
    context_robot_hash = _sha256_digest(
        robot["model_content_hash"],
        label="robot.model_content_hash",
    )
    context_self_mask_exclusions = _string_tuple(
        robot["self_mask_excluded_link_names"],
        label="robot.self_mask_excluded_link_names",
    )
    context_self_mask_backend = _non_empty_string(
        robot["self_mask_render_backend"],
        label="robot.self_mask_render_backend",
    )
    context_hand_eye_hash = _sha256_digest(
        hand_eye["artifact_sha256"],
        label="hand_eye.artifact_sha256",
    )
    context_flange_t_left_ir = _pose_from_payload(
        hand_eye["flange_T_left_ir"],
        parent_frame="flange",
        child_frame="left_ir",
        label="hand_eye.flange_T_left_ir",
    )

    if _sha256(hand_eye_path) != context_hand_eye_hash:
        raise ValueError("Bound hand-eye file SHA-256 differs from mapping context")
    verified_hand_eye = dependencies.hand_eye_reader(hand_eye_path)
    if verified_hand_eye.artifact_sha256 != context_hand_eye_hash:
        raise ValueError("Re-read hand-eye SHA-256 differs from mapping context")
    _require_pose_close(
        verified_hand_eye.flange_t_left_ir,
        context_flange_t_left_ir,
        label="Bound hand-eye flange_T_left_ir",
    )

    renderer = dependencies.renderer_factory(offsets)
    if renderer.model_content_hash != context_robot_hash:
        raise ValueError("Mapping context robot hash is not the active robot geometry hash")
    if tuple(renderer.self_mask_excluded_link_names) != context_self_mask_exclusions:
        raise ValueError("Active robot renderer self-mask exclusions differ from mapping context")
    if renderer.self_mask_render_backend != context_self_mask_backend:
        raise ValueError("Active robot renderer self-mask backend differs from mapping context")
    renderer_offsets = tuple(float(value) for value in renderer.joint_zero_offsets_rad)
    if renderer_offsets != offsets:
        raise ValueError("Active robot renderer offsets differ from mapping context")

    context_rectified = _mapping(payload, "rectified_stereo")
    context_foundation = _mapping(payload, "foundation_stereo")
    acquisition_contract = AcquisitionConfig.model_validate(
        _mapping(payload, "acquisition_contract")
    )
    left_intrinsics = _intrinsics_from_context(_mapping(context_rectified, "left"))
    for update, stereo_root, session_root in zip(
        updates,
        stereo_roots,
        session_roots,
        strict=True,
    ):
        evidence = update.evidence
        stereo_root = Path(stereo_root).resolve()
        session_root = Path(session_root).resolve()
        if _sha256(stereo_root / "metadata.json") != evidence.source_stereo_metadata_sha256:
            raise ValueError("Occupancy stereo metadata SHA-256 differs from frame evidence")
        if _sha256(session_root / "manifest.json") != evidence.source_session_manifest_sha256:
            raise ValueError("Occupancy session manifest SHA-256 differs from frame evidence")

        stored_stereo = dependencies.stereo_reader(stereo_root)
        dependencies.stereo_source_verifier(stored_stereo, session_root)
        stereo = stored_stereo.observation
        stereo_source = stored_stereo.metadata.get("source")
        if not isinstance(stereo_source, Mapping):
            raise ValueError("Occupancy stereo source metadata is missing")
        try:
            bound_session = Path(str(stereo_source["session"])).resolve()
            source_identity = (
                str(stereo_source["view_id"]),
                int(stereo_source["sequence_index"]),
                int(stereo_source["frame_number"]),
            )
            source_monotonic_time_ns = int(stereo_source["monotonic_time_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Occupancy stereo source identity metadata is invalid") from exc
        expected_identity = (
            evidence.source_view_id,
            evidence.source_sequence_index,
            evidence.frame_number,
        )
        observation_identity = (
            stereo.source_view_id,
            stereo.source_sequence_index,
            stereo.rectified.source_frame_number,
        )
        if (
            source_identity != expected_identity
            or observation_identity != expected_identity
            or bound_session != session_root
            or source_monotonic_time_ns != int(stereo.rectified.source_monotonic_time_ns)
        ):
            raise ValueError("Occupancy stereo source identity does not match frame evidence")
        _require_array_equal(
            np.asarray(stereo.depth_m, dtype=np.float64),
            update.source_depth_m,
            label="Occupancy source stereo depth",
            equal_nan=True,
        )
        _require_array_equal(
            np.asarray(stereo.result.valid_mask, dtype=np.bool_),
            update.stereo_valid_mask,
            label="Occupancy source stereo valid mask",
        )
        if stereo.result.confidence is None:
            raise ValueError("Occupancy source stereo confidence is missing")
        _require_array_equal(
            np.asarray(stereo.result.confidence, dtype=np.float64),
            update.stereo_confidence,
            label="Occupancy source stereo confidence",
        )
        reproduced_depth = stereo.result.depth_m(stereo.rectified.calibration)
        _require_array_equal(
            np.asarray(reproduced_depth, dtype=np.float64),
            np.asarray(stereo.depth_m, dtype=np.float64),
            label="Occupancy source stereo disparity/depth",
            equal_nan=True,
        )
        if _rectified_stereo_payload(stereo.rectified.calibration) != dict(context_rectified):
            raise ValueError("Occupancy stereo calibration differs from mapping context")
        if dict(stereo.result.metadata) != dict(context_foundation):
            raise ValueError("Occupancy FoundationStereo metadata differs from mapping context")

        reader = dependencies.session_reader_factory(session_root)
        if Path(reader.path).resolve() != session_root:
            raise ValueError("Occupancy session reader resolved a different root")
        bundle = reader.load_bundle(evidence.source_sequence_index)
        bundle_identity = (
            bundle.view_id,
            bundle.sequence_index,
            bundle.stereo.frame_number,
        )
        if bundle_identity != expected_identity:
            raise ValueError("Occupancy session source identity does not match frame evidence")
        _validate_session_capture_contract(bundle, acquisition_contract)
        if int(bundle.stereo.monotonic_time_ns) != int(stereo.rectified.source_monotonic_time_ns):
            raise ValueError("Occupancy session/stereo monotonic time differs")
        _require_array_equal(
            np.asarray(bundle.selected_robot_state.joint_positions_rad, dtype=np.float64),
            np.asarray(evidence.joint_positions_rad, dtype=np.float64),
            label="Occupancy session joints",
        )
        _require_pose_close(
            bundle.selected_robot_state.base_t_tcp,
            PoseSE3("base", "tcp", evidence.observed_base_t_tcp_matrix),
            label="Occupancy session observed base_T_tcp",
        )
        captured = _normalized_utc_text(
            reader.manifest["created_at_utc"],
            label="Occupancy session created_at_utc",
        )
        if captured != evidence.captured_at_utc:
            raise ValueError("Occupancy session timestamp differs from frame evidence")
        descriptor = reader.descriptor(evidence.source_sequence_index)
        relative_view = Path(str(descriptor.relative_path))
        view_metadata = (session_root / relative_view / "metadata.json").resolve()
        if relative_view.is_absolute() or not view_metadata.is_relative_to(session_root):
            raise ValueError("Occupancy session view metadata escapes its source root")
        if _sha256(view_metadata) != evidence.source_session_view_metadata_sha256:
            raise ValueError("Occupancy session view metadata SHA-256 differs from frame evidence")
        reproduced_physical_source_id = occupancy_physical_source_id(
            source_session_manifest_sha256=_sha256(session_root / "manifest.json"),
            source_session_view_metadata_sha256=_sha256(view_metadata),
            source_sequence_index=bundle.sequence_index,
            frame_number=bundle.stereo.frame_number,
            source_view_id=bundle.view_id,
        )
        if reproduced_physical_source_id != evidence.physical_source_id:
            raise ValueError("Occupancy physical source identity does not reproduce")

        rerendered = renderer.render_robot_depth(
            left_intrinsics,
            evidence.joint_positions_rad,
            np.asarray(evidence.base_t_camera_matrix, dtype=np.float64),
        )
        _require_array_equal(
            np.asarray(rerendered, dtype=np.float64),
            update.predicted_robot_depth_m,
            label="Occupancy robot depth does not reproduce from active geometry",
        )


def _validate_session_capture_contract(
    bundle: Any,
    config: AcquisitionConfig,
) -> None:
    """Recompute stop-and-capture evidence instead of trusting stored metrics."""

    before = bundle.robot_state_before
    after = bundle.robot_state_after
    selected = bundle.selected_robot_state
    stereo = bundle.stereo
    before_time = int(before.monotonic_time_ns)
    stereo_time = int(stereo.monotonic_time_ns)
    after_time = int(after.monotonic_time_ns)
    if not before_time <= stereo_time <= after_time:
        raise ValueError("Occupancy stereo timestamp is outside its robot-state bracket")
    device_times_ms = np.asarray(
        [stereo.left_device_time_ms, stereo.right_device_time_ms],
        dtype=np.float64,
    )
    if not np.isfinite(device_times_ms).all() or np.any(device_times_ms < 0.0):
        raise ValueError("Occupancy raw stereo device timestamps are invalid")

    bracket_ms = (after_time - before_time) / 1e6
    joint_delta_rad = float(
        np.max(
            np.abs(
                np.asarray(after.joint_positions_rad, dtype=np.float64)
                - np.asarray(before.joint_positions_rad, dtype=np.float64)
            )
        )
    )
    tcp_translation_delta_m = float(
        np.linalg.norm(after.base_t_tcp.translation_m - before.base_t_tcp.translation_m)
    )
    tcp_rotation_delta_rad = math.radians(
        _rotation_error_deg(before.base_t_tcp.rotation, after.base_t_tcp.rotation)
    )
    before_offset_ns = abs(stereo_time - before_time)
    after_offset_ns = abs(after_time - stereo_time)
    expected_selected = before if before_offset_ns <= after_offset_ns else after
    if int(selected.monotonic_time_ns) != int(expected_selected.monotonic_time_ns):
        raise ValueError("Occupancy selected robot state is not nearest to the stereo frame")
    _require_array_equal(
        np.asarray(selected.joint_positions_rad, dtype=np.float64),
        np.asarray(expected_selected.joint_positions_rad, dtype=np.float64),
        label="Occupancy selected robot-state joints",
    )
    _require_pose_close(
        selected.base_t_tcp,
        expected_selected.base_t_tcp,
        label="Occupancy selected robot-state base_T_tcp",
    )

    recomputed = (
        bracket_ms,
        joint_delta_rad,
        tcp_translation_delta_m,
        tcp_rotation_delta_rad,
        min(before_offset_ns, after_offset_ns) / 1e6,
    )
    recorded = (
        float(bundle.metrics.bracket_ms),
        float(bundle.metrics.max_joint_delta_rad),
        float(bundle.metrics.tcp_translation_delta_m),
        float(bundle.metrics.tcp_rotation_delta_rad),
        float(bundle.metrics.selected_robot_state_offset_ms),
    )
    if not np.isfinite(recorded).all() or not np.allclose(
        recorded,
        recomputed,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Occupancy stop-and-capture metrics do not reproduce")
    violations: list[str] = []
    if bracket_ms > config.max_bracket_ms:
        violations.append("robot/camera bracket")
    if joint_delta_rad > config.max_joint_delta_rad:
        violations.append("joint motion")
    if tcp_translation_delta_m > config.max_tcp_translation_delta_m:
        violations.append("TCP translation")
    if tcp_rotation_delta_rad > config.max_tcp_rotation_delta_rad:
        violations.append("TCP rotation")
    if violations:
        raise ValueError(
            "Occupancy source was not captured at a settled robot pose: " + ", ".join(violations)
        )


def _intrinsics_from_context(payload: Mapping[str, Any]) -> CameraIntrinsics:
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


def _rectified_stereo_payload(calibration: Any) -> dict[str, Any]:
    def intrinsics(value: CameraIntrinsics) -> dict[str, Any]:
        return {
            "width": value.width,
            "height": value.height,
            "fx": value.fx,
            "fy": value.fy,
            "cx": value.cx,
            "cy": value.cy,
            "distortion_model": value.distortion_model,
            "distortion_coefficients": list(value.distortion_coefficients),
        }

    return {
        "left": intrinsics(calibration.left),
        "right": intrinsics(calibration.right),
        "right_rectified_T_left_rectified": (
            calibration.right_rectified_t_left_rectified.matrix.tolist()
        ),
        "left_rectified_T_left_ir": calibration.left_rectified_t_left_ir.matrix.tolist(),
        "right_rectified_T_right_ir": (calibration.right_rectified_t_right_ir.matrix.tolist()),
        "disparity_to_depth_q": calibration.disparity_to_depth_q.tolist(),
        "left_valid_roi": list(calibration.left_valid_roi),
        "right_valid_roi": list(calibration.right_valid_roi),
    }


def _require_array_equal(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
    equal_nan: bool = False,
) -> None:
    if actual.dtype != expected.dtype:
        actual = actual.astype(expected.dtype, copy=False)
    if not np.array_equal(actual, expected, equal_nan=equal_nan):
        raise ValueError(f"{label} differs from stored frame evidence")


def _normalized_utc_text(value: Any, *, label: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def _validate_update_chain(
    updates: Sequence[OccupancyFrameUpdate],
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    *,
    replay_depth_rays: bool = True,
) -> None:
    if not occupancy_config.enabled:
        raise ValueError("Occupancy artifact requires enabled occupancy mapping")
    context = updates[0].mapping_context
    pose_contract = _validate_context_configuration(
        context,
        occupancy_config,
        acquisition_config,
    )
    kinematics = Es68KinematicModel.from_resources(
        joint_zero_offsets_rad=pose_contract.joint_zero_offsets_rad,
    )
    expected_views: list[str] = []
    previous: OccupancyFrameUpdate | None = None
    for index, update in enumerate(updates):
        if update.mapping_context != context:
            raise ValueError("All occupancy frames must share one mapping context")
        _validate_configured_geometry(update.mapping_snapshot, occupancy_config)
        _validate_configured_geometry(update.snapshot, occupancy_config)
        expected_views.append(update.evidence.physical_source_id)
        expected_view_tuple = tuple(expected_views)
        if update.evidence.source_sequence_index < 0:
            raise ValueError("Occupancy source sequence index must be non-negative")
        _validate_frame_pose_evidence(
            update.evidence,
            pose_contract,
            kinematics,
            occupancy_config,
        )
        if update.mapping_snapshot.source_view_ids != expected_view_tuple:
            raise ValueError("Occupancy snapshot view order does not match frame evidence")
        if index + 1 < occupancy_config.minimum_source_views:
            if update.snapshot.map_state is not OccupancyMapState.MAPPING:
                raise ValueError("Insufficient-view occupancy prefix must remain MAPPING")
        elif update.snapshot.map_state not in {
            OccupancyMapState.MAP_READY,
            OccupancyMapState.STALE,
        }:
            raise ValueError("Sufficient-view occupancy result must be MAP_READY or STALE")
        if update.snapshot.map_state is OccupancyMapState.STALE and index != len(updates) - 1:
            raise ValueError("Only the final offline occupancy result may be STALE")

        if previous is None:
            if update.mapping_snapshot.sequence != 1:
                raise ValueError("First MAPPING snapshot sequence must be one")
            if update.evidence.previous_evidence_hash is not None:
                raise ValueError("First occupancy evidence must not have a parent")
        else:
            if update.evidence.previous_evidence_hash != previous.evidence.quality_evidence_hash:
                raise ValueError("Occupancy previous evidence hash is not chain-bound")
            if update.mapping_snapshot.sequence != previous.snapshot.sequence + 1:
                raise ValueError("Occupancy snapshot sequence chain is discontinuous")
            if not previous.mapping_snapshot.occupied_indices.issubset(
                update.mapping_snapshot.occupied_indices
            ):
                raise ValueError("Occupied evidence must be monotonic across frames")
            previous_free_counts = dict(previous.mapping_snapshot.free_observation_counts)
            current_free_counts = dict(update.mapping_snapshot.free_observation_counts)
            if any(
                current_free_counts.get(voxel, 0) < count
                for voxel, count in previous_free_counts.items()
            ):
                raise ValueError("Free observation counts must be monotonic across frames")
            if any(
                count - previous_free_counts.get(voxel, 0) not in {0, 1}
                for voxel, count in current_free_counts.items()
            ):
                raise ValueError("One occupancy frame may cast at most one free vote per voxel")
            surviving_free = previous.mapping_snapshot.free_indices.difference(
                update.mapping_snapshot.occupied_indices
            )
            if not surviving_free.issubset(update.mapping_snapshot.free_indices):
                raise ValueError("Free evidence disappeared without an occupied observation")
            if update.mapping_snapshot.created_at_utc < previous.snapshot.created_at_utc:
                raise ValueError("Occupancy frame timestamps are not ordered")
            if (
                update.mapping_snapshot.rebuild_started_at_utc
                != previous.mapping_snapshot.rebuild_started_at_utc
            ):
                raise ValueError("Occupancy rebuild freshness reference changed within one cycle")
        if replay_depth_rays:
            _validate_replayed_mapping(update, previous, occupancy_config)
        previous = update


def _validate_context_configuration(
    context: OccupancyMappingContext,
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
) -> _PoseReplayContract:
    payload = context.to_payload()
    expected_occupancy = occupancy_config.model_dump(mode="json")
    # Schema-7 artifacts written before the CUDA backend selector omit this field.
    # Pydantic correctly supplies the historical CPU behavior, but comparison must
    # retain the exact old payload shape so an immutable pre-change artifact is not
    # mistaken for tampering. New artifacts set and hash-bind the field explicitly.
    context_occupancy = payload.get("occupancy_contract")
    if (
        isinstance(context_occupancy, Mapping)
        and "ray_integration_backend" not in context_occupancy
        and occupancy_config.ray_integration_backend == "cpu"
    ):
        expected_occupancy.pop("ray_integration_backend")
    if context_occupancy != expected_occupancy:
        raise ValueError("Mapping context occupancy configuration mismatch")
    if payload.get("acquisition_contract") != acquisition_config.model_dump(mode="json"):
        raise ValueError("Mapping context acquisition configuration mismatch")
    bounds_min = occupancy_config.workspace_bounds_min_m
    bounds_max = occupancy_config.workspace_bounds_max_m
    if bounds_min is None or bounds_max is None:
        raise ValueError("Occupancy artifact requires measured workspace bounds")
    expected_shape = _configured_grid_shape(occupancy_config)
    grid = payload.get("grid")
    expected_grid = {
        "frame_id": occupancy_config.frame_id,
        "voxel_size_m": occupancy_config.voxel_size_m,
        "origin_m": list(bounds_min),
        "grid_shape": list(expected_shape),
    }
    if grid != expected_grid:
        raise ValueError("Mapping context grid geometry mismatch")

    robot = _mapping(payload, "robot")
    _require_exact_keys(
        robot,
        {
            "model_content_hash",
            "self_mask_excluded_link_names",
            "self_mask_render_backend",
            "joint_zero_offsets_rad",
            "flange_T_tcp",
            "flange_tcp_asset_sha256",
        },
        label="robot",
    )
    robot_model_hash = _sha256_digest(
        robot["model_content_hash"],
        label="robot.model_content_hash",
    )
    flange_tcp_asset_hash = _sha256_digest(
        robot["flange_tcp_asset_sha256"],
        label="robot.flange_tcp_asset_sha256",
    )
    offsets = _finite_six_vector(
        robot["joint_zero_offsets_rad"],
        label="robot.joint_zero_offsets_rad",
    )
    flange_t_tcp = _pose_from_payload(
        robot["flange_T_tcp"],
        parent_frame="flange",
        child_frame="tcp",
        label="robot.flange_T_tcp",
    )

    resources = Es68ModelResources.packaged()
    if _sha256(resources.tcp_offset_json) != flange_tcp_asset_hash:
        raise ValueError("Mapping context flange/TCP asset hash is not the packaged ES68 asset")
    packaged_flange_t_tcp = load_es68_flange_t_tcp(resources)
    _require_pose_close(
        flange_t_tcp,
        packaged_flange_t_tcp,
        label="Mapping context flange_T_tcp",
    )

    hand_eye = _mapping(payload, "hand_eye")
    _require_exact_keys(
        hand_eye,
        {"artifact_sha256", "flange_T_left_ir"},
        label="hand_eye",
    )
    hand_eye_hash = _sha256_digest(
        hand_eye["artifact_sha256"],
        label="hand_eye.artifact_sha256",
    )
    flange_t_left_ir = _pose_from_payload(
        hand_eye["flange_T_left_ir"],
        parent_frame="flange",
        child_frame="left_ir",
        label="hand_eye.flange_T_left_ir",
    )

    rectified_stereo = _mapping(payload, "rectified_stereo")
    left_rectified_t_left_ir = _pose_from_payload(
        rectified_stereo["left_rectified_T_left_ir"],
        parent_frame="left_rectified",
        child_frame="left_ir",
        label="rectified_stereo.left_rectified_T_left_ir",
    )
    return _PoseReplayContract(
        robot_model_hash,
        hand_eye_hash,
        offsets,
        flange_t_tcp,
        flange_t_left_ir,
        left_rectified_t_left_ir,
    )


def _validate_frame_pose_evidence(
    evidence: OccupancyFrameEvidence,
    contract: _PoseReplayContract,
    kinematics: Es68KinematicModel,
    occupancy_config: OccupancyConfig,
) -> None:
    if evidence.robot_model_hash != contract.robot_model_hash:
        raise ValueError("Occupancy frame robot model hash does not match mapping context")
    if evidence.hand_eye_hash != contract.hand_eye_hash:
        raise ValueError("Occupancy frame hand-eye hash does not match mapping context")

    recomputed_base_t_flange = kinematics.base_t_flange(evidence.joint_positions_rad)
    recorded_base_t_flange = PoseSE3(
        "base",
        "flange",
        evidence.base_t_flange_matrix,
    )
    _require_pose_close(
        recorded_base_t_flange,
        recomputed_base_t_flange,
        label="Occupancy frame base_T_flange FK",
    )

    recomputed_base_t_tcp = recomputed_base_t_flange.compose(contract.flange_t_tcp)
    recorded_predicted_base_t_tcp = PoseSE3(
        "base",
        "tcp",
        evidence.predicted_base_t_tcp_matrix,
    )
    _require_pose_close(
        recorded_predicted_base_t_tcp,
        recomputed_base_t_tcp,
        label="Occupancy frame predicted base_T_tcp",
    )
    observed_base_t_tcp = PoseSE3(
        "base",
        "tcp",
        evidence.observed_base_t_tcp_matrix,
    )
    translation_error_m = float(
        np.linalg.norm(recomputed_base_t_tcp.translation_m - observed_base_t_tcp.translation_m)
    )
    rotation_error_deg = _rotation_error_deg(
        recomputed_base_t_tcp.rotation,
        observed_base_t_tcp.rotation,
    )
    if not math.isclose(
        evidence.fk_tcp_translation_error_m,
        translation_error_m,
        rel_tol=0.0,
        abs_tol=_SCALAR_TRANSLATION_ATOL_M,
    ):
        raise ValueError("Occupancy frame FK/TCP translation error does not reproduce")
    if not math.isclose(
        evidence.fk_tcp_rotation_error_deg,
        rotation_error_deg,
        rel_tol=0.0,
        abs_tol=_SCALAR_ROTATION_ATOL_DEG,
    ):
        raise ValueError("Occupancy frame FK/TCP rotation error does not reproduce")
    if (
        translation_error_m > occupancy_config.maximum_fk_tcp_translation_error_m
        or rotation_error_deg > occupancy_config.maximum_fk_tcp_rotation_error_deg
    ):
        raise ValueError("Occupancy FK/TCP evidence exceeds its safety contract")

    recomputed_base_t_camera = recomputed_base_t_flange.compose(contract.flange_t_left_ir).compose(
        contract.left_rectified_t_left_ir.inverse()
    )
    recorded_base_t_camera = PoseSE3(
        "base",
        "left_rectified",
        evidence.base_t_camera_matrix,
    )
    _require_pose_close(
        recorded_base_t_camera,
        recomputed_base_t_camera,
        label="Occupancy frame base_T_camera",
    )


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Mapping context {label} fields are invalid "
            f"(missing={missing}, unexpected={unexpected})"
        )


def _sha256_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Mapping context {label} must be a lowercase SHA-256 digest")
    return value


def _finite_six_vector(
    value: Any,
    *,
    label: str,
) -> tuple[float, float, float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Mapping context {label} must be a six-vector")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"Mapping context {label} must be a finite six-vector")
    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
        float(array[3]),
        float(array[4]),
        float(array[5]),
    )


def _string_tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Mapping context {label} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(
            f"Mapping context {label} must contain unique non-empty strings"
        )
    return result


def _non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Mapping context {label} must be a non-empty string")
    return value.strip()


def _pose_from_payload(
    value: Any,
    *,
    parent_frame: str,
    child_frame: str,
    label: str,
) -> PoseSE3:
    try:
        return PoseSE3(parent_frame, child_frame, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Mapping context {label} is not a finite SE(3) transform") from exc


def _require_pose_close(
    recorded: PoseSE3,
    expected: PoseSE3,
    *,
    label: str,
) -> None:
    translation_error_m = float(np.linalg.norm(recorded.translation_m - expected.translation_m))
    rotation_error_deg = _rotation_error_deg(recorded.rotation, expected.rotation)
    if (
        translation_error_m > _POSE_TRANSLATION_ATOL_M
        or rotation_error_deg > _POSE_ROTATION_ATOL_DEG
    ):
        raise ValueError(
            f"{label} does not reproduce "
            f"(translation={translation_error_m:.9g} m, "
            f"rotation={rotation_error_deg:.9g} deg)"
        )


def _rotation_error_deg(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
) -> float:
    relative = np.asarray(first_rotation, dtype=np.float64).T @ np.asarray(
        second_rotation,
        dtype=np.float64,
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _validate_configured_geometry(
    snapshot: OccupancySnapshot,
    occupancy_config: OccupancyConfig,
) -> None:
    bounds_min = occupancy_config.workspace_bounds_min_m
    if bounds_min is None:
        raise ValueError("Occupancy artifact requires measured workspace bounds")
    if (
        snapshot.frame_id != occupancy_config.frame_id
        or not np.isclose(snapshot.voxel_size_m, occupancy_config.voxel_size_m)
        or not np.allclose(snapshot.origin_m, bounds_min)
        or snapshot.grid_shape != _configured_grid_shape(occupancy_config)
        or snapshot.minimum_free_observations != occupancy_config.minimum_free_observations
        or not np.isclose(
            snapshot.minimum_free_view_translation_m,
            occupancy_config.minimum_free_view_translation_m,
        )
        or not np.isclose(
            snapshot.minimum_free_view_direction_deg,
            occupancy_config.minimum_free_view_direction_deg,
        )
    ):
        raise ValueError("Occupancy snapshot does not match its mapping configuration")


def _configured_grid_shape(config: OccupancyConfig) -> tuple[int, int, int]:
    bounds_min = config.workspace_bounds_min_m
    bounds_max = config.workspace_bounds_max_m
    if bounds_min is None or bounds_max is None:
        raise ValueError("Occupancy mapping bounds are incomplete")
    shape = np.ceil(
        (np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64))
        / config.voxel_size_m
    ).astype(np.int64)
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _validate_replayed_mapping(
    update: OccupancyFrameUpdate,
    previous: OccupancyFrameUpdate | None,
    config: OccupancyConfig,
) -> None:
    context_payload = update.mapping_context.to_payload()
    stereo_payload = context_payload.get("rectified_stereo")
    if not isinstance(stereo_payload, Mapping):
        raise ValueError("Mapping context rectified_stereo must be an object")
    left_payload = stereo_payload.get("left")
    if not isinstance(left_payload, Mapping):
        raise ValueError("Mapping context left intrinsics must be an object")
    intrinsics = CameraIntrinsics(
        int(left_payload["width"]),
        int(left_payload["height"]),
        float(left_payload["fx"]),
        float(left_payload["fy"]),
        float(left_payload["cx"]),
        float(left_payload["cy"]),
        str(left_payload["distortion_model"]),
        tuple(float(value) for value in left_payload["distortion_coefficients"]),
    )
    pose = PoseSE3(
        "base",
        "left_rectified",
        update.evidence.base_t_camera_matrix,
    )
    grid = OccupancyGridSpec(
        frame_id=update.mapping_snapshot.frame_id,
        voxel_size_m=update.mapping_snapshot.voxel_size_m,
        origin_m=update.mapping_snapshot.origin_m,
        grid_shape=update.mapping_snapshot.grid_shape,
    )
    replayed = DepthRayIntegrator(
        grid,
        DepthIntegrationConfig(
            minimum_depth_m=config.minimum_depth_m,
            maximum_depth_m=config.maximum_depth_m,
            pixel_stride=config.integration_stride,
            minimum_valid_rays=1,
            free_space_margin_m=config.free_space_margin_m,
            minimum_free_observations=config.minimum_free_observations,
            minimum_free_view_translation_m=(config.minimum_free_view_translation_m),
            minimum_free_view_direction_deg=(config.minimum_free_view_direction_deg),
            ray_integration_backend=config.ray_integration_backend,
        ),
        mapping_context_hash=update.mapping_context.content_hash,
    ).integrate(
        None if previous is None else previous.snapshot,
        update.source_depth_m,
        intrinsics,
        pose,
        valid_mask=update.integration_valid_mask,
        source_view_id=update.evidence.physical_source_id,
        observed_at_utc=datetime.fromisoformat(update.evidence.captured_at_utc),
    )
    if replayed != update.mapping_snapshot:
        raise ValueError("Stored MAPPING snapshot does not reproduce from frame evidence")


def _evidence_from_payload(raw: Mapping[str, Any]) -> OccupancyFrameEvidence:
    base_t_flange = raw["base_t_flange_matrix"]
    predicted_base_t_tcp = raw["predicted_base_t_tcp_matrix"]
    observed_base_t_tcp = raw["observed_base_t_tcp_matrix"]
    base_t_camera = raw["base_t_camera_matrix"]
    joints = raw["joint_positions_rad"]
    return OccupancyFrameEvidence(
        source_view_id=str(raw["source_view_id"]),
        physical_source_id=str(raw["physical_source_id"]),
        source_sequence_index=int(raw["source_sequence_index"]),
        frame_number=int(raw["frame_number"]),
        captured_at_utc=str(raw["captured_at_utc"]),
        robot_model_hash=str(raw["robot_model_hash"]),
        hand_eye_hash=str(raw["hand_eye_hash"]),
        source_stereo_metadata_sha256=str(raw["source_stereo_metadata_sha256"]),
        source_session_manifest_sha256=str(raw["source_session_manifest_sha256"]),
        source_session_view_metadata_sha256=str(raw["source_session_view_metadata_sha256"]),
        base_t_flange_matrix=tuple(tuple(float(value) for value in row) for row in base_t_flange),
        predicted_base_t_tcp_matrix=tuple(
            tuple(float(value) for value in row) for row in predicted_base_t_tcp
        ),
        observed_base_t_tcp_matrix=tuple(
            tuple(float(value) for value in row) for row in observed_base_t_tcp
        ),
        base_t_camera_matrix=tuple(tuple(float(value) for value in row) for row in base_t_camera),
        joint_positions_rad=tuple(float(value) for value in joints),
        fk_tcp_translation_error_m=float(raw["fk_tcp_translation_error_m"]),
        fk_tcp_rotation_error_deg=float(raw["fk_tcp_rotation_error_deg"]),
        valid_depth_fraction=float(raw["valid_depth_fraction"]),
        stereo_valid_fraction=float(raw["stereo_valid_fraction"]),
        confidence_accepted_fraction=float(raw["confidence_accepted_fraction"]),
        mean_accepted_confidence=float(raw["mean_accepted_confidence"]),
        lr_consistency_threshold_px=float(raw["lr_consistency_threshold_px"]),
        self_mask=RobotSelfMaskReport(**_mapping(raw, "self_mask")),
        mapping_context_hash=str(raw["mapping_context_hash"]),
        previous_evidence_hash=(
            None if raw["previous_evidence_hash"] is None else str(raw["previous_evidence_hash"])
        ),
        mapping_snapshot_content_hash=str(raw["mapping_snapshot_content_hash"]),
        mapping_snapshot_sequence=int(raw["mapping_snapshot_sequence"]),
        mapping_source_view_ids=tuple(str(value) for value in raw["mapping_source_view_ids"]),
        source_depth_content_hash=str(raw["source_depth_content_hash"]),
        stereo_valid_mask_content_hash=str(raw["stereo_valid_mask_content_hash"]),
        stereo_confidence_content_hash=str(raw["stereo_confidence_content_hash"]),
        predicted_robot_depth_content_hash=str(raw["predicted_robot_depth_content_hash"]),
        robot_mask_content_hash=str(raw["robot_mask_content_hash"]),
        integration_valid_mask_content_hash=str(raw["integration_valid_mask_content_hash"]),
        quality_evidence_hash=str(raw["quality_evidence_hash"]),
    )


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    return safe[:80] or "view"


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


def _snapshot_record(path: Path, snapshot: OccupancySnapshot) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "content_hash": snapshot.content_hash,
        "sequence": snapshot.sequence,
        "map_state": snapshot.map_state.value,
    }


def _load_snapshot_record(root: Path, record: Mapping[str, Any]) -> OccupancySnapshot:
    path = _contained(root, str(record["path"]))
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Occupancy snapshot checksum mismatch: {path.name}")
    snapshot = load_occupancy_snapshot(path)
    if (
        snapshot.content_hash != str(record["content_hash"])
        or snapshot.sequence != int(record["sequence"])
        or snapshot.map_state.value != str(record["map_state"])
    ):
        raise ValueError(f"Occupancy snapshot identity mismatch: {path.name}")
    return snapshot


def _source_record(root: Path, filename: str) -> dict[str, str]:
    path = root / filename
    if not path.is_file():
        raise ValueError(f"Occupancy source does not exist: {path}")
    return {"root": str(root), "file": filename, "sha256": _sha256(path)}


def _verify_source(
    record: Mapping[str, Any],
    *,
    expected_filename: str | None = None,
    relocation_root: Path | None = None,
) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Occupancy source escapes its artifact root")
    if expected_filename is not None and relative != Path(expected_filename):
        raise ValueError(f"Occupancy source filename must be {expected_filename!r}")
    expected_hash = str(record["sha256"])
    if path.is_file():
        if _sha256(path) != expected_hash:
            raise ValueError(f"Occupancy source checksum mismatch: {path}")
        return path
    if relocation_root is not None:
        # Stored datasets are commonly mirrored between the eiai acquisition host
        # and the analysis workstation. Preserve the suffix beginning at `data/`
        # and accept it only when the local file is byte-identical to the record.
        source_parts = path.parts
        with suppress(ValueError, StopIteration):
            data_index = source_parts.index("data")
            resolved_root = relocation_root.resolve()
            local_data = next(
                candidate
                for candidate in (resolved_root, *resolved_root.parents)
                if candidate.name == "data"
            )
            candidate = local_data.joinpath(*source_parts[data_index + 1 :]).resolve()
            if candidate.is_file() and _sha256(candidate) == expected_hash:
                return candidate
    raise ValueError(f"Occupancy source checksum mismatch: {path}")


def _contained(root: Path, value: str) -> Path:
    relative = Path(str(value))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Occupancy artifact path escapes its root")
    return path


def _load_array(root: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = _contained(root, str(record["path"]))
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Occupancy frame checksum mismatch: {path.name}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"Occupancy frame manifest mismatch: {path.name}")
    return array


def _validate_array_dtypes(
    *,
    source_depth: np.ndarray,
    stereo_valid: np.ndarray,
    stereo_confidence: np.ndarray,
    predicted_depth: np.ndarray,
    robot_mask: np.ndarray,
    integration_mask: np.ndarray,
) -> None:
    if not np.issubdtype(source_depth.dtype, np.floating):
        raise ValueError("Occupancy source_depth_m array must be floating point")
    if not np.issubdtype(stereo_confidence.dtype, np.floating):
        raise ValueError("Occupancy stereo_confidence array must be floating point")
    if not np.issubdtype(predicted_depth.dtype, np.floating):
        raise ValueError("Occupancy predicted_robot_depth_m array must be floating point")
    if stereo_valid.dtype != np.bool_:
        raise ValueError("Occupancy stereo_valid_mask array must be boolean")
    if robot_mask.dtype != np.bool_:
        raise ValueError("Occupancy robot_mask array must be boolean")
    if integration_mask.dtype != np.bool_:
        raise ValueError("Occupancy integration_valid_mask array must be boolean")


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"Occupancy metadata {key} must be an object")
    return value


def _sequence(parent: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = parent[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Occupancy metadata {key} must be an array")
    return value
