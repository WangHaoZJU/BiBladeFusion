"""Immutable evidence chain for the unknown-blade coarse-science phase.

The ordinary reconstructed-view artifact intentionally predates an unknown-object
foreground contract.  This module wraps it with the exact bootstrap mask and the
occupancy integration-valid mask that produced that foreground.  A generation is
append-only: recovery always names one exact predecessor and never discovers a
mutable ``latest`` directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.perception.bootstrap_foreground import (
    BOOTSTRAP_FOREGROUND_ALGORITHM,
    BootstrapForegroundConfig,
    BootstrapForegroundResult,
    BootstrapSeed,
    array_content_sha256,
    bootstrap_blade_foreground,
    bootstrap_policy_sha256,
    bootstrap_seed_payload,
)
from biblade_fusion.perception.coarse_foreground import (
    PROJECTED_COARSE_FOREGROUND_ALGORITHM,
    ProjectedCoarseForegroundGuide,
    ProjectedCoarseForegroundResult,
    projected_coarse_blade_foreground,
    projected_coarse_foreground_policy_sha256,
)
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import ProxySupportSelection, select_proxy_support
from biblade_fusion.planning import BladeSide
from biblade_fusion.storage.coarse_model import read_coarse_model_summary
from biblade_fusion.storage.coverage import read_coverage_ledger
from biblade_fusion.storage.initialization import INITIALIZATION_METADATA_FILENAME
from biblade_fusion.storage.occupancy_mapping import (
    _assert_occupancy_storage_authorities_current,
    _bind_occupancy_storage_authority,
    _OccupancyStorageAuthority,
    read_occupancy_mapping,
    read_occupancy_mapping_for_replay,
)
from biblade_fusion.storage.reconstructed_view import (
    StoredReconstructedBladeView,
    read_reconstructed_view,
)
from biblade_fusion.storage.stereo_inference import (
    read_stereo_inference,
    verify_stereo_inference_source,
)
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash

COARSE_SCAN_VIEW_SCHEMA_VERSION = 3
COARSE_SCAN_GENERATION_SCHEMA_VERSION = 1
COARSE_SCAN_VIEW_KIND = "biblade_fusion.coarse_scan_view"
COARSE_SCAN_GENERATION_KIND = "biblade_fusion.coarse_scan_generation"

CoarseTargetKind = Literal[
    "operator_seed",
    "proxy_normal",
    "fin_discovery_major_negative",
    "fin_discovery_major_positive",
    "fin_discovery_minor_negative",
    "fin_discovery_minor_positive",
]
CoarseForegroundResult = BootstrapForegroundResult | ProjectedCoarseForegroundResult
_COARSE_TARGET_KINDS = frozenset(
    {
        "operator_seed",
        "proxy_normal",
        "fin_discovery_major_negative",
        "fin_discovery_major_positive",
        "fin_discovery_minor_negative",
        "fin_discovery_minor_positive",
    }
)


@dataclass(frozen=True, slots=True)
class StoredCoarseScanView:
    root: Path
    reconstructed: StoredReconstructedBladeView
    foreground: CoarseForegroundResult
    target_view_id: str
    target_kind: CoarseTargetKind
    target_side: BladeSide
    proxy_support: ProxySupportSelection
    proxy_config: ProxyModelConfig
    metadata: dict[str, Any]
    metadata_sha256: str
    metadata_size_bytes: int
    occupancy_storage_authority: _OccupancyStorageAuthority | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def motion_authorized(self) -> bool:
        return False

    @property
    def support_cloud(self) -> PointCloud:
        """Return the exact per-view cloud allowed to influence coarse geometry."""

        cloud = self.reconstructed.view.base_cloud
        mask = self.proxy_support.mask
        return PointCloud(
            cloud.frame,
            cloud.points_m[mask],
            cloud.pixel_uv[mask],
            cloud.source_image_shape,
        )


@dataclass(frozen=True, slots=True)
class _CoarseScanViewReadback:
    """Transaction-local proof for read-only reuse of one strict view read.

    The token is intentionally private and short lived.  It is not a cache and
    is never persisted or accepted by a motion/science decision boundary.  The
    immutable scalar records let a read-only consumer close the TOCTOU window
    without replaying occupancy rays a second time.
    """

    view: StoredCoarseScanView
    root: Path
    metadata_sha256: str
    metadata_size_bytes: int
    source_records: tuple[tuple[str, str, str, str, int], ...]


@dataclass(frozen=True, slots=True)
class StoredCoarseScanGeneration:
    root: Path
    generation_index: int
    views: tuple[StoredCoarseScanView, ...]
    coverage_path: Path
    previous_generation_path: Path | None
    coarse_model_path: Path | None
    metadata: dict[str, Any]
    metadata_sha256: str
    metadata_size_bytes: int

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class StoredCoarseIntegrationSource:
    """Final integration-valid mask from one fully verified occupancy asset."""

    mask: np.ndarray
    source_view_id: str
    source_sequence_index: int
    frame_number: int
    occupancy_content_sha256: str

    def __post_init__(self) -> None:
        mask = np.array(self.mask, dtype=np.bool_, copy=True)
        if mask.ndim != 2:
            raise ValueError("Coarse integration-valid mask must be two-dimensional")
        if not self.source_view_id or self.source_sequence_index < 0 or self.frame_number < 0:
            raise ValueError("Coarse integration source identity is invalid")
        if len(self.occupancy_content_sha256) != 64:
            raise ValueError("Coarse integration source hash is malformed")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


_DirectoryAuthorityIdentity = tuple[str, str, str, int]


class _StrictReadContext:
    """Memoize immutable authorities only within one top-level strict read.

    Projected coarse views bind their predecessor generation.  Since each newer
    generation also lists every accepted historical view, naively following those
    bindings replays the same view, occupancy, and predecessor generation many
    times.  This context is created by the outermost public reader and discarded
    when that call returns; no result can therefore survive a later disk change.
    """

    def __init__(self) -> None:
        self.views: dict[_DirectoryAuthorityIdentity, StoredCoarseScanView] = {}
        self.generations: dict[
            _DirectoryAuthorityIdentity,
            StoredCoarseScanGeneration,
        ] = {}
        self.views_in_progress: set[_DirectoryAuthorityIdentity] = set()
        self.generations_in_progress: set[_DirectoryAuthorityIdentity] = set()
        self.expected_views: dict[Path, _DirectoryAuthorityIdentity] = {}
        self.expected_generations: dict[Path, _DirectoryAuthorityIdentity] = {}
        self.occupancy_authorities: dict[
            _DirectoryAuthorityIdentity,
            _OccupancyStorageAuthority,
        ] = {}


_STRICT_READ_CONTEXT: ContextVar[_StrictReadContext | None] = ContextVar(
    "coarse_scan_strict_read_context",
    default=None,
)


def _enter_strict_read_context() -> tuple[_StrictReadContext, Token | None]:
    context = _STRICT_READ_CONTEXT.get()
    if context is not None:
        return context, None
    context = _StrictReadContext()
    return context, _STRICT_READ_CONTEXT.set(context)


@contextmanager
def _strict_coarse_read_transaction() -> Iterator[None]:
    """Share strict-read proofs within one caller-defined read transaction."""

    _, context_token = _enter_strict_read_context()
    try:
        yield
    finally:
        if context_token is not None:
            _STRICT_READ_CONTEXT.reset(context_token)


def _authority_identity_from_bytes(
    root: Path,
    *,
    authority: str,
    content: bytes,
) -> _DirectoryAuthorityIdentity:
    return (
        str(root.resolve()),
        authority,
        hashlib.sha256(content).hexdigest(),
        len(content),
    )


def _coverage_observation_ids(
    views: tuple[StoredCoarseScanView, ...],
) -> tuple[str, ...]:
    """Reconstruct the stable identity of every accepted physical frame."""

    # Keep this import local so the storage re-export graph stays acyclic.
    from biblade_fusion.planning.coverage import coverage_observation_id

    identities: list[str] = []
    for item in views:
        source = item.reconstructed.metadata["source"]
        identities.append(
            coverage_observation_id(
                source["session"],
                item.reconstructed.view.source_view_id,
                item.reconstructed.view.source_sequence_index,
                item.reconstructed.view.source_frame_number,
            )
        )
    return tuple(identities)


def _assert_coverage_replays(
    *,
    views: tuple[StoredCoarseScanView, ...],
    coverage_path: Path,
    initialization_path: Path,
    view_plan_path: Path,
) -> None:
    """Recompute proxy coverage from the exact generation instead of trusting counts."""

    from biblade_fusion.planning.coverage import (
        create_coverage_ledger,
        update_coverage,
    )
    from biblade_fusion.storage.initialization import read_initialization
    from biblade_fusion.storage.view_plan import read_view_plan

    stored = read_coverage_ledger(coverage_path)
    metadata = stored.metadata
    if Path(str(metadata["source_plan"])).resolve() != view_plan_path:
        raise ValueError("Coarse coverage view-plan source differs from generation")
    if Path(str(metadata["source_initialization"])).resolve() != initialization_path:
        raise ValueError("Coarse coverage initialization source differs from generation")
    expected_ids = _coverage_observation_ids(views)
    if stored.ledger.observation_ids != expected_ids:
        raise ValueError("Coarse coverage physical observation identities differ from views")

    initialization = read_initialization(initialization_path)
    proxy_config = ProxyModelConfig.model_validate(
        initialization.metadata["processing"]["proxy_model"]
    )
    plan = read_view_plan(view_plan_path).result.geometric_plan
    replayed = create_coverage_ledger(plan, stored.ledger.config)
    for item, observation_id in zip(views, expected_ids, strict=True):
        if (
            int(item.metadata.get("schema_version", 1))
            == COARSE_SCAN_VIEW_SCHEMA_VERSION
            and item.proxy_config != proxy_config
        ):
            raise ValueError("Coarse-view proxy-support configuration differs from initialization")
        replayed = update_coverage(
            replayed,
            plan,
            initialization.observation.proxy,
            item.support_cloud,
            item.reconstructed.view.base_t_projection_camera,
            observation_id,
        )
    if (
        replayed.rows != stored.ledger.rows
        or replayed.columns != stored.ledger.columns
        or replayed.config != stored.ledger.config
        or replayed.observation_ids != stored.ledger.observation_ids
        or replayed.completed_patch_ids != stored.ledger.completed_patch_ids
        or len(replayed.patches) != len(stored.ledger.patches)
    ):
        raise ValueError("Coarse coverage ledger differs from deterministic replay")
    for expected, actual in zip(replayed.patches, stored.ledger.patches, strict=True):
        if (
            expected.patch_id != actual.patch_id
            or expected.side is not actual.side
            or expected.row != actual.row
            or expected.column != actual.column
            or expected.observation_ids != actual.observation_ids
            or not np.array_equal(expected.bin_point_counts, actual.bin_point_counts)
        ):
            raise ValueError("Coarse coverage patch differs from deterministic replay")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_record(path: str | Path, filename: str) -> dict[str, Any]:
    root = Path(path).resolve()
    authority = root / filename
    if not root.is_dir() or not authority.is_file():
        raise ValueError(f"Coarse-scan source is missing: {authority}")
    return {
        "root": str(root),
        "authority": filename,
        "sha256": _sha256(authority),
        "size_bytes": authority.stat().st_size,
    }


def _resolve_directory_record(record: dict[str, Any]) -> Path:
    raw = Path(str(record["root"]))
    root = raw.resolve()
    relative = Path(str(record["authority"]))
    authority = (root / relative).resolve()
    if (
        not raw.is_absolute()
        or raw != root
        or relative.is_absolute()
        or not authority.is_relative_to(root)
        or not authority.is_file()
        or _sha256(authority) != str(record["sha256"])
        or authority.stat().st_size != int(record["size_bytes"])
    ):
        raise ValueError("Coarse-scan directory source changed")
    return root


def _resolve_bound_directory_record(
    record: dict[str, Any],
    *,
    expected_authority: str,
) -> tuple[Path, _DirectoryAuthorityIdentity]:
    """Resolve and fingerprint one record before a context-local cached read."""

    root = _resolve_directory_record(record)
    authority = Path(str(record["authority"]))
    if authority != Path(expected_authority):
        raise ValueError(
            f"Coarse-scan directory source authority must be {expected_authority}"
        )
    return root, (
        str(root),
        expected_authority,
        str(record["sha256"]),
        int(record["size_bytes"]),
    )


def _read_bound_coarse_scan_view(
    record: dict[str, Any],
    context: _StrictReadContext,
) -> StoredCoarseScanView:
    root, identity = _resolve_bound_directory_record(
        record,
        expected_authority="metadata.json",
    )
    previous = context.expected_views.get(root)
    if previous is not None and previous != identity:
        raise ValueError("One coarse view has conflicting authority identities")
    context.expected_views[root] = identity
    try:
        return read_coarse_scan_view(root)
    finally:
        if previous is None:
            context.expected_views.pop(root, None)
        else:
            context.expected_views[root] = previous


def _expect_bound_generation(
    record: dict[str, Any],
    context: _StrictReadContext,
) -> tuple[Path, _DirectoryAuthorityIdentity | None]:
    """Stage one exact predecessor identity for `_replay_foreground`."""

    root, identity = _resolve_bound_directory_record(
        record,
        expected_authority="generation.json",
    )
    previous = context.expected_generations.get(root)
    if previous is not None and previous != identity:
        raise ValueError("One coarse generation has conflicting authority identities")
    context.expected_generations[root] = identity
    return root, previous


def _restore_expected_generation(
    root: Path,
    previous: _DirectoryAuthorityIdentity | None,
    context: _StrictReadContext,
) -> None:
    if previous is None:
        context.expected_generations.pop(root, None)
    else:
        context.expected_generations[root] = previous


def _stored_view_authority_records(
    view: StoredCoarseScanView,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if type(view) is not StoredCoarseScanView:
        raise ValueError("Coarse view authority requires a typed strict-reader result")
    record = {
        "root": str(view.root.resolve()),
        "authority": "metadata.json",
        "sha256": view.metadata_sha256,
        "size_bytes": view.metadata_size_bytes,
    }
    _resolve_directory_record(record)
    payload = json.loads((view.root / "metadata.json").read_text(encoding="utf-8"))
    if payload != view.metadata:
        raise ValueError("Coarse view metadata differs from its strict readback")
    source_names = ["reconstructed_view", "stereo_inference", "occupancy_mapping"]
    if "foreground_reference_generation" in view.metadata["sources"]:
        source_names.append("foreground_reference_generation")
    sources = tuple(dict(view.metadata["sources"][name]) for name in source_names)
    for source in sources:
        _resolve_directory_record(source)
    return record, sources


def _stored_generation_authority_record(
    generation: StoredCoarseScanGeneration,
) -> dict[str, Any]:
    if type(generation) is not StoredCoarseScanGeneration:
        raise ValueError("Coarse generation authority requires a typed strict-reader result")
    record = {
        "root": str(generation.root.resolve()),
        "authority": "generation.json",
        "sha256": generation.metadata_sha256,
        "size_bytes": generation.metadata_size_bytes,
    }
    _resolve_directory_record(record)
    payload = json.loads((generation.root / "generation.json").read_text(encoding="utf-8"))
    if payload != generation.metadata:
        raise ValueError("Coarse generation metadata differs from its strict readback")
    return record


def _recheck_cached_view_integrity(
    view: StoredCoarseScanView,
    *,
    verify_occupancy: bool = True,
) -> None:
    """Recheck subordinate checksums without repeating CUDA ray integration."""

    _stored_view_authority_records(view)
    for record in view.metadata["files"].values():
        _load_array(view.root, record)
    reconstructed_root = _resolve_directory_record(
        view.metadata["sources"]["reconstructed_view"]
    )
    _recheck_subordinate_array_records(reconstructed_root)
    stereo_root = _resolve_directory_record(
        view.metadata["sources"]["stereo_inference"]
    )
    _recheck_subordinate_array_records(stereo_root)
    occupancy_root = _resolve_directory_record(
        view.metadata["sources"]["occupancy_mapping"]
    )
    if not verify_occupancy:
        return
    authority = _bound_view_occupancy_storage_authority(view)
    if authority is None:
        read_occupancy_mapping_for_replay(occupancy_root)
    else:
        _assert_occupancy_storage_authorities_current((authority,))


def _recheck_subordinate_array_records(root: Path) -> None:
    """Verify one authority's declared arrays without rebuilding derived products."""

    payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    files = payload["files"]
    if not isinstance(files, dict):
        raise ValueError("Subordinate artifact array manifest must be an object")
    for record in files.values():
        _load_array(root, record)


def _bound_view_occupancy_storage_authority(
    view: StoredCoarseScanView,
) -> _OccupancyStorageAuthority | None:
    authority = view.occupancy_storage_authority
    if authority is None:
        return None
    root, identity = _resolve_bound_directory_record(
        view.metadata["sources"]["occupancy_mapping"],
        expected_authority="metadata.json",
    )
    if (
        authority.root != root
        or authority.metadata_sha256 != identity[2]
        or authority.metadata_size_bytes != identity[3]
    ):
        raise ValueError("Coarse view occupancy storage authority changed")
    return authority


def _recheck_cached_generation_authorities(
    generation: StoredCoarseScanGeneration,
) -> None:
    """Recheck the immutable authority roots before reusing a typed generation."""

    _stored_generation_authority_record(generation)
    payload = generation.metadata
    previous = payload["previous_generation"]
    if previous is not None:
        _resolve_directory_record(previous)
    sources = payload["sources"]
    for name in (
        "initialization",
        "view_plan",
        "discovery_plan",
        "coverage",
        "coarse_model",
    ):
        record = sources[name]
        if record is not None:
            _resolve_directory_record(record)
    occupancy_authorities: list[_OccupancyStorageAuthority] = []
    fallback_occupancy_roots: list[Path] = []
    for view in generation.views:
        _recheck_cached_view_integrity(view, verify_occupancy=False)
        authority = _bound_view_occupancy_storage_authority(view)
        if authority is None:
            fallback_occupancy_roots.append(
                _resolve_directory_record(
                    view.metadata["sources"]["occupancy_mapping"]
                )
            )
        else:
            occupancy_authorities.append(authority)
    _assert_occupancy_storage_authorities_current(tuple(occupancy_authorities))
    for root in fallback_occupancy_roots:
        read_occupancy_mapping_for_replay(root)


def _bind_coarse_scan_view_readback(
    view: StoredCoarseScanView,
) -> _CoarseScanViewReadback:
    """Bind a strict view result to exact on-disk authorities for one transaction."""

    if type(view) is not StoredCoarseScanView:
        raise ValueError("Coarse view readback requires a typed strict-reader result")
    root = view.root.resolve()
    metadata_record, sources = _stored_view_authority_records(view)
    frozen_sources: list[tuple[str, str, str, str, int]] = []
    source_names = ["reconstructed_view", "stereo_inference", "occupancy_mapping"]
    if "foreground_reference_generation" in view.metadata["sources"]:
        source_names.append("foreground_reference_generation")
    for name, source in zip(source_names, sources, strict=True):
        record = dict(source)
        source_root = _resolve_directory_record(record)
        frozen_sources.append(
            (
                name,
                str(source_root),
                str(record["authority"]),
                str(record["sha256"]),
                int(record["size_bytes"]),
            )
        )
    return _CoarseScanViewReadback(
        view=view,
        root=root,
        metadata_sha256=str(metadata_record["sha256"]),
        metadata_size_bytes=int(metadata_record["size_bytes"]),
        source_records=tuple(frozen_sources),
    )


def _revalidate_coarse_scan_view_readback(
    readback: _CoarseScanViewReadback,
    *,
    expected_root: str | Path,
) -> StoredCoarseScanView:
    """Recheck a read-only reuse token without semantic occupancy replay."""

    if type(readback) is not _CoarseScanViewReadback:
        raise ValueError("Coarse view reuse requires a typed transaction readback")
    view = readback.view
    root = Path(expected_root).resolve()
    if (
        type(view) is not StoredCoarseScanView
        or root != readback.root
        or view.root.resolve() != root
    ):
        raise ValueError("Coarse view reuse root differs from its transaction readback")
    metadata_path = root / "metadata.json"
    if (
        not metadata_path.is_file()
        or _sha256(metadata_path) != readback.metadata_sha256
        or metadata_path.stat().st_size != readback.metadata_size_bytes
    ):
        raise ValueError("Coarse view metadata changed after transaction readback")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload != view.metadata:
        raise ValueError("Coarse view metadata no longer matches its typed readback")
    sources = payload["sources"]
    for name, source_root, authority, sha256, size_bytes in readback.source_records:
        frozen = {
            "root": source_root,
            "authority": authority,
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        if dict(sources[name]) != frozen:
            raise ValueError("Coarse view source binding changed after transaction readback")
        _resolve_directory_record(frozen)
    return view


def _array_record(path: Path) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    finally:
        del value


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(str(record["path"]))
    path = (root.resolve() / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Coarse-scan array escapes its artifact")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError("Coarse-scan array checksum mismatch")
    value = np.load(path, allow_pickle=False)
    if str(value.dtype) != str(record["dtype"]) or list(value.shape) != record["shape"]:
        raise ValueError("Coarse-scan array manifest mismatch")
    return value


def _seed_from_payload(payload: Any) -> BootstrapSeed | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Coarse bootstrap seed must be an object or null")
    return BootstrapSeed(
        kind=str(payload["kind"]),  # type: ignore[arg-type]
        mode=str(payload["mode"]),  # type: ignore[arg-type]
        vertices_uv=tuple(
            (float(vertex[0]), float(vertex[1])) for vertex in payload["vertices_uv"]
        ),
    )


def _load_final_integration_mask(
    occupancy_root: Path,
    *,
    verify_occupancy: bool = True,
) -> np.ndarray:
    """Load the last integration mask only after full occupancy verification."""

    if verify_occupancy:
        occupancy = read_occupancy_mapping(occupancy_root)
        authority = _bind_occupancy_storage_authority(occupancy_root, occupancy)
        context = _STRICT_READ_CONTEXT.get()
        if context is not None:
            identity: _DirectoryAuthorityIdentity = (
                str(authority.root),
                "metadata.json",
                authority.metadata_sha256,
                authority.metadata_size_bytes,
            )
            context.occupancy_authorities[identity] = authority
    payload = json.loads((occupancy_root / "metadata.json").read_text(encoding="utf-8"))
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("Coarse-scan occupancy source has no frames")
    return np.asarray(
        _load_array(occupancy_root, frames[-1]["files"]["integration_valid_mask"]),
        dtype=np.bool_,
    )


def read_coarse_integration_source(
    occupancy_mapping: str | Path,
) -> StoredCoarseIntegrationSource:
    """Expose the current frame's science mask only after motion-grade verification."""

    root = Path(occupancy_mapping).resolve()
    occupancy = read_occupancy_mapping(root)
    evidence = occupancy.frame_evidence[-1]
    mask = _load_final_integration_mask(root, verify_occupancy=False)
    if occupancy_array_content_hash(mask) != evidence.integration_valid_mask_content_hash:
        raise ValueError("Coarse integration-valid mask differs from occupancy evidence")
    return StoredCoarseIntegrationSource(
        mask,
        evidence.source_view_id,
        evidence.source_sequence_index,
        evidence.frame_number,
        evidence.integration_valid_mask_content_hash,
    )


def _replay_foreground(
    *,
    stereo_root: Path,
    occupancy_root: Path,
    config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
    algorithm: str = BOOTSTRAP_FOREGROUND_ALGORITHM,
    guide: ProjectedCoarseForegroundGuide | None = None,
    reconstructed: StoredReconstructedBladeView | None = None,
    verified_integration: StoredCoarseIntegrationSource | None = None,
) -> CoarseForegroundResult:
    stereo = read_stereo_inference(stereo_root)
    source_session = Path(str(stereo.metadata["source"]["session"])).resolve()
    verify_stereo_inference_source(stereo, expected_session=source_session)
    if verified_integration is None:
        integration_valid = _load_final_integration_mask(occupancy_root)
    else:
        integration_valid = verified_integration.mask
    if integration_valid.shape != stereo.observation.depth_m.shape:
        raise ValueError("Coarse integration-valid mask shape changed")
    if algorithm == BOOTSTRAP_FOREGROUND_ALGORITHM:
        if guide is not None:
            raise ValueError("Operator bootstrap foreground cannot carry projected guidance")
        return bootstrap_blade_foreground(
            stereo.observation.rectified.left_ir,
            stereo.observation.depth_m,
            integration_valid,
            config,
            seed,
        )
    if algorithm != PROJECTED_COARSE_FOREGROUND_ALGORITHM:
        raise ValueError("Unsupported coarse foreground algorithm")
    if seed is not None or guide is None or reconstructed is None:
        raise ValueError("Projected coarse foreground evidence is incomplete")
    generation = read_coarse_scan_generation(guide.source_generation_path)
    if generation.metadata_sha256 != guide.source_generation_metadata_sha256:
        raise ValueError("Projected coarse source generation changed")
    reference_points = np.vstack(
        [item.support_cloud.points_m for item in generation.views]
    )
    if array_content_sha256(reference_points) != guide.reference_points_content_sha256:
        raise ValueError("Projected coarse reference points changed")
    return projected_coarse_blade_foreground(
        stereo.observation.rectified.left_ir,
        stereo.observation.depth_m,
        integration_valid,
        config,
        intrinsics=reconstructed.view.planning_intrinsics,
        base_t_left_rectified=reconstructed.view.base_t_projection_camera,
        reference_points_base_m=reference_points,
        guide=guide,
    )


def write_coarse_scan_view(
    output_dir: str | Path,
    foreground: CoarseForegroundResult,
    *,
    reconstructed_view: str | Path,
    source_stereo_inference: str | Path,
    source_occupancy_mapping: str | Path,
    target_view_id: str,
    target_kind: CoarseTargetKind,
    target_side: BladeSide,
    proxy_config: ProxyModelConfig,
) -> Path:
    """Bind one coarse reconstruction to its replayable unknown-object mask."""

    target_view_id = str(target_view_id).strip()
    if not target_view_id:
        raise ValueError("Coarse target view ID must be non-empty")
    if target_kind not in _COARSE_TARGET_KINDS:
        raise ValueError("Coarse target kind is unsupported")
    reconstructed_root = Path(reconstructed_view).resolve()
    stereo_root = Path(source_stereo_inference).resolve()
    occupancy_root = Path(source_occupancy_mapping).resolve()
    reconstructed = read_reconstructed_view(reconstructed_root)
    proxy_support = select_proxy_support(
        reconstructed.view.base_cloud.points_m,
        proxy_config,
        frame=reconstructed.view.base_cloud.frame,
    )
    integration = read_coarse_integration_source(occupancy_root)
    occupancy_source_record = _directory_record(occupancy_root, "metadata.json")
    replayed = _replay_foreground(
        stereo_root=stereo_root,
        occupancy_root=occupancy_root,
        config=foreground.config,
        seed=foreground.seed,
        algorithm=foreground.algorithm,
        guide=(
            foreground.guide
            if isinstance(foreground, ProjectedCoarseForegroundResult)
            else None
        ),
        reconstructed=reconstructed,
        verified_integration=integration,
    )
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
    if any(getattr(foreground, name) != getattr(replayed, name) for name in scalar_fields):
        raise ValueError("Coarse bootstrap foreground does not replay")
    if isinstance(foreground, ProjectedCoarseForegroundResult) and (
        not isinstance(replayed, ProjectedCoarseForegroundResult)
        or foreground.guide != replayed.guide
    ):
        raise ValueError("Projected coarse foreground guide does not replay")
    if not np.array_equal(foreground.mask, replayed.mask) or not np.array_equal(
        foreground.seed_mask, replayed.seed_mask
    ):
        raise ValueError("Coarse bootstrap foreground arrays do not replay")
    if not np.array_equal(reconstructed.blade_mask, foreground.mask):
        raise ValueError("Coarse reconstructed view does not use its bootstrap foreground")
    source = reconstructed.metadata["source"]
    if (
        Path(str(source["stereo_inference"])).resolve() != stereo_root
        or str(source["view_id"]) == ""
    ):
        raise ValueError("Coarse reconstructed-view stereo source changed")
    integration_valid = integration.mask
    if (
        integration.source_view_id != reconstructed.view.source_view_id
        or integration.source_sequence_index != reconstructed.view.source_sequence_index
        or integration.frame_number != reconstructed.view.source_frame_number
        or integration.occupancy_content_sha256
        != occupancy_array_content_hash(integration_valid)
        or foreground.valid_mask_content_sha256 != array_content_sha256(integration_valid)
    ):
        raise ValueError("Coarse reconstruction and occupancy frame identities differ")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Coarse-scan view output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        np.save(temporary / "mask.npy", foreground.mask, allow_pickle=False)
        np.save(temporary / "seed_mask.npy", foreground.seed_mask, allow_pickle=False)
        np.save(
            temporary / "proxy_support_mask.npy",
            proxy_support.mask,
            allow_pickle=False,
        )
        foreground_reference_record = None
        if isinstance(foreground, ProjectedCoarseForegroundResult):
            foreground_reference_record = _directory_record(
                foreground.guide.source_generation_path,
                "generation.json",
            )
        payload = {
            "schema_version": COARSE_SCAN_VIEW_SCHEMA_VERSION,
            "artifact_kind": COARSE_SCAN_VIEW_KIND,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "target": {
                "view_id": target_view_id,
                "kind": target_kind,
                "side": target_side.value,
            },
            "identity": {
                "view_id": reconstructed.view.source_view_id,
                "sequence_index": reconstructed.view.source_sequence_index,
                "frame_number": reconstructed.view.source_frame_number,
            },
            "foreground": {
                "algorithm": foreground.algorithm,
                "config": asdict(foreground.config),
                "seed": bootstrap_seed_payload(foreground.seed),
                "guide": (
                    foreground.guide.payload()
                    if isinstance(foreground, ProjectedCoarseForegroundResult)
                    else None
                ),
                "policy_sha256": foreground.policy_sha256,
                "diagnostics": asdict(foreground.diagnostics),
                "input_content_sha256": {
                    "left_rectified": foreground.left_image_content_sha256,
                    "depth_m": foreground.depth_content_sha256,
                    "integration_valid_mask": foreground.valid_mask_content_sha256,
                },
            },
            "proxy_support": {
                "configuration": proxy_config.model_dump(mode="json"),
                "diagnostics": proxy_support.metadata_payload(),
            },
            "files": {
                "mask": _array_record(temporary / "mask.npy"),
                "seed_mask": _array_record(temporary / "seed_mask.npy"),
                "proxy_support_mask": _array_record(
                    temporary / "proxy_support_mask.npy"
                ),
            },
            "sources": {
                "reconstructed_view": _directory_record(reconstructed_root, "metadata.json"),
                "stereo_inference": _directory_record(stereo_root, "metadata.json"),
                "occupancy_mapping": occupancy_source_record,
                **(
                    {"foreground_reference_generation": foreground_reference_record}
                    if foreground_reference_record is not None
                    else {}
                ),
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rechecked_integration = read_coarse_integration_source(occupancy_root)
        if (
            rechecked_integration.source_view_id != integration.source_view_id
            or rechecked_integration.source_sequence_index
            != integration.source_sequence_index
            or rechecked_integration.frame_number != integration.frame_number
            or rechecked_integration.occupancy_content_sha256
            != integration.occupancy_content_sha256
            or not np.array_equal(rechecked_integration.mask, integration.mask)
        ):
            raise ValueError("Coarse occupancy integration source changed before publication")
        _resolve_directory_record(occupancy_source_record)
        if foreground_reference_record is not None:
            _resolve_directory_record(foreground_reference_record)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_coarse_scan_view(path: str | Path) -> StoredCoarseScanView:
    """Independently replay and verify a coarse reconstruction evidence wrapper."""

    read_context, context_token = _enter_strict_read_context()
    root = Path(path).resolve()
    try:
        metadata_bytes = (root / "metadata.json").read_bytes()
        authority_identity = _authority_identity_from_bytes(
            root,
            authority="metadata.json",
            content=metadata_bytes,
        )
        expected_identity = read_context.expected_views.get(root)
        if expected_identity is not None and authority_identity != expected_identity:
            raise ValueError("Coarse view authority changed after its directory binding")
        cached = read_context.views.get(authority_identity)
        if cached is not None:
            _recheck_cached_view_integrity(cached)
            return cached
        if authority_identity in read_context.views_in_progress:
            raise ValueError("Coarse view authority graph is cyclic")
        read_context.views_in_progress.add(authority_identity)
        payload = json.loads(metadata_bytes.decode("utf-8"))
        if (
            int(payload["schema_version"]) not in {1, 2, COARSE_SCAN_VIEW_SCHEMA_VERSION}
            or payload.get("artifact_kind") != COARSE_SCAN_VIEW_KIND
            or payload.get("motion_authorized") is not False
        ):
            raise ValueError("unsupported or motion-authorized coarse-scan view")
        files = payload["files"]
        schema_version = int(payload["schema_version"])
        expected_files = (
            {"mask", "seed_mask", "proxy_support_mask"}
            if schema_version >= 2
            else {"mask", "seed_mask"}
        )
        if set(files) != expected_files:
            raise ValueError("coarse-scan view file set changed")
        mask = np.asarray(_load_array(root, files["mask"]), dtype=np.bool_)
        seed_mask = np.asarray(_load_array(root, files["seed_mask"]), dtype=np.bool_)
        sources = payload["sources"]
        reconstructed_root = _resolve_directory_record(sources["reconstructed_view"])
        stereo_root = _resolve_directory_record(sources["stereo_inference"])
        occupancy_root = _resolve_directory_record(sources["occupancy_mapping"])
        reconstructed = read_reconstructed_view(reconstructed_root)
        foreground_payload = payload["foreground"]
        config = BootstrapForegroundConfig(**foreground_payload["config"])
        seed = _seed_from_payload(foreground_payload["seed"])
        algorithm = str(foreground_payload["algorithm"])
        guide = None
        if algorithm == BOOTSTRAP_FOREGROUND_ALGORITHM:
            expected_policy = bootstrap_policy_sha256(config, seed)
            if foreground_payload.get("guide") is not None:
                raise ValueError("Operator bootstrap unexpectedly carries a projected guide")
            if "foreground_reference_generation" in sources:
                raise ValueError("Operator bootstrap unexpectedly binds a reference generation")
        elif algorithm == PROJECTED_COARSE_FOREGROUND_ALGORITHM:
            if schema_version != COARSE_SCAN_VIEW_SCHEMA_VERSION or seed is not None:
                raise ValueError("Projected coarse foreground requires schema 3 and no seed")
            guide_payload = foreground_payload.get("guide")
            if not isinstance(guide_payload, dict):
                raise ValueError("Projected coarse foreground guide is missing")
            reference_root, previous_expected_generation = _expect_bound_generation(
                sources["foreground_reference_generation"],
                read_context,
            )
            guide = ProjectedCoarseForegroundGuide(
                source_generation_path=reference_root,
                source_generation_metadata_sha256=str(
                    guide_payload["source_generation_metadata_sha256"]
                ),
                reference_points_content_sha256=str(
                    guide_payload["reference_points_content_sha256"]
                ),
                blade_envelope_min_m=tuple(guide_payload["blade_envelope_min_m"]),
                blade_envelope_max_m=tuple(guide_payload["blade_envelope_max_m"]),
            )
            if str(guide_payload["source_generation_path"]) != str(reference_root):
                raise ValueError("Projected coarse source path differs from its directory record")
            expected_policy = projected_coarse_foreground_policy_sha256(config)
        else:
            raise ValueError("unsupported coarse foreground algorithm")
        if foreground_payload["policy_sha256"] != expected_policy:
            raise ValueError("coarse foreground policy changed")
        try:
            replayed = _replay_foreground(
                stereo_root=stereo_root,
                occupancy_root=occupancy_root,
                config=config,
                seed=seed,
                algorithm=algorithm,
                guide=guide,
                reconstructed=reconstructed,
            )
        finally:
            if algorithm == PROJECTED_COARSE_FOREGROUND_ALGORITHM:
                _restore_expected_generation(
                    reference_root,
                    previous_expected_generation,
                    read_context,
                )
        if (
            foreground_payload["diagnostics"] != asdict(replayed.diagnostics)
            or foreground_payload["input_content_sha256"]
            != {
                "left_rectified": replayed.left_image_content_sha256,
                "depth_m": replayed.depth_content_sha256,
                "integration_valid_mask": replayed.valid_mask_content_sha256,
            }
            or not np.array_equal(mask, replayed.mask)
            or not np.array_equal(seed_mask, replayed.seed_mask)
        ):
            raise ValueError("coarse bootstrap foreground no longer replays")
        identity = payload["identity"]
        if (
            identity["view_id"] != reconstructed.view.source_view_id
            or int(identity["sequence_index"]) != reconstructed.view.source_sequence_index
            or int(identity["frame_number"]) != reconstructed.view.source_frame_number
            or not np.array_equal(mask, reconstructed.blade_mask)
        ):
            raise ValueError("coarse reconstructed-view binding changed")
        if schema_version >= 2:
            proxy_payload = payload["proxy_support"]
            proxy_config = ProxyModelConfig.model_validate(proxy_payload["configuration"])
            proxy_support_mask = np.asarray(
                _load_array(root, files["proxy_support_mask"]),
                dtype=np.bool_,
            )
            proxy_support = select_proxy_support(
                reconstructed.view.base_cloud.points_m,
                proxy_config,
                frame=reconstructed.view.base_cloud.frame,
            )
            if (
                not np.array_equal(proxy_support_mask, proxy_support.mask)
                or proxy_payload["diagnostics"] != proxy_support.metadata_payload()
            ):
                raise ValueError("coarse proxy support no longer replays")
        else:
            proxy_config = ProxyModelConfig()
            proxy_support = select_proxy_support(
                reconstructed.view.base_cloud.points_m,
                proxy_config,
                frame=reconstructed.view.base_cloud.frame,
            )
        target = payload["target"]
        if str(target["kind"]) not in _COARSE_TARGET_KINDS:
            raise ValueError("coarse target kind is unsupported")
        occupancy_storage_authority = read_context.occupancy_authorities.get(
            (
                str(occupancy_root),
                "metadata.json",
                str(sources["occupancy_mapping"]["sha256"]),
                int(sources["occupancy_mapping"]["size_bytes"]),
            )
        )
        if occupancy_storage_authority is not None:
            _assert_occupancy_storage_authorities_current(
                (occupancy_storage_authority,)
            )
        stored = StoredCoarseScanView(
            root,
            reconstructed,
            replayed,
            str(target["view_id"]),
            str(target["kind"]),  # type: ignore[arg-type]
            BladeSide(str(target["side"])),
            proxy_support,
            proxy_config,
            payload,
            hashlib.sha256(metadata_bytes).hexdigest(),
            len(metadata_bytes),
            occupancy_storage_authority,
        )
        read_context.views[authority_identity] = stored
        read_context.views_in_progress.remove(authority_identity)
        return stored
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coarse-scan view artifact {root}: {exc}") from exc
    finally:
        if context_token is not None:
            _STRICT_READ_CONTEXT.reset(context_token)


def write_coarse_scan_generation(
    output_dir: str | Path,
    *,
    views: tuple[str | Path, ...],
    coverage: str | Path,
    source_initialization: str | Path,
    source_view_plan: str | Path,
    source_discovery_plan: str | Path,
    previous_generation: str | Path | None = None,
    coarse_model: str | Path | None = None,
) -> Path:
    """Persist one append-only coarse view list and its exact proxy coverage."""

    if not views:
        raise ValueError("Coarse-scan generation requires at least one view")
    previous_path = Path(previous_generation).resolve() if previous_generation else None
    stored_views = tuple(read_coarse_scan_view(path) for path in views)
    previous = (
        read_coarse_scan_generation(previous_path)
        if previous_path is not None
        else None
    )
    output, _stored = _write_coarse_scan_generation_from_verified(
        output_dir,
        views=views,
        verified_views=stored_views,
        coverage=coverage,
        source_initialization=source_initialization,
        source_view_plan=source_view_plan,
        source_discovery_plan=source_discovery_plan,
        previous_generation=previous_generation,
        verified_previous_generation=previous,
        coarse_model=coarse_model,
    )
    return output


def _write_coarse_scan_generation_from_verified(
    output_dir: str | Path,
    *,
    views: tuple[str | Path, ...],
    verified_views: tuple[StoredCoarseScanView, ...],
    coverage: str | Path,
    source_initialization: str | Path,
    source_view_plan: str | Path,
    source_discovery_plan: str | Path,
    previous_generation: str | Path | None,
    verified_previous_generation: StoredCoarseScanGeneration | None,
    coarse_model: str | Path | None = None,
) -> tuple[Path, StoredCoarseScanGeneration]:
    """Write from strict readers already completed in this append transaction.

    This is deliberately private and accepts typed storage objects, not arbitrary
    decoded dictionaries.  It only removes repeated semantic replay inside one
    append call; public readers and all later checkpoint/resume boundaries remain
    independent full verifications.
    """

    expected_view_roots = tuple(Path(path).resolve() for path in views)
    actual_view_roots = tuple(item.root.resolve() for item in verified_views)
    if not verified_views or actual_view_roots != expected_view_roots:
        raise ValueError("Verified coarse views do not match generation sources")
    previous_path = Path(previous_generation).resolve() if previous_generation else None
    if (verified_previous_generation is None) != (previous_path is None):
        raise ValueError("Verified predecessor presence differs from generation source")
    if (
        verified_previous_generation is not None
        and verified_previous_generation.root.resolve() != previous_path
    ):
        raise ValueError("Verified predecessor root differs from generation source")

    stored_views = verified_views
    physical_ids = tuple(
        (
            item.reconstructed.view.source_view_id,
            item.reconstructed.view.source_sequence_index,
            item.reconstructed.view.source_frame_number,
        )
        for item in stored_views
    )
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("Coarse-scan generation contains a duplicate physical frame")
    initialization_root = Path(source_initialization).resolve()
    view_plan_root = Path(source_view_plan).resolve()
    discovery_plan_root = Path(source_discovery_plan).resolve()
    coverage_root = Path(coverage).resolve()
    coarse_root = Path(coarse_model).resolve() if coarse_model is not None else None
    # Capture every top-level authority before the strict readers below.  The
    # same records are rechecked immediately before publication, so a source
    # cannot be replaced in the read-to-record gap and silently become the new
    # authority for an old typed object.
    initialization_record = _directory_record(
        initialization_root,
        INITIALIZATION_METADATA_FILENAME,
    )
    view_plan_record = _directory_record(view_plan_root, "view_plan.json")
    discovery_plan_record = _directory_record(discovery_plan_root, "discovery.json")
    coverage_record = _directory_record(coverage_root, "coverage.json")
    previous_record = (
        _stored_generation_authority_record(verified_previous_generation)
        if verified_previous_generation is not None
        else None
    )
    coarse_record = (
        _directory_record(coarse_root, "metadata.json")
        if coarse_root is not None
        else None
    )
    view_authorities = tuple(
        _stored_view_authority_records(item) for item in stored_views
    )
    view_records = tuple(item[0] for item in view_authorities)
    view_source_records = tuple(item[1] for item in view_authorities)
    coverage_asset = read_coverage_ledger(coverage_root)
    _assert_coverage_replays(
        views=stored_views,
        coverage_path=coverage_root,
        initialization_path=initialization_root,
        view_plan_path=view_plan_root,
    )
    if previous_path is None:
        generation_index = 0
    else:
        assert verified_previous_generation is not None
        previous = verified_previous_generation
        generation_index = previous.generation_index + 1
        previous_roots = tuple(item.root for item in previous.views)
        current_roots = tuple(item.root for item in stored_views)
        view_append = current_roots[:-1] == previous_roots
        phase_transition = current_roots == previous_roots
        if not view_append and not phase_transition:
            raise ValueError("Coarse-scan generation is not an append-only successor")
        if phase_transition and (coarse_model is None or previous.coarse_model_path is not None):
            raise ValueError(
                "A view-preserving coarse generation must be the one-way schema-5 transition"
            )
        previous_sources = previous.metadata["sources"]
        initialization_changed = (
            Path(str(previous_sources["initialization"]["root"])).resolve()
            != initialization_root
        )
        view_plan_changed = (
            Path(str(previous_sources["view_plan"]["root"])).resolve()
            != view_plan_root
        )
        discovery_changed = (
            Path(str(previous_sources["discovery_plan"]["root"])).resolve()
            != discovery_plan_root
        )
        if initialization_changed or view_plan_changed:
            raise ValueError("Coarse-scan generation changed a bound planning source")
        # A newly appended physical view may bind a new immutable IK evaluation
        # from its latest stopped posture.  A view-preserving schema-5 transition,
        # however, must retain the exact discovery revision of its predecessor.
        if phase_transition and discovery_changed:
            raise ValueError("Schema-5 transition changed its discovery revision")
        previous_coverage = previous.coverage_path
        coverage_predecessor = coverage_asset.metadata["previous_ledger"]
        if phase_transition:
            # A phase-only transition must preserve the exact existing ledger;
            # no new predecessor relation is introduced.
            if coverage_root != previous_coverage:
                raise ValueError("Schema-5 transition changed proxy coverage")
        elif (
            coverage_predecessor is None
            or Path(str(coverage_predecessor)).resolve() != previous_coverage
        ):
            raise ValueError("Coarse coverage predecessor differs from generation predecessor")
    if coarse_root is not None:
        summary = read_coarse_model_summary(coarse_root)
        source_roots = tuple(
            Path(item["path"]).resolve() for item in summary.metadata["source_views"]
        )
        # Compare through the persisted wrapper sources; ``view`` is typed data, not a path.
        wrapper_roots = tuple(
            Path(item.metadata["sources"]["reconstructed_view"]["root"]).resolve()
            for item in stored_views
        )
        if source_roots != wrapper_roots:
            raise ValueError("Schema-5 coarse model is not built from this exact generation")
        support = summary.metadata.get("proxy_support")
        support_roots = (
            tuple(
                Path(item["path"]).resolve()
                for item in support["source_coarse_views"]
            )
            if support is not None
            else ()
        )
        if support_roots != tuple(item.root for item in stored_views):
            raise ValueError("Schema-5 coarse model lacks exact per-view proxy support")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Coarse-scan generation already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload = {
            "schema_version": COARSE_SCAN_GENERATION_SCHEMA_VERSION,
            "artifact_kind": COARSE_SCAN_GENERATION_KIND,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "generation_index": generation_index,
            "previous_generation": previous_record,
            "sources": {
                "initialization": initialization_record,
                "view_plan": view_plan_record,
                "discovery_plan": discovery_plan_record,
                "coverage": coverage_record,
                "coarse_model": coarse_record,
            },
            "views": list(view_records),
            "summary": {
                "view_count": len(stored_views),
                "front_view_count": sum(
                    item.target_side is BladeSide.FRONT for item in stored_views
                ),
                "back_view_count": sum(item.target_side is BladeSide.BACK for item in stored_views),
                "schema5_ready": coarse_root is not None,
            },
        }
        metadata_bytes = (
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        (temporary / "generation.json").write_bytes(metadata_bytes)
        # Close the transaction-local TOCTOU window without replaying occupancy
        # rays again: every typed source was fully verified above, and its bound
        # authority must still have the exact path, digest and size immediately
        # before the append-only generation is published.
        records = (
            initialization_record,
            view_plan_record,
            discovery_plan_record,
            coverage_record,
            *view_records,
        )
        for record in records:
            _resolve_directory_record(record)
        for source_records in view_source_records:
            for record in source_records:
                _resolve_directory_record(record)
        if previous_record is not None:
            _resolve_directory_record(previous_record)
        if coarse_record is not None:
            _resolve_directory_record(coarse_record)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    root = output.resolve()
    return root, StoredCoarseScanGeneration(
        root=root,
        generation_index=generation_index,
        views=stored_views,
        coverage_path=coverage_root,
        previous_generation_path=previous_path,
        coarse_model_path=coarse_root,
        metadata=payload,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        metadata_size_bytes=len(metadata_bytes),
    )


def read_coarse_scan_generation(path: str | Path) -> StoredCoarseScanGeneration:
    """Verify a generation, its exact predecessor and every immutable source."""

    read_context, context_token = _enter_strict_read_context()
    root = Path(path).resolve()
    try:
        metadata_bytes = (root / "generation.json").read_bytes()
        authority_identity = _authority_identity_from_bytes(
            root,
            authority="generation.json",
            content=metadata_bytes,
        )
        expected_identity = read_context.expected_generations.get(root)
        if expected_identity is not None and authority_identity != expected_identity:
            raise ValueError(
                "Coarse generation authority changed after its directory binding"
            )
        cached = read_context.generations.get(authority_identity)
        if cached is not None:
            _recheck_cached_generation_authorities(cached)
            return cached
        if authority_identity in read_context.generations_in_progress:
            raise ValueError("Coarse generation authority graph is cyclic")
        read_context.generations_in_progress.add(authority_identity)
        payload = json.loads(metadata_bytes.decode("utf-8"))
        if (
            int(payload["schema_version"]) != COARSE_SCAN_GENERATION_SCHEMA_VERSION
            or payload.get("artifact_kind") != COARSE_SCAN_GENERATION_KIND
            or payload.get("motion_authorized") is not False
        ):
            raise ValueError("unsupported or motion-authorized coarse generation")
        generation_index = int(payload["generation_index"])
        previous_record = payload["previous_generation"]
        previous_path = (
            _resolve_directory_record(previous_record) if previous_record is not None else None
        )
        if (generation_index == 0) != (previous_path is None):
            raise ValueError("coarse generation predecessor/index mismatch")
        previous_payload = None
        if previous_path is not None:
            previous_payload = json.loads(
                (previous_path / "generation.json").read_text(encoding="utf-8")
            )
            if int(previous_payload["generation_index"]) != generation_index - 1:
                raise ValueError("coarse generation predecessor is not consecutive")
        sources = payload["sources"]
        initialization_path = _resolve_directory_record(sources["initialization"])
        view_plan_path = _resolve_directory_record(sources["view_plan"])
        _resolve_directory_record(sources["discovery_plan"])
        coverage_path = _resolve_directory_record(sources["coverage"])
        coverage = read_coverage_ledger(coverage_path)
        views = tuple(
            _read_bound_coarse_scan_view(record, read_context)
            for record in payload["views"]
        )
        _assert_coverage_replays(
            views=views,
            coverage_path=coverage_path,
            initialization_path=initialization_path,
            view_plan_path=view_plan_path,
        )
        # Import lazily to keep the storage/workflow package re-export graph acyclic.
        from biblade_fusion.storage.initialization import read_initialization

        proxy = read_initialization(initialization_path).observation.proxy
        proxy_t_base = proxy.frame_T_proxy.inverse()
        for item in views:
            local_camera = proxy_t_base.transform_points(
                item.reconstructed.view.base_t_projection_camera.translation_m
            )
            if abs(float(local_camera[2])) <= 1e-9:
                raise ValueError("coarse generation camera lies on the proxy mid-plane")
            actual_side = BladeSide.FRONT if local_camera[2] > 0.0 else BladeSide.BACK
            if item.target_side is not actual_side:
                raise ValueError("coarse generation target side disagrees with camera geometry")
        physical_ids = tuple(
            (
                item.reconstructed.view.source_view_id,
                item.reconstructed.view.source_sequence_index,
                item.reconstructed.view.source_frame_number,
            )
            for item in views
        )
        if (
            len(set(physical_ids)) != len(physical_ids)
            or int(payload["summary"]["view_count"]) != len(views)
            or int(payload["summary"]["front_view_count"])
            != sum(item.target_side is BladeSide.FRONT for item in views)
            or int(payload["summary"]["back_view_count"])
            != sum(item.target_side is BladeSide.BACK for item in views)
        ):
            raise ValueError("coarse generation summary or physical identities changed")
        coarse_record = sources["coarse_model"]
        coarse_path = (
            _resolve_directory_record(coarse_record) if coarse_record is not None else None
        )
        if coarse_path is not None:
            coarse = read_coarse_model_summary(coarse_path)
            source_roots = tuple(
                Path(record["path"]).resolve() for record in coarse.metadata["source_views"]
            )
            wrapper_roots = tuple(
                Path(item.metadata["sources"]["reconstructed_view"]["root"]).resolve()
                for item in views
            )
            if source_roots != wrapper_roots:
                raise ValueError("coarse model sources differ from its scan generation")
            support = coarse.metadata.get("proxy_support")
            support_roots = (
                tuple(
                    Path(record["path"]).resolve()
                    for record in support["source_coarse_views"]
                )
                if support is not None
                else ()
            )
            if support_roots != tuple(item.root for item in views):
                raise ValueError("coarse model proxy support differs from its scan generation")
        if bool(payload["summary"]["schema5_ready"]) != (coarse_path is not None):
            raise ValueError("coarse generation schema-5 readiness changed")
        if previous_payload is not None:
            previous_view_roots = tuple(
                Path(record["root"]).resolve() for record in previous_payload["views"]
            )
            current_view_roots = tuple(item.root for item in views)
            view_append = current_view_roots[:-1] == previous_view_roots
            phase_transition = current_view_roots == previous_view_roots
            if not view_append and not phase_transition:
                raise ValueError("coarse generation no longer appends its predecessor")
            previous_sources = previous_payload["sources"]
            for source_name in ("initialization", "view_plan"):
                if (
                    Path(str(previous_sources[source_name]["root"])).resolve()
                    != Path(str(sources[source_name]["root"])).resolve()
                ):
                    raise ValueError("coarse generation changed its proxy source")
            discovery_changed = (
                Path(str(previous_sources["discovery_plan"]["root"])).resolve()
                != Path(str(sources["discovery_plan"]["root"])).resolve()
            )
            previous_coverage = Path(str(previous_sources["coverage"]["root"])).resolve()
            if phase_transition:
                if coarse_path is None or previous_sources["coarse_model"] is not None:
                    raise ValueError("invalid view-preserving coarse phase transition")
                if discovery_changed:
                    raise ValueError("schema-5 transition changed its discovery revision")
                if coverage_path != previous_coverage:
                    raise ValueError("schema-5 transition changed proxy coverage")
            elif Path(str(coverage.metadata["previous_ledger"])).resolve() != previous_coverage:
                raise ValueError("coarse coverage predecessor differs from view predecessor")
        stored = StoredCoarseScanGeneration(
            root,
            generation_index,
            views,
            coverage_path,
            previous_path,
            coarse_path,
            payload,
            hashlib.sha256(metadata_bytes).hexdigest(),
            len(metadata_bytes),
        )
        read_context.generations[authority_identity] = stored
        read_context.generations_in_progress.remove(authority_identity)
        return stored
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coarse-scan generation {root}: {exc}") from exc
    finally:
        if context_token is not None:
            _STRICT_READ_CONTEXT.reset(context_token)
