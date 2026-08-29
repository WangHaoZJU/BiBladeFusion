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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

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
from biblade_fusion.planning import BladeSide
from biblade_fusion.storage.coarse_model import read_coarse_model_summary
from biblade_fusion.storage.coverage import read_coverage_ledger
from biblade_fusion.storage.occupancy_mapping import read_occupancy_mapping
from biblade_fusion.storage.reconstructed_view import (
    StoredReconstructedBladeView,
    read_reconstructed_view,
)
from biblade_fusion.storage.stereo_inference import (
    read_stereo_inference,
    verify_stereo_inference_source,
)
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash

COARSE_SCAN_VIEW_SCHEMA_VERSION = 1
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
    foreground: BootstrapForegroundResult
    target_view_id: str
    target_kind: CoarseTargetKind
    target_side: BladeSide
    metadata: dict[str, Any]

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class StoredCoarseScanGeneration:
    root: Path
    generation_index: int
    views: tuple[StoredCoarseScanView, ...]
    coverage_path: Path
    previous_generation_path: Path | None
    coarse_model_path: Path | None
    metadata: dict[str, Any]

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
    plan = read_view_plan(view_plan_path).result.geometric_plan
    replayed = create_coverage_ledger(plan, stored.ledger.config)
    for item, observation_id in zip(views, expected_ids, strict=True):
        replayed = update_coverage(
            replayed,
            plan,
            initialization.observation.proxy,
            item.reconstructed.view.base_cloud,
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
        read_occupancy_mapping(occupancy_root)
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
) -> BootstrapForegroundResult:
    stereo = read_stereo_inference(stereo_root)
    source_session = Path(str(stereo.metadata["source"]["session"])).resolve()
    verify_stereo_inference_source(stereo, expected_session=source_session)
    integration_valid = _load_final_integration_mask(occupancy_root)
    if integration_valid.shape != stereo.observation.depth_m.shape:
        raise ValueError("Coarse integration-valid mask shape changed")
    return bootstrap_blade_foreground(
        stereo.observation.rectified.left_ir,
        stereo.observation.depth_m,
        integration_valid,
        config,
        seed,
    )


def write_coarse_scan_view(
    output_dir: str | Path,
    foreground: BootstrapForegroundResult,
    *,
    reconstructed_view: str | Path,
    source_stereo_inference: str | Path,
    source_occupancy_mapping: str | Path,
    target_view_id: str,
    target_kind: CoarseTargetKind,
    target_side: BladeSide,
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
    replayed = _replay_foreground(
        stereo_root=stereo_root,
        occupancy_root=occupancy_root,
        config=foreground.config,
        seed=foreground.seed,
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
    occupancy = read_occupancy_mapping(occupancy_root)
    evidence = occupancy.frame_evidence[-1]
    integration_valid = _load_final_integration_mask(occupancy_root)
    if (
        evidence.source_view_id != reconstructed.view.source_view_id
        or evidence.source_sequence_index != reconstructed.view.source_sequence_index
        or evidence.frame_number != reconstructed.view.source_frame_number
        or evidence.integration_valid_mask_content_hash
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
                "algorithm": BOOTSTRAP_FOREGROUND_ALGORITHM,
                "config": asdict(foreground.config),
                "seed": bootstrap_seed_payload(foreground.seed),
                "policy_sha256": foreground.policy_sha256,
                "diagnostics": asdict(foreground.diagnostics),
                "input_content_sha256": {
                    "left_rectified": foreground.left_image_content_sha256,
                    "depth_m": foreground.depth_content_sha256,
                    "integration_valid_mask": foreground.valid_mask_content_sha256,
                },
            },
            "files": {
                "mask": _array_record(temporary / "mask.npy"),
                "seed_mask": _array_record(temporary / "seed_mask.npy"),
            },
            "sources": {
                "reconstructed_view": _directory_record(reconstructed_root, "metadata.json"),
                "stereo_inference": _directory_record(stereo_root, "metadata.json"),
                "occupancy_mapping": _directory_record(occupancy_root, "metadata.json"),
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_coarse_scan_view(path: str | Path) -> StoredCoarseScanView:
    """Independently replay and verify a coarse reconstruction evidence wrapper."""

    root = Path(path).resolve()
    try:
        payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if (
            int(payload["schema_version"]) != COARSE_SCAN_VIEW_SCHEMA_VERSION
            or payload.get("artifact_kind") != COARSE_SCAN_VIEW_KIND
            or payload.get("motion_authorized") is not False
        ):
            raise ValueError("unsupported or motion-authorized coarse-scan view")
        files = payload["files"]
        if set(files) != {"mask", "seed_mask"}:
            raise ValueError("coarse-scan view file set changed")
        mask = np.asarray(_load_array(root, files["mask"]), dtype=np.bool_)
        seed_mask = np.asarray(_load_array(root, files["seed_mask"]), dtype=np.bool_)
        sources = payload["sources"]
        reconstructed_root = _resolve_directory_record(sources["reconstructed_view"])
        stereo_root = _resolve_directory_record(sources["stereo_inference"])
        occupancy_root = _resolve_directory_record(sources["occupancy_mapping"])
        foreground_payload = payload["foreground"]
        config = BootstrapForegroundConfig(**foreground_payload["config"])
        seed = _seed_from_payload(foreground_payload["seed"])
        if foreground_payload["policy_sha256"] != bootstrap_policy_sha256(config, seed):
            raise ValueError("coarse bootstrap policy changed")
        replayed = _replay_foreground(
            stereo_root=stereo_root,
            occupancy_root=occupancy_root,
            config=config,
            seed=seed,
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
        reconstructed = read_reconstructed_view(reconstructed_root)
        identity = payload["identity"]
        if (
            identity["view_id"] != reconstructed.view.source_view_id
            or int(identity["sequence_index"]) != reconstructed.view.source_sequence_index
            or int(identity["frame_number"]) != reconstructed.view.source_frame_number
            or not np.array_equal(mask, reconstructed.blade_mask)
        ):
            raise ValueError("coarse reconstructed-view binding changed")
        target = payload["target"]
        if str(target["kind"]) not in _COARSE_TARGET_KINDS:
            raise ValueError("coarse target kind is unsupported")
        return StoredCoarseScanView(
            root,
            reconstructed,
            replayed,
            str(target["view_id"]),
            str(target["kind"]),  # type: ignore[arg-type]
            BladeSide(str(target["side"])),
            payload,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coarse-scan view artifact {root}: {exc}") from exc


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
    stored_views = tuple(read_coarse_scan_view(path) for path in views)
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
    coverage_asset = read_coverage_ledger(coverage_root)
    _assert_coverage_replays(
        views=stored_views,
        coverage_path=coverage_root,
        initialization_path=initialization_root,
        view_plan_path=view_plan_root,
    )
    previous_path = Path(previous_generation).resolve() if previous_generation else None
    if previous_path is None:
        generation_index = 0
    else:
        previous = read_coarse_scan_generation(previous_path)
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
        if (
            Path(str(previous_sources["initialization"]["root"])).resolve()
            != initialization_root
            or Path(str(previous_sources["view_plan"]["root"])).resolve()
            != view_plan_root
            or Path(str(previous_sources["discovery_plan"]["root"])).resolve()
            != discovery_plan_root
        ):
            raise ValueError("Coarse-scan generation changed a bound planning source")
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
    coarse_root = Path(coarse_model).resolve() if coarse_model is not None else None
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
            "previous_generation": (
                _directory_record(previous_path, "generation.json")
                if previous_path is not None
                else None
            ),
            "sources": {
                "initialization": _directory_record(initialization_root, "initialization.json"),
                "view_plan": _directory_record(view_plan_root, "view_plan.json"),
                "discovery_plan": _directory_record(
                    discovery_plan_root,
                    "discovery.json",
                ),
                "coverage": _directory_record(coverage_root, "coverage.json"),
                "coarse_model": (
                    _directory_record(coarse_root, "metadata.json")
                    if coarse_root is not None
                    else None
                ),
            },
            "views": [_directory_record(item.root, "metadata.json") for item in stored_views],
            "summary": {
                "view_count": len(stored_views),
                "front_view_count": sum(
                    item.target_side is BladeSide.FRONT for item in stored_views
                ),
                "back_view_count": sum(item.target_side is BladeSide.BACK for item in stored_views),
                "schema5_ready": coarse_root is not None,
            },
        }
        (temporary / "generation.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_coarse_scan_generation(path: str | Path) -> StoredCoarseScanGeneration:
    """Verify a generation, its exact predecessor and every immutable source."""

    root = Path(path).resolve()
    try:
        payload = json.loads((root / "generation.json").read_text(encoding="utf-8"))
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
            read_coarse_scan_view(_resolve_directory_record(record)) for record in payload["views"]
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
            for source_name in ("initialization", "view_plan", "discovery_plan"):
                if (
                    Path(str(previous_sources[source_name]["root"])).resolve()
                    != Path(str(sources[source_name]["root"])).resolve()
                ):
                    raise ValueError("coarse generation changed its proxy source")
            previous_coverage = Path(str(previous_sources["coverage"]["root"])).resolve()
            if phase_transition:
                if coarse_path is None or previous_sources["coarse_model"] is not None:
                    raise ValueError("invalid view-preserving coarse phase transition")
                if coverage_path != previous_coverage:
                    raise ValueError("schema-5 transition changed proxy coverage")
            elif Path(str(coverage.metadata["previous_ledger"])).resolve() != previous_coverage:
                raise ValueError("coarse coverage predecessor differs from view predecessor")
        return StoredCoarseScanGeneration(
            root,
            generation_index,
            views,
            coverage_path,
            previous_path,
            coarse_path,
            payload,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coarse-scan generation {root}: {exc}") from exc
