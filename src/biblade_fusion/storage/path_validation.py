"""Immutable, re-derived collision reports for explicit offline view sequences."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from biblade_fusion.calibration import load_cs68_kinematics
from biblade_fusion.core.settings import CollisionConfig
from biblade_fusion.storage.initialization import read_initialization
from biblade_fusion.storage.view_plan import read_view_plan
from biblade_fusion.workflows import (
    ViewSequenceCollisionReport,
    validate_view_sequence_collision,
)

PATH_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredPathValidation:
    report: ViewSequenceCollisionReport
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: str | Path, filename: str) -> dict[str, str]:
    resolved_root = Path(root).resolve()
    path = resolved_root / filename
    if not path.is_file():
        raise ValueError(f"Path-validation source does not exist: {path}")
    return {"root": str(resolved_root), "file": filename, "sha256": _sha256(path)}


def _file_source(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Path-validation source does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _verify_directory_source(record: dict[str, Any]) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Path-validation source escapes its artifact root")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Path-validation source checksum mismatch: {path}")
    return root


def _verify_file_source(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"])).resolve()
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Path-validation source checksum mismatch: {path}")
    return path


def _derive(
    plan: Path,
    initialization: Path,
    kinematics: Path,
    view_ids: tuple[str, ...],
    config: CollisionConfig,
) -> ViewSequenceCollisionReport:
    stored_plan = read_view_plan(plan)
    stored_initialization = read_initialization(initialization)
    expected_initialization = Path(
        str(stored_plan.metadata["source_initialization"])
    ).resolve()
    if expected_initialization != initialization.resolve():
        raise ValueError("View plan does not belong to the supplied initialization")
    return validate_view_sequence_collision(
        stored_plan.result.filtered_plan,
        view_ids,
        stored_initialization.observation.seed_joint_positions_rad,
        load_cs68_kinematics(kinematics),
        stored_initialization.hand_eye,
        config,
    )


def write_path_validation(
    output_dir: str | Path,
    ordered_view_ids: tuple[str, ...],
    config: CollisionConfig,
    *,
    source_plan: str | Path,
    source_initialization: str | Path,
    source_kinematics: str | Path,
) -> Path:
    """Validate and persist an explicit sequence; never authorize execution."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Path-validation output already exists: {output}")
    plan = Path(source_plan).resolve()
    initialization = Path(source_initialization).resolve()
    kinematics = Path(source_kinematics).resolve()
    report = _derive(plan, initialization, kinematics, ordered_view_ids, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload = {
            "schema_version": PATH_VALIDATION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "sources": {
                "view_plan": _source(plan, "view_plan.json"),
                "initialization": _source(initialization, "metadata.json"),
                "kinematics": _file_source(kinematics),
            },
            "configuration": config.model_dump(mode="json"),
            "ordered_view_ids": list(ordered_view_ids),
            "report": asdict(report),
        }
        (temporary / "path_validation.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_path_validation(path: str | Path) -> StoredPathValidation:
    """Verify sources and re-run the full continuous collision validation."""

    root = Path(path)
    try:
        payload = json.loads((root / "path_validation.json").read_text(encoding="utf-8"))
        if int(payload["schema_version"]) != PATH_VALIDATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("Path validation must explicitly forbid motion")
        sources = payload["sources"]
        plan = _verify_directory_source(sources["view_plan"])
        initialization = _verify_directory_source(sources["initialization"])
        kinematics = _verify_file_source(sources["kinematics"])
        config = CollisionConfig.model_validate(payload["configuration"])
        view_ids = tuple(str(value) for value in payload["ordered_view_ids"])
        report = _derive(plan, initialization, kinematics, view_ids, config)
        normalized = json.loads(json.dumps(asdict(report), allow_nan=False))
        if payload["report"] != normalized:
            raise ValueError("Path-validation report does not match its sources")
        return StoredPathValidation(report, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid path-validation artifact {root}: {exc}") from exc
