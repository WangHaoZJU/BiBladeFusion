"""Immutable, re-derived HoloRobot motion-preflight artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from biblade_fusion.core.settings import MotionPreflightConfig
from biblade_fusion.robotics import Cs68PinocchioCollisionChecker
from biblade_fusion.storage.initialization import read_initialization
from biblade_fusion.storage.view_plan import read_view_plan
from biblade_fusion.workflows import (
    ViewSequenceMotionPreflight,
    preflight_view_sequence_motion,
)

MOTION_PREFLIGHT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredMotionPreflight:
    report: ViewSequenceMotionPreflight
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_source(root: str | Path, filename: str) -> dict[str, str]:
    resolved = Path(root).resolve()
    path = resolved / filename
    if not path.is_file():
        raise ValueError(f"Motion-preflight source does not exist: {path}")
    return {"root": str(resolved), "file": filename, "sha256": _sha256(path)}


def _verify_source(record: dict[str, Any]) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Motion-preflight source escapes its artifact root")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Motion-preflight source checksum mismatch: {path}")
    return root


def _derive(
    plan: Path,
    initialization: Path,
    ordered_view_ids: tuple[str, ...],
    config: MotionPreflightConfig,
) -> ViewSequenceMotionPreflight:
    stored_plan = read_view_plan(plan)
    stored_initialization = read_initialization(initialization)
    expected_initialization = Path(
        str(stored_plan.metadata["source_initialization"])
    ).resolve()
    if expected_initialization != initialization.resolve():
        raise ValueError("View plan does not belong to the supplied initialization")
    if stored_plan.metadata.get("source_kinematics") is None:
        raise ValueError(
            "View plan lacks controller-kinematics provenance; regenerate the plan"
        )
    checker = Cs68PinocchioCollisionChecker.from_resources()
    return preflight_view_sequence_motion(
        stored_plan.result.filtered_plan,
        ordered_view_ids,
        stored_initialization.observation.seed_joint_positions_rad,
        config,
        collision_checker=checker,
    )


def write_motion_preflight(
    output_dir: str | Path,
    ordered_view_ids: tuple[str, ...],
    config: MotionPreflightConfig,
    *,
    source_plan: str | Path,
    source_initialization: str | Path,
) -> Path:
    """Persist a self-collision/ServoJ preflight that never authorizes execution."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Motion-preflight output already exists: {output}")
    plan = Path(source_plan).resolve()
    initialization = Path(source_initialization).resolve()
    report = _derive(plan, initialization, ordered_view_ids, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload = {
            "schema_version": MOTION_PREFLIGHT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "ready_for_approval": report.ready_for_approval,
            "sources": {
                "view_plan": _directory_source(plan, "view_plan.json"),
                "initialization": _directory_source(
                    initialization, "metadata.json"
                ),
            },
            "configuration": config.model_dump(mode="json"),
            "ordered_view_ids": list(ordered_view_ids),
            "report": asdict(report),
        }
        (temporary / "motion_preflight.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def read_motion_preflight(path: str | Path) -> StoredMotionPreflight:
    """Verify bound inputs and re-run the full motion preflight."""

    root = Path(path)
    try:
        payload = json.loads(
            (root / "motion_preflight.json").read_text(encoding="utf-8")
        )
        if int(payload["schema_version"]) != MOTION_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if payload.get("motion_authorized") is not False:
            raise ValueError("Motion preflight must explicitly forbid motion")
        sources = payload["sources"]
        plan = _verify_source(sources["view_plan"])
        initialization = _verify_source(sources["initialization"])
        config = MotionPreflightConfig.model_validate(payload["configuration"])
        ordered_view_ids = tuple(str(value) for value in payload["ordered_view_ids"])
        report = _derive(plan, initialization, ordered_view_ids, config)
        normalized = json.loads(json.dumps(asdict(report), allow_nan=False))
        if payload["report"] != normalized:
            raise ValueError("Motion-preflight report does not match its sources")
        if bool(payload.get("ready_for_approval")) != report.ready_for_approval:
            raise ValueError("Motion-preflight approval readiness does not match")
        return StoredMotionPreflight(report, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid motion-preflight artifact {root}: {exc}") from exc
