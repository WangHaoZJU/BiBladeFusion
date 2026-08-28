"""Immutable, re-derived HoloRobot motion-preflight artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from biblade_fusion.core.settings import (
    CollisionConfig,
    MotionPreflightConfig,
    OccupancyConfig,
)
from biblade_fusion.robotics import (
    Es68PinocchioCollisionChecker,
    OccupancyRobotCollisionChecker,
)
from biblade_fusion.storage.initialization import read_initialization
from biblade_fusion.storage.occupancy_mapping import read_occupancy_mapping
from biblade_fusion.storage.view_plan import read_view_plan
from biblade_fusion.workflows import (
    ViewSequenceMotionPreflight,
    preflight_view_sequence_motion,
)

MOTION_PREFLIGHT_SCHEMA_VERSION = 5


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


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Motion-preflight evaluation time must be timezone-aware")
    return value.astimezone(UTC)


def _joint_zero_offsets(
    values: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    offsets = tuple(float(value) for value in values)
    if not offsets:
        offsets = (0.0,) * 6
    if len(offsets) != 6 or not all(math.isfinite(value) for value in offsets):
        raise ValueError("ES68 joint-zero offsets must be a finite six-vector")
    return offsets  # type: ignore[return-value]


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
    collision_config: CollisionConfig,
    occupancy_config: OccupancyConfig | None,
    occupancy: Path | None,
    joint_zero_offsets_rad: tuple[float, float, float, float, float, float],
    evaluated_at_utc: datetime,
    execution_freshness_margin_s: float,
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
    plan_offsets_raw = stored_plan.metadata["source_kinematics"].get(
        "joint_zero_offsets_rad"
    )
    if plan_offsets_raw is None:
        raise ValueError(
            "View plan lacks joint-zero-offset provenance; regenerate the plan"
        )
    plan_offsets = _joint_zero_offsets(plan_offsets_raw)
    if plan_offsets != joint_zero_offsets_rad:
        raise ValueError(
            "Motion-preflight joint-zero offsets differ from the view plan"
        )
    if collision_config.require_obstacles and not collision_config.obstacles:
        raise ValueError(
            "Motion preflight requires at least one configured workcell obstacle"
        )
    checker_unavailable_reason = "checker_unavailable"
    try:
        checker = Es68PinocchioCollisionChecker.from_es68_resources(
            joint_zero_offsets_rad=joint_zero_offsets_rad,
            environment_obstacles=collision_config.obstacles,
            minimum_clearance_m=collision_config.minimum_clearance_m,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        checker = None
        checker_unavailable_reason = (
            "es68_d435i_collision_model_unavailable:"
            f"{type(exc).__name__}"
        )
    occupancy_checker = None
    if occupancy is not None:
        if occupancy_config is None:
            raise ValueError("Occupancy configuration is required with an occupancy artifact")
        stored_occupancy = read_occupancy_mapping(occupancy)
        if (
            stored_occupancy.motion_eligible is not True
            or stored_occupancy.verification_status
            != "full_semantic_verified_for_motion_preflight"
        ):
            raise ValueError(
                "Occupancy artifact lacks full semantic motion-preflight verification"
            )
        if (
            stored_occupancy.semantic_attestation.occupancy_metadata_sha256
            != _sha256(occupancy / "metadata.json")
        ):
            raise ValueError(
                "Occupancy semantic attestation metadata hash differs from source"
            )
        artifact_config = OccupancyConfig.model_validate(
            stored_occupancy.metadata["configuration"]["occupancy"]
        )
        if artifact_config != occupancy_config:
            raise ValueError("Active occupancy configuration differs from the map artifact")
        snapshot = stored_occupancy.snapshot
        if checker is not None:
            if checker.robot_geometry_hash is None:
                raise ValueError("ES68 collision checker lacks robot-geometry identity")
            if any(
                evidence.robot_model_hash != checker.robot_geometry_hash
                for evidence in stored_occupancy.frame_evidence
            ):
                raise ValueError(
                    "Occupancy self-mask robot geometry differs from motion preflight"
                )
            hand_eye_hash = _sha256(stored_initialization.hand_eye.source_path)
            if any(
                evidence.hand_eye_hash != hand_eye_hash
                for evidence in stored_occupancy.frame_evidence
            ):
                raise ValueError(
                    "Occupancy hand-eye calibration differs from initialization"
                )
            occupancy_checker = OccupancyRobotCollisionChecker(
                checker,
                lambda: snapshot,
                maximum_map_age_s=occupancy_config.maximum_map_age_s,
                additional_clearance_m=(
                    occupancy_config.obstacle_inflation_m
                    + checker.minimum_clearance_m
                ),
                semantic_attestation=stored_occupancy.semantic_attestation,
                utc_clock=lambda: evaluated_at_utc,
            )
    return preflight_view_sequence_motion(
        stored_plan.result.filtered_plan,
        ordered_view_ids,
        stored_initialization.observation.seed_joint_positions_rad,
        config,
        hand_eye=stored_initialization.hand_eye,
        collision_checker=checker,
        collision_checker_unavailable_reason=checker_unavailable_reason,
        occupancy_checker=occupancy_checker,
        execution_freshness_margin_s=execution_freshness_margin_s,
        evaluated_at_utc=evaluated_at_utc,
    )


def write_motion_preflight(
    output_dir: str | Path,
    ordered_view_ids: tuple[str, ...],
    config: MotionPreflightConfig,
    collision_config: CollisionConfig,
    occupancy_config: OccupancyConfig | None = None,
    *,
    source_plan: str | Path,
    source_initialization: str | Path,
    source_occupancy: str | Path | None = None,
    joint_zero_offsets_rad: Sequence[float] = (),
    execution_freshness_margin_s: float = 1.0,
) -> Path:
    """Persist a self-collision/ServoJ preflight that never authorizes execution."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Motion-preflight output already exists: {output}")
    plan = Path(source_plan).resolve()
    initialization = Path(source_initialization).resolve()
    occupancy = (
        Path(source_occupancy).resolve() if source_occupancy is not None else None
    )
    offsets = _joint_zero_offsets(joint_zero_offsets_rad)
    freshness_margin_s = float(execution_freshness_margin_s)
    if not math.isfinite(freshness_margin_s) or freshness_margin_s < 0.0:
        raise ValueError("execution_freshness_margin_s must be finite and non-negative")
    evaluated_at = _utc(_now_utc())
    report = _derive(
        plan,
        initialization,
        ordered_view_ids,
        config,
        collision_config,
        occupancy_config,
        occupancy,
        offsets,
        evaluated_at,
        freshness_margin_s,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload = {
            "schema_version": MOTION_PREFLIGHT_SCHEMA_VERSION,
            "created_at_utc": evaluated_at.isoformat(),
            "evaluated_at_utc": evaluated_at.isoformat(),
            "motion_authorized": False,
            "ready_for_approval": report.ready_for_approval,
            "sources": {
                "view_plan": _directory_source(plan, "view_plan.json"),
                "initialization": _directory_source(
                    initialization, "metadata.json"
                ),
                "occupancy": (
                    _directory_source(occupancy, "metadata.json")
                    if occupancy is not None
                    else None
                ),
            },
            "configuration": {
                "motion_preflight": config.model_dump(mode="json"),
                "collision": collision_config.model_dump(mode="json"),
                "occupancy": (
                    occupancy_config.model_dump(mode="json")
                    if occupancy_config is not None
                    else None
                ),
                "joint_zero_offsets_rad": list(offsets),
                "execution_freshness_margin_s": freshness_margin_s,
            },
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
        occupancy = (
            _verify_source(sources["occupancy"])
            if sources.get("occupancy") is not None
            else None
        )
        configuration = payload["configuration"]
        config = MotionPreflightConfig.model_validate(
            configuration["motion_preflight"]
        )
        collision_config = CollisionConfig.model_validate(configuration["collision"])
        occupancy_config = (
            OccupancyConfig.model_validate(configuration["occupancy"])
            if configuration.get("occupancy") is not None
            else None
        )
        offsets = _joint_zero_offsets(configuration["joint_zero_offsets_rad"])
        freshness_margin_s = float(configuration["execution_freshness_margin_s"])
        if not math.isfinite(freshness_margin_s) or freshness_margin_s < 0.0:
            raise ValueError(
                "execution_freshness_margin_s must be finite and non-negative"
            )
        evaluated_at = _utc(datetime.fromisoformat(str(payload["evaluated_at_utc"])))
        created_at = _utc(datetime.fromisoformat(str(payload["created_at_utc"])))
        if created_at != evaluated_at:
            raise ValueError(
                "Motion-preflight creation and evaluation instants must match"
            )
        ordered_view_ids = tuple(str(value) for value in payload["ordered_view_ids"])
        report = _derive(
            plan,
            initialization,
            ordered_view_ids,
            config,
            collision_config,
            occupancy_config,
            occupancy,
            offsets,
            evaluated_at,
            freshness_margin_s,
        )
        normalized = json.loads(json.dumps(asdict(report), allow_nan=False))
        if payload["report"] != normalized:
            raise ValueError("Motion-preflight report does not match its sources")
        if bool(payload.get("ready_for_approval")) != report.ready_for_approval:
            raise ValueError("Motion-preflight approval readiness does not match")
        return StoredMotionPreflight(report, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid motion-preflight artifact {root}: {exc}") from exc
