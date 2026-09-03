"""Immutable, non-executable motion-envelope commissioning candidates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.core.settings import CollisionConfig, MotionPreflightConfig
from biblade_fusion.planning import CandidateStatus
from biblade_fusion.robotics import (
    Es68D435iCollisionResources,
    Es68PinocchioCollisionChecker,
    preflight_linear_joint_motion,
)
from biblade_fusion.storage.initialization import read_initialization
from biblade_fusion.storage.occupancy_mapping import read_occupancy_mapping
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.storage.view_plan import read_view_plan

MOTION_ENVELOPE_COMMISSIONING_SCHEMA_VERSION = 1
MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD = 0.02


@dataclass(frozen=True, slots=True)
class CommissioningTrialCandidate:
    """One planner-directed short segment that cannot authorize execution."""

    candidate_id: str
    target_view_id: str
    start_view_id: str
    start_joint_positions_rad: tuple[float, ...]
    raw_target_joint_positions_rad: tuple[float, ...]
    normalized_target_joint_positions_rad: tuple[float, ...]
    target_joint_turn_offsets: tuple[int, ...]
    goal_joint_positions_rad: tuple[float, ...]
    direction_scale: float
    maximum_candidate_joint_delta_rad: float
    maximum_remaining_target_joint_delta_rad: float
    mesh_status: str
    mesh_continuous_swept_volume_verified: bool
    mesh_minimum_certificate_margin_m: float | None
    estimated_servoj_duration_s: float | None
    servoj_command_count: int | None
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def motion_authorized(self) -> bool:
        return False

    @property
    def execution_capable(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class StoredCommissioningTrialCandidate:
    path: Path
    candidate: CommissioningTrialCandidate
    metadata: dict[str, Any]

    @property
    def motion_authorized(self) -> bool:
        return False

    @property
    def execution_capable(self) -> bool:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _source(root: Path, filename: str) -> dict[str, str]:
    resolved = root.resolve()
    path = (resolved / filename).resolve()
    if not path.is_file() or not path.is_relative_to(resolved):
        raise ValueError(f"Commissioning source is missing or escapes its root: {path}")
    return {"root": str(resolved), "file": filename, "sha256": _sha256(path)}


def _verify_source(record: dict[str, Any]) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Commissioning source escapes its artifact root")
    if not path.is_file() or _sha256(path) != str(record["sha256"]):
        raise ValueError(f"Commissioning source checksum mismatch: {path}")
    return root


def _six(values: Any, *, label: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite six-vector")
    return tuple(float(value) for value in vector)


def _validate_initialization_safety_boundary(stored_initialization: Any) -> None:
    authorization = stored_initialization.metadata.get("motion_authorized", False)
    if authorization is not False:
        raise ValueError("Initialization unexpectedly authorizes motion")


def _nearest_equivalent_target(
    start: tuple[float, ...],
    target: tuple[float, ...],
    limits: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    normalized: list[float] = []
    turns: list[int] = []
    for begin, end, (lower, upper) in zip(start, target, limits, strict=True):
        candidates = tuple(
            (end + 2.0 * math.pi * turn, turn)
            for turn in range(-3, 4)
            if lower - 1e-12 <= end + 2.0 * math.pi * turn <= upper + 1e-12
        )
        if not candidates:
            raise ValueError("Planner target has no joint-limit-valid 2*pi equivalent")
        selected, turn = min(candidates, key=lambda item: (abs(item[0] - begin), abs(item[1])))
        normalized.append(float(selected))
        turns.append(int(turn))
    return tuple(normalized), tuple(turns)


def _derive_candidate(
    *,
    plan: Path,
    initialization: Path,
    start_session: Path,
    start_view_id: str,
    occupancy: Path,
    target_view_id: str,
    maximum_candidate_joint_delta_rad: float,
    motion_config: MotionPreflightConfig,
    collision_config: CollisionConfig,
    joint_zero_offsets_rad: tuple[float, ...],
) -> CommissioningTrialCandidate:
    bound = float(maximum_candidate_joint_delta_rad)
    if (
        not math.isfinite(bound)
        or bound <= 0.0
        or bound > MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD
    ):
        raise ValueError(
            "Commissioning candidate joint delta must be in (0, 0.02] rad"
        )

    stored_plan = read_view_plan(plan)
    stored_initialization = read_initialization(initialization)
    if Path(str(stored_plan.metadata["source_initialization"])).resolve() != initialization:
        raise ValueError("Commissioning plan does not belong to the supplied initialization")
    _validate_initialization_safety_boundary(stored_initialization)

    evaluations = {
        evaluation.candidate.view_id: evaluation
        for evaluation in stored_plan.result.filtered_plan.candidates
    }
    try:
        evaluation = evaluations[target_view_id]
    except KeyError as exc:
        raise ValueError(f"Unknown commissioning target view: {target_view_id}") from exc
    if (
        evaluation.status is not CandidateStatus.ENDPOINT_FEASIBLE
        or evaluation.joint_positions_rad is None
    ):
        raise ValueError("Commissioning target must be endpoint-feasible")

    reader = SessionReader(start_session)
    bundle = reader.load_bundle(start_view_id)
    if bundle.metrics.max_joint_delta_rad != 0.0:
        raise ValueError("Commissioning start capture must have zero bracket joint motion")
    start = _six(bundle.selected_robot_state.joint_positions_rad, label="commissioning start")
    raw_target = _six(evaluation.joint_positions_rad, label="planner target")

    stored_occupancy = read_occupancy_mapping(occupancy)
    evidence = tuple(
        item for item in stored_occupancy.frame_evidence if item.source_view_id == start_view_id
    )
    if len(evidence) != 1:
        raise ValueError("Commissioning start view is not uniquely bound to the occupancy map")
    descriptor = reader.descriptor(start_view_id)
    view_metadata = (start_session / descriptor.relative_path / "metadata.json").resolve()
    if (
        evidence[0].source_session_manifest_sha256 != _sha256(start_session / "manifest.json")
        or evidence[0].source_session_view_metadata_sha256 != _sha256(view_metadata)
    ):
        raise ValueError("Commissioning start session differs from occupancy provenance")

    checker = Es68PinocchioCollisionChecker.from_es68_resources(
        Es68D435iCollisionResources.packaged_template(),
        joint_zero_offsets_rad=joint_zero_offsets_rad,
        environment_obstacles=collision_config.obstacles,
        minimum_clearance_m=collision_config.minimum_clearance_m,
    )
    normalized_target, turn_offsets = _nearest_equivalent_target(
        start,
        raw_target,
        checker.kinematic_model.joint_limit_pairs(),
    )
    start_array = np.asarray(start, dtype=np.float64)
    target_array = np.asarray(normalized_target, dtype=np.float64)
    delta = target_array - start_array
    maximum_delta = float(np.max(np.abs(delta)))
    if maximum_delta <= 1e-12:
        raise ValueError("Commissioning start already equals the planner target")
    scale = min(1.0, bound / maximum_delta)
    goal = tuple(float(value) for value in start_array + scale * delta)

    preflight = preflight_linear_joint_motion(
        start,
        goal,
        collision_checker=checker,
        require_occupancy=False,
        maximum_joint_step_rad=motion_config.maximum_joint_step_rad,
        servoj_dt_s=motion_config.servoj_dt_s,
        speed_scaling=motion_config.speed_scaling,
        velocity_margin=motion_config.velocity_margin,
    )
    collision = preflight.collision
    proof = collision.proof_evidence if collision is not None else None
    stream = preflight.servoj_stream
    core = {
        "target_view_id": target_view_id,
        "start_view_id": start_view_id,
        "start_joint_positions_rad": list(start),
        "raw_target_joint_positions_rad": list(raw_target),
        "normalized_target_joint_positions_rad": list(normalized_target),
        "target_joint_turn_offsets": list(turn_offsets),
        "goal_joint_positions_rad": list(goal),
        "direction_scale": scale,
        "maximum_candidate_joint_delta_rad": float(np.max(np.abs(np.asarray(goal) - start_array))),
        "maximum_remaining_target_joint_delta_rad": maximum_delta,
        "mesh_status": preflight.status.value,
        "mesh_continuous_swept_volume_verified": bool(
            collision is not None and collision.continuous_swept_volume_evidence_valid
        ),
        "mesh_minimum_certificate_margin_m": (
            proof.minimum_certificate_margin_m if proof is not None else None
        ),
        "estimated_servoj_duration_s": (
            max(0, len(stream.commands) - 1) * stream.dt_s if stream is not None else None
        ),
        "servoj_command_count": len(stream.commands) if stream is not None else None,
        "blocking_reasons": list(preflight.blocking_reasons),
        "warnings": list(preflight.warnings),
    }
    return CommissioningTrialCandidate(
        candidate_id=_canonical_sha256(core),
        target_view_id=target_view_id,
        start_view_id=start_view_id,
        start_joint_positions_rad=start,
        raw_target_joint_positions_rad=raw_target,
        normalized_target_joint_positions_rad=normalized_target,
        target_joint_turn_offsets=turn_offsets,
        goal_joint_positions_rad=goal,
        direction_scale=scale,
        maximum_candidate_joint_delta_rad=core["maximum_candidate_joint_delta_rad"],
        maximum_remaining_target_joint_delta_rad=maximum_delta,
        mesh_status=preflight.status.value,
        mesh_continuous_swept_volume_verified=core[
            "mesh_continuous_swept_volume_verified"
        ],
        mesh_minimum_certificate_margin_m=core["mesh_minimum_certificate_margin_m"],
        estimated_servoj_duration_s=core["estimated_servoj_duration_s"],
        servoj_command_count=core["servoj_command_count"],
        blocking_reasons=tuple(preflight.blocking_reasons),
        warnings=tuple(preflight.warnings),
    )


def write_commissioning_trial_candidate(
    output_dir: str | Path,
    *,
    plan: str | Path,
    initialization: str | Path,
    start_session: str | Path,
    start_view_id: str,
    occupancy: str | Path,
    target_view_id: str,
    maximum_candidate_joint_delta_rad: float,
    motion_config: MotionPreflightConfig,
    collision_config: CollisionConfig,
    joint_zero_offsets_rad: tuple[float, ...] = (),
) -> StoredCommissioningTrialCandidate:
    """Write one immutable diagnostic candidate without an executable stream."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Commissioning output already exists: {output}")
    resolved_plan = Path(plan).resolve()
    resolved_initialization = Path(initialization).resolve()
    resolved_session = Path(start_session).resolve()
    resolved_occupancy = Path(occupancy).resolve()
    offsets = _six(joint_zero_offsets_rad or (0.0,) * 6, label="joint zero offsets")
    candidate = _derive_candidate(
        plan=resolved_plan,
        initialization=resolved_initialization,
        start_session=resolved_session,
        start_view_id=start_view_id,
        occupancy=resolved_occupancy,
        target_view_id=target_view_id,
        maximum_candidate_joint_delta_rad=maximum_candidate_joint_delta_rad,
        motion_config=motion_config,
        collision_config=collision_config,
        joint_zero_offsets_rad=offsets,
    )
    descriptor = SessionReader(resolved_session).descriptor(start_view_id)
    payload = {
        "schema_version": MOTION_ENVELOPE_COMMISSIONING_SCHEMA_VERSION,
        "artifact_kind": "biblade_fusion.motion_envelope_commissioning_candidate",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "motion_authorized": False,
        "execution_capable": False,
        "ready_for_execution": False,
        "candidate": asdict(candidate),
        "configuration": {
            "maximum_commissioning_candidate_joint_delta_rad": (
                maximum_candidate_joint_delta_rad
            ),
            "motion_preflight": motion_config.model_dump(mode="json"),
            "collision": collision_config.model_dump(mode="json"),
            "joint_zero_offsets_rad": list(offsets),
        },
        "sources": {
            "view_plan": _source(resolved_plan, "view_plan.json"),
            "initialization": _source(resolved_initialization, "metadata.json"),
            "start_session": _source(resolved_session, "manifest.json"),
            "start_view": _source(
                resolved_session,
                str(Path(descriptor.relative_path) / "metadata.json"),
            ),
            "occupancy": _source(resolved_occupancy, "metadata.json"),
        },
        "safety_boundary": {
            "planner_direction_required": True,
            "continuous_mesh_proof_required": True,
            "occupancy_replay_is_diagnostic_only": True,
            "servoj_commands_persisted": False,
            "robot_write_interface_present": False,
            "requires_separate_live_revalidation_before_any_future_trial": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        (temporary / "candidate.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return StoredCommissioningTrialCandidate(output.resolve(), candidate, payload)


def read_commissioning_trial_candidate(
    path: str | Path,
) -> StoredCommissioningTrialCandidate:
    """Verify sources and re-derive a non-executable commissioning candidate."""

    root = Path(path).resolve()
    payload = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != MOTION_ENVELOPE_COMMISSIONING_SCHEMA_VERSION
        or payload.get("motion_authorized") is not False
        or payload.get("execution_capable") is not False
        or payload.get("ready_for_execution") is not False
    ):
        raise ValueError("Commissioning candidate safety boundary is invalid")
    sources = payload["sources"]
    plan = _verify_source(sources["view_plan"])
    initialization = _verify_source(sources["initialization"])
    start_session = _verify_source(sources["start_session"])
    _verify_source(sources["start_view"])
    occupancy = _verify_source(sources["occupancy"])
    configuration = payload["configuration"]
    stored_candidate = payload["candidate"]
    derived = _derive_candidate(
        plan=plan,
        initialization=initialization,
        start_session=start_session,
        start_view_id=str(stored_candidate["start_view_id"]),
        occupancy=occupancy,
        target_view_id=str(stored_candidate["target_view_id"]),
        maximum_candidate_joint_delta_rad=float(
            configuration["maximum_commissioning_candidate_joint_delta_rad"]
        ),
        motion_config=MotionPreflightConfig.model_validate(
            configuration["motion_preflight"]
        ),
        collision_config=CollisionConfig.model_validate(configuration["collision"]),
        joint_zero_offsets_rad=_six(
            configuration["joint_zero_offsets_rad"], label="joint zero offsets"
        ),
    )
    if _canonical_sha256(asdict(derived)) != _canonical_sha256(stored_candidate):
        raise ValueError("Commissioning candidate differs from deterministic re-derivation")
    boundary = payload.get("safety_boundary")
    if not isinstance(boundary, dict) or boundary != {
        "planner_direction_required": True,
        "continuous_mesh_proof_required": True,
        "occupancy_replay_is_diagnostic_only": True,
        "servoj_commands_persisted": False,
        "robot_write_interface_present": False,
        "requires_separate_live_revalidation_before_any_future_trial": True,
    }:
        raise ValueError("Commissioning candidate safety-boundary declaration is invalid")
    return StoredCommissioningTrialCandidate(root, derived, payload)


__all__ = [
    "MAXIMUM_COMMISSIONING_CANDIDATE_JOINT_DELTA_RAD",
    "MOTION_ENVELOPE_COMMISSIONING_SCHEMA_VERSION",
    "CommissioningTrialCandidate",
    "StoredCommissioningTrialCandidate",
    "read_commissioning_trial_candidate",
    "write_commissioning_trial_candidate",
]
