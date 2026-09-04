"""Fail-closed robot-envelope collision queries against immutable occupancy snapshots.

The mapping package owns construction and serialization of occupancy snapshots.
Motion preflight accepts only the concrete frozen :class:`OccupancySnapshot` type,
not a structural lookalike.  Every successful report carries the exact snapshot
sequence and recomputed content hash used for the query.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from biblade_fusion.mapping.occupancy import (
    OccupancySnapshot,
    compute_content_hash,
)
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
    _canonical_sha256,
    _joint_path_sha256,
)


class OccupancyQueryState(StrEnum):
    """Motion-side normalization of mapping occupancy states."""

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class OccupancyGeometryQueryLike(Protocol):
    state: Any
    blocked: bool
    occupied_count: int
    unknown_count: int
    free_count: int
    queried_count: int


OccupancySnapshotProvider = Callable[[], OccupancySnapshot]


class OccupancyEvidenceError(ValueError):
    """Raised when a snapshot cannot be used as motion-safety evidence."""


_SEMANTIC_VERIFIER_CONTRACT_PAYLOAD = {
    "schema": "biblade_fusion.occupancy_semantic_verifier.v1",
    "occupancy_mapping_schema": 7,
    "snapshot_type": "biblade_fusion.mapping.occupancy.OccupancySnapshot",
    "snapshot_content_hash": "canonical_sha256_recomputed",
    "verification": ("integrity_chain+raw_stereo_source+hand_eye+es68_fk+active_robot_rerender"),
    "replay_eligible": False,
}
OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH = hashlib.sha256(
    json.dumps(
        _SEMANTIC_VERIFIER_CONTRACT_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

def _semantic_attestation_hash(
    *,
    occupancy_metadata_sha256: str,
    snapshot_sequence: int,
    snapshot_content_hash: str,
    mapping_context_hash: str,
    quality_evidence_hash: str,
    robot_geometry_hash: str,
    semantic_verifier_contract_hash: str,
) -> str:
    payload = {
        "schema": "biblade_fusion.occupancy_semantic_attestation.v1",
        "occupancy_metadata_sha256": occupancy_metadata_sha256,
        "snapshot_sequence": snapshot_sequence,
        "snapshot_content_hash": snapshot_content_hash,
        "mapping_context_hash": mapping_context_hash,
        "quality_evidence_hash": quality_evidence_hash,
        "robot_geometry_hash": robot_geometry_hash,
        "semantic_verifier_contract_hash": semantic_verifier_contract_hash,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class OccupancySemanticAttestation:
    """Process-local proof emitted only after the full semantic asset reader passes.

    The constructor is deliberately disabled.  Storage's strict full reader issues
    this value through the private factory below; replay/integrity-only readers never
    receive one.  Motion still independently revalidates the concrete immutable
    snapshot and every hash in this contract.
    """

    occupancy_metadata_sha256: str
    snapshot_sequence: int
    snapshot_content_hash: str
    mapping_context_hash: str
    quality_evidence_hash: str
    robot_geometry_hash: str
    semantic_verifier_contract_hash: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("OccupancySemanticAttestation is issued only by full semantic verification")

    @property
    def attestation_hash(self) -> str:
        return _semantic_attestation_hash(
            occupancy_metadata_sha256=self.occupancy_metadata_sha256,
            snapshot_sequence=self.snapshot_sequence,
            snapshot_content_hash=self.snapshot_content_hash,
            mapping_context_hash=self.mapping_context_hash,
            quality_evidence_hash=self.quality_evidence_hash,
            robot_geometry_hash=self.robot_geometry_hash,
            semantic_verifier_contract_hash=self.semantic_verifier_contract_hash,
        )

    def assert_matches(
        self,
        snapshot: OccupancySnapshot,
        *,
        robot_geometry_hash: str,
    ) -> None:
        if self.semantic_verifier_contract_hash != (OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH):
            raise OccupancyEvidenceError("occupancy_semantic_verifier_contract_is_not_current")
        expected = (
            snapshot.sequence,
            snapshot.content_hash,
            snapshot.mapping_context_hash,
            snapshot.quality_evidence_hash,
            robot_geometry_hash,
        )
        recorded = (
            self.snapshot_sequence,
            self.snapshot_content_hash,
            self.mapping_context_hash,
            self.quality_evidence_hash,
            self.robot_geometry_hash,
        )
        if recorded != expected:
            raise OccupancyEvidenceError(
                "occupancy_semantic_attestation_does_not_match_current_snapshot"
            )


def _issue_occupancy_semantic_attestation(
    *,
    occupancy_metadata_sha256: str,
    snapshot: OccupancySnapshot,
    robot_geometry_hash: str,
) -> OccupancySemanticAttestation:
    """Issue the strong type used by storage after full semantic verification."""

    if type(snapshot) is not OccupancySnapshot:
        raise OccupancyEvidenceError("occupancy_semantic_attestation_requires_concrete_snapshot")
    metadata_hash = _sha256_digest(
        occupancy_metadata_sha256,
        reason="occupancy_metadata_hash_must_be_sha256",
    )
    snapshot_hash = _sha256_digest(
        snapshot.content_hash,
        reason="occupancy_content_hash_must_be_sha256",
    )
    if compute_content_hash(snapshot) != snapshot_hash:
        raise OccupancyEvidenceError("occupancy_snapshot_content_hash_mismatch")
    context_hash = _sha256_digest(
        snapshot.mapping_context_hash,
        reason="occupancy_mapping_context_hash_must_be_sha256",
    )
    quality_hash = _sha256_digest(
        snapshot.quality_evidence_hash,
        reason="occupancy_quality_evidence_hash_must_be_sha256",
    )
    geometry_hash = _sha256_digest(
        robot_geometry_hash,
        reason="occupancy_robot_geometry_hash_must_be_verified",
    )
    attestation = object.__new__(OccupancySemanticAttestation)
    for name, value in (
        ("occupancy_metadata_sha256", metadata_hash),
        ("snapshot_sequence", int(snapshot.sequence)),
        ("snapshot_content_hash", snapshot_hash),
        ("mapping_context_hash", context_hash),
        ("quality_evidence_hash", quality_hash),
        ("robot_geometry_hash", geometry_hash),
        (
            "semantic_verifier_contract_hash",
            OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH,
        ),
    ):
        object.__setattr__(attestation, name, value)
    attestation.assert_matches(snapshot, robot_geometry_hash=geometry_hash)
    return attestation


@dataclass(frozen=True, slots=True)
class OccupancyMapEvidence:
    """Stable identity of one immutable map used during collision checking."""

    frame_id: str
    sequence: int
    content_hash: str
    mapping_context_hash: str
    quality_evidence_hash: str
    robot_geometry_hash: str
    created_at_utc: str
    source_view_ids: tuple[str, ...]
    occupancy_metadata_sha256: str | None = None
    semantic_verifier_contract_hash: str | None = None
    semantic_attestation_hash: str | None = None

    @property
    def semantic_attestation_valid(self) -> bool:
        digests = (
            self.occupancy_metadata_sha256,
            self.semantic_verifier_contract_hash,
            self.semantic_attestation_hash,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            return False
        if self.semantic_verifier_contract_hash != OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH:
            return False
        expected = _semantic_attestation_hash(
            occupancy_metadata_sha256=self.occupancy_metadata_sha256,
            snapshot_sequence=self.sequence,
            snapshot_content_hash=self.content_hash,
            mapping_context_hash=self.mapping_context_hash,
            quality_evidence_hash=self.quality_evidence_hash,
            robot_geometry_hash=self.robot_geometry_hash,
            semantic_verifier_contract_hash=self.semantic_verifier_contract_hash,
        )
        return self.semantic_attestation_hash == expected

    @property
    def binding(
        self,
    ) -> tuple[
        str,
        int,
        str,
        str,
        str,
        str,
        str | None,
        str | None,
        str | None,
    ]:
        return (
            self.frame_id,
            self.sequence,
            self.content_hash,
            self.mapping_context_hash,
            self.quality_evidence_hash,
            self.robot_geometry_hash,
            self.occupancy_metadata_sha256,
            self.semantic_verifier_contract_hash,
            self.semantic_attestation_hash,
        )


@dataclass(frozen=True, slots=True)
class _PlacedRobotCollisionGeometry:
    geometry_name: str
    geometry_index: int
    collision_geometry: Any
    transform_base: Any
    world_aabb_minimum_m: tuple[float, float, float]
    world_aabb_maximum_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AcceptedStaticFreeAabb:
    """One strongly identified operator-accepted free region in ``base``.

    Only UNKNOWN voxels whose complete metric voxel AABB lies inside this box may
    be accepted.  OCCUPIED voxels are never downgraded and partial intersections
    remain UNKNOWN.  The checker separately binds the acceptance asset and the
    exact occupancy mapping context in which it was approved.
    """

    name: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        lower = tuple(float(value) for value in self.minimum_m)
        upper = tuple(float(value) for value in self.maximum_m)
        if not name:
            raise ValueError("accepted static-free AABB name must be non-empty")
        if len(lower) != 3 or len(upper) != 3 or not np.isfinite((lower, upper)).all():
            raise ValueError("accepted static-free AABB bounds must be finite triplets")
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise ValueError("accepted static-free AABB maxima must exceed minima")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "minimum_m", lower)
        object.__setattr__(self, "maximum_m", upper)

    def contains_voxel(
        self,
        snapshot: OccupancySnapshot,
        index: tuple[int, int, int],
    ) -> bool:
        tolerance = max(1e-12, snapshot.voxel_size_m * 1e-9)
        lower = tuple(
            snapshot.origin_m[axis] + index[axis] * snapshot.voxel_size_m for axis in range(3)
        )
        upper = tuple(value + snapshot.voxel_size_m for value in lower)
        return all(
            voxel_low >= accepted_low - tolerance
            and voxel_high <= accepted_high + tolerance
            for voxel_low, voxel_high, accepted_low, accepted_high in zip(
                lower,
                upper,
                self.minimum_m,
                self.maximum_m,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _RobotGeometryVoxelQuery:
    state: OccupancyQueryState
    blocked: bool
    occupied_count: int
    unknown_count: int
    free_count: int
    accepted_unknown_count: int
    outside_grid_unknown_count: int
    outside_acceptance_unknown_count: int
    separated_dangerous_count: int
    distance_query_count: int
    minimum_dangerous_distance_m: float | None
    blocking_voxel_index: tuple[int, int, int] | None
    queried_count: int


@dataclass(frozen=True, slots=True)
class OccupancyCollisionCheckResult:
    status: CollisionCheckStatus
    blocking_reasons: tuple[str, ...]
    evidence: OccupancyMapEvidence | None
    checked_geometry_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def checked_sphere_count(self) -> int:
        """Compatibility alias for schema-2 callers; values now count STL geometries."""

        return self.checked_geometry_count

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(slots=True)
class BoundOccupancyConfigurationQuery:
    """Many state-validity queries against one integrity-checked map snapshot."""

    checker: OccupancyRobotCollisionChecker
    snapshot: OccupancySnapshot
    evidence: OccupancyMapEvidence
    required_freshness_horizon_s: float
    _closed: bool = False

    def check(
        self,
        joint_positions_rad: Sequence[float],
    ) -> OccupancyCollisionCheckResult:
        if self._closed:
            raise OccupancyEvidenceError("bound_occupancy_query_is_closed")
        try:
            return self.checker._check_bound_configuration(
                self.snapshot,
                self.evidence,
                joint_positions_rad,
                required_freshness_horizon_s=self.required_freshness_horizon_s,
                stop_on_first_block=True,
            )
        except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
            return self.checker._unknown_result(
                f"occupancy_checker_error:{exc}",
                evidence=self.evidence,
            )
        except Exception as exc:  # pragma: no cover - fail closed across FCL
            return self.checker._unknown_result(
                f"occupancy_query_failed:{type(exc).__name__}:{exc}",
                evidence=self.evidence,
            )

    def close(self) -> None:
        if self._closed:
            return
        self.checker.assert_current_evidence(
            self.evidence,
            required_freshness_horizon_s=self.required_freshness_horizon_s,
        )
        self._closed = True


def _occupancy_evidence_binding_sha256(evidence: OccupancyMapEvidence) -> str:
    return _canonical_sha256(
        {
            "schema": "biblade_fusion.occupancy_map_motion_binding.v1",
            "binding": list(evidence.binding),
            "created_at_utc": evidence.created_at_utc,
            "source_view_ids": list(evidence.source_view_ids),
        }
    )


@dataclass(frozen=True, slots=True)
class SweptOccupancyProofEvidence:
    """Integrity-bound proof using exact STL-to-voxel distance margins."""

    trajectory_sha256: str
    map_binding_sha256: str
    occupancy_policy_contract_hash: str
    robot_motion_bound_contract_sha256: str
    motion_envelope_acceptance_id: str | None
    motion_envelope_metadata_sha256: str | None
    accepted_joint_uncertainty_rad: tuple[float, float, float, float, float, float]
    maximum_joint_step_rad: float
    maximum_subdivision_depth: int
    minimum_interval_joint_span_rad: float
    initial_interval_count: int
    certified_interval_count: int
    evaluated_configuration_count: int
    geometry_voxel_distance_query_count: int
    accepted_unknown_voxel_query_count: int
    deepest_subdivision: int
    termination_reason: str
    evidence_sha256: str
    schema: str = "biblade_fusion.swept_occupancy_proof.v3"
    method: str = "adaptive_midpoint_exact_stl_voxel_distance_sweep"

    @property
    def expanded_sphere_query_count(self) -> int:
        """Compatibility alias; no sphere queries are used by schema 3."""

        return self.geometry_voxel_distance_query_count

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": self.method,
            "trajectory_sha256": self.trajectory_sha256,
            "map_binding_sha256": self.map_binding_sha256,
            "occupancy_policy_contract_hash": self.occupancy_policy_contract_hash,
            "robot_motion_bound_contract_sha256": (self.robot_motion_bound_contract_sha256),
            "motion_envelope_acceptance_id": self.motion_envelope_acceptance_id,
            "motion_envelope_metadata_sha256": self.motion_envelope_metadata_sha256,
            "accepted_joint_uncertainty_rad": list(self.accepted_joint_uncertainty_rad),
            "maximum_joint_step_rad": self.maximum_joint_step_rad,
            "maximum_subdivision_depth": self.maximum_subdivision_depth,
            "minimum_interval_joint_span_rad": (self.minimum_interval_joint_span_rad),
            "initial_interval_count": self.initial_interval_count,
            "certified_interval_count": self.certified_interval_count,
            "evaluated_configuration_count": self.evaluated_configuration_count,
            "geometry_voxel_distance_query_count": (
                self.geometry_voxel_distance_query_count
            ),
            "accepted_unknown_voxel_query_count": (self.accepted_unknown_voxel_query_count),
            "deepest_subdivision": self.deepest_subdivision,
            "termination_reason": self.termination_reason,
        }

    @property
    def integrity_valid(self) -> bool:
        if self.schema != "biblade_fusion.swept_occupancy_proof.v3" or self.method != (
            "adaptive_midpoint_exact_stl_voxel_distance_sweep"
        ):
            return False
        digests = (
            self.trajectory_sha256,
            self.map_binding_sha256,
            self.occupancy_policy_contract_hash,
            self.robot_motion_bound_contract_sha256,
            self.evidence_sha256,
        )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            return False
        envelope = np.asarray(self.accepted_joint_uncertainty_rad, dtype=np.float64)
        if envelope.shape != (6,) or not np.isfinite(envelope).all() or np.any(envelope < 0.0):
            return False
        if np.any(envelope > 0.0):
            acceptance_id = self.motion_envelope_acceptance_id
            metadata_hash = self.motion_envelope_metadata_sha256
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (acceptance_id, metadata_hash)
            ):
                return False
        elif (
            self.motion_envelope_acceptance_id is not None
            or self.motion_envelope_metadata_sha256 is not None
        ):
            return False
        try:
            return self.evidence_sha256 == _canonical_sha256(self._payload())
        except (TypeError, ValueError):
            return False

    def matches_path(
        self,
        start_joint_positions_rad: Sequence[float],
        end_joint_positions_rad: Sequence[float],
    ) -> bool:
        return self.trajectory_sha256 == _joint_path_sha256(
            start_joint_positions_rad,
            end_joint_positions_rad,
        )


@dataclass(frozen=True, slots=True)
class JointPathOccupancyCollisionReport:
    status: CollisionCheckStatus
    sample_count: int
    blocked_sample_index: int | None
    blocked_path_fraction: float | None
    result: OccupancyCollisionCheckResult
    maximum_joint_step_rad: float
    continuous_swept_volume_verified: bool = False
    proof_evidence: SweptOccupancyProofEvidence | None = None

    @property
    def evidence(self) -> OccupancyMapEvidence | None:
        return self.result.evidence

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def continuous_swept_volume_evidence_valid(self) -> bool:
        proof = self.proof_evidence
        map_evidence = self.result.evidence
        return (
            self.status is CollisionCheckStatus.CLEAR
            and self.result.status is CollisionCheckStatus.CLEAR
            and self.continuous_swept_volume_verified
            and self.result.diagnostics.get("continuous_swept_volume_verified") is True
            and proof is not None
            and proof.integrity_valid
            and proof.termination_reason
            in {"all_intervals_certified", "constant_path_configuration_clear"}
            and map_evidence is not None
            and proof.map_binding_sha256 == _occupancy_evidence_binding_sha256(map_evidence)
            and proof.occupancy_policy_contract_hash
            == self.result.diagnostics.get("occupancy_policy_contract_hash")
            and proof.robot_motion_bound_contract_sha256
            == self.result.diagnostics.get("robot_motion_bound_contract_sha256")
            and proof.evidence_sha256
            == self.result.diagnostics.get("swept_occupancy_proof_evidence_sha256")
            and proof.termination_reason
            == self.result.diagnostics.get("swept_occupancy_termination_reason")
            and math.isclose(
                proof.maximum_joint_step_rad,
                self.maximum_joint_step_rad,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

    @property
    def motion_authorized(self) -> bool:
        return False


def _enum_value(value: Any) -> str:
    normalized = getattr(value, "value", value)
    return str(normalized).strip().lower()


def _normalized_created_at(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OccupancyEvidenceError("occupancy_created_at_must_be_timezone_aware")
    return value.astimezone(UTC).isoformat()


def _sha256_digest(value: object, *, reason: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise OccupancyEvidenceError(reason)
    return text


def occupancy_evidence_from_snapshot(
    snapshot: OccupancySnapshot,
    *,
    now_utc: datetime,
    max_age_s: float | None,
    authorization_started_at_utc: datetime | None = None,
    required_freshness_horizon_s: float = 0.0,
    verified_robot_geometry_hash: str | None = None,
    semantic_attestation: OccupancySemanticAttestation | None = None,
    allow_mapping_prefix: bool = False,
) -> OccupancyMapEvidence:
    """Validate one snapshot and return the identity bound to motion evidence."""

    if type(snapshot) is not OccupancySnapshot:
        raise OccupancyEvidenceError("occupancy_snapshot_must_be_concrete_immutable_snapshot")
    if compute_content_hash(snapshot) != snapshot.content_hash:
        raise OccupancyEvidenceError("occupancy_snapshot_content_hash_mismatch")
    if max_age_s is not None and (
        not math.isfinite(max_age_s) or max_age_s <= 0.0
    ):
        raise OccupancyEvidenceError("occupancy_max_age_must_be_positive")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise OccupancyEvidenceError("occupancy_check_time_must_be_timezone_aware")
    horizon = float(required_freshness_horizon_s)
    if not math.isfinite(horizon) or horizon < 0.0:
        raise OccupancyEvidenceError("occupancy_freshness_horizon_must_be_non_negative")
    if str(snapshot.frame_id) != "base":
        raise OccupancyEvidenceError(
            f"occupancy_frame_mismatch:{snapshot.frame_id!s}:expected_base"
        )
    map_state = _enum_value(snapshot.map_state)
    if map_state != "map_ready" and not (
        allow_mapping_prefix and map_state == "mapping"
    ):
        raise OccupancyEvidenceError(f"occupancy_map_not_ready:{_enum_value(snapshot.map_state)}")
    sequence = int(snapshot.sequence)
    if sequence < 0:
        raise OccupancyEvidenceError("occupancy_sequence_must_be_non_negative")
    content_hash = _sha256_digest(
        snapshot.content_hash,
        reason="occupancy_content_hash_must_be_sha256",
    )
    mapping_context_hash = _sha256_digest(
        getattr(snapshot, "mapping_context_hash", None),
        reason="occupancy_mapping_context_hash_must_be_sha256",
    )
    quality_evidence_hash = _sha256_digest(
        getattr(snapshot, "quality_evidence_hash", None),
        reason="occupancy_quality_evidence_hash_must_be_sha256",
    )
    robot_geometry_hash = _sha256_digest(
        verified_robot_geometry_hash,
        reason="occupancy_robot_geometry_hash_must_be_verified",
    )
    if semantic_attestation is not None:
        if type(semantic_attestation) is not OccupancySemanticAttestation:
            raise OccupancyEvidenceError("occupancy_semantic_attestation_has_invalid_type")
        semantic_attestation.assert_matches(
            snapshot,
            robot_geometry_hash=robot_geometry_hash,
        )
    try:
        valid_until = now_utc + timedelta(seconds=horizon)
        if max_age_s is None:
            # Publication replacement, not wall-clock age, owns lifecycle.
            stale = map_state == "stale"
            usable = map_state == "map_ready" or (
                allow_mapping_prefix and map_state == "mapping"
            )
        elif authorization_started_at_utc is None:
            stale = bool(snapshot.is_stale(valid_until, max_age_s))
            usable = not stale and (
                map_state == "map_ready"
                or (allow_mapping_prefix and map_state == "mapping")
            )
        else:
            if authorization_started_at_utc.tzinfo is None:
                raise ValueError("authorization start must be timezone-aware")
            authorization_started = authorization_started_at_utc.astimezone(UTC)
            authorization_age_s = (valid_until - authorization_started).total_seconds()
            stale = (
                _enum_value(snapshot.map_state) == "stale"
                or authorization_age_s < 0.0
                or authorization_age_s > max_age_s
            )
            usable = not stale and (
                map_state == "map_ready"
                or (allow_mapping_prefix and map_state == "mapping")
            )
    except Exception as exc:  # pragma: no cover - defensive adapter boundary
        raise OccupancyEvidenceError(f"occupancy_lifecycle_query_failed:{exc}") from exc
    if stale or not usable:
        raise OccupancyEvidenceError("occupancy_map_stale_or_unusable")
    source_view_ids = tuple(str(value).strip() for value in snapshot.source_view_ids)
    if (
        not source_view_ids
        or any(not value for value in source_view_ids)
        or len(set(source_view_ids)) != len(source_view_ids)
    ):
        raise OccupancyEvidenceError("occupancy_map_source_views_are_invalid")
    return OccupancyMapEvidence(
        frame_id="base",
        sequence=sequence,
        content_hash=content_hash,
        mapping_context_hash=mapping_context_hash,
        quality_evidence_hash=quality_evidence_hash,
        robot_geometry_hash=robot_geometry_hash,
        created_at_utc=_normalized_created_at(snapshot.created_at_utc),
        source_view_ids=source_view_ids,
        occupancy_metadata_sha256=(
            semantic_attestation.occupancy_metadata_sha256
            if semantic_attestation is not None
            else None
        ),
        semantic_verifier_contract_hash=(
            semantic_attestation.semantic_verifier_contract_hash
            if semantic_attestation is not None
            else None
        ),
        semantic_attestation_hash=(
            semantic_attestation.attestation_hash if semantic_attestation is not None else None
        ),
    )


@dataclass(slots=True)
class OccupancyRobotCollisionChecker:
    """Query original URDF collision STLs against immutable occupancy voxels.

    HPP-FCL measures each moving collision geometry against exact unions of
    dangerous voxel cells.  Adjacent same-state cells may be represented as one
    axis-aligned run box; the robot remains its original STL and AABBs never decide
    collision.  Fixed base geometry is excluded because its designed support
    contact cannot change during a robot motion segment.
    """

    robot_checker: Cs68PinocchioCollisionChecker
    snapshot_provider: OccupancySnapshotProvider
    maximum_map_age_s: float | None = None
    authorization_started_at_utc: datetime | None = None
    additional_clearance_m: float = 0.0
    ignored_geometry_names: tuple[str, ...] = ()
    accepted_static_free_aabbs: tuple[AcceptedStaticFreeAabb, ...] = ()
    accepted_static_free_acceptance_id: str | None = None
    accepted_static_free_mapping_context_hash: str | None = None
    allow_mapping_prefix_in_accepted_static_free: bool = False
    verified_robot_geometry_hash: str | None = None
    semantic_attestation: OccupancySemanticAttestation | None = None
    accepted_joint_uncertainty_rad: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    motion_envelope_acceptance_id: str | None = None
    motion_envelope_metadata_sha256: str | None = None
    utc_clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC), repr=False)
    _voxel_classification_content_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _voxel_classification: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.maximum_map_age_s is not None and (
            not math.isfinite(self.maximum_map_age_s)
            or self.maximum_map_age_s <= 0.0
        ):
            raise ValueError("maximum_map_age_s must be finite and positive")
        if self.authorization_started_at_utc is not None:
            if self.authorization_started_at_utc.tzinfo is None:
                raise ValueError("authorization_started_at_utc must be timezone-aware")
            self.authorization_started_at_utc = self.authorization_started_at_utc.astimezone(UTC)
        if not math.isfinite(self.additional_clearance_m) or self.additional_clearance_m < 0.0:
            raise ValueError("additional_clearance_m must be finite and non-negative")
        if len(set(self.ignored_geometry_names)) != len(self.ignored_geometry_names):
            raise ValueError("ignored occupancy geometry names must be unique")
        if any(
            type(item) is not AcceptedStaticFreeAabb for item in self.accepted_static_free_aabbs
        ):
            raise ValueError("accepted static-free regions require strong AABB values")
        accepted_names = tuple(item.name for item in self.accepted_static_free_aabbs)
        if len(set(accepted_names)) != len(accepted_names):
            raise ValueError("accepted static-free AABB names must be unique")
        if self.accepted_static_free_aabbs:
            self.accepted_static_free_acceptance_id = _sha256_digest(
                self.accepted_static_free_acceptance_id,
                reason="accepted_static_free_acceptance_id_must_be_sha256",
            )
            self.accepted_static_free_mapping_context_hash = _sha256_digest(
                self.accepted_static_free_mapping_context_hash,
                reason="accepted_static_free_mapping_context_hash_must_be_sha256",
            )
        elif (
            self.accepted_static_free_acceptance_id is not None
            or self.accepted_static_free_mapping_context_hash is not None
        ):
            raise ValueError("static-free acceptance identity/context require at least one AABB")
        if (
            self.allow_mapping_prefix_in_accepted_static_free
            and not self.accepted_static_free_aabbs
        ):
            raise ValueError(
                "mapping-prefix preflight requires accepted static-free AABBs"
            )
        checker_identity = getattr(self.robot_checker, "robot_geometry_hash", None)
        explicit_identity = self.verified_robot_geometry_hash
        if checker_identity is not None:
            checker_identity = _sha256_digest(
                checker_identity,
                reason="occupancy_robot_checker_geometry_hash_must_be_sha256",
            )
        if explicit_identity is not None:
            explicit_identity = _sha256_digest(
                explicit_identity,
                reason="occupancy_verified_robot_geometry_hash_must_be_sha256",
            )
        if checker_identity is None and explicit_identity is None:
            raise ValueError("occupancy checker requires a hash-bound robot geometry identity")
        if (
            checker_identity is not None
            and explicit_identity is not None
            and checker_identity != explicit_identity
        ):
            raise ValueError("explicit occupancy robot geometry differs from robot checker")
        self.verified_robot_geometry_hash = checker_identity or explicit_identity
        if self.semantic_attestation is not None:
            if type(self.semantic_attestation) is not OccupancySemanticAttestation:
                raise ValueError("occupancy semantic attestation has invalid type")
            if self.semantic_attestation.robot_geometry_hash != self.verified_robot_geometry_hash:
                raise ValueError(
                    "occupancy semantic attestation robot geometry differs from checker"
                )
        uncertainty = np.asarray(self.accepted_joint_uncertainty_rad, dtype=np.float64)
        if (
            uncertainty.shape != (6,)
            or not np.isfinite(uncertainty).all()
            or np.any(uncertainty < 0.0)
        ):
            raise ValueError("accepted joint uncertainty must be a non-negative six-vector")
        acceptance_id = self.motion_envelope_acceptance_id
        if np.any(uncertainty > 0.0):
            acceptance_id = _sha256_digest(
                acceptance_id,
                reason="motion_envelope_acceptance_id_must_be_sha256",
            )
            metadata_sha256 = _sha256_digest(
                self.motion_envelope_metadata_sha256,
                reason="motion_envelope_metadata_sha256_must_be_sha256",
            )
        elif acceptance_id is not None:
            raise ValueError("motion-envelope identity cannot bind a zero uncertainty vector")
        elif self.motion_envelope_metadata_sha256 is not None:
            raise ValueError("motion-envelope metadata cannot bind a zero uncertainty vector")
        else:
            metadata_sha256 = None
        self.accepted_joint_uncertainty_rad = tuple(float(value) for value in uncertainty)
        self.motion_envelope_acceptance_id = acceptance_id
        self.motion_envelope_metadata_sha256 = metadata_sha256

    @property
    def semantic_attestation_hash(self) -> str | None:
        attestation = self.semantic_attestation
        return attestation.attestation_hash if attestation is not None else None

    @property
    def motion_semantic_attestation_valid(self) -> bool:
        attestation = self.semantic_attestation
        return bool(
            type(attestation) is OccupancySemanticAttestation
            and attestation.semantic_verifier_contract_hash
            == OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH
            and attestation.robot_geometry_hash == self.verified_robot_geometry_hash
        )

    @property
    def continuous_swept_volume_supported(self) -> bool:
        """Exact midpoint distances plus link displacement bounds cover sweeps."""

        return True

    @property
    def policy_contract_hash(self) -> str:
        """Identity of every occupancy-query rule relevant to motion safety."""

        payload = {
            "schema": "biblade_fusion.occupancy_robot_collision_policy.v8",
            "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
            "path_semantic": "sampled_or_offline_continuous_original_stl_clearance",
            "continuous_swept_volume_supported": (self.continuous_swept_volume_supported),
            "robot_geometry_hash": self.verified_robot_geometry_hash,
            "robot_motion_bound_contract_sha256": (
                self.robot_checker.geometry_motion_bound_contract_sha256
            ),
            "motion_envelope_acceptance_id": self.motion_envelope_acceptance_id,
            "motion_envelope_metadata_sha256": self.motion_envelope_metadata_sha256,
            "accepted_joint_uncertainty_rad": list(self.accepted_joint_uncertainty_rad),
            "maximum_map_age_s": self.maximum_map_age_s,
            "authorization_started_at_utc": (
                self.authorization_started_at_utc.isoformat()
                if self.authorization_started_at_utc is not None
                else None
            ),
            "additional_clearance_m": self.additional_clearance_m,
            "ignored_geometry_names": list(self.ignored_geometry_names),
            "accepted_static_free": {
                "acceptance_id": self.accepted_static_free_acceptance_id,
                "required_mapping_context_hash": (self.accepted_static_free_mapping_context_hash),
                "whole_voxel_aabb_containment_required": True,
                "occupied_never_downgraded": True,
                "aabbs": [
                    {
                        "name": item.name,
                        "minimum_m": list(item.minimum_m),
                        "maximum_m": list(item.maximum_m),
                    }
                    for item in self.accepted_static_free_aabbs
                ],
            },
            "mapping_prefix_policy": {
                "enabled": self.allow_mapping_prefix_in_accepted_static_free,
                "accepted_static_free_only": True,
                "unknown_outside_acceptance_blocks": True,
                "robot_geometry": "original_urdf_collision_stl",
            },
            "voxel_geometry": "hppfcl_axis_aligned_box",
            "clearance_semantic": "exact_distance_greater_than_required_margin",
            "fixed_geometry_policy": {
                "excluded_parent_joint": 0,
                "reason": "fixed_base_support_contact_does_not_change_during_motion",
            },
            "unknown_is_occupied": True,
            "semantic_attestation_required_for_motion": True,
            "semantic_verifier_contract_hash": (OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH),
            "semantic_attestation_hash": self.semantic_attestation_hash,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def current_evidence(
        self, *, required_freshness_horizon_s: float = 0.0
    ) -> OccupancyMapEvidence:
        try:
            snapshot = self.snapshot_provider()
            evidence = occupancy_evidence_from_snapshot(
                snapshot,
                now_utc=self.utc_clock(),
                max_age_s=self.maximum_map_age_s,
                authorization_started_at_utc=self.authorization_started_at_utc,
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
                allow_mapping_prefix=(
                    self.allow_mapping_prefix_in_accepted_static_free
                ),
            )
            self._assert_static_free_acceptance_context(evidence)
            return evidence
        except OccupancyEvidenceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            raise OccupancyEvidenceError(
                f"occupancy_snapshot_provider_failed:{type(exc).__name__}:{exc}"
            ) from exc

    def assert_current_evidence(
        self,
        expected: OccupancyMapEvidence,
        *,
        required_freshness_horizon_s: float = 0.0,
    ) -> None:
        current = self.current_evidence(required_freshness_horizon_s=required_freshness_horizon_s)
        if current.binding != expected.binding:
            raise OccupancyEvidenceError(
                "occupancy_snapshot_changed:"
                f"expected={expected.sequence}:{expected.content_hash}:"
                f"current={current.sequence}:{current.content_hash}"
            )

    def check(
        self,
        joint_positions_rad: Sequence[float],
        *,
        expected_evidence: OccupancyMapEvidence | None = None,
        required_freshness_horizon_s: float = 0.0,
    ) -> OccupancyCollisionCheckResult:
        try:
            snapshot, evidence = self._bind_snapshot(
                expected_evidence=expected_evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            result = self._check_bound_configuration(
                snapshot,
                evidence,
                joint_positions_rad,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            self.assert_current_evidence(
                evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            return result
        except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
            return self._unknown_result(f"occupancy_checker_error:{exc}")
        except Exception as exc:  # pragma: no cover - fail closed across plugin boundaries
            return self._unknown_result(
                f"occupancy_query_failed:{type(exc).__name__}:{exc}"
            )

    @contextmanager
    def bind_configuration_queries(
        self,
        *,
        required_freshness_horizon_s: float = 0.0,
    ) -> Iterator[BoundOccupancyConfigurationQuery]:
        """Bind one immutable map for an OMPL/state-search transaction."""

        snapshot, evidence = self._bind_snapshot(
            expected_evidence=None,
            required_freshness_horizon_s=required_freshness_horizon_s,
        )
        query = BoundOccupancyConfigurationQuery(
            self,
            snapshot,
            evidence,
            float(required_freshness_horizon_s),
        )
        try:
            yield query
        finally:
            query.close()

    def check_sampled_configurations(
        self,
        configurations: Sequence[Sequence[float]],
        path_fractions: Sequence[float],
        *,
        maximum_joint_step_rad: float,
        required_freshness_horizon_s: float = 0.0,
        precheck_last_configuration: bool = False,
    ) -> JointPathOccupancyCollisionReport:
        """Check a sampled path while hashing and binding its immutable map once.

        The public :meth:`check` method intentionally verifies snapshot integrity
        before and after one independent query.  Calling it once per HoloRobot path
        sample used to recompute the complete voxel-set SHA-256 twice per sample.
        A path is one transaction: bind once, query every sampled pose against that
        exact frozen object, then verify the publisher still exposes the same map.
        """

        samples = tuple(tuple(float(value) for value in item) for item in configurations)
        fractions = tuple(float(value) for value in path_fractions)
        step = float(maximum_joint_step_rad)
        if not samples or len(samples) != len(fractions):
            raise ValueError("sampled occupancy path requires aligned non-empty inputs")
        if any(len(item) != 6 or not np.isfinite(item).all() for item in samples):
            raise ValueError("sampled occupancy path configurations must be finite six-vectors")
        if not np.isfinite(fractions).all() or any(
            fraction < 0.0 or fraction > 1.0 for fraction in fractions
        ):
            raise ValueError("sampled occupancy path fractions must be finite in [0, 1]")
        if any(
            later < earlier
            for earlier, later in zip(fractions, fractions[1:], strict=False)
        ):
            raise ValueError("sampled occupancy path fractions must be monotonic")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_joint_step_rad must be finite and positive")
        try:
            snapshot, evidence = self._bind_snapshot(
                expected_evidence=None,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
        except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
            result = self._unknown_result(f"occupancy_checker_error:{exc}")
            return JointPathOccupancyCollisionReport(
                status=result.status,
                sample_count=0,
                blocked_sample_index=0,
                blocked_path_fraction=0.0,
                result=result,
                maximum_joint_step_rad=step,
            )

        last_result: OccupancyCollisionCheckResult | None = None
        ordered_indices = (
            (len(samples) - 1, *range(len(samples) - 1))
            if precheck_last_configuration
            else tuple(range(len(samples)))
        )
        for checked_count, sample_index in enumerate(ordered_indices, start=1):
            configuration = samples[sample_index]
            fraction = fractions[sample_index]
            try:
                result = self._check_bound_configuration(
                    snapshot,
                    evidence,
                    configuration,
                    required_freshness_horizon_s=required_freshness_horizon_s,
                    stop_on_first_block=True,
                )
            except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
                result = self._unknown_result(
                    f"occupancy_checker_error:{exc}",
                    evidence=evidence,
                )
            except Exception as exc:  # pragma: no cover - plugin boundary
                result = self._unknown_result(
                    f"occupancy_query_failed:{type(exc).__name__}:{exc}",
                    evidence=evidence,
                )
            last_result = result
            if result.status is not CollisionCheckStatus.CLEAR:
                try:
                    self.assert_current_evidence(
                        evidence,
                        required_freshness_horizon_s=required_freshness_horizon_s,
                    )
                except OccupancyEvidenceError as exc:
                    result = self._unknown_result(
                        f"occupancy_checker_error:{exc}",
                        evidence=evidence,
                    )
                return self._finish_sampled_path(
                    result,
                    sample_count=checked_count,
                    blocked_sample_index=sample_index,
                    blocked_path_fraction=fraction,
                    maximum_joint_step_rad=step,
                    total_sample_count=len(samples),
                )

        assert last_result is not None
        try:
            self.assert_current_evidence(
                evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
        except OccupancyEvidenceError as exc:
            changed = self._unknown_result(
                f"occupancy_checker_error:{exc}",
                evidence=evidence,
            )
            return self._finish_sampled_path(
                changed,
                sample_count=len(samples),
                blocked_sample_index=len(samples) - 1,
                blocked_path_fraction=fractions[-1],
                maximum_joint_step_rad=step,
                total_sample_count=len(samples),
            )
        return self._finish_sampled_path(
            last_result,
            sample_count=len(samples),
            blocked_sample_index=None,
            blocked_path_fraction=None,
            maximum_joint_step_rad=step,
            total_sample_count=len(samples),
        )

    def _bind_snapshot(
        self,
        *,
        expected_evidence: OccupancyMapEvidence | None,
        required_freshness_horizon_s: float,
    ) -> tuple[OccupancySnapshot, OccupancyMapEvidence]:
        snapshot = self.snapshot_provider()
        evidence = occupancy_evidence_from_snapshot(
            snapshot,
            now_utc=self.utc_clock(),
            max_age_s=self.maximum_map_age_s,
            authorization_started_at_utc=self.authorization_started_at_utc,
            required_freshness_horizon_s=required_freshness_horizon_s,
            verified_robot_geometry_hash=self.verified_robot_geometry_hash,
            semantic_attestation=self.semantic_attestation,
            allow_mapping_prefix=self.allow_mapping_prefix_in_accepted_static_free,
        )
        self._assert_static_free_acceptance_context(evidence)
        if expected_evidence is not None and evidence.binding != expected_evidence.binding:
            raise OccupancyEvidenceError(
                "occupancy_snapshot_changed_before_query:"
                f"expected={expected_evidence.sequence}:{expected_evidence.content_hash}:"
                f"current={evidence.sequence}:{evidence.content_hash}"
            )
        return snapshot, evidence

    def _check_bound_configuration(
        self,
        snapshot: OccupancySnapshot,
        evidence: OccupancyMapEvidence,
        joint_positions_rad: Sequence[float],
        *,
        required_freshness_horizon_s: float,
        stop_on_first_block: bool = False,
    ) -> OccupancyCollisionCheckResult:
        geometries = self._robot_collision_geometries(joint_positions_rad)
        reasons: list[str] = []
        query_diagnostics: list[dict[str, Any]] = []
        for placed in geometries:
            uncertainty = self.robot_checker.geometry_displacement_bound_m(
                placed.geometry_index,
                self.accepted_joint_uncertainty_rad,
            )
            required_distance = self.additional_clearance_m + uncertainty
            query = self._query_robot_geometry(
                snapshot,
                placed,
                required_distance_m=required_distance,
            )
            state = self._validate_query_result(query)
            query_diagnostics.append(
                {
                    "geometry": placed.geometry_name,
                    "state": state,
                    "geometry_representation": "original_urdf_collision_stl",
                    "required_distance_m": required_distance,
                    "occupied_count": int(query.occupied_count),
                    "unknown_count": int(query.unknown_count),
                    "free_count": int(query.free_count),
                    "accepted_unknown_count": int(query.accepted_unknown_count),
                    "outside_grid_unknown_count": int(query.outside_grid_unknown_count),
                    "outside_acceptance_unknown_count": int(
                        query.outside_acceptance_unknown_count
                    ),
                    "separated_dangerous_count": int(query.separated_dangerous_count),
                    "distance_query_count": int(query.distance_query_count),
                    "minimum_dangerous_distance_m": query.minimum_dangerous_distance_m,
                    "blocking_voxel_index": query.blocking_voxel_index,
                    "queried_count": int(query.queried_count),
                }
            )
            if state == OccupancyQueryState.FREE.value and not bool(query.blocked):
                continue
            if state == OccupancyQueryState.OCCUPIED.value:
                reason = f"environment_occupancy_occupied:{placed.geometry_name}"
            elif state == OccupancyQueryState.UNKNOWN.value:
                reason = f"environment_occupancy_unknown:{placed.geometry_name}"
            else:
                reason = (
                    "environment_occupancy_invalid_query_state:"
                    f"{placed.geometry_name}:{state}"
                )
            reasons.append(reason)
            # Match HoloRobot's online state/path behavior: one unsafe robot
            # geometry is already a complete veto.  Evaluating all remaining
            # STLs only enriches diagnostics and made a known-invalid eiai goal
            # spend 20.94 seconds in one occupancy call.  Standalone diagnostic
            # queries keep the all-geometry default.
            if stop_on_first_block:
                break
        return OccupancyCollisionCheckResult(
            status=(CollisionCheckStatus.BLOCKED if reasons else CollisionCheckStatus.CLEAR),
            blocking_reasons=tuple(reasons),
            evidence=evidence,
            checked_geometry_count=len(query_diagnostics),
            diagnostics={
                "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
                "occupancy_policy_contract_hash": self.policy_contract_hash,
                "robot_motion_bound_contract_sha256": (
                    self.robot_checker.geometry_motion_bound_contract_sha256
                ),
                "unknown_policy": "conservative",
                "robot_geometry": "original_urdf_collision_stl",
                "additional_clearance_m": self.additional_clearance_m,
                "required_freshness_horizon_s": required_freshness_horizon_s,
                "fail_fast": bool(stop_on_first_block),
                "continuous_swept_volume_verified": False,
                "semantic_attestation_valid": evidence.semantic_attestation_valid,
                "semantic_attestation_hash": evidence.semantic_attestation_hash,
                "queries": query_diagnostics,
                "motion_authorized": False,
            },
        )

    def _unknown_result(
        self,
        reason: str,
        *,
        evidence: OccupancyMapEvidence | None = None,
    ) -> OccupancyCollisionCheckResult:
        return OccupancyCollisionCheckResult(
            status=CollisionCheckStatus.UNKNOWN,
            blocking_reasons=(reason,),
            evidence=evidence,
            checked_geometry_count=0,
            diagnostics={
                "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
                "occupancy_policy_contract_hash": self.policy_contract_hash,
                "unknown_policy": "conservative",
                "continuous_swept_volume_verified": False,
                "motion_authorized": False,
            },
        )

    @staticmethod
    def _finish_sampled_path(
        result: OccupancyCollisionCheckResult,
        *,
        sample_count: int,
        blocked_sample_index: int | None,
        blocked_path_fraction: float | None,
        maximum_joint_step_rad: float,
        total_sample_count: int,
    ) -> JointPathOccupancyCollisionReport:
        clear = result.status is CollisionCheckStatus.CLEAR
        enriched = OccupancyCollisionCheckResult(
            status=result.status,
            blocking_reasons=result.blocking_reasons,
            evidence=result.evidence,
            checked_geometry_count=result.checked_geometry_count,
            diagnostics={
                **result.diagnostics,
                "path_validation_mode": "holorobot_sampled_joint_v2",
                "sampled_path_verified": clear,
                "sample_count": total_sample_count,
            },
        )
        return JointPathOccupancyCollisionReport(
            status=result.status,
            sample_count=sample_count,
            blocked_sample_index=blocked_sample_index,
            blocked_path_fraction=blocked_path_fraction,
            result=enriched,
            maximum_joint_step_rad=maximum_joint_step_rad,
        )

    def check_path(
        self,
        start_joint_positions_rad: Sequence[float],
        end_joint_positions_rad: Sequence[float],
        *,
        maximum_joint_step_rad: float,
        expected_evidence: OccupancyMapEvidence | None = None,
        required_freshness_horizon_s: float = 0.0,
        maximum_subdivision_depth: int = 14,
        minimum_interval_joint_span_rad: float = 1e-7,
    ) -> JointPathOccupancyCollisionReport:
        """Prove the robot's complete swept envelope against one immutable map.

        At every midpoint the original URDF collision STL is measured directly
        against dangerous voxel boxes with HPP-FCL.  An interval is certified only
        when every exact separation exceeds clearance, accepted tracking uncertainty,
        and the serial-chain displacement bound for that interval.  Otherwise it is
        bisected; a limit without an actual-pose witness returns ``UNKNOWN``.
        """

        start = np.asarray(start_joint_positions_rad, dtype=np.float64)
        end = np.asarray(end_joint_positions_rad, dtype=np.float64)
        if start.shape != (6,) or end.shape != (6,) or not np.isfinite((start, end)).all():
            raise ValueError("occupancy path endpoints must be finite six-vectors")
        step = float(maximum_joint_step_rad)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_joint_step_rad must be finite and positive")
        depth_limit = int(maximum_subdivision_depth)
        if depth_limit < 0:
            raise ValueError("maximum_subdivision_depth must be non-negative")
        minimum_span = float(minimum_interval_joint_span_rad)
        if not math.isfinite(minimum_span) or minimum_span <= 0.0:
            raise ValueError("minimum_interval_joint_span_rad must be finite and positive")
        segment_count = max(1, math.ceil(float(np.max(np.abs(end - start))) / step))
        evaluated = 0
        checked_geometries = 0
        distance_queries = 0
        accepted_unknown_queries = 0
        certified = 0
        deepest = 0
        try:
            snapshot = self.snapshot_provider()
            bound_evidence = occupancy_evidence_from_snapshot(
                snapshot,
                now_utc=self.utc_clock(),
                max_age_s=self.maximum_map_age_s,
                authorization_started_at_utc=self.authorization_started_at_utc,
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
                allow_mapping_prefix=(
                    self.allow_mapping_prefix_in_accepted_static_free
                ),
            )
            self._assert_static_free_acceptance_context(bound_evidence)
            if (
                expected_evidence is not None
                and bound_evidence.binding != expected_evidence.binding
            ):
                raise OccupancyEvidenceError("occupancy_snapshot_changed_before_swept_query")
        except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
            failure = OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=(f"occupancy_checker_error:{exc}",),
                evidence=None,
                checked_geometry_count=0,
                diagnostics={
                    "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "continuous_swept_volume_verified": False,
                    "motion_authorized": False,
                },
            )
            return JointPathOccupancyCollisionReport(
                status=CollisionCheckStatus.UNKNOWN,
                sample_count=0,
                blocked_sample_index=0,
                blocked_path_fraction=0.0,
                result=failure,
                maximum_joint_step_rad=step,
            )

        path_hash = _joint_path_sha256(start, end)
        map_hash = _occupancy_evidence_binding_sha256(bound_evidence)
        motion_bound_hash = self.robot_checker.geometry_motion_bound_contract_sha256

        def issue_evidence(reason: str) -> SweptOccupancyProofEvidence:
            provisional = SweptOccupancyProofEvidence(
                trajectory_sha256=path_hash,
                map_binding_sha256=map_hash,
                occupancy_policy_contract_hash=self.policy_contract_hash,
                robot_motion_bound_contract_sha256=motion_bound_hash,
                motion_envelope_acceptance_id=self.motion_envelope_acceptance_id,
                motion_envelope_metadata_sha256=self.motion_envelope_metadata_sha256,
                accepted_joint_uncertainty_rad=self.accepted_joint_uncertainty_rad,
                maximum_joint_step_rad=step,
                maximum_subdivision_depth=depth_limit,
                minimum_interval_joint_span_rad=minimum_span,
                initial_interval_count=segment_count,
                certified_interval_count=certified,
                evaluated_configuration_count=evaluated,
                geometry_voxel_distance_query_count=distance_queries,
                accepted_unknown_voxel_query_count=accepted_unknown_queries,
                deepest_subdivision=deepest,
                termination_reason=reason,
                evidence_sha256="",
            )
            return SweptOccupancyProofEvidence(
                **{
                    name: getattr(provisional, name)
                    for name in provisional.__dataclass_fields__
                    if name != "evidence_sha256"
                },
                evidence_sha256=_canonical_sha256(provisional._payload()),
            )

        def finish(
            result: OccupancyCollisionCheckResult,
            *,
            fraction: float,
            reason: str,
        ) -> JointPathOccupancyCollisionReport:
            evidence = issue_evidence(reason)
            enriched = OccupancyCollisionCheckResult(
                status=result.status,
                blocking_reasons=result.blocking_reasons,
                evidence=result.evidence or bound_evidence,
                checked_geometry_count=checked_geometries,
                diagnostics={
                    **result.diagnostics,
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "continuous_swept_volume_verified": False,
                    "swept_occupancy_proof_evidence_sha256": (evidence.evidence_sha256),
                    "swept_occupancy_termination_reason": reason,
                    "motion_authorized": False,
                },
            )
            return JointPathOccupancyCollisionReport(
                status=result.status,
                sample_count=evaluated,
                blocked_sample_index=max(0, evaluated - 1),
                blocked_path_fraction=float(fraction),
                result=enriched,
                maximum_joint_step_rad=step,
                continuous_swept_volume_verified=False,
                proof_evidence=evidence,
            )

        checked_fractions: dict[float, OccupancyCollisionCheckResult] = {}

        def blocking_termination_reason(
            result: OccupancyCollisionCheckResult,
        ) -> str:
            if result.status is not CollisionCheckStatus.BLOCKED:
                return "checker_error"
            if any(
                reason.startswith("environment_occupancy_occupied:")
                for reason in result.blocking_reasons
            ):
                return "occupied_voxel_witness"
            return "unknown_or_policy_block_at_pose"

        def evaluate_actual(fraction: float) -> OccupancyCollisionCheckResult:
            nonlocal accepted_unknown_queries, checked_geometries, distance_queries, evaluated
            key = float(fraction)
            cached = checked_fractions.get(key)
            if cached is not None:
                return cached
            result = self.check(
                start + key * (end - start),
                expected_evidence=bound_evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            checked_fractions[key] = result
            evaluated += 1
            checked_geometries += result.checked_geometry_count
            distance_queries += sum(
                int(item.get("distance_query_count", 0))
                for item in result.diagnostics.get("queries", ())
            )
            accepted_unknown_queries += sum(
                int(item.get("accepted_unknown_count", 0))
                for item in result.diagnostics.get("queries", ())
            )
            return result

        for endpoint_fraction in (0.0, 1.0):
            endpoint = evaluate_actual(endpoint_fraction)
            if endpoint.status is not CollisionCheckStatus.CLEAR:
                return finish(
                    endpoint,
                    fraction=endpoint_fraction,
                    reason=blocking_termination_reason(endpoint),
                )

        if np.array_equal(start, end):
            evidence = issue_evidence("constant_path_configuration_clear")
            clear = OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.CLEAR,
                blocking_reasons=(),
                evidence=bound_evidence,
                checked_geometry_count=checked_geometries,
                diagnostics={
                    "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "robot_motion_bound_contract_sha256": motion_bound_hash,
                    "path_semantic": evidence.method,
                    "continuous_swept_volume_verified": True,
                    "swept_occupancy_proof_evidence_sha256": (evidence.evidence_sha256),
                    "swept_occupancy_termination_reason": evidence.termination_reason,
                    "accepted_unknown_voxel_query_count": (accepted_unknown_queries),
                    "semantic_attestation_valid": (bound_evidence.semantic_attestation_valid),
                    "motion_authorized": False,
                },
            )
            return JointPathOccupancyCollisionReport(
                status=CollisionCheckStatus.CLEAR,
                sample_count=evaluated,
                blocked_sample_index=None,
                blocked_path_fraction=None,
                result=clear,
                maximum_joint_step_rad=step,
                continuous_swept_volume_verified=True,
                proof_evidence=evidence,
            )

        intervals: list[tuple[float, float, int]] = [
            (index / segment_count, (index + 1) / segment_count, 0)
            for index in range(segment_count)
        ]
        while intervals:
            lower_fraction, upper_fraction, depth = intervals.pop()
            deepest = max(deepest, depth)
            midpoint_fraction = (lower_fraction + upper_fraction) / 2.0
            midpoint = start + midpoint_fraction * (end - start)
            lower = start + lower_fraction * (end - start)
            upper = start + upper_fraction * (end - start)
            maximum_deviation = np.abs(upper - lower) / 2.0
            try:
                geometries = self._robot_collision_geometries(midpoint)
                evaluated += 1
                interval_clear = True
                for placed in geometries:
                    displacement = self.robot_checker.geometry_displacement_bound_m(
                        placed.geometry_index,
                        maximum_deviation,
                    )
                    uncertainty = self.robot_checker.geometry_displacement_bound_m(
                        placed.geometry_index,
                        self.accepted_joint_uncertainty_rad,
                    )
                    query = self._query_robot_geometry(
                        snapshot,
                        placed,
                        required_distance_m=(
                            self.additional_clearance_m + uncertainty + displacement
                        ),
                    )
                    distance_queries += query.distance_query_count
                    checked_geometries += 1
                    accepted_unknown_queries += int(getattr(query, "accepted_unknown_count", 0))
                    if self._validate_query_result(query) != OccupancyQueryState.FREE.value:
                        interval_clear = False
                if interval_clear:
                    certified += 1
                    continue
            except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
                unknown = OccupancyCollisionCheckResult(
                    status=CollisionCheckStatus.UNKNOWN,
                    blocking_reasons=(f"continuous_swept_occupancy_proof_error:{exc}",),
                    evidence=bound_evidence,
                    checked_geometry_count=checked_geometries,
                    diagnostics={},
                )
                return finish(
                    unknown,
                    fraction=midpoint_fraction,
                    reason="proof_error",
                )

            midpoint_actual = evaluate_actual(midpoint_fraction)
            if midpoint_actual.status is not CollisionCheckStatus.CLEAR:
                return finish(
                    midpoint_actual,
                    fraction=midpoint_fraction,
                    reason=blocking_termination_reason(midpoint_actual),
                )
            joint_span = float(np.max(np.abs(upper - lower)))
            if depth >= depth_limit or joint_span <= minimum_span:
                unknown = OccupancyCollisionCheckResult(
                    status=CollisionCheckStatus.UNKNOWN,
                    blocking_reasons=("continuous_swept_occupancy_unproven:subdivision_limit",),
                    evidence=bound_evidence,
                    checked_geometry_count=checked_geometries,
                    diagnostics={},
                )
                return finish(
                    unknown,
                    fraction=midpoint_fraction,
                    reason="subdivision_limit",
                )
            intervals.append((midpoint_fraction, upper_fraction, depth + 1))
            intervals.append((lower_fraction, midpoint_fraction, depth + 1))

        try:
            self.assert_current_evidence(
                bound_evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
        except OccupancyEvidenceError as exc:
            unknown = OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=(f"occupancy_checker_error:{exc}",),
                evidence=None,
                checked_geometry_count=checked_geometries,
                diagnostics={},
            )
            return finish(
                unknown,
                fraction=1.0,
                reason="map_changed_during_proof",
            )
        evidence = issue_evidence("all_intervals_certified")
        clear = OccupancyCollisionCheckResult(
            status=CollisionCheckStatus.CLEAR,
            blocking_reasons=(),
            evidence=bound_evidence,
            checked_geometry_count=checked_geometries,
            diagnostics={
                "backend": "hppfcl_original_stl_vs_exact_voxel_run_union",
                "occupancy_policy_contract_hash": self.policy_contract_hash,
                "robot_motion_bound_contract_sha256": motion_bound_hash,
                "unknown_policy": "conservative",
                "path_semantic": evidence.method,
                "continuous_swept_volume_verified": True,
                "swept_occupancy_proof_evidence_sha256": evidence.evidence_sha256,
                "swept_occupancy_termination_reason": evidence.termination_reason,
                "certified_interval_count": certified,
                "deepest_subdivision": deepest,
                "accepted_unknown_voxel_query_count": accepted_unknown_queries,
                "semantic_attestation_valid": (bound_evidence.semantic_attestation_valid),
                "semantic_attestation_hash": (bound_evidence.semantic_attestation_hash),
                "required_freshness_horizon_s": required_freshness_horizon_s,
                "motion_authorized": False,
            },
        )
        return JointPathOccupancyCollisionReport(
            status=CollisionCheckStatus.CLEAR,
            sample_count=evaluated,
            blocked_sample_index=None,
            blocked_path_fraction=None,
            result=clear,
            maximum_joint_step_rad=step,
            continuous_swept_volume_verified=True,
            proof_evidence=evidence,
        )

    @staticmethod
    def _validate_query_result(query: OccupancyGeometryQueryLike) -> str:
        accepted_unknown = int(getattr(query, "accepted_unknown_count", 0))
        separated_dangerous = int(
            getattr(query, "separated_dangerous_count", 0)
        )
        counts = (
            int(query.occupied_count),
            int(query.unknown_count),
            int(query.free_count),
            accepted_unknown,
            separated_dangerous,
            int(query.queried_count),
        )
        if any(value < 0 for value in counts):
            raise OccupancyEvidenceError("occupancy_query_has_negative_count")
        occupied, unknown, free, accepted_unknown, separated_dangerous, queried = counts
        if queried <= 0 or (
            occupied
            + unknown
            + free
            + accepted_unknown
            + separated_dangerous
            != queried
        ):
            raise OccupancyEvidenceError("occupancy_query_count_contract_invalid")
        expected_state = (
            OccupancyQueryState.OCCUPIED.value
            if occupied
            else OccupancyQueryState.UNKNOWN.value
            if unknown
            else OccupancyQueryState.FREE.value
        )
        state = _enum_value(query.state)
        if state != expected_state:
            raise OccupancyEvidenceError(
                f"occupancy_query_state_count_mismatch:{state}:{expected_state}"
            )
        expected_blocked = bool(occupied or unknown)
        if bool(query.blocked) is not expected_blocked:
            raise OccupancyEvidenceError(
                "occupancy_query_did_not_apply_conservative_unknown_policy"
            )
        return state

    def _assert_static_free_acceptance_context(
        self,
        evidence: OccupancyMapEvidence,
    ) -> None:
        if not self.accepted_static_free_aabbs:
            return
        if evidence.mapping_context_hash != self.accepted_static_free_mapping_context_hash:
            raise OccupancyEvidenceError(
                "accepted_static_free_mapping_context_does_not_match_snapshot"
            )

    def _query_robot_geometry(
        self,
        snapshot: OccupancySnapshot,
        placed: _PlacedRobotCollisionGeometry,
        *,
        required_distance_m: float,
    ) -> _RobotGeometryVoxelQuery:
        """Measure the original STL against exact unions of dangerous voxel runs.

        Adjacent voxels with the same conservative state are merged only along X.
        The resulting boxes cover exactly the same occupied/unknown volume, so this
        removes per-voxel FCL calls without replacing the robot by a sphere or
        weakening UNKNOWN-as-blocked semantics.
        """

        try:
            import hppfcl
        except ImportError as exc:  # pragma: no cover - Pinocchio already requires it
            raise ImportError("hpp-fcl is required for exact STL occupancy checking") from exc
        margin = float(required_distance_m)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("required STL-to-voxel distance must be non-negative")
        voxel_size = float(snapshot.voxel_size_m)
        origin = np.asarray(snapshot.origin_m, dtype=np.float64)
        broadphase_minimum = np.asarray(
            placed.world_aabb_minimum_m,
            dtype=np.float64,
        ) - margin
        broadphase_maximum = np.asarray(
            placed.world_aabb_maximum_m,
            dtype=np.float64,
        ) + margin
        lower = np.floor((broadphase_minimum - origin) / voxel_size).astype(np.int64) - 1
        upper = np.floor((broadphase_maximum - origin) / voxel_size).astype(np.int64) + 1
        distance_request = hppfcl.DistanceRequest()
        occupied = 0
        unknown = 0
        free = 0
        accepted_unknown = 0
        outside_grid_unknown = 0
        outside_acceptance_unknown = 0
        separated_dangerous = 0
        distance_queries = 0
        minimum_dangerous_distance = math.inf
        blocking_voxel: tuple[int, int, int] | None = None
        tolerance = 1e-9
        # 0=in-grid UNKNOWN, 1=FREE, 2=OCCUPIED, 3=accepted UNKNOWN,
        # 4=out-of-grid UNKNOWN.  Keeping 0 and 4 distinct preserves diagnostics
        # and prevents run merging across the workspace boundary.
        local_shape = tuple(int(upper[axis] - lower[axis] + 1) for axis in range(3))
        local = np.full(local_shape, 4, dtype=np.uint8)
        grid = self._classification_grid(snapshot)
        clipped_lower = np.maximum(lower, 0)
        clipped_upper = np.minimum(upper, np.asarray(snapshot.grid_shape) - 1)
        if np.all(clipped_lower <= clipped_upper):
            source_slices = tuple(
                slice(int(clipped_lower[axis]), int(clipped_upper[axis]) + 1)
                for axis in range(3)
            )
            target_slices = tuple(
                slice(
                    int(clipped_lower[axis] - lower[axis]),
                    int(clipped_upper[axis] - lower[axis]) + 1,
                )
                for axis in range(3)
            )
            local[target_slices] = grid[source_slices]
        tolerance = max(1e-12, voxel_size * 1e-9)
        for region in self.accepted_static_free_aabbs:
            region_lower = np.asarray(
                [
                    math.ceil(
                        (region.minimum_m[axis] - tolerance - origin[axis])
                        / voxel_size
                    )
                    for axis in range(3)
                ],
                dtype=np.int64,
            )
            region_upper = np.asarray(
                [
                    math.floor(
                        (region.maximum_m[axis] + tolerance - origin[axis])
                        / voxel_size
                        - 1.0
                    )
                    for axis in range(3)
                ],
                dtype=np.int64,
            )
            accepted_lower = np.maximum(lower, region_lower)
            accepted_upper = np.minimum(upper, region_upper)
            if np.any(accepted_lower > accepted_upper):
                continue
            accepted_slices = tuple(
                slice(
                    int(accepted_lower[axis] - lower[axis]),
                    int(accepted_upper[axis] - lower[axis]) + 1,
                )
                for axis in range(3)
            )
            accepted_view = local[accepted_slices]
            accepted_view[(accepted_view == 0) | (accepted_view == 4)] = 3

        free = int(np.count_nonzero(local == 1))
        accepted_unknown = int(np.count_nonzero(local == 3))
        for local_y in range(local_shape[1]):
            for local_z in range(local_shape[2]):
                row = local[:, local_y, local_z]
                local_x = 0
                while local_x < local_shape[0]:
                    category = int(row[local_x])
                    if category in {1, 3}:
                        local_x += 1
                        continue
                    run_start = local_x
                    local_x += 1
                    while local_x < local_shape[0] and int(row[local_x]) == category:
                        local_x += 1
                    run_length = local_x - run_start
                    first_index = np.asarray(
                        (
                            int(lower[0] + run_start),
                            int(lower[1] + local_y),
                            int(lower[2] + local_z),
                        ),
                        dtype=np.int64,
                    )
                    dimensions = np.asarray(
                        (run_length * voxel_size, voxel_size, voxel_size),
                        dtype=np.float64,
                    )
                    center = origin + first_index * voxel_size + dimensions / 2.0
                    voxel_geometry = hppfcl.Box(*dimensions)
                    voxel_transform = hppfcl.Transform3f(np.eye(3), center)
                    distance_result = hppfcl.DistanceResult()
                    distance = float(
                        hppfcl.distance(
                            placed.collision_geometry,
                            placed.transform_base,
                            voxel_geometry,
                            voxel_transform,
                            distance_request,
                            distance_result,
                        )
                    )
                    distance_queries += 1
                    if not math.isfinite(distance):
                        raise ValueError(
                            f"non-finite STL-to-voxel-run distance for {placed.geometry_name}"
                        )
                    if distance < -1e100:
                        distance = 0.0
                    minimum_dangerous_distance = min(
                        minimum_dangerous_distance,
                        distance,
                    )
                    if distance > margin + tolerance:
                        separated_dangerous += run_length
                        continue
                    # A merged run proves clear with one distance call.  If it is
                    # close, refine only that rare run to retain exact diagnostic
                    # voxel counts and the true first blocking voxel.
                    for run_offset in range(run_length):
                        index_array = first_index + np.asarray(
                            (run_offset, 0, 0),
                            dtype=np.int64,
                        )
                        if run_length == 1:
                            voxel_distance = distance
                        else:
                            voxel_center = origin + (
                                index_array.astype(np.float64) + 0.5
                            ) * voxel_size
                            voxel_result = hppfcl.DistanceResult()
                            voxel_distance = float(
                                hppfcl.distance(
                                    placed.collision_geometry,
                                    placed.transform_base,
                                    hppfcl.Box(voxel_size, voxel_size, voxel_size),
                                    hppfcl.Transform3f(np.eye(3), voxel_center),
                                    distance_request,
                                    voxel_result,
                                )
                            )
                            distance_queries += 1
                            if voxel_distance < -1e100:
                                voxel_distance = 0.0
                            if not math.isfinite(voxel_distance):
                                raise ValueError(
                                    "non-finite STL-to-voxel distance for "
                                    f"{placed.geometry_name}"
                                )
                            minimum_dangerous_distance = min(
                                minimum_dangerous_distance,
                                voxel_distance,
                            )
                        if voxel_distance > margin + tolerance:
                            separated_dangerous += 1
                            continue
                        index = tuple(int(value) for value in index_array)
                        if category == 2:
                            occupied += 1
                        else:
                            unknown += 1
                            if category == 4:
                                outside_grid_unknown += 1
                            else:
                                outside_acceptance_unknown += 1
                        if blocking_voxel is None:
                            blocking_voxel = index
        queried = (
            occupied
            + unknown
            + free
            + accepted_unknown
            + separated_dangerous
        )
        state = (
            OccupancyQueryState.OCCUPIED
            if occupied
            else OccupancyQueryState.UNKNOWN
            if unknown
            else OccupancyQueryState.FREE
        )
        return _RobotGeometryVoxelQuery(
            state=state,
            blocked=bool(occupied or unknown),
            occupied_count=occupied,
            unknown_count=unknown,
            free_count=free,
            accepted_unknown_count=accepted_unknown,
            outside_grid_unknown_count=outside_grid_unknown,
            outside_acceptance_unknown_count=outside_acceptance_unknown,
            separated_dangerous_count=separated_dangerous,
            distance_query_count=distance_queries,
            minimum_dangerous_distance_m=(
                minimum_dangerous_distance
                if math.isfinite(minimum_dangerous_distance)
                else None
            ),
            blocking_voxel_index=blocking_voxel,
            queried_count=queried,
        )

    def _classification_grid(self, snapshot: OccupancySnapshot) -> np.ndarray:
        cached = self._voxel_classification
        if (
            cached is not None
            and self._voxel_classification_content_hash == snapshot.content_hash
        ):
            return cached
        grid = np.zeros(snapshot.grid_shape, dtype=np.uint8)
        if snapshot.free_indices:
            indices = np.asarray(tuple(snapshot.free_indices), dtype=np.int64)
            grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        if snapshot.occupied_indices:
            indices = np.asarray(tuple(snapshot.occupied_indices), dtype=np.int64)
            grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 2
        self._voxel_classification_content_hash = snapshot.content_hash
        self._voxel_classification = grid
        return grid

    def _robot_collision_geometries(
        self, joint_positions_rad: Sequence[float]
    ) -> tuple[_PlacedRobotCollisionGeometry, ...]:
        """Place the original URDF collision geometries without envelope substitution."""

        try:
            import hppfcl
        except ImportError as exc:  # pragma: no cover - Pinocchio already requires it
            raise ImportError("hpp-fcl is required for exact STL occupancy checking") from exc
        from biblade_fusion.robotics.pinocchio_collision import _require_pinocchio

        joints = self.robot_checker.pinocchio_model._to_configuration(joint_positions_rad)
        pin = _require_pinocchio()
        model = self.robot_checker.pinocchio_model
        pin.forwardKinematics(model.model, model.data, joints)
        pin.updateGeometryPlacements(
            model.model,
            model.data,
            self.robot_checker.geometry_model,
            self.robot_checker.geometry_data,
        )
        ignored = set(self.ignored_geometry_names)
        placed_geometries: list[_PlacedRobotCollisionGeometry] = []
        for index, geometry_object in enumerate(
            self.robot_checker.geometry_model.geometryObjects
        ):
            name = str(geometry_object.name)
            if (
                name.startswith("environment::")
                or name in ignored
                or int(geometry_object.parentJoint) == 0
            ):
                continue
            geometry = geometry_object.geometry
            geometry.computeLocalAABB()
            local_minimum = np.asarray(geometry.aabb_local.min_, dtype=np.float64)
            local_maximum = np.asarray(geometry.aabb_local.max_, dtype=np.float64)
            local_center = (local_minimum + local_maximum) / 2.0
            local_half_extents = (local_maximum - local_minimum) / 2.0
            placement = self.robot_checker.geometry_data.oMg[index]
            rotation = np.asarray(placement.rotation, dtype=np.float64)
            translation = np.asarray(placement.translation, dtype=np.float64)
            world_center = rotation @ local_center + translation
            world_half_extents = np.abs(rotation) @ local_half_extents
            if not np.isfinite((world_center, world_half_extents)).all():
                raise ValueError(f"non-finite STL placement for {name}")
            placed_geometries.append(
                _PlacedRobotCollisionGeometry(
                    geometry_name=name,
                    geometry_index=index,
                    collision_geometry=geometry,
                    transform_base=hppfcl.Transform3f(rotation, translation),
                    world_aabb_minimum_m=tuple(
                        float(value) for value in world_center - world_half_extents
                    ),
                    world_aabb_maximum_m=tuple(
                        float(value) for value in world_center + world_half_extents
                    ),
                )
            )
        if not placed_geometries:
            raise ValueError("robot occupancy checker contains no collision STL geometry")
        return tuple(placed_geometries)
