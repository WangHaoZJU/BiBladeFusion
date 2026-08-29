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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from biblade_fusion.mapping.occupancy import (
    OccupancySnapshot,
    compute_content_hash,
    sphere_intersecting_indices,
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


class SphereQueryLike(Protocol):
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
class RobotEnvelopeSphere:
    geometry_name: str
    center_base_m: tuple[float, float, float]
    radius_m: float
    geometry_index: int = -1


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
        lower = tuple(
            snapshot.origin_m[axis] + index[axis] * snapshot.voxel_size_m for axis in range(3)
        )
        upper = tuple(value + snapshot.voxel_size_m for value in lower)
        return all(
            voxel_low >= accepted_low and voxel_high <= accepted_high
            for voxel_low, voxel_high, accepted_low, accepted_high in zip(
                lower,
                upper,
                self.minimum_m,
                self.maximum_m,
                strict=True,
            )
        )


@dataclass(frozen=True, slots=True)
class _AcceptedStaticFreeSphereQuery:
    state: OccupancyQueryState
    blocked: bool
    occupied_count: int
    unknown_count: int
    free_count: int
    accepted_unknown_count: int
    queried_count: int


@dataclass(frozen=True, slots=True)
class OccupancyCollisionCheckResult:
    status: CollisionCheckStatus
    blocking_reasons: tuple[str, ...]
    evidence: OccupancyMapEvidence | None
    checked_sphere_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def motion_authorized(self) -> bool:
        return False


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
    """Integrity-bound proof that expanded robot envelopes cover a full path."""

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
    expanded_sphere_query_count: int
    accepted_unknown_voxel_query_count: int
    deepest_subdivision: int
    termination_reason: str
    evidence_sha256: str
    schema: str = "biblade_fusion.swept_occupancy_proof.v2"
    method: str = "adaptive_midpoint_expanded_tracking_envelope_sphere_sweep"

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
            "expanded_sphere_query_count": self.expanded_sphere_query_count,
            "accepted_unknown_voxel_query_count": (self.accepted_unknown_voxel_query_count),
            "deepest_subdivision": self.deepest_subdivision,
            "termination_reason": self.termination_reason,
        }

    @property
    def integrity_valid(self) -> bool:
        if self.schema != "biblade_fusion.swept_occupancy_proof.v2" or self.method != (
            "adaptive_midpoint_expanded_tracking_envelope_sphere_sweep"
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
    max_age_s: float,
    required_freshness_horizon_s: float = 0.0,
    verified_robot_geometry_hash: str | None = None,
    semantic_attestation: OccupancySemanticAttestation | None = None,
) -> OccupancyMapEvidence:
    """Validate one snapshot and return the identity bound to motion evidence."""

    if type(snapshot) is not OccupancySnapshot:
        raise OccupancyEvidenceError("occupancy_snapshot_must_be_concrete_immutable_snapshot")
    if compute_content_hash(snapshot) != snapshot.content_hash:
        raise OccupancyEvidenceError("occupancy_snapshot_content_hash_mismatch")
    if not math.isfinite(max_age_s) or max_age_s <= 0.0:
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
    if _enum_value(snapshot.map_state) != "map_ready":
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
        usable = bool(snapshot.is_usable_for_preflight(valid_until, max_age_s))
        stale = bool(snapshot.is_stale(valid_until, max_age_s))
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
    """Query conservative Pinocchio geometry envelopes against a map snapshot.

    Each robot collision STL is represented by its transformed local-AABB bounding
    sphere.  This intentionally over-approximates the mesh: false positive blocks are
    acceptable at this safety boundary, while a geometric under-approximation is not.
    """

    robot_checker: Cs68PinocchioCollisionChecker
    snapshot_provider: OccupancySnapshotProvider
    maximum_map_age_s: float = 30.0
    additional_clearance_m: float = 0.0
    ignored_geometry_names: tuple[str, ...] = ()
    accepted_static_free_aabbs: tuple[AcceptedStaticFreeAabb, ...] = ()
    accepted_static_free_acceptance_id: str | None = None
    accepted_static_free_mapping_context_hash: str | None = None
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_map_age_s) or self.maximum_map_age_s <= 0.0:
            raise ValueError("maximum_map_age_s must be finite and positive")
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
        """Adaptive expanded-sphere intervals conservatively cover full sweeps."""

        return True

    @property
    def policy_contract_hash(self) -> str:
        """Identity of every occupancy-query rule relevant to motion safety."""

        payload = {
            "schema": "biblade_fusion.occupancy_robot_collision_policy.v5",
            "backend": "occupancy_snapshot_robot_aabb_spheres",
            "path_semantic": "adaptive_conservative_expanded_sphere_sweep",
            "continuous_swept_volume_supported": (self.continuous_swept_volume_supported),
            "robot_geometry_hash": self.verified_robot_geometry_hash,
            "robot_motion_bound_contract_sha256": (
                self.robot_checker.geometry_motion_bound_contract_sha256
            ),
            "motion_envelope_acceptance_id": self.motion_envelope_acceptance_id,
            "motion_envelope_metadata_sha256": self.motion_envelope_metadata_sha256,
            "accepted_joint_uncertainty_rad": list(self.accepted_joint_uncertainty_rad),
            "maximum_map_age_s": self.maximum_map_age_s,
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
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
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
            snapshot = self.snapshot_provider()
            evidence = occupancy_evidence_from_snapshot(
                snapshot,
                now_utc=self.utc_clock(),
                max_age_s=self.maximum_map_age_s,
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
            )
            self._assert_static_free_acceptance_context(evidence)
            if expected_evidence is not None and evidence.binding != expected_evidence.binding:
                raise OccupancyEvidenceError(
                    "occupancy_snapshot_changed_before_query:"
                    f"expected={expected_evidence.sequence}:{expected_evidence.content_hash}:"
                    f"current={evidence.sequence}:{evidence.content_hash}"
                )
            spheres = self._robot_envelope_spheres(joint_positions_rad)
            reasons: list[str] = []
            query_diagnostics: list[dict[str, Any]] = []
            for sphere in spheres:
                query = self._query_sphere(snapshot, sphere)
                state = self._validate_query_result(query)
                query_diagnostics.append(
                    {
                        "geometry": sphere.geometry_name,
                        "state": state,
                        "radius_m": sphere.radius_m,
                        "occupied_count": int(query.occupied_count),
                        "unknown_count": int(query.unknown_count),
                        "free_count": int(query.free_count),
                        "accepted_unknown_count": int(getattr(query, "accepted_unknown_count", 0)),
                        "queried_count": int(query.queried_count),
                    }
                )
                if state == OccupancyQueryState.FREE.value and not bool(query.blocked):
                    continue
                if state == OccupancyQueryState.OCCUPIED.value:
                    reason = f"environment_occupancy_occupied:{sphere.geometry_name}"
                elif state == OccupancyQueryState.UNKNOWN.value:
                    reason = f"environment_occupancy_unknown:{sphere.geometry_name}"
                else:
                    reason = (
                        f"environment_occupancy_invalid_query_state:{sphere.geometry_name}:{state}"
                    )
                reasons.append(reason)
            self.assert_current_evidence(
                evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            return OccupancyCollisionCheckResult(
                status=(CollisionCheckStatus.BLOCKED if reasons else CollisionCheckStatus.CLEAR),
                blocking_reasons=tuple(reasons),
                evidence=evidence,
                checked_sphere_count=len(spheres),
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "robot_motion_bound_contract_sha256": (
                        self.robot_checker.geometry_motion_bound_contract_sha256
                    ),
                    "unknown_policy": "conservative",
                    "additional_clearance_m": self.additional_clearance_m,
                    "required_freshness_horizon_s": required_freshness_horizon_s,
                    "continuous_swept_volume_verified": False,
                    "semantic_attestation_valid": evidence.semantic_attestation_valid,
                    "semantic_attestation_hash": evidence.semantic_attestation_hash,
                    "queries": query_diagnostics,
                    "motion_authorized": False,
                },
            )
        except (OccupancyEvidenceError, TypeError, ValueError, RuntimeError) as exc:
            return OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=(f"occupancy_checker_error:{exc}",),
                evidence=None,
                checked_sphere_count=0,
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "unknown_policy": "conservative",
                    "continuous_swept_volume_verified": False,
                    "motion_authorized": False,
                },
            )
        except Exception as exc:  # pragma: no cover - fail closed across plugin boundaries
            return OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=(f"occupancy_query_failed:{type(exc).__name__}:{exc}",),
                evidence=None,
                checked_sphere_count=0,
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
                    "unknown_policy": "conservative",
                    "continuous_swept_volume_verified": False,
                    "motion_authorized": False,
                },
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

        Each STL is already over-approximated by a local-AABB sphere.  For an
        interval, the sphere at the midpoint is enlarged by a serial-chain motion
        upper bound, so it contains that geometry for every joint state in the
        interval.  Known-free queries certify the whole interval.  Blocked expanded
        queries trigger bisection; a collision is reported only with an actual-pose
        witness, and a limit without such a witness returns ``UNKNOWN``.
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
        checked_spheres = 0
        expanded_queries = 0
        accepted_unknown_queries = 0
        certified = 0
        deepest = 0
        try:
            snapshot = self.snapshot_provider()
            bound_evidence = occupancy_evidence_from_snapshot(
                snapshot,
                now_utc=self.utc_clock(),
                max_age_s=self.maximum_map_age_s,
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
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
                checked_sphere_count=0,
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
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
                expanded_sphere_query_count=expanded_queries,
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
                checked_sphere_count=checked_spheres,
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
            nonlocal accepted_unknown_queries, checked_spheres, evaluated
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
            checked_spheres += result.checked_sphere_count
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
                checked_sphere_count=checked_spheres,
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
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
                spheres = self._robot_envelope_spheres(midpoint)
                evaluated += 1
                interval_clear = True
                for sphere in spheres:
                    displacement = self.robot_checker.geometry_displacement_bound_m(
                        sphere.geometry_index,
                        maximum_deviation,
                    )
                    query = self._query_sphere(
                        snapshot,
                        RobotEnvelopeSphere(
                            geometry_name=sphere.geometry_name,
                            center_base_m=sphere.center_base_m,
                            radius_m=sphere.radius_m + displacement,
                            geometry_index=sphere.geometry_index,
                        ),
                    )
                    expanded_queries += 1
                    checked_spheres += 1
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
                    checked_sphere_count=checked_spheres,
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
                    checked_sphere_count=checked_spheres,
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
                checked_sphere_count=checked_spheres,
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
            checked_sphere_count=checked_spheres,
            diagnostics={
                "backend": "occupancy_snapshot_robot_aabb_spheres",
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
    def _validate_query_result(query: SphereQueryLike) -> str:
        accepted_unknown = int(getattr(query, "accepted_unknown_count", 0))
        counts = (
            int(query.occupied_count),
            int(query.unknown_count),
            int(query.free_count),
            accepted_unknown,
            int(query.queried_count),
        )
        if any(value < 0 for value in counts):
            raise OccupancyEvidenceError("occupancy_query_has_negative_count")
        occupied, unknown, free, accepted_unknown, queried = counts
        if queried <= 0 or occupied + unknown + free + accepted_unknown != queried:
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

    def _query_sphere(
        self,
        snapshot: OccupancySnapshot,
        sphere: RobotEnvelopeSphere,
    ) -> SphereQueryLike:
        if not self.accepted_static_free_aabbs:
            return snapshot.query_sphere(
                sphere.center_base_m,
                sphere.radius_m,
                unknown_is_occupied=True,
            )
        occupied = 0
        unknown = 0
        free = 0
        accepted_unknown = 0
        for index in sphere_intersecting_indices(
            center_m=sphere.center_base_m,
            radius_m=sphere.radius_m,
            origin_m=snapshot.origin_m,
            voxel_size_m=snapshot.voxel_size_m,
        ):
            state = _enum_value(snapshot.state_at_index(index))
            if state == OccupancyQueryState.OCCUPIED.value:
                occupied += 1
            elif state == OccupancyQueryState.FREE.value:
                free += 1
            elif any(
                region.contains_voxel(snapshot, index) for region in self.accepted_static_free_aabbs
            ):
                accepted_unknown += 1
            else:
                unknown += 1
        queried = occupied + unknown + free + accepted_unknown
        state = (
            OccupancyQueryState.OCCUPIED
            if occupied
            else OccupancyQueryState.UNKNOWN
            if unknown
            else OccupancyQueryState.FREE
        )
        return _AcceptedStaticFreeSphereQuery(
            state=state,
            blocked=bool(occupied or unknown),
            occupied_count=occupied,
            unknown_count=unknown,
            free_count=free,
            accepted_unknown_count=accepted_unknown,
            queried_count=queried,
        )

    def _robot_envelope_spheres(
        self, joint_positions_rad: Sequence[float]
    ) -> tuple[RobotEnvelopeSphere, ...]:
        """Transform each robot STL's local-AABB sphere into ``base``."""

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
        spheres: list[RobotEnvelopeSphere] = []
        for index, geometry_object in enumerate(self.robot_checker.geometry_model.geometryObjects):
            name = str(geometry_object.name)
            if name.startswith("environment::") or name in ignored:
                continue
            geometry = geometry_object.geometry
            geometry.computeLocalAABB()
            local_center = np.asarray(geometry.aabb_center, dtype=np.float64)
            radius = (
                float(geometry.aabb_radius)
                + self.additional_clearance_m
                + self.robot_checker.geometry_displacement_bound_m(
                    index,
                    self.accepted_joint_uncertainty_rad,
                )
            )
            placement = self.robot_checker.geometry_data.oMg[index]
            center = np.asarray(placement.rotation, dtype=np.float64) @ local_center + np.asarray(
                placement.translation, dtype=np.float64
            )
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError(f"non-finite occupancy envelope center for {name}")
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"invalid occupancy envelope radius for {name}")
            spheres.append(
                RobotEnvelopeSphere(
                    geometry_name=name,
                    center_base_m=tuple(float(value) for value in center),
                    radius_m=radius,
                    geometry_index=index,
                )
            )
        if not spheres:
            raise ValueError("robot occupancy envelope contains no geometry")
        return tuple(spheres)
