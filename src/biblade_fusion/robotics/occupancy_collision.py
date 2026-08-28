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

from biblade_fusion.mapping.occupancy import OccupancySnapshot, compute_content_hash
from biblade_fusion.robotics.pinocchio_collision import (
    CollisionCheckStatus,
    Cs68PinocchioCollisionChecker,
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
    "occupancy_mapping_schema": 6,
    "snapshot_type": "biblade_fusion.mapping.occupancy.OccupancySnapshot",
    "snapshot_content_hash": "canonical_sha256_recomputed",
    "verification": (
        "integrity_chain+raw_stereo_source+hand_eye+es68_fk+active_robot_rerender"
    ),
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
        raise TypeError(
            "OccupancySemanticAttestation is issued only by full semantic verification"
        )

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
        if self.semantic_verifier_contract_hash != (
            OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH
        ):
            raise OccupancyEvidenceError(
                "occupancy_semantic_verifier_contract_is_not_current"
            )
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
        raise OccupancyEvidenceError(
            "occupancy_semantic_attestation_requires_concrete_snapshot"
        )
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
        if (
            self.semantic_verifier_contract_hash
            != OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH
        ):
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
    def binding(self) -> tuple[
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


@dataclass(frozen=True, slots=True)
class JointPathOccupancyCollisionReport:
    status: CollisionCheckStatus
    sample_count: int
    blocked_sample_index: int | None
    blocked_path_fraction: float | None
    result: OccupancyCollisionCheckResult
    maximum_joint_step_rad: float
    continuous_swept_volume_verified: bool = False

    @property
    def evidence(self) -> OccupancyMapEvidence | None:
        return self.result.evidence

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def continuous_swept_volume_evidence_valid(self) -> bool:
        return (
            self.continuous_swept_volume_verified
            and self.result.diagnostics.get("continuous_swept_volume_verified")
            is True
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
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
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
        raise OccupancyEvidenceError(
            "occupancy_snapshot_must_be_concrete_immutable_snapshot"
        )
    if compute_content_hash(snapshot) != snapshot.content_hash:
        raise OccupancyEvidenceError("occupancy_snapshot_content_hash_mismatch")
    if not math.isfinite(max_age_s) or max_age_s <= 0.0:
        raise OccupancyEvidenceError("occupancy_max_age_must_be_positive")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise OccupancyEvidenceError("occupancy_check_time_must_be_timezone_aware")
    horizon = float(required_freshness_horizon_s)
    if not math.isfinite(horizon) or horizon < 0.0:
        raise OccupancyEvidenceError(
            "occupancy_freshness_horizon_must_be_non_negative"
        )
    if str(snapshot.frame_id) != "base":
        raise OccupancyEvidenceError(
            f"occupancy_frame_mismatch:{snapshot.frame_id!s}:expected_base"
        )
    if _enum_value(snapshot.map_state) != "map_ready":
        raise OccupancyEvidenceError(
            f"occupancy_map_not_ready:{_enum_value(snapshot.map_state)}"
        )
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
            raise OccupancyEvidenceError(
                "occupancy_semantic_attestation_has_invalid_type"
            )
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
            semantic_attestation.attestation_hash
            if semantic_attestation is not None
            else None
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
    verified_robot_geometry_hash: str | None = None
    semantic_attestation: OccupancySemanticAttestation | None = None
    utc_clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC), repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_map_age_s) or self.maximum_map_age_s <= 0.0:
            raise ValueError("maximum_map_age_s must be finite and positive")
        if (
            not math.isfinite(self.additional_clearance_m)
            or self.additional_clearance_m < 0.0
        ):
            raise ValueError("additional_clearance_m must be finite and non-negative")
        if len(set(self.ignored_geometry_names)) != len(self.ignored_geometry_names):
            raise ValueError("ignored occupancy geometry names must be unique")
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
            raise ValueError(
                "occupancy checker requires a hash-bound robot geometry identity"
            )
        if (
            checker_identity is not None
            and explicit_identity is not None
            and checker_identity != explicit_identity
        ):
            raise ValueError(
                "explicit occupancy robot geometry differs from robot checker"
            )
        self.verified_robot_geometry_hash = checker_identity or explicit_identity
        if self.semantic_attestation is not None:
            if type(self.semantic_attestation) is not OccupancySemanticAttestation:
                raise ValueError("occupancy semantic attestation has invalid type")
            if (
                self.semantic_attestation.robot_geometry_hash
                != self.verified_robot_geometry_hash
            ):
                raise ValueError(
                    "occupancy semantic attestation robot geometry differs from checker"
                )

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
        """The current discrete joint-sampling adapter has no continuous proof."""

        return False

    @property
    def policy_contract_hash(self) -> str:
        """Identity of every occupancy-query rule relevant to motion safety."""

        payload = {
            "schema": "biblade_fusion.occupancy_robot_collision_policy.v2",
            "backend": "occupancy_snapshot_robot_aabb_spheres",
            "path_semantic": "discrete_joint_samples_only",
            "continuous_swept_volume_supported": (
                self.continuous_swept_volume_supported
            ),
            "robot_geometry_hash": self.verified_robot_geometry_hash,
            "maximum_map_age_s": self.maximum_map_age_s,
            "additional_clearance_m": self.additional_clearance_m,
            "ignored_geometry_names": list(self.ignored_geometry_names),
            "unknown_is_occupied": True,
            "semantic_attestation_required_for_motion": True,
            "semantic_verifier_contract_hash": (
                OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH
            ),
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
            return occupancy_evidence_from_snapshot(
                snapshot,
                now_utc=self.utc_clock(),
                max_age_s=self.maximum_map_age_s,
                required_freshness_horizon_s=required_freshness_horizon_s,
                verified_robot_geometry_hash=self.verified_robot_geometry_hash,
                semantic_attestation=self.semantic_attestation,
            )
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
        current = self.current_evidence(
            required_freshness_horizon_s=required_freshness_horizon_s
        )
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
                query = snapshot.query_sphere(
                    sphere.center_base_m,
                    sphere.radius_m,
                    unknown_is_occupied=True,
                )
                state = self._validate_query_result(query)
                query_diagnostics.append(
                    {
                        "geometry": sphere.geometry_name,
                        "state": state,
                        "radius_m": sphere.radius_m,
                        "occupied_count": int(query.occupied_count),
                        "unknown_count": int(query.unknown_count),
                        "free_count": int(query.free_count),
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
                        f"environment_occupancy_invalid_query_state:{sphere.geometry_name}:"
                        f"{state}"
                    )
                reasons.append(reason)
            self.assert_current_evidence(
                evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            return OccupancyCollisionCheckResult(
                status=(
                    CollisionCheckStatus.BLOCKED
                    if reasons
                    else CollisionCheckStatus.CLEAR
                ),
                blocking_reasons=tuple(reasons),
                evidence=evidence,
                checked_sphere_count=len(spheres),
                diagnostics={
                    "backend": "occupancy_snapshot_robot_aabb_spheres",
                    "occupancy_policy_contract_hash": self.policy_contract_hash,
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
    ) -> JointPathOccupancyCollisionReport:
        start = np.asarray(start_joint_positions_rad, dtype=np.float64)
        end = np.asarray(end_joint_positions_rad, dtype=np.float64)
        if (
            start.shape != (6,)
            or end.shape != (6,)
            or not np.isfinite((start, end)).all()
        ):
            raise ValueError("occupancy path endpoints must be finite six-vectors")
        step = float(maximum_joint_step_rad)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_joint_step_rad must be finite and positive")
        segment_count = max(1, math.ceil(float(np.max(np.abs(end - start))) / step))
        bound_evidence = expected_evidence
        checked_spheres = 0
        last_result: OccupancyCollisionCheckResult | None = None
        for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, segment_count + 1)):
            result = self.check(
                start + fraction * (end - start),
                expected_evidence=bound_evidence,
                required_freshness_horizon_s=required_freshness_horizon_s,
            )
            last_result = result
            checked_spheres += result.checked_sphere_count
            if bound_evidence is None and result.evidence is not None:
                bound_evidence = result.evidence
            if result.status is not CollisionCheckStatus.CLEAR:
                return JointPathOccupancyCollisionReport(
                    status=result.status,
                    sample_count=segment_count + 1,
                    blocked_sample_index=sample_index,
                    blocked_path_fraction=float(fraction),
                    result=result,
                    maximum_joint_step_rad=step,
                    continuous_swept_volume_verified=False,
                )
        if last_result is None or bound_evidence is None:  # defensive; loop is non-empty
            clear = OccupancyCollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=("occupancy_path_produced_no_evidence",),
                evidence=None,
                checked_sphere_count=checked_spheres,
            )
            return JointPathOccupancyCollisionReport(
                status=CollisionCheckStatus.UNKNOWN,
                sample_count=segment_count + 1,
                blocked_sample_index=0,
                blocked_path_fraction=0.0,
                result=clear,
                maximum_joint_step_rad=step,
                continuous_swept_volume_verified=False,
            )
        clear = OccupancyCollisionCheckResult(
            status=CollisionCheckStatus.CLEAR,
            blocking_reasons=(),
            evidence=bound_evidence,
            checked_sphere_count=checked_spheres,
            diagnostics={
                **last_result.diagnostics,
                "path_sample_count": segment_count + 1,
                "path_semantic": "discrete_joint_samples_only",
                "continuous_swept_volume_verified": False,
            },
        )
        return JointPathOccupancyCollisionReport(
            status=CollisionCheckStatus.CLEAR,
            sample_count=segment_count + 1,
            blocked_sample_index=None,
            blocked_path_fraction=None,
            result=clear,
            maximum_joint_step_rad=step,
            continuous_swept_volume_verified=False,
        )

    @staticmethod
    def _validate_query_result(query: SphereQueryLike) -> str:
        counts = (
            int(query.occupied_count),
            int(query.unknown_count),
            int(query.free_count),
            int(query.queried_count),
        )
        if any(value < 0 for value in counts):
            raise OccupancyEvidenceError("occupancy_query_has_negative_count")
        occupied, unknown, free, queried = counts
        if queried <= 0 or occupied + unknown + free != queried:
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

    def _robot_envelope_spheres(
        self, joint_positions_rad: Sequence[float]
    ) -> tuple[RobotEnvelopeSphere, ...]:
        """Transform each robot STL's local-AABB sphere into ``base``."""

        from biblade_fusion.robotics.pinocchio_collision import _require_pinocchio

        joints = self.robot_checker.pinocchio_model._to_configuration(
            joint_positions_rad
        )
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
        for index, geometry_object in enumerate(
            self.robot_checker.geometry_model.geometryObjects
        ):
            name = str(geometry_object.name)
            if name.startswith("environment::") or name in ignored:
                continue
            geometry = geometry_object.geometry
            geometry.computeLocalAABB()
            local_center = np.asarray(geometry.aabb_center, dtype=np.float64)
            radius = float(geometry.aabb_radius) + self.additional_clearance_m
            placement = self.robot_checker.geometry_data.oMg[index]
            center = (
                np.asarray(placement.rotation, dtype=np.float64) @ local_center
                + np.asarray(placement.translation, dtype=np.float64)
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
                )
            )
        if not spheres:
            raise ValueError("robot occupancy envelope contains no geometry")
        return tuple(spheres)
