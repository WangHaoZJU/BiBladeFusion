"""Immutable operator acceptance for UNKNOWN-only static free workcell volumes.

This asset records a physical acceptance; it never authorizes motion by itself.
The guarded segment factory additionally requires a fresh semantic occupancy map,
continuous path proofs, an exact segment approval, and all normal driver gates.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.robotics import AcceptedStaticFreeAabb

STATIC_FREE_ACCEPTANCE_SCHEMA_VERSION = 1
_ASSET_TYPE = "biblade_fusion.static_free_acceptance"
_DECLARATION = (
    "The named volumes were physically inspected as permanently free for the "
    "accepted robot/camera assembly; OCCUPIED evidence remains blocking."
)
_CHECK_NAMES = (
    "final_collision_assembly_verified",
    "workcell_cleared_verified",
    "emergency_stop_verified",
    "low_speed_known_safe_path_verified",
)


@dataclass(frozen=True, slots=True)
class StoredStaticFreeAcceptance:
    path: Path
    acceptance_id: str
    workcell_id: str
    operator_id: str
    accepted_at_utc: datetime
    robot_geometry_hash: str
    workspace_minimum_m: tuple[float, float, float]
    workspace_maximum_m: tuple[float, float, float]
    regions: tuple[AcceptedStaticFreeAabb, ...]
    checklist: tuple[str, ...]
    metadata_sha256: str

    def assert_matches(
        self,
        *,
        acceptance_id: str,
        robot_geometry_hash: str,
        workspace_minimum_m: tuple[float, float, float],
        workspace_maximum_m: tuple[float, float, float],
        regions: tuple[AcceptedStaticFreeAabb, ...],
    ) -> None:
        """Reject a record that is not the exact configured physical acceptance."""

        if self.acceptance_id != acceptance_id:
            raise ValueError("static-free acceptance ID differs from configuration")
        if self.robot_geometry_hash != robot_geometry_hash:
            raise ValueError("static-free acceptance robot geometry differs from runtime")
        if self.workspace_minimum_m != workspace_minimum_m or (
            self.workspace_maximum_m != workspace_maximum_m
        ):
            raise ValueError("static-free acceptance workspace differs from mapping policy")
        if self.regions != regions:
            raise ValueError("static-free acceptance regions differ from mapping policy")


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_load(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("static-free acceptance metadata must be an object")
    return value


def _triplet(value: object, *, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be a three-element array")
    result = tuple(float(item) for item in value)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result  # type: ignore[return-value]


def _validated_payload(
    *,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    robot_geometry_hash: str,
    workspace_minimum_m: tuple[float, float, float],
    workspace_maximum_m: tuple[float, float, float],
    regions: tuple[AcceptedStaticFreeAabb, ...],
    checklist: dict[str, bool],
) -> dict[str, Any]:
    workcell = str(workcell_id).strip()
    operator = str(operator_id).strip()
    if not workcell or not operator:
        raise ValueError("workcell_id and operator_id must be non-empty")
    if accepted_at_utc.tzinfo is None:
        raise ValueError("accepted_at_utc must be timezone-aware")
    timestamp = accepted_at_utc.astimezone(UTC)
    if len(robot_geometry_hash) != 64 or any(
        character not in "0123456789abcdef" for character in robot_geometry_hash
    ):
        raise ValueError("robot_geometry_hash must be a lowercase SHA-256 digest")
    if not regions or len({item.name for item in regions}) != len(regions):
        raise ValueError("at least one uniquely named static-free region is required")
    lower = tuple(float(item) for item in workspace_minimum_m)
    upper = tuple(float(item) for item in workspace_maximum_m)
    if len(lower) != 3 or len(upper) != 3 or not np.isfinite((lower, upper)).all():
        raise ValueError("workspace bounds must be finite triplets")
    if any(high <= low for low, high in zip(lower, upper, strict=True)):
        raise ValueError("workspace bounds are not ordered")
    if set(checklist) != set(_CHECK_NAMES) or not all(
        checklist[name] is True for name in _CHECK_NAMES
    ):
        raise ValueError("all physical static-free acceptance checks must be true")
    for region in regions:
        if any(
            region.minimum_m[axis] < lower[axis] or region.maximum_m[axis] > upper[axis]
            for axis in range(3)
        ):
            raise ValueError("static-free acceptance region lies outside workspace")
    return {
        "schema_version": STATIC_FREE_ACCEPTANCE_SCHEMA_VERSION,
        "asset_type": _ASSET_TYPE,
        "workcell_id": workcell,
        "operator_id": operator,
        "accepted_at_utc": timestamp.isoformat(),
        "robot_geometry_sha256": robot_geometry_hash,
        "workspace_bounds_m": {"minimum": list(lower), "maximum": list(upper)},
        "accepted_static_free_aabbs": [
            {
                "name": region.name,
                "minimum_m": list(region.minimum_m),
                "maximum_m": list(region.maximum_m),
            }
            for region in regions
        ],
        "checklist": {name: True for name in _CHECK_NAMES},
        "declaration": _DECLARATION,
        "motion_authorized": False,
    }


def write_static_free_acceptance(
    path: str | Path,
    *,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    robot_geometry_hash: str,
    workspace_minimum_m: tuple[float, float, float],
    workspace_maximum_m: tuple[float, float, float],
    regions: tuple[AcceptedStaticFreeAabb, ...],
    checklist: dict[str, bool],
) -> StoredStaticFreeAcceptance:
    """Atomically write one declaration, then independently read it back."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"static-free acceptance already exists: {destination}")
    payload = _validated_payload(
        workcell_id=workcell_id,
        operator_id=operator_id,
        accepted_at_utc=accepted_at_utc,
        robot_geometry_hash=robot_geometry_hash,
        workspace_minimum_m=workspace_minimum_m,
        workspace_maximum_m=workspace_maximum_m,
        regions=regions,
        checklist=checklist,
    )
    payload["acceptance_id"] = _sha256_bytes(_canonical_json(payload))
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        metadata = temporary / "metadata.json"
        metadata.write_bytes(_canonical_json(payload) + b"\n")
        with metadata.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return read_static_free_acceptance(destination)


def read_static_free_acceptance(path: str | Path) -> StoredStaticFreeAcceptance:
    """Strictly verify one immutable acceptance record and recompute its identity."""

    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    payload = _strict_load(metadata_path)
    expected_fields = {
        "schema_version",
        "asset_type",
        "workcell_id",
        "operator_id",
        "accepted_at_utc",
        "robot_geometry_sha256",
        "workspace_bounds_m",
        "accepted_static_free_aabbs",
        "checklist",
        "declaration",
        "motion_authorized",
        "acceptance_id",
    }
    if set(payload) != expected_fields:
        raise ValueError("static-free acceptance fields differ from schema")
    acceptance_id = str(payload.pop("acceptance_id"))
    if acceptance_id != _sha256_bytes(_canonical_json(payload)):
        raise ValueError("static-free acceptance identity mismatch")
    if (
        payload["schema_version"] != STATIC_FREE_ACCEPTANCE_SCHEMA_VERSION
        or payload["asset_type"] != _ASSET_TYPE
        or payload["declaration"] != _DECLARATION
        or payload["motion_authorized"] is not False
    ):
        raise ValueError("static-free acceptance schema contract is invalid")
    bounds = payload["workspace_bounds_m"]
    if not isinstance(bounds, dict) or set(bounds) != {"minimum", "maximum"}:
        raise ValueError("static-free acceptance workspace fields are invalid")
    region_values = payload["accepted_static_free_aabbs"]
    if not isinstance(region_values, list):
        raise ValueError("static-free acceptance regions must be an array")
    regions: list[AcceptedStaticFreeAabb] = []
    for item in region_values:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "minimum_m",
            "maximum_m",
        }:
            raise ValueError("static-free acceptance region fields are invalid")
        regions.append(
            AcceptedStaticFreeAabb(
                str(item["name"]),
                _triplet(item["minimum_m"], label="region minimum"),
                _triplet(item["maximum_m"], label="region maximum"),
            )
        )
    checklist = payload["checklist"]
    validated = _validated_payload(
        workcell_id=str(payload["workcell_id"]),
        operator_id=str(payload["operator_id"]),
        accepted_at_utc=datetime.fromisoformat(str(payload["accepted_at_utc"])),
        robot_geometry_hash=str(payload["robot_geometry_sha256"]),
        workspace_minimum_m=_triplet(bounds["minimum"], label="workspace minimum"),
        workspace_maximum_m=_triplet(bounds["maximum"], label="workspace maximum"),
        regions=tuple(regions),
        checklist=(dict(checklist) if isinstance(checklist, dict) else {}),
    )
    if validated != payload:
        raise ValueError("static-free acceptance does not reproduce canonically")
    return StoredStaticFreeAcceptance(
        path=root,
        acceptance_id=acceptance_id,
        workcell_id=str(payload["workcell_id"]),
        operator_id=str(payload["operator_id"]),
        accepted_at_utc=datetime.fromisoformat(str(payload["accepted_at_utc"])),
        robot_geometry_hash=str(payload["robot_geometry_sha256"]),
        workspace_minimum_m=_triplet(bounds["minimum"], label="workspace minimum"),
        workspace_maximum_m=_triplet(bounds["maximum"], label="workspace maximum"),
        regions=tuple(regions),
        checklist=_CHECK_NAMES,
        metadata_sha256=_sha256_path(metadata_path),
    )
