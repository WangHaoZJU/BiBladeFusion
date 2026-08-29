"""HoloRobot-compatible Pinocchio/FCL kinematics and ES68 collision checks.

Adapted from HoloRobot's ``pinocchio_robot_model.py`` and
``pinocchio_collision.py`` at the pinned provenance commit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.settings import CollisionObstacleConfig
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    es68_d435i_collision_content_hash,
    es68_d435i_motion_model_contract_hash,
    es68_d435i_robot_geometry_hash,
    write_es68_d435i_collision_urdf,
)
from biblade_fusion.robotics.cs68_model import (
    CS68_JOINT_NAMES,
    Cs68KinematicModel,
    Cs68ModelResources,
)
from biblade_fusion.robotics.es68_model import Es68KinematicModel
from biblade_fusion.robotics.provenance import robot_stack_provenance
from biblade_fusion.robotics.urdf import write_cs68_urdf


class CollisionCheckStatus(StrEnum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CollisionPairFinding:
    pair_id: str
    link_a: str
    link_b: str
    geometry_a: str
    geometry_b: str
    minimum_distance_m: float | None = None
    required_clearance_m: float | None = None


@dataclass(frozen=True, slots=True)
class CollisionCheckResult:
    status: CollisionCheckStatus
    blocking_reasons: tuple[str, ...] = ()
    pairs: tuple[CollisionPairFinding, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def motion_authorized(self) -> bool:
        return False


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _joint_path_sha256(
    start_joint_positions_rad: Sequence[float],
    end_joint_positions_rad: Sequence[float],
) -> str:
    return _canonical_sha256(
        {
            "schema": "biblade_fusion.linear_joint_path.v1",
            "start_joint_positions_rad": [float(value) for value in start_joint_positions_rad],
            "end_joint_positions_rad": [float(value) for value in end_joint_positions_rad],
        }
    )


def _mesh_model_binding_sha256(diagnostics: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema": "biblade_fusion.swept_mesh_model_binding.v1",
            "model": diagnostics.get("model"),
            "collision_model_id": diagnostics.get("collision_model_id"),
            "collision_model_hash": diagnostics.get("collision_model_hash"),
            "robot_geometry_hash": diagnostics.get("robot_geometry_hash"),
            "motion_model_contract_hash": diagnostics.get("motion_model_contract_hash"),
            "geometry_motion_bound_contract_sha256": diagnostics.get(
                "geometry_motion_bound_contract_sha256"
            ),
            "minimum_clearance_m": diagnostics.get("minimum_clearance_m"),
            "collision_backend_versions": diagnostics.get("collision_backend_versions"),
            "geometry_count": diagnostics.get("geometry_count"),
            "collision_pair_count": diagnostics.get("collision_pair_count"),
        }
    )


@dataclass(frozen=True, slots=True)
class SweptMeshProofEvidence:
    """Integrity-bound certificate for one adaptively proven joint segment.

    This is not a collection of collision-free samples.  Every certified interval
    carries a Lipschitz upper bound on how far either rigid geometry can move away
    from its checked midpoint.  FCL separation at the midpoint must exceed that
    complete two-body motion bound plus ``proof_tolerance_m``.
    """

    trajectory_sha256: str
    model_binding_sha256: str
    motion_bound_contract_sha256: str
    motion_envelope_acceptance_id: str | None
    motion_envelope_metadata_sha256: str | None
    accepted_joint_uncertainty_rad: tuple[float, float, float, float, float, float]
    maximum_joint_step_rad: float
    proof_tolerance_m: float
    maximum_subdivision_depth: int
    minimum_interval_joint_span_rad: float
    initial_interval_count: int
    certified_interval_count: int
    evaluated_configuration_count: int
    deepest_subdivision: int
    minimum_certificate_margin_m: float | None
    termination_reason: str
    evidence_sha256: str
    schema: str = "biblade_fusion.swept_mesh_proof.v2"
    method: str = "adaptive_midpoint_fcl_lipschitz_tracking_envelope_sweep"

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": self.method,
            "trajectory_sha256": self.trajectory_sha256,
            "model_binding_sha256": self.model_binding_sha256,
            "motion_bound_contract_sha256": self.motion_bound_contract_sha256,
            "motion_envelope_acceptance_id": self.motion_envelope_acceptance_id,
            "motion_envelope_metadata_sha256": self.motion_envelope_metadata_sha256,
            "accepted_joint_uncertainty_rad": list(self.accepted_joint_uncertainty_rad),
            "maximum_joint_step_rad": self.maximum_joint_step_rad,
            "proof_tolerance_m": self.proof_tolerance_m,
            "maximum_subdivision_depth": self.maximum_subdivision_depth,
            "minimum_interval_joint_span_rad": (self.minimum_interval_joint_span_rad),
            "initial_interval_count": self.initial_interval_count,
            "certified_interval_count": self.certified_interval_count,
            "evaluated_configuration_count": self.evaluated_configuration_count,
            "deepest_subdivision": self.deepest_subdivision,
            "minimum_certificate_margin_m": self.minimum_certificate_margin_m,
            "termination_reason": self.termination_reason,
        }

    @property
    def integrity_valid(self) -> bool:
        if self.schema != "biblade_fusion.swept_mesh_proof.v2" or self.method != (
            "adaptive_midpoint_fcl_lipschitz_tracking_envelope_sweep"
        ):
            return False
        digests = (
            self.trajectory_sha256,
            self.model_binding_sha256,
            self.motion_bound_contract_sha256,
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
class JointPathMeshCollisionReport:
    status: CollisionCheckStatus
    sample_count: int
    blocked_sample_index: int | None
    blocked_path_fraction: float | None
    result: CollisionCheckResult
    maximum_joint_step_rad: float
    continuous_swept_volume_verified: bool = False
    proof_evidence: SweptMeshProofEvidence | None = None

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def continuous_swept_volume_evidence_valid(self) -> bool:
        evidence = self.proof_evidence
        return bool(
            self.status is CollisionCheckStatus.CLEAR
            and self.result.status is CollisionCheckStatus.CLEAR
            and self.continuous_swept_volume_verified
            and evidence is not None
            and evidence.integrity_valid
            and evidence.termination_reason
            in {"all_intervals_certified", "constant_path_configuration_clear"}
            and evidence.model_binding_sha256 == _mesh_model_binding_sha256(self.result.diagnostics)
            and evidence.motion_bound_contract_sha256
            == self.result.diagnostics.get("geometry_motion_bound_contract_sha256")
            and evidence.evidence_sha256
            == self.result.diagnostics.get("swept_mesh_proof_evidence_sha256")
            and evidence.termination_reason
            == self.result.diagnostics.get("swept_mesh_termination_reason")
            and math.isclose(
                evidence.maximum_joint_step_rad,
                self.maximum_joint_step_rad,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )

    @property
    def motion_authorized(self) -> bool:
        return False


def _require_pinocchio() -> Any:
    try:
        import pinocchio
    except ImportError as exc:
        raise ImportError("Pinocchio/FCL is required for the HoloRobot collision backend") from exc
    return pinocchio


def pinocchio_collision_available() -> bool:
    try:
        _require_pinocchio()
    except ImportError:
        return False
    return True


def _collision_backend_versions(pin: Any) -> dict[str, str]:
    try:
        import hppfcl
    except ImportError as exc:  # pragma: no cover - geometry load already requires FCL
        raise ImportError("hpp-fcl is required for ES68 collision checking") from exc
    return {
        "pinocchio": str(getattr(pin, "__version__", "unknown")),
        "hppfcl": str(getattr(hppfcl, "__version__", "unknown")),
    }


@dataclass(slots=True)
class PinocchioCs68Model:
    """Pinocchio view of the copied HoloRobot CS68 URDF."""

    urdf_path: Path
    model: Any
    data: Any
    tool_frame_id: int
    joint_name_to_id: dict[str, int]
    joint_zero_offsets_rad: tuple[float, ...] = ()

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        *,
        joint_zero_offsets_rad: Sequence[float] = (),
    ) -> PinocchioCs68Model:
        pin = _require_pinocchio()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"CS68 URDF not found: {urdf_path}")
        offsets = tuple(float(value) for value in joint_zero_offsets_rad)
        if offsets and (len(offsets) != 6 or not np.isfinite(offsets).all()):
            raise ValueError("CS68 joint-zero offsets must be a finite six-vector")
        model = pin.buildModelFromUrdf(str(urdf_path))
        if int(model.nq) != 6:
            raise ValueError(f"Expected a six-DOF CS68 URDF, got nq={model.nq}")
        if not model.existFrame("tool0"):
            raise ValueError("CS68 URDF is missing tool0")
        joint_ids = {name: int(model.getJointId(name)) for name in CS68_JOINT_NAMES}
        missing = [name for name, joint_id in joint_ids.items() if joint_id == 0]
        if missing:
            raise ValueError("CS68 URDF is missing joints: " + ", ".join(missing))
        return cls(
            urdf_path=urdf_path,
            model=model,
            data=model.createData(),
            tool_frame_id=int(model.getFrameId("tool0")),
            joint_name_to_id=joint_ids,
            joint_zero_offsets_rad=offsets,
        )

    def _to_configuration(self, joint_positions_rad: Sequence[float]) -> NDArray[np.float64]:
        joints = np.asarray(joint_positions_rad, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("CS68 joint positions must be a finite six-vector")
        if self.joint_zero_offsets_rad:
            joints = joints + np.asarray(self.joint_zero_offsets_rad, dtype=np.float64)
        configuration = np.zeros(int(self.model.nq), dtype=np.float64)
        for index, name in enumerate(CS68_JOINT_NAMES):
            joint_id = self.joint_name_to_id[name]
            configuration[int(self.model.joints[joint_id].idx_q)] = joints[index]
        return configuration

    def forward_kinematics(self, joint_positions_rad: Sequence[float]) -> NDArray[np.float64]:
        pin = _require_pinocchio()
        configuration = self._to_configuration(joint_positions_rad)
        pin.forwardKinematics(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[self.tool_frame_id].homogeneous, dtype=np.float64).copy()


def _add_holorobot_collision_pairs(
    geometry_model: Any,
    *,
    max_parent_joint_hop: int = 1,
) -> None:
    """Apply HoloRobot's adjacent-parent-joint exclusion policy."""

    if max_parent_joint_hop < 0:
        raise ValueError("max_parent_joint_hop must be non-negative")
    pin = _require_pinocchio()
    for first in range(int(geometry_model.ngeoms)):
        for second in range(first + 1, int(geometry_model.ngeoms)):
            first_geometry = geometry_model.geometryObjects[first]
            second_geometry = geometry_model.geometryObjects[second]
            if (
                abs(int(first_geometry.parentJoint) - int(second_geometry.parentJoint))
                <= max_parent_joint_hop
            ):
                continue
            geometry_model.addCollisionPair(pin.CollisionPair(first, second))


def _add_environment_boxes(
    geometry_model: Any,
    obstacles: Sequence[CollisionObstacleConfig],
    *,
    minimum_clearance_m: float,
    robot_geometry_count: int,
) -> None:
    """Add configured workcell AABBs and robot-to-environment FCL pairs."""

    if not obstacles:
        return
    pin = _require_pinocchio()
    try:
        import hppfcl
    except ImportError as exc:
        raise ImportError("hpp-fcl is required for workcell collision boxes") from exc
    clearance = float(minimum_clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative")
    for obstacle in obstacles:
        lower = np.asarray(obstacle.minimum_m, dtype=np.float64) - clearance
        upper = np.asarray(obstacle.maximum_m, dtype=np.float64) + clearance
        size = upper - lower
        center = (lower + upper) / 2.0
        geometry = pin.GeometryObject(
            f"environment::{obstacle.name}",
            0,
            0,
            hppfcl.Box(size),
            pin.SE3(np.eye(3), center),
        )
        environment_index = int(geometry_model.addGeometryObject(geometry))
        ignored = set(obstacle.ignored_capsule_indices)
        for robot_index in range(robot_geometry_count):
            if robot_index in ignored:
                continue
            geometry_model.addCollisionPair(pin.CollisionPair(robot_index, environment_index))


@dataclass(slots=True)
class Cs68PinocchioCollisionChecker:
    """Fail-closed CS68+D435i mesh collision checker copied from HoloRobot semantics."""

    resources: Cs68ModelResources | Es68D435iCollisionResources
    kinematic_model: Cs68KinematicModel
    pinocchio_model: PinocchioCs68Model
    geometry_model: Any
    geometry_data: Any
    pair_links: tuple[tuple[str, str], ...]
    pair_geometries: tuple[tuple[str, str], ...]
    include_d435i_mount: bool
    environment_obstacle_names: tuple[str, ...] = ()
    model_name: str = "cs68"
    collision_model_id: str | None = None
    collision_model_hash: str | None = None
    robot_geometry_hash: str | None = None
    motion_model_contract_hash: str | None = None
    minimum_clearance_m: float = 0.0
    continuous_swept_volume_supported: bool = True
    collision_backend_versions: dict[str, str] = field(default_factory=dict)
    _geometry_motion_coefficients_cache: tuple[tuple[float, ...], ...] | None = field(
        default=None,
        repr=False,
    )
    _temporary_directory: TemporaryDirectory[str] | None = field(default=None, repr=False)

    @classmethod
    def from_resources(
        cls,
        resources: Cs68ModelResources | None = None,
        *,
        joint_zero_offsets_rad: Sequence[float] = (),
        include_d435i_mount: bool = True,
        environment_obstacles: Sequence[CollisionObstacleConfig] = (),
        minimum_clearance_m: float = 0.0,
    ) -> Cs68PinocchioCollisionChecker:
        pin = _require_pinocchio()
        clearance_m = float(minimum_clearance_m)
        if not math.isfinite(clearance_m) or clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be finite and non-negative")
        resolved = resources or Cs68ModelResources.packaged()
        resolved.validate()
        temporary_directory: TemporaryDirectory[str] | None = None
        if include_d435i_mount:
            temporary_directory = TemporaryDirectory(prefix="biblade-cs68-urdf-")
            urdf_path = write_cs68_urdf(
                Path(temporary_directory.name) / "cs68_d435i.urdf",
                resolved,
                include_d435i_mount=True,
            )
        else:
            urdf_path = resolved.urdf_path
        model = PinocchioCs68Model.from_urdf(
            urdf_path, joint_zero_offsets_rad=joint_zero_offsets_rad
        )
        geometry_model = pin.buildGeomFromUrdf(
            model.model,
            str(urdf_path),
            pin.COLLISION,
            package_dirs=[str(resolved.root)],
        )
        robot_geometry_count = int(geometry_model.ngeoms)
        _add_holorobot_collision_pairs(geometry_model)
        _add_environment_boxes(
            geometry_model,
            environment_obstacles,
            minimum_clearance_m=clearance_m,
            robot_geometry_count=robot_geometry_count,
        )
        geometry_data = pin.GeometryData(geometry_model)
        pair_links: list[tuple[str, str]] = []
        pair_geometries: list[tuple[str, str]] = []
        for pair in geometry_model.collisionPairs:
            first = geometry_model.geometryObjects[pair.first]
            second = geometry_model.geometryObjects[pair.second]
            pair_links.append(
                (
                    str(model.model.frames[int(first.parentFrame)].name),
                    str(model.model.frames[int(second.parentFrame)].name),
                )
            )
            pair_geometries.append((str(first.name), str(second.name)))
        return cls(
            resources=resolved,
            kinematic_model=Cs68KinematicModel.from_resources(
                resolved, joint_zero_offsets_rad=joint_zero_offsets_rad
            ),
            pinocchio_model=model,
            geometry_model=geometry_model,
            geometry_data=geometry_data,
            pair_links=tuple(pair_links),
            pair_geometries=tuple(pair_geometries),
            include_d435i_mount=bool(include_d435i_mount),
            environment_obstacle_names=tuple(obstacle.name for obstacle in environment_obstacles),
            model_name="cs68",
            minimum_clearance_m=clearance_m,
            collision_backend_versions=_collision_backend_versions(pin),
            _temporary_directory=temporary_directory,
        )

    @classmethod
    def from_es68_resources(
        cls,
        resources: Es68D435iCollisionResources | None = None,
        *,
        joint_zero_offsets_rad: Sequence[float] = (),
        environment_obstacles: Sequence[CollisionObstacleConfig] = (),
        minimum_clearance_m: float = 0.0,
    ) -> Cs68PinocchioCollisionChecker:
        """Load only the completed ES68+D435i manifest; never fall back to CS68."""

        pin = _require_pinocchio()
        resolved = resources or Es68D435iCollisionResources.packaged_template()
        template = resolved.load_active()
        requested_clearance_m = float(minimum_clearance_m)
        if not math.isfinite(requested_clearance_m) or requested_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be finite and non-negative")
        effective_clearance_m = max(
            requested_clearance_m,
            float(template.minimum_clearance_m),
        )
        backend_versions = _collision_backend_versions(pin)
        temporary = TemporaryDirectory(prefix="biblade-es68-urdf-")
        urdf_path = write_es68_d435i_collision_urdf(
            Path(temporary.name) / "es68_d435i.urdf",
            template,
        )
        model = PinocchioCs68Model.from_urdf(
            urdf_path,
            joint_zero_offsets_rad=joint_zero_offsets_rad,
        )
        geometry_model = pin.buildGeomFromUrdf(
            model.model,
            str(urdf_path),
            pin.COLLISION,
            package_dirs=[str(resolved.root)],
        )
        robot_geometry_count = int(geometry_model.ngeoms)
        _add_holorobot_collision_pairs(
            geometry_model,
            max_parent_joint_hop=template.max_parent_joint_hop,
        )
        _add_environment_boxes(
            geometry_model,
            environment_obstacles,
            minimum_clearance_m=effective_clearance_m,
            robot_geometry_count=robot_geometry_count,
        )
        geometry_data = pin.GeometryData(geometry_model)
        pair_links: list[tuple[str, str]] = []
        pair_geometries: list[tuple[str, str]] = []
        for pair in geometry_model.collisionPairs:
            first = geometry_model.geometryObjects[pair.first]
            second = geometry_model.geometryObjects[pair.second]
            pair_links.append(
                (
                    str(model.model.frames[int(first.parentFrame)].name),
                    str(model.model.frames[int(second.parentFrame)].name),
                )
            )
            pair_geometries.append((str(first.name), str(second.name)))
        return cls(
            resources=resolved,
            kinematic_model=Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=joint_zero_offsets_rad
            ),
            pinocchio_model=model,
            geometry_model=geometry_model,
            geometry_data=geometry_data,
            pair_links=tuple(pair_links),
            pair_geometries=tuple(pair_geometries),
            include_d435i_mount=True,
            environment_obstacle_names=tuple(obstacle.name for obstacle in environment_obstacles),
            model_name="es68",
            collision_model_id=template.model_id,
            collision_model_hash=es68_d435i_collision_content_hash(template),
            robot_geometry_hash=es68_d435i_robot_geometry_hash(
                template,
                joint_zero_offsets_rad=joint_zero_offsets_rad,
            ),
            motion_model_contract_hash=es68_d435i_motion_model_contract_hash(
                template,
                joint_zero_offsets_rad=joint_zero_offsets_rad,
                environment_obstacles=environment_obstacles,
                minimum_clearance_m=effective_clearance_m,
                resolved_collision_pairs=pair_geometries,
                collision_backend_versions=backend_versions,
            ),
            minimum_clearance_m=effective_clearance_m,
            collision_backend_versions=backend_versions,
            _temporary_directory=temporary,
        )

    def check(self, joint_positions_rad: Sequence[float]) -> CollisionCheckResult:
        try:
            joints = self.pinocchio_model._to_configuration(joint_positions_rad)
            controller_joints = np.asarray(joint_positions_rad, dtype=np.float64)
            violations = tuple(
                name
                for name, value, limits in zip(
                    CS68_JOINT_NAMES,
                    controller_joints,
                    self.kinematic_model.joint_limit_pairs(),
                    strict=True,
                )
                if not limits[0] <= float(value) <= limits[1]
            )
            if violations:
                return CollisionCheckResult(
                    status=CollisionCheckStatus.BLOCKED,
                    blocking_reasons=tuple(
                        f"joint_limit:{joint_name}" for joint_name in violations
                    ),
                    diagnostics=self._diagnostics(),
                )
            pin = _require_pinocchio()
            pin.forwardKinematics(self.pinocchio_model.model, self.pinocchio_model.data, joints)
            pin.updateGeometryPlacements(
                self.pinocchio_model.model,
                self.pinocchio_model.data,
                self.geometry_model,
                self.geometry_data,
            )
            pin.computeCollisions(self.geometry_model, self.geometry_data, False)
            collision_indices = tuple(
                index
                for index, result in enumerate(self.geometry_data.collisionResults)
                if bool(result.isCollision())
            )
            clearance_distances: dict[int, float] = {}
            if self.minimum_clearance_m > 0.0:
                pin.computeDistances(self.geometry_model, self.geometry_data)
                for index, (geometries, result) in enumerate(
                    zip(
                        self.pair_geometries,
                        self.geometry_data.distanceResults,
                        strict=True,
                    )
                ):
                    if index in collision_indices or any(
                        name.startswith("environment::") for name in geometries
                    ):
                        continue
                    distance = float(result.min_distance)
                    if not math.isfinite(distance):
                        raise ValueError(
                            f"non-finite self-clearance distance for pair {geometries}"
                        )
                    if distance < self.minimum_clearance_m:
                        clearance_distances[index] = distance
            pairs = tuple(self._finding(index) for index in collision_indices) + tuple(
                self._finding(index, clearance_distance_m=distance)
                for index, distance in clearance_distances.items()
            )
            return CollisionCheckResult(
                status=(CollisionCheckStatus.BLOCKED if pairs else CollisionCheckStatus.CLEAR),
                blocking_reasons=tuple(pair.pair_id for pair in pairs),
                pairs=pairs,
                diagnostics=self._diagnostics(),
            )

        except (TypeError, ValueError, RuntimeError) as exc:
            return CollisionCheckResult(
                status=CollisionCheckStatus.UNKNOWN,
                blocking_reasons=(f"collision_checker_error:{exc}",),
                diagnostics=self._diagnostics(),
            )

    @property
    def model_binding(
        self,
    ) -> tuple[str, str | None, str | None, str | None]:
        """Stable identity that preflight and execution must agree on."""

        return (
            f"elite_{self.model_name}",
            self.collision_model_id,
            self.robot_geometry_hash,
            self.motion_model_contract_hash,
        )

    def check_path(
        self,
        start_joint_positions_rad: Sequence[float],
        end_joint_positions_rad: Sequence[float],
        *,
        maximum_joint_step_rad: float,
        proof_tolerance_m: float = 1e-6,
        maximum_subdivision_depth: int = 14,
        minimum_interval_joint_span_rad: float = 1e-7,
        maximum_joint_path_deviation_rad: Sequence[float] = (0.0,) * 6,
        motion_envelope_acceptance_id: str | None = None,
        motion_envelope_metadata_sha256: str | None = None,
    ) -> JointPathMeshCollisionReport:
        """Prove a complete linear joint sweep using conservative interval bounds.

        The Python hpp-fcl binding used by this project has no continuous-collision
        API.  For each interval this routine evaluates exact FCL separation at the
        midpoint and subtracts a serial-chain Lipschitz displacement bound for both
        geometries.  An interval is certified only when the remaining margin is
        positive.  Otherwise it is bisected.  Exhausting a configured limit returns
        ``UNKNOWN``; clear samples alone can never yield a clear report.
        """

        start = np.asarray(start_joint_positions_rad, dtype=np.float64)
        end = np.asarray(end_joint_positions_rad, dtype=np.float64)
        if start.shape != (6,) or end.shape != (6,) or not np.isfinite((start, end)).all():
            raise ValueError("CS68 path endpoints must be finite six-vectors")
        step = float(maximum_joint_step_rad)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_joint_step_rad must be finite and positive")
        tolerance = float(proof_tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("proof_tolerance_m must be finite and non-negative")
        depth_limit = int(maximum_subdivision_depth)
        if depth_limit < 0:
            raise ValueError("maximum_subdivision_depth must be non-negative")
        minimum_span = float(minimum_interval_joint_span_rad)
        if not math.isfinite(minimum_span) or minimum_span <= 0.0:
            raise ValueError("minimum_interval_joint_span_rad must be finite and positive")
        accepted_uncertainty = np.asarray(
            maximum_joint_path_deviation_rad,
            dtype=np.float64,
        )
        if (
            accepted_uncertainty.shape != (6,)
            or not np.isfinite(accepted_uncertainty).all()
            or np.any(accepted_uncertainty < 0.0)
        ):
            raise ValueError(
                "maximum_joint_path_deviation_rad must be a finite non-negative six-vector"
            )
        acceptance_id = motion_envelope_acceptance_id
        envelope_metadata_sha256 = motion_envelope_metadata_sha256
        if np.any(accepted_uncertainty > 0.0):
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (acceptance_id, envelope_metadata_sha256)
            ):
                raise ValueError(
                    "A non-zero joint-path deviation requires acceptance/metadata SHA-256s"
                )
        elif acceptance_id is not None or envelope_metadata_sha256 is not None:
            raise ValueError("Motion-envelope hashes cannot bind a zero uncertainty vector")
        accepted_uncertainty_tuple = tuple(float(value) for value in accepted_uncertainty)
        segment_count = max(1, math.ceil(float(np.max(np.abs(end - start))) / step))
        diagnostics = self._diagnostics()
        path_hash = _joint_path_sha256(start, end)
        model_hash = _mesh_model_binding_sha256(diagnostics)
        motion_bound_hash = self.geometry_motion_bound_contract_sha256
        evaluated = 0
        certified = 0
        deepest = 0
        minimum_margin: float | None = None

        def issue_evidence(reason: str) -> SweptMeshProofEvidence:
            provisional = SweptMeshProofEvidence(
                trajectory_sha256=path_hash,
                model_binding_sha256=model_hash,
                motion_bound_contract_sha256=motion_bound_hash,
                motion_envelope_acceptance_id=acceptance_id,
                motion_envelope_metadata_sha256=envelope_metadata_sha256,
                accepted_joint_uncertainty_rad=accepted_uncertainty_tuple,
                maximum_joint_step_rad=step,
                proof_tolerance_m=tolerance,
                maximum_subdivision_depth=depth_limit,
                minimum_interval_joint_span_rad=minimum_span,
                initial_interval_count=segment_count,
                certified_interval_count=certified,
                evaluated_configuration_count=evaluated,
                deepest_subdivision=deepest,
                minimum_certificate_margin_m=minimum_margin,
                termination_reason=reason,
                evidence_sha256="",
            )
            return SweptMeshProofEvidence(
                **{
                    name: getattr(provisional, name)
                    for name in provisional.__dataclass_fields__
                    if name != "evidence_sha256"
                },
                evidence_sha256=_canonical_sha256(provisional._payload()),
            )

        def finish_nonclear(
            result: CollisionCheckResult,
            *,
            fraction: float,
            reason: str,
        ) -> JointPathMeshCollisionReport:
            evidence = issue_evidence(reason)
            enriched = CollisionCheckResult(
                status=result.status,
                blocking_reasons=result.blocking_reasons,
                pairs=result.pairs,
                diagnostics={
                    **result.diagnostics,
                    "continuous_swept_volume_verified": False,
                    "swept_mesh_proof_evidence_sha256": evidence.evidence_sha256,
                    "swept_mesh_termination_reason": reason,
                },
            )
            return JointPathMeshCollisionReport(
                status=result.status,
                sample_count=evaluated,
                blocked_sample_index=max(0, evaluated - 1),
                blocked_path_fraction=float(fraction),
                result=enriched,
                maximum_joint_step_rad=step,
                continuous_swept_volume_verified=False,
                proof_evidence=evidence,
            )

        checked_fractions: dict[float, CollisionCheckResult] = {}

        def evaluate(fraction: float) -> CollisionCheckResult:
            nonlocal evaluated
            key = float(fraction)
            cached = checked_fractions.get(key)
            if cached is not None:
                return cached
            result = self.check(start + key * (end - start))
            checked_fractions[key] = result
            evaluated += 1
            return result

        for endpoint_fraction in (0.0, 1.0):
            endpoint_result = evaluate(endpoint_fraction)
            if endpoint_result.status is not CollisionCheckStatus.CLEAR:
                return finish_nonclear(
                    endpoint_result,
                    fraction=endpoint_fraction,
                    reason=(
                        "collision_witness"
                        if endpoint_result.status is CollisionCheckStatus.BLOCKED
                        else "checker_error"
                    ),
                )

        intervals: list[tuple[float, float, int]] = [
            (index / segment_count, (index + 1) / segment_count, 0)
            for index in range(segment_count)
        ]
        while intervals:
            lower_fraction, upper_fraction, depth = intervals.pop()
            deepest = max(deepest, depth)
            midpoint_fraction = (lower_fraction + upper_fraction) / 2.0
            midpoint_result = evaluate(midpoint_fraction)
            if midpoint_result.status is not CollisionCheckStatus.CLEAR:
                return finish_nonclear(
                    midpoint_result,
                    fraction=midpoint_fraction,
                    reason=(
                        "collision_witness"
                        if midpoint_result.status is CollisionCheckStatus.BLOCKED
                        else "checker_error"
                    ),
                )
            lower = start + lower_fraction * (end - start)
            upper = start + upper_fraction * (end - start)
            midpoint = start + midpoint_fraction * (end - start)
            try:
                slacks = self._pair_clearance_slacks(midpoint)
                maximum_deviation = np.abs(upper - lower) / 2.0 + accepted_uncertainty
                pair_margins: list[float] = []
                for pair_index, slack in enumerate(slacks):
                    pair = self.geometry_model.collisionPairs[pair_index]
                    displacement = self.geometry_displacement_bound_m(
                        int(pair.first), maximum_deviation
                    ) + self.geometry_displacement_bound_m(int(pair.second), maximum_deviation)
                    pair_margins.append(slack - displacement - tolerance)
            except (TypeError, ValueError, RuntimeError) as exc:
                unknown = CollisionCheckResult(
                    status=CollisionCheckStatus.UNKNOWN,
                    blocking_reasons=(f"continuous_swept_mesh_proof_error:{exc}",),
                    diagnostics=diagnostics,
                )
                return finish_nonclear(
                    unknown,
                    fraction=midpoint_fraction,
                    reason="proof_error",
                )
            interval_margin = min(pair_margins, default=math.inf)
            if interval_margin > 0.0 or not pair_margins:
                if math.isfinite(interval_margin):
                    minimum_margin = (
                        interval_margin
                        if minimum_margin is None
                        else min(minimum_margin, interval_margin)
                    )
                certified += 1
                continue
            joint_span = float(np.max(np.abs(upper - lower)))
            if depth >= depth_limit or joint_span <= minimum_span:
                unknown = CollisionCheckResult(
                    status=CollisionCheckStatus.UNKNOWN,
                    blocking_reasons=("continuous_swept_mesh_unproven:subdivision_limit",),
                    diagnostics=diagnostics,
                )
                return finish_nonclear(
                    unknown,
                    fraction=midpoint_fraction,
                    reason="subdivision_limit",
                )
            intervals.append((midpoint_fraction, upper_fraction, depth + 1))
            intervals.append((lower_fraction, midpoint_fraction, depth + 1))

        evidence = issue_evidence("all_intervals_certified")
        clear = CollisionCheckResult(
            status=CollisionCheckStatus.CLEAR,
                diagnostics={
                    **diagnostics,
                    "continuous_swept_volume_verified": True,
                    "continuous_sweep_backend": evidence.method,
                    "motion_envelope_acceptance_id": acceptance_id,
                    "motion_envelope_metadata_sha256": envelope_metadata_sha256,
                    "accepted_joint_uncertainty_rad": list(accepted_uncertainty_tuple),
                "swept_mesh_proof_evidence_sha256": evidence.evidence_sha256,
                "swept_mesh_termination_reason": evidence.termination_reason,
                "certified_interval_count": certified,
                "deepest_subdivision": deepest,
            },
        )
        return JointPathMeshCollisionReport(
            status=CollisionCheckStatus.CLEAR,
            sample_count=evaluated,
            blocked_sample_index=None,
            blocked_path_fraction=None,
            result=clear,
            maximum_joint_step_rad=step,
            continuous_swept_volume_verified=True,
            proof_evidence=evidence,
        )

    def _pair_clearance_slacks(
        self,
        joint_positions_rad: Sequence[float],
    ) -> tuple[float, ...]:
        """Return exact FCL separation remaining beyond the configured clearance."""

        pin = _require_pinocchio()
        joints = self.pinocchio_model._to_configuration(joint_positions_rad)
        pin.forwardKinematics(
            self.pinocchio_model.model,
            self.pinocchio_model.data,
            joints,
        )
        pin.updateGeometryPlacements(
            self.pinocchio_model.model,
            self.pinocchio_model.data,
            self.geometry_model,
            self.geometry_data,
        )
        pin.computeDistances(self.geometry_model, self.geometry_data)
        slacks: list[float] = []
        for geometries, result in zip(
            self.pair_geometries,
            self.geometry_data.distanceResults,
            strict=True,
        ):
            distance = float(result.min_distance)
            if not math.isfinite(distance):
                raise ValueError(f"non-finite FCL distance for swept pair {geometries}")
            environment_pair = any(name.startswith("environment::") for name in geometries)
            required = 0.0 if environment_pair else self.minimum_clearance_m
            slacks.append(distance - required)
        return tuple(slacks)

    def _geometry_motion_coefficients(self) -> tuple[tuple[float, ...], ...]:
        """Conservative metres/radian bounds for every geometry and ES68 joint.

        For an ancestor revolute joint, the maximum displacement of any point on a
        descendant mesh is bounded by ``radius * |delta_angle|``.  ``radius`` is
        formed from the sum of invariant inter-joint translation norms plus the
        mesh's local-AABB enclosing radius.  Triangle inequalities make this an
        over-approximation for every configuration, not just the checked pose.
        """

        cached = self._geometry_motion_coefficients_cache
        if cached is not None:
            return cached
        model = self.pinocchio_model.model
        rows: list[tuple[float, ...]] = []
        for geometry_object in self.geometry_model.geometryObjects:
            name = str(geometry_object.name)
            coefficients = np.zeros(6, dtype=np.float64)
            if name.startswith("environment::") or int(geometry_object.parentJoint) == 0:
                rows.append(tuple(float(value) for value in coefficients))
                continue
            geometry = geometry_object.geometry
            geometry.computeLocalAABB()
            local_extent = (
                float(np.linalg.norm(np.asarray(geometry_object.placement.translation)))
                + float(np.linalg.norm(np.asarray(geometry.aabb_center)))
                + float(geometry.aabb_radius)
            )
            downstream_translation = 0.0
            joint_id = int(geometry_object.parentJoint)
            while joint_id != 0:
                joint = model.joints[joint_id]
                if not str(joint.shortname()).startswith("JointModelR"):
                    raise ValueError(
                        "continuous swept proof supports only revolute ES68 joints; "
                        f"got {joint.shortname()}"
                    )
                configuration_index = int(joint.idx_q)
                if not 0 <= configuration_index < 6:
                    raise ValueError("ES68 swept proof found an invalid joint index")
                coefficients[configuration_index] = local_extent + downstream_translation
                downstream_translation += float(
                    np.linalg.norm(np.asarray(model.jointPlacements[joint_id].translation))
                )
                joint_id = int(model.parents[joint_id])
            if not np.isfinite(coefficients).all() or np.any(coefficients < 0.0):
                raise ValueError("invalid ES68 geometry motion coefficients")
            rows.append(tuple(float(value) for value in coefficients))
        result = tuple(rows)
        self._geometry_motion_coefficients_cache = result
        return result

    @property
    def geometry_motion_bound_contract_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": "biblade_fusion.es68_geometry_motion_bound.v1",
                "derivation": "serial_revolute_chain_triangle_inequality",
                "geometry_names": [str(item.name) for item in self.geometry_model.geometryObjects],
                "coefficients_m_per_rad": [
                    list(row) for row in self._geometry_motion_coefficients()
                ],
            }
        )

    def geometry_displacement_bound_m(
        self,
        geometry_index: int,
        maximum_joint_deviation_rad: Sequence[float],
    ) -> float:
        deviations = np.asarray(maximum_joint_deviation_rad, dtype=np.float64)
        if (
            deviations.shape != (6,)
            or not np.isfinite(deviations).all()
            or np.any(deviations < 0.0)
        ):
            raise ValueError("maximum_joint_deviation_rad must be a finite non-negative six-vector")
        coefficients = self._geometry_motion_coefficients()
        index = int(geometry_index)
        if not 0 <= index < len(coefficients):
            raise ValueError("geometry index is outside the collision model")
        bound = float(np.dot(np.asarray(coefficients[index]), deviations))
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("computed geometry displacement bound is invalid")
        return bound

    def _finding(
        self,
        pair_index: int,
        *,
        clearance_distance_m: float | None = None,
    ) -> CollisionPairFinding:
        links = self.pair_links[pair_index]
        geometries = self.pair_geometries[pair_index]
        environment_geometry = next(
            (name for name in geometries if name.startswith("environment::")),
            None,
        )
        if environment_geometry is None:
            prefix = "self_clearance" if clearance_distance_m is not None else "self_collision"
            pair_id = f"{prefix}:{geometries[0]}:{geometries[1]}"
            link_a, link_b = links
        else:
            robot_geometry = next(name for name in geometries if name != environment_geometry)
            obstacle_name = environment_geometry.removeprefix("environment::")
            pair_id = f"workcell_collision:{robot_geometry}:{obstacle_name}"
            link_a, link_b = (
                next(
                    link
                    for link, name in zip(links, geometries, strict=True)
                    if name == robot_geometry
                ),
                obstacle_name,
            )
        return CollisionPairFinding(
            pair_id=pair_id,
            link_a=link_a,
            link_b=link_b,
            geometry_a=geometries[0],
            geometry_b=geometries[1],
            minimum_distance_m=clearance_distance_m,
            required_clearance_m=(
                self.minimum_clearance_m if clearance_distance_m is not None else None
            ),
        )

    def _diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "pinocchio_fcl",
            "model": f"elite_{self.model_name}",
            "collision_model_id": self.collision_model_id,
            "collision_model_hash": self.collision_model_hash,
            "robot_geometry_hash": self.robot_geometry_hash,
            "motion_model_contract_hash": self.motion_model_contract_hash,
            "geometry_motion_bound_contract_sha256": (self.geometry_motion_bound_contract_sha256),
            "minimum_clearance_m": self.minimum_clearance_m,
            "self_clearance_enforced": True,
            "collision_backend_versions": dict(self.collision_backend_versions),
            "include_d435i_mount": self.include_d435i_mount,
            "geometry_count": int(self.geometry_model.ngeoms),
            "collision_pair_count": len(self.pair_links),
            "environment_obstacles": list(self.environment_obstacle_names),
            "motion_authorized": False,
            "provenance": robot_stack_provenance(),
        }


class Es68PinocchioCollisionChecker(Cs68PinocchioCollisionChecker):
    """Production-facing name for the strict ES68+D435i loader.

    The legacy base class remains available only for regression against the copied
    HoloRobot CS68 fixtures.  Production code must call ``from_es68_resources``.
    """
