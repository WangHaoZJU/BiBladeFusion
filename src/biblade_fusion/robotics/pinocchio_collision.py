"""HoloRobot-compatible Pinocchio/FCL kinematics and CS68 self-collision checks.

Adapted from HoloRobot's ``pinocchio_robot_model.py`` and
``pinocchio_collision.py`` at the pinned provenance commit.
"""

from __future__ import annotations

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
from biblade_fusion.robotics.cs68_model import (
    CS68_JOINT_NAMES,
    Cs68KinematicModel,
    Cs68ModelResources,
)
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


@dataclass(frozen=True, slots=True)
class JointPathMeshCollisionReport:
    status: CollisionCheckStatus
    sample_count: int
    blocked_sample_index: int | None
    blocked_path_fraction: float | None
    result: CollisionCheckResult
    maximum_joint_step_rad: float

    @property
    def collision_free(self) -> bool:
        return self.status is CollisionCheckStatus.CLEAR

    @property
    def motion_authorized(self) -> bool:
        return False


def _require_pinocchio() -> Any:
    try:
        import pinocchio
    except ImportError as exc:
        raise ImportError(
            "Pinocchio/FCL is required for the HoloRobot collision backend"
        ) from exc
    return pinocchio


def pinocchio_collision_available() -> bool:
    try:
        _require_pinocchio()
    except ImportError:
        return False
    return True


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
        return np.asarray(
            self.data.oMf[self.tool_frame_id].homogeneous, dtype=np.float64
        ).copy()


def _add_holorobot_collision_pairs(geometry_model: Any) -> None:
    """Apply HoloRobot's adjacent-parent-joint exclusion policy."""

    pin = _require_pinocchio()
    for first in range(int(geometry_model.ngeoms)):
        for second in range(first + 1, int(geometry_model.ngeoms)):
            first_geometry = geometry_model.geometryObjects[first]
            second_geometry = geometry_model.geometryObjects[second]
            if abs(int(first_geometry.parentJoint) - int(second_geometry.parentJoint)) <= 1:
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
            geometry_model.addCollisionPair(
                pin.CollisionPair(robot_index, environment_index)
            )


@dataclass(slots=True)
class Cs68PinocchioCollisionChecker:
    """Fail-closed CS68+D435i mesh collision checker copied from HoloRobot semantics."""

    resources: Cs68ModelResources
    kinematic_model: Cs68KinematicModel
    pinocchio_model: PinocchioCs68Model
    geometry_model: Any
    geometry_data: Any
    pair_links: tuple[tuple[str, str], ...]
    pair_geometries: tuple[tuple[str, str], ...]
    include_d435i_mount: bool
    environment_obstacle_names: tuple[str, ...] = ()
    _temporary_directory: TemporaryDirectory[str] | None = field(
        default=None, repr=False
    )

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
            minimum_clearance_m=minimum_clearance_m,
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
            environment_obstacle_names=tuple(
                obstacle.name for obstacle in environment_obstacles
            ),
            _temporary_directory=temporary_directory,
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
            pin.forwardKinematics(
                self.pinocchio_model.model, self.pinocchio_model.data, joints
            )
            pin.updateGeometryPlacements(
                self.pinocchio_model.model,
                self.pinocchio_model.data,
                self.geometry_model,
                self.geometry_data,
            )
            pin.computeCollisions(self.geometry_model, self.geometry_data, False)
            blocked_indices = tuple(
                index
                for index, result in enumerate(self.geometry_data.collisionResults)
                if bool(result.isCollision())
            )
            pairs = tuple(self._finding(index) for index in blocked_indices)
            return CollisionCheckResult(
                status=(
                    CollisionCheckStatus.BLOCKED
                    if blocked_indices
                    else CollisionCheckStatus.CLEAR
                ),
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

    def check_path(
        self,
        start_joint_positions_rad: Sequence[float],
        end_joint_positions_rad: Sequence[float],
        *,
        maximum_joint_step_rad: float,
    ) -> JointPathMeshCollisionReport:
        start = np.asarray(start_joint_positions_rad, dtype=np.float64)
        end = np.asarray(end_joint_positions_rad, dtype=np.float64)
        if (
            start.shape != (6,)
            or end.shape != (6,)
            or not np.isfinite((start, end)).all()
        ):
            raise ValueError("CS68 path endpoints must be finite six-vectors")
        step = float(maximum_joint_step_rad)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("maximum_joint_step_rad must be finite and positive")
        segment_count = max(1, math.ceil(float(np.max(np.abs(end - start))) / step))
        for sample_index, fraction in enumerate(np.linspace(0.0, 1.0, segment_count + 1)):
            result = self.check(start + fraction * (end - start))
            if result.status is not CollisionCheckStatus.CLEAR:
                return JointPathMeshCollisionReport(
                    status=result.status,
                    sample_count=segment_count + 1,
                    blocked_sample_index=sample_index,
                    blocked_path_fraction=float(fraction),
                    result=result,
                    maximum_joint_step_rad=step,
                )
        clear = CollisionCheckResult(
            status=CollisionCheckStatus.CLEAR,
            diagnostics=self._diagnostics(),
        )
        return JointPathMeshCollisionReport(
            status=CollisionCheckStatus.CLEAR,
            sample_count=segment_count + 1,
            blocked_sample_index=None,
            blocked_path_fraction=None,
            result=clear,
            maximum_joint_step_rad=step,
        )

    def _finding(self, pair_index: int) -> CollisionPairFinding:
        links = self.pair_links[pair_index]
        geometries = self.pair_geometries[pair_index]
        environment_geometry = next(
            (name for name in geometries if name.startswith("environment::")),
            None,
        )
        if environment_geometry is None:
            pair_id = f"self_collision:{geometries[0]}:{geometries[1]}"
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
        )

    def _diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "pinocchio_fcl",
            "model": "elite_cs68",
            "include_d435i_mount": self.include_d435i_mount,
            "geometry_count": int(self.geometry_model.ngeoms),
            "collision_pair_count": len(self.pair_links),
            "environment_obstacles": list(self.environment_obstacle_names),
            "motion_authorized": False,
            "provenance": robot_stack_provenance(),
        }
