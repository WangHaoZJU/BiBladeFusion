"""HoloRobot CS68 kinematics and packaged robot-model resource resolution.

Adapted from HoloRobot's ``elite_robot_model.py``, ``elite_yaml_loader.py``, and
``robot_model_paths.py`` at the commit recorded in :mod:`robotics.provenance`.
The fixed-transform-then-RotZ chain and joint-zero offset convention are kept intact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

CS68_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

CS68_COLLISION_LINK_NAMES: tuple[str, ...] = (
    "base_link_inertia",
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)

_SEGMENT_KEYS: tuple[str, ...] = (
    "shoulder",
    "upperarm",
    "forearm",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
_SEGMENT_FIELDS = frozenset(("x", "y", "z", "roll", "pitch", "yaw"))


def _degrees_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> float:
    return math.radians(float(loader.construct_scalar(node)))


class _EliteSafeLoader(yaml.SafeLoader):
    """Project-local loader so registering ``!degrees`` has no global side effects."""


_EliteSafeLoader.add_constructor("!degrees", _degrees_constructor)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_EliteSafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in robot-model file: {path}")
    return payload


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric robot-model value, got {type(value)!r}")


@dataclass(frozen=True, slots=True)
class Cs68ModelResources:
    """Paths to the copied HoloRobot CS68 model bundle."""

    root: Path

    @classmethod
    def packaged(cls) -> Cs68ModelResources:
        root = Path(__file__).resolve().parent / "resources" / "elite_cs"
        resources = cls(root=root)
        resources.validate()
        return resources

    @property
    def kinematics_yaml(self) -> Path:
        return self.root / "config" / "cs68" / "default_kinematics.yaml"

    @property
    def joint_limits_yaml(self) -> Path:
        return self.root / "config" / "cs68" / "joint_limits.yaml"

    @property
    def physical_parameters_yaml(self) -> Path:
        return self.root / "config" / "cs68" / "physical_parameters.yaml"

    @property
    def urdf_path(self) -> Path:
        return self.root / "urdf" / "generated" / "cs68.urdf"

    @property
    def collision_mesh_dir(self) -> Path:
        return self.root / "meshes" / "cs68" / "collision"

    @property
    def primitive_collision_manifest(self) -> Path:
        return self.root / "collision_models" / "cs68" / "manifest.yaml"

    @property
    def license_path(self) -> Path:
        return self.root / "LICENSE"

    def validate(self) -> None:
        required = (
            self.kinematics_yaml,
            self.joint_limits_yaml,
            self.physical_parameters_yaml,
            self.urdf_path,
            self.primitive_collision_manifest,
            self.license_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        mesh_names = (
            "base",
            "shoulder",
            "upperarm",
            "forearm",
            "wrist1",
            "wrist2",
            "wrist3",
        )
        missing.extend(
            str(self.collision_mesh_dir / f"{name}.stl")
            for name in mesh_names
            if not (self.collision_mesh_dir / f"{name}.stl").is_file()
        )
        if missing:
            raise FileNotFoundError(
                "Incomplete HoloRobot CS68 resource bundle: " + ", ".join(missing)
            )


def _parse_segments(path: Path) -> tuple[dict[str, float], ...]:
    raw = _load_yaml(path).get("kinematics", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"Missing kinematics mapping in {path}")
    segments: list[dict[str, float]] = []
    for key in _SEGMENT_KEYS:
        item = raw.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"Missing kinematics segment {key!r} in {path}")
        segments.append(
            {
                field: _as_float(item.get(field, 0.0))
                for field in ("x", "y", "z", "roll", "pitch", "yaw")
            }
        )
    return tuple(segments)


def _apply_overrides(
    segments: Sequence[dict[str, float]],
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, float], ...]:
    adjusted = [dict(segment) for segment in segments]
    for raw_key, raw_values in (overrides or {}).items():
        key = str(raw_key)
        if key not in _SEGMENT_KEYS:
            raise ValueError(f"Unknown kinematics segment override: {key!r}")
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"Kinematics override for {key!r} must be a mapping")
        target = adjusted[_SEGMENT_KEYS.index(key)]
        for raw_field, raw_value in raw_values.items():
            field = str(raw_field)
            if field not in _SEGMENT_FIELDS:
                raise ValueError(f"Unknown kinematics override field: {key}.{field}")
            target[field] = _as_float(raw_value)
    return tuple(adjusted)


def _parse_joint_limits(path: Path) -> dict[str, tuple[float, float]]:
    raw = _load_yaml(path).get("joint_limits", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"Missing joint_limits mapping in {path}")
    limits: dict[str, tuple[float, float]] = {}
    for joint_name in CS68_JOINT_NAMES:
        item = raw.get(joint_name)
        if not isinstance(item, Mapping) or not item.get("has_position_limits", True):
            raise ValueError(f"Missing position limits for {joint_name!r} in {path}")
        limits[joint_name] = (
            _as_float(item["min_position"]),
            _as_float(item["max_position"]),
        )
    return limits


def _parse_joint_velocity_limits(path: Path) -> dict[str, float]:
    raw = _load_yaml(path).get("joint_limits", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"Missing joint_limits mapping in {path}")
    limits: dict[str, float] = {}
    for joint_name in CS68_JOINT_NAMES:
        item = raw.get(joint_name)
        if not isinstance(item, Mapping) or not item.get("has_velocity_limits", False):
            raise ValueError(f"Missing velocity limit for {joint_name!r} in {path}")
        value = _as_float(item["max_velocity"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid velocity limit for {joint_name!r} in {path}")
        limits[joint_name] = value
    return limits


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _fixed_transform(segment: Mapping[str, float]) -> NDArray[np.float64]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(
        segment["roll"], segment["pitch"], segment["yaw"]
    )
    transform[:3, 3] = (segment["x"], segment["y"], segment["z"])
    return transform


def _rot_z(theta: float) -> NDArray[np.float64]:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        (
            (c, -s, 0.0, 0.0),
            (s, c, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class Cs68KinematicModel:
    """HoloRobot-compatible fixed-transform CS68 model."""

    segments: tuple[dict[str, float], ...]
    joint_limits: dict[str, tuple[float, float]]
    joint_zero_offsets_rad: tuple[float, ...] = ()

    @classmethod
    def from_resources(
        cls,
        resources: Cs68ModelResources | None = None,
        *,
        kinematics_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        joint_zero_offsets_rad: Sequence[float] = (),
    ) -> Cs68KinematicModel:
        resolved = resources or Cs68ModelResources.packaged()
        resolved.validate()
        offsets = tuple(float(value) for value in joint_zero_offsets_rad)
        if offsets and len(offsets) != len(CS68_JOINT_NAMES):
            raise ValueError("CS68 joint_zero_offsets_rad must contain six values")
        if not np.isfinite(offsets).all():
            raise ValueError("CS68 joint_zero_offsets_rad must be finite")
        return cls(
            segments=_apply_overrides(
                _parse_segments(resolved.kinematics_yaml), kinematics_overrides
            ),
            joint_limits=_parse_joint_limits(resolved.joint_limits_yaml),
            joint_zero_offsets_rad=offsets,
        )

    @property
    def dof(self) -> int:
        return len(CS68_JOINT_NAMES)

    def joint_limit_pairs(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.joint_limits[name] for name in CS68_JOINT_NAMES)

    def joint_velocity_limits_rad_s(self) -> tuple[float, ...]:
        """Return copied HoloRobot controller-profile velocity limits in joint order."""

        resources = Cs68ModelResources.packaged()
        limits = _parse_joint_velocity_limits(resources.joint_limits_yaml)
        return tuple(limits[name] for name in CS68_JOINT_NAMES)

    def _model_joints(self, joint_positions_rad: Sequence[float]) -> NDArray[np.float64]:
        joints = np.asarray(joint_positions_rad, dtype=np.float64)
        if joints.shape != (self.dof,) or not np.isfinite(joints).all():
            raise ValueError("CS68 joint positions must be a finite six-vector")
        if self.joint_zero_offsets_rad:
            joints = joints + np.asarray(self.joint_zero_offsets_rad, dtype=np.float64)
        return joints

    def forward_kinematics(
        self, joint_positions_rad: Sequence[float]
    ) -> NDArray[np.float64]:
        transform = np.eye(4, dtype=np.float64)
        for segment, joint in zip(
            self.segments, self._model_joints(joint_positions_rad), strict=True
        ):
            transform = transform @ _fixed_transform(segment) @ _rot_z(float(joint))
        return transform

    def link_transforms(
        self, joint_positions_rad: Sequence[float]
    ) -> dict[str, NDArray[np.float64]]:
        transform = np.eye(4, dtype=np.float64)
        transforms = {CS68_COLLISION_LINK_NAMES[0]: transform.copy()}
        for index, (segment, joint) in enumerate(
            zip(self.segments, self._model_joints(joint_positions_rad), strict=True), start=1
        ):
            transform = transform @ _fixed_transform(segment) @ _rot_z(float(joint))
            transforms[CS68_COLLISION_LINK_NAMES[index]] = transform.copy()
        return transforms
