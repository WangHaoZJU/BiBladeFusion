"""Calibrated HoloRobot ES68 kinematics for the lab arm used by BiBladeFusion.

The ES68 and CS68 share the same fixed-transform-then-RotZ implementation, but their
calibrated link parameters are different.  This module deliberately gives the ES68
bundle its own resource and type names so an experiment cannot silently load CS68
geometry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.robotics.cs68_model import (
    CS68_JOINT_NAMES,
    Cs68KinematicModel,
    _apply_overrides,
    _parse_joint_limits,
    _parse_joint_velocity_limits,
    _parse_segments,
)

ES68_JOINT_NAMES = CS68_JOINT_NAMES


@dataclass(frozen=True, slots=True)
class Es68ModelResources:
    """Paths to the exact HoloRobot calibration of the laboratory ES68."""

    root: Path

    @classmethod
    def packaged(cls) -> Es68ModelResources:
        resources = cls(Path(__file__).resolve().parent / "resources" / "elite_cs")
        resources.validate()
        return resources

    @property
    def kinematics_yaml(self) -> Path:
        return self.root / "config" / "es68" / "default_kinematics.yaml"

    @property
    def joint_limits_yaml(self) -> Path:
        return self.root / "config" / "es68" / "joint_limits.yaml"

    @property
    def tcp_offset_json(self) -> Path:
        return self.root / "config" / "es68" / "lab_tcp_offset.json"

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (
                self.kinematics_yaml,
                self.joint_limits_yaml,
                self.tcp_offset_json,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Incomplete HoloRobot ES68 calibration bundle: " + ", ".join(missing)
            )


@dataclass(frozen=True, slots=True)
class Es68KinematicModel(Cs68KinematicModel):
    """HoloRobot-compatible FK for the 709-pose calibrated laboratory ES68."""

    @classmethod
    def from_resources(
        cls,
        resources: Es68ModelResources | None = None,
        *,
        kinematics_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        joint_zero_offsets_rad: Sequence[float] = (),
    ) -> Es68KinematicModel:
        resolved = resources or Es68ModelResources.packaged()
        resolved.validate()
        offsets = tuple(float(value) for value in joint_zero_offsets_rad)
        if offsets and len(offsets) != len(ES68_JOINT_NAMES):
            raise ValueError("ES68 joint_zero_offsets_rad must contain six values")
        if not np.isfinite(offsets).all():
            raise ValueError("ES68 joint_zero_offsets_rad must be finite")
        return cls(
            segments=_apply_overrides(
                _parse_segments(resolved.kinematics_yaml), kinematics_overrides
            ),
            joint_limits=_parse_joint_limits(resolved.joint_limits_yaml),
            joint_zero_offsets_rad=offsets,
        )

    def base_t_flange(self, joint_positions_rad: Sequence[float]) -> PoseSE3:
        """Return the calibrated ``base_T_flange`` for controller joint readings."""

        return PoseSE3("base", "flange", self.forward_kinematics(joint_positions_rad))

    def joint_velocity_limits_rad_s(self) -> tuple[float, ...]:
        """Return the ES68 controller-profile limits without touching CS68 assets."""

        resources = Es68ModelResources.packaged()
        limits = _parse_joint_velocity_limits(resources.joint_limits_yaml)
        return tuple(limits[name] for name in ES68_JOINT_NAMES)


def load_es68_flange_t_tcp(
    resources: Es68ModelResources | None = None,
) -> PoseSE3:
    """Load HoloRobot's independently fitted flange-to-RTSI-TCP validation offset."""

    resolved = resources or Es68ModelResources.packaged()
    try:
        payload = json.loads(resolved.tcp_offset_json.read_text(encoding="utf-8"))
        offset = payload["tcp_offset"]
        if str(offset["frame"]) != "flange":
            raise ValueError("TCP offset frame is not flange")
        translation = np.asarray(offset["translation_m"], dtype=np.float64)
        roll, pitch, yaw = (float(value) for value in offset["rotation_rpy_rad"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid ES68 TCP offset {resolved.tcp_offset_json}: {exc}") from exc

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return PoseSE3.from_rotation_translation("flange", "tcp", rotation, translation)
