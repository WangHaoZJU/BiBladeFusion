"""Read-only acquisition and persistence of controller-specific ES68 MDH data.

The public Python symbols retain their historical ``cs68`` spelling for artifact and
downstream API compatibility.  New assets identify the physical robot unambiguously as
ES68 and legacy schema-1 CS68 assets are not accepted by this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

KINEMATICS_SCHEMA_VERSION = 2


class RobotKinematicsError(ValueError):
    """Controller MDH data could not be safely acquired or validated."""


def _vector6(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite six-vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class Cs68KinematicsModel:
    dh_alpha_rad: NDArray[np.float64]
    dh_a_m: NDArray[np.float64]
    dh_d_m: NDArray[np.float64]
    source: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Kinematics model source must be non-empty")
        object.__setattr__(self, "dh_alpha_rad", _vector6(self.dh_alpha_rad, "DH alpha"))
        object.__setattr__(self, "dh_a_m", _vector6(self.dh_a_m, "DH a"))
        object.__setattr__(self, "dh_d_m", _vector6(self.dh_d_m, "DH d"))


def fetch_cs68_kinematics(
    robot_ip: str,
    *,
    timeout_ms: int = 1000,
    sdk_module: ModuleType | Any | None = None,
) -> Cs68KinematicsModel:
    """Read MDH calibration through the Primary interface without issuing motion."""

    if not robot_ip:
        raise RobotKinematicsError("Robot IP is required to read kinematics")
    sdk = sdk_module or import_module("elite_cs_sdk")
    client = sdk.PrimaryClientInterface()
    try:
        if not client.connect(robot_ip):
            raise RobotKinematicsError(f"Cannot connect to Elite Primary interface at {robot_ip}")
        package = sdk.KinematicsInfo()
        if not client.getPackage(package, timeout_ms):
            raise RobotKinematicsError("Controller did not return KinematicsInfo")
        try:
            return Cs68KinematicsModel(
                package.dh_alpha_,
                package.dh_a_,
                package.dh_d_,
                source=f"Elite Primary interface at {robot_ip}",
            )
        except ValueError as exc:
            raise RobotKinematicsError(str(exc)) from exc
    finally:
        client.disconnect()


def write_cs68_kinematics(path: str | Path, model: Cs68KinematicsModel) -> Path:
    """Atomically store controller-specific ES68 MDH parameters as YAML."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Kinematics artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": KINEMATICS_SCHEMA_VERSION,
        "robot_model": "es68",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": model.source,
        "modified_dh": {
            "alpha_rad": model.dh_alpha_rad.tolist(),
            "a_m": model.dh_a_m.tolist(),
            "d_m": model.dh_d_m.tolist(),
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_cs68_kinematics(path: str | Path) -> Cs68KinematicsModel:
    """Load a controller-specific ES68 MDH artifact for offline IK."""

    source_path = Path(path)
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("kinematics root must be a mapping")
        if int(payload["schema_version"]) != KINEMATICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        if str(payload["robot_model"]).lower() != "es68":
            raise ValueError("kinematics artifact is not for ES68")
        mdh = payload["modified_dh"]
        return Cs68KinematicsModel(
            mdh["alpha_rad"],
            mdh["a_m"],
            mdh["d_m"],
            source=str(payload["source"]),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise RobotKinematicsError(
            f"Invalid ES68 kinematics artifact {source_path}: {exc}"
        ) from exc
