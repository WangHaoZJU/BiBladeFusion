"""Immutable pose-registered blade-view artifacts for multi-view coverage updates."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.workflows import ReconstructedBladeView

RECONSTRUCTED_VIEW_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredReconstructedBladeView:
    view: ReconstructedBladeView
    blade_mask: NDArray[np.bool_]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        mask = np.array(self.blade_mask, dtype=np.bool_, copy=True)
        if mask.shape != self.view.base_cloud.source_image_shape:
            raise ValueError("Stored blade mask does not match source image shape")
        mask.setflags(write=False)
        object.__setattr__(self, "blade_mask", mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion_model": intrinsics.distortion_model,
        "distortion_coefficients": list(intrinsics.distortion_coefficients),
    }


def _intrinsics(payload: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        int(payload["width"]),
        int(payload["height"]),
        float(payload["fx"]),
        float(payload["fy"]),
        float(payload["cx"]),
        float(payload["cy"]),
        str(payload["distortion_model"]),
        tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            "path": path.name,
            "sha256": _sha256(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def write_reconstructed_view(
    output_dir: str | Path,
    view: ReconstructedBladeView,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    *,
    source_session: str | Path,
    source_stereo_inference: str | Path | None = None,
) -> Path:
    """Persist one registered view without changing the reference proxy."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Reconstructed view output already exists: {output}")
    mask = np.asarray(blade_mask, dtype=np.bool_)
    if mask.shape != view.base_cloud.source_image_shape:
        raise ValueError("Blade mask does not match reconstructed source image shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    arrays = {
        "base_points_m": view.base_cloud.points_m,
        "pixel_uv": view.base_cloud.pixel_uv,
        "blade_mask": mask,
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / f"{name}.npy", array, allow_pickle=False)
        payload: dict[str, Any] = {
            "schema_version": RECONSTRUCTED_VIEW_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": {
                "session": str(Path(source_session).resolve()),
                "stereo_inference": (
                    str(Path(source_stereo_inference).resolve())
                    if source_stereo_inference is not None
                    else None
                ),
                "view_id": view.source_view_id,
                "sequence_index": view.source_sequence_index,
                "frame_number": view.source_frame_number,
                "depth_source": view.depth_source,
            },
            "files": {
                name: _record(temporary / f"{name}.npy") for name in arrays
            },
            "source_image_shape": list(view.base_cloud.source_image_shape),
            "planning_intrinsics": _intrinsics_payload(view.planning_intrinsics),
            "joint_positions_rad": view.joint_positions_rad.tolist(),
            "transforms": {
                "base_T_left_ir": view.base_t_left_ir.matrix.tolist(),
                "base_T_projection_camera": view.base_t_projection_camera.matrix.tolist(),
                "projection_camera_frame": view.base_t_projection_camera.child_frame,
            },
            "hand_eye": {
                "source_path": str(hand_eye.source_path.resolve()),
                "method": hand_eye.method,
                "tcp_T_left_ir": hand_eye.tcp_t_left_ir.matrix.tolist(),
            },
            "processing": {"point_cloud": point_cloud_config.model_dump(mode="json")},
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = Path(str(record["path"]))
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"reconstructed-view path escapes output: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"reconstructed-view checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"reconstructed-view array manifest mismatch: {relative}")
    return array


def read_reconstructed_view(path: str | Path) -> StoredReconstructedBladeView:
    """Validate and reconstruct one registered blade view."""

    root = Path(path)
    try:
        payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if int(payload["schema_version"]) != RECONSTRUCTED_VIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        arrays = {
            name: _load_array(root, record) for name, record in payload["files"].items()
        }
        source = payload["source"]
        transforms = payload["transforms"]
        cloud = PointCloud(
            "base",
            arrays["base_points_m"],
            arrays["pixel_uv"],
            tuple(int(value) for value in payload["source_image_shape"]),
        )
        view = ReconstructedBladeView(
            str(source["view_id"]),
            int(source["sequence_index"]),
            int(source["frame_number"]),
            _intrinsics(payload["planning_intrinsics"]),
            payload["joint_positions_rad"],
            PoseSE3("base", "left_ir", transforms["base_T_left_ir"]),
            PoseSE3(
                "base",
                str(transforms["projection_camera_frame"]),
                transforms["base_T_projection_camera"],
            ),
            cloud,
            str(source["depth_source"]),
        )
        return StoredReconstructedBladeView(view, arrays["blade_mask"], payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reconstructed-view artifact {root}: {exc}") from exc
