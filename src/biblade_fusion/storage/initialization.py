"""Atomic persistence for initial point-cloud and proxy artifacts."""

from __future__ import annotations

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
from biblade_fusion.core.settings import PointCloudConfig, ProxyModelConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.workflows import InitialObservation

INITIALIZATION_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class StoredInitialization:
    observation: InitialObservation
    hand_eye: HandEyeCalibration
    blade_mask: NDArray[np.bool_]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        mask = np.array(self.blade_mask, dtype=np.bool_, copy=True)
        if mask.shape != self.observation.base_cloud.source_image_shape:
            raise ValueError("Stored blade mask does not match source image shape")
        mask.setflags(write=False)
        object.__setattr__(self, "blade_mask", mask)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _load_contained_array(root: Path, relative_path: Any) -> np.ndarray:
    stored_path = Path(str(relative_path))
    if stored_path.is_absolute():
        raise ValueError(f"absolute artifact path is forbidden: {stored_path}")
    resolved_root = root.resolve()
    path = (resolved_root / stored_path).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"artifact path escapes output directory: {stored_path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"artifact file must be a single .npy array: {stored_path}")
    return array


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


def _intrinsics_from_payload(payload: dict[str, Any]) -> CameraIntrinsics:
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


def write_initialization(
    output_dir: str | Path,
    observation: InitialObservation,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    proxy_config: ProxyModelConfig,
    *,
    source_session: str | Path,
) -> Path:
    """Atomically write an initialization result without overwriting prior work."""

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"Initialization output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    mask = np.asarray(blade_mask, dtype=np.bool_)
    if mask.shape != observation.base_cloud.source_image_shape:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Blade mask does not match source image shape")

    proxy = observation.proxy
    metadata: dict[str, Any] = {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "session": str(Path(source_session).resolve()),
            "view_id": observation.source_view_id,
        },
        "files": {
            "base_points_m": "base_points_m.npy",
            "pixel_uv": "pixel_uv.npy",
            "blade_mask": "blade_mask.npy",
        },
        "source_image_shape": list(observation.base_cloud.source_image_shape),
        "left_intrinsics": _intrinsics_payload(observation.left_intrinsics),
        "seed_joint_positions_rad": observation.seed_joint_positions_rad.tolist(),
        "transforms": {
            "base_T_left_ir": observation.base_t_left_ir.matrix.tolist(),
            "base_T_depth": observation.base_t_depth.matrix.tolist(),
        },
        "proxy": {
            "base_T_proxy": proxy.frame_T_proxy.matrix.tolist(),
            "extents_m": proxy.extents_m.tolist(),
            "observed_surface_centroid_m": proxy.observed_surface_centroid_m.tolist(),
            "pca_eigenvalues_m2": proxy.pca_eigenvalues_m2.tolist(),
            "raw_point_count": proxy.raw_point_count,
            "finite_point_count": proxy.finite_point_count,
            "voxel_point_count": proxy.voxel_point_count,
            "camera_normal_cosine": proxy.camera_normal_cosine,
        },
        "hand_eye": {
            "source_path": str(hand_eye.source_path.resolve()),
            "method": hand_eye.method,
            "tcp_T_left_ir": hand_eye.tcp_t_left_ir.matrix.tolist(),
            "sample_count": hand_eye.sample_count,
            "translation_rmse_m": hand_eye.translation_rmse_m,
            "rotation_rmse_deg": hand_eye.rotation_rmse_deg,
        },
        "processing": {
            "point_cloud": point_cloud_config.model_dump(mode="json"),
            "proxy_model": proxy_config.model_dump(mode="json"),
        },
    }
    try:
        np.save(
            temporary / "base_points_m.npy",
            observation.base_cloud.points_m,
            allow_pickle=False,
        )
        np.save(temporary / "pixel_uv.npy", observation.base_cloud.pixel_uv, allow_pickle=False)
        np.save(temporary / "blade_mask.npy", mask, allow_pickle=False)
        _atomic_json(temporary / "metadata.json", metadata)
        temporary.replace(output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def read_initialization(path: str | Path) -> StoredInitialization:
    """Load an initialization artifact into immutable planning contracts."""

    root = Path(path)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("metadata root must be an object")
        if int(metadata["schema_version"]) != INITIALIZATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {metadata['schema_version']}")
        files = metadata["files"]
        points = _load_contained_array(root, files["base_points_m"])
        pixels = _load_contained_array(root, files["pixel_uv"])
        mask = _load_contained_array(root, files["blade_mask"])
        source_shape = tuple(int(value) for value in metadata["source_image_shape"])
        transforms = metadata["transforms"]
        proxy_data = metadata["proxy"]
        base_t_left_ir = PoseSE3("base", "left_ir", transforms["base_T_left_ir"])
        base_t_depth = PoseSE3("base", "depth", transforms["base_T_depth"])
        cloud = PointCloud("base", points, pixels, source_shape)
        proxy = BilateralBladeProxy(
            frame_T_proxy=PoseSE3("base", "blade_proxy", proxy_data["base_T_proxy"]),
            extents_m=proxy_data["extents_m"],
            observed_surface_centroid_m=proxy_data["observed_surface_centroid_m"],
            pca_eigenvalues_m2=proxy_data["pca_eigenvalues_m2"],
            raw_point_count=int(proxy_data["raw_point_count"]),
            finite_point_count=int(proxy_data["finite_point_count"]),
            voxel_point_count=int(proxy_data["voxel_point_count"]),
            camera_normal_cosine=float(proxy_data["camera_normal_cosine"]),
        )
        observation = InitialObservation(
            source_view_id=str(metadata["source"]["view_id"]),
            left_intrinsics=_intrinsics_from_payload(metadata["left_intrinsics"]),
            seed_joint_positions_rad=metadata["seed_joint_positions_rad"],
            base_t_left_ir=base_t_left_ir,
            base_t_depth=base_t_depth,
            base_cloud=cloud,
            proxy=proxy,
        )
        hand_eye_data = metadata["hand_eye"]
        hand_eye = HandEyeCalibration(
            tcp_t_left_ir=PoseSE3("tcp", "left_ir", hand_eye_data["tcp_T_left_ir"]),
            method=str(hand_eye_data["method"]),
            sample_count=(
                int(hand_eye_data["sample_count"])
                if hand_eye_data.get("sample_count") is not None
                else None
            ),
            translation_rmse_m=(
                float(hand_eye_data["translation_rmse_m"])
                if hand_eye_data.get("translation_rmse_m") is not None
                else None
            ),
            rotation_rmse_deg=(
                float(hand_eye_data["rotation_rmse_deg"])
                if hand_eye_data.get("rotation_rmse_deg") is not None
                else None
            ),
            source_path=Path(str(hand_eye_data["source_path"])),
        )
        return StoredInitialization(observation, hand_eye, mask, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid initialization artifact {root}: {exc}") from exc
