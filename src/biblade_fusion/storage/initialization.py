"""Atomic persistence for initial point-cloud and proxy artifacts."""

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

from biblade_fusion.calibration import HandEyeCalibration, load_hand_eye_calibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
    ProxyModelConfig,
)
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.robotics import (
    Es68KinematicModel,
    Es68ModelResources,
    load_es68_flange_t_tcp,
)
from biblade_fusion.workflows import AuthoritativeRobotPose, InitialObservation

INITIALIZATION_SCHEMA_VERSION = 7


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(path: Path) -> dict[str, Any]:
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


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Initialization source asset is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _pose_authority_payload(authority: AuthoritativeRobotPose) -> dict[str, Any]:
    return {
        "method": "joints_to_packaged_es68_fk_v1",
        "controller_tcp_role": "validation_only",
        "joint_zero_offsets_rad": list(authority.joint_zero_offsets_rad),
        "base_T_flange": authority.base_t_flange.matrix.tolist(),
        "predicted_base_T_tcp": authority.predicted_base_t_tcp.matrix.tolist(),
        "observed_base_T_tcp": authority.observed_base_t_tcp.matrix.tolist(),
        "fk_tcp_translation_error_m": authority.fk_tcp_translation_error_m,
        "fk_tcp_rotation_error_deg": authority.fk_tcp_rotation_error_deg,
        "maximum_fk_tcp_translation_error_m": (
            authority.maximum_fk_tcp_translation_error_m
        ),
        "maximum_fk_tcp_rotation_error_deg": (
            authority.maximum_fk_tcp_rotation_error_deg
        ),
    }


def _verified_source_hand_eye(
    hand_eye: HandEyeCalibration,
    config: HandEyeConfig,
) -> HandEyeCalibration:
    loaded = load_hand_eye_calibration(
        config.model_copy(update={"calibration_path": hand_eye.source_path})
    )
    if (
        loaded.method != hand_eye.method
        or loaded.flange_t_left_ir is None
        or hand_eye.flange_t_left_ir is None
        or not np.allclose(
            loaded.flange_t_left_ir.matrix,
            hand_eye.flange_t_left_ir.matrix,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            loaded.tcp_t_left_ir.matrix,
            hand_eye.tcp_t_left_ir.matrix,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Initialization in-memory hand-eye does not match its source")
    return loaded


def _load_contained_array(root: Path, record: Any) -> np.ndarray:
    relative_path = record["path"] if isinstance(record, dict) else record
    stored_path = Path(str(relative_path))
    if stored_path.is_absolute():
        raise ValueError(f"absolute artifact path is forbidden: {stored_path}")
    resolved_root = root.resolve()
    path = (resolved_root / stored_path).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"artifact path escapes output directory: {stored_path}")
    if isinstance(record, dict) and _sha256(path) != str(record["sha256"]):
        raise ValueError(f"initialization checksum mismatch: {stored_path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"artifact file must be a single .npy array: {stored_path}")
    if isinstance(record, dict) and (
        str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]
    ):
        raise ValueError(f"initialization array manifest mismatch: {stored_path}")
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
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
    *,
    source_session: str | Path,
    source_stereo_inference: str | Path | None = None,
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

    hand_eye = _verified_source_hand_eye(hand_eye, hand_eye_config)
    flange_t_left_ir = hand_eye.require_flange_primary()
    authority = observation.pose_authority
    if authority is None:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Initialization requires authoritative ES68 FK pose evidence")
    if tuple(kinematics_config.joint_zero_offsets_rad) != authority.joint_zero_offsets_rad:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Initialization kinematics offsets do not match pose evidence")
    if (
        authority.maximum_fk_tcp_translation_error_m
        != hand_eye_config.maximum_fk_tcp_translation_error_m
        or authority.maximum_fk_tcp_rotation_error_deg
        != hand_eye_config.maximum_fk_tcp_rotation_error_deg
    ):
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Initialization hand-eye gate does not match pose evidence")
    if not np.allclose(
        authority.base_t_flange.compose(flange_t_left_ir).matrix,
        observation.base_t_left_ir.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Initialization camera pose is not derived from flange-primary hand-eye")
    flange_t_tcp = load_es68_flange_t_tcp()
    if not np.allclose(
        authority.base_t_flange.compose(flange_t_tcp).matrix,
        authority.predicted_base_t_tcp.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError("Initialization predicted TCP is not derived from ES68 flange FK")
    resources = Es68ModelResources.packaged()
    left_ir_t_projection_camera = observation.base_t_left_ir.inverse().compose(
        observation.base_t_projection_camera
    )

    proxy = observation.proxy
    metadata: dict[str, Any] = {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "session": str(Path(source_session).resolve()),
            "view_id": observation.source_view_id,
            "sequence_index": observation.source_sequence_index,
            "frame_number": observation.source_frame_number,
            "stereo_inference": (
                str(Path(source_stereo_inference).resolve())
                if source_stereo_inference is not None
                else None
            ),
            "depth_source": observation.depth_source,
        },
        "files": {},
        "source_image_shape": list(observation.base_cloud.source_image_shape),
        "planning_intrinsics": _intrinsics_payload(observation.planning_intrinsics),
        "seed_joint_positions_rad": observation.seed_joint_positions_rad.tolist(),
        "transforms": {
            "base_T_left_ir": observation.base_t_left_ir.matrix.tolist(),
            "base_T_projection_camera": observation.base_t_projection_camera.matrix.tolist(),
            "projection_camera_frame": observation.base_t_projection_camera.child_frame,
            "left_ir_T_projection_camera": left_ir_t_projection_camera.matrix.tolist(),
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
            "source": _source_record(hand_eye.source_path),
            "method": hand_eye.method,
            "flange_T_left_ir": flange_t_left_ir.matrix.tolist(),
            "tcp_T_left_ir": hand_eye.tcp_t_left_ir.matrix.tolist(),
            "flange_T_tcp": flange_t_tcp.matrix.tolist(),
            "sample_count": hand_eye.sample_count,
            "translation_rmse_m": hand_eye.translation_rmse_m,
            "rotation_rmse_deg": hand_eye.rotation_rmse_deg,
            "rotation_span_deg": hand_eye.rotation_span_deg,
            "translation_span_m": hand_eye.translation_span_m,
            "rotation_axis_diversity": hand_eye.rotation_axis_diversity,
        },
        "pose_authority": _pose_authority_payload(authority),
        "kinematics_assets": {
            "model": _source_record(resources.kinematics_yaml),
            "joint_limits": _source_record(resources.joint_limits_yaml),
            "flange_tcp": _source_record(resources.tcp_offset_json),
        },
        "processing": {
            "point_cloud": point_cloud_config.model_dump(mode="json"),
            "proxy_model": proxy_config.model_dump(mode="json"),
            "kinematics": kinematics_config.model_dump(mode="json"),
            "hand_eye_gate": hand_eye_config.model_dump(mode="json"),
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
        metadata["files"] = {
            name: _array_record(temporary / filename)
            for name, filename in {
                "base_points_m": "base_points_m.npy",
                "pixel_uv": "pixel_uv.npy",
                "blade_mask": "blade_mask.npy",
            }.items()
        }
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
        schema_version = int(metadata["schema_version"])
        if schema_version not in {4, 5, 6, INITIALIZATION_SCHEMA_VERSION}:
            raise ValueError(f"unsupported schema {schema_version}")
        files = metadata["files"]
        points = _load_contained_array(root, files["base_points_m"])
        pixels = _load_contained_array(root, files["pixel_uv"])
        mask = _load_contained_array(root, files["blade_mask"])
        source_shape = tuple(int(value) for value in metadata["source_image_shape"])
        transforms = metadata["transforms"]
        proxy_data = metadata["proxy"]
        base_t_left_ir = PoseSE3("base", "left_ir", transforms["base_T_left_ir"])
        if schema_version == 4:
            planning_intrinsics_data = metadata["left_intrinsics"]
            base_t_projection_camera = PoseSE3("base", "depth", transforms["base_T_depth"])
            depth_source = "native_realsense"
        else:
            planning_intrinsics_data = metadata["planning_intrinsics"]
            projection_frame = str(transforms["projection_camera_frame"])
            base_t_projection_camera = PoseSE3(
                "base",
                projection_frame,
                transforms["base_T_projection_camera"],
            )
            depth_source = str(metadata["source"]["depth_source"])
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
            planning_intrinsics=_intrinsics_from_payload(planning_intrinsics_data),
            seed_joint_positions_rad=metadata["seed_joint_positions_rad"],
            base_t_left_ir=base_t_left_ir,
            base_t_projection_camera=base_t_projection_camera,
            base_cloud=cloud,
            proxy=proxy,
            depth_source=depth_source,
            source_sequence_index=int(metadata["source"].get("sequence_index", 0)),
            source_frame_number=int(metadata["source"].get("frame_number", 0)),
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
            source_path=Path(
                str(
                    hand_eye_data["source"]["path"]
                    if schema_version == INITIALIZATION_SCHEMA_VERSION
                    else hand_eye_data["source_path"]
                )
            ),
            rotation_span_deg=(
                float(hand_eye_data["rotation_span_deg"])
                if hand_eye_data.get("rotation_span_deg") is not None
                else None
            ),
            translation_span_m=(
                float(hand_eye_data["translation_span_m"])
                if hand_eye_data.get("translation_span_m") is not None
                else None
            ),
            rotation_axis_diversity=(
                float(hand_eye_data["rotation_axis_diversity"])
                if hand_eye_data.get("rotation_axis_diversity") is not None
                else None
            ),
            flange_t_left_ir=(
                PoseSE3("flange", "left_ir", hand_eye_data["flange_T_left_ir"])
                if schema_version == INITIALIZATION_SCHEMA_VERSION
                else None
            ),
        )
        pose_authority = None
        if schema_version == INITIALIZATION_SCHEMA_VERSION:
            for record in metadata["kinematics_assets"].values():
                asset_path = Path(str(record["path"])).resolve()
                if (
                    _sha256(asset_path) != str(record["sha256"])
                    or asset_path.stat().st_size != int(record["size_bytes"])
                ):
                    raise ValueError(f"initialization kinematics asset changed: {asset_path}")
            source_record = hand_eye_data["source"]
            source_path = Path(str(source_record["path"])).resolve()
            if (
                _sha256(source_path) != str(source_record["sha256"])
                or source_path.stat().st_size != int(source_record["size_bytes"])
            ):
                raise ValueError("initialization hand-eye source changed")
            processing = metadata["processing"]
            kinematics_config = KinematicsConfig.model_validate(processing["kinematics"])
            hand_eye_config = HandEyeConfig.model_validate(processing["hand_eye_gate"])
            loaded_hand_eye = _verified_source_hand_eye(hand_eye, hand_eye_config)
            hand_eye = loaded_hand_eye
            authority_data = metadata["pose_authority"]
            if authority_data["method"] != "joints_to_packaged_es68_fk_v1":
                raise ValueError("initialization pose authority method is unsupported")
            if authority_data["controller_tcp_role"] != "validation_only":
                raise ValueError("initialization controller TCP role is invalid")
            recorded_offsets = tuple(
                float(value) for value in authority_data["joint_zero_offsets_rad"]
            )
            if recorded_offsets != tuple(kinematics_config.joint_zero_offsets_rad):
                raise ValueError("initialization joint offsets changed")
            expected_base_t_flange = Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad
            ).base_t_flange(observation.seed_joint_positions_rad)
            flange_t_tcp = load_es68_flange_t_tcp()
            if not np.allclose(
                hand_eye_data["flange_T_tcp"],
                flange_t_tcp.matrix,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("initialization flange_T_tcp changed")
            pose_authority = AuthoritativeRobotPose(
                PoseSE3("base", "flange", authority_data["base_T_flange"]),
                PoseSE3("base", "tcp", authority_data["predicted_base_T_tcp"]),
                PoseSE3("base", "tcp", authority_data["observed_base_T_tcp"]),
                float(authority_data["fk_tcp_translation_error_m"]),
                float(authority_data["fk_tcp_rotation_error_deg"]),
                float(authority_data["maximum_fk_tcp_translation_error_m"]),
                float(authority_data["maximum_fk_tcp_rotation_error_deg"]),
                recorded_offsets,
            )
            if not np.allclose(
                pose_authority.base_t_flange.matrix,
                expected_base_t_flange.matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("initialization base_T_flange does not match ES68 FK")
            if not np.allclose(
                pose_authority.predicted_base_t_tcp.matrix,
                expected_base_t_flange.compose(flange_t_tcp).matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("initialization predicted base_T_tcp does not match ES68 FK")
            if (
                pose_authority.maximum_fk_tcp_translation_error_m
                != hand_eye_config.maximum_fk_tcp_translation_error_m
                or pose_authority.maximum_fk_tcp_rotation_error_deg
                != hand_eye_config.maximum_fk_tcp_rotation_error_deg
            ):
                raise ValueError("initialization FK/TCP gate changed")
            if not np.allclose(
                observation.base_t_left_ir.matrix,
                expected_base_t_flange.compose(hand_eye.require_flange_primary()).matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("initialization base_T_left_ir is not authoritative")
            left_ir_t_projection = PoseSE3(
                "left_ir",
                observation.base_t_projection_camera.child_frame,
                transforms["left_ir_T_projection_camera"],
            )
            if not np.allclose(
                observation.base_t_projection_camera.matrix,
                observation.base_t_left_ir.compose(left_ir_t_projection).matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("initialization projection-camera transform is inconsistent")
            observation = InitialObservation(
                source_view_id=observation.source_view_id,
                planning_intrinsics=observation.planning_intrinsics,
                seed_joint_positions_rad=observation.seed_joint_positions_rad,
                base_t_left_ir=observation.base_t_left_ir,
                base_t_projection_camera=observation.base_t_projection_camera,
                base_cloud=observation.base_cloud,
                proxy=observation.proxy,
                depth_source=observation.depth_source,
                source_sequence_index=observation.source_sequence_index,
                source_frame_number=observation.source_frame_number,
                pose_authority=pose_authority,
            )
        return StoredInitialization(observation, hand_eye, mask, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid initialization artifact {root}: {exc}") from exc
