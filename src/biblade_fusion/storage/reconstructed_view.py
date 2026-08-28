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

from biblade_fusion.calibration import HandEyeCalibration, load_hand_eye_calibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.robotics import (
    Es68KinematicModel,
    Es68ModelResources,
    load_es68_flange_t_tcp,
)
from biblade_fusion.workflows import AuthoritativeRobotPose, ReconstructedBladeView

RECONSTRUCTED_VIEW_SCHEMA_VERSION = 2


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


def _source_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Reconstructed-view source asset is missing: {resolved}")
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
        raise ValueError("Reconstructed-view hand-eye does not match its source")
    return loaded


def write_reconstructed_view(
    output_dir: str | Path,
    view: ReconstructedBladeView,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
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
    hand_eye = _verified_source_hand_eye(hand_eye, hand_eye_config)
    flange_t_left_ir = hand_eye.require_flange_primary()
    authority = view.pose_authority
    if authority is None:
        raise ValueError("Reconstructed view requires authoritative ES68 FK pose evidence")
    if tuple(kinematics_config.joint_zero_offsets_rad) != authority.joint_zero_offsets_rad:
        raise ValueError("Reconstructed-view offsets do not match pose evidence")
    if (
        authority.maximum_fk_tcp_translation_error_m
        != hand_eye_config.maximum_fk_tcp_translation_error_m
        or authority.maximum_fk_tcp_rotation_error_deg
        != hand_eye_config.maximum_fk_tcp_rotation_error_deg
    ):
        raise ValueError("Reconstructed-view hand-eye gate does not match pose evidence")
    if not np.allclose(
        authority.base_t_flange.compose(flange_t_left_ir).matrix,
        view.base_t_left_ir.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Reconstructed camera pose is not derived from flange-primary hand-eye")
    flange_t_tcp = load_es68_flange_t_tcp()
    if not np.allclose(
        authority.base_t_flange.compose(flange_t_tcp).matrix,
        authority.predicted_base_t_tcp.matrix,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Reconstructed predicted TCP is not derived from ES68 flange FK")
    resources = Es68ModelResources.packaged()
    left_ir_t_projection_camera = view.base_t_left_ir.inverse().compose(
        view.base_t_projection_camera
    )
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
                "left_ir_T_projection_camera": left_ir_t_projection_camera.matrix.tolist(),
            },
            "hand_eye": {
                "source": _source_record(hand_eye.source_path),
                "method": hand_eye.method,
                "flange_T_left_ir": flange_t_left_ir.matrix.tolist(),
                "tcp_T_left_ir": hand_eye.tcp_t_left_ir.matrix.tolist(),
                "flange_T_tcp": flange_t_tcp.matrix.tolist(),
            },
            "pose_authority": _pose_authority_payload(authority),
            "kinematics_assets": {
                "model": _source_record(resources.kinematics_yaml),
                "joint_limits": _source_record(resources.joint_limits_yaml),
                "flange_tcp": _source_record(resources.tcp_offset_json),
            },
            "processing": {
                "point_cloud": point_cloud_config.model_dump(mode="json"),
                "kinematics": kinematics_config.model_dump(mode="json"),
                "hand_eye_gate": hand_eye_config.model_dump(mode="json"),
            },
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
        schema_version = int(payload["schema_version"])
        if schema_version not in {1, RECONSTRUCTED_VIEW_SCHEMA_VERSION}:
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
        pose_authority = None
        if schema_version == RECONSTRUCTED_VIEW_SCHEMA_VERSION:
            for record in payload["kinematics_assets"].values():
                asset_path = Path(str(record["path"])).resolve()
                if (
                    _sha256(asset_path) != str(record["sha256"])
                    or asset_path.stat().st_size != int(record["size_bytes"])
                ):
                    raise ValueError(f"reconstructed-view kinematics asset changed: {asset_path}")
            hand_eye_data = payload["hand_eye"]
            hand_eye_source = hand_eye_data["source"]
            source_path = Path(str(hand_eye_source["path"])).resolve()
            if (
                _sha256(source_path) != str(hand_eye_source["sha256"])
                or source_path.stat().st_size != int(hand_eye_source["size_bytes"])
            ):
                raise ValueError("reconstructed-view hand-eye source changed")
            hand_eye = HandEyeCalibration(
                PoseSE3("tcp", "left_ir", hand_eye_data["tcp_T_left_ir"]),
                str(hand_eye_data["method"]),
                None,
                None,
                None,
                source_path,
                flange_t_left_ir=PoseSE3(
                    "flange",
                    "left_ir",
                    hand_eye_data["flange_T_left_ir"],
                ),
            )
            processing = payload["processing"]
            kinematics_config = KinematicsConfig.model_validate(processing["kinematics"])
            hand_eye_config = HandEyeConfig.model_validate(processing["hand_eye_gate"])
            hand_eye = _verified_source_hand_eye(hand_eye, hand_eye_config)
            authority_data = payload["pose_authority"]
            if authority_data["method"] != "joints_to_packaged_es68_fk_v1":
                raise ValueError("reconstructed-view pose authority method is unsupported")
            if authority_data["controller_tcp_role"] != "validation_only":
                raise ValueError("reconstructed-view controller TCP role is invalid")
            offsets = tuple(
                float(value) for value in authority_data["joint_zero_offsets_rad"]
            )
            if offsets != tuple(kinematics_config.joint_zero_offsets_rad):
                raise ValueError("reconstructed-view joint offsets changed")
            expected_base_t_flange = Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad
            ).base_t_flange(payload["joint_positions_rad"])
            flange_t_tcp = load_es68_flange_t_tcp()
            if not np.allclose(
                hand_eye_data["flange_T_tcp"],
                flange_t_tcp.matrix,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("reconstructed-view flange_T_tcp changed")
            pose_authority = AuthoritativeRobotPose(
                PoseSE3("base", "flange", authority_data["base_T_flange"]),
                PoseSE3("base", "tcp", authority_data["predicted_base_T_tcp"]),
                PoseSE3("base", "tcp", authority_data["observed_base_T_tcp"]),
                float(authority_data["fk_tcp_translation_error_m"]),
                float(authority_data["fk_tcp_rotation_error_deg"]),
                float(authority_data["maximum_fk_tcp_translation_error_m"]),
                float(authority_data["maximum_fk_tcp_rotation_error_deg"]),
                offsets,
            )
            if not np.allclose(
                pose_authority.base_t_flange.matrix,
                expected_base_t_flange.matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("reconstructed-view base_T_flange does not match ES68 FK")
            if not np.allclose(
                pose_authority.predicted_base_t_tcp.matrix,
                expected_base_t_flange.compose(flange_t_tcp).matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("reconstructed-view predicted base_T_tcp is not authoritative")
            if (
                pose_authority.maximum_fk_tcp_translation_error_m
                != hand_eye_config.maximum_fk_tcp_translation_error_m
                or pose_authority.maximum_fk_tcp_rotation_error_deg
                != hand_eye_config.maximum_fk_tcp_rotation_error_deg
            ):
                raise ValueError("reconstructed-view FK/TCP gate changed")
            expected_base_t_left_ir = expected_base_t_flange.compose(
                hand_eye.require_flange_primary()
            )
            if not np.allclose(
                transforms["base_T_left_ir"],
                expected_base_t_left_ir.matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("reconstructed-view base_T_left_ir is not authoritative")
            left_ir_t_projection = PoseSE3(
                "left_ir",
                str(transforms["projection_camera_frame"]),
                transforms["left_ir_T_projection_camera"],
            )
            if not np.allclose(
                transforms["base_T_projection_camera"],
                expected_base_t_left_ir.compose(left_ir_t_projection).matrix,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("reconstructed-view projection transform is inconsistent")
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
            pose_authority,
        )
        return StoredReconstructedBladeView(view, arrays["blade_mask"], payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reconstructed-view artifact {root}: {exc}") from exc
