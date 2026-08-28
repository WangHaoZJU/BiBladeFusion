"""Pose-register one masked native or stereo blade depth view into the robot base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig, KinematicsConfig, PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import (
    PointCloud,
    depth_image_to_point_cloud,
    native_depth_to_meters,
    realsense_depth_image_to_point_cloud,
)
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


class ReconstructionError(ValueError):
    """A captured view cannot safely produce a pose-registered blade cloud."""


def _rotation_error_deg(predicted: PoseSE3, observed: PoseSE3) -> float:
    relative = predicted.rotation.T @ observed.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


@dataclass(frozen=True, slots=True)
class AuthoritativeRobotPose:
    """Auditable joints-to-flange pose with controller TCP as validation only."""

    base_t_flange: PoseSE3
    predicted_base_t_tcp: PoseSE3
    observed_base_t_tcp: PoseSE3
    fk_tcp_translation_error_m: float
    fk_tcp_rotation_error_deg: float
    maximum_fk_tcp_translation_error_m: float
    maximum_fk_tcp_rotation_error_deg: float
    joint_zero_offsets_rad: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        frames = (
            (self.base_t_flange, "flange"),
            (self.predicted_base_t_tcp, "tcp"),
            (self.observed_base_t_tcp, "tcp"),
        )
        if any(
            pose.parent_frame != "base" or pose.child_frame != child
            for pose, child in frames
        ):
            raise ValueError("Authoritative robot-pose evidence has invalid frames")
        offsets = tuple(float(value) for value in self.joint_zero_offsets_rad)
        if len(offsets) != 6 or not np.isfinite(offsets).all():
            raise ValueError("Authoritative robot-pose offsets must be a finite six-vector")
        try:
            reproduced_base_t_tcp = self.base_t_flange.compose(
                load_es68_flange_t_tcp()
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot validate authoritative pose against packaged ES68 flange_T_tcp"
            ) from exc
        if not np.allclose(
            self.predicted_base_t_tcp.matrix,
            reproduced_base_t_tcp.matrix,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Authoritative predicted base_T_tcp is not "
                "base_T_flange composed with packaged flange_T_tcp"
            )
        errors = (
            float(self.fk_tcp_translation_error_m),
            float(self.fk_tcp_rotation_error_deg),
        )
        reproduced = (
            float(
                np.linalg.norm(
                    self.predicted_base_t_tcp.translation_m
                    - self.observed_base_t_tcp.translation_m
                )
            ),
            _rotation_error_deg(self.predicted_base_t_tcp, self.observed_base_t_tcp),
        )
        if not np.allclose(errors, reproduced, rtol=0.0, atol=1e-12):
            raise ValueError("Authoritative FK/TCP residuals do not match pose evidence")
        limits = (
            float(self.maximum_fk_tcp_translation_error_m),
            float(self.maximum_fk_tcp_rotation_error_deg),
        )
        if (
            not np.isfinite((*errors, *limits)).all()
            or any(value < 0.0 for value in errors)
            or limits[0] <= 0.0
            or not 0.0 < limits[1] <= 180.0
        ):
            raise ValueError("Authoritative FK/TCP residuals or limits are invalid")
        if errors[0] > limits[0] or errors[1] > limits[1]:
            raise ValueError("Authoritative controller TCP residual exceeds its gate")
        object.__setattr__(self, "joint_zero_offsets_rad", offsets)


def resolve_authoritative_robot_pose(
    bundle: SynchronizedFrameBundle,
    hand_eye: HandEyeCalibration,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> tuple[PoseSE3, AuthoritativeRobotPose]:
    """Resolve ``base_T_left_ir`` from ES68 FK and gate the observed controller TCP."""

    try:
        flange_t_left_ir = hand_eye.require_flange_primary()
        model = Es68KinematicModel.from_resources(
            joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad
        )
        base_t_flange = model.base_t_flange(
            bundle.selected_robot_state.joint_positions_rad
        )
        predicted_base_t_tcp = base_t_flange.compose(load_es68_flange_t_tcp())
    except (OSError, TypeError, ValueError) as exc:
        raise ReconstructionError(f"Cannot resolve authoritative ES68 FK pose: {exc}") from exc
    observed_base_t_tcp = bundle.selected_robot_state.base_t_tcp
    translation_error_m = float(
        np.linalg.norm(
            predicted_base_t_tcp.translation_m - observed_base_t_tcp.translation_m
        )
    )
    rotation_error_deg = _rotation_error_deg(
        predicted_base_t_tcp,
        observed_base_t_tcp,
    )
    violations = []
    if translation_error_m > hand_eye_config.maximum_fk_tcp_translation_error_m:
        violations.append("translation")
    if rotation_error_deg > hand_eye_config.maximum_fk_tcp_rotation_error_deg:
        violations.append("rotation")
    if violations:
        raise ReconstructionError(
            "ES68 FK and observed controller TCP disagree: "
            + ", ".join(violations)
            + f" (translation={translation_error_m:.6f} m, "
            f"rotation={rotation_error_deg:.6f} deg)"
        )
    authority = AuthoritativeRobotPose(
        base_t_flange=base_t_flange,
        predicted_base_t_tcp=predicted_base_t_tcp,
        observed_base_t_tcp=observed_base_t_tcp,
        fk_tcp_translation_error_m=translation_error_m,
        fk_tcp_rotation_error_deg=rotation_error_deg,
        maximum_fk_tcp_translation_error_m=(
            hand_eye_config.maximum_fk_tcp_translation_error_m
        ),
        maximum_fk_tcp_rotation_error_deg=(
            hand_eye_config.maximum_fk_tcp_rotation_error_deg
        ),
        joint_zero_offsets_rad=kinematics_config.joint_zero_offsets_rad,
    )
    return base_t_flange.compose(flange_t_left_ir), authority


@dataclass(frozen=True, slots=True)
class ReconstructedBladeView:
    source_view_id: str
    source_sequence_index: int
    source_frame_number: int
    planning_intrinsics: CameraIntrinsics
    joint_positions_rad: NDArray[np.float64]
    base_t_left_ir: PoseSE3
    base_t_projection_camera: PoseSE3
    base_cloud: PointCloud
    depth_source: Literal["native_realsense", "foundation_stereo"]
    pose_authority: AuthoritativeRobotPose | None = None

    def __post_init__(self) -> None:
        if (
            not self.source_view_id
            or self.source_sequence_index < 0
            or self.source_frame_number < 0
        ):
            raise ValueError("Reconstructed view source identity is invalid")
        joints = np.array(self.joint_positions_rad, dtype=np.float64, copy=True)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Reconstructed view joints must be a finite six-vector")
        joints.setflags(write=False)
        object.__setattr__(self, "joint_positions_rad", joints)
        if (self.base_t_left_ir.parent_frame, self.base_t_left_ir.child_frame) != (
            "base",
            "left_ir",
        ):
            raise ValueError("Reconstructed view requires base_T_left_ir")
        expected_frame = {
            "native_realsense": "depth",
            "foundation_stereo": "left_rectified",
        }[self.depth_source]
        if (
            self.base_t_projection_camera.parent_frame != "base"
            or self.base_t_projection_camera.child_frame != expected_frame
        ):
            raise ValueError(f"{self.depth_source} requires base_T_{expected_frame}")
        if self.base_cloud.frame != "base":
            raise ValueError("Reconstructed blade cloud must be in base")


def reconstruct_native_depth_view(
    bundle: SynchronizedFrameBundle,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    config: PointCloudConfig,
    *,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> ReconstructedBladeView:
    """Deproject a masked native D435i depth image and register it through robot pose."""

    stereo = bundle.stereo
    calibration = stereo.calibration
    if stereo.native_depth is None:
        raise ReconstructionError("Stored view has no native depth")
    if calibration.native_depth_scale_m is None:
        raise ReconstructionError("Stored view has no native depth scale")
    if calibration.depth is None or calibration.left_t_depth is None:
        raise ReconstructionError("Stored view has no depth-stream calibration")
    depth_m = native_depth_to_meters(stereo.native_depth, calibration.native_depth_scale_m)
    depth_cloud = realsense_depth_image_to_point_cloud(
        depth_m,
        calibration.depth,
        config,
        frame="depth",
        valid_mask=blade_mask,
    )
    base_t_left_ir, pose_authority = resolve_authoritative_robot_pose(
        bundle,
        hand_eye,
        kinematics_config,
        hand_eye_config,
    )
    base_t_depth = base_t_left_ir.compose(calibration.left_t_depth)
    return ReconstructedBladeView(
        bundle.view_id,
        bundle.sequence_index,
        stereo.frame_number,
        calibration.left,
        bundle.selected_robot_state.joint_positions_rad,
        base_t_left_ir,
        base_t_depth,
        depth_cloud.transformed(base_t_depth),
        "native_realsense",
        pose_authority,
    )


def reconstruct_foundation_stereo_view(
    bundle: SynchronizedFrameBundle,
    stereo_observation: StereoInferenceObservation,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    config: PointCloudConfig,
    *,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> ReconstructedBladeView:
    """Back-project calibrated stereo depth and register it through robot pose."""

    if (
        stereo_observation.source_view_id != bundle.view_id
        or stereo_observation.source_sequence_index != bundle.sequence_index
        or stereo_observation.rectified.source_frame_number != bundle.stereo.frame_number
    ):
        raise ReconstructionError("Stereo inference artifact does not match the stored view")
    mask = np.asarray(blade_mask, dtype=np.bool_)
    if mask.shape != stereo_observation.depth_m.shape:
        raise ReconstructionError("Blade mask must match the rectified stereo depth image")
    calibration = stereo_observation.rectified.calibration
    cloud = depth_image_to_point_cloud(
        stereo_observation.depth_m,
        calibration.left,
        config,
        frame="left_rectified",
        valid_mask=mask & stereo_observation.result.valid_mask,
    )
    base_t_left_ir, pose_authority = resolve_authoritative_robot_pose(
        bundle,
        hand_eye,
        kinematics_config,
        hand_eye_config,
    )
    base_t_left_rectified = base_t_left_ir.compose(
        calibration.left_rectified_t_left_ir.inverse()
    )
    return ReconstructedBladeView(
        bundle.view_id,
        bundle.sequence_index,
        bundle.stereo.frame_number,
        calibration.left,
        bundle.selected_robot_state.joint_positions_rad,
        base_t_left_ir,
        base_t_left_rectified,
        cloud.transformed(base_t_left_rectified),
        "foundation_stereo",
        pose_authority,
    )
