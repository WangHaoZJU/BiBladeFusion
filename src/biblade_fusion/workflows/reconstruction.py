"""Pose-register one masked native or stereo blade depth view into the robot base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import (
    PointCloud,
    depth_image_to_point_cloud,
    native_depth_to_meters,
    realsense_depth_image_to_point_cloud,
)
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


class ReconstructionError(ValueError):
    """A captured view cannot safely produce a pose-registered blade cloud."""


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
    base_t_left_ir = bundle.selected_robot_state.base_t_tcp.compose(hand_eye.tcp_t_left_ir)
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
    )


def reconstruct_foundation_stereo_view(
    bundle: SynchronizedFrameBundle,
    stereo_observation: StereoInferenceObservation,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    config: PointCloudConfig,
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
    base_t_left_ir = bundle.selected_robot_state.base_t_tcp.compose(hand_eye.tcp_t_left_ir)
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
    )
