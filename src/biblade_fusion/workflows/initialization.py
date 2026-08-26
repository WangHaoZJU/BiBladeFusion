"""Initial visible-face observation to bilateral planning proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition.bundle import SynchronizedFrameBundle
from biblade_fusion.calibration.hand_eye import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig, ProxyModelConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import (
    PointCloud,
    depth_image_to_point_cloud,
    native_depth_to_meters,
    realsense_depth_image_to_point_cloud,
)
from biblade_fusion.perception.proxy import BilateralBladeProxy, build_bilateral_proxy
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


class InitializationError(ValueError):
    """A synchronized view cannot safely initialize bilateral planning."""


@dataclass(frozen=True, slots=True)
class InitialObservation:
    source_view_id: str
    planning_intrinsics: CameraIntrinsics
    seed_joint_positions_rad: NDArray[np.float64]
    base_t_left_ir: PoseSE3
    base_t_projection_camera: PoseSE3
    base_cloud: PointCloud
    proxy: BilateralBladeProxy
    depth_source: Literal["native_realsense", "foundation_stereo"] = "native_realsense"

    def __post_init__(self) -> None:
        joints = np.array(self.seed_joint_positions_rad, dtype=np.float64, copy=True)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Initial seed joints must be a finite six-vector")
        joints.setflags(write=False)
        object.__setattr__(self, "seed_joint_positions_rad", joints)
        if (
            self.base_t_left_ir.parent_frame != "base"
            or self.base_t_left_ir.child_frame != "left_ir"
        ):
            raise ValueError("Initial camera pose must be base_T_left_ir")
        if self.base_t_projection_camera.parent_frame != "base":
            raise ValueError("Initial projection-camera pose must have base as parent")
        expected_projection_frame = {
            "native_realsense": "depth",
            "foundation_stereo": "left_rectified",
        }[self.depth_source]
        if self.base_t_projection_camera.child_frame != expected_projection_frame:
            raise ValueError(
                f"{self.depth_source} requires base_T_{expected_projection_frame}"
            )
        if self.base_cloud.frame != "base":
            raise ValueError("Initial point cloud must be in the base frame")
        if self.proxy.frame_T_proxy.parent_frame != "base":
            raise ValueError("Initial proxy must be in the base frame")


def initialize_native_depth(
    bundle: SynchronizedFrameBundle,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    proxy_config: ProxyModelConfig,
) -> InitialObservation:
    """Build a base-frame cloud and conservative proxy from one masked native depth view."""

    stereo = bundle.stereo
    calibration = stereo.calibration
    if stereo.native_depth is None:
        raise InitializationError("Stored view has no native depth")
    if calibration.native_depth_scale_m is None:
        raise InitializationError("Stored view has no native depth scale")
    if calibration.depth is None or calibration.left_t_depth is None:
        raise InitializationError("Stored view has no depth-stream calibration")

    depth_m = native_depth_to_meters(stereo.native_depth, calibration.native_depth_scale_m)
    depth_cloud = realsense_depth_image_to_point_cloud(
        depth_m,
        calibration.depth,
        point_cloud_config,
        frame="depth",
        valid_mask=blade_mask,
    )
    base_t_left_ir = bundle.selected_robot_state.base_t_tcp.compose(hand_eye.tcp_t_left_ir)
    base_t_depth = base_t_left_ir.compose(calibration.left_t_depth)
    base_cloud = depth_cloud.transformed(base_t_depth)
    proxy = build_bilateral_proxy(
        base_cloud.points_m,
        base_t_left_ir,
        proxy_config,
    )
    return InitialObservation(
        source_view_id=bundle.view_id,
        planning_intrinsics=calibration.left,
        seed_joint_positions_rad=bundle.selected_robot_state.joint_positions_rad,
        base_t_left_ir=base_t_left_ir,
        base_t_projection_camera=base_t_depth,
        base_cloud=base_cloud,
        proxy=proxy,
    )


def initialize_foundation_stereo_depth(
    bundle: SynchronizedFrameBundle,
    stereo_observation: StereoInferenceObservation,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    proxy_config: ProxyModelConfig,
) -> InitialObservation:
    """Build the same bilateral proxy from calibrated FoundationStereo depth."""

    if (
        stereo_observation.source_view_id != bundle.view_id
        or stereo_observation.source_sequence_index != bundle.sequence_index
        or stereo_observation.rectified.source_frame_number != bundle.stereo.frame_number
    ):
        raise InitializationError("Stereo inference artifact does not match the stored view")
    mask = np.asarray(blade_mask, dtype=np.bool_)
    if mask.shape != stereo_observation.depth_m.shape:
        raise InitializationError("Blade mask must match the rectified stereo depth image")

    calibration = stereo_observation.rectified.calibration
    combined_valid = mask & stereo_observation.result.valid_mask
    projection_cloud = depth_image_to_point_cloud(
        stereo_observation.depth_m,
        calibration.left,
        point_cloud_config,
        frame="left_rectified",
        valid_mask=combined_valid,
    )
    base_t_left_ir = bundle.selected_robot_state.base_t_tcp.compose(hand_eye.tcp_t_left_ir)
    base_t_left_rectified = base_t_left_ir.compose(
        calibration.left_rectified_t_left_ir.inverse()
    )
    base_cloud = projection_cloud.transformed(base_t_left_rectified)
    proxy = build_bilateral_proxy(
        base_cloud.points_m,
        base_t_left_rectified,
        proxy_config,
    )
    return InitialObservation(
        source_view_id=bundle.view_id,
        planning_intrinsics=calibration.left,
        seed_joint_positions_rad=bundle.selected_robot_state.joint_positions_rad,
        base_t_left_ir=base_t_left_ir,
        base_t_projection_camera=base_t_left_rectified,
        base_cloud=base_cloud,
        proxy=proxy,
        depth_source="foundation_stereo",
    )
