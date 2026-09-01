"""Initial visible-face observation to bilateral planning proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition.bundle import SynchronizedFrameBundle
from biblade_fusion.calibration.hand_eye import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
    ProxyModelConfig,
)
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import (
    BilateralBladeProxy,
    build_bilateral_proxy,
    select_proxy_support,
)
from biblade_fusion.workflows.reconstruction import (
    AuthoritativeRobotPose,
    reconstruct_foundation_stereo_view,
    reconstruct_native_depth_view,
)
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
    source_sequence_index: int = 0
    source_frame_number: int = 0
    pose_authority: AuthoritativeRobotPose | None = None
    proxy_support_mask: NDArray[np.bool_] | None = None

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
        if self.source_sequence_index < 0 or self.source_frame_number < 0:
            raise ValueError("Initial source sequence and frame numbers must be non-negative")
        explicit_support_mask = self.proxy_support_mask is not None
        support_mask = (
            np.ones(self.base_cloud.points_m.shape[0], dtype=np.bool_)
            if self.proxy_support_mask is None
            else np.array(self.proxy_support_mask, dtype=np.bool_, copy=True)
        )
        if support_mask.shape != (self.base_cloud.points_m.shape[0],):
            raise ValueError("Initial proxy-support mask must match base-cloud points")
        support_count = int(np.count_nonzero(support_mask))
        if explicit_support_mask and support_count != self.proxy.raw_point_count:
            raise ValueError("Initial proxy-support mask does not match proxy input count")
        support_mask.setflags(write=False)
        object.__setattr__(self, "proxy_support_mask", support_mask)

    @property
    def proxy_support_points_m(self) -> NDArray[np.float64]:
        assert self.proxy_support_mask is not None
        return self.base_cloud.points_m[self.proxy_support_mask]


def initialize_native_depth(
    bundle: SynchronizedFrameBundle,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    proxy_config: ProxyModelConfig,
    *,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> InitialObservation:
    """Build a base-frame cloud and conservative proxy from one masked native depth view."""

    reconstructed = reconstruct_native_depth_view(
        bundle,
        blade_mask,
        hand_eye,
        point_cloud_config,
        kinematics_config=kinematics_config,
        hand_eye_config=hand_eye_config,
    )
    support = select_proxy_support(
        reconstructed.base_cloud.points_m,
        proxy_config,
        frame=reconstructed.base_cloud.frame,
    )
    proxy = build_bilateral_proxy(
        reconstructed.base_cloud.points_m[support.mask],
        reconstructed.base_t_projection_camera,
        proxy_config,
    )
    return InitialObservation(
        source_view_id=reconstructed.source_view_id,
        planning_intrinsics=reconstructed.planning_intrinsics,
        seed_joint_positions_rad=reconstructed.joint_positions_rad,
        base_t_left_ir=reconstructed.base_t_left_ir,
        base_t_projection_camera=reconstructed.base_t_projection_camera,
        base_cloud=reconstructed.base_cloud,
        proxy=proxy,
        source_sequence_index=reconstructed.source_sequence_index,
        source_frame_number=reconstructed.source_frame_number,
        pose_authority=reconstructed.pose_authority,
        proxy_support_mask=support.mask,
    )


def initialize_foundation_stereo_depth(
    bundle: SynchronizedFrameBundle,
    stereo_observation: StereoInferenceObservation,
    blade_mask: ArrayLike,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    proxy_config: ProxyModelConfig,
    *,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> InitialObservation:
    """Build the same bilateral proxy from calibrated FoundationStereo depth."""

    reconstructed = reconstruct_foundation_stereo_view(
        bundle,
        stereo_observation,
        blade_mask,
        hand_eye,
        point_cloud_config,
        kinematics_config=kinematics_config,
        hand_eye_config=hand_eye_config,
    )
    support = select_proxy_support(
        reconstructed.base_cloud.points_m,
        proxy_config,
        frame=reconstructed.base_cloud.frame,
    )
    proxy = build_bilateral_proxy(
        reconstructed.base_cloud.points_m[support.mask],
        reconstructed.base_t_projection_camera,
        proxy_config,
    )
    return InitialObservation(
        source_view_id=reconstructed.source_view_id,
        planning_intrinsics=reconstructed.planning_intrinsics,
        seed_joint_positions_rad=reconstructed.joint_positions_rad,
        base_t_left_ir=reconstructed.base_t_left_ir,
        base_t_projection_camera=reconstructed.base_t_projection_camera,
        base_cloud=reconstructed.base_cloud,
        proxy=proxy,
        depth_source="foundation_stereo",
        source_sequence_index=reconstructed.source_sequence_index,
        source_frame_number=reconstructed.source_frame_number,
        pose_authority=reconstructed.pose_authority,
        proxy_support_mask=support.mask,
    )
