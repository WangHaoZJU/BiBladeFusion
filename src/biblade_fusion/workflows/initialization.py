"""Initial visible-face observation to bilateral planning proxy."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from biblade_fusion.acquisition.bundle import SynchronizedFrameBundle
from biblade_fusion.calibration.hand_eye import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import PointCloudConfig, ProxyModelConfig
from biblade_fusion.perception.pointcloud import (
    PointCloud,
    native_depth_to_meters,
    realsense_depth_image_to_point_cloud,
)
from biblade_fusion.perception.proxy import BilateralBladeProxy, build_bilateral_proxy


class InitializationError(ValueError):
    """A synchronized view cannot safely initialize bilateral planning."""


@dataclass(frozen=True, slots=True)
class InitialObservation:
    source_view_id: str
    base_t_left_ir: PoseSE3
    base_t_depth: PoseSE3
    base_cloud: PointCloud
    proxy: BilateralBladeProxy

    def __post_init__(self) -> None:
        if (
            self.base_t_left_ir.parent_frame != "base"
            or self.base_t_left_ir.child_frame != "left_ir"
        ):
            raise ValueError("Initial camera pose must be base_T_left_ir")
        if self.base_t_depth.parent_frame != "base" or self.base_t_depth.child_frame != "depth":
            raise ValueError("Initial depth pose must be base_T_depth")
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
        base_t_left_ir=base_t_left_ir,
        base_t_depth=base_t_depth,
        base_cloud=base_cloud,
        proxy=proxy,
    )
