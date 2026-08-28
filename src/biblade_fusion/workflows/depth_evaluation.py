"""Calibrated comparison of native D435i and FoundationStereo blade depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.settings import (
    DepthComparisonConfig,
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
)
from biblade_fusion.perception.pointcloud import (
    native_depth_to_meters,
    point_cloud_to_depth_image,
    realsense_depth_image_to_point_cloud,
)
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import BladeSide
from biblade_fusion.workflows.reconstruction import resolve_authoritative_robot_pose
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


class DepthComparisonError(ValueError):
    """Paired depth sources cannot be compared in a shared calibrated frame."""


@dataclass(frozen=True, slots=True)
class DepthComparisonMetrics:
    blade_pixel_count: int
    native_valid_pixel_count: int
    stereo_valid_pixel_count: int
    overlap_pixel_count: int
    native_coverage_fraction: float
    stereo_coverage_fraction: float
    overlap_fraction: float
    signed_mean_error_m: float
    signed_median_error_m: float
    mean_absolute_error_m: float
    root_mean_square_error_m: float
    p95_absolute_error_m: float
    median_stereo_to_native_ratio: float
    agreement_fractions: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class DepthViewGeometry:
    side: BladeSide
    camera_side_offset_m: float
    incidence_cosine: float
    incidence_angle_deg: float


@dataclass(frozen=True, slots=True)
class PairedDepthComparison:
    source_view_id: str
    source_sequence_index: int
    source_frame_number: int
    native_depth_left_rectified_m: NDArray[np.float32]
    comparison_mask: NDArray[np.bool_]
    signed_error_m: NDArray[np.float32]
    metrics: DepthComparisonMetrics

    def __post_init__(self) -> None:
        native = np.array(self.native_depth_left_rectified_m, dtype=np.float32, copy=True)
        mask = np.array(self.comparison_mask, dtype=np.bool_, copy=True)
        error = np.array(self.signed_error_m, dtype=np.float32, copy=True)
        if native.shape != mask.shape or error.shape != mask.shape:
            raise ValueError("Paired depth comparison arrays must have equal shapes")
        if np.isfinite(error[~mask]).any():
            raise ValueError("Depth error must be NaN outside the comparison mask")
        native.setflags(write=False)
        mask.setflags(write=False)
        error.setflags(write=False)
        object.__setattr__(self, "native_depth_left_rectified_m", native)
        object.__setattr__(self, "comparison_mask", mask)
        object.__setattr__(self, "signed_error_m", error)


def classify_depth_view_geometry(
    bundle: SynchronizedFrameBundle,
    stereo: StereoInferenceObservation,
    proxy: BilateralBladeProxy,
    hand_eye: HandEyeCalibration,
    minimum_camera_side_offset_m: float,
    *,
    kinematics_config: KinematicsConfig,
    hand_eye_config: HandEyeConfig,
) -> DepthViewGeometry:
    """Classify achieved rectified-camera pose against the fixed bilateral proxy."""

    if (
        stereo.source_view_id != bundle.view_id
        or stereo.source_sequence_index != bundle.sequence_index
        or stereo.rectified.source_frame_number != bundle.stereo.frame_number
    ):
        raise DepthComparisonError("Stereo inference artifact does not match the stored view")
    if minimum_camera_side_offset_m <= 0.0:
        raise DepthComparisonError("Minimum camera side offset must be positive")
    try:
        base_t_left_ir, _ = resolve_authoritative_robot_pose(
            bundle,
            hand_eye,
            kinematics_config,
            hand_eye_config,
        )
    except ValueError as exc:
        raise DepthComparisonError(
            f"Cannot classify view without authoritative ES68 FK: {exc}"
        ) from exc
    base_t_left_rectified = base_t_left_ir.compose(
        stereo.rectified.calibration.left_rectified_t_left_ir.inverse()
    )
    proxy_t_base = proxy.frame_T_proxy.inverse()
    camera_local = proxy_t_base.transform_points(base_t_left_rectified.translation_m)
    offset = float(camera_local[2])
    if abs(offset) < minimum_camera_side_offset_m:
        raise DepthComparisonError(
            "Camera is too close to the proxy mid-plane to classify blade side"
        )
    side = BladeSide.FRONT if offset > 0.0 else BladeSide.BACK
    outward = proxy.outward_normal * (1.0 if side is BladeSide.FRONT else -1.0)
    optical_axis = base_t_left_rectified.rotation[:, 2]
    incidence_cosine = float(np.clip((-optical_axis) @ outward, -1.0, 1.0))
    incidence_angle_deg = float(np.degrees(np.arccos(incidence_cosine)))
    if incidence_angle_deg > 90.0:
        raise DepthComparisonError("Camera optical axis points away from the blade face")
    return DepthViewGeometry(side, offset, incidence_cosine, incidence_angle_deg)


def compare_paired_depth(
    bundle: SynchronizedFrameBundle,
    stereo: StereoInferenceObservation,
    blade_mask: ArrayLike,
    point_cloud_config: PointCloudConfig,
    comparison_config: DepthComparisonConfig,
) -> PairedDepthComparison:
    """Compare both depth sources on shared blade pixels; native depth is not ground truth."""

    frame = bundle.stereo
    calibration = frame.calibration
    if (
        stereo.source_view_id != bundle.view_id
        or stereo.source_sequence_index != bundle.sequence_index
        or stereo.rectified.source_frame_number != frame.frame_number
    ):
        raise DepthComparisonError("Stereo inference artifact does not match the stored view")
    if frame.native_depth is None or calibration.native_depth_scale_m is None:
        raise DepthComparisonError("Stored view has no native RealSense depth")
    if calibration.depth is None or calibration.left_t_depth is None:
        raise DepthComparisonError("Stored view has no depth-stream calibration")
    mask = np.asarray(blade_mask, dtype=np.bool_)
    if mask.shape != stereo.depth_m.shape:
        raise DepthComparisonError("Blade mask must match left-rectified stereo depth")
    blade_pixels = int(np.count_nonzero(mask))
    if blade_pixels < comparison_config.minimum_overlap_points:
        raise DepthComparisonError("Blade mask has too few pixels for depth comparison")

    native_m = native_depth_to_meters(
        frame.native_depth,
        calibration.native_depth_scale_m,
    )
    native_cloud = realsense_depth_image_to_point_cloud(
        native_m,
        calibration.depth,
        point_cloud_config,
        frame="depth",
    )
    left_rectified_t_depth = (
        stereo.rectified.calibration.left_rectified_t_left_ir.compose(
            calibration.left_t_depth
        )
    )
    native_rectified = point_cloud_to_depth_image(
        native_cloud.transformed(left_rectified_t_depth),
        stereo.rectified.calibration.left,
    )
    native_valid = mask & np.isfinite(native_rectified)
    stereo_valid = mask & stereo.result.valid_mask & np.isfinite(stereo.depth_m)
    overlap = native_valid & stereo_valid
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count < comparison_config.minimum_overlap_points:
        raise DepthComparisonError(
            f"Depth sources overlap at {overlap_count} blade pixels; at least "
            f"{comparison_config.minimum_overlap_points} are required"
        )

    differences = stereo.depth_m[overlap].astype(np.float64) - native_rectified[
        overlap
    ].astype(np.float64)
    absolute = np.abs(differences)
    ratios = stereo.depth_m[overlap].astype(np.float64) / native_rectified[
        overlap
    ].astype(np.float64)
    signed_error = np.full(mask.shape, np.nan, dtype=np.float32)
    signed_error[overlap] = differences.astype(np.float32)
    metrics = DepthComparisonMetrics(
        blade_pixel_count=blade_pixels,
        native_valid_pixel_count=int(np.count_nonzero(native_valid)),
        stereo_valid_pixel_count=int(np.count_nonzero(stereo_valid)),
        overlap_pixel_count=overlap_count,
        native_coverage_fraction=float(np.count_nonzero(native_valid) / blade_pixels),
        stereo_coverage_fraction=float(np.count_nonzero(stereo_valid) / blade_pixels),
        overlap_fraction=float(overlap_count / blade_pixels),
        signed_mean_error_m=float(np.mean(differences)),
        signed_median_error_m=float(np.median(differences)),
        mean_absolute_error_m=float(np.mean(absolute)),
        root_mean_square_error_m=float(np.sqrt(np.mean(np.square(differences)))),
        p95_absolute_error_m=float(np.percentile(absolute, 95)),
        median_stereo_to_native_ratio=float(np.median(ratios)),
        agreement_fractions=tuple(
            (threshold, float(np.mean(absolute <= threshold)))
            for threshold in comparison_config.agreement_thresholds_m
        ),
    )
    return PairedDepthComparison(
        bundle.view_id,
        bundle.sequence_index,
        frame.frame_number,
        native_rectified,
        overlap,
        signed_error,
        metrics,
    )
