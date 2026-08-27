"""Static-scene validation of native D435i clouds in the ES68 base frame.

Primary metrics are symmetric projective depth residuals computed from the unmodified
robot/hand-eye poses.  The optional ICP result is diagnostic evidence only and is never
fed back into those metrics or into the exported overlay cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    NativeOverlapValidationConfig,
    PointCloudConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.fusion import estimate_normals
from biblade_fusion.perception.pointcloud import (
    PointCloud,
    native_depth_to_meters,
    realsense_project_points_to_pixels,
)
from biblade_fusion.workflows.reconstruction import reconstruct_native_depth_view


class NativeOverlapValidationError(ValueError):
    """Static native-depth observations cannot form auditable overlap evidence."""


def _readonly(value: ArrayLike, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class IcpCorrectionDiagnostic:
    correction_matrix: NDArray[np.float64]
    correspondence_count: int
    point_to_plane_rmse_before_m: float | None
    point_to_plane_rmse_after_m: float | None
    translation_correction_m: float | None
    rotation_correction_deg: float | None
    converged: bool
    reason: str

    def __post_init__(self) -> None:
        matrix = np.array(self.correction_matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("ICP diagnostic correction must be a finite 4x4 matrix")
        matrix.setflags(write=False)
        object.__setattr__(self, "correction_matrix", matrix)
        if self.correspondence_count < 0 or not self.reason:
            raise ValueError("ICP diagnostic identity is invalid")
        for value in (
            self.point_to_plane_rmse_before_m,
            self.point_to_plane_rmse_after_m,
            self.translation_correction_m,
            self.rotation_correction_deg,
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError("ICP diagnostic metrics must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NativeOverlapPairMetrics:
    reference_projected_count: int
    comparison_projected_count: int
    reference_surface_inlier_count: int
    comparison_surface_inlier_count: int
    surface_inlier_fraction: float
    signed_mean_error_m: float
    signed_median_error_m: float
    mean_absolute_error_m: float
    median_absolute_error_m: float
    root_mean_square_error_m: float
    p95_absolute_error_m: float
    agreement_fractions: tuple[tuple[float, float], ...]
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        counts = (
            self.reference_projected_count,
            self.comparison_projected_count,
            self.reference_surface_inlier_count,
            self.comparison_surface_inlier_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("Native-overlap counts must be non-negative")
        if not 0.0 <= self.surface_inlier_fraction <= 1.0:
            raise ValueError("Native-overlap inlier fraction must be in [0, 1]")
        values = (
            self.signed_mean_error_m,
            self.signed_median_error_m,
            self.mean_absolute_error_m,
            self.median_absolute_error_m,
            self.root_mean_square_error_m,
            self.p95_absolute_error_m,
        )
        if not np.isfinite(values).all():
            raise ValueError("Native-overlap error metrics must be finite")
        if any(value < 0.0 for value in values[2:]):
            raise ValueError("Native-overlap absolute metrics must be non-negative")
        if any(
            threshold <= 0.0 or not 0.0 <= fraction <= 1.0
            for threshold, fraction in self.agreement_fractions
        ):
            raise ValueError("Native-overlap agreement fractions are invalid")
        if self.passed == bool(self.failure_reasons):
            raise ValueError("Native-overlap pass state and reasons disagree")


@dataclass(frozen=True, slots=True)
class NativeOverlapPairResult:
    reference_view_id: str
    comparison_view_id: str
    symmetric_signed_residuals_m: NDArray[np.float64]
    metrics: NativeOverlapPairMetrics
    icp_diagnostic: IcpCorrectionDiagnostic | None

    def __post_init__(self) -> None:
        if not self.reference_view_id or not self.comparison_view_id:
            raise ValueError("Native-overlap pair view IDs must be non-empty")
        if self.reference_view_id == self.comparison_view_id:
            raise ValueError("Native-overlap comparison view must differ from reference")
        residuals = np.array(self.symmetric_signed_residuals_m, dtype=np.float64, copy=True)
        if residuals.ndim != 1 or len(residuals) < 1 or not np.isfinite(residuals).all():
            raise ValueError("Native-overlap residuals must be a nonempty finite vector")
        residuals.setflags(write=False)
        object.__setattr__(self, "symmetric_signed_residuals_m", residuals)


@dataclass(frozen=True, slots=True)
class NativeOverlapReport:
    reference_view_id: str
    view_ids: tuple[str, ...]
    pairs: tuple[NativeOverlapPairResult, ...]
    translation_span_m: float
    rotation_span_deg: float
    overlay_points_m: NDArray[np.float64]
    overlay_view_indices: NDArray[np.uint16]
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.view_ids) < 2 or self.view_ids[0] != self.reference_view_id:
            raise ValueError("Native-overlap report must place its reference view first")
        if len(set(self.view_ids)) != len(self.view_ids):
            raise ValueError("Native-overlap report view IDs must be unique")
        if len(self.pairs) != len(self.view_ids) - 1:
            raise ValueError("Native-overlap report requires one pair per comparison view")
        if tuple(pair.comparison_view_id for pair in self.pairs) != self.view_ids[1:]:
            raise ValueError("Native-overlap pair order does not match report view order")
        if not np.isfinite((self.translation_span_m, self.rotation_span_deg)).all():
            raise ValueError("Native-overlap pose spans must be finite")
        points = _readonly(self.overlay_points_m, (3,), "Overlay points").astype(
            np.float64, copy=False
        )
        points.setflags(write=False)
        labels = np.array(self.overlay_view_indices, dtype=np.uint16, copy=True)
        if labels.shape != (len(points),) or len(points) < 1:
            raise ValueError("Overlay labels must match nonempty overlay points")
        if int(labels.max()) >= len(self.view_ids):
            raise ValueError("Overlay label references an unknown view")
        labels.setflags(write=False)
        object.__setattr__(self, "overlay_points_m", points)
        object.__setattr__(self, "overlay_view_indices", labels)
        if self.passed == bool(self.failure_reasons):
            raise ValueError("Native-overlap report pass state and reasons disagree")


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    view_id: str
    intrinsics: CameraIntrinsics
    depth_m: NDArray[np.float32]
    smooth_mask: NDArray[np.bool_]
    base_t_depth: PoseSE3
    base_cloud: PointCloud


def _point_cloud_config(
    source: PointCloudConfig, config: NativeOverlapValidationConfig
) -> PointCloudConfig:
    return source.model_copy(
        update={
            "minimum_depth_m": config.minimum_depth_m,
            "maximum_depth_m": config.maximum_depth_m,
            "pixel_stride": config.pixel_stride,
            "minimum_valid_points": min(
                source.minimum_valid_points, config.minimum_projected_points
            ),
        }
    )


def _smooth_depth_mask(
    depth_m: NDArray[np.float32], config: NativeOverlapValidationConfig
) -> NDArray[np.bool_]:
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= config.minimum_depth_m)
        & (depth_m <= config.maximum_depth_m)
    )
    radius = config.edge_window_radius_px
    if radius == 0:
        return valid
    height, width = depth_m.shape
    padded_depth = np.pad(depth_m, radius, constant_values=np.nan)
    local_min = np.full((height, width), np.inf, dtype=np.float32)
    local_max = np.full((height, width), -np.inf, dtype=np.float32)
    all_valid = np.ones((height, width), dtype=np.bool_)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            window = padded_depth[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            window_valid = (
                np.isfinite(window)
                & (window >= config.minimum_depth_m)
                & (window <= config.maximum_depth_m)
            )
            all_valid &= window_valid
            local_min = np.minimum(local_min, np.where(window_valid, window, np.inf))
            local_max = np.maximum(local_max, np.where(window_valid, window, -np.inf))
    return valid & all_valid & (local_max - local_min <= config.maximum_local_depth_range_m)


def _same_calibration(first: _PreparedObservation, second: _PreparedObservation) -> bool:
    a, b = first.intrinsics, second.intrinsics
    return (
        (a.width, a.height, a.distortion_model) == (b.width, b.height, b.distortion_model)
        and len(a.distortion_coefficients) == len(b.distortion_coefficients)
        and np.allclose((a.fx, a.fy, a.cx, a.cy), (b.fx, b.fy, b.cx, b.cy), atol=1e-9)
        and np.allclose(a.distortion_coefficients, b.distortion_coefficients, atol=1e-12)
    )


def _prepare(
    bundle: SynchronizedFrameBundle,
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    config: NativeOverlapValidationConfig,
) -> _PreparedObservation:
    stereo = bundle.stereo
    calibration = stereo.calibration
    if stereo.native_depth is None or calibration.native_depth_scale_m is None:
        raise NativeOverlapValidationError(f"{bundle.view_id} has no native depth")
    if calibration.depth is None or calibration.left_t_depth is None:
        raise NativeOverlapValidationError(f"{bundle.view_id} has no depth-stream calibration")
    depth_m = native_depth_to_meters(stereo.native_depth, calibration.native_depth_scale_m)
    smooth = _smooth_depth_mask(depth_m, config)
    reconstructed = reconstruct_native_depth_view(
        bundle,
        smooth,
        hand_eye,
        point_cloud_config,
    )
    return _PreparedObservation(
        bundle.view_id,
        calibration.depth,
        depth_m,
        smooth,
        reconstructed.base_t_projection_camera,
        reconstructed.base_cloud,
    )


def _transform_points(matrix: NDArray[np.float64], points: NDArray[np.float64]) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _directed_residuals(
    source: _PreparedObservation,
    target: _PreparedObservation,
    config: NativeOverlapValidationConfig,
) -> tuple[NDArray[np.float64], int]:
    target_t_base = target.base_t_depth.inverse().matrix
    camera_points = _transform_points(target_t_base, source.base_cloud.points_m)
    pixels = realsense_project_points_to_pixels(camera_points, target.intrinsics)
    finite = np.isfinite(pixels).all(axis=1) & np.isfinite(camera_points[:, 2])
    finite &= (camera_points[:, 2] >= config.minimum_depth_m) & (
        camera_points[:, 2] <= config.maximum_depth_m
    )
    indices = np.flatnonzero(finite)
    rounded = np.rint(pixels[indices]).astype(np.int64)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target.intrinsics.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target.intrinsics.height)
    )
    indices = indices[inside]
    rounded = rounded[inside]
    if len(indices) == 0:
        raise NativeOverlapValidationError(
            f"{source.view_id} has no projection overlap with {target.view_id}"
        )
    target_valid = target.smooth_mask[rounded[:, 1], rounded[:, 0]]
    indices = indices[target_valid]
    rounded = rounded[target_valid]
    if len(indices) == 0:
        raise NativeOverlapValidationError(
            f"{source.view_id} has no valid depth overlap with {target.view_id}"
        )
    measured = target.depth_m[rounded[:, 1], rounded[:, 0]].astype(np.float64)
    residuals = measured - camera_points[indices, 2]
    projected_count = len(residuals)
    inliers = np.abs(residuals) <= config.maximum_surface_residual_m
    return residuals[inliers], projected_count


def _voxel_centroids(points: NDArray[np.float64], voxel_size_m: float) -> np.ndarray:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centroids = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(centroids, inverse, points)
    centroids /= counts[:, None]
    return centroids


def _subsample(points: NDArray[np.float64], maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]


def _nearest(
    query: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    indices = np.empty(len(query), dtype=np.int64)
    squared = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 128):
        chunk = query[start : start + 128]
        distances = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        local = np.argmin(distances, axis=1)
        indices[start : start + len(chunk)] = local
        squared[start : start + len(chunk)] = distances[np.arange(len(chunk)), local]
    return indices, squared


def _rotation_vector_matrix(vector: NDArray[np.float64]) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    if angle < 1e-12:
        return np.eye(3) + skew
    axis_skew = skew / angle
    return np.eye(3) + np.sin(angle) * axis_skew + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)


def _rotation_angle_deg(rotation: NDArray[np.float64]) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _diagnose_icp(
    source: _PreparedObservation,
    target: _PreparedObservation,
    config: NativeOverlapValidationConfig,
) -> IcpCorrectionDiagnostic:
    if not config.diagnostic_icp_enabled:
        raise AssertionError("disabled ICP diagnostic must not be invoked")
    source_points = _subsample(
        _voxel_centroids(source.base_cloud.points_m, config.diagnostic_icp_voxel_size_m),
        config.diagnostic_icp_maximum_points,
    )
    target_points = _subsample(
        _voxel_centroids(target.base_cloud.points_m, config.diagnostic_icp_voxel_size_m),
        config.diagnostic_icp_maximum_points,
    )
    neighbors = min(config.diagnostic_icp_normal_neighbors, len(target_points) - 1)
    if min(len(source_points), len(target_points)) < max(
        config.diagnostic_icp_minimum_correspondences, neighbors + 1
    ):
        return IcpCorrectionDiagnostic(
            np.eye(4), 0, None, None, None, None, False, "insufficient diagnostic points"
        )
    target_normals = estimate_normals(target_points, neighbors)
    correction = np.eye(4)
    maximum_squared = config.diagnostic_icp_maximum_correspondence_m**2
    before: float | None = None
    count = 0
    for _ in range(config.diagnostic_icp_iterations):
        moved = _transform_points(correction, source_points)
        indices, squared = _nearest(moved, target_points)
        valid = squared <= maximum_squared
        count = int(np.count_nonzero(valid))
        if count < config.diagnostic_icp_minimum_correspondences:
            return IcpCorrectionDiagnostic(
                correction,
                count,
                before,
                None,
                None,
                None,
                False,
                "insufficient ICP correspondences",
            )
        p = moved[valid]
        q = target_points[indices[valid]]
        n = target_normals[indices[valid]]
        residual = np.einsum("ij,ij->i", n, q - p)
        rmse = float(np.sqrt(np.mean(residual**2)))
        if before is None:
            before = rmse
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-6)
        robust = 1.0 / np.sqrt(1.0 + (residual / (2.5 * scale)) ** 2)
        design = np.column_stack((np.cross(p, n), n)) * robust[:, None]
        rhs = residual * robust
        if config.diagnostic_icp_pose_prior_weight > 0.0:
            prior = np.sqrt(config.diagnostic_icp_pose_prior_weight) * np.eye(6)
            design = np.vstack((design, prior))
            rhs = np.concatenate((rhs, np.zeros(6)))
        delta, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        if np.linalg.norm(delta) < 1e-9:
            break
        update = np.eye(4)
        update[:3, :3] = _rotation_vector_matrix(delta[:3])
        update[:3, 3] = delta[3:]
        correction = update @ correction
    moved = _transform_points(correction, source_points)
    indices, squared = _nearest(moved, target_points)
    valid = squared <= maximum_squared
    count = int(np.count_nonzero(valid))
    if count < config.diagnostic_icp_minimum_correspondences:
        return IcpCorrectionDiagnostic(
            correction,
            count,
            before,
            None,
            None,
            None,
            False,
            "ICP overlap vanished",
        )
    residual = np.einsum(
        "ij,ij->i",
        target_normals[indices[valid]],
        target_points[indices[valid]] - moved[valid],
    )
    after = float(np.sqrt(np.mean(residual**2)))
    return IcpCorrectionDiagnostic(
        correction,
        count,
        before,
        after,
        float(np.linalg.norm(correction[:3, 3])),
        _rotation_angle_deg(correction[:3, :3]),
        True,
        "diagnostic only; correction was not applied to primary metrics",
    )


def _pair_result(
    reference: _PreparedObservation,
    comparison: _PreparedObservation,
    config: NativeOverlapValidationConfig,
) -> NativeOverlapPairResult:
    forward, forward_projected = _directed_residuals(reference, comparison, config)
    reverse, reverse_projected = _directed_residuals(comparison, reference, config)
    residuals = np.concatenate((forward, -reverse))
    if len(residuals) < 1:
        raise NativeOverlapValidationError(
            f"{reference.view_id}/{comparison.view_id} has no same-surface residuals"
        )
    absolute = np.abs(residuals)
    projected = forward_projected + reverse_projected
    inlier_fraction = len(residuals) / projected
    agreements = tuple(
        (threshold, float(np.mean(absolute <= threshold)))
        for threshold in config.agreement_thresholds_m
    )
    five_mm = next(
        fraction for threshold, fraction in agreements if np.isclose(threshold, 0.005, atol=1e-12)
    )
    median = float(np.median(absolute))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    p95 = float(np.percentile(absolute, 95))
    reasons: list[str] = []
    if min(forward_projected, reverse_projected) < config.minimum_projected_points:
        reasons.append("projected overlap count is below the configured minimum")
    if inlier_fraction < config.minimum_surface_inlier_fraction:
        reasons.append("same-surface inlier fraction is below the configured minimum")
    if median > config.maximum_median_absolute_error_m:
        reasons.append("median absolute depth residual exceeds the configured maximum")
    if rmse > config.maximum_root_mean_square_error_m:
        reasons.append("depth-residual RMSE exceeds the configured maximum")
    if p95 > config.maximum_p95_absolute_error_m:
        reasons.append("P95 absolute depth residual exceeds the configured maximum")
    if five_mm < config.minimum_five_mm_agreement_fraction:
        reasons.append("5 mm agreement fraction is below the configured minimum")
    metrics = NativeOverlapPairMetrics(
        forward_projected,
        reverse_projected,
        len(forward),
        len(reverse),
        float(inlier_fraction),
        float(np.mean(residuals)),
        float(np.median(residuals)),
        float(np.mean(absolute)),
        median,
        rmse,
        p95,
        agreements,
        not reasons,
        tuple(reasons),
    )
    diagnostic = (
        _diagnose_icp(comparison, reference, config) if config.diagnostic_icp_enabled else None
    )
    return NativeOverlapPairResult(
        reference.view_id,
        comparison.view_id,
        residuals,
        metrics,
        diagnostic,
    )


def _pose_spans(observations: tuple[_PreparedObservation, ...]) -> tuple[float, float]:
    translation_span = 0.0
    rotation_span = 0.0
    for first_index, first in enumerate(observations):
        for second in observations[first_index + 1 :]:
            translation_span = max(
                translation_span,
                float(
                    np.linalg.norm(
                        first.base_t_depth.translation_m - second.base_t_depth.translation_m
                    )
                ),
            )
            relative = first.base_t_depth.rotation.T @ second.base_t_depth.rotation
            rotation_span = max(rotation_span, _rotation_angle_deg(relative))
    return translation_span, rotation_span


def _overlay(
    observations: tuple[_PreparedObservation, ...],
    config: NativeOverlapValidationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for index, observation in enumerate(observations):
        selected = _subsample(
            _voxel_centroids(observation.base_cloud.points_m, config.overlay_voxel_size_m),
            config.maximum_overlay_points_per_view,
        )
        points.append(selected)
        labels.append(np.full(len(selected), index, dtype=np.uint16))
    return np.vstack(points), np.concatenate(labels)


def evaluate_native_overlap(
    bundles: tuple[SynchronizedFrameBundle, ...],
    hand_eye: HandEyeCalibration,
    point_cloud_config: PointCloudConfig,
    config: NativeOverlapValidationConfig,
) -> NativeOverlapReport:
    """Evaluate one reference plus static-scene comparison views without pose correction."""

    if len(bundles) < config.minimum_views:
        raise NativeOverlapValidationError(
            f"native-overlap validation needs at least {config.minimum_views} views"
        )
    view_ids = tuple(bundle.view_id for bundle in bundles)
    if len(set(view_ids)) != len(view_ids):
        raise NativeOverlapValidationError("native-overlap view IDs must be unique")
    processing_cloud = _point_cloud_config(point_cloud_config, config)
    observations = tuple(_prepare(bundle, hand_eye, processing_cloud, config) for bundle in bundles)
    reference = observations[0]
    if any(not _same_calibration(reference, item) for item in observations[1:]):
        raise NativeOverlapValidationError(
            "native-overlap observations use different depth-stream calibrations"
        )
    pairs = tuple(_pair_result(reference, comparison, config) for comparison in observations[1:])
    translation_span, rotation_span = _pose_spans(observations)
    reasons: list[str] = []
    failed_pairs = [pair.comparison_view_id for pair in pairs if not pair.metrics.passed]
    if failed_pairs:
        reasons.append("failed comparison views: " + ", ".join(failed_pairs))
    if translation_span < config.minimum_translation_span_m:
        reasons.append("camera translation span is below the configured minimum")
    if rotation_span < config.minimum_rotation_span_deg:
        reasons.append("camera rotation span is below the configured minimum")
    overlay_points, overlay_labels = _overlay(observations, config)
    return NativeOverlapReport(
        reference.view_id,
        view_ids,
        pairs,
        translation_span,
        rotation_span,
        overlay_points,
        overlay_labels,
        not reasons,
        tuple(reasons),
    )
