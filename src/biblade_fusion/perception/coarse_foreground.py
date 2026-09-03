"""Projection-guided foreground extraction before a schema-5 blade exists.

The first coarse observation remains an operator-authored hard ROI.  Every later
coarse observation is constrained by two independent pieces of that accepted
history: accumulated blade support projected into the current rectified image,
and the configured blade-only envelope in the base frame.  The safety occupancy
mask is never passed through this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapForegroundError,
    array_content_sha256,
)

PROJECTED_COARSE_FOREGROUND_ALGORITHM = (
    "accumulated_blade_projection_depth_band_base_envelope_v2"
)


def _sha256_digest(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _readonly_mask(value: ArrayLike, shape: tuple[int, int], label: str) -> NDArray[np.bool_]:
    mask = np.array(value, dtype=np.bool_, copy=True)
    if mask.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class ProjectedCoarseForegroundGuide:
    """Immutable identity of the accepted coarse generation used as a guide."""

    source_generation_path: Path
    source_generation_metadata_sha256: str
    reference_points_content_sha256: str
    blade_envelope_min_m: tuple[float, float, float]
    blade_envelope_max_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_generation_path",
            Path(self.source_generation_path).resolve(),
        )
        object.__setattr__(
            self,
            "source_generation_metadata_sha256",
            _sha256_digest(
                self.source_generation_metadata_sha256,
                label="source_generation_metadata_sha256",
            ),
        )
        object.__setattr__(
            self,
            "reference_points_content_sha256",
            _sha256_digest(
                self.reference_points_content_sha256,
                label="reference_points_content_sha256",
            ),
        )
        lower = tuple(float(value) for value in self.blade_envelope_min_m)
        upper = tuple(float(value) for value in self.blade_envelope_max_m)
        if (
            len(lower) != 3
            or len(upper) != 3
            or not np.isfinite((lower, upper)).all()
            or any(high <= low for low, high in zip(lower, upper, strict=True))
        ):
            raise ValueError("Projected coarse blade envelope must be finite and ordered")
        object.__setattr__(self, "blade_envelope_min_m", lower)
        object.__setattr__(self, "blade_envelope_max_m", upper)

    def payload(self) -> dict[str, object]:
        return {
            "source_generation_path": str(self.source_generation_path),
            "source_generation_metadata_sha256": self.source_generation_metadata_sha256,
            "reference_points_content_sha256": self.reference_points_content_sha256,
            "blade_envelope_min_m": list(self.blade_envelope_min_m),
            "blade_envelope_max_m": list(self.blade_envelope_max_m),
        }


@dataclass(frozen=True, slots=True)
class ProjectedCoarseForegroundDiagnostics:
    image_pixel_count: int
    supplied_valid_pixel_count: int
    depth_valid_pixel_count: int
    reference_point_count: int
    projected_reference_pixel_count: int
    eligible_projected_pixel_count: int
    predicted_depth_consistent_pixel_count: int
    base_envelope_pixel_count: int
    mask_pixel_count: int
    mask_fraction: float
    projected_match_fraction: float
    minimum_mask_depth_m: float
    median_mask_depth_m: float
    maximum_mask_depth_m: float


def projected_coarse_foreground_policy_sha256(
    config: BootstrapForegroundConfig,
) -> str:
    payload = {
        "algorithm": PROJECTED_COARSE_FOREGROUND_ALGORITHM,
        "configuration": asdict(config),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectedCoarseForegroundResult:
    """A blade-only mask tied to one prior accepted coarse generation."""

    mask: NDArray[np.bool_]
    projected_reference_mask: NDArray[np.bool_]
    diagnostics: ProjectedCoarseForegroundDiagnostics
    config: BootstrapForegroundConfig
    guide: ProjectedCoarseForegroundGuide
    algorithm: str
    policy_sha256: str
    left_image_content_sha256: str
    depth_content_sha256: str
    valid_mask_content_sha256: str

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in np.asarray(self.mask).shape)
        if len(shape) != 2:
            raise ValueError("Projected coarse mask must be two-dimensional")
        mask = _readonly_mask(self.mask, shape, "mask")
        projection = _readonly_mask(
            self.projected_reference_mask,
            shape,
            "projected_reference_mask",
        )
        if np.any(mask & ~projection):
            raise ValueError("Projected coarse mask escapes its reference projection")
        if self.algorithm != PROJECTED_COARSE_FOREGROUND_ALGORITHM:
            raise ValueError("Unsupported projected coarse foreground algorithm")
        if self.policy_sha256 != projected_coarse_foreground_policy_sha256(self.config):
            raise ValueError("Projected coarse foreground policy hash changed")
        for label, digest in (
            ("left_image_content_sha256", self.left_image_content_sha256),
            ("depth_content_sha256", self.depth_content_sha256),
            ("valid_mask_content_sha256", self.valid_mask_content_sha256),
        ):
            _sha256_digest(digest, label=label)
        diagnostics = self.diagnostics
        if (
            diagnostics.image_pixel_count != mask.size
            or diagnostics.projected_reference_pixel_count != int(np.count_nonzero(projection))
            or diagnostics.mask_pixel_count != int(np.count_nonzero(mask))
            or diagnostics.mask_fraction
            != diagnostics.mask_pixel_count / diagnostics.image_pixel_count
        ):
            raise ValueError("Projected coarse foreground diagnostics do not reproduce")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "projected_reference_mask", projection)

    @property
    def seed(self) -> None:
        """Compatibility with the operator-bootstrap foreground result."""

        return None

    @property
    def seed_mask(self) -> NDArray[np.bool_]:
        """Compatibility name used by the coarse-view evidence container."""

        return self.projected_reference_mask


def projected_coarse_blade_foreground(
    left_rectified: ArrayLike,
    depth_m: ArrayLike,
    valid_mask: ArrayLike,
    config: BootstrapForegroundConfig,
    *,
    intrinsics: CameraIntrinsics,
    base_t_left_rectified: PoseSE3,
    reference_points_base_m: ArrayLike,
    guide: ProjectedCoarseForegroundGuide,
) -> ProjectedCoarseForegroundResult:
    """Propagate accepted blade support into a new coarse observation.

    The projected support is deliberately dilated so newly exposed fin pixels and
    small stereo holes are not clipped.  The final decision is nevertheless made
    in metric base coordinates and cannot escape the blade-only envelope.
    """

    left = np.asarray(left_rectified)
    original_depth = np.asarray(depth_m)
    depth = np.asarray(depth_m, dtype=np.float64)
    supplied_valid = np.asarray(valid_mask, dtype=np.bool_)
    expected_shape = (intrinsics.height, intrinsics.width)
    if (
        left.shape != expected_shape
        or depth.shape != expected_shape
        or supplied_valid.shape != expected_shape
    ):
        raise BootstrapForegroundError(
            "Projected coarse inputs must match left-rectified intrinsics"
        )
    if left.ndim != 2 or not np.issubdtype(left.dtype, np.number) or not np.isfinite(left).all():
        raise BootstrapForegroundError("Projected coarse left image is invalid")
    if intrinsics.distortion_model != "none" or intrinsics.distortion_coefficients:
        raise BootstrapForegroundError("Projected coarse guidance requires rectified intrinsics")
    if (
        base_t_left_rectified.parent_frame != "base"
        or base_t_left_rectified.child_frame != "left_rectified"
    ):
        raise BootstrapForegroundError(
            "Projected coarse guidance requires base_T_left_rectified"
        )
    reference = np.asarray(reference_points_base_m, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) < config.minimum_projected_reference_points
        or not np.isfinite(reference).all()
    ):
        raise BootstrapForegroundError(
            "Accepted coarse reference has insufficient finite blade points"
        )
    if array_content_sha256(reference) != guide.reference_points_content_sha256:
        raise BootstrapForegroundError("Accepted coarse reference point identity changed")

    depth_valid = (
        supplied_valid
        & np.isfinite(depth)
        & (depth >= config.minimum_depth_m)
        & (depth <= config.maximum_depth_m)
    )
    depth_valid_count = int(np.count_nonzero(depth_valid))
    if depth_valid_count < config.minimum_valid_pixels:
        raise BootstrapForegroundError(
            "Projected coarse depth-valid support is below minimum_valid_pixels"
        )

    camera_points = base_t_left_rectified.inverse().transform_points(reference)
    camera_z = camera_points[:, 2]
    projectable = (
        np.isfinite(camera_points).all(axis=1)
        & (camera_z >= config.minimum_depth_m)
        & (camera_z <= config.maximum_depth_m)
    )
    camera_points = camera_points[projectable]
    camera_z = camera_z[projectable]
    if len(camera_points) < config.minimum_projected_reference_points:
        raise BootstrapForegroundError(
            "Too few accepted blade points project into the current depth range"
        )
    pixels_u = np.rint(
        intrinsics.fx * camera_points[:, 0] / camera_z + intrinsics.cx
    ).astype(np.int64)
    pixels_v = np.rint(
        intrinsics.fy * camera_points[:, 1] / camera_z + intrinsics.cy
    ).astype(np.int64)
    inside_image = (
        (pixels_u >= 0)
        & (pixels_u < intrinsics.width)
        & (pixels_v >= 0)
        & (pixels_v < intrinsics.height)
    )
    projected = np.zeros(expected_shape, dtype=np.uint8)
    projected_u = pixels_u[inside_image]
    projected_v = pixels_v[inside_image]
    projected_z = camera_z[inside_image]
    projected[projected_v, projected_u] = 1
    radius = config.projected_reference_dilation_px
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    projected_reference = cv2.dilate(projected, kernel).astype(np.bool_)
    projected_count = int(np.count_nonzero(projected_reference))
    if projected_count < config.minimum_projected_reference_pixels:
        raise BootstrapForegroundError(
            "Projected accepted-blade support is below its pixel minimum"
        )

    # A dilated 2-D projection alone can include a fixture or table region that
    # happens to lie inside the conservative blade AABB.  Carry the projected
    # reference depth through the same neighbourhood so each current pixel must
    # also agree with a locally predicted surface band.  The configured depth
    # jump remains deliberately loose enough for newly exposed thin fins.
    finite_fill = float(config.maximum_depth_m + config.maximum_neighbour_depth_jump_m + 1.0)
    projected_depth_min = np.full(expected_shape, finite_fill, dtype=np.float32)
    projected_depth_max = np.full(expected_shape, -finite_fill, dtype=np.float32)
    np.minimum.at(
        projected_depth_min,
        (projected_v, projected_u),
        projected_z.astype(np.float32),
    )
    np.maximum.at(
        projected_depth_max,
        (projected_v, projected_u),
        projected_z.astype(np.float32),
    )
    local_depth_min = cv2.erode(projected_depth_min, kernel)
    local_depth_max = cv2.dilate(projected_depth_max, kernel)
    depth_tolerance = float(config.maximum_neighbour_depth_jump_m)
    predicted_depth_consistent = (
        projected_reference
        & (local_depth_min < finite_fill)
        & (local_depth_max > -finite_fill)
        & (depth >= local_depth_min - depth_tolerance)
        & (depth <= local_depth_max + depth_tolerance)
    )

    eligible_projected = depth_valid & projected_reference
    rows, columns = np.nonzero(eligible_projected)
    eligible_projected_count = len(rows)
    if eligible_projected_count < config.minimum_mask_pixels:
        raise BootstrapForegroundError(
            "Projected accepted-blade region has insufficient eligible depth"
        )
    z_m = depth[rows, columns]
    points_camera = np.column_stack(
        (
            (columns - intrinsics.cx) * z_m / intrinsics.fx,
            (rows - intrinsics.cy) * z_m / intrinsics.fy,
            z_m,
        )
    )
    points_base = base_t_left_rectified.transform_points(points_camera)
    lower = np.asarray(guide.blade_envelope_min_m, dtype=np.float64)
    upper = np.asarray(guide.blade_envelope_max_m, dtype=np.float64)
    inside_envelope = np.all((points_base >= lower) & (points_base <= upper), axis=1)
    inside_predicted_depth = predicted_depth_consistent[rows, columns]
    selected = inside_envelope & inside_predicted_depth
    mask = np.zeros(expected_shape, dtype=np.bool_)
    mask[rows[selected], columns[selected]] = True
    mask_count = int(np.count_nonzero(mask))
    mask_fraction = mask_count / mask.size
    projected_match_fraction = mask_count / eligible_projected_count
    if mask_count < config.minimum_mask_pixels:
        raise BootstrapForegroundError("Projected coarse blade mask is below minimum_mask_pixels")
    if not config.minimum_mask_fraction <= mask_fraction <= config.maximum_mask_fraction:
        raise BootstrapForegroundError(
            "Projected coarse blade mask fraction is outside configured bounds"
        )
    if projected_match_fraction < config.minimum_projected_match_fraction:
        raise BootstrapForegroundError(
            "Projected coarse blade support disagrees with the base-frame envelope"
        )

    mask_depth = depth[mask]
    diagnostics = ProjectedCoarseForegroundDiagnostics(
        image_pixel_count=int(mask.size),
        supplied_valid_pixel_count=int(np.count_nonzero(supplied_valid)),
        depth_valid_pixel_count=depth_valid_count,
        reference_point_count=len(reference),
        projected_reference_pixel_count=projected_count,
        eligible_projected_pixel_count=eligible_projected_count,
        predicted_depth_consistent_pixel_count=int(
            np.count_nonzero(eligible_projected & predicted_depth_consistent)
        ),
        base_envelope_pixel_count=int(np.count_nonzero(inside_envelope)),
        mask_pixel_count=mask_count,
        mask_fraction=mask_fraction,
        projected_match_fraction=projected_match_fraction,
        minimum_mask_depth_m=float(np.min(mask_depth)),
        median_mask_depth_m=float(np.median(mask_depth)),
        maximum_mask_depth_m=float(np.max(mask_depth)),
    )
    return ProjectedCoarseForegroundResult(
        mask=mask,
        projected_reference_mask=projected_reference,
        diagnostics=diagnostics,
        config=config,
        guide=guide,
        algorithm=PROJECTED_COARSE_FOREGROUND_ALGORITHM,
        policy_sha256=projected_coarse_foreground_policy_sha256(config),
        left_image_content_sha256=array_content_sha256(left),
        depth_content_sha256=array_content_sha256(original_depth),
        valid_mask_content_sha256=array_content_sha256(supplied_valid),
    )


__all__ = [
    "PROJECTED_COARSE_FOREGROUND_ALGORITHM",
    "ProjectedCoarseForegroundDiagnostics",
    "ProjectedCoarseForegroundGuide",
    "ProjectedCoarseForegroundResult",
    "projected_coarse_blade_foreground",
    "projected_coarse_foreground_policy_sha256",
]
