"""Reference-guided blade foreground extraction for scientific depth products.

The safety occupancy mapper must retain every eligible scene depth.  Scientific
reconstruction has a different contract: once a schema-5 coarse blade model is
pinned, only depths geometrically consistent with that immutable reference may be
labelled as blade foreground.  This module implements that second contract without
connected-component filtering or erosion, preserving narrow fins and edge pixels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import BladeForegroundConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.surface import CurvedBladeSurface

REFERENCE_PROJECTED_ALGORITHM = "reference_projected_visible_owner_v2"
_Z_BUFFER_OWNERSHIP_ATOL_M = 1e-9


class BladeForegroundMaskError(ValueError):
    """The current observation cannot support a trustworthy blade mask."""


@dataclass(frozen=True, slots=True)
class BladeForegroundDiagnostics:
    """Scalar evidence used by fail-closed mask gates and persisted audit records."""

    target_patch_id: str
    target_incidence_cosine: float
    image_pixel_count: int
    eligible_pixel_count: int
    valid_eligible_depth_pixel_count: int
    reference_pixel_count: int
    eligible_reference_pixel_count: int
    target_reference_pixel_count: int
    eligible_target_reference_pixel_count: int
    mask_pixel_count: int
    target_mask_pixel_count: int
    mask_fraction: float
    reference_match_fraction: float
    target_match_fraction: float


def _readonly_array(
    value: ArrayLike,
    *,
    dtype: np.dtype[np.generic] | type[np.generic],
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BladeForegroundMaskResult:
    """Immutable mask plus its projected-depth evidence and policy identity."""

    mask: NDArray[np.bool_]
    reference_depth_m: NDArray[np.float64]
    target_reference_depth_m: NDArray[np.float64]
    eligible_mask: NDArray[np.bool_]
    diagnostics: BladeForegroundDiagnostics
    config: BladeForegroundConfig
    algorithm: str
    policy_sha256: str

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("Blade foreground mask must be two-dimensional")
        shape = (int(mask.shape[0]), int(mask.shape[1]))
        immutable_mask = _readonly_array(mask, dtype=np.bool_, shape=shape, name="mask")
        reference = _readonly_array(
            self.reference_depth_m,
            dtype=np.float64,
            shape=shape,
            name="reference_depth_m",
        )
        target_reference = _readonly_array(
            self.target_reference_depth_m,
            dtype=np.float64,
            shape=shape,
            name="target_reference_depth_m",
        )
        eligible = _readonly_array(
            self.eligible_mask,
            dtype=np.bool_,
            shape=shape,
            name="eligible_mask",
        )
        if np.isinf(reference).any() or np.isinf(target_reference).any():
            raise ValueError("Reference depths may contain finite values or NaN, not infinity")
        if np.any(np.isfinite(reference) & (reference <= 0.0)) or np.any(
            np.isfinite(target_reference) & (target_reference <= 0.0)
        ):
            raise ValueError("Finite reference depths must be positive")
        if np.any(immutable_mask & ~eligible):
            raise ValueError("Blade foreground mask must be a subset of eligible_mask")
        if not isinstance(self.diagnostics, BladeForegroundDiagnostics):
            raise TypeError("diagnostics must be BladeForegroundDiagnostics")
        if not isinstance(self.config, BladeForegroundConfig):
            raise TypeError("config must be BladeForegroundConfig")
        if self.algorithm != REFERENCE_PROJECTED_ALGORITHM:
            raise ValueError("Unsupported blade-foreground algorithm identity")
        if len(self.policy_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.policy_sha256
        ):
            raise ValueError("Blade-foreground policy SHA-256 is malformed")
        if self.policy_sha256 != _policy_sha256(self.config):
            raise ValueError("Blade-foreground policy SHA-256 does not match config")
        _validate_diagnostics_and_policy(
            self.diagnostics,
            config=self.config,
            mask=immutable_mask,
            reference_depth_m=reference,
            target_reference_depth_m=target_reference,
            eligible_mask=eligible,
        )
        object.__setattr__(self, "mask", immutable_mask)
        object.__setattr__(self, "reference_depth_m", reference)
        object.__setattr__(self, "target_reference_depth_m", target_reference)
        object.__setattr__(self, "eligible_mask", eligible)


def _validate_diagnostics_and_policy(
    diagnostics: BladeForegroundDiagnostics,
    *,
    config: BladeForegroundConfig,
    mask: NDArray[np.bool_],
    reference_depth_m: NDArray[np.float64],
    target_reference_depth_m: NDArray[np.float64],
    eligible_mask: NDArray[np.bool_],
) -> None:
    image_pixels = int(mask.size)
    eligible_pixels = int(np.count_nonzero(eligible_mask))
    reference_support = np.isfinite(reference_depth_m)
    target_support = np.isfinite(target_reference_depth_m)
    eligible_reference = int(np.count_nonzero(eligible_mask & reference_support))
    eligible_target = int(np.count_nonzero(eligible_mask & target_support))
    mask_pixels = int(np.count_nonzero(mask))
    expected_counts = {
        "image_pixel_count": image_pixels,
        "eligible_pixel_count": eligible_pixels,
        "reference_pixel_count": int(np.count_nonzero(reference_support)),
        "eligible_reference_pixel_count": eligible_reference,
        "target_reference_pixel_count": int(np.count_nonzero(target_support)),
        "eligible_target_reference_pixel_count": eligible_target,
        "mask_pixel_count": mask_pixels,
    }
    if any(
        int(getattr(diagnostics, name)) != expected for name, expected in expected_counts.items()
    ):
        raise ValueError("Blade-foreground diagnostics do not match their arrays")
    if not diagnostics.target_patch_id:
        raise ValueError("Blade-foreground target patch identity must be non-empty")
    if (
        not np.isfinite(diagnostics.target_incidence_cosine)
        or diagnostics.target_incidence_cosine < config.minimum_target_incidence_cosine
        or diagnostics.target_incidence_cosine > 1.0 + 1e-12
    ):
        raise ValueError("Blade-foreground target incidence violates its policy")
    if (
        not 0 <= diagnostics.valid_eligible_depth_pixel_count <= eligible_pixels
        or mask_pixels > diagnostics.valid_eligible_depth_pixel_count
        or not 0 <= diagnostics.target_mask_pixel_count <= min(mask_pixels, eligible_target)
    ):
        raise ValueError("Blade-foreground diagnostic support is outside its bounds")
    if eligible_reference <= 0 or eligible_target <= 0 or image_pixels <= 0:
        raise ValueError("Blade-foreground eligible reference support must be non-empty")
    expected_ratios = (
        mask_pixels / image_pixels,
        mask_pixels / eligible_reference,
        diagnostics.target_mask_pixel_count / eligible_target,
    )
    actual_ratios = (
        diagnostics.mask_fraction,
        diagnostics.reference_match_fraction,
        diagnostics.target_match_fraction,
    )
    if any(
        not np.isfinite(actual) or actual != expected
        for actual, expected in zip(actual_ratios, expected_ratios, strict=True)
    ):
        raise ValueError("Blade-foreground diagnostic ratios do not reproduce")
    if (
        diagnostics.reference_pixel_count < config.minimum_reference_pixels
        or diagnostics.eligible_reference_pixel_count < config.minimum_reference_pixels
        or diagnostics.target_reference_pixel_count < config.minimum_target_reference_pixels
        or diagnostics.eligible_target_reference_pixel_count
        < config.minimum_target_reference_pixels
        or mask_pixels < config.minimum_mask_pixels
        or diagnostics.target_mask_pixel_count < config.minimum_target_mask_pixels
        or diagnostics.reference_match_fraction < config.minimum_reference_match_fraction
        or diagnostics.target_match_fraction < config.minimum_target_match_fraction
        or not config.minimum_mask_fraction
        <= diagnostics.mask_fraction
        <= config.maximum_mask_fraction
    ):
        raise ValueError("Blade-foreground result violates its fail-closed policy")


def _policy_sha256(config: BladeForegroundConfig) -> str:
    payload = {
        "algorithm": REFERENCE_PROJECTED_ALGORITHM,
        "configuration": config.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _projected_z_buffer(
    points_base_m: NDArray[np.float64],
    *,
    left_rectified_t_base: PoseSE3,
    intrinsics: CameraIntrinsics,
    config: BladeForegroundConfig,
) -> NDArray[np.float64]:
    camera_points = left_rectified_t_base.transform_points(points_base_m)
    z_m = camera_points[:, 2]
    valid = (
        np.isfinite(camera_points).all(axis=1)
        & (z_m >= config.minimum_projection_depth_m)
        & (z_m <= config.maximum_projection_depth_m)
    )
    camera_points = camera_points[valid]
    z_m = z_m[valid]
    output = np.full((intrinsics.height, intrinsics.width), np.inf, dtype=np.float64)
    if len(camera_points) == 0:
        output.fill(np.nan)
        return output

    centers_u = np.rint(intrinsics.fx * camera_points[:, 0] / z_m + intrinsics.cx).astype(np.int64)
    centers_v = np.rint(intrinsics.fy * camera_points[:, 1] / z_m + intrinsics.cy).astype(np.int64)
    radius = config.projection_radius_px
    offsets = tuple(
        (du, dv)
        for dv in range(-radius, radius + 1)
        for du in range(-radius, radius + 1)
        if du * du + dv * dv <= radius * radius
    )
    for du, dv in offsets:
        pixels_u = centers_u + du
        pixels_v = centers_v + dv
        inside = (
            (pixels_u >= 0)
            & (pixels_u < intrinsics.width)
            & (pixels_v >= 0)
            & (pixels_v < intrinsics.height)
        )
        np.minimum.at(output, (pixels_v[inside], pixels_u[inside]), z_m[inside])
    output[~np.isfinite(output)] = np.nan
    return output


def reference_guided_blade_mask(
    depth_m: NDArray[np.float32] | NDArray[np.float64],
    eligible_mask: NDArray[np.bool_],
    intrinsics: CameraIntrinsics,
    base_t_left_rectified: PoseSE3,
    surface: CurvedBladeSurface,
    target_patch_id: str,
    config: BladeForegroundConfig,
) -> BladeForegroundMaskResult:
    """Extract a fine-scan blade mask using the fixed base-frame coarse surface.

    A nearest-depth point-splat z-buffer is formed from every surface sample.  A measurement
    is accepted only when it is eligible and lies within the configured asymmetric
    front/back interval around the prediction.  The selected target patch has
    independent projected-support and matched-depth gates so a nominal candidate
    cannot be accepted when its intended patch is actually absent or occluded.
    Visibility is therefore discrete at the configured splat radius; this function
    does not claim continuous triangle-rasterised occlusion between coarse samples.
    """

    if not isinstance(config, BladeForegroundConfig):
        raise TypeError("config must be BladeForegroundConfig")
    if not config.enabled:
        raise BladeForegroundMaskError("Blade foreground extraction is disabled")
    if config.method != "reference_projected":
        raise BladeForegroundMaskError("Unsupported blade foreground method")
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be CameraIntrinsics")
    if intrinsics.distortion_model != "none" or intrinsics.distortion_coefficients:
        raise BladeForegroundMaskError(
            "Reference projection requires distortion-free left_rectified intrinsics"
        )
    if not isinstance(base_t_left_rectified, PoseSE3) or (
        base_t_left_rectified.parent_frame != "base"
        or base_t_left_rectified.child_frame != "left_rectified"
    ):
        raise BladeForegroundMaskError("Reference projection requires base_T_left_rectified")
    if not isinstance(surface, CurvedBladeSurface) or surface.frame != "base":
        raise BladeForegroundMaskError("Reference surface must be a base-frame model")
    if not target_patch_id:
        raise BladeForegroundMaskError("Target patch ID must be non-empty")

    depth_input = np.asarray(depth_m)
    eligible_input = np.asarray(eligible_mask)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth_input.ndim != 2 or depth_input.shape != expected_shape:
        raise BladeForegroundMaskError(
            f"Depth shape must match rectified intrinsics {expected_shape}"
        )
    if not np.issubdtype(depth_input.dtype, np.floating):
        raise BladeForegroundMaskError("Depth must use floating-point metres")
    if eligible_input.dtype != np.bool_ or eligible_input.shape != expected_shape:
        raise BladeForegroundMaskError(f"eligible_mask must be boolean with shape {expected_shape}")
    depth = np.asarray(depth_input, dtype=np.float64)
    eligible = np.asarray(eligible_input, dtype=np.bool_)
    if np.isinf(depth).any() or np.any(np.isfinite(depth) & (depth < 0.0)):
        raise BladeForegroundMaskError(
            "Depth may contain positive metres, zero, or NaN, but not negative/infinite values"
        )
    if not np.any(eligible):
        raise BladeForegroundMaskError("eligible_mask is empty")

    target_patches = tuple(patch for patch in surface.patches if patch.patch_id == target_patch_id)
    if len(target_patches) != 1:
        raise BladeForegroundMaskError(
            f"Target patch {target_patch_id!r} is not unique in the reference surface"
        )
    target_patch = target_patches[0]
    camera_to_target = base_t_left_rectified.translation_m - target_patch.obb_center_m
    camera_distance = float(np.linalg.norm(camera_to_target))
    if camera_distance <= 1e-12:
        raise BladeForegroundMaskError("Target patch centre coincides with the camera")
    target_incidence_cosine = float(
        np.dot(target_patch.main_normal, camera_to_target / camera_distance)
    )
    if target_incidence_cosine < config.minimum_target_incidence_cosine:
        raise BladeForegroundMaskError(
            "Target patch does not face the current camera with sufficient incidence"
        )

    all_points = np.vstack([patch.points_m for patch in surface.patches])
    left_rectified_t_base = base_t_left_rectified.inverse()
    reference_depth = _projected_z_buffer(
        all_points,
        left_rectified_t_base=left_rectified_t_base,
        intrinsics=intrinsics,
        config=config,
    )
    target_reference_depth = _projected_z_buffer(
        target_patch.points_m,
        left_rectified_t_base=left_rectified_t_base,
        intrinsics=intrinsics,
        config=config,
    )

    reference_support = np.isfinite(reference_depth)
    # A target pixel is visible only when the target patch itself wins the full
    # surface z-buffer.  Merely falling within the asymmetric depth tolerance is
    # insufficient for thin walls: a front surface 2 mm ahead of a back target
    # would otherwise be counted as a successful back-side observation.
    target_support = (
        np.isfinite(target_reference_depth)
        & np.isfinite(reference_depth)
        & (np.abs(target_reference_depth - reference_depth) <= _Z_BUFFER_OWNERSHIP_ATOL_M)
    )
    target_reference_depth = np.where(
        target_support,
        target_reference_depth,
        np.nan,
    )
    eligible_reference = eligible & reference_support
    eligible_target_reference = eligible & target_support
    reference_pixel_count = int(np.count_nonzero(reference_support))
    eligible_reference_pixel_count = int(np.count_nonzero(eligible_reference))
    target_reference_pixel_count = int(np.count_nonzero(target_support))
    eligible_target_reference_pixel_count = int(np.count_nonzero(eligible_target_reference))
    if reference_pixel_count < config.minimum_reference_pixels:
        raise BladeForegroundMaskError(
            "Projected reference support is below minimum_reference_pixels"
        )
    if eligible_reference_pixel_count < config.minimum_reference_pixels:
        raise BladeForegroundMaskError(
            "Eligible projected reference support is below minimum_reference_pixels"
        )
    if target_reference_pixel_count < config.minimum_target_reference_pixels:
        raise BladeForegroundMaskError(
            "Projected target-patch support is below minimum_target_reference_pixels"
        )
    if eligible_target_reference_pixel_count < config.minimum_target_reference_pixels:
        raise BladeForegroundMaskError(
            "Eligible target-patch support is below minimum_target_reference_pixels"
        )

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    matched = (
        eligible_reference
        & valid_depth
        & (depth >= reference_depth - config.front_depth_tolerance_m)
        & (depth <= reference_depth + config.back_depth_tolerance_m)
    )
    target_matched = (
        matched
        & target_support
        & (depth >= target_reference_depth - config.front_depth_tolerance_m)
        & (depth <= target_reference_depth + config.back_depth_tolerance_m)
    )
    image_pixel_count = int(depth.size)
    mask_pixel_count = int(np.count_nonzero(matched))
    target_mask_pixel_count = int(np.count_nonzero(target_matched))
    mask_fraction = mask_pixel_count / image_pixel_count
    reference_match_fraction = mask_pixel_count / eligible_reference_pixel_count
    target_match_fraction = target_mask_pixel_count / eligible_target_reference_pixel_count

    if mask_pixel_count < config.minimum_mask_pixels:
        raise BladeForegroundMaskError("Blade foreground is below minimum_mask_pixels")
    if target_mask_pixel_count < config.minimum_target_mask_pixels:
        raise BladeForegroundMaskError(
            "Target-patch foreground is below minimum_target_mask_pixels"
        )
    if reference_match_fraction < config.minimum_reference_match_fraction:
        raise BladeForegroundMaskError(
            "Reference foreground match fraction is below its configured minimum"
        )
    if target_match_fraction < config.minimum_target_match_fraction:
        raise BladeForegroundMaskError(
            "Target-patch foreground match fraction is below its configured minimum"
        )
    if not config.minimum_mask_fraction <= mask_fraction <= config.maximum_mask_fraction:
        raise BladeForegroundMaskError(
            "Blade foreground image fraction is outside configured bounds"
        )

    diagnostics = BladeForegroundDiagnostics(
        target_patch_id=target_patch_id,
        target_incidence_cosine=target_incidence_cosine,
        image_pixel_count=image_pixel_count,
        eligible_pixel_count=int(np.count_nonzero(eligible)),
        valid_eligible_depth_pixel_count=int(np.count_nonzero(eligible & valid_depth)),
        reference_pixel_count=reference_pixel_count,
        eligible_reference_pixel_count=eligible_reference_pixel_count,
        target_reference_pixel_count=target_reference_pixel_count,
        eligible_target_reference_pixel_count=eligible_target_reference_pixel_count,
        mask_pixel_count=mask_pixel_count,
        target_mask_pixel_count=target_mask_pixel_count,
        mask_fraction=mask_fraction,
        reference_match_fraction=reference_match_fraction,
        target_match_fraction=target_match_fraction,
    )
    return BladeForegroundMaskResult(
        mask=matched,
        reference_depth_m=reference_depth,
        target_reference_depth_m=target_reference_depth,
        eligible_mask=eligible,
        diagnostics=diagnostics,
        config=config,
        algorithm=REFERENCE_PROJECTED_ALGORITHM,
        policy_sha256=_policy_sha256(config),
    )
