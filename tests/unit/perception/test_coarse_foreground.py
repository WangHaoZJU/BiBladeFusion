from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapForegroundError,
    array_content_sha256,
)
from biblade_fusion.perception.coarse_foreground import (
    ProjectedCoarseForegroundGuide,
    projected_coarse_blade_foreground,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(40, 30, 40.0, 40.0, 19.5, 14.5, "none", ())


def _points_from_pixels(
    pixels_uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_m: float = 1.0,
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    return np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx) * depth_m / intrinsics.fx,
            (pixels[:, 1] - intrinsics.cy) * depth_m / intrinsics.fy,
            np.full(len(pixels), depth_m),
        )
    )


def _case(
    tmp_path: Path,
    *,
    reference_pixels: np.ndarray,
    dilation_px: int = 2,
    minimum_match_fraction: float = 0.5,
) -> tuple[
    CameraIntrinsics,
    np.ndarray,
    ProjectedCoarseForegroundGuide,
    BootstrapForegroundConfig,
]:
    intrinsics = _intrinsics()
    reference = _points_from_pixels(reference_pixels, intrinsics)
    guide = ProjectedCoarseForegroundGuide(
        source_generation_path=tmp_path / "accepted-generation",
        source_generation_metadata_sha256="a" * 64,
        reference_points_content_sha256=array_content_sha256(reference),
        blade_envelope_min_m=(-0.22, -0.17, 0.9),
        blade_envelope_max_m=(0.22, 0.17, 1.1),
    )
    config = BootstrapForegroundConfig(
        minimum_depth_m=0.2,
        maximum_depth_m=1.5,
        minimum_valid_pixels=1,
        minimum_component_pixels=1,
        minimum_mask_pixels=1,
        minimum_mask_fraction=0.0,
        maximum_mask_fraction=1.0,
        projected_reference_dilation_px=dilation_px,
        minimum_projected_reference_points=1,
        minimum_projected_reference_pixels=1,
        minimum_projected_match_fraction=minimum_match_fraction,
    )
    return intrinsics, reference, guide, config


def test_projected_foreground_rejects_larger_fixture_and_table_components(
    tmp_path: Path,
) -> None:
    reference_pixels = np.asarray(
        [(u, v) for v in range(10, 20) for u in range(13, 27)],
        dtype=np.int64,
    )
    intrinsics, reference, guide, config = _case(
        tmp_path,
        reference_pixels=reference_pixels,
    )
    depth = np.full((30, 40), 0.5, dtype=np.float64)
    expected_blade = np.zeros(depth.shape, dtype=np.bool_)
    expected_blade[10:20, 13:27] = True
    depth[expected_blade] = 1.0

    result = projected_coarse_blade_foreground(
        np.zeros(depth.shape, dtype=np.uint16),
        depth,
        np.ones(depth.shape, dtype=np.bool_),
        config,
        intrinsics=intrinsics,
        base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
        reference_points_base_m=reference,
        guide=guide,
    )

    assert np.array_equal(result.mask, expected_blade)
    assert result.diagnostics.mask_pixel_count == int(np.count_nonzero(expected_blade))
    assert result.diagnostics.eligible_projected_pixel_count > result.diagnostics.mask_pixel_count


def test_projected_foreground_keeps_disconnected_fin_support(tmp_path: Path) -> None:
    upper_fin = np.asarray(
        [(u, v) for v in range(9, 12) for u in range(15, 25)],
        dtype=np.int64,
    )
    lower_fin = np.asarray(
        [(u, v) for v in range(18, 21) for u in range(15, 25)],
        dtype=np.int64,
    )
    reference_pixels = np.vstack((upper_fin, lower_fin))
    intrinsics, reference, guide, config = _case(
        tmp_path,
        reference_pixels=reference_pixels,
        dilation_px=1,
    )
    depth = np.full((30, 40), np.nan, dtype=np.float64)
    expected = np.zeros(depth.shape, dtype=np.bool_)
    expected[9:12, 15:25] = True
    expected[18:21, 15:25] = True
    depth[expected] = 1.0

    result = projected_coarse_blade_foreground(
        np.zeros(depth.shape, dtype=np.uint16),
        depth,
        np.isfinite(depth),
        config,
        intrinsics=intrinsics,
        base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
        reference_points_base_m=reference,
        guide=guide,
    )

    assert np.array_equal(result.mask, expected)
    assert np.any(result.mask[9:12])
    assert np.any(result.mask[18:21])
    assert not np.any(result.mask[12:18])


def test_projected_foreground_uses_local_depth_band_inside_conservative_aabb(
    tmp_path: Path,
) -> None:
    reference_pixels = np.asarray(
        [(u, v) for v in range(10, 20) for u in range(13, 27)],
        dtype=np.int64,
    )
    intrinsics, reference, guide, config = _case(
        tmp_path,
        reference_pixels=reference_pixels,
        dilation_px=2,
        minimum_match_fraction=0.1,
    )
    depth = np.full((30, 40), np.nan, dtype=np.float64)
    depth[10:20, 13:27] = 1.0
    depth[15, 11] = 0.90  # Inside the broad AABB, but not the predicted surface band.
    depth[15, 28] = 1.025  # A newly exposed fin remains within the loose 30 mm band.

    result = projected_coarse_blade_foreground(
        np.zeros(depth.shape, dtype=np.uint16),
        depth,
        np.isfinite(depth),
        config,
        intrinsics=intrinsics,
        base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
        reference_points_base_m=reference,
        guide=guide,
    )

    assert result.projected_reference_mask[15, 11]
    assert result.projected_reference_mask[15, 28]
    assert not result.mask[15, 11]
    assert result.mask[15, 28]
    assert (
        result.diagnostics.predicted_depth_consistent_pixel_count
        < result.diagnostics.eligible_projected_pixel_count
    )


def test_projected_foreground_stops_when_current_depth_disagrees_with_envelope(
    tmp_path: Path,
) -> None:
    reference_pixels = np.asarray(
        [(u, v) for v in range(10, 20) for u in range(13, 27)],
        dtype=np.int64,
    )
    intrinsics, reference, guide, config = _case(
        tmp_path,
        reference_pixels=reference_pixels,
        dilation_px=1,
        minimum_match_fraction=0.75,
    )
    depth = np.full((30, 40), np.nan, dtype=np.float64)
    depth[10:20, 13:20] = 1.0
    depth[10:20, 20:27] = 0.5

    with pytest.raises(BootstrapForegroundError, match="disagrees"):
        projected_coarse_blade_foreground(
            np.zeros(depth.shape, dtype=np.uint16),
            depth,
            np.isfinite(depth),
            config,
            intrinsics=intrinsics,
            base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
            reference_points_base_m=reference,
            guide=guide,
        )


def test_projected_foreground_rejects_changed_reference_identity(tmp_path: Path) -> None:
    reference_pixels = np.asarray(((19, 14), (20, 14)), dtype=np.int64)
    intrinsics, reference, guide, config = _case(
        tmp_path,
        reference_pixels=reference_pixels,
        dilation_px=1,
    )
    changed_reference = reference.copy()
    changed_reference[0, 0] += 0.001

    with pytest.raises(BootstrapForegroundError, match="identity changed"):
        projected_coarse_blade_foreground(
            np.zeros((30, 40), dtype=np.uint16),
            np.ones((30, 40), dtype=np.float64),
            np.ones((30, 40), dtype=np.bool_),
            config,
            intrinsics=intrinsics,
            base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
            reference_points_base_m=changed_reference,
            guide=guide,
        )
