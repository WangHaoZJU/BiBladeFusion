from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import BladeForegroundConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.blade_foreground import (
    REFERENCE_PROJECTED_ALGORITHM,
    BladeForegroundMaskError,
    reference_guided_blade_mask,
)
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    CurvedSurfacePatch,
    SurfaceRegion,
)
from biblade_fusion.planning.views import BladeSide


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(31, 21, 10.0, 10.0, 15.0, 10.0, "none", ())


def _camera_pose() -> PoseSE3:
    # A non-identity transform makes an accidental base-frame projection fail.
    return PoseSE3.from_rotation_translation(
        "base",
        "left_rectified",
        np.eye(3),
        [0.20, -0.10, 0.05],
    )


def _patch(
    patch_id: str,
    side: BladeSide,
    image_u: int,
    image_rows: tuple[int, ...],
    *,
    z_m: float = 1.0,
    normal: np.ndarray | None = None,
) -> CurvedSurfacePatch:
    intrinsics = _intrinsics()
    camera_points = np.asarray(
        [
            [
                (image_u - intrinsics.cx) * z_m / intrinsics.fx,
                (row - intrinsics.cy) * z_m / intrinsics.fy,
                z_m,
            ]
            for row in image_rows
        ],
        dtype=np.float64,
    )
    points_base = _camera_pose().transform_points(camera_points)
    patch_normal = (
        np.array([0.0, 0.0, -1.0 if side is BladeSide.FRONT else 1.0])
        if normal is None
        else np.asarray(normal, dtype=np.float64)
    )
    return CurvedSurfacePatch(
        patch_id=patch_id,
        side=side,
        region=SurfaceRegion.FIN_FACE if side is BladeSide.FRONT else SurfaceRegion.SURFACE,
        row=0,
        column=0,
        adaptive_depth=0,
        points_m=points_base,
        normals=np.repeat(patch_normal[None, :], len(points_base), axis=0),
        section_coordinates=np.column_stack(
            (np.linspace(0.0, 1.0, len(points_base)), np.zeros(len(points_base)))
        ),
        obb_center_m=np.mean(points_base, axis=0),
        obb_axes=np.eye(3),
        obb_extents_m=np.array([0.05, 0.01, 0.001]),
        main_normal=patch_normal,
        curvature_deg=0.0,
        boundary_fraction=0.0,
    )


def _surface() -> CurvedBladeSurface:
    return CurvedBladeSurface(
        frame="base",
        patches=(
            _patch("front_fin", BladeSide.FRONT, 12, (7, 8, 9, 10, 11, 12)),
            _patch("back_main", BladeSide.BACK, 18, (7, 8, 9, 10, 11, 12)),
        ),
        axes=np.eye(3),
        center_m=np.array([0.2, -0.1, 1.05]),
        section_arc_lengths_m=(0.2, 0.1),
        angle_boundary_counts=(6, 6),
        base_grid_counts=(1, 1),
        base_footprint_m=(0.1, 0.1),
        footprint_source="configured_override",
    )


def _config(**updates: object) -> BladeForegroundConfig:
    values: dict[str, object] = {
        "enabled": True,
        "projection_radius_px": 0,
        "front_depth_tolerance_m": 0.001,
        "back_depth_tolerance_m": 0.001,
        "minimum_reference_pixels": 1,
        "minimum_target_reference_pixels": 1,
        "minimum_mask_pixels": 1,
        "minimum_target_mask_pixels": 1,
        "minimum_reference_match_fraction": 0.01,
        "minimum_target_match_fraction": 0.01,
        "minimum_mask_fraction": 0.0,
        "maximum_mask_fraction": 0.90,
    }
    values.update(updates)
    return BladeForegroundConfig(**values)


def _depth() -> np.ndarray:
    # Background deliberately has the same depth as the blade: the projected
    # reference support, not a depth-only connected component, excludes it.
    return np.ones((_intrinsics().height, _intrinsics().width), dtype=np.float32)


def test_non_identity_pose_excludes_background_and_preserves_thin_feature() -> None:
    eligible = np.ones_like(_depth(), dtype=np.bool_)

    result = reference_guided_blade_mask(
        _depth(),
        eligible,
        _intrinsics(),
        _camera_pose(),
        _surface(),
        "front_fin",
        _config(),
    )

    expected = np.zeros_like(eligible)
    expected[7:13, 12] = True
    expected[7:13, 18] = True
    assert np.array_equal(result.mask, expected)
    assert np.count_nonzero(result.mask[:, 12]) == 6
    assert result.mask[:, 11].sum() == 0
    assert result.mask[:, 13].sum() == 0
    assert result.diagnostics.target_mask_pixel_count == 6
    assert result.algorithm == REFERENCE_PROJECTED_ALGORITHM
    assert len(result.policy_sha256) == 64
    assert not result.mask.flags.writeable
    assert not result.reference_depth_m.flags.writeable


def test_eligible_mask_excludes_robot_pixel_without_eroding_neighbours() -> None:
    eligible = np.ones_like(_depth(), dtype=np.bool_)
    eligible[9, 12] = False

    result = reference_guided_blade_mask(
        _depth(),
        eligible,
        _intrinsics(),
        _camera_pose(),
        _surface(),
        "front_fin",
        _config(),
    )

    assert result.mask[9, 12] == np.False_
    assert result.mask[8, 12] == np.True_
    assert result.mask[10, 12] == np.True_
    assert not np.any(result.mask & ~eligible)
    assert result.diagnostics.target_mask_pixel_count == 5


def test_target_patch_has_an_independent_projected_support_gate() -> None:
    with pytest.raises(BladeForegroundMaskError, match="target-patch support"):
        reference_guided_blade_mask(
            _depth(),
            np.ones_like(_depth(), dtype=np.bool_),
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(minimum_target_reference_pixels=7),
        )


def test_target_patch_has_an_independent_depth_match_gate() -> None:
    depth = _depth()
    depth[7:13, 12] = 0.5

    with pytest.raises(BladeForegroundMaskError, match="Target-patch foreground"):
        reference_guided_blade_mask(
            depth,
            np.ones_like(depth, dtype=np.bool_),
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(),
        )


def test_occluded_thin_back_surface_does_not_own_target_support() -> None:
    front = _patch(
        "front_main",
        BladeSide.FRONT,
        12,
        (7, 8, 9, 10, 11, 12),
        z_m=1.0,
    )
    hidden_back = _patch(
        "back_target_hidden",
        BladeSide.BACK,
        12,
        (7, 8, 9, 10, 11, 12),
        z_m=1.002,
        # Deliberately face the test camera so this test isolates z-buffer
        # ownership from the independent incidence gate.
        normal=np.array([0.0, 0.0, -1.0]),
    )
    visible_back = _patch(
        "back_target_visible",
        BladeSide.BACK,
        18,
        (7, 8, 9, 10, 11, 12),
        z_m=1.002,
        normal=np.array([0.0, 0.0, -1.0]),
    )
    target = CurvedSurfacePatch(
        patch_id="back_target",
        side=BladeSide.BACK,
        region=SurfaceRegion.SURFACE,
        row=0,
        column=0,
        adaptive_depth=0,
        points_m=np.vstack((hidden_back.points_m, visible_back.points_m)),
        normals=np.vstack((hidden_back.normals, visible_back.normals)),
        section_coordinates=np.vstack(
            (hidden_back.section_coordinates, visible_back.section_coordinates)
        ),
        obb_center_m=np.mean(np.vstack((hidden_back.points_m, visible_back.points_m)), axis=0),
        obb_axes=np.eye(3),
        obb_extents_m=np.array([0.6, 0.5, 0.002]),
        main_normal=np.array([0.0, 0.0, -1.0]),
        curvature_deg=0.0,
        boundary_fraction=0.0,
    )
    surface = CurvedBladeSurface(
        frame="base",
        patches=(front, target),
        axes=np.eye(3),
        center_m=np.array([0.2, -0.1, 1.05]),
        section_arc_lengths_m=(0.2, 0.1),
        angle_boundary_counts=(6, 6),
        base_grid_counts=(1, 1),
        base_footprint_m=(0.1, 0.1),
        footprint_source="configured_override",
    )
    depth = np.full((_intrinsics().height, _intrinsics().width), np.nan, dtype=np.float32)
    depth[7:13, 12] = 1.0
    depth[7:13, 18] = 1.002

    result = reference_guided_blade_mask(
        depth,
        np.ones_like(depth, dtype=np.bool_),
        _intrinsics(),
        _camera_pose(),
        surface,
        "back_target",
        _config(
            front_depth_tolerance_m=0.006,
            back_depth_tolerance_m=0.010,
        ),
    )

    assert np.isnan(result.target_reference_depth_m[7:13, 12]).all()
    assert np.isfinite(result.target_reference_depth_m[7:13, 18]).all()
    assert result.diagnostics.target_reference_pixel_count == 6
    assert result.diagnostics.target_mask_pixel_count == 6


def test_target_patch_incidence_gate_rejects_opposite_side() -> None:
    with pytest.raises(BladeForegroundMaskError, match="sufficient incidence"):
        reference_guided_blade_mask(
            _depth(),
            np.ones_like(_depth(), dtype=np.bool_),
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "back_main",
            _config(minimum_target_incidence_cosine=0.1),
        )


def test_default_projection_radius_preserves_thin_fin_and_rejects_far_background() -> None:
    depth = np.full_like(_depth(), 1.05)
    depth[7:13, 12] = 1.0

    result = reference_guided_blade_mask(
        depth,
        np.ones_like(depth, dtype=np.bool_),
        _intrinsics(),
        _camera_pose(),
        _surface(),
        "front_fin",
        _config(projection_radius_px=BladeForegroundConfig().projection_radius_px),
    )

    assert np.count_nonzero(result.mask) == 6
    assert np.all(result.mask[7:13, 12])
    assert not np.any(result.mask[:, :9])
    assert not np.any(result.mask[:, 21:])


def test_empty_small_and_oversized_masks_fail_closed() -> None:
    eligible = np.ones_like(_depth(), dtype=np.bool_)
    with pytest.raises(BladeForegroundMaskError, match="eligible_mask is empty"):
        reference_guided_blade_mask(
            _depth(),
            np.zeros_like(eligible),
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(),
        )

    with pytest.raises(BladeForegroundMaskError, match="minimum_mask_pixels"):
        reference_guided_blade_mask(
            _depth(),
            eligible,
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(minimum_mask_pixels=13),
        )

    with pytest.raises(BladeForegroundMaskError, match="outside configured bounds"):
        reference_guided_blade_mask(
            _depth(),
            eligible,
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(projection_radius_px=10, maximum_mask_fraction=0.10),
        )


def test_frame_shape_and_rectification_contracts_are_strict() -> None:
    eligible = np.ones_like(_depth(), dtype=np.bool_)
    with pytest.raises(BladeForegroundMaskError, match="base_T_left_rectified"):
        reference_guided_blade_mask(
            _depth(),
            eligible,
            _intrinsics(),
            PoseSE3.identity("base", "left_ir"),
            _surface(),
            "front_fin",
            _config(),
        )
    with pytest.raises(BladeForegroundMaskError, match="Depth shape"):
        reference_guided_blade_mask(
            _depth()[:-1],
            eligible[:-1],
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(),
        )
    distorted = CameraIntrinsics(31, 21, 10.0, 10.0, 15.0, 10.0, "brown5", (0.1,))
    with pytest.raises(BladeForegroundMaskError, match="distortion-free"):
        reference_guided_blade_mask(
            _depth(),
            eligible,
            distorted,
            _camera_pose(),
            _surface(),
            "front_fin",
            _config(),
        )


def test_disabled_policy_cannot_be_used_implicitly() -> None:
    with pytest.raises(BladeForegroundMaskError, match="disabled"):
        reference_guided_blade_mask(
            _depth(),
            np.ones_like(_depth(), dtype=np.bool_),
            _intrinsics(),
            _camera_pose(),
            _surface(),
            "front_fin",
            BladeForegroundConfig(),
        )
