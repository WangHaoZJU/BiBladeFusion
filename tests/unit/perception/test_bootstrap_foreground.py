from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapForegroundError,
    BootstrapSeed,
    array_content_sha256,
    bootstrap_blade_foreground,
)


def _config(**updates: object) -> BootstrapForegroundConfig:
    config = BootstrapForegroundConfig(
        minimum_depth_m=0.2,
        maximum_depth_m=2.0,
        maximum_neighbour_depth_jump_m=0.04,
        maximum_neighbour_relative_depth_jump=0.0,
        boundary_margin_px=1,
        minimum_valid_pixels=1,
        minimum_component_pixels=1,
        minimum_mask_pixels=1,
        minimum_mask_fraction=0.0,
        maximum_mask_fraction=0.9,
        minimum_seed_valid_pixels=1,
        minimum_seed_valid_fraction=0.0,
        minimum_component_hint_selection_fraction=0.01,
    )
    return replace(config, **updates)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    depth = np.full((20, 30), 1.50, dtype=np.float32)
    expected = np.zeros_like(depth, dtype=np.bool_)
    expected[5:15, 10:18] = True
    # One-pixel-wide fin remains connected to the main blade at a modest depth jump.
    expected[9, 18:25] = True
    depth[expected] = 0.80
    depth[9, 18:25] = 0.82
    valid = np.ones_like(expected)
    return left, depth, valid, expected


def test_automatic_bootstrap_excludes_boundary_background_and_preserves_fin() -> None:
    left, depth, valid, expected = _scene()

    result = bootstrap_blade_foreground(left, depth, valid, _config())

    np.testing.assert_array_equal(result.mask, expected)
    assert np.all(result.mask[9, 18:25])
    assert result.diagnostics.boundary_component_count == 1
    assert result.diagnostics.selected_component_count == 1
    assert result.diagnostics.median_mask_depth_m == pytest.approx(0.8)
    assert result.left_image_content_sha256 == array_content_sha256(left)
    assert not result.mask.flags.writeable
    assert not result.seed_mask.flags.writeable


def test_automatic_bootstrap_fails_closed_for_two_similar_objects() -> None:
    left, depth, valid, _ = _scene()
    depth[5:15, 10:18] = 1.50
    depth[9, 18:25] = 1.50
    depth[4:10, 5:10] = 0.70
    depth[11:17, 20:25] = 0.90

    with pytest.raises(BootstrapForegroundError, match="ambiguous"):
        bootstrap_blade_foreground(
            left,
            depth,
            valid,
            _config(maximum_unseeded_ambiguity_ratio=0.5),
        )


def test_component_hint_selects_one_complete_component_without_clipping() -> None:
    left, depth, valid, expected = _scene()
    depth[3:7, 3:7] = 1.0
    hint = BootstrapSeed.rectangle(11, 6, 14, 8, mode="component_hint")

    result = bootstrap_blade_foreground(left, depth, valid, _config(), hint)

    np.testing.assert_array_equal(result.mask, expected)
    assert np.any(result.mask & ~result.seed_mask)
    assert result.diagnostics.selected_seed_fraction > 0.0


def test_hard_polygon_is_deterministic_and_retains_disconnected_thin_support() -> None:
    left, depth, valid, _ = _scene()
    depth.fill(1.5)
    valid.fill(True)
    depth[5:16, 8:26] = np.nan
    valid[5:16, 8:26] = False
    depth[6:14, 9:20] = 0.8
    valid[6:14, 9:20] = True
    # Deliberately disconnected one-pixel fin support inside the human polygon.
    depth[10, 21:24] = 0.83
    valid[10, 21:24] = True
    seed = BootstrapSeed.polygon(
        [(8, 5), (25, 5), (25, 15), (8, 15)],
        mode="hard_roi",
    )

    result = bootstrap_blade_foreground(
        left,
        depth,
        valid,
        _config(minimum_component_pixels=10),
        seed,
    )

    expected = valid & result.seed_mask
    np.testing.assert_array_equal(result.mask, expected)
    assert np.all(result.mask[10, 21:24])
    assert result.diagnostics.selected_component_count == 2


def test_boundary_object_seed_and_invalid_inputs_fail_closed() -> None:
    left, depth, valid, _ = _scene()
    depth.fill(1.5)
    depth[:, :5] = 0.8
    with pytest.raises(BootstrapForegroundError, match="No interior"):
        bootstrap_blade_foreground(left, depth, valid, _config())

    outside = BootstrapSeed.rectangle(-1, 2, 5, 8)
    with pytest.raises(BootstrapForegroundError, match="outside"):
        bootstrap_blade_foreground(left, depth, valid, _config(), outside)

    with pytest.raises(BootstrapForegroundError, match="must match"):
        bootstrap_blade_foreground(left[:, :-1], depth, valid, _config())


def test_depth_gap_policy_and_content_hash_are_auditable() -> None:
    left, depth, valid, expected = _scene()
    depth[9, 18:25] = 1.0

    result = bootstrap_blade_foreground(left, depth, valid, _config())

    assert not np.any(result.mask[9, 18:25])
    assert np.count_nonzero(result.mask) == np.count_nonzero(expected) - 7
    changed = left.copy()
    changed[0, 0] += 1
    assert array_content_sha256(left) != array_content_sha256(changed)
