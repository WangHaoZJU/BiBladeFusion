from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.mapping.self_mask import (
    RobotSelfMaskConfig,
    depth_consistent_robot_self_mask,
)


def test_depth_consistent_mask_blocks_matching_and_behind_robot_rays() -> None:
    measured = np.array([[1.0, 0.7, 1.3, np.nan]], dtype=np.float64)
    predicted = np.ones_like(measured)

    result = depth_consistent_robot_self_mask(
        measured,
        predicted,
        config=RobotSelfMaskConfig(
            front_tolerance_m=0.05,
            back_tolerance_m=0.05,
            dilation_px=0,
        ),
    )

    np.testing.assert_array_equal(result.robot_mask, [[True, False, True, False]])
    np.testing.assert_array_equal(
        result.integration_valid_mask,
        [[False, True, False, False]],
    )
    assert result.report.depth_matched_pixels == 1
    assert result.report.retained_valid_pixels == 1


def test_farther_measurement_cannot_ray_clear_through_projected_robot() -> None:
    result = depth_consistent_robot_self_mask(
        [[1.20]],
        [[0.75]],
        config=RobotSelfMaskConfig(dilation_px=0),
    )

    assert bool(result.robot_mask[0, 0]) is True
    assert bool(result.integration_valid_mask[0, 0]) is False
    assert result.report.depth_matched_pixels == 0


def test_closer_unknown_object_in_front_of_projected_robot_is_preserved() -> None:
    result = depth_consistent_robot_self_mask(
        [[0.40]],
        [[0.75]],
        config=RobotSelfMaskConfig(dilation_px=0),
    )

    assert bool(result.robot_mask[0, 0]) is False
    assert bool(result.integration_valid_mask[0, 0]) is True


def test_masked_robot_pixel_is_not_converted_into_free_space_evidence() -> None:
    result = depth_consistent_robot_self_mask(
        [[0.75]],
        [[0.75]],
        config=RobotSelfMaskConfig(dilation_px=0),
    )

    assert bool(result.robot_mask[0, 0]) is True
    assert bool(result.integration_valid_mask[0, 0]) is False


def test_self_mask_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching 2-D"):
        depth_consistent_robot_self_mask(np.ones((2, 2)), np.ones((3, 3)))
