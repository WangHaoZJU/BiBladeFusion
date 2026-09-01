import numpy as np
import pytest

from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.perception.proxy import (
    PROXY_SUPPORT_ALGORITHM,
    ProxySupportError,
    select_proxy_support,
)


def envelope_config(**updates: object) -> ProxyModelConfig:
    values: dict[str, object] = {
        "minimum_points": 6,
        "estimated_thickness_m": 0.01,
        "blade_envelope_min_m": (0.0, -0.1, 0.0),
        "blade_envelope_max_m": (0.2, 0.1, 0.3),
        "minimum_envelope_retained_fraction": 0.6,
    }
    values.update(updates)
    return ProxyModelConfig(**values)


def test_proxy_support_intersects_hard_roi_cloud_with_base_envelope() -> None:
    points = np.array(
        [
            [0.05, 0.00, 0.10],
            [0.10, -0.05, 0.20],
            [0.20, 0.10, 0.30],
            [0.06, 0.01, 0.11],
            [0.11, -0.04, 0.21],
            [0.19, 0.09, 0.29],
            [0.25, 0.00, 0.10],
            [0.10, 0.00, -0.01],
            [0.30, 0.00, 0.10],
            [0.10, 0.20, 0.10],
        ]
    )

    support = select_proxy_support(points, envelope_config(), frame="base")

    np.testing.assert_array_equal(
        support.mask,
        [True, True, True, True, True, True, False, False, False, False],
    )
    assert support.input_point_count == 10
    assert support.retained_point_count == 6
    assert support.rejected_point_count == 4
    assert support.retained_fraction == pytest.approx(0.6)
    assert support.metadata_payload()["algorithm"] == PROXY_SUPPORT_ALGORITHM
    np.testing.assert_allclose(support.retained_bounds_min_m, [0.05, -0.05, 0.1])
    np.testing.assert_allclose(support.retained_bounds_max_m, [0.2, 0.1, 0.3])


def test_proxy_support_fails_when_envelope_rejects_too_much_hard_roi() -> None:
    points = np.array(
        [
            [0.05, 0.00, 0.10],
            [0.10, -0.05, 0.20],
            [0.20, 0.10, 0.30],
            [0.06, 0.01, 0.11],
            [0.11, -0.04, 0.21],
            [0.19, 0.09, 0.29],
            [0.25, 0.00, 0.10],
            [0.30, 0.00, 0.10],
            [0.35, 0.00, 0.10],
            [0.40, 0.00, 0.10],
        ]
    )

    with pytest.raises(ProxySupportError, match="60.000%"):
        select_proxy_support(
            points,
            envelope_config(minimum_envelope_retained_fraction=0.8),
            frame="base",
        )


def test_proxy_support_fails_when_too_few_points_remain() -> None:
    points = np.array(
        [
            [0.05, 0.00, 0.10],
            [0.10, -0.05, 0.20],
            [0.25, 0.00, 0.10],
            [0.30, 0.00, 0.10],
            [0.35, 0.00, 0.10],
            [0.40, 0.00, 0.10],
        ]
    )

    with pytest.raises(ProxySupportError, match="2/6 points"):
        select_proxy_support(
            points,
            envelope_config(minimum_envelope_retained_fraction=0.5),
            frame="base",
        )


def test_proxy_support_rejects_envelope_in_camera_frame() -> None:
    points = np.array(
        [
            [0.05, 0.0, 0.1],
            [0.10, 0.0, 0.2],
            [0.15, 0.0, 0.25],
            [0.06, 0.0, 0.11],
            [0.11, 0.0, 0.21],
            [0.16, 0.0, 0.26],
        ]
    )

    with pytest.raises(ProxySupportError, match="base-frame"):
        select_proxy_support(points, envelope_config(), frame="camera")


def test_proxy_support_without_measured_envelope_preserves_every_point() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    support = select_proxy_support(
        points,
        ProxyModelConfig(estimated_thickness_m=0.01),
        frame="base",
    )

    assert support.envelope_enabled is False
    assert support.retained_fraction == 1.0
    assert support.mask.all()
