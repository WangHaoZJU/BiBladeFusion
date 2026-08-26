import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.perception.proxy import ProxyBuildError, build_bilateral_proxy


def make_planar_points() -> np.ndarray:
    major = np.linspace(-0.1, 0.1, 21)
    minor = np.linspace(-0.05, 0.05, 11)
    x, y = np.meshgrid(major, minor)
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def camera_pose(z_m: float = 1.0) -> PoseSE3:
    return PoseSE3.from_rotation_translation("base", "camera", np.eye(3), [0, 0, z_m])


def test_proxy_extrudes_hidden_side_and_contains_observation() -> None:
    points = make_planar_points()
    config = ProxyModelConfig(
        voxel_size_m=0.001,
        minimum_points=100,
        estimated_thickness_m=0.01,
        tangential_margin_m=0.002,
        visible_side_margin_m=0.001,
        hidden_side_margin_m=0.003,
    )

    proxy = build_bilateral_proxy(points, camera_pose(), config)

    np.testing.assert_allclose(proxy.outward_normal, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(proxy.extents_m, [0.204, 0.104, 0.014], atol=1e-12)
    np.testing.assert_allclose(proxy.center_m, [0.0, 0.0, -0.006], atol=1e-12)
    assert proxy.frame_T_proxy.parent_frame == "base"
    assert proxy.frame_T_proxy.child_frame == "blade_proxy"
    assert proxy.corners_m().shape == (8, 3)
    assert proxy.contains(points).all()


def test_proxy_uses_conservative_planar_prior() -> None:
    config = ProxyModelConfig(
        voxel_size_m=0.001,
        minimum_points=100,
        estimated_planar_extents_m=(0.25, 0.12),
        estimated_thickness_m=0.01,
        tangential_margin_m=0.002,
    )

    proxy = build_bilateral_proxy(make_planar_points(), camera_pose(), config)

    np.testing.assert_allclose(proxy.extents_m[:2], [0.254, 0.124], atol=1e-12)


def test_proxy_orients_normal_toward_camera_below_surface() -> None:
    config = ProxyModelConfig(
        voxel_size_m=0.001,
        minimum_points=100,
        estimated_thickness_m=0.01,
    )

    proxy = build_bilateral_proxy(make_planar_points(), camera_pose(-1.0), config)

    np.testing.assert_allclose(proxy.outward_normal, [0.0, 0.0, -1.0], atol=1e-12)
    assert proxy.center_m[2] > 0.0


def test_proxy_requires_thickness_prior() -> None:
    config = ProxyModelConfig(voxel_size_m=0.001, minimum_points=100)

    with pytest.raises(ProxyBuildError, match="estimated_thickness_m"):
        build_bilateral_proxy(make_planar_points(), camera_pose(), config)


def test_proxy_rejects_line_like_observation() -> None:
    x = np.linspace(-0.1, 0.1, 100)
    points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    config = ProxyModelConfig(
        voxel_size_m=0.001,
        minimum_points=50,
        estimated_thickness_m=0.01,
    )

    with pytest.raises(ProxyBuildError, match="line-like"):
        build_bilateral_proxy(points, camera_pose(), config)


def test_proxy_rejects_grazing_initial_view() -> None:
    config = ProxyModelConfig(
        voxel_size_m=0.001,
        minimum_points=100,
        estimated_thickness_m=0.01,
        minimum_camera_normal_cosine=0.5,
    )
    grazing_camera = PoseSE3.from_rotation_translation(
        "base", "camera", np.eye(3), [1.0, 0.0, 0.01]
    )

    with pytest.raises(ProxyBuildError, match="too grazing"):
        build_bilateral_proxy(make_planar_points(), grazing_camera, config)
