import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ViewPlanningConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    BladeSide,
    ViewPlanningError,
    generate_bilateral_view_plan,
)


def make_proxy(extents=(0.4, 0.2, 0.02)) -> BilateralBladeProxy:
    return BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.asarray(extents),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(101, 101, 50.0, 50.0, 50.0, 50.0, "none", ())


def test_bilateral_plan_partitions_both_faces_and_looks_at_targets() -> None:
    config = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
    )

    plan = generate_bilateral_view_plan(make_proxy(), make_intrinsics(), config)

    assert plan.rows == 1
    assert plan.columns == 2
    assert len(plan.candidates) == 4
    front = plan.for_side(BladeSide.FRONT)
    back = plan.for_side(BladeSide.BACK)
    assert len(front) == len(back) == 2
    for candidate in plan.candidates:
        camera_to_target = candidate.patch.target_m - candidate.base_t_left_ir.translation_m
        camera_to_target /= np.linalg.norm(camera_to_target)
        np.testing.assert_allclose(candidate.optical_axis, camera_to_target, atol=1e-12)
        assert np.linalg.det(candidate.base_t_left_ir.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(front[0].base_t_left_ir.translation_m[2], 0.11)
    np.testing.assert_allclose(back[0].base_t_left_ir.translation_m[2], -0.11)


def test_overlap_increases_candidate_count() -> None:
    no_overlap = generate_bilateral_view_plan(
        make_proxy((0.6, 0.2, 0.02)),
        make_intrinsics(),
        ViewPlanningConfig(
            standoff_distance_m=0.1,
            overlap_fraction=0.0,
            footprint_utilization=1.0,
            edge_margin_m=0.0,
        ),
    )
    overlap = generate_bilateral_view_plan(
        make_proxy((0.6, 0.2, 0.02)),
        make_intrinsics(),
        ViewPlanningConfig(
            standoff_distance_m=0.1,
            overlap_fraction=0.5,
            footprint_utilization=1.0,
            edge_margin_m=0.0,
        ),
    )

    assert overlap.columns > no_overlap.columns


def test_view_plan_requires_explicit_standoff() -> None:
    with pytest.raises(ViewPlanningError, match="standoff_distance_m"):
        generate_bilateral_view_plan(make_proxy(), make_intrinsics(), ViewPlanningConfig())


def test_view_plan_enforces_candidate_limit() -> None:
    with pytest.raises(ViewPlanningError, match="exceeding"):
        generate_bilateral_view_plan(
            make_proxy((1.0, 1.0, 0.02)),
            make_intrinsics(),
            ViewPlanningConfig(
                standoff_distance_m=0.05,
                footprint_utilization=0.5,
                maximum_candidates=10,
            ),
        )


def test_view_plan_rejects_margin_larger_than_coverage() -> None:
    with pytest.raises(ViewPlanningError, match="edge_margin_m"):
        generate_bilateral_view_plan(
            make_proxy(),
            make_intrinsics(),
            ViewPlanningConfig(
                standoff_distance_m=0.01,
                footprint_utilization=0.5,
                edge_margin_m=0.02,
            ),
        )
