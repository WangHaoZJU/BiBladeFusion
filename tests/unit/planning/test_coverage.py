import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CoverageConfig, ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    BladeSide,
    CoverageError,
    create_coverage_ledger,
    filter_candidate_views,
    generate_bilateral_view_plan,
    select_uncovered_candidates,
    update_coverage,
)


def make_proxy() -> BilateralBladeProxy:
    return BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.4, 0.2, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )


def make_plan():
    return generate_bilateral_view_plan(
        make_proxy(),
        CameraIntrinsics(101, 101, 50.0, 50.0, 50.0, 50.0, "none", ()),
        ViewPlanningConfig(
            standoff_distance_m=0.1,
            overlap_fraction=0.0,
            footprint_utilization=1.0,
            edge_margin_m=0.0,
        ),
    )


def patch_cloud(*, front: bool) -> PointCloud:
    bins = 4
    x = np.linspace(-0.2 + 0.2 / (2 * bins), -0.2 / (2 * bins), bins)
    y = np.linspace(-0.1 + 0.2 / (2 * bins), 0.1 - 0.2 / (2 * bins), bins)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, 0.01 if front else -0.01)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    pixels = np.column_stack(np.unravel_index(np.arange(len(points)), (bins, bins)))[..., ::-1]
    return PointCloud("base", points, pixels, (bins, bins))


def coverage_config() -> CoverageConfig:
    return CoverageConfig(
        bins_per_axis=4,
        minimum_points_per_bin=1,
        completed_fraction=0.75,
        maximum_surface_distance_m=0.005,
        minimum_camera_side_offset_m=0.02,
        minimum_surface_points_per_view=10,
    )


def test_bilateral_coverage_keeps_front_and_back_evidence_separate() -> None:
    plan = make_plan()
    ledger = create_coverage_ledger(plan, coverage_config())
    ledger = update_coverage(
        ledger,
        plan,
        make_proxy(),
        patch_cloud(front=True),
        PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0, 0, 0.2]),
        "front-view",
    )

    assert ledger.is_complete("front_r00_c00")
    assert not ledger.is_complete("front_r00_c01")
    assert ledger.completion_fraction(BladeSide.FRONT) == pytest.approx(0.5)
    assert ledger.completion_fraction(BladeSide.BACK) == 0.0

    ledger = update_coverage(
        ledger,
        plan,
        make_proxy(),
        patch_cloud(front=False),
        PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0, 0, -0.2]),
        "back-view",
    )

    assert ledger.is_complete("back_r00_c00")
    assert ledger.completion_fraction() == pytest.approx(0.5)


def test_replanning_returns_only_accepted_incomplete_patches() -> None:
    plan = make_plan()
    filtered = filter_candidate_views(
        plan.candidates,
        make_proxy(),
        ViewFilterConfig(camera_clearance_radius_m=0.01),
    )
    ledger = update_coverage(
        create_coverage_ledger(plan, coverage_config()),
        plan,
        make_proxy(),
        patch_cloud(front=True),
        PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0, 0, 0.2]),
        "front-view",
    )

    replanned = select_uncovered_candidates(filtered, ledger)

    assert "front_r00_c00" in replanned.completed_patch_ids
    assert {item.candidate.view_id for item in replanned.remaining} == {
        "front_r00_c01",
        "back_r00_c00",
        "back_r00_c01",
    }
    assert not replanned.blocked_patch_ids
    assert replanned.motion_authorized is False


def test_coverage_rejects_duplicate_observation_and_midplane_camera() -> None:
    plan = make_plan()
    ledger = create_coverage_ledger(plan, coverage_config())
    camera = PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0, 0, 0.2])
    ledger = update_coverage(
        ledger,
        plan,
        make_proxy(),
        patch_cloud(front=True),
        camera,
        "same-view",
    )

    with pytest.raises(CoverageError, match="already recorded"):
        update_coverage(
            ledger,
            plan,
            make_proxy(),
            patch_cloud(front=True),
            camera,
            "same-view",
        )
    with pytest.raises(CoverageError, match="mid-plane"):
        update_coverage(
            create_coverage_ledger(plan, coverage_config()),
            plan,
            make_proxy(),
            patch_cloud(front=True),
            PoseSE3.identity("base", "left_ir"),
            "ambiguous-side",
        )
