import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    CoverageConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import create_coverage_ledger, update_coverage
from biblade_fusion.storage import (
    read_coverage_driven_plan,
    write_coverage_driven_plan,
    write_coverage_ledger,
    write_view_plan,
)
from biblade_fusion.workflows import InitialObservation, plan_initial_observation


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.4, 0.2, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )
    planning = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
    )
    filtering = ViewFilterConfig(camera_clearance_radius_m=0.01)
    observation = InitialObservation(
        "seed",
        CameraIntrinsics(101, 101, 50, 50, 50, 50, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        PointCloud(
            "base",
            np.zeros((3, 3)),
            np.array([[0, 0], [1, 0], [2, 0]]),
            (101, 101),
        ),
        proxy,
    )
    result = plan_initial_observation(observation, planning, filtering)
    view_plan = write_view_plan(
        tmp_path / "view-plan",
        result,
        planning,
        filtering,
        source_initialization=tmp_path / "initialization",
    )
    config = CoverageConfig(
        bins_per_axis=2,
        minimum_points_per_bin=1,
        completed_fraction=1.0,
        maximum_surface_distance_m=0.01,
        minimum_surface_points_per_view=4,
    )
    x, y = np.meshgrid([-0.15, -0.05], [-0.05, 0.05])
    cloud = PointCloud(
        "base",
        np.column_stack((x.ravel(), y.ravel(), np.full(4, 0.01))),
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        (2, 2),
    )
    ledger = update_coverage(
        create_coverage_ledger(result.geometric_plan, config),
        result.geometric_plan,
        proxy,
        cloud,
        PoseSE3.from_rotation_translation(
            "base", "left_ir", np.eye(3), [0, 0, 0.2]
        ),
        "seed",
    )
    coverage = write_coverage_ledger(
        tmp_path / "coverage",
        ledger,
        source_plan=view_plan,
        source_initialization=tmp_path / "initialization",
    )
    return view_plan, coverage


def test_coverage_driven_plan_round_trip_is_non_executable(tmp_path: Path) -> None:
    view_plan, coverage = _write_sources(tmp_path)
    output = write_coverage_driven_plan(
        tmp_path / "next-plan",
        source_plan=view_plan,
        source_coverage=coverage,
    )

    stored = read_coverage_driven_plan(output)

    assert stored.metadata["motion_authorized"] is False
    assert stored.plan.motion_authorized is False
    assert stored.plan.completed_patch_ids == ("front_r00_c00",)
    assert len(stored.plan.remaining) == 3
    assert not stored.plan.blocked_patch_ids


def test_coverage_driven_plan_detects_source_and_summary_tampering(
    tmp_path: Path,
) -> None:
    view_plan, coverage = _write_sources(tmp_path)
    output = write_coverage_driven_plan(
        tmp_path / "next-plan",
        source_plan=view_plan,
        source_coverage=coverage,
    )
    payload_path = output / "coverage_plan.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["remaining_view_ids"] = []
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="remaining_view_ids"):
        read_coverage_driven_plan(output)
