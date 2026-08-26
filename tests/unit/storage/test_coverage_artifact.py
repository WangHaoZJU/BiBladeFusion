from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CoverageConfig, ViewPlanningConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    create_coverage_ledger,
    generate_bilateral_view_plan,
    update_coverage,
)
from biblade_fusion.storage import read_coverage_ledger, write_coverage_ledger


def make_ledger():
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.2, 0.2, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )
    plan = generate_bilateral_view_plan(
        proxy,
        CameraIntrinsics(101, 101, 50.0, 50.0, 50.0, 50.0, "none", ()),
        ViewPlanningConfig(
            standoff_distance_m=0.1,
            overlap_fraction=0.0,
            footprint_utilization=1.0,
            edge_margin_m=0.0,
        ),
    )
    config = CoverageConfig(
        bins_per_axis=2,
        minimum_points_per_bin=1,
        completed_fraction=1.0,
        maximum_surface_distance_m=0.01,
        minimum_surface_points_per_view=4,
    )
    x, y = np.meshgrid([-0.05, 0.05], [-0.05, 0.05])
    points = np.column_stack((x.ravel(), y.ravel(), np.full(4, 0.01)))
    cloud = PointCloud("base", points, np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), (2, 2))
    ledger = update_coverage(
        create_coverage_ledger(plan, config),
        plan,
        proxy,
        cloud,
        PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0, 0, 0.2]),
        "seed",
    )
    return ledger


def test_coverage_artifact_round_trip_and_checksum(tmp_path: Path) -> None:
    output = write_coverage_ledger(
        tmp_path / "coverage",
        make_ledger(),
        source_plan=tmp_path / "plan",
        source_initialization=tmp_path / "initialization",
    )
    stored = read_coverage_ledger(output)

    assert stored.ledger.observation_ids == ("seed",)
    assert stored.ledger.completion_fraction() == pytest.approx(0.5)
    assert stored.metadata["motion_authorized"] is False

    counts = np.load(output / "bin_point_counts.npy", allow_pickle=False)
    counts[0, 0, 0] += 1
    np.save(output / "bin_point_counts.npy", counts, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_coverage_ledger(output)


def test_coverage_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_coverage_ledger(
            output,
            make_ledger(),
            source_plan=tmp_path / "plan",
            source_initialization=tmp_path / "initialization",
        )
