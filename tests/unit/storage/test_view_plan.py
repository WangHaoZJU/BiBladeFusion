from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import CandidateStatus, FilteredViewPlan
from biblade_fusion.storage import read_view_plan, write_view_plan
from biblade_fusion.workflows import InitialObservation, plan_initial_observation


def test_view_plan_round_trip_is_explicitly_non_executable(tmp_path: Path) -> None:
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
    observation = InitialObservation(
        "seed",
        CameraIntrinsics(101, 101, 50, 50, 50, 50, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        PointCloud("base", np.zeros((3, 3)), np.array([[0, 0], [1, 0], [2, 0]]), (101, 101)),
        proxy,
    )
    planning_config = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
    )
    filter_config = ViewFilterConfig(camera_clearance_radius_m=0.01)
    result = plan_initial_observation(observation, planning_config, filter_config)

    output = write_view_plan(
        tmp_path / "plan",
        result,
        planning_config,
        filter_config,
        source_initialization=tmp_path / "initialization",
    )
    stored = read_view_plan(output)

    assert stored.metadata["motion_authorized"] is False
    assert stored.result.motion_authorized is False
    assert len(stored.result.geometric_plan.candidates) == 4
    assert len(stored.result.filtered_plan.accepted) == 4


def test_endpoint_plan_requires_and_verifies_kinematics_provenance(
    tmp_path: Path,
) -> None:
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
    planning = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
    )
    filtering = ViewFilterConfig(camera_clearance_radius_m=0.01)
    result = plan_initial_observation(observation, planning, filtering)
    endpoint = replace(
        result,
        filtered_plan=FilteredViewPlan(
            tuple(
                replace(
                    item,
                    status=CandidateStatus.ENDPOINT_FEASIBLE,
                    joint_positions_rad=np.zeros(6),
                )
                for item in result.filtered_plan.candidates
            ),
            (),
        ),
    )

    with pytest.raises(ValueError, match="must record"):
        write_view_plan(
            tmp_path / "missing",
            endpoint,
            planning,
            filtering,
            source_initialization=tmp_path / "initialization",
        )
    assert not tuple(tmp_path.glob(".*.partial"))

    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("controller-specific-mdh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="joint-zero offsets"):
        write_view_plan(
            tmp_path / "missing-offsets",
            endpoint,
            planning,
            filtering,
            source_initialization=tmp_path / "initialization",
            source_kinematics=kinematics,
        )
    assert not tuple(tmp_path.glob(".*.partial"))
    output = write_view_plan(
        tmp_path / "plan",
        endpoint,
        planning,
        filtering,
        source_initialization=tmp_path / "initialization",
        source_kinematics=kinematics,
        joint_zero_offsets_rad=(0.0,) * 6,
    )
    stored = read_view_plan(output)
    assert stored.metadata["source_kinematics"]["sha256"]
    assert stored.metadata["source_kinematics"]["joint_zero_offsets_rad"] == [0.0] * 6

    kinematics.write_text("changed-mdh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kinematics checksum mismatch"):
        read_view_plan(output)
