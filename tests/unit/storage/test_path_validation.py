import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import (
    Cs68KinematicsModel,
    HandEyeCalibration,
    write_cs68_kinematics,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    CollisionConfig,
    CollisionObstacleConfig,
    PointCloudConfig,
    ProxyModelConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    CandidateMetrics,
    CandidateStatus,
    EvaluatedCandidate,
    FilteredViewPlan,
    generate_bilateral_view_plan,
)
from biblade_fusion.storage import (
    read_path_validation,
    write_initialization,
    write_path_validation,
    write_view_plan,
)
from biblade_fusion.workflows import InitialObservation, OfflineViewPlanningResult


def _sources(tmp_path: Path):
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
    intrinsics = CameraIntrinsics(101, 101, 50, 50, 50, 50, "none", ())
    observation = InitialObservation(
        "seed",
        intrinsics,
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
    hand_eye = HandEyeCalibration(
        PoseSE3.from_rotation_translation("tcp", "left_ir", np.eye(3), [0.1, 0, 0]),
        "test",
        20,
        0.001,
        0.2,
        tmp_path / "hand_eye.yaml",
    )
    point_config = PointCloudConfig(minimum_valid_points=3)
    proxy_config = ProxyModelConfig(estimated_thickness_m=0.01)
    initialization = write_initialization(
        tmp_path / "initialization",
        observation,
        np.ones((101, 101), dtype=bool),
        hand_eye,
        point_config,
        proxy_config,
        source_session=tmp_path / "session",
    )
    planning = ViewPlanningConfig(
        standoff_distance_m=0.1,
        overlap_fraction=0.0,
        footprint_utilization=1.0,
        edge_margin_m=0.0,
    )
    geometric = generate_bilateral_view_plan(proxy, intrinsics, planning)
    metrics = CandidateMetrics(1.0, 1.0, 1.0, 0.1, 0.0, 0.1, 1.0)
    evaluated = tuple(
        EvaluatedCandidate(
            candidate,
            CandidateStatus.ENDPOINT_FEASIBLE,
            metrics,
            (),
            np.full(6, index * 0.01),
        )
        for index, candidate in enumerate(geometric.candidates, start=1)
    )
    result = OfflineViewPlanningResult(geometric, FilteredViewPlan(evaluated, ()))
    filtering = ViewFilterConfig(camera_clearance_radius_m=0.01)
    plan = write_view_plan(
        tmp_path / "plan",
        result,
        planning,
        filtering,
        source_initialization=initialization,
    )
    kinematics = write_cs68_kinematics(
        tmp_path / "kinematics.yaml",
        Cs68KinematicsModel(np.zeros(6), np.full(6, 0.2), np.zeros(6), "test"),
    )
    collision = CollisionConfig(
        link_radii_m=(0.01,) * 6,
        camera_tool_radius_m=0.01,
        minimum_joint_positions_rad=(-np.pi,) * 6,
        maximum_joint_positions_rad=(np.pi,) * 6,
        obstacles=(
            CollisionObstacleConfig(
                name="far",
                minimum_m=(10, 10, 10),
                maximum_m=(11, 11, 11),
            ),
        ),
        minimum_clearance_m=0.0,
    )
    return initialization, plan, kinematics, collision, geometric.candidates[0].view_id


def test_path_validation_round_trip_is_rederived_and_non_executable(
    tmp_path: Path,
) -> None:
    initialization, plan, kinematics, collision, view_id = _sources(tmp_path)
    output = write_path_validation(
        tmp_path / "validation",
        (view_id,),
        collision,
        source_plan=plan,
        source_initialization=initialization,
        source_kinematics=kinematics,
    )

    stored = read_path_validation(output)

    assert stored.report.collision_free
    assert stored.report.motion_authorized is False
    assert stored.metadata["motion_authorized"] is False
    assert stored.report.ordered_view_ids == (view_id,)

    metadata_path = output / "path_validation.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["report"]["legs"] = []
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        read_path_validation(output)


def test_path_validation_refuses_unknown_view(tmp_path: Path) -> None:
    initialization, plan, kinematics, collision, _ = _sources(tmp_path)

    with pytest.raises(ValueError, match="unknown views"):
        write_path_validation(
            tmp_path / "validation",
            ("missing",),
            collision,
            source_plan=plan,
            source_initialization=initialization,
            source_kinematics=kinematics,
        )
