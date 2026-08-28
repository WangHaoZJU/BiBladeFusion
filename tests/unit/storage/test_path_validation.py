import json
from datetime import datetime, timedelta
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
    HandEyeConfig,
    KinematicsConfig,
    MotionPreflightConfig,
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
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import (
    read_motion_preflight,
    read_path_validation,
    write_initialization,
    write_motion_preflight,
    write_path_validation,
    write_view_plan,
)
from biblade_fusion.workflows import (
    AuthoritativeRobotPose,
    InitialObservation,
    OfflineViewPlanningResult,
)


def _sources(tmp_path: Path):
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    flange_t_tcp = load_es68_flange_t_tcp()
    flange_t_left_ir = base_t_flange.inverse().compose(
        PoseSE3.identity("base", "left_ir")
    )
    authority = AuthoritativeRobotPose(
        base_t_flange,
        base_t_flange.compose(flange_t_tcp),
        base_t_flange.compose(flange_t_tcp),
        0.0,
        0.0,
        0.002,
        0.3,
        (0.0,) * 6,
    )
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
        pose_authority=authority,
    )
    hand_eye_source = tmp_path / "hand_eye.yaml"
    tcp_t_left_ir = flange_t_tcp.inverse().compose(flange_t_left_ir)
    hand_eye_source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "test",
                "matrix": flange_t_left_ir.matrix.tolist(),
                "derived_runtime": {
                    "tcp_T_left_ir": tcp_t_left_ir.matrix.tolist(),
                },
                "quality": {
                    "sample_count": 20,
                    "translation_rmse_m": 0.001,
                    "rotation_rmse_deg": 0.2,
                    "rotation_span_deg": 45.0,
                    "translation_span_m": 0.1,
                    "rotation_axis_diversity": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "test",
        20,
        0.001,
        0.2,
        hand_eye_source,
        flange_t_left_ir=flange_t_left_ir,
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
        KinematicsConfig(),
        HandEyeConfig(),
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
    kinematics = write_cs68_kinematics(
        tmp_path / "kinematics.yaml",
        Cs68KinematicsModel(np.zeros(6), np.full(6, 0.2), np.zeros(6), "test"),
    )
    plan = write_view_plan(
        tmp_path / "plan",
        result,
        planning,
        filtering,
        source_initialization=initialization,
        source_kinematics=kinematics,
        joint_zero_offsets_rad=(0.0,) * 6,
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
    assert stored.metadata["schema_version"] == 1
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


def test_path_validation_refuses_different_kinematics_artifact(tmp_path: Path) -> None:
    initialization, plan, _, collision, view_id = _sources(tmp_path)
    other_kinematics = write_cs68_kinematics(
        tmp_path / "other_kinematics.yaml",
        Cs68KinematicsModel(
            np.zeros(6), np.full(6, 0.25), np.zeros(6), "different-controller"
        ),
    )

    with pytest.raises(ValueError, match="different kinematics"):
        write_path_validation(
            tmp_path / "validation",
            (view_id,),
            collision,
            source_plan=plan,
            source_initialization=initialization,
            source_kinematics=other_kinematics,
        )


def test_motion_preflight_without_occupancy_is_rederived_as_blocked(tmp_path: Path) -> None:
    initialization, plan, _, collision, view_id = _sources(tmp_path)
    output = write_motion_preflight(
        tmp_path / "motion_preflight",
        (view_id,),
        MotionPreflightConfig(maximum_joint_step_rad=0.02),
        collision,
        source_plan=plan,
        source_initialization=initialization,
        joint_zero_offsets_rad=(0.0,) * 6,
        execution_freshness_margin_s=1.25,
    )

    stored = read_motion_preflight(output)

    assert stored.report.ready_for_approval is False
    assert stored.report.motion_authorized is False
    assert stored.metadata["motion_authorized"] is False
    assert stored.metadata["evaluated_at_utc"] == stored.metadata["created_at_utc"]
    assert stored.report.evaluated_at_utc == stored.metadata["evaluated_at_utc"]
    assert (
        stored.report.legs[0].preflight.diagnostics["evaluated_at_utc"]
        == stored.metadata["evaluated_at_utc"]
    )
    assert stored.metadata["configuration"]["joint_zero_offsets_rad"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert (
        stored.metadata["configuration"]["execution_freshness_margin_s"]
        == 1.25
    )
    assert stored.report.legs[0].preflight.servoj_stream is None
    assert stored.report.legs[0].preflight.blocking_reasons == (
        "es68_d435i_collision_model_unavailable:FileNotFoundError",
        "endpoint_pose_consistency_es68_model_unavailable",
    )
    assert stored.report.cost.estimated_servoj_duration_s == 0.0
    assert stored.report.cost.total_joint_travel_l1_rad == pytest.approx(0.06)
    assert np.asarray(stored.report.legs[0].goal_base_t_tcp_matrix).shape == (4, 4)

    metadata_path = output / "motion_preflight.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema 3"):
        read_motion_preflight(output)

    payload["schema_version"] = 5
    created = datetime.fromisoformat(payload["created_at_utc"])
    payload["created_at_utc"] = (created + timedelta(seconds=1)).isoformat()
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="creation and evaluation instants"):
        read_motion_preflight(output)

    payload["created_at_utc"] = created.isoformat()
    payload["report"]["legs"] = []
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        read_motion_preflight(output)


def test_motion_preflight_refuses_offsets_different_from_view_plan(
    tmp_path: Path,
) -> None:
    initialization, plan, _, collision, view_id = _sources(tmp_path)

    with pytest.raises(ValueError, match="joint-zero offsets differ"):
        write_motion_preflight(
            tmp_path / "motion_preflight",
            (view_id,),
            MotionPreflightConfig(maximum_joint_step_rad=0.02),
            collision,
            source_plan=plan,
            source_initialization=initialization,
            joint_zero_offsets_rad=(0.001, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


def test_motion_preflight_requires_workcell_obstacles(tmp_path: Path) -> None:
    initialization, plan, _, _, view_id = _sources(tmp_path)

    with pytest.raises(ValueError, match="workcell obstacle"):
        write_motion_preflight(
            tmp_path / "motion_preflight",
            (view_id,),
            MotionPreflightConfig(),
            CollisionConfig(require_obstacles=True),
            source_plan=plan,
            source_initialization=initialization,
        )
