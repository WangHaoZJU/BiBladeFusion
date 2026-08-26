from pathlib import Path

import numpy as np

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    PointCloudConfig,
    ProxyModelConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.planning import BladeSide
from biblade_fusion.storage import (
    SessionReader,
    SessionWriter,
    read_initialization,
    read_view_plan,
    write_initialization,
    write_view_plan,
)
from biblade_fusion.workflows import initialize_native_depth, plan_initial_observation


def make_bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(20, 20, 50.0, 50.0, 9.5, 9.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        0.001,
        intrinsics,
        PoseSE3.identity("left_ir", "depth"),
    )
    image = np.zeros((20, 20), dtype=np.uint8)
    stereo = StereoFrame(
        1_050_000_000,
        1,
        10.0,
        10.0,
        image,
        image,
        np.full((20, 20), 500, dtype=np.uint16),
        calibration,
    )
    state = RobotState(
        1_000_000_000,
        1.0,
        np.zeros(6),
        PoseSE3.identity("base", "tcp"),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        "seed",
        0,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0, 0, 0, 0, 0),
    )


def test_raw_session_to_non_executable_bilateral_plan(tmp_path: Path) -> None:
    settings = load_settings("configs/default.yaml")
    with SessionWriter.create(tmp_path, settings, label="integration") as writer:
        writer.write_bundle(make_bundle())
    bundle = SessionReader(writer.path).load_bundle("seed")

    hand_eye = HandEyeCalibration(
        PoseSE3.identity("tcp", "left_ir"),
        "integration-test",
        20,
        0.001,
        0.2,
        tmp_path / "hand_eye.yaml",
    )
    point_config = PointCloudConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_points=100,
    )
    proxy_config = ProxyModelConfig(
        voxel_size_m=0.0001,
        minimum_points=100,
        estimated_thickness_m=0.01,
    )
    mask = np.ones((20, 20), dtype=bool)
    observation = initialize_native_depth(
        bundle,
        mask,
        hand_eye,
        point_config,
        proxy_config,
    )
    initialization_path = write_initialization(
        tmp_path / "initialization",
        observation,
        mask,
        hand_eye,
        point_config,
        proxy_config,
        source_session=writer.path,
    )
    stored_initialization = read_initialization(initialization_path)

    planning_config = ViewPlanningConfig(
        standoff_distance_m=0.2,
        overlap_fraction=0.3,
        footprint_utilization=0.8,
        edge_margin_m=0.005,
    )
    filter_config = ViewFilterConfig(camera_clearance_radius_m=0.05)
    planning_result = plan_initial_observation(
        stored_initialization.observation,
        planning_config,
        filter_config,
    )
    plan_path = write_view_plan(
        tmp_path / "view_plan",
        planning_result,
        planning_config,
        filter_config,
        source_initialization=initialization_path,
    )
    stored_plan = read_view_plan(plan_path)

    assert stored_plan.result.geometric_plan.for_side(BladeSide.FRONT)
    assert stored_plan.result.geometric_plan.for_side(BladeSide.BACK)
    assert all(
        item.status.value == "geometry_only"
        for item in stored_plan.result.filtered_plan.accepted
    )
    assert stored_plan.result.motion_authorized is False
