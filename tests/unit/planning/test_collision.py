from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    CollisionConfig,
    CollisionObstacleConfig,
)
from biblade_fusion.planning import (
    CollisionValidationError,
    cs68_mdh_joint_origins,
    validate_joint_path_collision,
)


def model(*, a: tuple[float, ...] = (0.2,) * 6) -> Cs68KinematicsModel:
    return Cs68KinematicsModel(np.zeros(6), a, np.zeros(6), "unit-test")


def hand_eye() -> HandEyeCalibration:
    return HandEyeCalibration(
        PoseSE3.from_rotation_translation("tcp", "left_ir", np.eye(3), [0.1, 0, 0]),
        "test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
    )


def config(*, obstacles=()) -> CollisionConfig:
    return CollisionConfig(
        link_radii_m=(0.01,) * 6,
        camera_tool_radius_m=0.01,
        minimum_joint_positions_rad=(-np.pi,) * 6,
        maximum_joint_positions_rad=(np.pi,) * 6,
        obstacles=obstacles,
        require_obstacles=True,
        minimum_clearance_m=0.0,
        maximum_joint_step_rad=0.1,
    )


def test_mdh_origins_follow_vendor_fixed_transform_then_rotz_chain() -> None:
    origins, base_t_tcp = cs68_mdh_joint_origins(
        model(),
        [np.pi / 2, 0, 0, 0, 0, 0],
    )

    np.testing.assert_allclose(origins[1], [0.2, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(origins[2], [0.2, 0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(base_t_tcp.translation_m, [0.2, 1.0, 0.0], atol=1e-12)


def test_collision_validation_is_fail_closed_without_geometry() -> None:
    with pytest.raises(CollisionValidationError, match="joint limits"):
        validate_joint_path_collision(
            np.zeros(6),
            np.zeros(6),
            model(),
            hand_eye(),
            CollisionConfig(),
        )


def test_clear_joint_path_remains_non_executable() -> None:
    far_box = CollisionObstacleConfig(
        name="far",
        minimum_m=(10.0, 10.0, 10.0),
        maximum_m=(11.0, 11.0, 11.0),
    )

    report = validate_joint_path_collision(
        np.zeros(6),
        np.full(6, 0.05),
        model(),
        hand_eye(),
        config(obstacles=(far_box,)),
    )

    assert report.sample_count == 2
    assert report.collision_free
    assert report.motion_authorized is False


def test_continuous_sampling_detects_mid_path_workcell_collision() -> None:
    obstacle = CollisionObstacleConfig(
        name="midpoint_fixture",
        minimum_m=(0.45, -0.05, -0.05),
        maximum_m=(0.55, 0.05, 0.05),
    )
    path_config = config(obstacles=(obstacle,)).model_copy(
        update={"maximum_joint_step_rad": 0.05}
    )

    report = validate_joint_path_collision(
        [-np.pi / 2, 0, 0, 0, 0, 0],
        [np.pi / 2, 0, 0, 0, 0, 0],
        model(a=(0.0, 1.0, 0.2, 0.2, 0.2, 0.2)),
        hand_eye(),
        path_config,
    )

    workcell = tuple(item for item in report.findings if item.kind == "workcell_collision")
    assert workcell
    assert any(0.0 < item.path_fraction < 1.0 for item in workcell)


def test_joint_limit_violation_is_reported() -> None:
    far_box = CollisionObstacleConfig(
        name="far",
        minimum_m=(10.0, 10.0, 10.0),
        maximum_m=(11.0, 11.0, 11.0),
    )

    report = validate_joint_path_collision(
        np.zeros(6),
        [4.0, 0, 0, 0, 0, 0],
        model(),
        hand_eye(),
        config(obstacles=(far_box,)),
    )

    assert any(item.kind == "joint_limit" for item in report.findings)
