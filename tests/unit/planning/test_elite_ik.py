from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.planning import EliteCs68IkChecker, ReachabilityState
from biblade_fusion.planning.collision import cs68_mdh_joint_origins
from biblade_fusion.robotics import load_es68_flange_t_tcp


class FakeSolver:
    def __init__(self, succeeds: bool = True) -> None:
        self.succeeds = succeeds
        self.target = None

    def setMDH(self, alpha, a, d) -> None:
        np.testing.assert_allclose(alpha, np.arange(6) * 0.1)

    def setDefaultTimeout(self, timeout: float) -> None:
        assert timeout == 0.05

    def getPositionIK(self, target, near):
        self.target = np.asarray(target)
        np.testing.assert_allclose(near, np.zeros(6))
        return self.succeeds, np.ones(6) * 0.2, SimpleNamespace(kinematic_error="OK")


def checker(solver: FakeSolver) -> EliteCs68IkChecker:
    model = Cs68KinematicsModel(
        np.arange(6) * 0.1,
        np.arange(6) * 0.01,
        np.arange(6) * 0.02,
        "unit-test",
    )
    tcp_t_left_ir = PoseSE3.from_rotation_translation(
        "tcp", "left_ir", np.eye(3), [0.1, 0, 0]
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
        flange_t_left_ir=load_es68_flange_t_tcp().compose(tcp_t_left_ir),
    )
    return EliteCs68IkChecker(
        model,
        hand_eye,
        np.zeros(6),
        KinematicsConfig(),
        solver=solver,
    )


def test_elite_ik_converts_camera_target_to_tcp_and_returns_joints() -> None:
    solver = FakeSolver()
    ik = checker(solver)
    camera_pose = PoseSE3.from_rotation_translation("base", "left_ir", np.eye(3), [0.5, 0, 0])

    result = ik.check(camera_pose)

    assert result.state is ReachabilityState.REACHABLE
    np.testing.assert_allclose(result.joint_positions_rad, np.ones(6) * 0.2)
    np.testing.assert_allclose(solver.target[:3], [0.4, 0, 0])


def test_elite_ik_accepts_unique_generated_view_camera_frame() -> None:
    solver = FakeSolver()
    ik = checker(solver)
    camera_pose = PoseSE3.from_rotation_translation(
        "base",
        "front_r00_c00_left_ir",
        np.eye(3),
        [0.5, 0, 0],
    )

    result = ik.check(camera_pose)

    assert result.state is ReachabilityState.REACHABLE
    np.testing.assert_allclose(solver.target[:3], [0.4, 0, 0])


def test_elite_ik_reports_no_solution_as_unreachable() -> None:
    result = checker(FakeSolver(succeeds=False)).check(PoseSE3.identity("base", "left_ir"))

    assert result.state is ReachabilityState.UNREACHABLE
    assert "no endpoint IK solution" in result.message


def test_elite_ik_passes_rpy_orientation_expected_by_vendor_plugin() -> None:
    solver = FakeSolver()
    ik = checker(solver)
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    camera_pose = PoseSE3.from_rotation_translation(
        "base", "left_ir", rotation, [0.5, 0, 0]
    )

    result = ik.check(camera_pose)

    assert result.state is ReachabilityState.REACHABLE
    np.testing.assert_allclose(
        solver.target[3:],
        [0.0, 0.0, np.pi / 2.0],
        atol=1e-12,
    )


def test_default_elite_ik_uses_holorobot_mdh_solver_without_sdk_plugin() -> None:
    model = Cs68KinematicsModel(
        np.arange(6) * 0.1,
        np.arange(6) * 0.01,
        np.arange(6) * 0.02,
        "unit-test",
    )
    tcp_t_left_ir = PoseSE3.from_rotation_translation(
        "tcp", "left_ir", np.eye(3), [0.1, 0, 0]
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
        flange_t_left_ir=load_es68_flange_t_tcp().compose(tcp_t_left_ir),
    )
    joints = np.zeros(6)
    _, base_t_flange = cs68_mdh_joint_origins(model, joints)
    camera_pose = base_t_flange.compose(hand_eye.require_flange_primary())
    ik = EliteCs68IkChecker(model, hand_eye, joints, KinematicsConfig())

    result = ik.check(camera_pose)

    assert result.state is ReachabilityState.REACHABLE
    assert "HoloRobot MDH" in result.message
    np.testing.assert_allclose(result.joint_positions_rad, joints, atol=1e-12)
    assert ik._loader is None
