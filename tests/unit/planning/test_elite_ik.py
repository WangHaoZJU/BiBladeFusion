from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.planning import EliteCs68IkChecker, ReachabilityState
from biblade_fusion.planning.collision import cs68_mdh_joint_origins
from biblade_fusion.planning.elite_ik import (
    _forward_pose_and_jacobian,
    _holorobot_near_seed_perturbations,
    _HoloRobotMdhIkSolver,
    _HoloRobotPinocchioIkSolver,
    _so3_error,
)
from biblade_fusion.robotics import Cs68ModelResources, load_es68_flange_t_tcp
from biblade_fusion.robotics.pinocchio_collision import PinocchioCs68Model


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
    assert "HoloRobot analytic MDH" in result.message
    np.testing.assert_allclose(result.joint_positions_rad, joints, atol=1e-12)
    assert ik._loader is None


def test_holorobot_analytic_mdh_jacobian_matches_finite_difference() -> None:
    model = Cs68KinematicsModel(
        np.array([0.0, np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2]),
        np.array([0.0, 0.0, -0.43, -0.37, 0.0, 0.0]),
        np.array([0.17, 0.0, 0.0, 0.15, 0.12, 0.11]),
        "analytic-jacobian-test",
    )
    joints = np.array([0.2, -1.1, 1.0, -0.8, 0.4, 0.15])
    pose, analytic = _forward_pose_and_jacobian(model, joints)
    numeric = np.zeros((6, 6), dtype=np.float64)
    epsilon = 1e-7
    for index in range(6):
        perturbed = joints.copy()
        perturbed[index] += epsilon
        shifted, _ = _forward_pose_and_jacobian(model, perturbed)
        numeric[:3, index] = (shifted.translation_m - pose.translation_m) / epsilon
        numeric[3:, index] = _so3_error(shifted.rotation, pose.rotation) / epsilon

    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=1e-5)


def test_holorobot_mdh_solver_reproduces_a_known_reachable_pose() -> None:
    model = Cs68KinematicsModel(
        np.array([0.0, np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2]),
        np.array([0.0, 0.0, -0.43, -0.37, 0.0, 0.0]),
        np.array([0.17, 0.0, 0.0, 0.15, 0.12, 0.11]),
        "known-pose-test",
    )
    expected_joints = np.array([0.2, -1.1, 1.0, -0.8, 0.4, 0.15])
    target, _ = _forward_pose_and_jacobian(model, expected_joints)
    solution = _HoloRobotMdhIkSolver(model).solve(
        target,
        expected_joints + np.array([0.05, -0.04, 0.03, -0.02, 0.01, -0.05]),
    )

    assert solution is not None
    reproduced, _ = _forward_pose_and_jacobian(model, solution)
    np.testing.assert_allclose(reproduced.translation_m, target.translation_m, atol=1e-4)
    assert np.linalg.norm(_so3_error(target.rotation, reproduced.rotation)) < 1e-3


def test_holorobot_near_seed_sweep_matches_active_mapping_pattern() -> None:
    seed = np.arange(6, dtype=np.float64)

    candidates = _holorobot_near_seed_perturbations(seed)

    assert len(candidates) == 12
    np.testing.assert_allclose(candidates[0] - seed, [0.12, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(candidates[1] - seed, [-0.12, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(candidates[-1] - seed, [-0.06, 0, 0, -0.12, 0, 0])


def test_holorobot_pinocchio_solver_reproduces_reachable_urdf_pose() -> None:
    pinocchio_model = PinocchioCs68Model.from_urdf(
        Cs68ModelResources.packaged().urdf_path
    )
    expected = np.array([0.2, -1.1, 1.0, -0.8, 0.4, 0.15])
    target = PoseSE3("base", "flange", pinocchio_model.forward_kinematics(expected))

    solution = _HoloRobotPinocchioIkSolver(pinocchio_model).solve(
        target,
        expected + np.array([0.05, -0.04, 0.03, -0.02, 0.01, -0.05]),
    )

    assert solution is not None
    np.testing.assert_allclose(
        pinocchio_model.forward_kinematics(solution),
        target.matrix,
        atol=1e-3,
    )


def test_holorobot_pinocchio_seed_sweep_retains_all_distinct_branches(
    monkeypatch,
) -> None:
    pinocchio_model = PinocchioCs68Model.from_urdf(
        Cs68ModelResources.packaged().urdf_path
    )
    solver = _HoloRobotPinocchioIkSolver(pinocchio_model)
    seed = np.array([0.2, -1.1, 1.0, -0.8, 0.4, 0.15])
    calls: list[np.ndarray] = []

    def solve_from_every_seed(_target, controller_seed):
        calls.append(controller_seed.copy())
        return controller_seed.copy()

    monkeypatch.setattr(solver, "_solve_single", solve_from_every_seed)

    solutions = solver.solve_all(PoseSE3.identity("base", "flange"), seed)

    assert len(calls) > 1
    assert len(solutions) > 1
    np.testing.assert_allclose(solutions[0], seed)
    assert any(np.allclose(item, calls[1]) for item in solutions)
