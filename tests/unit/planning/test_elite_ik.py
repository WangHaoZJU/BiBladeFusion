from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biblade_fusion.calibration import Cs68KinematicsModel, HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import KinematicsConfig
from biblade_fusion.planning import EliteCs68IkChecker, ReachabilityState


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
    hand_eye = HandEyeCalibration(
        PoseSE3.from_rotation_translation("tcp", "left_ir", np.eye(3), [0.1, 0, 0]),
        "test",
        20,
        0.001,
        0.2,
        Path("hand_eye.yaml"),
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
    np.testing.assert_allclose(solver.target[3:], [0.0, 0.0, np.pi / 2.0])
