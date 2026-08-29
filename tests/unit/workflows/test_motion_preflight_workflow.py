from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import MotionPreflightConfig
from biblade_fusion.planning import (
    BladeSide,
    CandidateMetrics,
    CandidateStatus,
    CandidateView,
    EvaluatedCandidate,
    FilteredViewPlan,
    SurfacePatch,
)
from biblade_fusion.robotics import (
    CollisionCheckResult,
    CollisionCheckStatus,
    JointPathMeshCollisionReport,
    load_es68_flange_t_tcp,
)
from biblade_fusion.workflows import (
    preflight_live_joint_segment,
    preflight_view_sequence_motion,
)


class _FakeEs68Kinematics:
    def forward_kinematics(self, joint_positions_rad) -> np.ndarray:
        del joint_positions_rad
        return np.eye(4)

    def joint_velocity_limits_rad_s(self) -> tuple[float, ...]:
        return (1.0,) * 6


class _FakeEs68Checker:
    model_name = "es68"
    collision_model_id = "fake-es68"
    collision_model_hash = "1" * 64
    robot_geometry_hash = "2" * 64
    motion_model_contract_hash = "3" * 64
    kinematic_model = _FakeEs68Kinematics()

    def check_path(
        self,
        start_joint_positions_rad,
        end_joint_positions_rad,
        *,
        maximum_joint_step_rad: float,
        **proof_contract,
    ) -> JointPathMeshCollisionReport:
        del start_joint_positions_rad, end_joint_positions_rad, proof_contract
        result = CollisionCheckResult(
            CollisionCheckStatus.CLEAR,
            diagnostics={
                "model": "elite_es68",
                "collision_model_id": self.collision_model_id,
                "collision_model_hash": self.collision_model_hash,
                "robot_geometry_hash": self.robot_geometry_hash,
                "motion_model_contract_hash": self.motion_model_contract_hash,
            },
        )
        return JointPathMeshCollisionReport(
            CollisionCheckStatus.CLEAR,
            2,
            None,
            None,
            result,
            maximum_joint_step_rad,
        )


def _filtered_plan(base_t_left_ir: np.ndarray) -> FilteredViewPlan:
    patch = SurfacePatch(
        "patch",
        BladeSide.FRONT,
        0,
        0,
        np.zeros(3),
        np.array((0.0, 0.0, 1.0)),
        (0.02, 0.02),
    )
    candidate = CandidateView(
        "view-0001",
        patch,
        PoseSE3("base", "left_ir", base_t_left_ir),
        0.3,
        (0.1, 0.1),
    )
    metrics = CandidateMetrics(1.0, 1.0, 1.0, 0.3, 0.0, 1.0, 1.0)
    return FilteredViewPlan(
        (
            EvaluatedCandidate(
                candidate,
                CandidateStatus.ENDPOINT_FEASIBLE,
                metrics,
                (),
                np.zeros(6),
            ),
        ),
        (),
    )


def test_view_sequence_blocks_endpoint_that_disagrees_with_es68_fk() -> None:
    target = load_es68_flange_t_tcp().matrix.copy()
    target[0, 3] += 0.01
    hand_eye = HandEyeCalibration(
        PoseSE3("tcp", "left_ir", np.eye(4)),
        "test",
        20,
        0.001,
        0.1,
        Path("hand-eye.yaml"),
        flange_t_left_ir=load_es68_flange_t_tcp().compose(
            PoseSE3.identity("tcp", "left_ir")
        ),
    )

    report = preflight_view_sequence_motion(
        _filtered_plan(target),
        ("view-0001",),
        np.zeros(6),
        MotionPreflightConfig(
            maximum_endpoint_translation_error_m=0.002,
            maximum_endpoint_rotation_error_deg=0.3,
        ),
        hand_eye=hand_eye,
        collision_checker=_FakeEs68Checker(),
    )

    leg = report.legs[0]
    assert leg.endpoint_consistency.status is CollisionCheckStatus.BLOCKED
    assert leg.endpoint_consistency.translation_error_m == 0.01
    assert leg.endpoint_consistency.rotation_error_deg == 0.0
    assert "endpoint_fk_tcp_translation_error_exceeded" in (
        leg.preflight.blocking_reasons
    )
    assert leg.preflight.servoj_stream is None
    assert report.ready_for_approval is False


def test_live_segment_uses_measured_start_and_checks_final_tcp_endpoint() -> None:
    result = preflight_live_joint_segment(
        (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        MotionPreflightConfig(),
        collision_checker=_FakeEs68Checker(),
        occupancy_checker=None,
        final_target=True,
        target_base_t_tcp_matrix=load_es68_flange_t_tcp().matrix,
    )

    assert result.preflight.start_joint_positions_rad[0] == 0.01
    assert result.preflight.goal_joint_positions_rad[0] == 0.02
    assert result.endpoint_consistency is not None
    assert result.endpoint_consistency.status is CollisionCheckStatus.CLEAR
    assert result.ready_for_approval is False
    assert "continuous_swept_mesh_unavailable" in result.preflight.blocking_reasons


def test_intermediate_segment_cannot_claim_a_view_endpoint() -> None:
    with pytest.raises(ValueError, match="must not claim"):
        preflight_live_joint_segment(
            np.zeros(6),
            np.full(6, 0.01),
            MotionPreflightConfig(),
            collision_checker=_FakeEs68Checker(),
            occupancy_checker=None,
            final_target=False,
            target_base_t_tcp_matrix=np.eye(4),
        )
