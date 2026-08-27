import numpy as np
import pytest

from biblade_fusion.robotics import (
    Es68KinematicModel,
    Es68ModelResources,
    load_es68_flange_t_tcp,
)


def test_es68_uses_dedicated_holorobot_709_pose_resources() -> None:
    resources = Es68ModelResources.packaged()

    assert "/config/es68/" in str(resources.kinematics_yaml)
    assert "709 poses" in resources.kinematics_yaml.read_text(encoding="utf-8")


def test_es68_calibrated_fk_is_stable_at_zero() -> None:
    transform = Es68KinematicModel.from_resources().forward_kinematics((0.0,) * 6)

    np.testing.assert_allclose(
        transform[:3, 3],
        (-0.723641989842, -0.229057201937, 0.064656406643),
        atol=1e-10,
    )
    np.testing.assert_allclose(transform[3], (0.0, 0.0, 0.0, 1.0))


def test_es68_flange_tcp_offset_matches_holorobot_fit() -> None:
    offset = load_es68_flange_t_tcp()

    assert offset.parent_frame == "flange"
    assert offset.child_frame == "tcp"
    assert offset.translation_m[2] == pytest.approx(0.0025845477600085483)


def test_es68_joint_zero_offsets_follow_controller_plus_offset_convention() -> None:
    offsets = (0.01, -0.02, 0.03, -0.04, 0.05, -0.06)
    corrected = Es68KinematicModel.from_resources(joint_zero_offsets_rad=offsets)
    plain = Es68KinematicModel.from_resources()
    controller = np.array((0.2, -0.3, 0.4, -0.5, 0.6, -0.7))

    np.testing.assert_allclose(
        corrected.forward_kinematics(controller),
        plain.forward_kinematics(controller + np.asarray(offsets)),
        atol=1e-12,
    )
