import numpy as np

from biblade_fusion.devices.robot.conversions import (
    elite_tcp_pose_to_se3,
    rotation_vector_to_matrix,
)


def test_zero_rotation_vector_is_identity() -> None:
    np.testing.assert_allclose(rotation_vector_to_matrix([0, 0, 0]), np.eye(3))


def test_rotation_vector_uses_axis_angle_not_rpy() -> None:
    rotation = rotation_vector_to_matrix([0, 0, np.pi / 2])

    np.testing.assert_allclose(
        rotation,
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )


def test_elite_tcp_pose_conversion_preserves_frames_and_translation() -> None:
    pose = elite_tcp_pose_to_se3([0.1, 0.2, 0.3, 0, 0, 0])

    assert pose.parent_frame == "base"
    assert pose.child_frame == "tcp"
    np.testing.assert_allclose(pose.translation_m, [0.1, 0.2, 0.3])

