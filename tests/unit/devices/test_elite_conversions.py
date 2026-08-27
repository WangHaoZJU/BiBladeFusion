import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot.conversions import (
    elite_tcp_pose_to_se3,
    matrix_to_rotation_vector,
    matrix_to_rpy_xyz,
    rotation_vector_to_matrix,
    rpy_xyz_to_matrix,
    se3_to_elite_kdl_pose,
    se3_to_elite_tcp_pose,
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


def test_elite_tcp_pose_conversion_follows_holorobot_rpy_contract() -> None:
    pose = elite_tcp_pose_to_se3([0.1, 0.2, 0.3, 0.3, -0.4, 0.5])

    assert pose.parent_frame == "base"
    assert pose.child_frame == "tcp"
    np.testing.assert_allclose(pose.translation_m, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose.rotation, rpy_xyz_to_matrix([0.3, -0.4, 0.5]))


def test_rotation_vector_round_trip_including_pi_rotation() -> None:
    for vector in (
        np.array([0.2, -0.3, 0.4]),
        np.array([0.0, np.pi, 0.0]),
        np.array([1e-10, -2e-10, 3e-10]),
    ):
        rotation = rotation_vector_to_matrix(vector)
        recovered = matrix_to_rotation_vector(rotation)
        np.testing.assert_allclose(
            rotation_vector_to_matrix(recovered),
            rotation,
            atol=1e-9,
        )


def test_se3_to_elite_pose_round_trip() -> None:
    original = elite_tcp_pose_to_se3([0.1, 0.2, 0.3, 0.2, -0.3, 0.4])

    encoded = se3_to_elite_tcp_pose(original)
    decoded = elite_tcp_pose_to_se3(encoded)

    np.testing.assert_allclose(decoded.matrix, original.matrix, atol=1e-10)
    assert encoded.flags.writeable is False


def test_rpy_xyz_round_trip() -> None:
    rpy = np.array([0.3, -0.4, 0.5])

    recovered = matrix_to_rpy_xyz(rpy_xyz_to_matrix(rpy))

    np.testing.assert_allclose(recovered, rpy, atol=1e-12)


def test_kdl_pose_conversion_uses_rpy_not_rotation_vector() -> None:
    roll, pitch, yaw = 0.3, -0.4, 0.5
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )
    pose = PoseSE3.from_rotation_translation("base", "tcp", rotation, [0.1, 0.2, 0.3])

    kdl_pose = se3_to_elite_kdl_pose(pose)

    np.testing.assert_allclose(kdl_pose, [0.1, 0.2, 0.3, roll, pitch, yaw])
    np.testing.assert_allclose(kdl_pose, se3_to_elite_tcp_pose(pose))
