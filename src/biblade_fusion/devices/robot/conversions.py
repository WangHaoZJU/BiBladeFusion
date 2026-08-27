"""Conversions at the Elite SDK boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3


def rotation_vector_to_matrix(rotation_vector: ArrayLike) -> NDArray[np.float64]:
    """Convert an axis-angle rotation vector in radians to a rotation matrix."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("Rotation vector must have shape (3,)")
    if not np.isfinite(vector).all():
        raise ValueError("Rotation vector must be finite")

    theta = float(np.linalg.norm(vector))
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    if theta < 1e-8:
        theta_sq = theta * theta
        a = 1.0 - theta_sq / 6.0
        b = 0.5 - theta_sq / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * skew + b * (skew @ skew)


def matrix_to_rotation_vector(rotation: ArrayLike) -> NDArray[np.float64]:
    """Convert a proper rotation matrix to an axis-angle rotation vector."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-7
    ):
        raise ValueError("Rotation matrix must be orthonormal with determinant +1")

    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew_vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-8:
        return skew_vector / 2.0
    if np.pi - theta < 1e-6:
        symmetric = (matrix + matrix.T) / 2.0
        _, eigenvectors = np.linalg.eigh(symmetric)
        axis = eigenvectors[:, -1]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis = -axis
        return axis * theta
    return skew_vector * (theta / (2.0 * np.sin(theta)))


def elite_tcp_pose_to_se3(
    tcp_pose: ArrayLike,
    *,
    parent_frame: str = "base",
    child_frame: str = "tcp",
) -> PoseSE3:
    """Convert Elite ``[x,y,z,rx,ry,rz]`` into a frame-aware pose.

    Elite RTSI defines ``rx, ry, rz`` as a rotation vector, not RPY Euler angles.
    """

    pose = np.asarray(tcp_pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("Elite TCP pose must have shape (6,)")
    return PoseSE3.from_rotation_translation(
        parent_frame,
        child_frame,
        rotation_vector_to_matrix(pose[3:]),
        pose[:3],
    )


def se3_to_elite_tcp_pose(pose: PoseSE3) -> NDArray[np.float64]:
    """Convert a frame-aware pose into Elite ``[x,y,z,rx,ry,rz]`` form."""

    result = np.concatenate((pose.translation_m, matrix_to_rotation_vector(pose.rotation)))
    result.setflags(write=False)
    return result


def se3_to_elite_kdl_pose(pose: PoseSE3) -> NDArray[np.float64]:
    """Convert SE(3) to the Elite KDL plugin's ``xyz + roll/pitch/yaw`` input."""

    rotation = pose.rotation
    horizontal = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = float(np.arctan2(-rotation[2, 0], horizontal))
    if horizontal > 1e-9:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        yaw = 0.0
    result = np.array([*pose.translation_m, roll, pitch, yaw], dtype=np.float64)
    result.setflags(write=False)
    return result
