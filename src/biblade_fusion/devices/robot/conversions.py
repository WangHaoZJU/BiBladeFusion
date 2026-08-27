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


def rpy_xyz_to_matrix(rpy_rad: ArrayLike) -> NDArray[np.float64]:
    """Convert HoloRobot/Elite XYZ roll-pitch-yaw radians to a rotation matrix."""

    rpy = np.asarray(rpy_rad, dtype=np.float64)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("RPY angles must be a finite three-vector")
    roll, pitch, yaw = (float(value) for value in rpy)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy_xyz(rotation: ArrayLike) -> NDArray[np.float64]:
    """Convert a rotation matrix to HoloRobot's XYZ roll-pitch-yaw convention."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-7
    ):
        raise ValueError("Rotation matrix must be orthonormal with determinant +1")
    horizontal = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    pitch = float(np.arctan2(-matrix[2, 0], horizontal))
    if horizontal > 1e-9:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        yaw = 0.0
    result = np.array([roll, pitch, yaw], dtype=np.float64)
    result.setflags(write=False)
    return result


def elite_tcp_pose_to_se3(
    tcp_pose: ArrayLike,
    *,
    parent_frame: str = "base",
    child_frame: str = "tcp",
) -> PoseSE3:
    """Convert Elite ``[x,y,z,rx,ry,rz]`` into a frame-aware pose.

    This intentionally follows the pinned HoloRobot backend: ``rx, ry, rz`` are XYZ
    roll-pitch-yaw angles in radians. It is the project integration contract even when
    other Elite SDK interfaces or documentation use rotation-vector terminology.
    """

    pose = np.asarray(tcp_pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("Elite TCP pose must have shape (6,)")
    return PoseSE3.from_rotation_translation(
        parent_frame,
        child_frame,
        rpy_xyz_to_matrix(pose[3:]),
        pose[:3],
    )


def se3_to_elite_tcp_pose(pose: PoseSE3) -> NDArray[np.float64]:
    """Convert a pose into HoloRobot's Elite ``xyz + RPY(xyz)`` boundary form."""

    result = np.concatenate((pose.translation_m, matrix_to_rpy_xyz(pose.rotation)))
    result.setflags(write=False)
    return result


def se3_to_elite_kdl_pose(pose: PoseSE3) -> NDArray[np.float64]:
    """Convert SE(3) to the Elite KDL plugin's ``xyz + roll/pitch/yaw`` input."""

    result = np.concatenate((pose.translation_m, matrix_to_rpy_xyz(pose.rotation)))
    result.setflags(write=False)
    return result
