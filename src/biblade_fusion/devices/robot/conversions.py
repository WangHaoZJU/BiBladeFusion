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

