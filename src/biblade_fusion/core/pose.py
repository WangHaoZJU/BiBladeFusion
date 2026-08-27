"""Frame-aware rigid transformations.

The project uses ``parent_T_child`` notation throughout. A pose maps coordinates from
``child_frame`` into ``parent_frame``. Translation is always expressed in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ATOL = 1e-7


@dataclass(frozen=True, slots=True)
class PoseSE3:
    """An immutable, frame-aware homogeneous rigid transform."""

    parent_frame: str
    child_frame: str
    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.parent_frame or not self.child_frame:
            raise ValueError("Pose frame names must be non-empty")

        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4):
            raise ValueError(f"Pose matrix must have shape (4, 4), got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Pose matrix must contain only finite values")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=_ATOL):
            raise ValueError("Pose matrix must have homogeneous last row [0, 0, 0, 1]")

        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ATOL):
            raise ValueError("Pose rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ATOL):
            raise ValueError("Pose rotation determinant must be +1")

        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls, parent_frame: str, child_frame: str | None = None) -> PoseSE3:
        """Create an identity transform between two named frames."""

        return cls(parent_frame, child_frame or parent_frame, np.eye(4))

    @classmethod
    def from_rotation_translation(
        cls,
        parent_frame: str,
        child_frame: str,
        rotation: ArrayLike,
        translation_m: ArrayLike,
    ) -> PoseSE3:
        """Create a pose from a 3x3 rotation and a three-vector translation."""

        rotation_array = np.asarray(rotation, dtype=np.float64)
        translation_array = np.asarray(translation_m, dtype=np.float64)
        if rotation_array.shape != (3, 3):
            raise ValueError("Rotation must have shape (3, 3)")
        if translation_array.shape != (3,):
            raise ValueError("Translation must have shape (3,)")

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_array
        matrix[:3, 3] = translation_array
        return cls(parent_frame, child_frame, matrix)

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Return a copy of the 3x3 rotation matrix."""

        return self.matrix[:3, :3].copy()

    @property
    def translation_m(self) -> NDArray[np.float64]:
        """Return a copy of the translation in metres."""

        return self.matrix[:3, 3].copy()

    def inverse(self) -> PoseSE3:
        """Invert the transform and swap its frame names."""

        rotation_t = self.matrix[:3, :3].T
        translation = -rotation_t @ self.matrix[:3, 3]
        return PoseSE3.from_rotation_translation(
            self.child_frame,
            self.parent_frame,
            rotation_t,
            translation,
        )

    def compose(self, other: PoseSE3) -> PoseSE3:
        """Compose ``parent_T_child`` with ``child_T_grandchild``."""

        if self.child_frame != other.parent_frame:
            raise ValueError(
                "Cannot compose poses with disconnected frames: "
                f"{self.parent_frame}_T_{self.child_frame} and "
                f"{other.parent_frame}_T_{other.child_frame}"
            )
        return PoseSE3(self.parent_frame, other.child_frame, self.matrix @ other.matrix)

    def transform_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform one point ``(3,)`` or a point array ``(..., 3)``."""

        point_array = np.asarray(points, dtype=np.float64)
        if point_array.shape == () or point_array.shape[-1] != 3:
            raise ValueError("Points must have shape (3,) or (..., 3)")
        return point_array @ self.matrix[:3, :3].T + self.matrix[:3, 3]
