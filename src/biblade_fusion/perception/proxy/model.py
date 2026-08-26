"""Immutable bilateral blade proxy model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.pose import PoseSE3


def _readonly_vector(value: ArrayLike, name: str, *, positive: bool = False) -> NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class BilateralBladeProxy:
    """A conservative oriented box enclosing observed and inferred blade geometry.

    The proxy frame axes are major, minor, and camera-facing normal respectively.
    Its center is the center of the conservative planning volume, not an estimate of
    the physical blade's center of mass.
    """

    frame_T_proxy: PoseSE3
    extents_m: NDArray[np.float64]
    observed_surface_centroid_m: NDArray[np.float64]
    pca_eigenvalues_m2: NDArray[np.float64]
    raw_point_count: int
    finite_point_count: int
    voxel_point_count: int
    camera_normal_cosine: float

    def __post_init__(self) -> None:
        extents = _readonly_vector(self.extents_m, "Proxy extents", positive=True)
        observed_centroid = _readonly_vector(
            self.observed_surface_centroid_m, "Observed surface centroid"
        )
        eigenvalues = _readonly_vector(self.pca_eigenvalues_m2, "PCA eigenvalues")
        if np.any(eigenvalues < 0.0):
            raise ValueError("PCA eigenvalues must be non-negative")
        if not (self.raw_point_count >= self.finite_point_count >= self.voxel_point_count >= 3):
            raise ValueError("Proxy point counts are inconsistent")
        if not 0.0 <= self.camera_normal_cosine <= 1.0:
            raise ValueError("Camera-normal cosine must be in [0, 1]")

        object.__setattr__(self, "extents_m", extents)
        object.__setattr__(self, "observed_surface_centroid_m", observed_centroid)
        object.__setattr__(self, "pca_eigenvalues_m2", eigenvalues)

    @property
    def center_m(self) -> NDArray[np.float64]:
        """Conservative planning-volume center in the parent frame."""

        return self.frame_T_proxy.translation_m

    @property
    def axes(self) -> NDArray[np.float64]:
        """Proxy major, minor, and camera-facing normal axes as columns."""

        return self.frame_T_proxy.rotation

    @property
    def outward_normal(self) -> NDArray[np.float64]:
        """Return the normal pointing from the observed face toward the initial camera."""

        return self.axes[:, 2]

    def corners_m(self) -> NDArray[np.float64]:
        """Return all eight oriented-box corners in the parent frame."""

        signs = np.asarray(list(product((-0.5, 0.5), repeat=3)), dtype=np.float64)
        return self.frame_T_proxy.transform_points(signs * self.extents_m)

    def contains(self, points_m: ArrayLike, tolerance_m: float = 1e-9) -> NDArray[np.bool_]:
        """Test whether parent-frame points lie inside the proxy volume."""

        if tolerance_m < 0.0:
            raise ValueError("Containment tolerance must be non-negative")
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        local_points = self.frame_T_proxy.inverse().transform_points(points)
        return np.all(np.abs(local_points) <= self.extents_m / 2.0 + tolerance_m, axis=1)
