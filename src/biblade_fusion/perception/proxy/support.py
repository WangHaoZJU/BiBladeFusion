"""Fail-closed base-frame support selection for initial proxy construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biblade_fusion.core.settings import ProxyModelConfig

PROXY_SUPPORT_ALGORITHM = "base_frame_blade_envelope_aabb_v1"


class ProxySupportError(ValueError):
    """The hard-ROI cloud cannot support the configured blade envelope."""


def _bounds_text(lower: NDArray[np.float64], upper: NDArray[np.float64]) -> str:
    return (
        f"min={np.array2string(lower, precision=6, separator=',')}, "
        f"max={np.array2string(upper, precision=6, separator=',')}"
    )


def _readonly_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class ProxySupportSelection:
    """Auditable point membership for the proxy-only support cloud."""

    mask: NDArray[np.bool_]
    frame: str
    envelope_enabled: bool
    envelope_min_m: NDArray[np.float64]
    envelope_max_m: NDArray[np.float64]
    minimum_retained_fraction: float | None
    input_bounds_min_m: NDArray[np.float64]
    input_bounds_max_m: NDArray[np.float64]
    retained_bounds_min_m: NDArray[np.float64]
    retained_bounds_max_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        mask = np.array(self.mask, dtype=np.bool_, copy=True)
        if mask.ndim != 1 or mask.size == 0:
            raise ValueError("Proxy-support mask must be a non-empty vector")
        if not self.frame:
            raise ValueError("Proxy-support frame must be non-empty")
        vectors = {
            "envelope_min_m": self.envelope_min_m,
            "envelope_max_m": self.envelope_max_m,
            "input_bounds_min_m": self.input_bounds_min_m,
            "input_bounds_max_m": self.input_bounds_max_m,
            "retained_bounds_min_m": self.retained_bounds_min_m,
            "retained_bounds_max_m": self.retained_bounds_max_m,
        }
        for name, value in vectors.items():
            object.__setattr__(self, name, _readonly_vector(value, name))
        retained = int(np.count_nonzero(mask))
        if retained == 0:
            raise ValueError("Proxy-support selection cannot be empty")
        if self.envelope_enabled:
            if self.frame != "base":
                raise ValueError("Blade-envelope support must be expressed in base")
            if self.minimum_retained_fraction is None:
                raise ValueError("Enabled blade envelope requires a retained fraction")
        elif self.minimum_retained_fraction is not None:
            raise ValueError("Disabled blade envelope cannot set a retained fraction")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)

    @property
    def input_point_count(self) -> int:
        return int(self.mask.size)

    @property
    def retained_point_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def rejected_point_count(self) -> int:
        return self.input_point_count - self.retained_point_count

    @property
    def retained_fraction(self) -> float:
        return self.retained_point_count / self.input_point_count

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "algorithm": PROXY_SUPPORT_ALGORITHM,
            "frame": self.frame,
            "envelope_enabled": self.envelope_enabled,
            "envelope_min_m": self.envelope_min_m.tolist(),
            "envelope_max_m": self.envelope_max_m.tolist(),
            "minimum_retained_fraction": self.minimum_retained_fraction,
            "input_point_count": self.input_point_count,
            "retained_point_count": self.retained_point_count,
            "rejected_point_count": self.rejected_point_count,
            "retained_fraction": self.retained_fraction,
            "input_bounds_min_m": self.input_bounds_min_m.tolist(),
            "input_bounds_max_m": self.input_bounds_max_m.tolist(),
            "retained_bounds_min_m": self.retained_bounds_min_m.tolist(),
            "retained_bounds_max_m": self.retained_bounds_max_m.tolist(),
        }


def select_proxy_support(
    points_m: ArrayLike,
    config: ProxyModelConfig,
    *,
    frame: str,
) -> ProxySupportSelection:
    """Intersect a hard-ROI cloud with its configured base-frame blade envelope."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ProxySupportError("Proxy-support point cloud must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ProxySupportError("Proxy-support point cloud must contain only finite points")

    input_min = points.min(axis=0)
    input_max = points.max(axis=0)
    envelope_min = config.blade_envelope_min_m
    envelope_max = config.blade_envelope_max_m
    if envelope_min is None or envelope_max is None:
        mask = np.ones(points.shape[0], dtype=np.bool_)
        return ProxySupportSelection(
            mask,
            frame,
            False,
            input_min,
            input_max,
            None,
            input_min,
            input_max,
            input_min,
            input_max,
        )
    if frame != "base":
        raise ProxySupportError("Blade-envelope filtering requires a base-frame cloud")

    lower = np.asarray(envelope_min, dtype=np.float64)
    upper = np.asarray(envelope_max, dtype=np.float64)
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    retained = points[mask]
    retained_count = retained.shape[0]
    retained_fraction = retained_count / points.shape[0]
    if retained_count < config.minimum_points:
        raise ProxySupportError(
            "Blade-envelope intersection retained "
            f"{retained_count}/{points.shape[0]} points; at least {config.minimum_points} "
            "are required; "
            f"input_bounds=({_bounds_text(input_min, input_max)}); "
            f"envelope=({_bounds_text(lower, upper)})"
        )
    minimum_fraction = config.minimum_envelope_retained_fraction
    assert minimum_fraction is not None
    if retained_fraction < minimum_fraction:
        raise ProxySupportError(
            "Blade-envelope intersection retained "
            f"{retained_count}/{points.shape[0]} points ({retained_fraction:.3%}); "
            f"at least {minimum_fraction:.3%} is required; "
            f"input_bounds=({_bounds_text(input_min, input_max)}); "
            f"envelope=({_bounds_text(lower, upper)})"
        )
    return ProxySupportSelection(
        mask,
        frame,
        True,
        lower,
        upper,
        minimum_fraction,
        input_min,
        input_max,
        retained.min(axis=0),
        retained.max(axis=0),
    )
