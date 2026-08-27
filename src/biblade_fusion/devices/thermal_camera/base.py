"""Vendor-neutral thermal-camera contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    """A radiometric temperature frame in degrees Celsius."""

    monotonic_time_ns: int
    device_time_ms: float | None
    temperature_c: NDArray[np.float32]
    raw_counts: NDArray[np.uint16] | None = None

    def __post_init__(self) -> None:
        temperature = np.array(self.temperature_c, dtype=np.float32, copy=True)
        if temperature.ndim != 2:
            raise ValueError("Thermal temperature data must be a two-dimensional array")
        if not np.isfinite(temperature).all():
            raise ValueError("Thermal temperature data must be finite")
        if self.monotonic_time_ns < 0:
            raise ValueError("Thermal timestamp must be non-negative")

        raw: NDArray[np.uint16] | None = None
        if self.raw_counts is not None:
            raw = np.array(self.raw_counts, dtype=np.uint16, copy=True)
            if raw.shape != temperature.shape:
                raise ValueError("Thermal raw-count shape must match temperature data")
            raw.setflags(write=False)

        temperature.setflags(write=False)
        object.__setattr__(self, "temperature_c", temperature)
        object.__setattr__(self, "raw_counts", raw)


@runtime_checkable
class ThermalCamera(Protocol):
    """Optional thermal-camera capability.

    ``capture`` returns ``None`` only when the configured implementation explicitly
    represents unavailable hardware, as done by :class:`NullThermalCamera`.
    """

    @property
    def is_available(self) -> bool: ...

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def capture(self) -> ThermalFrame | None: ...
