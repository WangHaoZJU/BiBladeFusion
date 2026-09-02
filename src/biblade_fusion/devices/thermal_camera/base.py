"""Vendor-neutral thermal-camera contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ThermalFrameProvenance:
    """Identity of the radiometric source that produced one temperature matrix."""

    manufacturer: str
    model: str
    serial_number: str
    transport: str
    sdk_name: str
    sdk_version: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "manufacturer",
            "model",
            "serial_number",
            "transport",
            "sdk_name",
        ):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"Thermal provenance {field_name} must be non-empty")
        if self.sdk_version is not None and not self.sdk_version.strip():
            raise ValueError("Thermal provenance sdk_version must be non-empty when present")


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    """A radiometric temperature frame in degrees Celsius."""

    monotonic_time_ns: int
    device_time_ms: float | None
    temperature_c: NDArray[np.float32]
    raw_counts: NDArray[np.uint16] | None = None
    provenance: ThermalFrameProvenance | None = None

    def __post_init__(self) -> None:
        temperature = np.array(self.temperature_c, dtype=np.float32, copy=True)
        if temperature.ndim != 2:
            raise ValueError("Thermal temperature data must be a two-dimensional array")
        if not np.isfinite(temperature).all():
            raise ValueError("Thermal temperature data must be finite")
        if self.monotonic_time_ns < 0:
            raise ValueError("Thermal timestamp must be non-negative")
        if self.device_time_ms is not None and (
            not np.isfinite(self.device_time_ms) or self.device_time_ms < 0.0
        ):
            raise ValueError("Thermal device timestamp must be finite and non-negative")

        raw: NDArray[np.uint16] | None = None
        if self.raw_counts is not None:
            raw_source = np.asarray(self.raw_counts)
            if raw_source.dtype != np.dtype(np.uint16):
                raise ValueError(
                    "Thermal raw counts must already use uint16; implicit wrapping is forbidden"
                )
            raw = np.array(raw_source, dtype=np.uint16, copy=True)
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
