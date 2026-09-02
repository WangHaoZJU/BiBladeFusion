"""Fail-closed TSR605 USB adapter boundary and local SDK audit.

The repository does not bundle a vendor USB binding.  This module deliberately keeps
the project-side contract separate from any vendor function names so that the Device
Network SDK cannot accidentally be treated as a local USB radiometric SDK.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.devices.thermal_camera.base import (
    ThermalFrame,
    ThermalFrameProvenance,
)
from biblade_fusion.devices.thermal_camera.errors import (
    ThermalCameraCaptureError,
    ThermalCameraUnavailableError,
)


class ThermalSdkKind(StrEnum):
    """Locally identifiable SDK package families."""

    MISSING = "missing"
    HIKVISION_DEVICE_NETWORK = "hikvision_device_network_sdk"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True, slots=True)
class ThermalSdkAudit:
    """Read-only classification of a candidate TSR605 SDK directory."""

    sdk_root: str | None
    kind: ThermalSdkKind
    compatible: bool
    reason: str
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "sdk_root": self.sdk_root,
            "kind": self.kind.value,
            "compatible": self.compatible,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


def _relative_evidence(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def audit_tsr605_usb_sdk(sdk_root: str | Path | None) -> ThermalSdkAudit:
    """Classify an SDK tree without loading native code or touching a USB device.

    Only a package with a reviewed TSR605 USB radiometric binding may eventually be
    marked compatible.  The supplied HCNetSDK is detected explicitly and rejected: its
    public entry points are network-device login/preview APIs, even though its large
    header also contains structures mentioning USB peripherals and thermal products.
    """

    if sdk_root is None:
        return ThermalSdkAudit(
            None,
            ThermalSdkKind.MISSING,
            False,
            "thermal.sdk_root is not configured",
        )

    root = Path(sdk_root).expanduser()
    if not root.exists():
        return ThermalSdkAudit(
            str(root),
            ThermalSdkKind.MISSING,
            False,
            "configured thermal SDK path does not exist",
        )
    if not root.is_dir():
        return ThermalSdkAudit(
            str(root),
            ThermalSdkKind.UNRECOGNIZED,
            False,
            "configured thermal SDK path is not a directory",
        )

    files = tuple(path for path in root.rglob("*") if path.is_file())
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name.casefold(), []).append(path)

    evidence: list[str] = []
    network_markers = {
        "hcnetsdk.h",
        "libhcnetsdk.so",
        "设备网络sdk使用手册.chm",
    }
    matched_markers = sorted(network_markers.intersection(by_name))
    for marker in matched_markers:
        evidence.append(_relative_evidence(root, by_name[marker][0]))

    network_header = False
    for header in by_name.get("hcnetsdk.h", ()):
        try:
            payload = header.read_bytes()
        except OSError:
            continue
        if b"NET_DVR_Login_V40" in payload and b"NET_DVR_RealPlay_V40" in payload:
            network_header = True
            break

    if network_header or len(matched_markers) >= 2:
        return ThermalSdkAudit(
            str(root),
            ThermalSdkKind.HIKVISION_DEVICE_NETWORK,
            False,
            (
                "this is the Hikvision Device Network SDK; it requires a logged-in "
                "network device and does not expose a reviewed local TSR605 USB "
                "radiometric-device binding"
            ),
            tuple(evidence),
        )

    return ThermalSdkAudit(
        str(root),
        ThermalSdkKind.UNRECOGNIZED,
        False,
        (
            "SDK package is not recognized by the reviewed TSR605 USB adapter; "
            "do not load it until its official headers and examples are mapped and tested"
        ),
        tuple(evidence),
    )


@dataclass(frozen=True, slots=True)
class UsbRadiometricDeviceIdentity:
    """Identity reported by a reviewed USB radiometric backend."""

    manufacturer: str
    model: str
    serial_number: str
    sdk_name: str
    sdk_version: str | None = None


@dataclass(frozen=True, slots=True)
class UsbRadiometricCapture:
    """One backend-owned per-pixel Celsius observation.

    ``monotonic_time_ns`` is required to use the host process monotonic clock domain;
    a device clock belongs separately in ``device_time_ms``.  A future native backend
    must establish that contract from the vendor callback rather than copying a device
    counter into the host-timestamp field.
    """

    monotonic_time_ns: int
    device_time_ms: float | None
    temperature_c: NDArray[np.float32]
    raw_counts: NDArray[np.uint16] | None = None


@runtime_checkable
class Tsr605UsbBackend(Protocol):
    """Narrow seam to be implemented from the official TSR605 USB SDK only."""

    @property
    def is_available(self) -> bool: ...

    @property
    def is_open(self) -> bool: ...

    @property
    def identity(self) -> UsbRadiometricDeviceIdentity | None: ...

    def open(self, serial_number: str) -> None: ...

    def close(self) -> None: ...

    def capture_radiometric(self, timeout_ms: int) -> UsbRadiometricCapture: ...


class Tsr605UsbThermalCamera:
    """Validate a reviewed backend before exposing it as a project thermal camera."""

    def __init__(
        self,
        backend: Tsr605UsbBackend,
        *,
        serial_number: str,
        capture_timeout_ms: int,
        expected_shape: tuple[int, int] | None = None,
    ) -> None:
        if not serial_number.strip():
            raise ValueError("TSR605 serial number must be pinned")
        if capture_timeout_ms <= 0:
            raise ValueError("TSR605 capture timeout must be positive")
        if expected_shape is not None and any(value <= 0 for value in expected_shape):
            raise ValueError("TSR605 expected shape must contain positive dimensions")
        self._backend = backend
        self._serial_number = serial_number
        self._capture_timeout_ms = capture_timeout_ms
        self._expected_shape = expected_shape
        self._identity: UsbRadiometricDeviceIdentity | None = None

    @property
    def is_available(self) -> bool:
        return bool(self._backend.is_available)

    @property
    def is_open(self) -> bool:
        return bool(self._backend.is_open and self._identity is not None)

    def __enter__(self) -> Tsr605UsbThermalCamera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self.is_open:
            return
        if not self._backend.is_available:
            raise ThermalCameraUnavailableError("configured TSR605 USB device is unavailable")
        try:
            self._backend.open(self._serial_number)
            if not self._backend.is_open:
                raise ThermalCameraUnavailableError(
                    "TSR605 USB backend returned from open without an open device"
                )
            identity = self._backend.identity
            if identity is None:
                raise ThermalCameraUnavailableError(
                    "TSR605 USB backend opened without reporting device identity"
                )
            if identity.model.strip().casefold() != "tsr605":
                raise ThermalCameraUnavailableError(
                    f"expected TSR605 but backend opened {identity.model!r}"
                )
            if identity.serial_number != self._serial_number:
                raise ThermalCameraUnavailableError(
                    "TSR605 serial mismatch: configured "
                    f"{self._serial_number!r}, opened {identity.serial_number!r}"
                )
            self._identity = identity
        except ThermalCameraUnavailableError:
            self._identity = None
            # Cleanup is best effort here; it must not replace the concrete
            # identity/model/serial validation failure that blocked the device.
            with suppress(Exception):
                self._backend.close()
            raise
        except Exception as exc:
            self._identity = None
            with suppress(Exception):
                self._backend.close()
            raise ThermalCameraUnavailableError(
                f"failed to open the pinned TSR605 USB device: {exc}"
            ) from exc

    def close(self) -> None:
        try:
            self._backend.close()
        finally:
            self._identity = None

    def capture(self) -> ThermalFrame:
        identity = self._identity
        if not self.is_open or identity is None:
            raise ThermalCameraCaptureError("TSR605 USB camera is not open")
        try:
            sample = self._backend.capture_radiometric(self._capture_timeout_ms)
            frame = ThermalFrame(
                monotonic_time_ns=sample.monotonic_time_ns,
                device_time_ms=sample.device_time_ms,
                temperature_c=sample.temperature_c,
                raw_counts=sample.raw_counts,
                provenance=ThermalFrameProvenance(
                    manufacturer=identity.manufacturer,
                    model=identity.model,
                    serial_number=identity.serial_number,
                    transport="usb",
                    sdk_name=identity.sdk_name,
                    sdk_version=identity.sdk_version,
                ),
            )
        except ThermalCameraCaptureError:
            raise
        except Exception as exc:
            raise ThermalCameraCaptureError(f"TSR605 radiometric capture failed: {exc}") from exc
        if self._expected_shape is not None and frame.temperature_c.shape != self._expected_shape:
            raise ThermalCameraCaptureError(
                "TSR605 radiometric shape mismatch: expected "
                f"{self._expected_shape}, received {frame.temperature_c.shape}"
            )
        return frame
