"""Configuration-to-device composition for optional thermal acquisition."""

from __future__ import annotations

from biblade_fusion.core.settings import ThermalConfig
from biblade_fusion.devices.thermal_camera.base import ThermalCamera
from biblade_fusion.devices.thermal_camera.errors import ThermalCameraConfigurationError
from biblade_fusion.devices.thermal_camera.null_camera import NullThermalCamera
from biblade_fusion.devices.thermal_camera.tsr605_usb import (
    ThermalSdkKind,
    Tsr605UsbBackend,
    Tsr605UsbThermalCamera,
    audit_tsr605_usb_sdk,
)


def create_thermal_camera(
    config: ThermalConfig,
    *,
    backend: Tsr605UsbBackend | None = None,
) -> ThermalCamera:
    """Create the configured adapter, refusing every unreviewed native SDK path."""

    if not config.enabled:
        return NullThermalCamera()
    if config.driver != "tsr605_usb":
        raise ThermalCameraConfigurationError(
            "enabled thermal acquisition requires thermal.driver='tsr605_usb'"
        )
    if config.model.strip().casefold() != "tsr605":
        raise ThermalCameraConfigurationError("the reviewed USB boundary accepts model TSR605 only")
    if config.transport != "usb":
        raise ThermalCameraConfigurationError("TSR605 acquisition is scoped to USB transport")
    if config.serial_number is None:
        raise ThermalCameraConfigurationError(
            "thermal.serial_number must pin the physical TSR605 before capture"
        )

    # An injected backend is the test/review seam.  Production composition deliberately
    # has no fallback to HCNetSDK: the supplied package is the Device Network SDK.
    if backend is None:
        audit = audit_tsr605_usb_sdk(config.sdk_root)
        if audit.kind is ThermalSdkKind.HIKVISION_DEVICE_NETWORK:
            raise ThermalCameraConfigurationError(audit.reason)
        if not audit.compatible:
            raise ThermalCameraConfigurationError(
                f"{audit.reason}; a reviewed TSR605 USB backend is not bundled"
            )
        raise ThermalCameraConfigurationError(
            "TSR605 USB SDK was audited but no reviewed native backend is registered"
        )

    expected_shape = None
    if config.expected_width is not None and config.expected_height is not None:
        expected_shape = (config.expected_height, config.expected_width)
    return Tsr605UsbThermalCamera(
        backend,
        serial_number=config.serial_number,
        capture_timeout_ms=config.capture_timeout_ms,
        expected_shape=expected_shape,
    )
