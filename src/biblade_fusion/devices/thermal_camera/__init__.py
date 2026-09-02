"""Thermal-camera contracts, SDK audit, and fail-closed adapters."""

from biblade_fusion.devices.thermal_camera.base import (
    ThermalCamera,
    ThermalFrame,
    ThermalFrameProvenance,
)
from biblade_fusion.devices.thermal_camera.errors import (
    ThermalCameraCaptureError,
    ThermalCameraConfigurationError,
    ThermalCameraError,
    ThermalCameraUnavailableError,
)
from biblade_fusion.devices.thermal_camera.factory import create_thermal_camera
from biblade_fusion.devices.thermal_camera.null_camera import NullThermalCamera
from biblade_fusion.devices.thermal_camera.tsr605_usb import (
    ThermalSdkAudit,
    ThermalSdkKind,
    Tsr605UsbBackend,
    Tsr605UsbThermalCamera,
    UsbRadiometricCapture,
    UsbRadiometricDeviceIdentity,
    audit_tsr605_usb_sdk,
)

__all__ = [
    "NullThermalCamera",
    "ThermalCamera",
    "ThermalCameraCaptureError",
    "ThermalCameraConfigurationError",
    "ThermalCameraError",
    "ThermalCameraUnavailableError",
    "ThermalFrame",
    "ThermalFrameProvenance",
    "ThermalSdkAudit",
    "ThermalSdkKind",
    "Tsr605UsbBackend",
    "Tsr605UsbThermalCamera",
    "UsbRadiometricCapture",
    "UsbRadiometricDeviceIdentity",
    "audit_tsr605_usb_sdk",
    "create_thermal_camera",
]
