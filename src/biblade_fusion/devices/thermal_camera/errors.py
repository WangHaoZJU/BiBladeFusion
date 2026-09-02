"""Thermal-camera adapter failures."""


class ThermalCameraError(RuntimeError):
    """Base class for a thermal-camera failure."""


class ThermalCameraConfigurationError(ThermalCameraError):
    """The selected driver or SDK cannot satisfy the radiometric contract."""


class ThermalCameraUnavailableError(ThermalCameraError):
    """The configured physical camera is unavailable or has the wrong identity."""


class ThermalCameraCaptureError(ThermalCameraError):
    """A capture did not produce a valid per-pixel radiometric observation."""
