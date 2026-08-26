"""Thermal-camera contracts and placeholder implementation."""

from biblade_fusion.devices.thermal_camera.base import ThermalCamera, ThermalFrame
from biblade_fusion.devices.thermal_camera.null_camera import NullThermalCamera

__all__ = ["NullThermalCamera", "ThermalCamera", "ThermalFrame"]

