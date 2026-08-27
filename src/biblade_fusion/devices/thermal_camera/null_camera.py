"""Explicit no-device thermal adapter."""

from __future__ import annotations

from biblade_fusion.devices.thermal_camera.base import ThermalFrame


class NullThermalCamera:
    """Preserve the thermal interface without fabricating temperature samples."""

    @property
    def is_available(self) -> bool:
        return False

    @property
    def is_open(self) -> bool:
        return False

    def __enter__(self) -> NullThermalCamera:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        """No-op: no thermal device has been configured."""

    def close(self) -> None:
        """No-op: no thermal device has been configured."""

    def capture(self) -> ThermalFrame | None:
        """Return no observation rather than a fake zero-valued temperature image."""

        return None
