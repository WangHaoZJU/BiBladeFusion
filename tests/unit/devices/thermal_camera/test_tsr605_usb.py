from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.settings import ThermalConfig
from biblade_fusion.devices.thermal_camera import (
    NullThermalCamera,
    ThermalCameraCaptureError,
    ThermalCameraConfigurationError,
    ThermalCameraUnavailableError,
    ThermalSdkKind,
    Tsr605UsbThermalCamera,
    UsbRadiometricCapture,
    UsbRadiometricDeviceIdentity,
    audit_tsr605_usb_sdk,
    create_thermal_camera,
)


class FakeUsbBackend:
    def __init__(
        self,
        *,
        model: str = "TSR605",
        serial_number: str = "TSR605-UNIT-01",
        shape: tuple[int, int] = (3, 4),
    ) -> None:
        self._available = True
        self._open = False
        self._identity = UsbRadiometricDeviceIdentity(
            manufacturer="HIKMICRO",
            model=model,
            serial_number=serial_number,
            sdk_name="official-usb-sdk-test-double",
            sdk_version="test",
        )
        self._shape = shape
        self.opened_serial: str | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def identity(self) -> UsbRadiometricDeviceIdentity | None:
        return self._identity if self._open else None

    def open(self, serial_number: str) -> None:
        self.opened_serial = serial_number
        self._open = True

    def close(self) -> None:
        self._open = False

    def capture_radiometric(self, timeout_ms: int) -> UsbRadiometricCapture:
        assert timeout_ms == 2500
        return UsbRadiometricCapture(
            monotonic_time_ns=123,
            device_time_ms=4.5,
            temperature_c=np.full(self._shape, 26.25, dtype=np.float32),
            raw_counts=np.full(self._shape, 1234, dtype=np.uint16),
        )


class CloseFailingBackend(FakeUsbBackend):
    def close(self) -> None:
        self._open = False
        raise RuntimeError("cleanup failed")


def test_network_sdk_is_identified_and_rejected_without_loading_it(tmp_path: Path) -> None:
    include = tmp_path / "头文件"
    libraries = tmp_path / "库文件"
    docs = tmp_path / "开发文档"
    include.mkdir()
    libraries.mkdir()
    docs.mkdir()
    (include / "HCNetSDK.h").write_bytes(
        b"NET_DVR_Login_V40 NET_DVR_RealPlay_V40 NET_DVR_CaptureJPEGPicture_WithAppendData"
    )
    (libraries / "libhcnetsdk.so").write_bytes(b"not loaded")
    (docs / "设备网络SDK使用手册.chm").write_bytes(b"not parsed")

    audit = audit_tsr605_usb_sdk(tmp_path)

    assert audit.kind is ThermalSdkKind.HIKVISION_DEVICE_NETWORK
    assert audit.compatible is False
    assert "Device Network SDK" in audit.reason
    assert set(audit.evidence) == {
        "头文件/HCNetSDK.h",
        "库文件/libhcnetsdk.so",
        "开发文档/设备网络SDK使用手册.chm",
    }


def test_factory_keeps_disabled_camera_explicitly_empty() -> None:
    camera = create_thermal_camera(ThermalConfig())

    assert isinstance(camera, NullThermalCamera)
    assert camera.capture() is None


def test_factory_refuses_network_sdk_as_usb_backend(tmp_path: Path) -> None:
    (tmp_path / "HCNetSDK.h").write_bytes(b"NET_DVR_Login_V40 NET_DVR_RealPlay_V40")
    (tmp_path / "libhcnetsdk.so").write_bytes(b"not loaded")
    config = ThermalConfig(
        enabled=True,
        driver="tsr605_usb",
        sdk_root=tmp_path,
        serial_number="TSR605-UNIT-01",
    )

    with pytest.raises(ThermalCameraConfigurationError, match="Device Network SDK"):
        create_thermal_camera(config)


def test_reviewed_backend_is_identity_pinned_and_returns_radiometric_frame() -> None:
    backend = FakeUsbBackend()
    camera = Tsr605UsbThermalCamera(
        backend,
        serial_number="TSR605-UNIT-01",
        capture_timeout_ms=2500,
        expected_shape=(3, 4),
    )

    with camera:
        frame = camera.capture()

    assert backend.opened_serial == "TSR605-UNIT-01"
    assert camera.is_open is False
    assert frame.temperature_c.shape == (3, 4)
    assert frame.raw_counts is not None
    assert frame.provenance is not None
    assert frame.provenance.model == "TSR605"
    assert frame.provenance.transport == "usb"


def test_adapter_rejects_wrong_model_and_closes_backend() -> None:
    backend = FakeUsbBackend(model="OTHER")
    camera = Tsr605UsbThermalCamera(
        backend,
        serial_number="TSR605-UNIT-01",
        capture_timeout_ms=2500,
    )

    with pytest.raises(ThermalCameraUnavailableError, match="expected TSR605"):
        camera.open()

    assert backend.is_open is False


def test_adapter_preserves_identity_failure_when_cleanup_also_fails() -> None:
    camera = Tsr605UsbThermalCamera(
        CloseFailingBackend(model="OTHER"),
        serial_number="TSR605-UNIT-01",
        capture_timeout_ms=2500,
    )

    with pytest.raises(ThermalCameraUnavailableError, match="expected TSR605"):
        camera.open()


def test_adapter_rejects_unexpected_radiometric_shape() -> None:
    camera = Tsr605UsbThermalCamera(
        FakeUsbBackend(shape=(2, 2)),
        serial_number="TSR605-UNIT-01",
        capture_timeout_ms=2500,
        expected_shape=(3, 4),
    )

    with camera, pytest.raises(ThermalCameraCaptureError, match="shape mismatch"):
        camera.capture()
