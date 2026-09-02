from pathlib import Path

from biblade_fusion.core.settings import load_settings
from biblade_fusion.diagnostics.doctor import _check_thermal
from biblade_fusion.diagnostics.types import CheckLevel


def _network_sdk_tree(root: Path) -> None:
    (root / "HCNetSDK.h").write_bytes(b"NET_DVR_Login_V40 NET_DVR_RealPlay_V40")
    (root / "libhcnetsdk.so").write_bytes(b"not loaded")


def test_disabled_thermal_scope_is_safe() -> None:
    result = _check_thermal(load_settings("configs/default.yaml"))

    assert result.level is CheckLevel.PASS
    assert result.details["unknown_blade_motion_scope"] == "blocked"


def test_doctor_rejects_network_sdk_as_enabled_direct_usb_binding(tmp_path: Path) -> None:
    _network_sdk_tree(tmp_path)
    settings = load_settings("configs/default.yaml")
    settings.thermal = settings.thermal.model_copy(
        update={
            "enabled": True,
            "driver": "tsr605_usb",
            "sdk_root": tmp_path,
            "serial_number": "TSR605-UNIT-01",
        }
    )

    result = _check_thermal(settings)

    assert result.level is CheckLevel.FAIL
    assert result.details["kind"] == "hikvision_device_network_sdk"
    assert result.details["compatible"] is False
    assert "Device Network SDK" in result.message
