import numpy as np
import pytest

from biblade_fusion.devices.thermal_camera import NullThermalCamera, ThermalFrame


def test_null_camera_returns_no_fake_data() -> None:
    camera = NullThermalCamera()

    with camera:
        assert camera.is_available is False
        assert camera.is_open is False
        assert camera.capture() is None


def test_thermal_frame_copies_and_freezes_temperature() -> None:
    source = np.full((2, 3), 25.0, dtype=np.float32)
    frame = ThermalFrame(1, None, source)
    source[0, 0] = 99.0

    assert frame.temperature_c[0, 0] == 25.0
    assert frame.temperature_c.flags.writeable is False


def test_thermal_frame_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="finite"):
        ThermalFrame(1, None, np.array([[np.nan]], dtype=np.float32))
