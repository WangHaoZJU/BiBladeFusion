from pathlib import Path

import pytest
from pydantic import ValidationError

from biblade_fusion.core.settings import AppSettings, load_settings


def test_default_settings_load_safely() -> None:
    settings = load_settings(Path("configs/default.yaml"))

    assert settings.robot.model == "cs68"
    assert settings.robot.motion_enabled is False
    assert settings.thermal.enabled is False
    assert settings.realsense.infrared_width == 1280
    assert settings.acquisition.max_bracket_ms == 250.0


def test_unknown_configuration_key_is_rejected() -> None:
    raw = {
        "project": {"data_root": "data", "log_root": "logs"},
        "robot": {
            "model": "cs68",
            "sdk_wheel": "/tmp/elite.whl",
            "unexpected": True,
        },
        "realsense": {},
        "thermal": {},
    }

    with pytest.raises(ValidationError, match="unexpected"):
        AppSettings.model_validate(raw)


def test_invalid_robot_ip_is_rejected() -> None:
    raw = {
        "project": {},
        "robot": {"robot_ip": "not-an-ip", "sdk_wheel": "/tmp/elite.whl"},
        "realsense": {},
        "thermal": {},
    }

    with pytest.raises(ValidationError, match="robot_ip"):
        AppSettings.model_validate(raw)
