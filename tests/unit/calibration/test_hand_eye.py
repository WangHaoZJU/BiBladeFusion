from pathlib import Path

import numpy as np
import pytest
import yaml

from biblade_fusion.calibration import HandEyeCalibrationError, load_hand_eye_calibration
from biblade_fusion.core.settings import HandEyeConfig


def write_calibration(path: Path, **quality_overrides: object) -> None:
    quality = {
        "sample_count": 20,
        "translation_rmse_m": 0.001,
        "rotation_rmse_deg": 0.2,
    }
    quality.update(quality_overrides)
    payload = {
        "schema_version": 1,
        "parent_frame": "tcp",
        "child_frame": "left_ir",
        "method": "Park-Martin",
        "matrix": np.eye(4).tolist(),
        "quality": quality,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def config(path: Path, **overrides: object) -> HandEyeConfig:
    values = {"calibration_path": path}
    values.update(overrides)
    return HandEyeConfig.model_validate(values)


def test_load_hand_eye_calibration_with_quality_gate(tmp_path: Path) -> None:
    path = tmp_path / "hand_eye.yaml"
    write_calibration(path)

    calibration = load_hand_eye_calibration(config(path))

    assert calibration.method == "Park-Martin"
    assert calibration.sample_count == 20
    assert calibration.tcp_t_left_ir.parent_frame == "tcp"
    assert calibration.tcp_t_left_ir.child_frame == "left_ir"
    np.testing.assert_allclose(calibration.tcp_t_left_ir.matrix, np.eye(4))


def test_hand_eye_rejects_quality_above_threshold(tmp_path: Path) -> None:
    path = tmp_path / "hand_eye.yaml"
    write_calibration(path, translation_rmse_m=0.004)

    with pytest.raises(HandEyeCalibrationError, match="translation RMSE"):
        load_hand_eye_calibration(config(path))


def test_hand_eye_rejects_wrong_frame_convention(tmp_path: Path) -> None:
    path = tmp_path / "hand_eye.yaml"
    write_calibration(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["parent_frame"] = "base"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(HandEyeCalibrationError, match="parent frame"):
        load_hand_eye_calibration(config(path))


def test_hand_eye_requires_configured_path() -> None:
    with pytest.raises(HandEyeCalibrationError, match="not configured"):
        load_hand_eye_calibration(HandEyeConfig())
