import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import biblade_fusion.calibration.stereo_validation as stereo_validation
from biblade_fusion.calibration import (
    CharucoImageDetection,
    RawInfraredStereoFrame,
    StereoValidationAssetSession,
    StereoValidationError,
    StereoValidationThresholds,
    validate_stereo_asset_session,
)
from biblade_fusion.core.settings import StereoRectificationConfig

TARGET = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml")


def _write_calibration(path: Path) -> Path:
    intrinsics = {
        "width": 640,
        "height": 480,
        "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "opencv_distortion_model": "brown_conrady",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    transform = np.eye(4)
    transform[0, 3] = -0.05
    payload = {
        "schema_version": 1,
        "calibration_type": "d435i_ir_stereo_charuco",
        "factory_intrinsics_used": False,
        "left_ir": intrinsics,
        "right_ir": intrinsics,
        "right_ir_T_left_ir": transform.tolist(),
        "baseline_m": 0.05,
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _detections() -> tuple[CharucoImageDetection, CharucoImageDetection]:
    grid_x, grid_y = np.meshgrid(np.arange(6), np.arange(4))
    objects = np.column_stack(
        (
            (grid_x.reshape(-1) - 2.5) * 0.02,
            (grid_y.reshape(-1) - 1.5) * 0.02,
            np.zeros(24),
        )
    ).astype(np.float32)
    matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros(5)
    left_points, _ = cv2.projectPoints(
        objects, np.zeros(3), np.array([0.0, 0.0, 1.0]), matrix, distortion
    )
    right_points, _ = cv2.projectPoints(
        objects, np.zeros(3), np.array([-0.05, 0.0, 1.0]), matrix, distortion
    )
    ids = np.arange(24, dtype=np.int32)
    return (
        CharucoImageDetection(ids, left_points.reshape(-1, 2), objects, marker_count=12),
        CharucoImageDetection(ids, right_points.reshape(-1, 2), objects, marker_count=12),
    )


class _Detector:
    def __init__(self, target) -> None:
        self.target = target
        self.left, self.right = _detections()

    def detect(self, image: np.ndarray) -> CharucoImageDetection | None:
        return self.left if int(image[0, 0]) == 100 else self.right


def _frame(number: int) -> RawInfraredStereoFrame:
    return RawInfraredStereoFrame(
        np.full((480, 640), 100, dtype=np.uint8),
        np.full((480, 640), 110, dtype=np.uint8),
        number,
        number,
        float(number),
        float(number),
        "hardware_clock",
        f"2026-08-27T12:00:{number:02d}+00:00",
    )


def _session(tmp_path: Path) -> StereoValidationAssetSession:
    return StereoValidationAssetSession.create(
        tmp_path / "validations",
        target_path=TARGET,
        calibration_path=_write_calibration(tmp_path / "stereo.yaml"),
        image_size=(640, 480),
        frames_per_second=30,
        serial_number="test-d435i",
        emitter_enabled=False,
        rectification=StereoRectificationConfig(),
        thresholds=StereoValidationThresholds(minimum_accepted_pairs=3),
    )


def test_validation_session_copies_and_binds_fixed_inputs(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_device_info({"serial_number": "test-d435i", "name": "D435I"})
    session.record_pair(_frame(1))

    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_type"] == "d435i_ir_stereo_independent_validation_session"
    assert manifest["calibration_refit_performed"] is False
    assert manifest["fixed_calibration"]["source_path"].endswith("stereo.yaml")
    assert len(manifest["fixed_calibration"]["sha256"]) == 64
    assert (session.root / "configuration/fixed_stereo_calibration.yaml").is_file()
    assert (session.root / "raw_pairs/pair_0000/left_ir.png").is_file()


def test_independent_validation_writes_metrics_and_epipolar_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stereo_validation, "StereoCharucoDetector", _Detector)
    session = _session(tmp_path)
    for number in range(1, 4):
        session.record_pair(_frame(number))

    result = validate_stereo_asset_session(session)

    assert result.metrics.passed
    assert result.metrics.accepted_pair_count == 3
    assert result.metrics.rejected_pair_count == 0
    assert result.metrics.vertical_disparity_rmse_px < 1e-5
    assert result.metrics.left_reprojection_rmse_px < 1e-4
    assert result.metrics.right_reprojection_rmse_px < 1e-4
    assert result.metrics.stereo_transfer_rmse_px < 1e-4
    assert result.report_json.is_file()
    assert result.report_text.is_file()
    assert (
        result.analysis_root
        / "pairs/pair_0000/rectified_epipolar_overlay.png"
    ).is_file()
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["calibration_refit_performed"] is False
    assert report["metrics"]["passed"] is True
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["result"]["metrics"]["passed"] is True
    with pytest.raises(StereoValidationError, match="immutable"):
        session.record_pair(_frame(9))


def test_validation_rejects_tampered_raw_asset(tmp_path: Path) -> None:
    session = _session(tmp_path)
    for number in range(1, 4):
        session.record_pair(_frame(number))
    (session.root / "raw_pairs/pair_0000/left_ir.png").write_bytes(b"tampered")

    with pytest.raises(StereoValidationError, match="checksum mismatch"):
        validate_stereo_asset_session(session)


def test_validation_refuses_stream_resolution_mismatch(tmp_path: Path) -> None:
    calibration = _write_calibration(tmp_path / "stereo.yaml")
    with pytest.raises(StereoValidationError, match="does not match calibration"):
        StereoValidationAssetSession.create(
            tmp_path / "validations",
            target_path=TARGET,
            calibration_path=calibration,
            image_size=(1280, 720),
            frames_per_second=30,
            serial_number=None,
            emitter_enabled=False,
            rectification=StereoRectificationConfig(),
            thresholds=StereoValidationThresholds(),
        )
