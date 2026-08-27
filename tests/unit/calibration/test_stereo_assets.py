import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.calibration import (
    CharucoImageDetection,
    LatestStereoFrameMailbox,
    RawInfraredStereoFrame,
    StereoCalibrationAssetError,
    StereoCalibrationAssetSession,
    StereoCharucoBoard,
)

TARGET = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml")


def _frame(value: int, number: int) -> RawInfraredStereoFrame:
    return RawInfraredStereoFrame(
        np.full((48, 64), value, dtype=np.uint8),
        np.full((48, 64), value, dtype=np.uint8),
        number,
        number,
        float(number) * 10.0,
        float(number) * 10.0,
        "hardware_clock",
        f"2026-08-27T12:00:{number:02d}+00:00",
    )


def _session(tmp_path: Path) -> StereoCalibrationAssetSession:
    return StereoCalibrationAssetSession.create(
        tmp_path / "calibrations",
        target_path=TARGET,
        image_size=(64, 48),
        frames_per_second=30,
        serial_number="test-d435i",
        emitter_enabled=False,
    )


class _Detector:
    def __init__(self) -> None:
        self.target = StereoCharucoBoard.read(TARGET)
        ids = np.arange(24, dtype=np.int32)
        points = np.column_stack((ids % 6, ids // 6)).astype(np.float32)
        objects = np.column_stack((points * 0.02, np.zeros(24))).astype(np.float32)
        self.detection = CharucoImageDetection(ids, points, objects, marker_count=12)

    def detect(self, image: np.ndarray) -> CharucoImageDetection | None:
        return self.detection if int(image[0, 0]) else None


def test_mailbox_retains_latest_pair_without_queue_growth() -> None:
    mailbox = LatestStereoFrameMailbox()
    first = _frame(10, 1)
    latest = _frame(20, 2)
    assert mailbox.publish(first)
    assert not mailbox.publish(latest)
    assert mailbox.take_for_preview() is latest
    assert mailbox.publish(first)
    assert mailbox.snapshot() is first


def test_session_creates_unique_append_only_raw_assets(tmp_path: Path) -> None:
    session = _session(tmp_path)
    another = _session(tmp_path)
    assert session.root != another.root
    assert session.root.parent == tmp_path / "calibrations"
    assert (session.root / "configuration" / "charuco_target.yaml").is_file()

    session.record_device_info(
        {
            "serial_number": "test-d435i",
            "name": "Intel RealSense D435I",
            "firmware_version": "test",
        }
    )
    assert session.record_pair(_frame(120, 1)) == "pair_0000"
    assert session.record_pair(_frame(130, 2)) == "pair_0001"
    with pytest.raises(StereoCalibrationAssetError, match="cannot be changed"):
        session.record_device_info({"serial_number": "test-d435i"})
    with pytest.raises(StereoCalibrationAssetError, match="already part"):
        session.record_pair(_frame(140, 2))

    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["factory_intrinsics_used"] is False
    assert manifest["factory_stereo_extrinsics_used"] is False
    assert manifest["device"]["name"] == "Intel RealSense D435I"
    assert manifest["status"] == "capturing"
    assert [item["pair_id"] for item in manifest["raw_pairs"]] == [
        "pair_0000",
        "pair_0001",
    ]
    assert (session.root / "raw_pairs/pair_0000/left_ir.png").is_file()
    assert (session.root / "raw_pairs/pair_0000/frame_metadata.json").is_file()


def test_session_reader_rejects_manifest_path_escape(tmp_path: Path) -> None:
    session = _session(tmp_path)
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    manifest["target"]["path"] = "../../outside.yaml"
    session.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StereoCalibrationAssetError, match="escapes"):
        StereoCalibrationAssetSession.open(session.root)


def test_offline_detection_preserves_acceptance_evidence_and_checks_integrity(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    session.record_pair(_frame(120, 1))
    session.record_pair(_frame(0, 2))

    run = session.detect_offline(_Detector())  # type: ignore[arg-type]
    assert run.accepted_pair_ids == ("pair_0000",)
    assert run.rejected_pair_ids == ("pair_0001",)
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["accepted_pair_count"] == 1
    assert summary["rejected_pair_count"] == 1
    rejected = summary["pairs"][1]
    assert set(rejected["rejection_reasons"]) == {
        "left_board_not_detected",
        "right_board_not_detected",
    }
    assert (session.root / "analyses/analysis_001/pairs/pair_0000/left_detection.png").is_file()

    raw = session.root / "raw_pairs/pair_0000/left_ir.png"
    raw.write_bytes(b"tampered")
    with pytest.raises(StereoCalibrationAssetError, match="checksum mismatch"):
        session.detect_offline(_Detector())  # type: ignore[arg-type]
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "analysis_failed"


def test_solution_is_bound_to_session_manifest(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.record_pair(_frame(120, 1))
    run = session.detect_offline(_Detector())  # type: ignore[arg-type]
    result_path = session.solution_path(run.run_id)
    result_path.write_text("calibration_type: test\n", encoding="utf-8")
    metrics = SimpleNamespace(
        sample_count=1,
        left_monocular_rms_px=0.1,
        right_monocular_rms_px=0.1,
        joint_stereo_rms_px=0.2,
        epipolar_rmse_px=0.1,
        epipolar_p95_px=0.2,
    )
    result = SimpleNamespace(
        metrics=metrics,
        distortion_model=SimpleNamespace(value="brown5"),
        calibration=SimpleNamespace(baseline_m=0.05),
    )
    session.record_solution(run.run_id, result_path, result)  # type: ignore[arg-type]

    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["result"]["path"].endswith("d435i_ir_stereo_calibration.yaml")
    assert len(manifest["result"]["sha256"]) == 64
    with pytest.raises(StereoCalibrationAssetError, match="immutable"):
        session.record_pair(_frame(130, 2))
