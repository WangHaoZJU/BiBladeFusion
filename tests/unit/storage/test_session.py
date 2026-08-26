import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition.bundle import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.depth_camera.base import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.storage import SessionWriter


def make_state(time_ns: int) -> RobotState:
    return RobotState(
        time_ns,
        time_ns / 1e9,
        np.zeros(6),
        PoseSE3.identity("base", "tcp"),
        "IDLE",
        "NORMAL",
        0.2,
    )


def make_bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(4, 3, 100, 100, 2, 1.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]
        ),
        0.001,
    )
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    depth = np.full((3, 4), 1000, dtype=np.uint16)
    stereo = StereoFrame(1_050_000_000, 7, 12.5, 12.6, image, image, depth, calibration)
    before = make_state(1_000_000_000)
    after = make_state(1_100_000_000)
    return SynchronizedFrameBundle(
        "seed/front",
        0,
        before,
        after,
        before,
        stereo,
        None,
        CaptureMetrics(100.0, 0.0, 0.0, 0.0, 50.0),
    )


def test_session_writer_preserves_raw_data_and_metadata(tmp_path: Path) -> None:
    settings = load_settings("configs/default.yaml")

    with SessionWriter.create(tmp_path, settings, label="unit") as writer:
        view_path = writer.write_bundle(make_bundle())

    assert view_path.name == "0000_seed_front"
    np.testing.assert_array_equal(
        np.load(view_path / "left_ir.npy", allow_pickle=False),
        np.arange(12, dtype=np.uint8).reshape(3, 4),
    )
    assert (view_path / "right_ir.npy").is_file()
    assert (view_path / "native_depth.npy").is_file()
    metadata = json.loads((view_path / "metadata.json").read_text())
    assert metadata["view_id"] == "seed/front"
    assert metadata["stereo"]["calibration"]["native_depth_scale_m"] == 0.001
    assert metadata["thermal"] is None

    manifest = json.loads((writer.path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["views"][0]["path"] == "views/0000_seed_front"


def test_session_rejects_duplicate_view(tmp_path: Path) -> None:
    settings = load_settings("configs/default.yaml")
    writer = SessionWriter.create(tmp_path, settings)
    bundle = make_bundle()
    writer.write_bundle(bundle)

    with pytest.raises(FileExistsError):
        writer.write_bundle(bundle)

    writer.close("aborted")
