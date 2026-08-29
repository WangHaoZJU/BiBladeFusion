from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    FoundationStereoConfig,
    StereoRectificationConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.bootstrap_foreground import BootstrapForegroundConfig
from biblade_fusion.perception.stereo import StereoResult
from biblade_fusion.storage.bootstrap_foreground import (
    read_bootstrap_foreground,
    write_bootstrap_foreground,
)
from biblade_fusion.storage.session import SessionWriter
from biblade_fusion.storage.stereo_inference import write_stereo_inference
from biblade_fusion.workflows.bootstrap_foreground import (
    bootstrap_foundation_stereo_foreground,
)
from biblade_fusion.workflows.stereo_inference import infer_rectified_stereo


def _bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(30, 20, 50.0, 50.0, 14.5, 9.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation(
            "right_ir",
            "left_ir",
            np.eye(3),
            [-0.05, 0.0, 0.0],
        ),
        None,
    )
    image = np.arange(600, dtype=np.uint16).reshape(20, 30).astype(np.uint8)
    stereo = StereoFrame(1234, 9, 1.0, 1.0, image, image, None, calibration)
    state = RobotState(
        1200,
        1.0,
        np.zeros(6),
        PoseSE3.identity("base", "tcp"),
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        "initial",
        3,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0, 0, 0, 0, 0),
    )


class _SceneFoundationBackend:
    def __init__(self, tmp_path: Path) -> None:
        source = tmp_path / "foundation" / "core" / "foundation_stereo.py"
        source.parent.mkdir(parents=True)
        source.write_text("# test source\n", encoding="utf-8")
        checkpoint = tmp_path / "model.pth"
        checkpoint.write_bytes(b"model")
        model_config = tmp_path / "cfg.yaml"
        model_config.write_text("vit_size: small\n", encoding="utf-8")

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        self.metadata = {
            "backend": "foundation_stereo",
            "runtime": "official_nvidia_foundation_stereo",
            "repository_path": str((tmp_path / "foundation").resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "model_config_path": str(model_config.resolve()),
            "source_sha256": digest(source),
            "checkpoint_sha256": digest(checkpoint),
            "model_config_sha256": digest(model_config),
        }

    def infer(self, left_rectified: np.ndarray, right_rectified: np.ndarray) -> StereoResult:
        disparity = np.full(left_rectified.shape, 5.0, dtype=np.float32)
        disparity[5:15, 10:18] = 10.0
        disparity[9, 18:25] = 9.5
        return StereoResult(
            disparity,
            np.ones_like(disparity, dtype=np.bool_),
            metadata=self.metadata,
        )


def _config() -> BootstrapForegroundConfig:
    return BootstrapForegroundConfig(
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        maximum_neighbour_depth_jump_m=0.04,
        maximum_neighbour_relative_depth_jump=0.0,
        boundary_margin_px=1,
        minimum_valid_pixels=1,
        minimum_component_pixels=1,
        minimum_mask_pixels=1,
        minimum_mask_fraction=0.0,
        maximum_mask_fraction=0.9,
        minimum_seed_valid_pixels=1,
    )


def _calibration_asset(path: Path, bundle: SynchronizedFrameBundle) -> Path:
    calibration = bundle.stereo.calibration

    def camera(intrinsics: CameraIntrinsics) -> dict[str, object]:
        return {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "camera_matrix": [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            "opencv_distortion_model": intrinsics.distortion_model,
            "distortion_coefficients": list(intrinsics.distortion_coefficients),
        }

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_type": "d435i_ir_stereo_charuco",
                "factory_intrinsics_used": False,
                "left_ir": camera(calibration.left),
                "right_ir": camera(calibration.right),
                "right_ir_T_left_ir": calibration.right_t_left.matrix.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _asset(tmp_path: Path) -> tuple[Path, Path]:
    bundle = _bundle()
    with SessionWriter.create(
        tmp_path / "sessions",
        load_settings("configs/default.yaml"),
        label="bootstrap-source",
    ) as writer:
        writer.write_bundle(bundle)
    calibration = _calibration_asset(tmp_path / "stereo.yaml", bundle)
    rectification = StereoRectificationConfig()
    stereo_observation = infer_rectified_stereo(
        bundle,
        _SceneFoundationBackend(tmp_path),
        rectification,
    )
    stereo_root = write_stereo_inference(
        tmp_path / "stereo",
        stereo_observation,
        FoundationStereoConfig(device="cpu"),
        rectification,
        source_session=writer.path,
        source_stereo_calibration=calibration,
    )
    bootstrap = bootstrap_foundation_stereo_foreground(
        stereo_observation,
        _config(),
    )
    output = write_bootstrap_foreground(
        tmp_path / "bootstrap",
        bootstrap,
        source_stereo_inference=stereo_root,
    )
    return output, stereo_root


def test_bootstrap_asset_round_trip_replays_source_and_is_immutable(tmp_path: Path) -> None:
    output, stereo_root = _asset(tmp_path)

    stored = read_bootstrap_foreground(output)

    assert stored.root == output
    assert stored.observation.source_view_id == "initial"
    assert stored.observation.source_sequence_index == 3
    assert stored.observation.source_frame_number == 9
    assert stored.metadata["motion_authorized"] is False
    assert stored.metadata["sources"]["stereo_inference"]["root"] == str(stereo_root)
    assert np.all(stored.observation.result.mask[9, 18:25])
    assert not stored.observation.result.mask.flags.writeable
    with pytest.raises(FileExistsError):
        write_bootstrap_foreground(
            output,
            stored.observation,
            source_stereo_inference=stereo_root,
        )


def test_bootstrap_asset_rejects_tampered_mask(tmp_path: Path) -> None:
    output, _ = _asset(tmp_path)
    np.save(output / "mask.npy", np.zeros((20, 30), dtype=np.bool_), allow_pickle=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_bootstrap_foreground(output)


def test_bootstrap_asset_rejects_tampered_stereo_array(tmp_path: Path) -> None:
    output, stereo_root = _asset(tmp_path)
    depth_path = stereo_root / "depth_m.npy"
    depth = np.load(depth_path, allow_pickle=False)
    depth[6, 11] = 0.9
    np.save(depth_path, depth, allow_pickle=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_bootstrap_foreground(output)


def test_bootstrap_asset_rejects_tampered_raw_left_image(tmp_path: Path) -> None:
    output, stereo_root = _asset(tmp_path)
    stereo_metadata = json.loads(
        (stereo_root / "metadata.json").read_text(encoding="utf-8")
    )
    session = Path(stereo_metadata["source"]["session"])
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    view_root = session / manifest["views"][0]["path"]
    left_path = view_root / "left_ir.npy"
    left = np.load(left_path, allow_pickle=False)
    left[0, 0] += 1
    np.save(left_path, left, allow_pickle=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_bootstrap_foreground(output)


def test_bootstrap_asset_rejects_noncanonical_source_path(tmp_path: Path) -> None:
    output, stereo_root = _asset(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sources"]["stereo_inference"]["root"] = (
        f"{stereo_root}/../{stereo_root.name}"
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="absolute and canonical"):
        read_bootstrap_foreground(output)


def test_bootstrap_workflow_rejects_non_official_backend(tmp_path: Path) -> None:
    rectification = StereoRectificationConfig()
    backend = _SceneFoundationBackend(tmp_path)
    backend.metadata["runtime"] = "fake"
    observation = infer_rectified_stereo(_bundle(), backend, rectification)

    with pytest.raises(ValueError, match="official NVIDIA"):
        bootstrap_foundation_stereo_foreground(observation, _config())
