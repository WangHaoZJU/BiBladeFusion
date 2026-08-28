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
from biblade_fusion.perception.stereo import StereoResult
from biblade_fusion.storage import (
    SessionWriter,
    read_stereo_inference,
    verify_stereo_inference_source,
    write_stereo_inference,
)
from biblade_fusion.workflows import infer_rectified_stereo


class ConstantStereoBackend:
    def __init__(self, metadata: dict[str, object] | None = None) -> None:
        self.metadata = metadata or {"runtime": "constant-test"}

    def infer(self, left_rectified, right_rectified):
        assert left_rectified.shape == right_rectified.shape
        return StereoResult(
            np.full(left_rectified.shape, 5.0, dtype=np.float32),
            np.ones(left_rectified.shape, dtype=bool),
            np.full(left_rectified.shape, 0.9, dtype=np.float32),
            metadata=self.metadata,
        )


def make_bundle() -> SynchronizedFrameBundle:
    intrinsics = CameraIntrinsics(20, 16, 50.0, 50.0, 9.5, 7.5, "none", ())
    calibration = StereoCalibrationSnapshot(
        intrinsics,
        intrinsics,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0, 0]),
        None,
    )
    image = np.arange(20 * 16, dtype=np.uint16).reshape(16, 20).astype(np.uint8)
    stereo = StereoFrame(1234, 7, 1.0, 1.0, image, image, None, calibration)
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
        "seed",
        0,
        state,
        state,
        state,
        stereo,
        None,
        CaptureMetrics(0, 0, 0, 0, 0),
    )


def write_calibration_asset(path: Path, bundle: SynchronizedFrameBundle) -> Path:
    calibration = bundle.stereo.calibration

    def camera_payload(intrinsics: CameraIntrinsics) -> dict[str, object]:
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
                "left_ir": camera_payload(calibration.left),
                "right_ir": camera_payload(calibration.right),
                "right_ir_T_left_ir": calibration.right_t_left.matrix.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return path


def official_test_backend(tmp_path: Path) -> ConstantStereoBackend:
    repository = tmp_path / "foundation"
    source = repository / "core" / "foundation_stereo.py"
    source.parent.mkdir(parents=True)
    source.write_text("# deterministic test source\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"deterministic checkpoint")
    model_config = tmp_path / "cfg.yaml"
    model_config.write_text("vit_size: small\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return ConstantStereoBackend(
        {
            "backend": "foundation_stereo",
            "runtime": "official_nvidia_foundation_stereo",
            "repository_path": str(repository),
            "checkpoint_path": str(checkpoint),
            "model_config_path": str(model_config),
            "source_sha256": digest(source),
            "checkpoint_sha256": digest(checkpoint),
            "model_config_sha256": digest(model_config),
        }
    )


def test_stereo_workflow_and_artifact_round_trip(tmp_path: Path) -> None:
    rectification_config = StereoRectificationConfig()
    observation = infer_rectified_stereo(
        make_bundle(),
        official_test_backend(tmp_path),
        rectification_config,
    )

    assert observation.result.valid_mask[:, :5].sum() == 0
    np.testing.assert_allclose(
        observation.depth_m[observation.result.valid_mask],
        0.5,
    )
    output = write_stereo_inference(
        tmp_path / "stereo",
        observation,
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=tmp_path / "session",
    )
    stored = read_stereo_inference(output)

    np.testing.assert_array_equal(
        stored.observation.result.disparity_px,
        observation.result.disparity_px,
    )
    np.testing.assert_array_equal(
        stored.observation.result.valid_mask,
        observation.result.valid_mask,
    )
    np.testing.assert_allclose(
        stored.observation.result.confidence,
        observation.result.confidence,
    )
    np.testing.assert_allclose(
        stored.observation.depth_m,
        observation.depth_m,
        equal_nan=True,
    )
    assert stored.metadata["files"]["depth_m"]["sha256"]
    assert stored.metadata["source"]["view_id"] == "seed"


def test_stereo_artifact_detects_array_tampering(tmp_path: Path) -> None:
    rectification_config = StereoRectificationConfig()
    observation = infer_rectified_stereo(
        make_bundle(),
        ConstantStereoBackend(),
        rectification_config,
    )
    output = write_stereo_inference(
        tmp_path / "stereo",
        observation,
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=tmp_path / "session",
    )
    np.save(output / "disparity_px.npy", np.zeros((16, 20), dtype=np.float32))

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_stereo_inference(output)


def test_stereo_semantic_source_reproduces_raw_rectification_and_calibration(
    tmp_path: Path,
) -> None:
    bundle = make_bundle()
    with SessionWriter.create(
        tmp_path / "sessions",
        load_settings("configs/default.yaml"),
        label="source-binding",
    ) as writer:
        writer.write_bundle(bundle)
    calibration_path = write_calibration_asset(tmp_path / "stereo.yaml", bundle)
    rectification_config = StereoRectificationConfig()
    observation = infer_rectified_stereo(
        bundle,
        official_test_backend(tmp_path),
        rectification_config,
    )
    output = write_stereo_inference(
        tmp_path / "stereo-bound",
        observation,
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=writer.path,
        source_stereo_calibration=calibration_path,
    )
    stored = read_stereo_inference(output)

    reproduced_bundle = verify_stereo_inference_source(
        stored,
        expected_session=writer.path,
    )

    assert reproduced_bundle.view_id == bundle.view_id
    assert stored.metadata["schema_version"] == 2
    assert stored.metadata["source"]["raw_session_integrity"][
        "left_ir_npy_sha256"
    ]
    Path(stored.metadata["inference"]["checkpoint_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="runtime source checksum mismatch"):
        verify_stereo_inference_source(stored, expected_session=writer.path)


def test_stereo_semantic_source_detects_raw_array_tampering(tmp_path: Path) -> None:
    bundle = make_bundle()
    with SessionWriter.create(
        tmp_path / "sessions",
        load_settings("configs/default.yaml"),
        label="source-tamper",
    ) as writer:
        view_path = writer.write_bundle(bundle)
    calibration_path = write_calibration_asset(tmp_path / "stereo.yaml", bundle)
    rectification_config = StereoRectificationConfig()
    output = write_stereo_inference(
        tmp_path / "stereo-bound",
        infer_rectified_stereo(
            bundle,
            official_test_backend(tmp_path),
            rectification_config,
        ),
        FoundationStereoConfig(device="cpu"),
        rectification_config,
        source_session=writer.path,
        source_stereo_calibration=calibration_path,
    )
    stored = read_stereo_inference(output)
    np.save(
        view_path / "left_ir.npy",
        np.zeros_like(bundle.stereo.left_ir),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_stereo_inference_source(stored, expected_session=writer.path)
