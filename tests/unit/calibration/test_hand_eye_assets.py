import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from biblade_fusion.calibration import (
    HandEyeAssetSession,
    HandEyeBundleAdjustment,
    HandEyeObservability,
    HandEyeSample,
    HandEyeSolution,
    LatestHandEyeBundleMailbox,
    evaluate_hand_eye_validation,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig, load_settings
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.devices.robot.conversions import rotation_vector_to_matrix

TARGET = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml")


def _write_stereo_calibration(path: Path) -> Path:
    intrinsics = {
        "width": 1280,
        "height": 720,
        "camera_matrix": [
            [900.0, 0.0, 640.0],
            [0.0, 905.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        "opencv_distortion_model": "brown_conrady",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    right_t_left = np.eye(4)
    right_t_left[0, 3] = -0.05
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "calibration_type": "d435i_ir_stereo_charuco",
                "factory_intrinsics_used": False,
                "left_ir": intrinsics,
                "right_ir": intrinsics,
                "right_ir_T_left_ir": right_t_left.tolist(),
                "baseline_m": 0.05,
            }
        ),
        encoding="utf-8",
    )
    return path


def _pose(parent: str, child: str, rotation_vector, translation) -> PoseSE3:
    return PoseSE3.from_rotation_translation(
        parent,
        child,
        rotation_vector_to_matrix(rotation_vector),
        translation,
    )


def _validation_case() -> tuple[
    HandEyeSolution,
    tuple[HandEyeSample, ...],
    CameraIntrinsics,
]:
    flange_t_left_ir = _pose(
        "flange", "left_ir", [0.10, -0.05, 0.08], [0.04, 0.01, 0.08]
    )
    base_t_target = _pose(
        "base", "target", [0.20, 0.10, -0.10], [0.55, -0.05, 0.20]
    )
    solution = HandEyeSolution(
        flange_t_left_ir=flange_t_left_ir,
        base_t_target=base_t_target,
        method="OpenCV Park-Martin + LM bundle adjustment",
        sample_count=20,
        translation_rmse_m=0.0,
        rotation_rmse_deg=0.0,
        rotation_max_deg=0.0,
        observability=HandEyeObservability(55.0, 0.2, 0.3),
        bundle_adjustment=HandEyeBundleAdjustment(
            True, True, 0.2, 0.1, 0.08, 0.3, "synthetic"
        ),
        initial_translation_rmse_m=0.0,
        initial_rotation_rmse_deg=0.0,
    )
    intrinsics = CameraIntrinsics(
        1280,
        720,
        900.0,
        905.0,
        640.0,
        360.0,
        "brown_conrady",
        (0.0, 0.0, 0.0, 0.0, 0.0),
    )
    matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]]
    )
    object_points = np.array(
        [[x * 0.02, y * 0.02, 0.0] for y in range(5) for x in range(7)],
        dtype=np.float64,
    )
    motions = (
        ([0.15, -0.05, 0.08], [0.31, -0.18, 0.36]),
        ([-0.18, 0.22, 0.11], [0.36, -0.12, 0.41]),
        ([0.28, 0.13, -0.16], [0.27, -0.08, 0.45]),
        ([-0.12, -0.26, 0.24], [0.43, -0.19, 0.34]),
        ([0.09, 0.31, 0.19], [0.34, -0.03, 0.39]),
    )
    samples: list[HandEyeSample] = []
    for index, (rotation_vector, translation) in enumerate(motions):
        base_t_flange = _pose("base", "flange", rotation_vector, translation)
        left_ir_t_target = (
            flange_t_left_ir.inverse()
            .compose(base_t_flange.inverse())
            .compose(base_t_target)
        )
        target_rotation_vector, _ = cv2.Rodrigues(left_ir_t_target.rotation)
        pixels, _ = cv2.projectPoints(
            object_points,
            target_rotation_vector,
            left_ir_t_target.translation_m,
            matrix,
            np.zeros(5),
        )
        samples.append(
            HandEyeSample(
                f"held-out-{index}",
                base_t_flange,
                left_ir_t_target,
                charuco_corner_count=len(object_points),
                reprojection_rmse_px=0.0,
                charuco_ids=np.arange(len(object_points), dtype=np.int32),
                image_points_px=pixels.reshape(-1, 2),
                object_points_m=object_points,
            )
        )
    return solution, tuple(samples), intrinsics


def test_latest_bundle_mailbox_drops_intermediate_preview_notifications() -> None:
    mailbox = LatestHandEyeBundleMailbox()
    first = object()
    newest = object()

    assert mailbox.publish(first) is True  # type: ignore[arg-type]
    assert mailbox.publish(newest) is False  # type: ignore[arg-type]
    assert mailbox.take_for_preview() is newest
    assert mailbox.take_for_preview() is newest
    assert mailbox.publish(first) is True  # type: ignore[arg-type]


def test_fixed_parameter_held_out_validation_passes_exact_geometry() -> None:
    solution, samples, intrinsics = _validation_case()

    result = evaluate_hand_eye_validation(
        solution,
        samples,
        intrinsics,
        HandEyeConfig(validation_minimum_samples=5),
    )

    assert result.metrics.passed is True
    assert result.metrics.translation_rmse_m < 1e-12
    assert result.metrics.rotation_rmse_deg < 1e-6
    assert result.metrics.reprojection_rmse_px < 1e-8


def test_fixed_parameter_validation_rejects_corner_error_without_refitting() -> None:
    solution, samples, intrinsics = _validation_case()
    shifted = tuple(
        replace(sample, image_points_px=sample.image_points_px + np.array([3.0, 0.0]))
        for sample in samples
    )

    result = evaluate_hand_eye_validation(
        solution,
        shifted,
        intrinsics,
        HandEyeConfig(validation_minimum_samples=5),
    )

    assert result.metrics.passed is False
    assert result.metrics.reprojection_rmse_px == 3.0
    assert result.metrics.translation_rmse_m < 1e-12


def test_asset_session_preserves_undo_and_publishes_only_validated_result(
    tmp_path: Path,
) -> None:
    settings = load_settings("configs/default.yaml")
    stereo_path = _write_stereo_calibration(tmp_path / "stereo.yaml")
    runtime_path = tmp_path / "runtime" / "hand_eye_active.yaml"
    hand_eye_config = settings.hand_eye.model_copy(
        update={
            "calibration_path": runtime_path,
            "minimum_samples": 5,
            "validation_minimum_samples": 5,
        }
    )
    session = HandEyeAssetSession.create(
        tmp_path / "sessions",
        target_path=TARGET,
        stereo_calibration_path=stereo_path,
        robot_config=settings.robot,
        realsense_config=settings.realsense.model_copy(
            update={"stereo_calibration_path": stereo_path}
        ),
        hand_eye_config=hand_eye_config,
        kinematics_config=settings.kinematics,
    )
    session.record_connection_info(
        {"robot_ip": "192.168.6.60", "d435i_serial_number": "test-d435i"}
    )
    solution, validation_samples, intrinsics = _validation_case()
    image = np.zeros((720, 1280), dtype=np.uint8)
    bundle = SimpleNamespace(stereo=SimpleNamespace(left_ir=image, right_ir=image))
    sample_root = session.record_sample(
        "training",
        validation_samples[0],
        bundle,  # type: ignore[arg-type]
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
    )

    assert sample_root.is_dir()
    assert session.exclude_last_sample("training") == validation_samples[0].sample_id
    assert (sample_root / "left_ir.png").is_file()
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_samples"][0]["active"] is False

    training_samples = tuple(
        replace(validation_samples[index % 5], sample_id=f"train-{index:02d}")
        for index in range(20)
    )
    candidate = session.record_candidate(solution, training_samples, intrinsics)
    validation = evaluate_hand_eye_validation(
        solution, validation_samples, intrinsics, hand_eye_config
    )
    report, published, validation_samples_path = session.finalize_validation(
        validation,
        validation_samples,
        runtime_path,
        hand_eye_config,
    )

    assert candidate.is_file()
    assert report.is_file()
    assert validation_samples_path.is_file()
    assert published == runtime_path
    assert runtime_path.is_file()
    published_payload = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    assert published_payload["independent_validation"]["passed"] is True
    assert published_payload["independent_validation"]["calibration_refit_performed"] is False
    manifest = json.loads(session.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["motion_commanded"] is False
