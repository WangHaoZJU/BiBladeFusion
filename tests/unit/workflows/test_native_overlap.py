from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration, load_hand_eye_calibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    NativeOverlapValidationConfig,
    PointCloudConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import (
    CameraIntrinsics,
    StereoCalibrationSnapshot,
    StereoFrame,
)
from biblade_fusion.devices.robot import RobotState
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import (
    SessionWriter,
    read_legacy_native_overlap_for_replay,
    read_native_overlap_report,
    write_native_overlap_report,
)
from biblade_fusion.workflows import evaluate_native_overlap

INTRINSICS = CameraIntrinsics(64, 48, 80.0, 80.0, 31.5, 23.5, "none", ())


def _plane_depth(
    base_t_camera: PoseSE3,
    plane_point_m: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    u, v = np.meshgrid(np.arange(INTRINSICS.width), np.arange(INTRINSICS.height))
    rays = np.column_stack(
        (
            (u.ravel() - INTRINSICS.cx) / INTRINSICS.fx,
            (v.ravel() - INTRINSICS.cy) / INTRINSICS.fy,
            np.ones(u.size),
        )
    )
    directions = rays @ base_t_camera.rotation.T
    numerator = float(plane_normal @ (plane_point_m - base_t_camera.translation_m))
    denominator = directions @ plane_normal
    axial_depth = numerator / denominator
    axial_depth[(denominator <= 0.0) | (axial_depth <= 0.0)] = np.nan
    return axial_depth.reshape((INTRINSICS.height, INTRINSICS.width))


def _bundle(
    view_id: str,
    index: int,
    joints: np.ndarray,
    plane_point_m: np.ndarray,
    plane_normal: np.ndarray,
) -> SynchronizedFrameBundle:
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(joints)
    base_t_tcp = base_t_flange.compose(load_es68_flange_t_tcp())
    depth_m = _plane_depth(base_t_tcp, plane_point_m, plane_normal)
    native = np.rint(depth_m * 1000.0).astype(np.uint16)
    calibration = StereoCalibrationSnapshot(
        INTRINSICS,
        INTRINSICS,
        PoseSE3.from_rotation_translation("right_ir", "left_ir", np.eye(3), [-0.05, 0.0, 0.0]),
        0.001,
        INTRINSICS,
        PoseSE3.identity("left_ir", "depth"),
    )
    image = np.zeros((INTRINSICS.height, INTRINSICS.width), dtype=np.uint8)
    frame = StereoFrame(
        1000 + index * 10,
        10 + index,
        2000.0 + index,
        2000.0 + index,
        image,
        image,
        native,
        calibration,
    )
    state = RobotState(
        1000 + index * 10,
        float(index),
        joints,
        base_t_tcp,
        "IDLE",
        "NORMAL",
        0.2,
    )
    return SynchronizedFrameBundle(
        view_id,
        0,
        state,
        state,
        state,
        frame,
        None,
        CaptureMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _bundles() -> tuple[SynchronizedFrameBundle, ...]:
    joints = (
        np.zeros(6),
        np.array([0.12, -0.08, 0.06, 0.0, 0.0, 0.0]),
        np.array([-0.10, 0.07, -0.08, 0.04, 0.0, 0.0]),
    )
    reference = Es68KinematicModel.from_resources().base_t_flange(joints[0]).compose(
        load_es68_flange_t_tcp()
    )
    normal = reference.rotation[:, 2]
    point = reference.translation_m + normal * 0.5
    return tuple(
        _bundle(f"view_{index}", index, pose, point, normal)
        for index, pose in enumerate(joints)
    )


def _config() -> NativeOverlapValidationConfig:
    return NativeOverlapValidationConfig(
        minimum_views=3,
        minimum_depth_m=0.3,
        maximum_depth_m=0.7,
        pixel_stride=2,
        edge_window_radius_px=1,
        minimum_projected_points=100,
        minimum_surface_inlier_fraction=0.95,
        maximum_median_absolute_error_m=0.0015,
        maximum_root_mean_square_error_m=0.002,
        maximum_p95_absolute_error_m=0.003,
        minimum_five_mm_agreement_fraction=0.98,
        minimum_translation_span_m=0.02,
        minimum_rotation_span_deg=5.0,
        diagnostic_icp_maximum_points=300,
        diagnostic_icp_minimum_correspondences=50,
        diagnostic_icp_normal_neighbors=8,
        maximum_overlay_points_per_view=1000,
    )


def _hand_eye(path: Path, translation=(0.0, 0.0, 0.0)) -> HandEyeCalibration:
    tcp_t_left_ir = PoseSE3.from_rotation_translation(
        "tcp", "left_ir", np.eye(3), translation
    )
    flange_t_left_ir = load_es68_flange_t_tcp().compose(tcp_t_left_ir)
    return HandEyeCalibration(
        tcp_t_left_ir,
        "synthetic",
        None,
        None,
        None,
        path,
        flange_t_left_ir=flange_t_left_ir,
    )


def _hand_eye_gate(path: Path | None = None) -> HandEyeConfig:
    return HandEyeConfig(
        calibration_path=path,
        require_quality_metrics=False,
        require_observability_metrics=False,
    )


def test_native_overlap_passes_consistent_static_plane_without_using_icp() -> None:
    report = evaluate_native_overlap(
        _bundles(),
        _hand_eye(Path("synthetic.yaml")),
        PointCloudConfig(minimum_valid_points=100),
        _config(),
        kinematics_config=KinematicsConfig(),
        hand_eye_config=_hand_eye_gate(),
    )

    assert report.passed
    assert report.translation_span_m > 0.02
    assert report.rotation_span_deg > 5.0
    assert all(pair.metrics.passed for pair in report.pairs)
    assert all(pair.metrics.p95_absolute_error_m < 0.003 for pair in report.pairs)
    assert all(pair.icp_diagnostic is not None for pair in report.pairs)


def test_native_overlap_detects_wrong_rotating_hand_eye_offset() -> None:
    report = evaluate_native_overlap(
        _bundles(),
        _hand_eye(Path("wrong.yaml"), (0.05, 0.0, 0.0)),
        PointCloudConfig(minimum_valid_points=100),
        _config(),
        kinematics_config=KinematicsConfig(),
        hand_eye_config=_hand_eye_gate(),
    )

    assert not report.passed
    assert any(not pair.metrics.passed for pair in report.pairs)


def _write_hand_eye(path: Path) -> tuple[HandEyeConfig, HandEyeCalibration]:
    flange_t_tcp = load_es68_flange_t_tcp()
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "synthetic",
                "matrix": flange_t_tcp.matrix.tolist(),
                "derived_runtime": {
                    "tcp_T_left_ir": np.eye(4).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    config = _hand_eye_gate(path)
    return config, load_hand_eye_calibration(config)


def test_native_overlap_artifact_recomputes_sources_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    settings = load_settings("configs/default.yaml")
    sessions: list[Path] = []
    for bundle in _bundles():
        with SessionWriter.create(tmp_path / "sessions", settings, label=bundle.view_id) as writer:
            writer.write_bundle(bundle)
        sessions.append(writer.path)
    hand_eye_config, hand_eye = _write_hand_eye(tmp_path / "hand_eye.yaml")
    point_config = PointCloudConfig(minimum_valid_points=100)
    validation_config = _config()
    report = evaluate_native_overlap(
        _bundles(),
        hand_eye,
        point_config,
        validation_config,
        kinematics_config=settings.kinematics,
        hand_eye_config=hand_eye_config,
    )

    output = write_native_overlap_report(
        tmp_path / "overlap",
        report,
        tuple(sessions),
        hand_eye,
        hand_eye_config,
        settings.kinematics,
        point_config,
        validation_config,
    )
    stored = read_native_overlap_report(output)

    assert stored.report.passed
    assert (output / "overlay_base_frame.ply").is_file()
    assert (output / "overview.png").is_file()
    assert (output / "metrics.csv").is_file()
    legacy_output = tmp_path / "legacy-overlap"
    shutil.copytree(output, legacy_output)
    legacy_metadata_path = legacy_output / "native_overlap_report.json"
    legacy_metadata = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
    legacy_metadata["schema_version"] = 1
    legacy_metadata["interpretation"]["primary"] = (
        "base_T_tcp · tcp_T_left_ir · left_ir_T_depth"
    )
    legacy_metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
    legacy = read_legacy_native_overlap_for_replay(legacy_output)
    assert legacy.verification_status == "legacy_tcp_primary_integrity_only"
    assert legacy.current_fk_authority_eligible is False
    with pytest.raises(ValueError, match="unsupported schema 1"):
        read_native_overlap_report(legacy_output)

    residual = next((output / "arrays").glob("residual_*.npy"))
    np.save(residual, np.zeros(5), allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_native_overlap_report(output)
