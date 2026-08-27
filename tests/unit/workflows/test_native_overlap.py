from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import numpy as np
import pytest
import yaml

from biblade_fusion.acquisition import CaptureMetrics, SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration, load_hand_eye_calibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
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
from biblade_fusion.storage import (
    SessionWriter,
    read_native_overlap_report,
    write_native_overlap_report,
)
from biblade_fusion.workflows import evaluate_native_overlap


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = radians(angle_deg)
    return np.array(
        [[cos(angle), 0.0, sin(angle)], [0.0, 1.0, 0.0], [-sin(angle), 0.0, cos(angle)]]
    )


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = radians(angle_deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos(angle), -sin(angle)], [0.0, sin(angle), cos(angle)]]
    )


INTRINSICS = CameraIntrinsics(64, 48, 80.0, 80.0, 31.5, 23.5, "none", ())


def _plane_depth(base_t_camera: PoseSE3, plane_z_m: float = 0.5) -> np.ndarray:
    u, v = np.meshgrid(np.arange(INTRINSICS.width), np.arange(INTRINSICS.height))
    rays = np.column_stack(
        (
            (u.ravel() - INTRINSICS.cx) / INTRINSICS.fx,
            (v.ravel() - INTRINSICS.cy) / INTRINSICS.fy,
            np.ones(u.size),
        )
    )
    directions = rays @ base_t_camera.rotation.T
    axial_depth = (plane_z_m - base_t_camera.translation_m[2]) / directions[:, 2]
    axial_depth[directions[:, 2] <= 0.0] = np.nan
    return axial_depth.reshape((INTRINSICS.height, INTRINSICS.width))


def _bundle(view_id: str, index: int, base_t_tcp: PoseSE3) -> SynchronizedFrameBundle:
    depth_m = _plane_depth(base_t_tcp)
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
        np.full(6, index * 0.01),
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
    poses = (
        PoseSE3.identity("base", "tcp"),
        PoseSE3.from_rotation_translation("base", "tcp", _rotation_y(8.0), [-0.03, 0.0, 0.0]),
        PoseSE3.from_rotation_translation("base", "tcp", _rotation_x(-7.0), [0.0, -0.03, 0.0]),
    )
    return tuple(_bundle(f"view_{index}", index, pose) for index, pose in enumerate(poses))


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
    return HandEyeCalibration(
        PoseSE3.from_rotation_translation("tcp", "left_ir", np.eye(3), translation),
        "synthetic",
        None,
        None,
        None,
        path,
    )


def test_native_overlap_passes_consistent_static_plane_without_using_icp() -> None:
    report = evaluate_native_overlap(
        _bundles(),
        _hand_eye(Path("synthetic.yaml")),
        PointCloudConfig(minimum_valid_points=100),
        _config(),
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
    )

    assert not report.passed
    assert any(not pair.metrics.passed for pair in report.pairs)


def _write_hand_eye(path: Path) -> tuple[HandEyeConfig, HandEyeCalibration]:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "parent_frame": "tcp",
                "child_frame": "left_ir",
                "method": "synthetic",
                "matrix": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )
    config = HandEyeConfig(
        calibration_path=path,
        require_quality_metrics=False,
        require_observability_metrics=False,
    )
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
    report = evaluate_native_overlap(_bundles(), hand_eye, point_config, validation_config)

    output = write_native_overlap_report(
        tmp_path / "overlap",
        report,
        tuple(sessions),
        hand_eye,
        hand_eye_config,
        point_config,
        validation_config,
    )
    stored = read_native_overlap_report(output)

    assert stored.report.passed
    assert (output / "overlay_base_frame.ply").is_file()
    assert (output / "overview.png").is_file()
    assert (output / "metrics.csv").is_file()
    residual = next((output / "arrays").glob("residual_*.npy"))
    np.save(residual, np.zeros(5), allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_native_overlap_report(output)
