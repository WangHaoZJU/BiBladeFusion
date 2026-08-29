import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import biblade_fusion.storage.reconstructed_view as reconstructed_view_storage
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig, KinematicsConfig, PointCloudConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud, depth_image_to_point_cloud
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import blade_foreground as foreground_storage
from biblade_fusion.storage import read_reconstructed_view, write_reconstructed_view
from biblade_fusion.storage.reconstructed_view import (
    RECONSTRUCTED_VIEW_SCHEMA_VERSION,
    SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION,
)
from biblade_fusion.workflows import AuthoritativeRobotPose, ReconstructedBladeView


def make_view(
    pose_authority: AuthoritativeRobotPose | None = None,
) -> ReconstructedBladeView:
    cloud = PointCloud(
        "base",
        np.array([[0.0, 0.0, 0.5], [0.01, 0.0, 0.5], [0.0, 0.01, 0.5]]),
        np.array([[0, 0], [1, 0], [0, 1]]),
        (2, 2),
    )
    return ReconstructedBladeView(
        "front_r00_c00",
        3,
        17,
        CameraIntrinsics(2, 2, 100.0, 100.0, 0.5, 0.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        cloud,
        "native_realsense",
        pose_authority,
    )


def make_science_view(
    pose_authority: AuthoritativeRobotPose,
) -> ReconstructedBladeView:
    ordinary = make_view(pose_authority)
    angle_rad = 0.08
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    base_t_left_rectified = PoseSE3.from_rotation_translation(
        "base",
        "left_rectified",
        rotation,
        [0.012, -0.006, 0.004],
    )
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    rectified_cloud = depth_image_to_point_cloud(
        np.full((2, 2), 0.5, dtype=np.float32),
        ordinary.planning_intrinsics,
        PointCloudConfig(minimum_valid_points=3),
        frame="left_rectified",
        valid_mask=mask,
    )
    return ReconstructedBladeView(
        ordinary.source_view_id,
        ordinary.source_sequence_index,
        ordinary.source_frame_number,
        ordinary.planning_intrinsics,
        ordinary.joint_positions_rad,
        ordinary.base_t_left_ir,
        base_t_left_rectified,
        rectified_cloud.transformed(base_t_left_rectified),
        "foundation_stereo",
        pose_authority,
    )


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion_model": intrinsics.distortion_model,
        "distortion_coefficients": list(intrinsics.distortion_coefficients),
    }


def _science_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    session = (tmp_path / "session").resolve()
    stereo = (tmp_path / "stereo").resolve()
    foreground = (tmp_path / "foreground").resolve()
    session.mkdir()
    stereo.mkdir()
    foreground.mkdir()
    (foreground / "metadata.json").write_text("{}\n", encoding="utf-8")
    return session, stereo, foreground


def _foreground_fixture(
    view: ReconstructedBladeView,
    mask: np.ndarray,
    *,
    session: Path,
    stereo: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        result=SimpleNamespace(mask=np.array(mask, dtype=np.bool_, copy=True)),
        metadata={
            "identity": {
                "view_id": view.source_view_id,
                "sequence_index": view.source_sequence_index,
                "frame_number": view.source_frame_number,
            },
            "camera": {
                "base_T_left_rectified": (view.base_t_projection_camera.matrix.tolist()),
                "intrinsics": _intrinsics_payload(view.planning_intrinsics),
            },
            "sources": {
                "session": {"root": str(session.resolve())},
                "stereo_inference": {"root": str(stereo.resolve())},
            },
        },
    )


def _left_rectified_t_left_ir(view: ReconstructedBladeView) -> PoseSE3:
    transform = view.base_t_projection_camera.inverse().compose(view.base_t_left_ir)
    assert transform.parent_frame == "left_rectified"
    assert transform.child_frame == "left_ir"
    assert not np.allclose(transform.matrix, np.eye(4))
    return transform


def _patch_foreground_rectification(
    monkeypatch: pytest.MonkeyPatch,
    view: ReconstructedBladeView,
) -> PoseSE3:
    transform = _left_rectified_t_left_ir(view)
    monkeypatch.setattr(
        reconstructed_view_storage,
        "_foreground_rectification",
        lambda _stereo_root: transform,
    )
    return transform


def _patch_science_stereo(
    monkeypatch: pytest.MonkeyPatch,
    view: ReconstructedBladeView,
) -> None:
    observation = SimpleNamespace(
        source_view_id=view.source_view_id,
        source_sequence_index=view.source_sequence_index,
        rectified=SimpleNamespace(
            source_frame_number=view.source_frame_number,
            calibration=SimpleNamespace(left=view.planning_intrinsics),
        ),
        result=SimpleNamespace(valid_mask=np.ones((2, 2), dtype=np.bool_)),
        depth_m=np.full((2, 2), 0.5, dtype=np.float32),
    )
    monkeypatch.setattr(
        reconstructed_view_storage,
        "_science_stereo_observation",
        lambda _stereo_root: observation,
    )


def _mismatched_foreground(
    foreground: SimpleNamespace,
    mismatch: str,
    tmp_path: Path,
) -> SimpleNamespace:
    changed = deepcopy(foreground)
    if mismatch == "mask":
        changed.result.mask[0, 0] = ~changed.result.mask[0, 0]
    elif mismatch == "view_id":
        changed.metadata["identity"]["view_id"] = "another-view"
    elif mismatch == "sequence_index":
        changed.metadata["identity"]["sequence_index"] += 1
    elif mismatch == "frame_number":
        changed.metadata["identity"]["frame_number"] += 1
    elif mismatch == "base_T_left_rectified":
        changed.metadata["camera"]["base_T_left_rectified"][0][3] += 0.001
    elif mismatch == "rectified_intrinsics":
        changed.metadata["camera"]["intrinsics"]["fx"] += 1.0
    elif mismatch == "session":
        changed.metadata["sources"]["session"]["root"] = str(
            (tmp_path / "another-session").resolve()
        )
    elif mismatch == "stereo":
        changed.metadata["sources"]["stereo_inference"]["root"] = str(
            (tmp_path / "another-stereo").resolve()
        )
    else:  # pragma: no cover - the parameter list below is the contract
        raise AssertionError(f"unknown mismatch: {mismatch}")
    return changed


def _write_science_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, ReconstructedBladeView, np.ndarray, SimpleNamespace]:
    hand_eye, authority = authoritative_inputs(tmp_path)
    view = make_science_view(authority)
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    session, stereo, foreground_path = _science_sources(tmp_path)
    foreground = _foreground_fixture(
        view,
        mask,
        session=session,
        stereo=stereo,
    )
    monkeypatch.setattr(
        foreground_storage,
        "read_blade_foreground_mask",
        lambda _path: foreground,
    )
    _patch_foreground_rectification(monkeypatch, view)
    _patch_science_stereo(monkeypatch, view)
    output = write_reconstructed_view(
        tmp_path / "view",
        view,
        mask,
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=session,
        source_stereo_inference=stereo,
        source_blade_foreground_mask=foreground_path,
    )
    return output, view, mask, foreground


def authoritative_inputs(
    tmp_path: Path,
) -> tuple[HandEyeCalibration, AuthoritativeRobotPose]:
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    flange_t_tcp = load_es68_flange_t_tcp()
    flange_t_left_ir = base_t_flange.inverse().compose(PoseSE3.identity("base", "left_ir"))
    source = tmp_path / "hand_eye.yaml"
    tcp_t_left_ir = flange_t_tcp.inverse().compose(flange_t_left_ir)
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "parent_frame": "flange",
                "child_frame": "left_ir",
                "method": "test",
                "matrix": flange_t_left_ir.matrix.tolist(),
                "derived_runtime": {
                    "tcp_T_left_ir": tcp_t_left_ir.matrix.tolist(),
                },
                "quality": {
                    "sample_count": 20,
                    "translation_rmse_m": 0.001,
                    "rotation_rmse_deg": 0.2,
                    "rotation_span_deg": 45.0,
                    "translation_span_m": 0.1,
                    "rotation_axis_diversity": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    hand_eye = HandEyeCalibration(
        tcp_t_left_ir,
        "test",
        20,
        0.001,
        0.2,
        source,
        flange_t_left_ir=flange_t_left_ir,
    )
    authority = AuthoritativeRobotPose(
        base_t_flange,
        base_t_flange.compose(flange_t_tcp),
        base_t_flange.compose(flange_t_tcp),
        0.0,
        0.0,
        0.002,
        0.3,
        (0.0,) * 6,
    )
    return hand_eye, authority


def test_reconstructed_view_round_trip_and_checksum(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_reconstructed_view(
        tmp_path / "view",
        make_view(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    stored = read_reconstructed_view(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["schema_version"] == RECONSTRUCTED_VIEW_SCHEMA_VERSION
    assert metadata["source"]["blade_foreground_mask"] is None
    assert stored.view.source_view_id == "front_r00_c00"
    assert stored.view.source_sequence_index == 3
    assert stored.view.source_frame_number == 17
    assert stored.view.depth_source == "native_realsense"
    assert stored.view.pose_authority is not None
    np.testing.assert_allclose(stored.view.base_cloud.points_m, make_view().base_cloud.points_m)

    points = np.load(output / "base_points_m.npy", allow_pickle=False)
    points[0, 0] += 1.0
    np.save(output / "base_points_m.npy", points, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_reconstructed_view(output)


def test_authoritative_pose_rejects_tcp_not_derived_from_packaged_flange(
    tmp_path: Path,
) -> None:
    _, authority = authoritative_inputs(tmp_path)
    tampered = PoseSE3.from_rotation_translation(
        "base",
        "tcp",
        authority.predicted_base_t_tcp.rotation,
        authority.predicted_base_t_tcp.translation_m + np.array([0.001, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="packaged flange_T_tcp"):
        AuthoritativeRobotPose(
            authority.base_t_flange,
            tampered,
            tampered,
            0.0,
            0.0,
            0.002,
            0.3,
            authority.joint_zero_offsets_rad,
        )


def test_reconstructed_view_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    hand_eye, authority = authoritative_inputs(tmp_path)

    with pytest.raises(FileExistsError):
        write_reconstructed_view(
            output,
            make_view(authority),
            np.ones((2, 2), dtype=bool),
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=tmp_path / "session",
        )


def test_reconstructed_view_reader_rejects_tampered_projection_chain(
    tmp_path: Path,
) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_reconstructed_view(
        tmp_path / "view",
        make_view(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["transforms"]["base_T_projection_camera"][0][3] += 0.001
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="projection transform is inconsistent"):
        read_reconstructed_view(output)


def test_science_reconstructed_view_round_trip_uses_schema_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, expected_view, expected_mask, _ = _write_science_view(
        tmp_path,
        monkeypatch,
    )

    stored = read_reconstructed_view(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["schema_version"] == SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION
    assert metadata["source"]["blade_foreground_mask"]["root"] == str(
        (tmp_path / "foreground").resolve()
    )
    assert stored.view.source_view_id == expected_view.source_view_id
    assert stored.view.source_sequence_index == expected_view.source_sequence_index
    assert stored.view.source_frame_number == expected_view.source_frame_number
    assert stored.view.depth_source == "foundation_stereo"
    assert stored.view.planning_intrinsics == expected_view.planning_intrinsics
    np.testing.assert_array_equal(stored.blade_mask, expected_mask)
    np.testing.assert_allclose(
        stored.view.base_t_projection_camera.matrix,
        expected_view.base_t_projection_camera.matrix,
        rtol=0.0,
        atol=1e-12,
    )
    assert not np.allclose(
        stored.view.base_t_projection_camera.matrix,
        stored.view.base_t_left_ir.matrix,
    )


def test_science_writer_rejects_cloud_not_replayed_from_bound_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    valid_view = make_science_view(authority)
    forged_cloud = PointCloud(
        "base",
        valid_view.base_cloud.points_m + np.array([0.01, 0.0, 0.0]),
        valid_view.base_cloud.pixel_uv,
        valid_view.base_cloud.source_image_shape,
    )
    forged_view = replace(valid_view, base_cloud=forged_cloud)
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    session, stereo, foreground_path = _science_sources(tmp_path)
    foreground = _foreground_fixture(
        valid_view,
        mask,
        session=session,
        stereo=stereo,
    )
    monkeypatch.setattr(
        foreground_storage,
        "read_blade_foreground_mask",
        lambda _path: foreground,
    )
    _patch_foreground_rectification(monkeypatch, valid_view)
    _patch_science_stereo(monkeypatch, valid_view)

    with pytest.raises(ValueError, match="point cloud does not replay"):
        write_reconstructed_view(
            tmp_path / "view",
            forged_view,
            mask,
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=session,
            source_stereo_inference=stereo,
            source_blade_foreground_mask=foreground_path,
        )


@pytest.mark.parametrize("tampered_array", ("base_points_m", "pixel_uv"))
def test_science_reader_replays_cloud_and_rejects_semantic_array_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_array: str,
) -> None:
    output, _, _, _ = _write_science_view(tmp_path, monkeypatch)
    array_path = output / f"{tampered_array}.npy"
    array = np.load(array_path, allow_pickle=False)
    if tampered_array == "base_points_m":
        array[0, 0] += 0.01
    else:
        array[[0, 1]] = array[[1, 0]]
    np.save(array_path, array, allow_pickle=False)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"][tampered_array]["sha256"] = hashlib.sha256(
        array_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="point cloud does not replay"):
        read_reconstructed_view(output)


_FOREGROUND_BINDING_MISMATCHES = (
    "mask",
    "view_id",
    "sequence_index",
    "frame_number",
    "base_T_left_rectified",
    "rectified_intrinsics",
    "session",
    "stereo",
)


@pytest.mark.parametrize("mismatch", _FOREGROUND_BINDING_MISMATCHES)
def test_science_writer_rejects_each_mismatched_foreground_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    view = make_science_view(authority)
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    session, stereo, foreground_path = _science_sources(tmp_path)
    foreground = _foreground_fixture(
        view,
        mask,
        session=session,
        stereo=stereo,
    )
    invalid = _mismatched_foreground(foreground, mismatch, tmp_path)
    monkeypatch.setattr(
        foreground_storage,
        "read_blade_foreground_mask",
        lambda _path: invalid,
    )
    _patch_foreground_rectification(monkeypatch, view)

    with pytest.raises(ValueError, match="foreground mask, identity, camera, or source"):
        write_reconstructed_view(
            tmp_path / "view",
            view,
            mask,
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=session,
            source_stereo_inference=stereo,
            source_blade_foreground_mask=foreground_path,
        )


@pytest.mark.parametrize("mismatch", _FOREGROUND_BINDING_MISMATCHES)
def test_science_reader_rejects_each_mismatched_foreground_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    output, _, _, foreground = _write_science_view(tmp_path, monkeypatch)
    invalid = _mismatched_foreground(foreground, mismatch, tmp_path)
    monkeypatch.setattr(
        foreground_storage,
        "read_blade_foreground_mask",
        lambda _path: invalid,
    )

    with pytest.raises(ValueError, match="foreground mask, identity, camera, or source"):
        read_reconstructed_view(output)


def test_science_reader_rejects_mismatched_raw_rectified_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, _, _ = _write_science_view(tmp_path, monkeypatch)
    monkeypatch.setattr(
        reconstructed_view_storage,
        "_foreground_rectification",
        lambda _stereo_root: PoseSE3.identity("left_rectified", "left_ir"),
    )

    with pytest.raises(ValueError, match="foreground mask, identity, camera, or source"):
        read_reconstructed_view(output)


@pytest.mark.parametrize("missing_source", ("blade_foreground_mask", "stereo_inference"))
def test_science_reader_rejects_missing_required_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_source: str,
) -> None:
    output, _, _, _ = _write_science_view(tmp_path, monkeypatch)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source"][missing_source] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    message = (
        "no blade-foreground source"
        if missing_source == "blade_foreground_mask"
        else "no stereo-inference source"
    )
    with pytest.raises(ValueError, match=message):
        read_reconstructed_view(output)


@pytest.mark.parametrize("missing", ("stereo", "foundation_stereo_depth"))
def test_science_writer_rejects_missing_required_foreground_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    science_view = make_science_view(authority)
    view = make_view(authority) if missing == "foundation_stereo_depth" else science_view
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    session, stereo, foreground_path = _science_sources(tmp_path)
    foreground = _foreground_fixture(
        science_view,
        mask,
        session=session,
        stereo=stereo,
    )
    monkeypatch.setattr(
        foreground_storage,
        "read_blade_foreground_mask",
        lambda _path: foreground,
    )

    with pytest.raises(ValueError, match="requires FoundationStereo reconstruction"):
        write_reconstructed_view(
            tmp_path / "view",
            view,
            mask,
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=session,
            source_stereo_inference=None if missing == "stereo" else stereo,
            source_blade_foreground_mask=foreground_path,
        )
