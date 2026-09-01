import json
from pathlib import Path

import numpy as np
import pytest

import biblade_fusion.storage.coarse_scan as coarse_scan_module
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    PointCloudConfig,
    ProxyModelConfig,
)
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy, build_bilateral_proxy
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage import read_initialization, write_initialization
from biblade_fusion.workflows import AuthoritativeRobotPose, InitialObservation


def make_observation(
    pose_authority: AuthoritativeRobotPose | None = None,
) -> InitialObservation:
    points = np.array([[0, 0, 0.5], [0.1, 0, 0.5], [0, 0.1, 0.5]])
    pixels = np.array([[0, 0], [1, 0], [0, 1]])
    cloud = PointCloud("base", points, pixels, (2, 2))
    proxy = BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.ones(3),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        3,
        3,
        3,
        1.0,
    )
    return InitialObservation(
        "seed",
        CameraIntrinsics(2, 2, 100, 100, 0.5, 0.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "depth"),
        cloud,
        proxy,
        pose_authority=pose_authority,
    )


def make_filtered_observation(
    pose_authority: AuthoritativeRobotPose,
    proxy_config: ProxyModelConfig,
) -> InitialObservation:
    points = np.array(
        [
            [0.50, 0.00, 0.10],
            [0.51, 0.01, 0.11],
            [0.52, -0.01, 0.12],
            [0.53, 0.02, 0.13],
            [0.54, -0.02, 0.14],
            [0.55, 0.00, 0.15],
            [0.30, 0.00, 0.10],
            [0.80, 0.00, 0.10],
            [0.50, 0.40, 0.10],
            [0.50, 0.00, -0.10],
        ]
    )
    pixels = np.array([(u, v) for v in range(2) for u in range(5)])
    cloud = PointCloud("base", points, pixels, (2, 5))
    base_t_projection_camera = PoseSE3.identity("base", "depth")
    proxy = build_bilateral_proxy(
        points[:6],
        base_t_projection_camera,
        proxy_config,
    )
    return InitialObservation(
        "seed",
        CameraIntrinsics(5, 2, 100, 100, 2, 0.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        base_t_projection_camera,
        cloud,
        proxy,
        pose_authority=pose_authority,
        proxy_support_mask=np.array([True] * 6 + [False] * 4),
    )


def authoritative_inputs(
    tmp_path: Path,
) -> tuple[HandEyeCalibration, AuthoritativeRobotPose]:
    base_t_flange = Es68KinematicModel.from_resources().base_t_flange(np.zeros(6))
    flange_t_tcp = load_es68_flange_t_tcp()
    flange_t_left_ir = base_t_flange.inverse().compose(
        PoseSE3.identity("base", "left_ir")
    )
    source = tmp_path / "he.yaml"
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


def test_initialization_artifact_round_trip(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    point_config = PointCloudConfig(minimum_valid_points=3)
    proxy_config = ProxyModelConfig(estimated_thickness_m=0.01)
    mask = np.array([[True, True], [True, False]])
    output = tmp_path / "initialization"

    write_initialization(
        output,
        make_observation(authority),
        mask,
        hand_eye,
        point_config,
        proxy_config,
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    stored = read_initialization(output)

    assert stored.observation.source_view_id == "seed"
    assert stored.observation.planning_intrinsics.fx == 100
    assert stored.observation.depth_source == "native_realsense"
    assert stored.observation.base_t_projection_camera.child_frame == "depth"
    np.testing.assert_allclose(stored.observation.seed_joint_positions_rad, np.zeros(6))
    np.testing.assert_allclose(
        stored.hand_eye.tcp_t_left_ir.matrix,
        hand_eye.tcp_t_left_ir.matrix,
    )
    np.testing.assert_allclose(
        stored.observation.base_cloud.points_m,
        make_observation().base_cloud.points_m,
    )
    np.testing.assert_array_equal(stored.blade_mask, mask)
    assert stored.metadata["processing"]["proxy_model"]["estimated_thickness_m"] == 0.01
    assert stored.metadata["schema_version"] == 8
    assert stored.hand_eye.flange_t_left_ir is not None
    assert stored.observation.pose_authority is not None
    assert stored.metadata["files"]["base_points_m"]["sha256"]
    assert stored.metadata["files"]["proxy_support_mask"]["sha256"]
    assert stored.observation.proxy_support_mask is not None
    assert stored.observation.proxy_support_mask.all()
    source_record = coarse_scan_module._directory_record(output, "metadata.json")
    assert source_record["authority"] == "metadata.json"

    points = np.load(output / "base_points_m.npy", allow_pickle=False)
    points[0, 0] += 1.0
    np.save(output / "base_points_m.npy", points, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_initialization(output)


def test_initialization_persists_proxy_support_diagnostics(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    proxy_config = ProxyModelConfig(
        estimated_thickness_m=0.01,
        minimum_points=6,
        blade_envelope_min_m=(0.45, -0.10, 0.00),
        blade_envelope_max_m=(0.65, 0.10, 0.30),
        minimum_envelope_retained_fraction=0.5,
    )
    output = write_initialization(
        tmp_path / "filtered_initialization",
        make_filtered_observation(authority, proxy_config),
        np.ones((2, 5), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        proxy_config,
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )

    stored = read_initialization(output)

    assert stored.observation.base_cloud.points_m.shape == (10, 3)
    assert stored.observation.proxy.raw_point_count == 6
    assert stored.metadata["proxy_support"]["input_point_count"] == 10
    assert stored.metadata["proxy_support"]["retained_point_count"] == 6
    assert stored.metadata["proxy_support"]["retained_fraction"] == pytest.approx(0.6)
    np.testing.assert_array_equal(
        stored.observation.proxy_support_mask,
        np.array([True] * 6 + [False] * 4),
    )

    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["proxy"]["extents_m"][0] += 0.01
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="proxy does not reproduce"):
        read_initialization(output)


def test_initialization_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    hand_eye, authority = authoritative_inputs(tmp_path)

    with pytest.raises(FileExistsError):
        write_initialization(
            output,
            make_observation(authority),
            np.ones((2, 2), dtype=bool),
            hand_eye,
            PointCloudConfig(minimum_valid_points=3),
            ProxyModelConfig(estimated_thickness_m=0.01),
            KinematicsConfig(),
            HandEyeConfig(),
            source_session=tmp_path / "session",
        )


def test_initialization_reader_rejects_path_escape(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "files": {
                    "base_points_m": "../outside.npy",
                    "pixel_uv": "pixels.npy",
                    "blade_mask": "mask.npy",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        read_initialization(artifact)


def test_initialization_reader_migrates_schema_four_native_depth(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_initialization(
        tmp_path / "initialization",
        make_observation(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        ProxyModelConfig(estimated_thickness_m=0.01),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 4
    metadata["left_intrinsics"] = metadata.pop("planning_intrinsics")
    metadata["transforms"]["base_T_depth"] = metadata["transforms"].pop(
        "base_T_projection_camera"
    )
    metadata["transforms"].pop("projection_camera_frame")
    metadata["source"].pop("depth_source")
    metadata["source"].pop("stereo_inference")
    metadata["hand_eye"]["source_path"] = metadata["hand_eye"].pop("source")["path"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    stored = read_initialization(output)

    assert stored.observation.depth_source == "native_realsense"
    assert stored.observation.base_t_projection_camera.child_frame == "depth"


def test_initialization_reader_rejects_tampered_fk_authority(tmp_path: Path) -> None:
    hand_eye, authority = authoritative_inputs(tmp_path)
    output = write_initialization(
        tmp_path / "initialization",
        make_observation(authority),
        np.ones((2, 2), dtype=bool),
        hand_eye,
        PointCloudConfig(minimum_valid_points=3),
        ProxyModelConfig(estimated_thickness_m=0.01),
        KinematicsConfig(),
        HandEyeConfig(),
        source_session=tmp_path / "session",
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pose_authority"]["base_T_flange"][0][3] += 0.001
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="base_T_flange does not match ES68 FK|packaged flange_T_tcp",
    ):
        read_initialization(output)
