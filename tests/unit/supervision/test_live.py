from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.supervision.live as live_module
from biblade_fusion.mapping import OccupancyMapState, OccupancySnapshot
from biblade_fusion.robotics import Es68KinematicModel
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    es68_d435i_collision_content_hash,
    es68_d435i_robot_geometry_hash,
)
from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentStatusSnapshot,
)
from biblade_fusion.supervision.live import (
    LiveCollisionGeometry,
    LiveSupervisionBridge,
    LiveSupervisionError,
    LiveSupervisionLayout,
    _CollisionMeshPart,
    _PerceptionState,
    _ScienceState,
    _SensorState,
)
from biblade_fusion.supervision.snapshot import (
    discover_supervisory_snapshots,
    load_snapshot_array,
)


def _status(
    root: Path,
    *,
    run_id: str = "live-run-001",
    phase: str = "bootstrap_map_required",
    disposition: ExperimentDisposition = ExperimentDisposition.NEEDS_CAPTURE,
    cycle_index: int = 0,
) -> ExperimentStatusSnapshot:
    return ExperimentStatusSnapshot(
        run_id=run_id,
        run_root=root,
        phase=phase,
        disposition=disposition,
        cycle_index=cycle_index,
        current_view_id=("bootstrap-00" if cycle_index else None),
        proposed_view_id=(
            "fine-01" if disposition is ExperimentDisposition.WAITING_APPROVAL else None
        ),
        expected_capture_view_id=None,
        expected_capture_purpose=None,
        blocking_reasons=(),
        event_count=0,
        latest_event_sha256=None,
        recovery_required=False,
        awaiting_external_approval=(disposition is ExperimentDisposition.WAITING_APPROVAL),
    )


def _collision_geometry(
    tmp_path: Path,
    *,
    collision_model_hash: str = "a" * 64,
    robot_geometry_hash: str = "b" * 64,
) -> LiveCollisionGeometry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "active-collision-manifest.yaml"
    mesh = tmp_path / "base-collision.stl"
    manifest.write_text("model_id: es68-d435i-lab\n", encoding="utf-8")
    mesh.write_bytes(b"unit-test-collision-mesh")
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    kinematics = Es68KinematicModel.from_resources()
    return LiveCollisionGeometry(
        model_id="es68-d435i-lab",
        collision_model_hash=collision_model_hash,
        robot_geometry_hash=robot_geometry_hash,
        manifest_path=manifest,
        manifest_sha256=sha256(manifest),
        parts=(
            _CollisionMeshPart(
                link_name="base_link_inertia",
                parent_link="base_link_inertia",
                vertices_m=np.asarray(
                    ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
                    dtype=np.float64,
                ),
                triangles=np.asarray(((0, 1, 2),), dtype=np.int64),
                parent_t_mesh=np.eye(4),
                source_path=mesh,
                source_sha256=sha256(mesh),
            ),
        ),
        kinematics=kinematics,
    )


def _bridge(
    tmp_path: Path,
    *,
    collision_geometry: LiveCollisionGeometry | None = None,
) -> LiveSupervisionBridge:
    kinematics = Es68KinematicModel.from_resources()
    return LiveSupervisionBridge(
        tmp_path / "timeline",
        layout=LiveSupervisionLayout(
            model_id="es68-d435i-lab",
            occupancy_bounds_min_m=(-0.5, -0.5, 0.0),
            occupancy_bounds_max_m=(0.8, 0.5, 1.5),
            occupancy_voxel_size_m=0.02,
        ),
        kinematics=kinematics,
        collision_geometry=(
            _collision_geometry(tmp_path)
            if collision_geometry is None
            else collision_geometry
        ),
        utc_clock=lambda: datetime(2026, 8, 29, 3, 0, tzinfo=UTC),
    )


def _ready_occupancy() -> OccupancySnapshot:
    observed = datetime(2026, 8, 29, 2, 59, 59, tzinfo=UTC)
    return OccupancySnapshot(
        frame_id="base",
        voxel_size_m=0.02,
        origin_m=(-0.1, -0.1, 0.0),
        grid_shape=(10, 10, 10),
        free_indices=frozenset({(1, 1, 1)}),
        free_observation_counts=(((1, 1, 1), 2),),
        minimum_free_observations=2,
        minimum_free_view_translation_m=0.02,
        minimum_free_view_direction_deg=5.0,
        occupied_indices=frozenset({(2, 2, 2)}),
        sequence=2,
        created_at_utc=observed,
        source_view_ids=("bootstrap-00", "bootstrap-01"),
        source_camera_centres_base_m=((0.0, 0.0, 0.5), (0.03, 0.0, 0.5)),
        source_camera_axes_base=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        rebuild_started_at_utc=observed,
        map_state=OccupancyMapState.MAP_READY,
        mapping_context_hash="1" * 64,
        parent_evidence_hash="3" * 64,
        quality_evidence_hash="2" * 64,
        state_reason="unit-test ready map",
    )


def _inject_verified_display_state(bridge: LiveSupervisionBridge) -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    depth = np.full((3, 4), 0.4, dtype=np.float32)
    depth[0, 0] = np.nan
    sensor = _SensorState(
        frame_number=7,
        captured_at_utc=datetime(2026, 8, 29, 2, 59, 59, tzinfo=UTC),
        left_ir=image,
        right_ir=image,
        depth_m=depth,
        confidence=np.ones((3, 4), dtype=np.float32),
        occupancy_quality_evidence_sha256="2" * 64,
        valid_depth_fraction=11 / 12,
        stereo_valid_fraction=11 / 12,
        confidence_accepted_fraction=1.0,
        mean_accepted_confidence=1.0,
        lr_consistency_threshold_px=1.0,
        fk_tcp_translation_error_m=0.001,
        fk_tcp_rotation_error_deg=0.1,
        projected_robot_pixel_count=3,
        measured_valid_pixel_count=11,
        depth_matched_pixel_count=1,
        masked_valid_pixel_count=2,
        retained_valid_pixel_count=9,
    )
    points = np.asarray(((0.3, 0.0, 0.5), (0.31, 0.01, 0.51)), dtype=np.float64)
    bridge._perception = _PerceptionState(
        robot_joint_positions_rad=(0.0,) * 6,
        robot_mode="IDLE",
        safety_status="NORMAL",
        camera_pose_matrix=tuple(tuple(float(value) for value in row) for row in np.eye(4)),
        captured_at_utc=sensor.captured_at_utc,
        occupancy=_ready_occupancy(),
        sensor=sensor,
        science=_ScienceState(
            current_points_m=points,
            fused_points_m=points,
            front_coverage=0.5,
            back_coverage=0.4,
            registered_view_count=1,
            model_version="registered-view-union:1",
        ),
        assets=(),
    )
    bridge._planned_joint_path = np.asarray(
        ((0.0,) * 6, (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    bridge._sampled_actual_joints = [(0.0,) * 6]


def test_initial_status_is_published_atomically_and_read_only(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    stored = bridge.publish_status(_status(tmp_path / "run"))

    assert stored.snapshot.sequence == 0
    assert stored.snapshot.safety.system_state == "BLOCKED"
    assert stored.snapshot.safety.viewer_motion_command_capable is False
    assert stored.snapshot.occupancy.state == "UNREADY"
    assert stored.snapshot.source_session_id == "live-run-001"
    assert not tuple(bridge.timeline_root.glob("*.partial"))
    discovered = discover_supervisory_snapshots(bridge.timeline_root)
    assert discovered.snapshots == (stored,)
    trajectory = next(
        asset
        for asset in stored.snapshot.assets
        if asset.logical_name == "read_only_joint_trajectory"
    )
    payload = json.loads((stored.root / trajectory.path).read_text(encoding="utf-8"))
    assert payload["motion_authorized"] is False
    assert payload["planned_servoj_joint_path_rad"] is None
    diagnostic = json.loads(
        (
            bridge.timeline_root
            / "performance_diagnostics"
            / "snapshot_00000000_bootstrap_map_required.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic["authority"] == "diagnostic_only_not_safety_or_science_authority"
    assert diagnostic["status"] == "completed"
    assert diagnostic["identity"]["snapshot_sequence"] == 0
    assert diagnostic["spans"]["live.snapshot_publication"]["count"] == 1
    assert diagnostic["spans"]["live.snapshot_commit"]["count"] == 1


def test_approval_without_live_evidence_publishes_block_then_raises(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    with pytest.raises(LiveSupervisionError, match="live_perception_unavailable"):
        bridge.publish_status(
            _status(
                tmp_path / "run",
                phase="waiting_approval",
                disposition=ExperimentDisposition.WAITING_APPROVAL,
            )
        )

    stored = discover_supervisory_snapshots(bridge.timeline_root).snapshots[-1]
    assert stored.snapshot.safety.system_state == "BLOCKED"
    assert "live_perception_unavailable" in stored.snapshot.safety.blocking_reasons
    assert stored.snapshot.plan.state == "PREFLIGHT_FAILED"


def test_map_ready_without_observer_wiring_fails_closed_before_planning(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path)

    with pytest.raises(LiveSupervisionError, match="live_perception_unavailable"):
        bridge.publish_status(
            _status(
                tmp_path / "run",
                phase="map_ready",
                disposition=ExperimentDisposition.READY,
            )
        )

    stored = discover_supervisory_snapshots(bridge.timeline_root).snapshots[-1]
    assert stored.snapshot.safety.system_state == "BLOCKED"


def test_verified_arrays_and_joint_paths_feed_existing_replay_schema(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    _inject_verified_display_state(bridge)

    stored = bridge.publish_status(
        _status(
            tmp_path / "run",
            phase="map_ready",
            disposition=ExperimentDisposition.READY,
            cycle_index=1,
        )
    )

    occupied = load_snapshot_array(
        stored,
        stored.snapshot.occupancy.occupied_centres_m,
    )
    planned = load_snapshot_array(
        stored,
        stored.snapshot.robot.planned_tcp_path_base_m,
    )
    depth = load_snapshot_array(stored, stored.snapshot.sensor.depth_m)
    fused = load_snapshot_array(stored, stored.snapshot.reconstruction.fused_points_m)
    collision_vertices = load_snapshot_array(
        stored,
        stored.snapshot.robot.collision_mesh_vertices_base_m,
    )
    collision_triangles = load_snapshot_array(
        stored,
        stored.snapshot.robot.collision_mesh_triangles,
    )
    assert occupied is not None and occupied.shape == (1, 3)
    assert planned is not None and planned.shape == (2, 3)
    assert depth is not None and np.isnan(depth[0, 0])
    assert fused is not None and fused.shape == (2, 3)
    assert collision_vertices is not None and collision_vertices.shape == (3, 3)
    assert collision_triangles is not None and collision_triangles.shape == (1, 3)
    binding = next(
        asset
        for asset in stored.snapshot.assets
        if asset.logical_name == "active_live_collision_mesh_binding"
    )
    binding_payload = json.loads((stored.root / binding.path).read_text(encoding="utf-8"))
    assert binding_payload["collision_model_sha256"] == "a" * 64
    assert binding_payload["robot_geometry_sha256"] == "b" * 64
    assert binding_payload["motion_authorized"] is False
    assert stored.snapshot.sensor.source == "FOUNDATION_STEREO"
    assert stored.snapshot.reconstruction.provenance_reasons == (
        "bounded_display_union_is_not_scientific_fusion",
    )
    trajectory = next(
        asset
        for asset in stored.snapshot.assets
        if asset.logical_name == "read_only_joint_trajectory"
    )
    payload = json.loads((stored.root / trajectory.path).read_text(encoding="utf-8"))
    assert len(payload["planned_servoj_joint_path_rad"]) == 2
    assert payload["actual_path_semantic"].endswith("not_high_rate_servoj_tracking")


def test_follow_timeline_is_append_only_and_bound_to_one_run(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    first = bridge.publish_status(_status(tmp_path / "run"))
    second = bridge.publish_status(_status(tmp_path / "run"))

    timeline = discover_supervisory_snapshots(bridge.timeline_root)
    assert [item.snapshot.sequence for item in timeline.snapshots] == [0, 1]
    assert timeline.snapshots[0].content_sha256 == first.content_sha256
    assert timeline.snapshots[1].content_sha256 == second.content_sha256

    other = _status(tmp_path / "other", run_id="another-run")
    with pytest.raises(LiveSupervisionError, match="another supervised run"):
        bridge.publish_status(other)


def test_bridge_surface_has_no_motion_or_approval_method(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)

    assert bridge.motion_command_capable is False
    for name in (
        "approve",
        "authorize",
        "execute",
        "execute_segment",
        "request_stop",
        "stop",
    ):
        assert not hasattr(bridge, name)


def test_phase_event_reset_preserves_coarse_clouds_and_snapshot_continuity(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path)
    _inject_verified_display_state(bridge)
    coarse_points = np.asarray(((0.2, 0.0, 0.4), (0.21, 0.01, 0.41)))
    display_points, display_keys = live_module._display_voxel_representatives(
        coarse_points,
        maximum_points=50_000,
    )
    bridge._display_union = {
        key: (0.0, tuple(float(value) for value in point))
        for key, point in zip(display_keys, display_points, strict=True)
    }
    bridge._registered_physical_source_ids = {"coarse-00"}
    first = bridge.publish_status(
        _status(
            tmp_path / "coarse-run",
            phase="map_ready",
            disposition=ExperimentDisposition.READY,
            cycle_index=1,
        )
    )

    bridge.begin_new_event_stream(run_id="live-run-001")

    assert bridge.motion_command_capable is False
    assert bridge._registered_physical_source_ids == {"coarse-00"}
    assert len(bridge._display_union) == 2
    assert bridge._perception is not None
    assert bridge._prepared is None
    assert bridge._planned_joint_path is None
    second = bridge.publish_status(_status(tmp_path / "fine-run", phase="bootstrap_map_required"))
    fused = load_snapshot_array(second, second.snapshot.reconstruction.fused_points_m)
    assert fused is not None
    assert fused.shape == (2, 3)
    assert second.snapshot.sequence == first.snapshot.sequence + 1
    assert second.snapshot.safety.viewer_motion_command_capable is False

    with pytest.raises(LiveSupervisionError, match="another supervised run"):
        bridge.begin_new_event_stream(run_id="different-run")


def test_coarse_scan_view_is_strictly_read_and_added_as_current_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse_root = tmp_path / "coarse-view"
    coarse_root.mkdir()
    (coarse_root / "metadata.json").write_text(
        '{"artifact_kind":"biblade_fusion.coarse_scan_view"}\n',
        encoding="utf-8",
    )
    points = np.asarray(((0.2, 0.0, 0.4), (0.21, 0.01, 0.41)), dtype=np.float64)
    reconstructed_root = coarse_root / "reconstructed"
    reconstructed_root.mkdir()
    np.save(reconstructed_root / "base_points_m.npy", points, allow_pickle=False)
    point_sha256 = hashlib.sha256(
        (reconstructed_root / "base_points_m.npy").read_bytes()
    ).hexdigest()
    (reconstructed_root / "metadata.json").write_text("{}\n", encoding="utf-8")
    reconstructed = SimpleNamespace(
        view=SimpleNamespace(
            source_view_id="coarse-00",
            source_sequence_index=3,
            source_frame_number=17,
            base_cloud=SimpleNamespace(points_m=points),
        ),
        metadata={
            "files": {
                "base_points_m": {
                    "path": "base_points_m.npy",
                    "sha256": point_sha256,
                }
            }
        },
    )
    calls = []

    def read_coarse(path: Path):
        calls.append(Path(path).resolve())
        return SimpleNamespace(
            reconstructed=reconstructed,
            metadata={"sources": {"reconstructed_view": {"root": str(reconstructed_root)}}},
        )

    monkeypatch.setattr(live_module, "read_coarse_scan_view", read_coarse)
    result = SimpleNamespace(
        reconstructed_view_path=None,
        coarse_scan_view_path=coarse_root,
        bundle=SimpleNamespace(
            view_id="coarse-00",
            sequence_index=3,
            stereo=SimpleNamespace(frame_number=17),
        ),
    )

    current = live_module._read_current_science_view(result)

    assert calls == [coarse_root.resolve()]
    assert current.points_m is not None
    assert np.array_equal(current.points_m, points)
    assert current.points_m.flags.writeable is False
    assert current.asset is not None
    assert current.asset.kind == "biblade_fusion.coarse_scan_view"
    assert current.asset.path == coarse_root / "metadata.json"


def test_coarse_scan_identity_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse_root = tmp_path / "coarse-view"
    coarse_root.mkdir()
    (coarse_root / "metadata.json").write_text("{}\n", encoding="utf-8")
    reconstructed_root = coarse_root / "reconstructed"
    reconstructed_root.mkdir()
    mismatch_points = np.zeros((1, 3))
    np.save(reconstructed_root / "base_points_m.npy", mismatch_points, allow_pickle=False)
    point_sha256 = hashlib.sha256(
        (reconstructed_root / "base_points_m.npy").read_bytes()
    ).hexdigest()
    (reconstructed_root / "metadata.json").write_text("{}\n", encoding="utf-8")
    reconstructed = SimpleNamespace(
        view=SimpleNamespace(
            source_view_id="another-view",
            source_sequence_index=3,
            source_frame_number=17,
            base_cloud=SimpleNamespace(points_m=mismatch_points),
        ),
        metadata={
            "files": {
                "base_points_m": {
                    "path": "base_points_m.npy",
                    "sha256": point_sha256,
                }
            }
        },
    )
    monkeypatch.setattr(
        live_module,
        "read_coarse_scan_view",
        lambda _: SimpleNamespace(
            reconstructed=reconstructed,
            metadata={"sources": {"reconstructed_view": {"root": str(reconstructed_root)}}},
        ),
    )
    result = SimpleNamespace(
        reconstructed_view_path=None,
        coarse_scan_view_path=coarse_root,
        bundle=SimpleNamespace(
            view_id="coarse-00",
            sequence_index=3,
            stereo=SimpleNamespace(frame_number=17),
        ),
    )

    with pytest.raises(LiveSupervisionError, match="identities differ"):
        live_module._read_current_science_view(result)


def test_active_collision_geometry_requires_exact_checker_hashes() -> None:
    resources = Es68D435iCollisionResources.packaged_template()
    template = resources.load_active()
    offsets = (0.0,) * 6
    collision_hash = es68_d435i_collision_content_hash(template)
    robot_hash = es68_d435i_robot_geometry_hash(
        template,
        joint_zero_offsets_rad=offsets,
    )

    geometry = LiveCollisionGeometry.from_active_resources(
        resources,
        joint_zero_offsets_rad=offsets,
        expected_model_id=template.model_id,
        expected_collision_model_hash=collision_hash,
        expected_robot_geometry_hash=robot_hash,
    )

    assert geometry.model_id == template.model_id
    assert geometry.collision_model_hash == collision_hash
    assert geometry.robot_geometry_hash == robot_hash
    assert len(geometry.parts) == 8
    with pytest.raises(LiveSupervisionError, match="manifest/STLs differ"):
        LiveCollisionGeometry.from_active_resources(
            resources,
            joint_zero_offsets_rad=offsets,
            expected_model_id=template.model_id,
            expected_collision_model_hash="0" * 64,
            expected_robot_geometry_hash=robot_hash,
        )


def test_live_collision_source_mutation_blocks_publication(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    _inject_verified_display_state(bridge)
    (tmp_path / "base-collision.stl").write_bytes(b"mutated")

    with pytest.raises(LiveSupervisionError, match="collision STL changed"):
        bridge.publish_status(
            _status(
                tmp_path / "run",
                phase="map_ready",
                disposition=ExperimentDisposition.READY,
                cycle_index=1,
            )
        )


def test_bridge_restart_replays_append_only_display_registry(tmp_path: Path) -> None:
    first = _bridge(tmp_path)
    first.publish_status(_status(tmp_path / "run"))
    source = tmp_path / "display-source"
    source.mkdir()
    metadata = source / "metadata.json"
    point_array = source / "base_points_m.npy"
    points = np.asarray(((0.101, 0.0, 0.2), (0.201, 0.0, 0.3)), dtype=np.float64)
    np.save(point_array, points, allow_pickle=False)
    metadata.write_text("{}\n", encoding="utf-8")
    representatives, _ = live_module._display_voxel_representatives(
        points,
        maximum_points=50_000,
    )
    entry = first._display_registry.append(
        source_kind="coarse_scan_view",
        view_id="coarse-restart-00",
        source_sequence_index=0,
        source_frame_number=4,
        metadata_path=metadata,
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        point_array_path=point_array,
        point_array_file_sha256=hashlib.sha256(point_array.read_bytes()).hexdigest(),
        points_f64le_sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(points, dtype="<f8")).tobytes()
        ).hexdigest(),
        raw_point_count=len(points),
        voxel_point_count=len(representatives),
        display_algorithm=live_module._DISPLAY_UNION_ALGORITHM,
        display_voxel_size_m=live_module._DISPLAY_VOXEL_SIZE_M,
        maximum_current_points=live_module._MAXIMUM_CURRENT_DISPLAY_POINTS,
        created_at_utc=datetime(2026, 8, 29, 3, 0, tzinfo=UTC),
    )

    restarted = _bridge(tmp_path)

    assert restarted._registered_physical_source_ids == {entry.physical_source_id}
    assert len(restarted._display_union) == len(representatives)
    assert restarted._display_registry.head.head_entry_sha256 == entry.entry_sha256


def test_bridge_restart_rejects_registry_truncated_after_bound_snapshot(
    tmp_path: Path,
) -> None:
    first = _bridge(tmp_path)
    source = tmp_path / "display-source"
    source.mkdir()
    metadata = source / "metadata.json"
    point_array = source / "base_points_m.npy"
    points = np.asarray(((0.101, 0.0, 0.2),), dtype=np.float64)
    np.save(point_array, points, allow_pickle=False)
    metadata.write_text("{}\n", encoding="utf-8")
    entry = first._display_registry.append(
        source_kind="coarse_scan_view",
        view_id="coarse-bound-00",
        source_sequence_index=0,
        source_frame_number=4,
        metadata_path=metadata,
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        point_array_path=point_array,
        point_array_file_sha256=hashlib.sha256(point_array.read_bytes()).hexdigest(),
        points_f64le_sha256=hashlib.sha256(
            np.ascontiguousarray(np.asarray(points, dtype="<f8")).tobytes()
        ).hexdigest(),
        raw_point_count=1,
        voxel_point_count=1,
        display_algorithm=live_module._DISPLAY_UNION_ALGORITHM,
        display_voxel_size_m=live_module._DISPLAY_VOXEL_SIZE_M,
        maximum_current_points=live_module._MAXIMUM_CURRENT_DISPLAY_POINTS,
        created_at_utc=datetime(2026, 8, 29, 3, 0, tzinfo=UTC),
    )
    first.publish_status(_status(tmp_path / "run"))
    entry.path.unlink()

    with pytest.raises(LiveSupervisionError, match="registry bounds differ"):
        _bridge(tmp_path)


def test_bridge_restart_rejects_different_active_collision_model(tmp_path: Path) -> None:
    first = _bridge(tmp_path)
    _inject_verified_display_state(first)
    first.publish_status(
        _status(
            tmp_path / "run",
            phase="map_ready",
            disposition=ExperimentDisposition.READY,
            cycle_index=1,
        )
    )
    replacement = _collision_geometry(
        tmp_path / "replacement-collision",
        collision_model_hash="c" * 64,
        robot_geometry_hash="d" * 64,
    )

    with pytest.raises(LiveSupervisionError, match="differs from the active model"):
        _bridge(tmp_path, collision_geometry=replacement)
