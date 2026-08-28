from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from biblade_fusion.core.settings import (
    CollisionConfig,
    CollisionObstacleConfig,
    OccupancyConfig,
    load_settings,
)
from biblade_fusion.diagnostics import doctor
from biblade_fusion.robotics.collision_template import Es68D435iCollisionResources


def _collision_resources(
    tmp_path: Path,
    *,
    ready: bool,
) -> Es68D435iCollisionResources:
    packaged = Es68D435iCollisionResources.packaged_template()
    payload = yaml.safe_load(packaged.manifest_template_path.read_text(encoding="utf-8"))
    payload["ready"] = ready
    resources = Es68D435iCollisionResources(tmp_path / "elite_cs")
    resources.manifest_path.parent.mkdir(parents=True)
    resources.manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    if ready:
        for spec in (*payload["links"].values(), payload["attachment"]):
            mesh = resources.root / spec["mesh"]
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_bytes(b"solid placeholder\nendsolid placeholder\n")
    return resources


def test_realsense_enumeration_failure_is_a_warning(monkeypatch) -> None:
    fake_module = SimpleNamespace(context=lambda: (_ for _ in ()).throw(RuntimeError("no udev")))
    monkeypatch.setattr(doctor, "import_module", lambda _: fake_module)

    result = doctor._check_realsense()

    assert result.level is doctor.CheckLevel.WARN
    assert "enumeration unavailable" in result.message


def test_collision_diagnostic_lists_fail_closed_missing_inputs() -> None:
    result = doctor._check_collision_configuration(load_settings("configs/default.yaml"))

    assert result.level is doctor.CheckLevel.WARN
    assert set(result.details["missing"]) == {
        "link_radii_m",
        "camera_tool_radius_m",
        "minimum_joint_positions_rad",
        "maximum_joint_positions_rad",
    }


def test_collision_diagnostic_passes_complete_configuration() -> None:
    settings = load_settings("configs/default.yaml")
    settings.collision = CollisionConfig(
        link_radii_m=(0.1,) * 6,
        camera_tool_radius_m=0.1,
        minimum_joint_positions_rad=(-3.0,) * 6,
        maximum_joint_positions_rad=(3.0,) * 6,
        obstacles=(
            CollisionObstacleConfig(
                name="table",
                minimum_m=(-1.0, -1.0, -1.0),
                maximum_m=(1.0, 1.0, 0.0),
            ),
        ),
    )

    result = doctor._check_collision_configuration(settings)

    assert result.level is doctor.CheckLevel.PASS
    assert result.details["motion_authorized"] is False


def test_occupancy_diagnostic_warns_when_workspace_is_not_configured() -> None:
    result = doctor._check_occupancy_configuration(load_settings("configs/default.yaml"))

    assert result.level is doctor.CheckLevel.WARN
    assert result.details["semantic_states"] == ["free", "occupied", "unknown"]
    assert result.details["unknown_blocks_motion"] is True
    assert result.details["workspace_bounds_configured"] is False
    assert result.details["hardware_connection_attempted"] is False


def test_occupancy_diagnostic_passes_enabled_three_state_workspace() -> None:
    settings = load_settings("configs/default.yaml")
    settings.occupancy = OccupancyConfig(
        enabled=True,
        workspace_bounds_min_m=(-0.1, -0.2, 0.0),
        workspace_bounds_max_m=(0.2, 0.2, 0.3),
        voxel_size_m=0.01,
    )

    result = doctor._check_occupancy_configuration(settings)

    assert result.level is doctor.CheckLevel.PASS
    assert result.details["grid_shape"] == [31, 40, 30]
    assert result.details["grid_voxels"] == 37_200
    assert result.details["motion_authorized"] is False


def test_occupancy_diagnostic_rejects_mutated_inverted_workspace() -> None:
    settings = load_settings("configs/default.yaml")
    settings.occupancy = OccupancyConfig(
        enabled=True,
        workspace_bounds_min_m=(-0.1, -0.1, 0.0),
        workspace_bounds_max_m=(0.1, 0.1, 0.2),
    )
    settings.occupancy.workspace_bounds_max_m = (-0.2, 0.1, 0.2)

    result = doctor._check_occupancy_configuration(settings)

    assert result.level is doctor.CheckLevel.FAIL
    assert result.details["workspace_bounds_valid"] is False


def test_final_collision_model_warns_when_manifest_is_missing(tmp_path: Path) -> None:
    resources = Es68D435iCollisionResources(tmp_path / "elite_cs")

    result = doctor._check_final_collision_model(
        load_settings("configs/default.yaml"),
        resources,
    )

    assert result.level is doctor.CheckLevel.WARN
    assert result.details["manifest_exists"] is False
    assert result.details["ready"] is False


def test_final_collision_model_warns_when_manifest_is_inactive(tmp_path: Path) -> None:
    resources = _collision_resources(tmp_path, ready=False)

    result = doctor._check_final_collision_model(
        load_settings("configs/default.yaml"),
        resources,
    )

    assert result.level is doctor.CheckLevel.WARN
    assert result.details["manifest_exists"] is True
    assert result.details["ready"] is False


def test_final_collision_model_validates_ready_assets_and_hashes(tmp_path: Path) -> None:
    resources = _collision_resources(tmp_path, ready=True)

    result = doctor._check_final_collision_model(
        load_settings("configs/default.yaml"),
        resources,
    )

    assert result.level is doctor.CheckLevel.PASS
    assert result.details["ready"] is True
    assert len(result.details["collision_content_sha256"]) == 64
    assert len(result.details["robot_geometry_sha256"]) == 64
    assert result.details["hashes_are_sha256"] is True
    assert result.details["hashes_stable_across_recompute"] is True
    assert result.details["hardware_connection_attempted"] is False


def test_final_collision_model_fails_when_hash_changes_during_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resources = _collision_resources(tmp_path, ready=True)
    collision_hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        doctor,
        "es68_d435i_collision_content_hash",
        lambda _template: next(collision_hashes),
    )
    monkeypatch.setattr(
        doctor,
        "es68_d435i_robot_geometry_hash",
        lambda _template, *, joint_zero_offsets_rad: "c" * 64,
    )

    result = doctor._check_final_collision_model(
        load_settings("configs/default.yaml"),
        resources,
    )

    assert result.level is doctor.CheckLevel.FAIL
    assert result.details["hashes_stable_across_recompute"] is False


def test_motion_readiness_warns_while_motion_is_disabled() -> None:
    result = doctor._check_motion_readiness(load_settings("configs/default.yaml"))

    assert result.level is doctor.CheckLevel.WARN
    assert result.details["motion_ready"] is False
    assert result.details["continuous_swept_mesh_supported"] is False
    assert result.details["continuous_swept_occupancy_supported"] is False
    assert result.details["doctor_authorizes_motion"] is False


def test_motion_readiness_fails_if_motion_is_enabled() -> None:
    settings = load_settings("configs/default.yaml")
    settings.robot.motion_enabled = True

    result = doctor._check_motion_readiness(settings)

    assert result.level is doctor.CheckLevel.FAIL
    assert result.details["motion_enabled"] is True
