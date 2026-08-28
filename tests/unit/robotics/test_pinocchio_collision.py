from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml

from biblade_fusion.core.settings import CollisionObstacleConfig
from biblade_fusion.mapping.robot_depth_renderer import Es68D435iRobotDepthRenderer
from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68KinematicModel,
    Cs68ModelResources,
    Cs68PinocchioCollisionChecker,
    Es68D435iCollisionResources,
    Es68PinocchioCollisionChecker,
)
from biblade_fusion.robotics.pinocchio_collision import PinocchioCs68Model
from biblade_fusion.robotics.urdf import (
    D435I_MOUNT_COLLISION_ORIGIN_XYZ_M,
    D435I_MOUNT_JOINT,
    D435I_MOUNT_LINK,
    build_cs68_urdf,
    write_cs68_urdf,
)


def test_materialized_urdf_adds_holorobot_d435i_mount() -> None:
    root = ElementTree.fromstring(build_cs68_urdf())

    assert root.find(f".//link[@name='{D435I_MOUNT_LINK}']") is not None
    joint = root.find(f".//joint[@name='{D435I_MOUNT_JOINT}']")
    assert joint is not None
    assert joint.find("parent").attrib["link"] == "wrist_3_link"
    collision_origin = root.find(
        f".//link[@name='{D435I_MOUNT_LINK}']/collision/origin"
    )
    assert collision_origin is not None
    assert tuple(float(value) for value in collision_origin.attrib["xyz"].split()) == (
        D435I_MOUNT_COLLISION_ORIGIN_XYZ_M
    )


@pytest.mark.parametrize(
    "joints",
    [
        (0.0,) * 6,
        (0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
        (0.5, -0.8, 1.2, -0.6, 0.4, -0.3),
    ],
)
def test_pinocchio_fk_matches_holorobot_yaml_model(
    tmp_path, joints: tuple[float, ...]
) -> None:
    resources = Cs68ModelResources.packaged()
    urdf_path = write_cs68_urdf(tmp_path / "cs68.urdf", include_d435i_mount=False)
    pin_model = PinocchioCs68Model.from_urdf(urdf_path)
    yaml_model = Cs68KinematicModel.from_resources(resources)

    np.testing.assert_allclose(
        pin_model.forward_kinematics(joints),
        yaml_model.forward_kinematics(joints),
        atol=1e-10,
    )


def test_pinocchio_collision_includes_d435i_and_holorobot_pairs() -> None:
    checker = Cs68PinocchioCollisionChecker.from_resources()

    assert checker.geometry_model.ngeoms == 8
    assert len(checker.pair_links) == 20
    assert any(D435I_MOUNT_LINK in pair for pair in checker.pair_links)


def test_pinocchio_collision_clear_at_zero() -> None:
    result = Cs68PinocchioCollisionChecker.from_resources().check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.CLEAR
    assert result.motion_authorized is False
    assert result.diagnostics["include_d435i_mount"] is True


def test_pinocchio_collision_enforces_self_clearance_before_contact() -> None:
    result = Cs68PinocchioCollisionChecker.from_resources(
        minimum_clearance_m=0.02
    ).check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.BLOCKED
    finding = next(
        item for item in result.pairs if item.pair_id.startswith("self_clearance:")
    )
    assert finding.minimum_distance_m is not None
    assert finding.minimum_distance_m < 0.02
    assert finding.required_clearance_m == 0.02


def test_pinocchio_collision_blocks_holorobot_folded_fixture() -> None:
    result = Cs68PinocchioCollisionChecker.from_resources().check(
        (0.0, -3.0, 3.0, -3.0, 0.0, 0.0)
    )

    assert result.status is CollisionCheckStatus.BLOCKED
    assert any(reason.startswith("self_collision:") for reason in result.blocking_reasons)


def test_pinocchio_collision_fails_closed_for_invalid_joint_state() -> None:
    result = Cs68PinocchioCollisionChecker.from_resources().check((0.0,) * 5)

    assert result.status is CollisionCheckStatus.UNKNOWN
    assert result.motion_authorized is False


def test_pinocchio_path_sampling_catches_folded_endpoint() -> None:
    report = Cs68PinocchioCollisionChecker.from_resources().check_path(
        (0.0,) * 6,
        (0.0, -3.0, 3.0, -3.0, 0.0, 0.0),
        maximum_joint_step_rad=0.1,
    )

    assert report.status is CollisionCheckStatus.BLOCKED
    assert report.sample_count == 31
    assert report.blocked_sample_index is not None
    assert report.motion_authorized is False


def test_pinocchio_workcell_box_is_checked_against_robot_meshes() -> None:
    model = Cs68KinematicModel.from_resources()
    tcp = model.forward_kinematics((0.0,) * 6)[:3, 3]
    checker = Cs68PinocchioCollisionChecker.from_resources(
        environment_obstacles=(
            CollisionObstacleConfig(
                name="tcp_keepout",
                minimum_m=tuple(float(value - 0.02) for value in tcp),
                maximum_m=tuple(float(value + 0.02) for value in tcp),
            ),
        ),
        minimum_clearance_m=0.005,
    )

    result = checker.check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.BLOCKED
    assert any(
        reason.startswith("workcell_collision:")
        for reason in result.blocking_reasons
    )
    assert result.diagnostics["environment_obstacles"] == ["tcp_keepout"]


def test_far_workcell_box_preserves_clear_state() -> None:
    checker = Cs68PinocchioCollisionChecker.from_resources(
        environment_obstacles=(
            CollisionObstacleConfig(
                name="far",
                minimum_m=(10.0, 10.0, 10.0),
                maximum_m=(11.0, 11.0, 11.0),
            ),
        ),
    )

    result = checker.check((0.0,) * 6)

    assert result.status is CollisionCheckStatus.CLEAR
    assert checker.geometry_model.ngeoms == 9
    assert len(checker.pair_links) == 28


def _write_ready_es68_resources(root: Path) -> Es68D435iCollisionResources:
    import trimesh

    packaged = Es68D435iCollisionResources.packaged_template()
    payload = yaml.safe_load(
        packaged.manifest_template_path.read_text(encoding="utf-8")
    )
    payload["ready"] = True
    manifest = root / "collision_models" / "es68_d435i" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for index, spec in enumerate((*payload["links"].values(), payload["attachment"])):
        mesh = root / spec["mesh"]
        mesh.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(
            extents=(0.01 + index * 0.0001, 0.01, 0.01)
        ).export(mesh)
    return Es68D435iCollisionResources(root)


def test_strict_es68_checker_binds_active_manifest_and_mesh_hash(tmp_path: Path) -> None:
    resources = _write_ready_es68_resources(tmp_path)

    checker = Es68PinocchioCollisionChecker.from_es68_resources(resources)

    assert checker.model_binding[0] == "elite_es68"
    assert checker.collision_model_id == "es68_d435i_collision"
    assert checker.collision_model_hash is not None
    assert len(checker.collision_model_hash) == 64
    assert checker.geometry_model.ngeoms == 8


def test_renderer_and_checker_share_nonzero_offset_robot_geometry_hash(
    tmp_path: Path,
) -> None:
    resources = _write_ready_es68_resources(tmp_path)
    offsets = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)

    checker = Es68PinocchioCollisionChecker.from_es68_resources(
        resources,
        joint_zero_offsets_rad=offsets,
    )
    renderer = Es68D435iRobotDepthRenderer.from_active_resources(
        resources,
        joint_zero_offsets_rad=offsets,
    )
    zero_renderer = Es68D435iRobotDepthRenderer.from_active_resources(resources)

    assert renderer.model_content_hash == checker.robot_geometry_hash
    assert renderer.model_content_hash != zero_renderer.model_content_hash
