from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from biblade_fusion.core.settings import CollisionObstacleConfig
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    Es68D435iCollisionTemplate,
    build_es68_d435i_collision_urdf,
    es68_d435i_motion_model_contract_hash,
    es68_d435i_robot_geometry_hash,
)


def test_packaged_collision_template_is_explicitly_inactive() -> None:
    resources = Es68D435iCollisionResources.packaged_template()
    template = resources.load_template()

    assert template.ready is False
    assert template.mesh_units == "m"
    assert len(template.links) == 7
    assert template.attachment.parent_link == "flange"
    assert resources.collision_mesh_dir.joinpath("README.md").is_file()
    assert resources.manifest_path.name == "manifest.yaml"
    with pytest.raises(ValueError, match="not ready"):
        template.validate_assets()


def test_packaged_active_collision_model_uses_holorobot_d435i_mount_relation() -> None:
    resources = Es68D435iCollisionResources.packaged_template()

    template = resources.load_active()
    root = ElementTree.fromstring(build_es68_d435i_collision_urdf(template))

    assert template.model_id == "es68_d435i_depth_camera_mount_v1"
    assert template.mesh_units == "m"
    assert template.mesh_scale == 1.0
    assert all(spec.origin_xyz_m == (0.0, 0.0, 0.0) for spec in template.links)
    assert template.attachment.mesh_path.name == "depth_camera_mount.stl"
    assert template.attachment.joint_xyz_m == (0.0, 0.0, 0.0)
    assert template.attachment.joint_rpy_rad == (0.0, 0.0, 0.0)
    assert template.attachment.origin_xyz_m == (-0.0505, -0.031815, 0.0)

    joint = root.find(".//joint[@name='flange-d435i_collision']")
    collision = root.find(".//link[@name='d435i_collision_link']/collision")
    assert joint is not None
    assert collision is not None
    assert joint.find("parent").attrib == {"link": "flange"}
    assert joint.find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
    assert collision.find("origin").attrib == {
        "xyz": "-0.0505 -0.031815 0",
        "rpy": "0 0 0",
    }
    assert collision.find("geometry/mesh").attrib == {
        "filename": "meshes/es68_d435i/collision/depth_camera_mount.stl",
        "scale": "1 1 1",
    }


def _ready_template(tmp_path: Path, *, units: str = "m") -> Es68D435iCollisionTemplate:
    packaged = Es68D435iCollisionResources.packaged_template()
    payload = yaml.safe_load(packaged.manifest_template_path.read_text(encoding="utf-8"))
    payload["ready"] = True
    payload["mesh_units"] = units
    manifest = tmp_path / "collision_models" / "es68_d435i" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for spec in (*payload["links"].values(), payload["attachment"]):
        mesh = tmp_path / spec["mesh"]
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"solid placeholder\nendsolid placeholder\n")
    return Es68D435iCollisionTemplate.load(manifest, model_root=tmp_path)


def test_materialized_template_uses_es68_chain_and_articulated_meshes(tmp_path: Path) -> None:
    template = _ready_template(tmp_path, units="mm")

    root = ElementTree.fromstring(build_es68_d435i_collision_urdf(template))

    shoulder = root.find(".//joint[@name='shoulder_pan_joint']/origin")
    assert shoulder is not None
    assert float(shoulder.attrib["xyz"].split()[2]) == pytest.approx(0.161444008581)
    base_mesh = root.find(".//link[@name='base_link_inertia']/collision/geometry/mesh")
    assert base_mesh is not None
    assert base_mesh.attrib["filename"] == "meshes/es68_d435i/collision/base.stl"
    assert base_mesh.attrib["scale"] == "0.001 0.001 0.001"
    attachment_joint = root.find(".//joint[@name='flange-d435i_collision']")
    assert attachment_joint is not None
    assert attachment_joint.find("parent").attrib["link"] == "flange"


def test_ready_template_fails_closed_when_a_mesh_is_missing(tmp_path: Path) -> None:
    template = _ready_template(tmp_path)
    template.attachment.mesh_path.unlink()

    with pytest.raises(FileNotFoundError, match="d435i_assembly.stl"):
        build_es68_d435i_collision_urdf(template)


def test_robot_and_motion_hashes_bind_offsets_environment_and_policy(
    tmp_path: Path,
) -> None:
    template = _ready_template(tmp_path)
    zero_geometry = es68_d435i_robot_geometry_hash(template)
    offset_geometry = es68_d435i_robot_geometry_hash(
        template,
        joint_zero_offsets_rad=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    obstacle = CollisionObstacleConfig(
        name="blade_fixture",
        minimum_m=(0.2, -0.1, 0.0),
        maximum_m=(0.4, 0.1, 0.3),
    )
    first_motion = es68_d435i_motion_model_contract_hash(
        template,
        environment_obstacles=(obstacle,),
        minimum_clearance_m=0.01,
    )
    changed_clearance = es68_d435i_motion_model_contract_hash(
        template,
        environment_obstacles=(obstacle,),
        minimum_clearance_m=0.02,
    )
    changed_resolved_pairs = es68_d435i_motion_model_contract_hash(
        template,
        environment_obstacles=(obstacle,),
        minimum_clearance_m=0.01,
        resolved_collision_pairs=(("base_mesh", "fixture"),),
        collision_backend_versions={"pinocchio": "2.7.0", "hppfcl": "2.4.4"},
    )

    assert len(zero_geometry) == 64
    assert zero_geometry != offset_geometry
    assert first_motion != changed_clearance
    assert first_motion != changed_resolved_pairs
