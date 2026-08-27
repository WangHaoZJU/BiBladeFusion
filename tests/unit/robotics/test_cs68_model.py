from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pytest

from biblade_fusion.robotics import (
    CS68_COLLISION_LINK_NAMES,
    CS68_JOINT_NAMES,
    HOLOROBOT_SOURCE_COMMIT,
    Cs68KinematicModel,
    Cs68ModelResources,
)
from biblade_fusion.robotics.provenance import robot_stack_provenance


def test_packaged_holorobot_cs68_resources_are_complete() -> None:
    resources = Cs68ModelResources.packaged()

    assert resources.urdf_path.is_file()
    assert resources.primitive_collision_manifest.is_file()
    assert "Apache License" in resources.license_path.read_text(encoding="utf-8")
    assert len(tuple(resources.collision_mesh_dir.glob("*.stl"))) == 8


def test_cs68_urdf_preserves_holorobot_joint_and_mesh_conventions() -> None:
    resources = Cs68ModelResources.packaged()
    root = ElementTree.parse(resources.urdf_path).getroot()

    joint_names = {joint.attrib["name"] for joint in root.findall("joint")}
    assert set(CS68_JOINT_NAMES) <= joint_names
    mesh_paths = {
        mesh.attrib["filename"] for mesh in root.findall(".//collision/geometry/mesh")
    }
    assert "meshes/cs68/collision/base.stl" in mesh_paths
    assert "meshes/cs68/collision/wrist3.stl" in mesh_paths


def test_cs68_holorobot_fk_is_stable_at_zero() -> None:
    model = Cs68KinematicModel.from_resources()

    transform = model.forward_kinematics((0.0,) * 6)

    assert transform[2, 3] == pytest.approx(0.066, rel=1e-3)
    np.testing.assert_allclose(transform[3], (0.0, 0.0, 0.0, 1.0))


def test_cs68_joint_zero_offsets_are_applied_in_model_space() -> None:
    offsets = (0.01, -0.02, 0.03, -0.04, 0.05, -0.06)
    offset_model = Cs68KinematicModel.from_resources(joint_zero_offsets_rad=offsets)
    plain_model = Cs68KinematicModel.from_resources()
    controller_joints = (0.2, -0.3, 0.4, -0.5, 0.6, -0.7)
    model_joints = tuple(
        value + offset for value, offset in zip(controller_joints, offsets, strict=True)
    )

    np.testing.assert_allclose(
        offset_model.forward_kinematics(controller_joints),
        plain_model.forward_kinematics(model_joints),
        atol=1e-12,
    )


def test_cs68_link_transforms_cover_holorobot_collision_links() -> None:
    transforms = Cs68KinematicModel.from_resources().link_transforms((0.0,) * 6)

    assert tuple(transforms) == CS68_COLLISION_LINK_NAMES
    assert all(transform.shape == (4, 4) for transform in transforms.values())


def test_robot_stack_provenance_is_pinned() -> None:
    provenance = robot_stack_provenance()

    assert provenance["source_commit"] == HOLOROBOT_SOURCE_COMMIT
    assert provenance["elite_tcp_orientation_convention"] == "rpy_xyz_rad"
