from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from biblade_fusion.robotics.collision_template import Es68D435iCollisionResources
from biblade_fusion.robotics.es68_model import Es68KinematicModel
from biblade_fusion.robotics.model_gui import (
    assembly_mesh_paths,
    assembly_mesh_transforms,
    parse_joint_degrees,
    validate_joint_positions,
)


def test_parse_joint_degrees_accepts_commas_or_sequence() -> None:
    parsed = parse_joint_degrees("0, -90, 90, -60, -90, 15")

    assert np.rad2deg(parsed) == pytest.approx((0.0, -90.0, 90.0, -60.0, -90.0, 15.0))
    assert parse_joint_degrees((0, 0, 0, 0, 0, 0)) == (0.0,) * 6
    with pytest.raises(ValueError, match="six-vector"):
        parse_joint_degrees("0, 1")


def test_assembly_uses_es68_link_frames_and_fixed_d435i_mount() -> None:
    template = Es68D435iCollisionResources.packaged_template().load_active()
    model = Es68KinematicModel.from_resources()
    joints = parse_joint_degrees("20,-70,85,-40,-100,30")

    placements = assembly_mesh_transforms(template, joints, kinematic_model=model)
    link_frames = model.link_transforms(joints)

    assert tuple(placements) == tuple(
        spec.link_name for spec in (*template.links, template.attachment)
    )
    for spec in template.links:
        assert placements[spec.link_name] == pytest.approx(link_frames[spec.link_name])

    flange_t_camera_mesh = np.linalg.inv(link_frames["wrist_3_link"]) @ placements[
        "d435i_collision_link"
    ]
    assert flange_t_camera_mesh[:3, :3] == pytest.approx(np.eye(3))
    assert flange_t_camera_mesh[:3, 3] == pytest.approx((-0.0505, -0.031815, 0.0))


def test_mesh_scale_does_not_scale_origins_expressed_in_metres() -> None:
    active = Es68D435iCollisionResources.packaged_template().load_active()
    template = replace(active, mesh_units="mm", mesh_scale=0.001)
    model = Es68KinematicModel.from_resources()

    camera = assembly_mesh_transforms(
        template,
        (0.0,) * 6,
        kinematic_model=model,
    )["d435i_collision_link"]
    flange_t_camera = np.linalg.inv(model.link_transforms((0.0,) * 6)["wrist_3_link"]) @ camera

    assert np.linalg.norm(flange_t_camera[:3, 0]) == pytest.approx(0.001)
    assert flange_t_camera[:3, 3] == pytest.approx((-0.0505, -0.031815, 0.0))


def test_attachment_composes_joint_then_mesh_origin_before_local_scale() -> None:
    active = Es68D435iCollisionResources.packaged_template().load_active()
    attachment = replace(
        active.attachment,
        joint_xyz_m=(0.1, 0.2, 0.3),
        joint_rpy_rad=(np.pi / 2.0, 0.0, 0.0),
        origin_xyz_m=(0.4, 0.5, 0.6),
        origin_rpy_rad=(0.0, np.pi / 2.0, 0.0),
    )
    template = replace(
        active,
        mesh_units="mm",
        mesh_scale=0.001,
        attachment=attachment,
    )
    model = Es68KinematicModel.from_resources()
    joints = parse_joint_degrees("20,-70,85,-40,-100,30")

    joint_transform = np.array(
        (
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 0.0, -1.0, 0.2),
            (0.0, 1.0, 0.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    mesh_origin = np.array(
        (
            (0.0, 0.0, 1.0, 0.4),
            (0.0, 1.0, 0.0, 0.5),
            (-1.0, 0.0, 0.0, 0.6),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    local_scale = np.diag((0.001, 0.001, 0.001, 1.0))
    expected = (
        model.base_t_flange(joints).matrix
        @ joint_transform
        @ mesh_origin
        @ local_scale
    )

    actual = assembly_mesh_transforms(
        template,
        joints,
        kinematic_model=model,
    )[attachment.link_name]

    assert actual == pytest.approx(expected)


def test_gui_pose_validation_rejects_out_of_limit_initial_values() -> None:
    model = Es68KinematicModel.from_resources()

    assert validate_joint_positions((0.0,) * 6, model) == (0.0,) * 6
    with pytest.raises(ValueError, match="J1=.*outside"):
        validate_joint_positions((100.0, 0.0, 0.0, 0.0, 0.0, 0.0), model)


def test_active_d435i_mesh_has_expected_holorobot_envelope() -> None:
    template = Es68D435iCollisionResources.packaged_template().load_active()
    path = assembly_mesh_paths(template)["d435i_collision_link"]

    mesh = trimesh.load_mesh(path, process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    translated = bounds + np.asarray(template.attachment.origin_xyz_m)

    assert translated[0] == pytest.approx((-0.05, -0.0315, 0.0), abs=1e-6)
    assert translated[1] == pytest.approx((0.05, 0.0315, 0.047), abs=1e-6)
    assert np.ptp(translated, axis=0) == pytest.approx((0.100, 0.063, 0.047), abs=1e-6)
