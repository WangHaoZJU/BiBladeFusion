from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pytest

from biblade_fusion.robotics import (
    CollisionCheckStatus,
    Cs68KinematicModel,
    Cs68ModelResources,
    Cs68PinocchioCollisionChecker,
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
