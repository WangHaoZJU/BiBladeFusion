from __future__ import annotations

from dataclasses import replace
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


def test_pinocchio_swept_proof_catches_folded_endpoint() -> None:
    report = Cs68PinocchioCollisionChecker.from_resources().check_path(
        (0.0,) * 6,
        (0.0, -3.0, 3.0, -3.0, 0.0, 0.0),
        maximum_joint_step_rad=0.1,
    )

    assert report.status is CollisionCheckStatus.BLOCKED
    assert report.sample_count >= 1
    assert report.blocked_sample_index is not None
    assert report.proof_evidence is not None
    assert report.proof_evidence.termination_reason == "collision_witness"
    assert report.continuous_swept_volume_verified is False
    assert report.motion_authorized is False


def test_pinocchio_swept_proof_catches_collision_between_clear_endpoints() -> None:
    start = (
        -0.2737750906072358,
        -2.54313081933306,
        2.279988088139635,
        0.22151955229468578,
        3.430021029598457,
        1.29868549579151,
    )
    goal = (
        -0.03207170843024487,
        -2.5216118685608597,
        2.230088509205807,
        0.6655138082761458,
        4.092408338680695,
        2.199323664730734,
    )
    checker = Cs68PinocchioCollisionChecker.from_resources()
    assert checker.check(start).status is CollisionCheckStatus.CLEAR
    assert checker.check(goal).status is CollisionCheckStatus.CLEAR

    report = checker.check_path(
        start,
        goal,
        maximum_joint_step_rad=1.0,
    )

    assert report.status is CollisionCheckStatus.BLOCKED
    assert report.blocked_path_fraction == 0.5
    assert report.proof_evidence is not None
    assert report.proof_evidence.termination_reason == "collision_witness"


def test_pinocchio_clear_path_has_integrity_bound_continuous_proof() -> None:
    start = (0.0,) * 6
    goal = (0.02, 0.0, 0.0, 0.0, 0.0, 0.0)

    report = Cs68PinocchioCollisionChecker.from_resources().check_path(
        start,
        goal,
        maximum_joint_step_rad=0.02,
    )

    assert report.status is CollisionCheckStatus.CLEAR
    assert report.continuous_swept_volume_verified is True
    assert report.continuous_swept_volume_evidence_valid is True
    assert report.proof_evidence is not None
    assert report.proof_evidence.matches_path(start, goal)
    assert report.proof_evidence.certified_interval_count >= 1
    assert report.proof_evidence.minimum_certificate_margin_m is not None
    assert report.proof_evidence.minimum_certificate_margin_m > 0.0
    assert report.result.diagnostics["continuous_sweep_backend"] == (
        "adaptive_fcl_relative_motion_tracking_box_sweep"
    )


def test_self_collision_pair_bound_cancels_common_ancestor_motion() -> None:
    checker = Cs68PinocchioCollisionChecker.from_resources()
    pair_index = checker.pair_links.index(("wrist_1_link", "wrist_3_link"))
    pair = checker.geometry_model.collisionPairs[pair_index]
    deviation = (0.02,) * 6

    absolute_bound = checker.geometry_displacement_bound_m(
        int(pair.first), deviation
    ) + checker.geometry_displacement_bound_m(int(pair.second), deviation)
    relative_bound = checker.pair_displacement_bound_m(pair_index, deviation)
    coefficients = checker._pair_motion_coefficients()[pair_index]

    assert coefficients[:4] == pytest.approx((0.0,) * 4)
    assert relative_bound > 0.0
    assert relative_bound < absolute_bound


def test_environment_pair_bound_retains_absolute_robot_motion() -> None:
    checker = Cs68PinocchioCollisionChecker.from_resources(
        environment_obstacles=(
            CollisionObstacleConfig(
                name="far",
                minimum_m=(10.0, 10.0, 10.0),
                maximum_m=(11.0, 11.0, 11.0),
            ),
        ),
    )
    pair_index = next(
        index
        for index, geometries in enumerate(checker.pair_geometries)
        if any(name.startswith("environment::") for name in geometries)
    )
    pair = checker.geometry_model.collisionPairs[pair_index]
    deviation = (0.02,) * 6

    absolute_bound = checker.geometry_displacement_bound_m(
        int(pair.first), deviation
    ) + checker.geometry_displacement_bound_m(int(pair.second), deviation)

    assert checker.pair_displacement_bound_m(pair_index, deviation) == pytest.approx(absolute_bound)


def test_tracking_error_box_is_subdivided_instead_of_becoming_a_fixed_floor() -> None:
    checker = Es68PinocchioCollisionChecker.from_es68_resources(
        Es68D435iCollisionResources.packaged_template(),
        minimum_clearance_m=0.01,
    )
    measured_first_view_joints = (
        3.7294016957032246,
        -1.982535364175026,
        2.0464796841372563,
        -2.189910131300226,
        -2.3942590974776885,
        0.04317535171214368,
    )
    accepted_tracking_error = (
        0.023921648606234358,
        0.0023597103910613093,
        0.008431400738262651,
        0.0032522917774432947,
        0.028725210605739182,
        0.023866973509621525,
    )

    report = checker.check_path(
        measured_first_view_joints,
        measured_first_view_joints,
        maximum_joint_step_rad=0.02,
        maximum_joint_path_deviation_rad=accepted_tracking_error,
        motion_envelope_acceptance_id="a" * 64,
        motion_envelope_metadata_sha256="b" * 64,
    )

    assert report.status is CollisionCheckStatus.CLEAR
    assert report.continuous_swept_volume_evidence_valid is True
    assert report.proof_evidence is not None
    assert report.proof_evidence.deepest_subdivision > 0
    assert report.proof_evidence.evaluated_configuration_count > 3


def test_pinocchio_swept_proof_limit_returns_unknown_not_sampled_clear() -> None:
    checker = Es68PinocchioCollisionChecker.from_es68_resources(
        Es68D435iCollisionResources.packaged_template(),
        minimum_clearance_m=0.01,
    )
    joints = (
        3.7294016957032246,
        -1.982535364175026,
        2.0464796841372563,
        -2.189910131300226,
        -2.3942590974776885,
        0.04317535171214368,
    )
    report = checker.check_path(
        joints,
        joints,
        maximum_joint_step_rad=0.02,
        maximum_subdivision_depth=0,
        maximum_joint_path_deviation_rad=(
            0.023921648606234358,
            0.0023597103910613093,
            0.008431400738262651,
            0.0032522917774432947,
            0.028725210605739182,
            0.023866973509621525,
        ),
        motion_envelope_acceptance_id="a" * 64,
        motion_envelope_metadata_sha256="b" * 64,
    )

    assert report.status is CollisionCheckStatus.UNKNOWN
    assert report.continuous_swept_volume_verified is False
    assert report.proof_evidence is not None
    assert report.proof_evidence.termination_reason == "subdivision_limit"
    assert "unproven:subdivision_limit" in report.result.blocking_reasons[0]
    limiting = report.result.diagnostics["swept_mesh_limiting_pair"]
    assert limiting["certificate_margin_m"] <= 0.0
    assert len(limiting["links"]) == 2
    assert len(limiting["geometries"]) == 2


def test_pinocchio_swept_proof_tampering_invalidates_certificate() -> None:
    report = Cs68PinocchioCollisionChecker.from_resources().check_path(
        (0.0,) * 6,
        (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
        maximum_joint_step_rad=0.02,
    )
    assert report.proof_evidence is not None

    tampered = replace(
        report,
        proof_evidence=replace(
            report.proof_evidence,
            certified_interval_count=report.proof_evidence.certified_interval_count + 1,
        ),
    )

    assert tampered.continuous_swept_volume_verified is True
    assert tampered.continuous_swept_volume_evidence_valid is False


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

    swept = checker.check_path(
        (0.0,) * 6,
        (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        maximum_joint_step_rad=0.01,
    )
    assert swept.status is CollisionCheckStatus.CLEAR
    assert swept.continuous_swept_volume_evidence_valid is True
    assert swept.proof_evidence is not None
    assert swept.result.diagnostics["model"] == "elite_es68"
    assert swept.result.diagnostics["collision_model_hash"] == (
        checker.collision_model_hash
    )
    assert swept.result.diagnostics["robot_geometry_hash"] == (
        checker.robot_geometry_hash
    )
    assert swept.result.diagnostics["motion_model_contract_hash"] == (
        checker.motion_model_contract_hash
    )


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
    assert renderer.self_mask_excluded_link_names == (renderer.template.attachment.link_name,)
    assert renderer.self_mask_render_backend.startswith(("open3d_raycasting:", "numpy_zbuffer:"))
    assert renderer.template.attachment.link_name not in {
        mesh.link_name for mesh in renderer.meshes
    }
