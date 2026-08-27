from types import SimpleNamespace

from biblade_fusion.core.settings import CollisionConfig, CollisionObstacleConfig, load_settings
from biblade_fusion.diagnostics import doctor


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
        "obstacles",
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
