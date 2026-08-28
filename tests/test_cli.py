import pytest
from typer.testing import CliRunner

import biblade_fusion.cli as cli_module
from biblade_fusion.cli import _with_emitter_override, app
from biblade_fusion.core.settings import load_settings
from biblade_fusion.robotics import model_gui

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "BiBladeFusion 0.1.0"


def test_short_version_command() -> None:
    result = runner.invoke(app, ["version", "--short"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_initialize_command_is_exposed() -> None:
    result = runner.invoke(app, ["initialize", "--help"])

    assert result.exit_code == 0
    assert "native-depth" in result.stdout
    assert "stereo-depth" in result.stdout


def test_stereo_inference_command_is_exposed() -> None:
    result = runner.invoke(app, ["stereo", "--help"])

    assert result.exit_code == 0
    assert "infer-session" in result.stdout


def test_hand_eye_solver_command_is_exposed() -> None:
    result = runner.invoke(app, ["calibration", "--help"])

    assert result.exit_code == 0
    assert "solve-hand-eye" in result.stdout
    assert "extract-hand-eye" in result.stdout
    assert "stereo-gui" in result.stdout
    assert "stereo-solve-assets" in result.stdout


def test_plan_command_is_exposed() -> None:
    result = runner.invoke(app, ["plan", "--help"])

    assert result.exit_code == 0
    assert "views" in result.stdout


def test_coverage_command_is_exposed() -> None:
    result = runner.invoke(app, ["coverage", "--help"])

    assert result.exit_code == 0
    assert "seed" in result.stdout
    assert "add" in result.stdout
    assert "next-plan" in result.stdout


def test_reconstruction_commands_are_exposed() -> None:
    result = runner.invoke(app, ["reconstruct", "--help"])

    assert result.exit_code == 0
    assert "native-depth" in result.stdout
    assert "stereo-depth" in result.stdout


def test_acquire_snapshot_exposes_temporary_emitter_override() -> None:
    result = runner.invoke(app, ["acquire", "snapshot", "--help"])

    assert result.exit_code == 0
    assert "--emitter" in result.stdout
    assert "--no-emitter" in result.stdout


def test_emitter_override_does_not_mutate_loaded_settings() -> None:
    settings = load_settings("configs/default.yaml")

    overridden = _with_emitter_override(settings, True)

    assert settings.realsense.infrared_emitter_enabled is False
    assert overridden.realsense.infrared_emitter_enabled is True
    assert _with_emitter_override(settings, None) is settings


def test_evaluation_command_is_exposed() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])

    assert result.exit_code == 0
    assert "depth-pair" in result.stdout
    assert "aggregate-depth" in result.stdout
    assert "make-depth-manifest" in result.stdout
    assert "native-overlap" in result.stdout


def test_robot_kinematics_export_is_exposed() -> None:
    result = runner.invoke(app, ["robot", "--help"])

    assert result.exit_code == 0
    assert "export-kinematics" in result.stdout
    assert "inspect-model" in result.stdout

    inspect_help = runner.invoke(app, ["robot", "inspect-model", "--help"])
    assert inspect_help.exit_code == 0
    assert "--joints-deg" in inspect_help.stdout
    assert "--config" in inspect_help.stdout
    assert "--ip" not in inspect_help.stdout


def test_robot_model_inspector_execution_does_not_construct_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_hardware(*_args, **_kwargs):
        raise AssertionError("offline model inspector attempted to construct hardware")

    launches: list[dict[str, object]] = []

    def fake_launch(**kwargs) -> int:
        launches.append(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "EliteReadOnlyRobot", forbidden_hardware)
    monkeypatch.setattr(cli_module, "RealSenseD435i", forbidden_hardware)
    monkeypatch.setattr(model_gui, "launch_es68_d435i_model_gui", fake_launch)

    result = runner.invoke(
        app,
        [
            "robot",
            "inspect-model",
            "--config",
            "configs/default.yaml",
            "--joints-deg",
            "0,-60,90,-60,-90,0",
        ],
    )

    assert result.exit_code == 0
    assert len(launches) == 1
    assert launches[0]["initial_joint_positions_rad"] == pytest.approx(
        (0.0, -1.0471975512, 1.5707963268, -1.0471975512, -1.5707963268, 0.0)
    )


def test_safety_path_validation_is_exposed() -> None:
    result = runner.invoke(app, ["safety", "--help"])

    assert result.exit_code == 0
    assert "validate-path" in result.stdout
    assert "preflight-path" in result.stdout


def test_read_only_supervisory_replay_is_exposed() -> None:
    result = runner.invoke(app, ["supervise", "--help"])

    assert result.exit_code == 0
    assert "replay" in result.stdout
    assert "build-replay" in result.stdout

    replay_help = runner.invoke(app, ["supervise", "replay", "--help"])
    assert replay_help.exit_code == 0
    assert "--follow" in replay_help.stdout
