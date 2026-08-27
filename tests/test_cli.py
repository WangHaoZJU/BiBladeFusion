from typer.testing import CliRunner

from biblade_fusion.cli import _with_emitter_override, app
from biblade_fusion.core.settings import load_settings

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


def test_robot_kinematics_export_is_exposed() -> None:
    result = runner.invoke(app, ["robot", "--help"])

    assert result.exit_code == 0
    assert "export-kinematics" in result.stdout


def test_safety_path_validation_is_exposed() -> None:
    result = runner.invoke(app, ["safety", "--help"])

    assert result.exit_code == 0
    assert "validate-path" in result.stdout
    assert "preflight-path" in result.stdout
