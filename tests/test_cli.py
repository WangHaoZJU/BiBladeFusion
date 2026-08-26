from typer.testing import CliRunner

from biblade_fusion.cli import app

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


def test_stereo_inference_command_is_exposed() -> None:
    result = runner.invoke(app, ["stereo", "--help"])

    assert result.exit_code == 0
    assert "infer-session" in result.stdout


def test_hand_eye_solver_command_is_exposed() -> None:
    result = runner.invoke(app, ["calibration", "--help"])

    assert result.exit_code == 0
    assert "solve-hand-eye" in result.stdout


def test_plan_command_is_exposed() -> None:
    result = runner.invoke(app, ["plan", "--help"])

    assert result.exit_code == 0
    assert "views" in result.stdout


def test_robot_kinematics_export_is_exposed() -> None:
    result = runner.invoke(app, ["robot", "--help"])

    assert result.exit_code == 0
    assert "export-kinematics" in result.stdout
