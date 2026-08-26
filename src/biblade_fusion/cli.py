"""BiBladeFusion command-line interface."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from biblade_fusion import __version__
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.robot import EliteReadOnlyRobot
from biblade_fusion.diagnostics import CheckLevel, run_doctor

app = typer.Typer(
    name="bbf",
    help="BiBladeFusion development and acquisition tools.",
    no_args_is_help=True,
)
robot_app = typer.Typer(help="Safe Elite CS68 state tools.", no_args_is_help=True)
app.add_typer(robot_app, name="robot")


@app.callback()
def main() -> None:
    """BiBladeFusion command group."""


@app.command()
def version(
    short: Annotated[
        bool,
        typer.Option("--short", help="Print only the semantic version."),
    ] = False,
) -> None:
    """Show the BiBladeFusion version."""

    if short:
        typer.echo(__version__)
        return
    typer.echo(f"BiBladeFusion {__version__}")


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Validated YAML configuration file.",
        ),
    ] = Path("configs/default.yaml"),
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Check the local environment without connecting to or moving the robot."""

    settings = load_settings(config)
    results = run_doctor(settings)

    if output_json:
        typer.echo(json.dumps([asdict(result) for result in results], indent=2))
    else:
        table = Table(title="BiBladeFusion doctor")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Message")
        for result in results:
            style = {
                CheckLevel.PASS: "green",
                CheckLevel.WARN: "yellow",
                CheckLevel.FAIL: "red",
            }[result.level]
            table.add_row(result.name, f"[{style}]{result.level.upper()}[/]", result.message)
        Console().print(table)

    if any(result.level is CheckLevel.FAIL for result in results):
        raise typer.Exit(code=1)


@robot_app.command("status")
def robot_status(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Validated YAML configuration file.",
        ),
    ] = Path("configs/default.yaml"),
    robot_ip: Annotated[
        str | None,
        typer.Option("--ip", help="Temporary robot IP override."),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Read one CS68 RTSI state snapshot; this command cannot move the robot."""

    settings = load_settings(config)
    if robot_ip is not None:
        settings.robot = type(settings.robot).model_validate(
            {**settings.robot.model_dump(), "robot_ip": robot_ip}
        )

    with EliteReadOnlyRobot(settings.robot) as robot:
        state = robot.read_state()
        result = {
            "controller_version": robot.controller_version(),
            "controller_time_s": state.controller_time_s,
            "joint_positions_rad": state.joint_positions_rad.tolist(),
            "base_T_tcp": state.base_t_tcp.matrix.tolist(),
            "robot_mode": state.robot_mode,
            "safety_status": state.safety_status,
            "speed_scaling": state.speed_scaling,
        }

    if output_json:
        typer.echo(json.dumps(result, indent=2))
        return

    console = Console()
    console.print("[bold]Elite CS68 read-only RTSI status[/bold]")
    for key, value in result.items():
        console.print(f"{key}: {value}")




if __name__ == "__main__":
    app()
