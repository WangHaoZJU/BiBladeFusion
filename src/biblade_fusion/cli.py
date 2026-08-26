"""BiBladeFusion command-line interface."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from biblade_fusion import __version__
from biblade_fusion.acquisition import SynchronizedAcquirer
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.depth_camera import RealSenseD435i, list_realsense_devices
from biblade_fusion.devices.robot import EliteReadOnlyRobot
from biblade_fusion.devices.thermal_camera import NullThermalCamera
from biblade_fusion.diagnostics import CheckLevel, run_doctor
from biblade_fusion.storage import SessionWriter

app = typer.Typer(
    name="bbf",
    help="BiBladeFusion development and acquisition tools.",
    no_args_is_help=True,
)
robot_app = typer.Typer(help="Safe Elite CS68 state tools.", no_args_is_help=True)
camera_app = typer.Typer(help="Intel RealSense D435i tools.", no_args_is_help=True)
acquire_app = typer.Typer(help="Synchronized read-only acquisition.", no_args_is_help=True)
app.add_typer(robot_app, name="robot")
app.add_typer(camera_app, name="camera")
app.add_typer(acquire_app, name="acquire")


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


@camera_app.command("list")
def camera_list(output_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """List connected RealSense devices without starting image streams."""

    try:
        devices = list_realsense_devices()
    except Exception as exc:
        typer.echo(f"RealSense enumeration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    rows = [asdict(device) for device in devices]
    if output_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        typer.echo("No RealSense devices detected.")
        return
    table = Table(title="RealSense devices")
    table.add_column("Serial")
    table.add_column("Name")
    table.add_column("Product line")
    for row in rows:
        table.add_row(row["serial_number"], row["name"], row["product_line"])
    Console().print(table)


@camera_app.command("capture")
def camera_capture(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination .npz file."),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/default.yaml"),
) -> None:
    """Capture one synchronized raw D435i frame bundle."""

    import numpy as np

    settings = load_settings(config)
    destination = output if output.suffix == ".npz" else output.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with RealSenseD435i(settings.realsense) as camera:
        frame = camera.capture()

    arrays: dict[str, object] = {
        "left_ir": frame.left_ir,
        "right_ir": frame.right_ir,
        "monotonic_time_ns": frame.monotonic_time_ns,
        "frame_number": frame.frame_number,
        "left_device_time_ms": frame.left_device_time_ms,
        "right_device_time_ms": frame.right_device_time_ms,
        "left_intrinsics": [
            frame.calibration.left.fx,
            frame.calibration.left.fy,
            frame.calibration.left.cx,
            frame.calibration.left.cy,
        ],
        "right_intrinsics": [
            frame.calibration.right.fx,
            frame.calibration.right.fy,
            frame.calibration.right.cx,
            frame.calibration.right.cy,
        ],
        "right_T_left": frame.calibration.right_t_left.matrix,
        "native_depth_scale_m": (
            frame.calibration.native_depth_scale_m
            if frame.calibration.native_depth_scale_m is not None
            else np.nan
        ),
    }
    if frame.native_depth is not None:
        arrays["native_depth"] = frame.native_depth
    np.savez_compressed(destination, **arrays)
    typer.echo(f"Saved D435i frame bundle: {destination}")


@acquire_app.command("snapshot")
def acquire_snapshot(
    view_id: Annotated[str, typer.Option("--view-id")] = "seed",
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/default.yaml"),
    robot_ip: Annotated[
        str | None,
        typer.Option("--ip", help="Temporary robot IP override."),
    ] = None,
) -> None:
    """Capture one D435i frame bracketed by read-only CS68 states."""

    settings = load_settings(config)
    if robot_ip is not None:
        settings.robot = type(settings.robot).model_validate(
            {**settings.robot.model_dump(), "robot_ip": robot_ip}
        )
    thermal = NullThermalCamera()
    with (
        SessionWriter.create(settings.project.data_root, settings, label=view_id) as session,
        EliteReadOnlyRobot(settings.robot) as robot,
        RealSenseD435i(settings.realsense) as camera,
    ):
        acquirer = SynchronizedAcquirer(
            robot,
            camera,
            thermal,
            settings.acquisition,
            require_thermal=settings.thermal.enabled,
        )
        bundle = acquirer.capture(view_id, 0)
        view_path = session.write_bundle(bundle)
    typer.echo(f"Saved synchronized session: {session.path}")
    typer.echo(f"Saved view: {view_path}")






if __name__ == "__main__":
    app()
