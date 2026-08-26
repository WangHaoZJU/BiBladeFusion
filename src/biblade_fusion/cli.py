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
from biblade_fusion.calibration import (
    fetch_cs68_kinematics,
    load_cs68_kinematics,
    load_hand_eye_calibration,
    read_hand_eye_samples,
    solve_hand_eye,
    write_cs68_kinematics,
    write_hand_eye_calibration,
    write_hand_eye_samples,
)
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.depth_camera import RealSenseD435i, list_realsense_devices
from biblade_fusion.devices.robot import EliteReadOnlyRobot
from biblade_fusion.devices.thermal_camera import NullThermalCamera
from biblade_fusion.diagnostics import CheckLevel, run_doctor
from biblade_fusion.perception.stereo import (
    FoundationStereoBackend,
    run_foundation_stereo_doctor,
)
from biblade_fusion.planning import (
    EliteCs68IkChecker,
    coverage_observation_id,
    create_coverage_ledger,
    select_uncovered_candidates,
    update_coverage,
)
from biblade_fusion.storage import (
    SessionReader,
    SessionWriter,
    read_coverage_driven_plan,
    read_coverage_ledger,
    read_depth_aggregate,
    read_depth_comparison,
    read_initialization,
    read_reconstructed_view,
    read_stereo_inference,
    read_view_plan,
    write_coverage_driven_plan,
    write_coverage_ledger,
    write_depth_aggregate,
    write_depth_comparison,
    write_initialization,
    write_reconstructed_view,
    write_stereo_inference,
    write_view_plan,
)
from biblade_fusion.workflows import (
    compare_paired_depth,
    extract_hand_eye_samples,
    infer_rectified_stereo,
    initialize_foundation_stereo_depth,
    initialize_native_depth,
    plan_initial_observation,
    reconstruct_foundation_stereo_view,
    reconstruct_native_depth_view,
)

app = typer.Typer(
    name="bbf",
    help="BiBladeFusion development and acquisition tools.",
    no_args_is_help=True,
)
robot_app = typer.Typer(help="Safe Elite CS68 state tools.", no_args_is_help=True)
camera_app = typer.Typer(help="Intel RealSense D435i tools.", no_args_is_help=True)
acquire_app = typer.Typer(help="Synchronized read-only acquisition.", no_args_is_help=True)
calibration_app = typer.Typer(help="Offline calibration tools.", no_args_is_help=True)
stereo_app = typer.Typer(help="Stereo inference tools.", no_args_is_help=True)
initialize_app = typer.Typer(help="Offline initial-model construction.", no_args_is_help=True)
plan_app = typer.Typer(help="Offline bilateral view planning.", no_args_is_help=True)
coverage_app = typer.Typer(help="Offline bilateral coverage tracking.", no_args_is_help=True)
reconstruct_app = typer.Typer(help="Pose-register stored blade depth views.", no_args_is_help=True)
evaluate_app = typer.Typer(help="Offline experiment evaluation.", no_args_is_help=True)
app.add_typer(robot_app, name="robot")
app.add_typer(camera_app, name="camera")
app.add_typer(acquire_app, name="acquire")
app.add_typer(calibration_app, name="calibration")
app.add_typer(stereo_app, name="stereo")
app.add_typer(initialize_app, name="initialize")
app.add_typer(plan_app, name="plan")
app.add_typer(coverage_app, name="coverage")
app.add_typer(reconstruct_app, name="reconstruct")
app.add_typer(evaluate_app, name="evaluate")


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


@robot_app.command("export-kinematics")
def robot_export_kinematics(
    output: Annotated[Path, typer.Option("--output", "-o")],
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
    """Read and store CS68 MDH parameters; this command cannot move the robot."""

    try:
        settings = load_settings(config)
        address = robot_ip or settings.robot.robot_ip
        if address is None:
            raise ValueError("Robot IP is not configured")
        model = fetch_cs68_kinematics(
            address,
            timeout_ms=settings.kinematics.primary_timeout_ms,
        )
        destination = write_cs68_kinematics(output, model)
    except Exception as exc:
        typer.echo(f"Kinematics export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved read-only CS68 kinematics artifact: {destination}")


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
        "depth_intrinsics": (
            [
                frame.calibration.depth.fx,
                frame.calibration.depth.fy,
                frame.calibration.depth.cx,
                frame.calibration.depth.cy,
            ]
            if frame.calibration.depth is not None
            else np.full(4, np.nan)
        ),
        "left_T_depth": (
            frame.calibration.left_t_depth.matrix
            if frame.calibration.left_t_depth is not None
            else np.full((4, 4), np.nan)
        ),
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


@stereo_app.command("doctor")
def stereo_doctor(
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
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate FoundationStereo source, weights, dependencies, and CUDA."""

    settings = load_settings(config)
    results = run_foundation_stereo_doctor(settings.foundation_stereo)
    if output_json:
        typer.echo(json.dumps([asdict(result) for result in results], indent=2))
    else:
        table = Table(title="FoundationStereo doctor")
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


@stereo_app.command("infer-session")
def stereo_infer_session(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
) -> None:
    """Infer and persist calibrated stereo depth from a stored view; no hardware is used."""

    try:
        settings = load_settings(config)
        bundle = SessionReader(session).load_bundle(view_id)
        backend = FoundationStereoBackend(settings.foundation_stereo)
        observation = infer_rectified_stereo(
            bundle,
            backend,
            settings.stereo_rectification,
        )
        destination = write_stereo_inference(
            output,
            observation,
            settings.foundation_stereo,
            settings.stereo_rectification,
            source_session=session,
        )
    except Exception as exc:
        typer.echo(f"Stereo inference failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    valid_count = int(observation.result.valid_mask.sum())
    typer.echo(f"Saved calibrated stereo inference: {destination}")
    typer.echo(f"Valid depth pixels: {valid_count}/{observation.result.valid_mask.size}")


@calibration_app.command("solve-hand-eye")
def calibration_solve_hand_eye(
    samples: Annotated[
        Path,
        typer.Option("--samples", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    method: Annotated[
        str,
        typer.Option(help="OpenCV method: park, tsai, horaud, andreff, or daniilidis."),
    ] = "park",
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
    """Solve and quality-gate eye-in-hand calibration from an offline sample set."""

    try:
        settings = load_settings(config)
        sample_set = read_hand_eye_samples(samples)
        solution = solve_hand_eye(sample_set, settings.hand_eye, method=method)
        destination = write_hand_eye_calibration(output, solution)
    except Exception as exc:
        typer.echo(f"Hand-eye calibration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved hand-eye calibration: {destination}")
    typer.echo(
        "Quality: "
        f"translation RMSE={solution.translation_rmse_m:.6f} m, "
        f"rotation RMSE={solution.rotation_rmse_deg:.3f} deg, "
        f"samples={solution.sample_count}"
    )


@calibration_app.command("extract-hand-eye")
def calibration_extract_hand_eye(
    sessions: Annotated[
        list[Path],
        typer.Option(
            "--session",
            exists=True,
            file_okay=False,
            readable=True,
            help="Stored session directory; repeat for multiple sessions.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
    """Extract eye-in-hand samples from stored views; no hardware is used."""

    try:
        settings = load_settings(config)
        observations = []
        for session in sessions:
            reader = SessionReader(session)
            observations.extend(
                (session, reader.load_bundle(descriptor.sequence_index))
                for descriptor in reader.views
            )
        result = extract_hand_eye_samples(observations, settings.hand_eye.target)
        destination = write_hand_eye_samples(output, result.samples, result.rejected)
    except Exception as exc:
        typer.echo(f"Hand-eye sample extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved hand-eye sample set: {destination}")
    typer.echo(f"Accepted views: {len(result.samples)}; rejected views: {len(result.rejected)}")
    if len(result.samples) < settings.hand_eye.minimum_samples:
        typer.echo(
            f"Need at least {settings.hand_eye.minimum_samples} accepted samples before solving.",
            err=True,
        )
        raise typer.Exit(code=2)


@initialize_app.command("native-depth")
def initialize_from_native_depth(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    mask: Annotated[
        Path,
        typer.Option("--mask", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
) -> None:
    """Build a base-frame initial proxy from one stored, masked D435i depth view."""

    import numpy as np

    try:
        settings = load_settings(config)
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        bundle = SessionReader(session).load_bundle(view_id)
        blade_mask = np.load(mask, allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Blade mask must be a single .npy array")
        observation = initialize_native_depth(
            bundle,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
        )
        destination = write_initialization(
            output,
            observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
            source_session=session,
        )
    except Exception as exc:
        typer.echo(f"Initialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved initialization artifact: {destination}")


@initialize_app.command("stereo-depth")
def initialize_from_stereo_depth(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    stereo: Annotated[
        Path,
        typer.Option("--stereo", exists=True, file_okay=False, readable=True),
    ],
    mask: Annotated[
        Path,
        typer.Option("--mask", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
) -> None:
    """Build an initial proxy from a stored calibrated FoundationStereo result."""

    import numpy as np

    try:
        settings = load_settings(config)
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        bundle = SessionReader(session).load_bundle(view_id)
        stereo_observation = read_stereo_inference(stereo).observation
        blade_mask = np.load(mask, allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Blade mask must be a single .npy array")
        observation = initialize_foundation_stereo_depth(
            bundle,
            stereo_observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
        )
        destination = write_initialization(
            output,
            observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
            source_session=session,
            source_stereo_inference=stereo,
        )
    except Exception as exc:
        typer.echo(f"Initialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved FoundationStereo initialization artifact: {destination}")


@reconstruct_app.command("native-depth")
def reconstruct_from_native_depth(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    mask: Annotated[
        Path,
        typer.Option("--mask", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    view_id: Annotated[str, typer.Option("--view-id")],
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
    """Pose-register a stored native-depth blade view without changing the proxy."""

    import numpy as np

    try:
        settings = load_settings(config)
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        bundle = SessionReader(session).load_bundle(view_id)
        blade_mask = np.load(mask, allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Blade mask must be a single .npy array")
        view = reconstruct_native_depth_view(
            bundle,
            blade_mask,
            hand_eye,
            settings.point_cloud,
        )
        destination = write_reconstructed_view(
            output,
            view,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            source_session=session,
        )
    except Exception as exc:
        typer.echo(f"View reconstruction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved pose-registered blade view: {destination}")


@reconstruct_app.command("stereo-depth")
def reconstruct_from_stereo_depth(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    stereo: Annotated[
        Path,
        typer.Option("--stereo", exists=True, file_okay=False, readable=True),
    ],
    mask: Annotated[
        Path,
        typer.Option("--mask", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    view_id: Annotated[str, typer.Option("--view-id")],
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
    """Pose-register a stored FoundationStereo blade view without changing the proxy."""

    import numpy as np

    try:
        settings = load_settings(config)
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        bundle = SessionReader(session).load_bundle(view_id)
        stereo_observation = read_stereo_inference(stereo).observation
        blade_mask = np.load(mask, allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Blade mask must be a single .npy array")
        view = reconstruct_foundation_stereo_view(
            bundle,
            stereo_observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
        )
        destination = write_reconstructed_view(
            output,
            view,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            source_session=session,
            source_stereo_inference=stereo,
        )
    except Exception as exc:
        typer.echo(f"View reconstruction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved pose-registered stereo blade view: {destination}")


@plan_app.command("views")
def plan_views(
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
    """Create a non-executable bilateral view plan from an initialization artifact."""

    try:
        settings = load_settings(config)
        stored = read_initialization(initialization)
        reachability_checker = None
        if settings.kinematics.model_path is not None:
            kinematics = load_cs68_kinematics(settings.kinematics.model_path)
            reachability_checker = EliteCs68IkChecker(
                kinematics,
                stored.hand_eye,
                stored.observation.seed_joint_positions_rad,
                settings.kinematics,
            )
        result = plan_initial_observation(
            stored.observation,
            settings.view_planning,
            settings.view_filter,
            reachability_checker,
        )
        destination = write_view_plan(
            output,
            result,
            settings.view_planning,
            settings.view_filter,
            source_initialization=initialization,
        )
    except Exception as exc:
        typer.echo(f"View planning failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved offline view plan: {destination}")
    typer.echo(
        f"Candidates: {len(result.geometric_plan.candidates)}, "
        f"geometry accepted: {len(result.filtered_plan.accepted)}, "
        f"endpoint feasible: {len(result.filtered_plan.endpoint_feasible)}"
    )
    typer.echo("Motion authorized: no")


@coverage_app.command("seed")
def coverage_seed(
    plan: Annotated[
        Path,
        typer.Option("--plan", exists=True, file_okay=False, readable=True),
    ],
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
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
    """Create coverage state from the pose-registered initial blade observation."""

    try:
        settings = load_settings(config)
        stored_plan = read_view_plan(plan)
        stored_initialization = read_initialization(initialization)
        expected_initialization = Path(
            str(stored_plan.metadata["source_initialization"])
        ).resolve()
        if expected_initialization != initialization.resolve():
            raise ValueError("View plan was not generated from the supplied initialization")
        observation = stored_initialization.observation
        ledger = create_coverage_ledger(
            stored_plan.result.geometric_plan,
            settings.coverage,
        )
        ledger = update_coverage(
            ledger,
            stored_plan.result.geometric_plan,
            observation.proxy,
            observation.base_cloud,
            observation.base_t_projection_camera,
            coverage_observation_id(
                stored_initialization.metadata["source"]["session"],
                observation.source_view_id,
                observation.source_sequence_index,
                observation.source_frame_number,
            ),
        )
        remaining = select_uncovered_candidates(stored_plan.result.filtered_plan, ledger)
        destination = write_coverage_ledger(
            output,
            ledger,
            source_plan=plan,
            source_initialization=initialization,
        )
    except Exception as exc:
        typer.echo(f"Coverage initialization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved bilateral coverage ledger: {destination}")
    typer.echo(
        f"Completed patches: {len(remaining.completed_patch_ids)}; "
        f"remaining accepted views: {len(remaining.remaining)}; "
        f"blocked patches: {len(remaining.blocked_patch_ids)}"
    )
    typer.echo("Motion authorized: no")


@coverage_app.command("add")
def coverage_add(
    ledger: Annotated[
        Path,
        typer.Option("--ledger", exists=True, file_okay=False, readable=True),
    ],
    plan: Annotated[
        Path,
        typer.Option("--plan", exists=True, file_okay=False, readable=True),
    ],
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    view: Annotated[
        Path,
        typer.Option("--view", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Append one immutable pose-registered view to a new coverage ledger."""

    import numpy as np

    try:
        stored_ledger = read_coverage_ledger(ledger)
        stored_plan = read_view_plan(plan)
        stored_initialization = read_initialization(initialization)
        stored_view = read_reconstructed_view(view)
        if Path(str(stored_ledger.metadata["source_plan"])).resolve() != plan.resolve():
            raise ValueError("Coverage ledger does not belong to the supplied view plan")
        if (
            Path(str(stored_ledger.metadata["source_initialization"])).resolve()
            != initialization.resolve()
        ):
            raise ValueError("Coverage ledger does not belong to the supplied initialization")
        if not np.allclose(
            stored_view.metadata["hand_eye"]["tcp_T_left_ir"],
            stored_initialization.hand_eye.tcp_t_left_ir.matrix,
            atol=1e-9,
        ):
            raise ValueError("Reconstructed view uses a different hand-eye calibration")
        source = stored_view.metadata["source"]
        updated = update_coverage(
            stored_ledger.ledger,
            stored_plan.result.geometric_plan,
            stored_initialization.observation.proxy,
            stored_view.view.base_cloud,
            stored_view.view.base_t_projection_camera,
            coverage_observation_id(
                source["session"],
                stored_view.view.source_view_id,
                stored_view.view.source_sequence_index,
                stored_view.view.source_frame_number,
            ),
        )
        remaining = select_uncovered_candidates(stored_plan.result.filtered_plan, updated)
        destination = write_coverage_ledger(
            output,
            updated,
            source_plan=plan,
            source_initialization=initialization,
            previous_ledger=ledger,
        )
    except Exception as exc:
        typer.echo(f"Coverage update failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved updated bilateral coverage ledger: {destination}")
    typer.echo(
        f"Completed patches: {len(remaining.completed_patch_ids)}; "
        f"remaining accepted views: {len(remaining.remaining)}; "
        f"blocked patches: {len(remaining.blocked_patch_ids)}"
    )
    typer.echo("Motion authorized: no")


@coverage_app.command("next-plan")
def coverage_next_plan(
    ledger: Annotated[
        Path,
        typer.Option("--ledger", exists=True, file_okay=False, readable=True),
    ],
    plan: Annotated[
        Path,
        typer.Option("--plan", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Export remaining, completed, and blocked patches without authorizing motion."""

    try:
        destination = write_coverage_driven_plan(
            output,
            source_plan=plan,
            source_coverage=ledger,
        )
        stored = read_coverage_driven_plan(destination)
    except Exception as exc:
        typer.echo(f"Coverage-driven planning failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved coverage-driven view plan: {destination}")
    typer.echo(
        f"Completed patches: {len(stored.plan.completed_patch_ids)}; "
        f"remaining views: {len(stored.plan.remaining)}; "
        f"blocked patches: {len(stored.plan.blocked_patch_ids)}"
    )
    typer.echo("Motion authorized: no")


@evaluate_app.command("depth-pair")
def evaluate_depth_pair(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    stereo: Annotated[
        Path,
        typer.Option("--stereo", exists=True, file_okay=False, readable=True),
    ],
    mask: Annotated[
        Path,
        typer.Option("--mask", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    view_id: Annotated[str, typer.Option("--view-id")],
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
    """Compare native and stereo depth in the calibrated left-rectified frame."""

    import numpy as np

    try:
        settings = load_settings(config)
        bundle = SessionReader(session).load_bundle(view_id)
        stereo_observation = read_stereo_inference(stereo).observation
        blade_mask = np.load(mask, allow_pickle=False)
        if not isinstance(blade_mask, np.ndarray):
            blade_mask.close()
            raise ValueError("Blade mask must be a single .npy array")
        comparison = compare_paired_depth(
            bundle,
            stereo_observation,
            blade_mask,
            settings.point_cloud,
            settings.depth_comparison,
        )
        destination = write_depth_comparison(
            output,
            comparison,
            settings.point_cloud,
            settings.depth_comparison,
            source_session=session,
            source_stereo_inference=stereo,
            source_blade_mask=mask,
        )
        verified = read_depth_comparison(destination).comparison
    except Exception as exc:
        typer.echo(f"Paired depth evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    metrics = verified.metrics
    typer.echo(f"Saved paired depth comparison: {destination}")
    typer.echo(
        f"Overlap: {metrics.overlap_pixel_count}/{metrics.blade_pixel_count} pixels; "
        f"MAE: {metrics.mean_absolute_error_m * 1000.0:.3f} mm; "
        f"RMSE: {metrics.root_mean_square_error_m * 1000.0:.3f} mm; "
        f"P95: {metrics.p95_absolute_error_m * 1000.0:.3f} mm"
    )
    typer.echo("Native RealSense depth is a comparison reference, not ground truth")


@evaluate_app.command("aggregate-depth")
def evaluate_aggregate_depth(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Aggregate paired comparisons while retaining side and incidence strata."""

    try:
        destination = write_depth_aggregate(output, manifest)
        stored = read_depth_aggregate(destination)
    except Exception as exc:
        typer.echo(f"Depth aggregation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    overall = next(group for group in stored.report.groups if group.group_id == "all")
    typer.echo(f"Saved stratified depth aggregate: {destination}")
    typer.echo(
        f"Views: {overall.metrics.view_count}; "
        f"shared pixels: {overall.metrics.overlap_pixel_count}; "
        f"view-mean MAE: {overall.metrics.view_mean_absolute_error_m * 1000.0:.3f} mm; "
        f"pooled MAE: {overall.metrics.pooled_mean_absolute_error_m * 1000.0:.3f} mm"
    )
    typer.echo("Native RealSense depth is a comparison reference, not ground truth")


if __name__ == "__main__":
    app()
