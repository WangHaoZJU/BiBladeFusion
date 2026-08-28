"""BiBladeFusion command-line interface."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from biblade_fusion import __version__
from biblade_fusion.acquisition import SynchronizedAcquirer
from biblade_fusion.calibration import (
    StereoCalibrationAssetSession,
    StereoValidationAssetSession,
    StereoValidationThresholds,
    fetch_cs68_kinematics,
    load_cs68_kinematics,
    load_hand_eye_calibration,
    load_stereo_calibration,
    read_hand_eye_samples,
    solve_hand_eye,
    solve_stereo_asset_session,
    validate_stereo_asset_session,
    write_cs68_kinematics,
    write_hand_eye_calibration,
    write_hand_eye_samples,
)
from biblade_fusion.core.settings import load_settings
from biblade_fusion.devices.depth_camera import RealSenseD435i, list_realsense_devices
from biblade_fusion.devices.robot import EliteReadOnlyRobot
from biblade_fusion.devices.thermal_camera import NullThermalCamera
from biblade_fusion.diagnostics import CheckLevel, run_doctor
from biblade_fusion.mapping import Es68D435iRobotDepthRenderer, OccupancyMapState
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
from biblade_fusion.robotics import Es68KinematicModel
from biblade_fusion.storage import (
    SessionReader,
    SessionWriter,
    read_coarse_model_summary,
    read_coverage_driven_plan,
    read_coverage_ledger,
    read_depth_aggregate,
    read_depth_comparison,
    read_initialization,
    read_motion_preflight,
    read_native_overlap_report,
    read_occupancy_mapping,
    read_path_validation,
    read_reconstructed_view,
    read_stereo_inference,
    read_view_plan,
    write_coarse_model,
    write_coverage_driven_plan,
    write_coverage_ledger,
    write_depth_aggregate,
    write_depth_aggregate_manifest,
    write_depth_comparison,
    write_initialization,
    write_motion_preflight,
    write_native_overlap_report,
    write_occupancy_mapping,
    write_path_validation,
    write_reconstructed_view,
    write_stereo_inference,
    write_view_plan,
)
from biblade_fusion.workflows import (
    build_coarse_blade_model,
    compare_paired_depth,
    evaluate_native_overlap,
    extract_hand_eye_samples,
    infer_rectified_stereo,
    initialize_foundation_stereo_depth,
    initialize_native_depth,
    integrate_foundation_stereo_occupancy,
    plan_initial_observation,
    reconstruct_foundation_stereo_view,
    reconstruct_native_depth_view,
    registered_cloud_view,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _with_emitter_override(settings, emitter_enabled: bool | None):
    """Return settings with an optional one-command RealSense emitter override."""

    if emitter_enabled is None:
        return settings
    return settings.model_copy(
        update={
            "realsense": settings.realsense.model_copy(
                update={"infrared_emitter_enabled": emitter_enabled}
            )
        }
    )


app = typer.Typer(
    name="bbf",
    help="BiBladeFusion development and acquisition tools.",
    no_args_is_help=True,
)
robot_app = typer.Typer(help="Safe Elite ES68 state tools.", no_args_is_help=True)
camera_app = typer.Typer(help="Intel RealSense D435i tools.", no_args_is_help=True)
acquire_app = typer.Typer(help="Synchronized read-only acquisition.", no_args_is_help=True)
calibration_app = typer.Typer(help="Offline calibration tools.", no_args_is_help=True)
stereo_app = typer.Typer(help="Stereo inference tools.", no_args_is_help=True)
initialize_app = typer.Typer(help="Offline initial-model construction.", no_args_is_help=True)
plan_app = typer.Typer(help="Offline bilateral view planning.", no_args_is_help=True)
coverage_app = typer.Typer(help="Offline bilateral coverage tracking.", no_args_is_help=True)
reconstruct_app = typer.Typer(help="Pose-register stored blade depth views.", no_args_is_help=True)
evaluate_app = typer.Typer(help="Offline experiment evaluation.", no_args_is_help=True)
safety_app = typer.Typer(help="Offline, non-executable safety validation.", no_args_is_help=True)
occupancy_app = typer.Typer(
    help="Depth-derived unknown-environment occupancy assets.",
    no_args_is_help=True,
)
supervise_app = typer.Typer(
    help="Read-only, snapshot-driven supervision and evidence replay.",
    no_args_is_help=True,
)
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
app.add_typer(safety_app, name="safety")
app.add_typer(occupancy_app, name="occupancy")
app.add_typer(supervise_app, name="supervise")


@app.callback()
def main() -> None:
    """BiBladeFusion command group."""


@occupancy_app.command("build-replay")
def occupancy_build_replay(
    stereo_inferences: Annotated[
        list[Path],
        typer.Option(
            "--stereo",
            exists=True,
            file_okay=False,
            readable=True,
            help="Repeat in settled capture order for each FoundationStereo artifact.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
    workspace_min: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--workspace-min", help="Temporary base-frame x y z minimum in metres."),
    ] = None,
    workspace_max: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--workspace-max", help="Temporary base-frame x y z maximum in metres."),
    ] = None,
    voxel_size_m: Annotated[
        float | None,
        typer.Option("--voxel-size-m", min=0.001, max=0.05),
    ] = None,
) -> None:
    """Build an immutable offline/replay map; it is never live motion evidence."""

    try:
        if not stereo_inferences:
            raise ValueError("At least one --stereo artifact is required")
        if (workspace_min is None) != (workspace_max is None):
            raise ValueError("--workspace-min and --workspace-max must be supplied together")
        settings = load_settings(config)
        occupancy_update = {
            "enabled": True,
            **(
                {
                    "workspace_bounds_min_m": workspace_min,
                    "workspace_bounds_max_m": workspace_max,
                }
                if workspace_min is not None and workspace_max is not None
                else {}
            ),
            **({"voxel_size_m": voxel_size_m} if voxel_size_m is not None else {}),
        }
        occupancy_config = type(settings.occupancy).model_validate(
            {**settings.occupancy.model_dump(), **occupancy_update}
        )
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        renderer = Es68D435iRobotDepthRenderer.from_active_resources(
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        )
        snapshot = None
        previous_evidence_hash = None
        updates = []
        source_sessions = []
        for stereo_path in stereo_inferences:
            stored = read_stereo_inference(stereo_path)
            resolved_stereo = Path(stereo_path).resolve()
            source_session = Path(str(stored.metadata["source"]["session"])).resolve()
            reader = SessionReader(source_session)
            bundle = reader.load_bundle(stored.observation.source_view_id)
            captured_at = datetime.fromisoformat(str(reader.manifest["created_at_utc"]))
            descriptor = reader.descriptor(stored.observation.source_view_id)
            relative_view = Path(descriptor.relative_path)
            view_metadata = (source_session / relative_view / "metadata.json").resolve()
            if relative_view.is_absolute() or not view_metadata.is_relative_to(
                source_session
            ):
                raise ValueError("Selected source view metadata escapes its session")
            update = integrate_foundation_stereo_occupancy(
                snapshot,
                bundle,
                stored.observation,
                hand_eye,
                occupancy_config,
                settings.acquisition,
                renderer,
                captured_at_utc=captured_at,
                source_stereo_metadata_sha256=_file_sha256(
                    resolved_stereo / "metadata.json"
                ),
                source_session_manifest_sha256=_file_sha256(
                    source_session / "manifest.json"
                ),
                source_session_view_metadata_sha256=_file_sha256(view_metadata),
                previous_evidence_hash=previous_evidence_hash,
            )
            snapshot = update.snapshot
            previous_evidence_hash = update.evidence.quality_evidence_hash
            updates.append(update)
            source_sessions.append(source_session)
        if snapshot is None:
            raise ValueError("No occupancy snapshot was produced")
        if snapshot.map_state is OccupancyMapState.MAP_READY:
            snapshot = snapshot.mark_stale(
                "offline replay build; acquire a fresh live snapshot before motion"
            )
            updates[-1] = replace(updates[-1], snapshot=snapshot)
        destination = write_occupancy_mapping(
            output,
            tuple(updates),
            occupancy_config,
            settings.acquisition,
            source_stereo_inferences=tuple(stereo_inferences),
            source_sessions=tuple(source_sessions),
            source_hand_eye=hand_eye.source_path,
        )
        stored_output = read_occupancy_mapping(destination)
    except Exception as exc:
        typer.echo(f"Occupancy replay build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved immutable occupancy replay asset: {destination}")
    typer.echo(
        f"Map: {stored_output.snapshot.version}; state: "
        f"{stored_output.snapshot.map_state.value}; views: "
        f"{len(stored_output.snapshot.source_view_ids)}"
    )
    typer.echo("Motion authorized: no (offline/replay occupancy is deliberately stale)")


@occupancy_app.command("inspect")
def occupancy_inspect(
    artifact: Annotated[
        Path,
        typer.Option("--artifact", exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Verify and summarize a stored occupancy-mapping digital asset."""

    try:
        stored = read_occupancy_mapping(artifact)
    except Exception as exc:
        typer.echo(f"Occupancy inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    snapshot = stored.snapshot
    typer.echo(f"Version: {snapshot.version}")
    typer.echo(f"State: {snapshot.map_state.value}")
    typer.echo(
        f"Known/free/occupied/unknown voxels: {snapshot.known_voxel_count}/"
        f"{len(snapshot.free_indices)}/{len(snapshot.occupied_indices)}/"
        f"{snapshot.unknown_voxel_count}"
    )
    typer.echo(f"Source views: {', '.join(snapshot.source_view_ids)}")
    typer.echo("Usable for motion: no unless a live process independently proves freshness")


@supervise_app.command("replay")
def supervise_replay(
    snapshot: Annotated[
        Path,
        typer.Option(
            "--snapshot",
            exists=True,
            readable=True,
            help=(
                "A snapshot JSON/directory, or a directory containing an ordered "
                "snapshot timeline."
            ),
        ),
    ],
    interval_ms: Annotated[
        int,
        typer.Option(
            "--interval-ms",
            min=100,
            help="Replay interval for multi-snapshot timelines.",
        ),
    ] = 800,
    follow: Annotated[
        bool,
        typer.Option(
            "--follow",
            help=(
                "Read-only poll a timeline root for atomically published child snapshots; "
                "this is not live obstacle avoidance."
            ),
        ),
    ] = False,
) -> None:
    """Open the immutable read-only console; never connect to command ports."""

    try:
        from biblade_fusion.supervision.gui import launch_supervisory_console

        return_code = launch_supervisory_console(
            snapshot,
            replay_interval_ms=interval_ms,
            follow=follow,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            typer.echo(
                "PySide6 is not installed; run `uv sync --extra supervision-gui`.",
                err=True,
            )
        else:
            typer.echo(f"Supervisory replay failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Supervisory replay failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if return_code:
        raise typer.Exit(code=return_code)


@supervise_app.command("build-replay")
def supervise_build_replay(
    occupancy: Annotated[
        Path,
        typer.Option(
            "--occupancy",
            exists=True,
            file_okay=False,
            readable=True,
            help="Required immutable occupancy-mapping artifact.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="New self-contained replay snapshot directory."),
    ],
    stereo: Annotated[
        Path | None,
        typer.Option(
            "--stereo",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional explicit latest stereo artifact; otherwise follow occupancy provenance.",
        ),
    ] = None,
    current_view: Annotated[
        Path | None,
        typer.Option(
            "--current-view",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional pose-registered view matching the occupancy map's latest frame.",
        ),
    ] = None,
    coarse_model: Annotated[
        Path | None,
        typer.Option(
            "--coarse-model",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional verified multi-view coarse-model artifact.",
        ),
    ] = None,
    preflight: Annotated[
        Path | None,
        typer.Option(
            "--preflight",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional historical preflight bound to the same occupancy asset.",
        ),
    ] = None,
) -> None:
    """Build an immutable offline replay snapshot; never authorize motion."""

    try:
        from biblade_fusion.workflows.supervision_replay import (
            build_supervisory_replay_snapshot,
        )

        stored = build_supervisory_replay_snapshot(
            output,
            source_occupancy=occupancy,
            source_stereo_inference=stereo,
            source_reconstructed_view=current_view,
            source_coarse_model=coarse_model,
            source_motion_preflight=preflight,
        )
    except Exception as exc:
        typer.echo(f"Supervisory replay build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    snapshot = stored.snapshot
    typer.echo(f"Saved immutable supervisory replay snapshot: {stored.root}")
    typer.echo(
        f"Map: {snapshot.occupancy.version}; state: {snapshot.occupancy.state}; "
        f"occupied/free: {snapshot.occupancy.occupied_centres_m.shape[0]}/"
        f"{snapshot.occupancy.free_centres_m.shape[0]}"
    )
    typer.echo("Viewer mode: REPLAY; system state: BLOCKED; motion authorized: no")


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
    """Read one ES68 RTSI state snapshot; this command cannot move the robot."""

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
    console.print("[bold]Elite ES68 read-only RTSI status[/bold]")
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
    """Read and store ES68 MDH parameters; this command cannot move the robot."""

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
    typer.echo(f"Saved read-only ES68 kinematics artifact: {destination}")


@robot_app.command("inspect-model")
def robot_inspect_model(
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
    joints_deg: Annotated[
        str,
        typer.Option(
            "--joints-deg",
            help="Six comma- or space-separated controller joint angles in degrees.",
        ),
    ] = "0,0,0,0,0,0",
) -> None:
    """Open the completely offline ES68+D435i articulated STL inspector."""

    try:
        settings = load_settings(config)
        from biblade_fusion.robotics.model_gui import (
            launch_es68_d435i_model_gui,
            parse_joint_degrees,
        )

        initial_joints_rad = parse_joint_degrees(joints_deg)
        exit_code = launch_es68_d435i_model_gui(
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
            initial_joint_positions_rad=initial_joints_rad,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6" or (exc.name or "").startswith("PySide6."):
            typer.echo(
                "PySide6/Qt3D is not installed; run "
                "`uv sync --extra robot-model-gui`.",
                err=True,
            )
        else:
            typer.echo(f"Offline robot-model viewer failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Offline robot-model viewer failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if exit_code:
        raise typer.Exit(code=exit_code)


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
    emitter: Annotated[
        bool | None,
        typer.Option(
            "--emitter/--no-emitter",
            help="Temporarily override the D435i projector for this capture only.",
        ),
    ] = None,
) -> None:
    """Capture one synchronized raw D435i frame bundle."""

    import numpy as np

    settings = _with_emitter_override(load_settings(config), emitter)
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
    emitter: Annotated[
        bool | None,
        typer.Option(
            "--emitter/--no-emitter",
            help="Temporarily override the D435i projector for this session only.",
        ),
    ] = None,
) -> None:
    """Capture one D435i frame bracketed by read-only ES68 states."""

    settings = _with_emitter_override(load_settings(config), emitter)
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
            source_stereo_calibration=settings.realsense.stereo_calibration_path,
        )
    except Exception as exc:
        typer.echo(f"Stereo inference failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    valid_count = int(observation.result.valid_mask.sum())
    typer.echo(f"Saved calibrated stereo inference: {destination}")
    typer.echo(f"Valid depth pixels: {valid_count}/{observation.result.valid_mask.size}")


@calibration_app.command("stereo-gui")
def calibration_stereo_gui(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Asset collection root; every launch creates a unique session directory.",
        ),
    ],
    target: Annotated[
        Path,
        typer.Option("--target", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml"),
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
    """Capture raw D435i IR assets, then detect and calibrate offline."""

    try:
        from biblade_fusion.calibration.stereo_gui import launch_stereo_calibration_gui

        settings = load_settings(config)
        raise_code = launch_stereo_calibration_gui(target, output, settings.realsense)
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            typer.echo("PySide6 is not installed; run `uv sync --extra calibration-gui`.", err=True)
        else:
            typer.echo(f"Stereo calibration GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Stereo calibration GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raise_code:
        raise typer.Exit(code=raise_code)


@calibration_app.command("stereo-solve-assets")
def calibration_stereo_solve_assets(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
    minimum_samples: Annotated[
        int,
        typer.Option("--minimum-samples", min=10, max=100),
    ] = 20,
    distortion_model: Annotated[
        str,
        typer.Option(
            "--distortion-model",
            help="auto, radial2, brown5, or rational8",
        ),
    ] = "auto",
) -> None:
    """Re-run offline ChArUco detection and solving from one stored asset session."""

    try:
        assets = StereoCalibrationAssetSession.open(session)
        detection, result, output = solve_stereo_asset_session(
            assets,
            minimum_samples=minimum_samples,
            distortion_model=distortion_model,
            runtime_calibration_path=Path("data/calibrations/d435i_ir_active.yaml"),
        )
    except Exception as exc:
        typer.echo(f"Stereo asset calibration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Accepted pairs: {len(detection.accepted_pair_ids)}")
    typer.echo(f"Rejected pairs: {len(detection.rejected_pair_ids)}")
    typer.echo(f"Selected distortion model: {result.distortion_model.value}")
    typer.echo(f"Joint stereo RMS: {result.metrics.joint_stereo_rms_px:.6f} px")
    typer.echo(f"Epipolar RMSE: {result.metrics.epipolar_rmse_px:.6f} px")
    typer.echo(f"Saved calibration: {output}")


@calibration_app.command("stereo-validate-gui")
def calibration_stereo_validate_gui(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Validation collection root; every launch creates a unique session.",
        ),
    ],
    target: Annotated[
        Path,
        typer.Option("--target", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml"),
    calibration: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Fixed user stereo calibration; defaults to realsense config path.",
        ),
    ] = None,
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
    minimum_pairs: Annotated[
        int,
        typer.Option("--minimum-pairs", min=3, max=100),
    ] = 8,
    maximum_vertical_rmse_px: Annotated[
        float,
        typer.Option("--maximum-vertical-rmse-px", min=0.01),
    ] = 0.5,
    maximum_vertical_p95_px: Annotated[
        float,
        typer.Option("--maximum-vertical-p95-px", min=0.01),
    ] = 1.0,
    maximum_reprojection_rmse_px: Annotated[
        float,
        typer.Option("--maximum-reprojection-rmse-px", min=0.01),
    ] = 0.5,
    maximum_stereo_transfer_rmse_px: Annotated[
        float,
        typer.Option("--maximum-stereo-transfer-rmse-px", min=0.01),
    ] = 1.0,
) -> None:
    """Capture hold-out ChArUco pairs and validate fixed stereo parameters."""

    try:
        from biblade_fusion.calibration.stereo_validation_gui import (
            launch_stereo_validation_gui,
        )

        settings = load_settings(config)
        calibration_path = calibration or settings.realsense.stereo_calibration_path
        if calibration_path is None:
            raise ValueError(
                "--calibration or realsense.stereo_calibration_path is required"
            )
        thresholds = StereoValidationThresholds(
            minimum_accepted_pairs=minimum_pairs,
            maximum_vertical_disparity_rmse_px=maximum_vertical_rmse_px,
            maximum_vertical_disparity_p95_px=maximum_vertical_p95_px,
            maximum_monocular_reprojection_rmse_px=maximum_reprojection_rmse_px,
            maximum_stereo_transfer_rmse_px=maximum_stereo_transfer_rmse_px,
        )
        raise_code = launch_stereo_validation_gui(
            target,
            calibration_path,
            output,
            settings.realsense,
            settings.stereo_rectification,
            thresholds,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            typer.echo(
                "PySide6 is not installed; run `uv sync --extra calibration-gui`.",
                err=True,
            )
        else:
            typer.echo(f"Stereo validation GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Stereo validation GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raise_code:
        raise typer.Exit(code=raise_code)


@calibration_app.command("stereo-validate-assets")
def calibration_stereo_validate_assets(
    session: Annotated[
        Path,
        typer.Option("--session", exists=True, file_okay=False, readable=True),
    ],
) -> None:
    """Re-run fixed-parameter validation from one preserved validation session."""

    try:
        assets = StereoValidationAssetSession.open(session)
        result = validate_stereo_asset_session(assets)
    except Exception as exc:
        typer.echo(f"Stereo validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    metrics = result.metrics
    typer.echo(f"Result: {'PASS' if metrics.passed else 'FAIL'}")
    typer.echo(
        f"Accepted/rejected pairs: {metrics.accepted_pair_count}/"
        f"{metrics.rejected_pair_count}"
    )
    typer.echo(
        "Vertical disparity RMSE/P95/max: "
        f"{metrics.vertical_disparity_rmse_px:.6f}/"
        f"{metrics.vertical_disparity_p95_px:.6f}/"
        f"{metrics.vertical_disparity_max_px:.6f} px"
    )
    typer.echo(
        "Left/right reprojection RMSE: "
        f"{metrics.left_reprojection_rmse_px:.6f}/"
        f"{metrics.right_reprojection_rmse_px:.6f} px"
    )
    typer.echo(f"Stereo transfer RMSE: {metrics.stereo_transfer_rmse_px:.6f} px")
    typer.echo(f"Saved report: {result.report_json}")


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
    stereo_calibration: Annotated[
        Path | None,
        typer.Option(
            "--stereo-calibration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="User-calibrated D435i IR stereo YAML; defaults to realsense config.",
        ),
    ] = None,
    bundle_adjustment: Annotated[
        bool,
        typer.Option("--ba/--no-ba", help="Run HoloRobot-aligned LM reprojection refinement."),
    ] = True,
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
    """Solve ES68 flange-to-left-IR hand-eye calibration from stored samples."""

    try:
        settings = load_settings(config)
        sample_set = read_hand_eye_samples(samples)
        calibration_path = stereo_calibration or settings.realsense.stereo_calibration_path
        if bundle_adjustment and calibration_path is None:
            raise ValueError(
                "--stereo-calibration or realsense.stereo_calibration_path is required for BA"
            )
        intrinsics = (
            load_stereo_calibration(calibration_path).left if calibration_path is not None else None
        )
        solution = solve_hand_eye(
            sample_set,
            settings.hand_eye,
            method=method,
            intrinsics=intrinsics,
            refine=bundle_adjustment,
        )
        destination = write_hand_eye_calibration(
            output,
            solution,
            intrinsics=intrinsics,
            stereo_calibration_path=calibration_path,
        )
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
    if solution.bundle_adjustment.enabled:
        typer.echo(
            "BA: "
            f"{solution.bundle_adjustment.initial_rmse_px:.4f} -> "
            f"{solution.bundle_adjustment.final_rmse_px:.4f} px"
        )


@calibration_app.command("hand-eye-gui")
def calibration_hand_eye_gui(
    output: Annotated[Path, typer.Option("--output", "-o")],
    target: Annotated[
        Path,
        typer.Option("--target", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml"),
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
    """Open synchronized ES68 + D435i raw-left-IR hand-eye calibration."""

    try:
        from biblade_fusion.calibration.hand_eye_gui import (
            launch_hand_eye_calibration_gui,
        )

        settings = load_settings(config)
        raise_code = launch_hand_eye_calibration_gui(
            target,
            output,
            settings.robot,
            settings.realsense,
            settings.acquisition,
            settings.hand_eye,
            settings.kinematics,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            typer.echo(
                "PySide6 is not installed; run `uv sync --extra calibration-gui`.",
                err=True,
            )
        else:
            typer.echo(f"Hand-eye calibration GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Hand-eye calibration GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raise_code:
        raise typer.Exit(code=raise_code)


@calibration_app.command("hand-eye-validate-gui")
def calibration_hand_eye_validate_gui(
    calibration: Annotated[
        Path,
        typer.Option(
            "--calibration",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Frozen schema-2 hand-eye YAML; parameters are never refit.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    target: Annotated[
        Path,
        typer.Option("--target", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml"),
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
    """Collect new held-out poses against one frozen hand-eye result."""

    try:
        from biblade_fusion.calibration.hand_eye_gui import (
            launch_hand_eye_calibration_gui,
        )

        settings = load_settings(config)
        raise_code = launch_hand_eye_calibration_gui(
            target,
            output,
            settings.robot,
            settings.realsense,
            settings.acquisition,
            settings.hand_eye,
            settings.kinematics,
            validation_calibration_path=calibration,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            typer.echo(
                "PySide6 is not installed; run `uv sync --extra calibration-gui`.",
                err=True,
            )
        else:
            typer.echo(f"Hand-eye validation GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Hand-eye validation GUI failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if raise_code:
        raise typer.Exit(code=raise_code)


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
        result = extract_hand_eye_samples(
            observations,
            settings.hand_eye.target,
            Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad
            ),
        )
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
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
        destination = write_initialization(
            output,
            observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
            settings.kinematics,
            settings.hand_eye,
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
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
        destination = write_initialization(
            output,
            observation,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.proxy_model,
            settings.kinematics,
            settings.hand_eye,
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
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
        destination = write_reconstructed_view(
            output,
            view,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.kinematics,
            settings.hand_eye,
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
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
        destination = write_reconstructed_view(
            output,
            view,
            blade_mask,
            hand_eye,
            settings.point_cloud,
            settings.kinematics,
            settings.hand_eye,
            source_session=session,
            source_stereo_inference=stereo,
        )
    except Exception as exc:
        typer.echo(f"View reconstruction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved pose-registered stereo blade view: {destination}")


@reconstruct_app.command("coarse-model")
def reconstruct_coarse_model(
    views: Annotated[
        list[Path],
        typer.Option(
            "--view",
            exists=True,
            file_okay=False,
            readable=True,
            help="Repeat for each front/back pose-registered coarse-scan artifact.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Fuse coarse scans, partition true surfaces, integrate TSDF, and plan fine views."""

    import numpy as np

    try:
        if len(views) < 2:
            raise ValueError("At least two reconstructed views are required")
        settings = load_settings(config)
        stored = tuple(read_reconstructed_view(path) for path in views)
        reference_hand_eye = np.asarray(stored[0].metadata["hand_eye"]["tcp_T_left_ir"])
        if any(
            not np.allclose(
                item.metadata["hand_eye"]["tcp_T_left_ir"], reference_hand_eye, atol=1e-9
            )
            for item in stored[1:]
        ):
            raise ValueError("Coarse views use different hand-eye calibration matrices")
        planning_intrinsics = stored[0].view.planning_intrinsics
        if any(item.view.planning_intrinsics != planning_intrinsics for item in stored[1:]):
            raise ValueError("Coarse views use different left-IR planning intrinsics")
        result = build_coarse_blade_model(
            tuple(registered_cloud_view(item.view) for item in stored),
            planning_intrinsics,
            settings.multi_view_fusion,
            settings.surface_partition,
            settings.view_planning,
            settings.tsdf,
            settings.surface_quality,
        )
        destination = write_coarse_model(
            output,
            result,
            settings,
            source_views=tuple(views),
        )
        verified = read_coarse_model_summary(destination).metadata
    except Exception as exc:
        typer.echo(f"Coarse-model reconstruction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    quality = verified["quality"]
    typer.echo(f"Saved paper-derived coarse model: {destination}")
    typer.echo(
        f"Patches: {len(verified['surface']['patches'])}; "
        f"fine views: {len(verified['view_plan']['candidate_ids'])}; "
        f"mesh triangles: {quality['mesh_triangle_count']}; "
        f"coarse coverage: {quality['completion_fraction']:.3f}"
    )
    typer.echo("Motion authorized: no")


@reconstruct_app.command("inspect-fine-plan")
def reconstruct_inspect_fine_plan(
    coarse_model: Annotated[
        Path,
        typer.Option(
            "--coarse-model",
            exists=True,
            file_okay=False,
            readable=True,
            help="Schema-4 paper-derived coarse-model artifact.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    gui: Annotated[
        bool,
        typer.Option("--gui/--no-gui", help="Open the read-only PySide6 orbit viewer."),
    ] = True,
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Audit fine-view geometry and export portable inspection evidence."""

    try:
        from biblade_fusion.storage.fine_plan_inspection import (
            read_fine_plan_inspection,
            write_fine_plan_inspection,
        )
        from biblade_fusion.workflows.fine_plan_inspection import inspect_fine_plan

        settings = load_settings(config)
        summary = read_coarse_model_summary(coarse_model)
        inspection = inspect_fine_plan(summary, settings.view_filter)
        destination = write_fine_plan_inspection(output, inspection)
        verified = read_fine_plan_inspection(destination).metadata
    except Exception as exc:
        typer.echo(f"Fine-plan inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    accepted = sum(bool(item["accepted"]) for item in verified["views"])
    typer.echo(f"Saved fine-plan inspection: {destination}")
    typer.echo(
        f"Geometry passed: {'yes' if verified['geometry_passed'] else 'no'}; "
        f"accepted views: {accepted}/{len(verified['views'])}"
    )
    typer.echo("Robot feasibility: unverified; motion authorized: no")
    if gui:
        try:
            from biblade_fusion.planning.fine_plan_gui import (
                launch_fine_plan_inspection_gui,
            )

            raise_code = launch_fine_plan_inspection_gui(destination)
        except ModuleNotFoundError as exc:
            if exc.name == "PySide6":
                typer.echo(
                    "PySide6 is not installed; run `uv sync --extra calibration-gui` "
                    "or use --no-gui.",
                    err=True,
                )
            else:
                typer.echo(f"Fine-plan viewer failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(f"Fine-plan viewer failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if raise_code:
            raise typer.Exit(code=raise_code)
    if not verified["geometry_passed"]:
        raise typer.Exit(code=2)


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
            source_kinematics=settings.kinematics.model_path,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
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
        expected_initialization = Path(str(stored_plan.metadata["source_initialization"])).resolve()
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
        if (
            stored_initialization.hand_eye.flange_t_left_ir is None
            or not np.allclose(
                stored_view.metadata["hand_eye"]["flange_T_left_ir"],
                stored_initialization.hand_eye.flange_t_left_ir.matrix,
                atol=1e-9,
            )
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


@evaluate_app.command("native-overlap")
def evaluate_native_overlap_command(
    reference: Annotated[
        Path,
        typer.Option("--reference", exists=True, file_okay=False, readable=True),
    ],
    sessions: Annotated[
        list[Path],
        typer.Option(
            "--session",
            exists=True,
            file_okay=False,
            readable=True,
            help="Comparison session; repeat once per static-scene view.",
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
    """Validate static native-depth overlap without applying ICP pose corrections."""

    try:
        settings = load_settings(config)
        hand_eye = load_hand_eye_calibration(settings.hand_eye)
        source_sessions = (reference, *sessions)
        if len({path.resolve() for path in source_sessions}) != len(source_sessions):
            raise ValueError("Native-overlap session paths must be unique")
        readers = tuple(SessionReader(path) for path in source_sessions)
        if any(len(reader.views) != 1 for reader in readers):
            raise ValueError("Native-overlap currently requires one immutable view per session")
        bundles = tuple(reader.load_bundle(reader.views[0].view_id) for reader in readers)
        report = evaluate_native_overlap(
            bundles,
            hand_eye,
            settings.point_cloud,
            settings.native_overlap_validation,
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
        destination = write_native_overlap_report(
            output,
            report,
            source_sessions,
            hand_eye,
            settings.hand_eye,
            settings.kinematics,
            settings.point_cloud,
            settings.native_overlap_validation,
        )
        verified = read_native_overlap_report(destination).report
    except Exception as exc:
        typer.echo(f"Native-overlap validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    table = Table(title="Static native-depth overlap (primary metrics: no ICP)")
    table.add_column("Comparison")
    table.add_column("Inliers")
    table.add_column("Median mm")
    table.add_column("RMSE mm")
    table.add_column("P95 mm")
    table.add_column("≤5 mm")
    table.add_column("ICP diagnostic")
    table.add_column("Result")
    for pair in verified.pairs:
        metrics = pair.metrics
        agreement = dict(metrics.agreement_fractions)[0.005]
        diagnostic = pair.icp_diagnostic
        diagnostic_text = "disabled"
        if diagnostic is not None:
            diagnostic_text = (
                f"{diagnostic.translation_correction_m * 1000:.2f} mm/"
                f"{diagnostic.rotation_correction_deg:.3f}°"
                if diagnostic.converged
                and diagnostic.translation_correction_m is not None
                and diagnostic.rotation_correction_deg is not None
                else diagnostic.reason
            )
        table.add_row(
            pair.comparison_view_id,
            f"{metrics.surface_inlier_fraction * 100:.2f}%",
            f"{metrics.median_absolute_error_m * 1000:.3f}",
            f"{metrics.root_mean_square_error_m * 1000:.3f}",
            f"{metrics.p95_absolute_error_m * 1000:.3f}",
            f"{agreement * 100:.2f}%",
            diagnostic_text,
            "PASS" if metrics.passed else "FAIL",
        )
    Console().print(table)
    typer.echo(
        f"Pose span: {verified.translation_span_m * 1000:.2f} mm, "
        f"{verified.rotation_span_deg:.3f} deg"
    )
    typer.echo(f"Saved immutable native-overlap report: {destination}")
    typer.echo("ICP changed primary metrics: no")
    if not verified.passed:
        typer.echo("Validation result: FAIL", err=True)
        for reason in verified.failure_reasons:
            typer.echo(f"- {reason}", err=True)
        raise typer.Exit(code=2)
    typer.echo("Validation result: PASS")


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


@evaluate_app.command("make-depth-manifest")
def evaluate_make_depth_manifest(
    comparisons: Annotated[
        list[Path],
        typer.Option(
            "--comparison",
            exists=True,
            file_okay=False,
            readable=True,
            help="Repeat for every paired depth-comparison artifact.",
        ),
    ],
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Label paired comparisons from achieved poses and the fixed blade proxy."""

    try:
        settings = load_settings(config)
        destination = write_depth_aggregate_manifest(
            output,
            tuple(comparisons),
            initialization,
            settings.depth_comparison,
        )
    except Exception as exc:
        typer.echo(f"Depth manifest generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Saved achieved-pose depth manifest: {destination}")


@safety_app.command("validate-path")
def safety_validate_path(
    plan: Annotated[
        Path,
        typer.Option("--plan", exists=True, file_okay=False, readable=True),
    ],
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    view_ids: Annotated[
        list[str],
        typer.Option(
            "--view-id",
            help="Repeat in the exact traversal order to validate.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Validate an explicit sequence without connecting to or moving the robot."""

    try:
        settings = load_settings(config)
        if settings.kinematics.model_path is None:
            raise ValueError("kinematics.model_path must be configured")
        destination = write_path_validation(
            output,
            tuple(view_ids),
            settings.collision,
            source_plan=plan,
            source_initialization=initialization,
            source_kinematics=settings.kinematics.model_path,
        )
        stored = read_path_validation(destination)
    except Exception as exc:
        typer.echo(f"Path safety validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finding_count = sum(len(leg.collision.findings) for leg in stored.report.legs)
    typer.echo(f"Saved non-executable path validation: {destination}")
    typer.echo(
        f"Legs: {len(stored.report.legs)}; findings: {finding_count}; "
        f"collision free: {'yes' if stored.report.collision_free else 'no'}"
    )
    typer.echo("Motion authorized: no")


@safety_app.command("preflight-path")
def safety_preflight_path(
    plan: Annotated[
        Path,
        typer.Option("--plan", exists=True, file_okay=False, readable=True),
    ],
    initialization: Annotated[
        Path,
        typer.Option("--initialization", exists=True, file_okay=False, readable=True),
    ],
    occupancy: Annotated[
        Path,
        typer.Option(
            "--occupancy",
            exists=True,
            file_okay=False,
            readable=True,
            help="Versioned occupancy-mapping artifact bound to this preflight.",
        ),
    ],
    view_ids: Annotated[
        list[str],
        typer.Option(
            "--view-id",
            help="Repeat in the exact traversal order to preflight.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/default.yaml"),
) -> None:
    """Persist HoloRobot mesh/ServoJ preflight without connecting to the robot."""

    try:
        settings = load_settings(config)
        destination = write_motion_preflight(
            output,
            tuple(view_ids),
            settings.motion_preflight,
            settings.collision,
            settings.occupancy,
            source_plan=plan,
            source_initialization=initialization,
            source_occupancy=occupancy,
            joint_zero_offsets_rad=settings.kinematics.joint_zero_offsets_rad,
        )
        stored = read_motion_preflight(destination)
    except Exception as exc:
        typer.echo(f"Motion preflight failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    blocking_count = sum(len(leg.preflight.blocking_reasons) for leg in stored.report.legs)
    typer.echo(f"Saved non-executable motion preflight: {destination}")
    typer.echo(
        f"Legs: {len(stored.report.legs)}; blocking reasons: {blocking_count}; "
        "ready for approval: "
        f"{'yes' if stored.report.ready_for_approval else 'no'}"
    )
    typer.echo(
        "Estimated ServoJ duration: "
        f"{stored.report.cost.estimated_servoj_duration_s:.3f} s; "
        "joint travel L1: "
        f"{stored.report.cost.total_joint_travel_l1_rad:.3f} rad"
    )
    typer.echo("Motion authorized: no")


if __name__ == "__main__":
    app()
