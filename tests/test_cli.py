import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

import biblade_fusion.cli as cli_module
import biblade_fusion.storage.runtime_timing_acceptance as timing_acceptance_module
import biblade_fusion.storage.science_acceptance as science_acceptance_module
import biblade_fusion.workflows.unknown_blade_runtime as unknown_runtime_module
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


def test_supervised_scan_preparation_commands_are_exposed() -> None:
    result = runner.invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "bootstrap-mask" in result.stdout
    assert "run-unknown" in result.stdout

    bootstrap_help = runner.invoke(app, ["scan", "bootstrap-mask", "--help"])
    assert bootstrap_help.exit_code == 0
    assert "--rectangle" in bootstrap_help.stdout
    assert "--polygon" in bootstrap_help.stdout
    assert "--seed-mode" in bootstrap_help.stdout

    runtime_help = runner.invoke(app, ["scan", "run-unknown", "--help"])
    assert runtime_help.exit_code == 0
    assert "--operator-id" in runtime_help.stdout
    assert "--output" in runtime_help.stdout
    assert "--resume" in runtime_help.stdout


def test_unknown_scan_cli_forwards_identity_and_output_without_hidden_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    fake_runtime = object()

    @contextmanager
    def fake_open(settings, *, output_root, operator_id, run_id, resume):
        calls.append((settings.robot.model, Path(output_root), operator_id, run_id, resume))
        yield fake_runtime

    def fake_console(runtime) -> int:
        assert runtime is fake_runtime
        calls.append(("console",))
        return 0

    monkeypatch.setattr(
        unknown_runtime_module,
        "open_production_unknown_blade_runtime",
        fake_open,
    )
    monkeypatch.setattr(
        unknown_runtime_module,
        "run_unknown_blade_operator_console",
        fake_console,
    )
    output = tmp_path / "new-run"

    result = runner.invoke(
        app,
        [
            "scan",
            "run-unknown",
            "--config",
            "configs/default.yaml",
            "--output",
            str(output),
            "--operator-id",
            "operator-7",
            "--run-id",
            "blade-run-7",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("es68", output, "operator-7", "blade-run-7", False),
        ("console",),
    ]


def test_unknown_scan_cli_forwards_explicit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    @contextmanager
    def fake_open(_settings, *, output_root, operator_id, run_id, resume):
        assert Path(output_root) == tmp_path / "existing-run"
        assert operator_id == "operator-7"
        assert run_id is None
        calls.append(resume)
        yield object()

    monkeypatch.setattr(
        unknown_runtime_module,
        "open_production_unknown_blade_runtime",
        fake_open,
    )
    monkeypatch.setattr(
        unknown_runtime_module,
        "run_unknown_blade_operator_console",
        lambda _runtime: 0,
    )

    result = runner.invoke(
        app,
        [
            "scan",
            "run-unknown",
            "--config",
            "configs/default.yaml",
            "--output",
            str(tmp_path / "existing-run"),
            "--operator-id",
            "operator-7",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert calls == [True]


def test_unknown_scan_doctor_routes_complete_coarse_to_fine_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        unknown_runtime_module,
        "unknown_blade_runtime_readiness",
        lambda settings: calls.append(settings.robot.model) or (),
    )

    result = runner.invoke(
        app,
        [
            "scan",
            "doctor",
            "--mode",
            "unknown",
            "--config",
            "configs/default.yaml",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["es68"]
    assert result.stdout.strip() == "[]"


def test_static_free_acceptance_recording_command_is_exposed() -> None:
    result = runner.invoke(app, ["safety", "--help"])

    assert result.exit_code == 0
    assert "record-static-free-acceptance" in result.stdout
    assert "build-runtime-timing-report" in result.stdout
    assert "record-runtime-timing-acceptance" in result.stdout


def test_science_acceptance_recording_refuses_insufficient_runtime_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        Path("configs/science_acceptance.template.json").read_text(encoding="utf-8")
    )
    payload["test_envelope"] = {
        "minimum_distance_m": 0.15,
        "maximum_distance_m": 0.75,
        "minimum_incidence_deg": 0.0,
        "maximum_incidence_deg": 75.0,
    }
    declaration = tmp_path / "science.json"
    declaration.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        science_acceptance_module,
        "write_science_acceptance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("insufficient acceptance must not be recorded")
        ),
    )

    result = runner.invoke(
        app,
        [
            "safety",
            "record-science-acceptance",
            "--config",
            "configs/default.yaml",
            "--declaration",
            str(declaration),
            "--output",
            str(tmp_path / "acceptance"),
        ],
    )

    assert result.exit_code == 1
    assert "maximum runtime depth" in result.stderr
    assert not (tmp_path / "acceptance").exists()


def test_science_runtime_contract_export_command_is_exposed() -> None:
    result = runner.invoke(app, ["safety", "--help"])

    assert result.exit_code == 0
    assert "science-runtime-contract" in result.stdout


def test_science_evidence_canonicalization_command_is_strict_and_nonmoving(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.pretty.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_type": "biblade_fusion.raw_science_acceptance_asset_manifest",
                "generator": {"name": "asset-indexer", "version": "1.0"},
                "created_at_utc": "2026-08-29T00:00:00+00:00",
                "assets": [
                    {
                        "asset_id": "annotation-001",
                        "role": "annotation",
                        "archive_path": "annotations/001.json",
                        "sha256": "1" * 64,
                        "size_bytes": 1,
                    },
                    {
                        "asset_id": "depth-001",
                        "role": "depth_reference",
                        "archive_path": "depth/001.npz",
                        "sha256": "2" * 64,
                        "size_bytes": 2,
                    },
                    {
                        "asset_id": "specimen-001",
                        "role": "specimen",
                        "archive_path": "specimens/001.json",
                        "sha256": "3" * 64,
                        "size_bytes": 3,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "raw.canonical.json"

    result = runner.invoke(
        app,
        [
            "safety",
            "canonicalize-science-evidence",
            "--kind",
            "raw-manifest",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes().endswith(b"\n")
    assert b"\n " not in output.read_bytes()
    assert "SHA-256:" in result.stdout
    assert "Size bytes:" in result.stdout
    assert "Motion authorized: no" in result.stdout

    second = runner.invoke(
        app,
        [
            "safety",
            "canonicalize-science-evidence",
            "--kind",
            "raw-manifest",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )
    assert second.exit_code == 1
    assert "File exists" in second.stderr


def test_runtime_timing_acceptance_recording_command_writes_strict_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_text = Path("configs/default.yaml").read_text(encoding="utf-8")
    replacements = {
        "maximum_perception_cycle_duration_s: null": ("maximum_perception_cycle_duration_s: 4.0"),
        "maximum_operator_reposition_interval_s: null": (
            "maximum_operator_reposition_interval_s: 30.0"
        ),
        "maximum_segment_execution_duration_s: null": (
            "maximum_segment_execution_duration_s: 12.0"
        ),
        "maximum_schema5_handoff_duration_s: null": ("maximum_schema5_handoff_duration_s: 20.0"),
    }
    for old, new in replacements.items():
        assert config_text.count(old) == 1
        config_text = config_text.replace(old, new)
    config = tmp_path / "settings.yaml"
    config.write_text(config_text, encoding="utf-8")

    declaration_payload = json.loads(
        Path("configs/runtime_timing_acceptance.template.json").read_text(encoding="utf-8")
    )
    declaration_payload["workcell_id"] = "cell-1"
    declaration_payload["operator_id"] = "operator-1"
    declaration_payload["accepted_at_utc"] = "2026-08-29T00:00:00+00:00"
    declaration_payload["checklist"] = {name: True for name in declaration_payload["checklist"]}
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps(declaration_payload), encoding="utf-8")

    monkeypatch.setattr(
        timing_acceptance_module,
        "science_runtime_contract_for_settings",
        lambda _settings: "a" * 64,
    )
    monkeypatch.setattr(
        timing_acceptance_module,
        "motion_control_contract_for_settings",
        lambda _settings: "b" * 64,
    )
    timing_settings = load_settings(config)
    timing_contract = timing_acceptance_module.runtime_timing_contract_for_settings(
        timing_settings
    )
    measurement_session_payload: dict[str, object] = {
        "schema": "biblade_fusion.runtime_timing_measurement_session.v1",
        "host_run_id": "host-run-1",
        "workcell_id": "cell-1",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "runtime_contract_sha256": timing_contract,
        "measurement_contract_sha256": (
            timing_acceptance_module._measurement_contract_sha256()
        ),
        "boot_id_sha256": "c" * 64,
        "motion_authorized": False,
    }
    measurement_session_payload["measurement_session_id"] = (
        timing_acceptance_module._sha256_bytes(
            timing_acceptance_module._canonical_json(measurement_session_payload)
        )
    )

    traces: list[Path] = []
    roles_and_durations = (
        ("perception_cycle_trace", 2.0),
        ("operator_reposition_trace", 20.0),
        ("segment_execution_trace", 8.0),
        ("schema5_handoff_trace", 15.0),
    )
    for trial_index, mode in enumerate(("cold", "warm", "warm")):
        for role_index, (role, duration) in enumerate(roles_and_durations):
            trace = tmp_path / f"trace-{trial_index}-{role}.json"
            operation_evidence = {
                "artifact_kind": "biblade_fusion.test_runtime_timing_operation",
                "role": role,
                "trial_id": f"trial-{trial_index}",
            }
            operation_evidence_bytes = (
                json.dumps(
                    operation_evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
            duration_ns = int(duration * 1_000_000_000)
            started_ns = 1_000_000_000_000 + (
                trial_index * 10 + role_index
            ) * 100_000_000_000
            trace.write_text(
                json.dumps(
                    {
                        "schema": "biblade_fusion.runtime_timing_trace.v2",
                        "captured_at_utc": (
                            f"2026-08-29T00:00:0{trial_index}+00:00"
                        ),
                        "duration_s": duration,
                        "host_run_id": "host-run-1",
                        "mode": mode,
                        "role": role,
                        "trial_id": f"trial-{trial_index}",
                        "measurement_method": (
                            "biblade_fusion.storage.measure_runtime_timing_trace.v2"
                        ),
                        "runtime_contract_sha256": timing_contract,
                        "measurement_session_id": measurement_session_payload[
                            "measurement_session_id"
                        ],
                        "measurement_session_payload": measurement_session_payload,
                        "boot_id_sha256": "c" * 64,
                        "operation_evidence_sha256": (
                            timing_acceptance_module._sha256_bytes(
                                operation_evidence_bytes
                            )
                        ),
                        "operation_evidence_kind": operation_evidence[
                            "artifact_kind"
                        ],
                        "operation_evidence_size_bytes": len(
                            operation_evidence_bytes
                        ),
                        "operation_evidence_payload": operation_evidence,
                        "measurement_contract_sha256": (
                            timing_acceptance_module._measurement_contract_sha256()
                        ),
                        "started_monotonic_ns": started_ns,
                        "completed_monotonic_ns": started_ns + duration_ns,
                        "duration_ns": duration_ns,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            traces.append(trace)
    trial_report = tmp_path / "trials.json"
    raw_manifest = tmp_path / "raw.json"
    timing_acceptance_module.build_runtime_timing_reports(
        traces,
        settings=timing_settings,
        trial_report=trial_report,
        raw_timing_manifest=raw_manifest,
    )
    output = tmp_path / "acceptance"
    trace_arguments = [item for trace in traces for item in ("--trace", str(trace))]
    result = runner.invoke(
        app,
        [
            "safety",
            "record-runtime-timing-acceptance",
            "--config",
            str(config),
            "--declaration",
            str(declaration),
            "--trial-report",
            str(trial_report),
            "--raw-timing-manifest",
            str(raw_manifest),
            *trace_arguments,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "metadata.json").is_file()
    assert "Motion authorized: no" in result.stdout


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

    next_plan_help = runner.invoke(app, ["coverage", "next-plan", "--help"])
    assert next_plan_help.exit_code == 0
    assert "--start-side" in next_plan_help.stdout


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

    preflight_help = runner.invoke(app, ["safety", "preflight-path", "--help"])
    assert preflight_help.exit_code == 0
    assert "--coverage-plan" in preflight_help.stdout


@pytest.mark.parametrize("supply_both", [False, True])
def test_safety_preflight_requires_exactly_one_ordering_source(
    tmp_path: Path,
    supply_both: bool,
) -> None:
    for directory in ("plan", "initialization", "occupancy", "coverage-plan"):
        (tmp_path / directory).mkdir()
    ordering_args = (
        [
            "--view-id",
            "front_r00_c00",
            "--coverage-plan",
            str(tmp_path / "coverage-plan"),
        ]
        if supply_both
        else []
    )
    args = [
        "safety",
        "preflight-path",
        "--plan",
        str(tmp_path / "plan"),
        "--initialization",
        str(tmp_path / "initialization"),
        "--occupancy",
        str(tmp_path / "occupancy"),
        "--output",
        str(tmp_path / "output"),
        *ordering_args,
    ]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "Supply exactly one ordering source" in result.stderr


def test_read_only_supervisory_replay_is_exposed() -> None:
    result = runner.invoke(app, ["supervise", "--help"])

    assert result.exit_code == 0
    assert "replay" in result.stdout
    assert "build-replay" in result.stdout

    replay_help = runner.invoke(app, ["supervise", "replay", "--help"])
    assert replay_help.exit_code == 0
    assert "--follow" in replay_help.stdout
