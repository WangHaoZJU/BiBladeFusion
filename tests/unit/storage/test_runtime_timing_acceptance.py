from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import biblade_fusion.storage as storage_api
import biblade_fusion.storage.runtime_timing_acceptance as timing_module
from biblade_fusion.core.settings import load_settings
from biblade_fusion.storage.runtime_timing_acceptance import (
    build_runtime_timing_reports,
    load_runtime_timing_acceptance_declaration,
    measure_runtime_timing_trace,
    read_runtime_timing_acceptance,
    read_runtime_timing_measurement_session,
    write_runtime_timing_acceptance,
    write_runtime_timing_measurement_session,
)

_LIMITS = {
    "maximum_perception_cycle_duration_s": 4.0,
    "maximum_operator_reposition_interval_s": 30.0,
    "maximum_segment_execution_duration_s": 12.0,
    "maximum_schema5_handoff_duration_s": 20.0,
}


def _settings():
    settings = load_settings("configs/default.yaml")
    stop = settings.stop_and_capture.model_copy(update=_LIMITS)
    return settings.model_copy(update={"stop_and_capture": stop})


def _write_json(path: Path, value: object, *, canonical: bool = False) -> Path:
    if canonical:
        text = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    else:
        text = json.dumps(value, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _trace_payload(
    *,
    trial_index: int,
    mode: str,
    role: str,
    duration_s: float,
) -> dict[str, object]:
    evidence_number = trial_index * 10 + (
        (
            "perception_cycle_trace",
            "operator_reposition_trace",
            "segment_execution_trace",
            "schema5_handoff_trace",
        ).index(role)
        + 1
    )
    evidence_payload = {
        "artifact_kind": "biblade_fusion.test_runtime_timing_operation",
        "operation_number": evidence_number,
        "role": role,
        "trial_id": f"trial-{trial_index}",
    }
    evidence_bytes = (
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    duration_ns = int(round(duration_s * 1_000_000_000.0))
    started_ns = 1_000_000_000_000 + evidence_number * 100_000_000_000
    session_payload: dict[str, object] = {
        "schema": "biblade_fusion.runtime_timing_measurement_session.v1",
        "host_run_id": "timing-run-001",
        "workcell_id": "cell-1",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "runtime_contract_sha256": (
            timing_module.runtime_timing_contract_for_settings(_settings())
        ),
        "measurement_contract_sha256": timing_module._measurement_contract_sha256(),
        "boot_id_sha256": "c" * 64,
        "motion_authorized": False,
    }
    session_payload["measurement_session_id"] = timing_module._sha256_bytes(
        timing_module._canonical_json(session_payload)
    )
    return {
        "schema": "biblade_fusion.runtime_timing_trace.v2",
        "host_run_id": "timing-run-001",
        "trial_id": f"trial-{trial_index}",
        "mode": mode,
        "role": role,
        "captured_at_utc": f"2026-08-29T00:00:0{trial_index}+00:00",
        "duration_s": duration_ns / 1_000_000_000.0,
        "measurement_method": (
            "biblade_fusion.storage.measure_runtime_timing_trace.v2"
        ),
        "runtime_contract_sha256": (
            timing_module.runtime_timing_contract_for_settings(_settings())
        ),
        "measurement_session_id": session_payload["measurement_session_id"],
        "measurement_session_payload": session_payload,
        "boot_id_sha256": "c" * 64,
        "operation_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "operation_evidence_kind": evidence_payload["artifact_kind"],
        "operation_evidence_size_bytes": len(evidence_bytes),
        "operation_evidence_payload": evidence_payload,
        "measurement_contract_sha256": timing_module._measurement_contract_sha256(),
        "started_monotonic_ns": started_ns,
        "completed_monotonic_ns": started_ns + duration_ns,
        "duration_ns": duration_ns,
    }


def _traces(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    roles_and_durations = (
        ("perception_cycle_trace", 2.0),
        ("operator_reposition_trace", 20.0),
        ("segment_execution_trace", 8.0),
        ("schema5_handoff_trace", 15.0),
    )
    for trial_index, mode in enumerate(("cold", "warm", "warm")):
        for role, duration in roles_and_durations:
            paths.append(
                _write_json(
                    tmp_path / f"trace-{trial_index}-{role}.json",
                    _trace_payload(
                        trial_index=trial_index,
                        mode=mode,
                        role=role,
                        duration_s=duration + trial_index * 0.1,
                    ),
                    canonical=True,
                )
            )
    return paths


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    traces = _traces(tmp_path)
    report, manifest = build_runtime_timing_reports(
        traces,
        settings=_settings(),
        trial_report=tmp_path / "trials.json",
        raw_timing_manifest=tmp_path / "manifest.json",
    )
    return report, manifest, traces


def _checklist() -> dict[str, bool]:
    return {
        "target_gpu_and_filesystem_used": True,
        "target_robot_controller_used": True,
        "all_four_intervals_measured_monotonically": True,
        "cold_and_warm_trials_included": True,
        "raw_timing_evidence_archived": True,
        "independent_result_review_completed": True,
    }


@pytest.fixture(autouse=True)
def _stable_external_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        timing_module,
        "science_runtime_contract_for_settings",
        lambda _settings: "a" * 64,
    )
    monkeypatch.setattr(
        timing_module,
        "motion_control_contract_for_settings",
        lambda _settings: "b" * 64,
    )
    monkeypatch.setattr(timing_module, "_boot_id_sha256", lambda: "c" * 64)


def _write_acceptance(tmp_path: Path):
    report, manifest, traces = _inputs(tmp_path)
    return write_runtime_timing_acceptance(
        tmp_path / "acceptance",
        settings=_settings(),
        workcell_id="cell-1",
        operator_id="operator-1",
        accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
        trial_report=report,
        raw_timing_manifest=manifest,
        raw_timing_traces=traces,
        checklist=_checklist(),
    )


def test_runtime_timing_acceptance_round_trips_and_binds_runtime(tmp_path: Path) -> None:
    stored = _write_acceptance(tmp_path)
    reread = read_runtime_timing_acceptance(stored.path)

    assert reread.acceptance_id == stored.acceptance_id
    assert reread.trial_count == 3
    assert reread.raw_evidence_count == 12
    assert (reread.path / "trial_report.json").is_file()
    assert (reread.path / "raw_timing_manifest.json").is_file()
    assert len(tuple((reread.path / "evidence").iterdir())) == 12
    reread.assert_matches(settings=_settings(), acceptance_id=reread.acceptance_id)
    changed = _settings()
    changed = changed.model_copy(
        update={
            "stop_and_capture": changed.stop_and_capture.model_copy(
                update={"maximum_perception_cycle_duration_s": 3.0}
            )
        }
    )
    with pytest.raises(ValueError, match="limits differ"):
        reread.assert_matches(settings=changed, acceptance_id=reread.acceptance_id)


def test_runtime_timing_acceptance_is_non_overwriting_and_tamper_evident(
    tmp_path: Path,
) -> None:
    stored = _write_acceptance(tmp_path)
    with pytest.raises(FileExistsError):
        write_runtime_timing_acceptance(
            stored.path,
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=tmp_path / "trials.json",
            raw_timing_manifest=tmp_path / "manifest.json",
            raw_timing_traces=(),
            checklist=_checklist(),
        )

    report = stored.path / "trial_report.json"
    report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="copy changed"):
        read_runtime_timing_acceptance(stored.path)


def test_runtime_timing_acceptance_rejects_an_existing_exclusive_claim(
    tmp_path: Path,
) -> None:
    report, manifest, traces = _inputs(tmp_path)
    destination = tmp_path / "acceptance"
    claim = tmp_path / ".acceptance.claim"
    claim.write_text("concurrent writer\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already claimed"):
        write_runtime_timing_acceptance(
            destination,
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report,
            raw_timing_manifest=manifest,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )

    assert not destination.exists()
    assert claim.read_text(encoding="utf-8") == "concurrent writer\n"


def test_runtime_timing_acceptance_does_not_replace_an_empty_destination(
    tmp_path: Path,
) -> None:
    report, manifest, traces = _inputs(tmp_path)
    destination = tmp_path / "acceptance"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        write_runtime_timing_acceptance(
            destination,
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report,
            raw_timing_manifest=manifest,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_runtime_timing_claim_cleanup_never_unlinks_another_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "acceptance"
    claim = tmp_path / ".acceptance.claim"

    def replace_claim(*_args, **_kwargs):
        claim.unlink()
        claim.write_text("replacement owner\n", encoding="utf-8")
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(
        timing_module,
        "_write_claimed_runtime_timing_acceptance",
        replace_claim,
    )

    with pytest.raises(RuntimeError, match="synthetic writer failure"):
        write_runtime_timing_acceptance(
            destination,
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=tmp_path / "unused-report",
            raw_timing_manifest=tmp_path / "unused-manifest",
            raw_timing_traces=(),
            checklist=_checklist(),
        )

    assert claim.read_text(encoding="utf-8") == "replacement owner\n"


def test_runtime_timing_acceptance_seals_and_revalidates_each_raw_trace(
    tmp_path: Path,
) -> None:
    stored = _write_acceptance(tmp_path)
    evidence = next((stored.path / "evidence").iterdir())
    evidence.write_bytes(evidence.read_bytes() + b" ")

    with pytest.raises(ValueError, match="raw evidence .* changed"):
        read_runtime_timing_acceptance(stored.path)


def test_runtime_timing_acceptance_binds_measurement_session_workcell(
    tmp_path: Path,
) -> None:
    report, manifest, traces = _inputs(tmp_path)

    with pytest.raises(ValueError, match="measurement session workcell differs"):
        write_runtime_timing_acceptance(
            tmp_path / "acceptance",
            settings=_settings(),
            workcell_id="cell-2",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report,
            raw_timing_manifest=manifest,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )


def test_runtime_timing_reader_cross_checks_measurement_session_workcell(
    tmp_path: Path,
) -> None:
    stored = _write_acceptance(tmp_path)
    metadata = stored.path / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload.pop("acceptance_id")
    payload["workcell_id"] = "cell-2"
    payload["acceptance_id"] = timing_module._sha256_bytes(
        timing_module._canonical_json(payload)
    )
    _write_json(metadata, payload, canonical=True)

    with pytest.raises(ValueError, match="measurement session workcell differs"):
        read_runtime_timing_acceptance(stored.path)


@pytest.mark.parametrize(
    "filename",
    ("metadata.json", "trial_report.json", "raw_timing_manifest.json"),
)
def test_runtime_timing_acceptance_rejects_symlinked_core_files(
    tmp_path: Path,
    filename: str,
) -> None:
    stored = _write_acceptance(tmp_path)
    source = stored.path / filename
    external = tmp_path / f"external-{filename}"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)

    with pytest.raises(ValueError, match="regular non-symlink file"):
        read_runtime_timing_acceptance(stored.path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.__setitem__("workcell_id", 7), "workcell_id"),
        (
            lambda payload: payload["timing_limits_s"].__setitem__(
                "maximum_perception_cycle_duration_s", True
            ),
            "limits must be numeric",
        ),
    ),
)
def test_runtime_timing_acceptance_rejects_metadata_type_confusion(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    stored = _write_acceptance(tmp_path)
    metadata = stored.path / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload.pop("acceptance_id")
    mutation(payload)
    payload["acceptance_id"] = timing_module._sha256_bytes(
        timing_module._canonical_json(payload)
    )
    _write_json(metadata, payload, canonical=True)

    with pytest.raises(ValueError, match=message):
        read_runtime_timing_acceptance(stored.path)


def test_runtime_timing_acceptance_requires_canonical_metadata(tmp_path: Path) -> None:
    stored = _write_acceptance(tmp_path)
    metadata = stored.path / "metadata.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    _write_json(metadata, payload, canonical=False)

    with pytest.raises(ValueError, match="canonical JSON"):
        read_runtime_timing_acceptance(stored.path)


def test_runtime_timing_acceptance_refuses_manifest_without_exact_raw_traces(
    tmp_path: Path,
) -> None:
    report, manifest, traces = _inputs(tmp_path)
    traces[0].write_bytes(traces[0].read_bytes() + b" ")

    with pytest.raises(ValueError):
        write_runtime_timing_acceptance(
            tmp_path / "invalid",
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report,
            raw_timing_manifest=manifest,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )


@pytest.mark.parametrize(
    "failure",
    ["zero", "duplicate", "single_mode", "over_limit", "noncanonical"],
)
def test_runtime_timing_acceptance_rejects_invalid_trials(
    tmp_path: Path,
    failure: str,
) -> None:
    report_path, manifest_path, traces = _inputs(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if failure == "zero":
        report["trials"][0]["maximum_perception_cycle_duration_s"] = 0.0
    elif failure == "duplicate":
        report["trials"][1]["trial_id"] = report["trials"][0]["trial_id"]
    elif failure == "single_mode":
        for trial in report["trials"]:
            trial["mode"] = "warm"
    elif failure == "over_limit":
        report["trials"][0]["maximum_perception_cycle_duration_s"] = 4.1
    _write_json(report_path, report, canonical=failure != "noncanonical")

    with pytest.raises(ValueError):
        write_runtime_timing_acceptance(
            tmp_path / "invalid",
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report_path,
            raw_timing_manifest=manifest_path,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )


@pytest.mark.parametrize("failure", ["duplicate_key", "duplicate_identity", "noncanonical"])
def test_runtime_timing_acceptance_rejects_ambiguous_raw_manifest(
    tmp_path: Path,
    failure: str,
) -> None:
    report_path, manifest_path, traces = _inputs(tmp_path)
    if failure == "duplicate_key":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence"][1]["role"] = manifest["evidence"][0]["role"]
        manifest["evidence"][1]["name"] = manifest["evidence"][0]["name"]
        _write_json(manifest_path, manifest, canonical=True)
    elif failure == "duplicate_identity":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evidence"][1]["sha256"] = manifest["evidence"][0]["sha256"]
        manifest["evidence"][1]["size_bytes"] = manifest["evidence"][0]["size_bytes"]
        _write_json(manifest_path, manifest, canonical=True)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _write_json(manifest_path, manifest, canonical=False)

    with pytest.raises(ValueError):
        write_runtime_timing_acceptance(
            tmp_path / "invalid",
            settings=_settings(),
            workcell_id="cell-1",
            operator_id="operator-1",
            accepted_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
            trial_report=report_path,
            raw_timing_manifest=manifest_path,
            raw_timing_traces=traces,
            checklist=_checklist(),
        )


def test_runtime_timing_storage_api_exports_authority() -> None:
    assert storage_api.build_runtime_timing_reports is build_runtime_timing_reports
    assert storage_api.measure_runtime_timing_trace is measure_runtime_timing_trace
    assert (
        storage_api.load_runtime_timing_acceptance_declaration
        is load_runtime_timing_acceptance_declaration
    )
    assert storage_api.read_runtime_timing_acceptance is read_runtime_timing_acceptance
    assert (
        storage_api.read_runtime_timing_measurement_session
        is read_runtime_timing_measurement_session
    )
    assert storage_api.write_runtime_timing_acceptance is write_runtime_timing_acceptance
    assert (
        storage_api.write_runtime_timing_measurement_session
        is write_runtime_timing_measurement_session
    )


def test_runtime_timing_measurement_session_is_sealed_and_boot_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = write_runtime_timing_measurement_session(
        tmp_path / "measurement-session.json",
        settings=_settings(),
        host_run_id="host-run-1",
        workcell_id="cell-1",
        created_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
    )
    reread = read_runtime_timing_measurement_session(session.path)

    assert reread == session
    reread.assert_current(_settings())
    with pytest.raises(FileExistsError):
        write_runtime_timing_measurement_session(
            session.path,
            settings=_settings(),
            host_run_id="host-run-1",
            workcell_id="cell-1",
        )

    monkeypatch.setattr(timing_module, "_boot_id_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="another boot"):
        reread.assert_current(_settings())


@pytest.mark.parametrize(
    "invalid_json",
    (
        '{"workcell_id":"a","workcell_id":"b","operator_id":"o",'
        '"accepted_at_utc":"2026-08-29T00:00:00+00:00","checklist":{}}',
        '{"workcell_id":"a","operator_id":"o","accepted_at_utc":NaN,'
        '"checklist":{}}',
    ),
)
def test_runtime_timing_declaration_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    invalid_json: str,
) -> None:
    declaration = tmp_path / "declaration.json"
    declaration.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ValueError):
        load_runtime_timing_acceptance_declaration(declaration)


def test_narrow_trial_operation_produces_exclusive_canonical_trace(
    tmp_path: Path,
) -> None:
    clock = iter((10_000_000_000, 11_250_000_000))
    destination = tmp_path / "trace.json"
    session = write_runtime_timing_measurement_session(
        tmp_path / "measurement-session.json",
        settings=_settings(),
        host_run_id="host-run-1",
        workcell_id="cell-1",
        created_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
    )
    evidence = _write_json(
        tmp_path / "operation.json",
        {
            "artifact_kind": "biblade_fusion.test_runtime_timing_operation",
            "result": "measured-result",
        },
        canonical=True,
    )

    result, path = measure_runtime_timing_trace(
        destination,
        trial_id="cold-001",
        mode="cold",
        role="perception_cycle_trace",
        settings=_settings(),
        measurement_session=session.path,
        operation_evidence_path=lambda _result: evidence,
        operation=lambda: "measured-result",
        monotonic_ns_clock=lambda: next(clock),
        utc_clock=lambda: datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert result == "measured-result"
    assert path == destination.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["duration_s"] == pytest.approx(1.25)
    assert payload["runtime_contract_sha256"] == (
        timing_module.runtime_timing_contract_for_settings(_settings())
    )
    assert payload["boot_id_sha256"] == "c" * 64
    assert payload["operation_evidence_sha256"] == hashlib.sha256(
        evidence.read_bytes()
    ).hexdigest()
    assert payload["duration_ns"] == 1_250_000_000
    assert path.read_bytes() == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        measure_runtime_timing_trace(
            destination,
            trial_id="cold-001",
            mode="cold",
            role="perception_cycle_trace",
            settings=_settings(),
            measurement_session=session.path,
            operation_evidence_path=lambda _result: evidence,
            operation=lambda: None,
            monotonic_ns_clock=lambda: 20_000_000_000,
        )


def test_narrow_trial_claims_output_before_physical_operation(tmp_path: Path) -> None:
    destination = tmp_path / "trace.json"
    session = write_runtime_timing_measurement_session(
        tmp_path / "measurement-session.json",
        settings=_settings(),
        host_run_id="host-run-1",
        workcell_id="cell-1",
        created_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
    )
    claim = tmp_path / ".trace.json.claim"
    claim.write_text("concurrent measurement\n", encoding="utf-8")
    operation_called = False

    def operation():
        nonlocal operation_called
        operation_called = True

    with pytest.raises(FileExistsError, match="already claimed"):
        measure_runtime_timing_trace(
            destination,
            trial_id="cold-001",
            mode="cold",
            role="perception_cycle_trace",
            settings=_settings(),
            measurement_session=session.path,
            operation_evidence_path=lambda _result: tmp_path / "never-used.json",
            operation=operation,
            monotonic_ns_clock=lambda: 20_000_000_000,
        )

    assert operation_called is False
    assert not destination.exists()
    assert claim.read_text(encoding="utf-8") == "concurrent measurement\n"


def test_narrow_trial_rejects_runtime_contract_change_during_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = write_runtime_timing_measurement_session(
        tmp_path / "measurement-session.json",
        settings=_settings(),
        host_run_id="host-run-1",
        workcell_id="cell-1",
        created_at_utc=datetime(2026, 8, 29, tzinfo=UTC),
    )
    contracts = iter(
        (
            session.runtime_contract_sha256,
            session.runtime_contract_sha256,
            "f" * 64,
        )
    )
    monkeypatch.setattr(
        timing_module,
        "runtime_timing_contract_for_settings",
        lambda _settings: next(contracts),
    )
    clock = iter((10_000_000_000, 11_000_000_000))
    evidence = _write_json(
        tmp_path / "operation.json",
        {"artifact_kind": "biblade_fusion.test_runtime_timing_operation"},
        canonical=True,
    )

    with pytest.raises(ValueError, match="contract changed"):
        measure_runtime_timing_trace(
            tmp_path / "trace.json",
            trial_id="cold-001",
            mode="cold",
            role="perception_cycle_trace",
            settings=_settings(),
            measurement_session=session.path,
            operation_evidence_path=lambda _result: evidence,
            operation=lambda: None,
            monotonic_ns_clock=lambda: next(clock),
            utc_clock=lambda: datetime(2026, 8, 29, tzinfo=UTC),
        )

    assert not (tmp_path / "trace.json").exists()
    assert not (tmp_path / ".trace.json.claim").exists()


def test_machine_role_traces_build_canonical_report_and_manifest(tmp_path: Path) -> None:
    traces: list[Path] = []
    roles = (
        "perception_cycle_trace",
        "operator_reposition_trace",
        "segment_execution_trace",
        "schema5_handoff_trace",
    )
    durations = (2.0, 20.0, 8.0, 15.0)
    for trial_index, mode in enumerate(("cold", "warm", "warm")):
        for role, duration in zip(roles, durations, strict=True):
            traces.append(
                _write_json(
                    tmp_path / f"trace-{trial_index}-{role}.json",
                    _trace_payload(
                        trial_index=trial_index,
                        mode=mode,
                        role=role,
                        duration_s=duration + trial_index * 0.1,
                    ),
                    canonical=True,
                )
            )

    report, manifest = build_runtime_timing_reports(
        traces,
        settings=_settings(),
        trial_report=tmp_path / "built-trials.json",
        raw_timing_manifest=tmp_path / "built-manifest.json",
    )

    assert report.read_bytes() == (
        json.dumps(
            json.loads(report.read_text(encoding="utf-8")),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(manifest_payload["evidence"]) == 12
    assert {item["role"] for item in manifest_payload["evidence"]} == set(roles)


def test_machine_trace_aggregation_rejects_missing_role(tmp_path: Path) -> None:
    traces: list[Path] = []
    for trial_index, mode in enumerate(("cold", "warm", "warm")):
        for role in (
            "perception_cycle_trace",
            "operator_reposition_trace",
            "segment_execution_trace",
        ):
            traces.append(
                _write_json(
                    tmp_path / f"trace-{trial_index}-{role}.json",
                    _trace_payload(
                        trial_index=trial_index,
                        mode=mode,
                        role=role,
                        duration_s=1.0,
                    ),
                    canonical=True,
                )
            )

    with pytest.raises(ValueError, match="four traces"):
        build_runtime_timing_reports(
            traces,
            settings=_settings(),
            trial_report=tmp_path / "built-trials.json",
            raw_timing_manifest=tmp_path / "built-manifest.json",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("runtime_contract_sha256", "f" * 64, "runtime contract differs"),
        ("measurement_session_id", "f" * 64, "measurement session binding"),
        ("boot_id_sha256", "f" * 64, "boot identities"),
        ("measurement_method", "manual", "measurement method differs"),
    ),
)
def test_machine_trace_aggregation_rejects_mixed_measurement_authority(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    traces = _traces(tmp_path)
    payload = json.loads(traces[-1].read_text(encoding="utf-8"))
    payload[field] = replacement
    _write_json(traces[-1], payload, canonical=True)

    with pytest.raises(ValueError, match=message):
        build_runtime_timing_reports(
            traces,
            settings=_settings(),
            trial_report=tmp_path / "built-trials.json",
            raw_timing_manifest=tmp_path / "built-manifest.json",
        )


def test_machine_trace_aggregation_rejects_reused_operation_evidence(
    tmp_path: Path,
) -> None:
    traces = _traces(tmp_path)
    first = json.loads(traces[0].read_text(encoding="utf-8"))
    second = json.loads(traces[1].read_text(encoding="utf-8"))
    second["operation_evidence_sha256"] = first["operation_evidence_sha256"]
    _write_json(traces[1], second, canonical=True)

    with pytest.raises(ValueError, match="repeat operation evidence"):
        build_runtime_timing_reports(
            traces,
            settings=_settings(),
            trial_report=tmp_path / "built-trials.json",
            raw_timing_manifest=tmp_path / "built-manifest.json",
        )


def test_machine_trace_aggregation_exclusively_publishes_both_outputs(
    tmp_path: Path,
) -> None:
    traces = _traces(tmp_path)
    report = tmp_path / "built-trials.json"
    manifest = tmp_path / "built-manifest.json"
    manifest.write_text("pre-existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_runtime_timing_reports(
            traces,
            settings=_settings(),
            trial_report=report,
            raw_timing_manifest=manifest,
        )

    assert not report.exists()
    assert manifest.read_text(encoding="utf-8") == "pre-existing\n"
