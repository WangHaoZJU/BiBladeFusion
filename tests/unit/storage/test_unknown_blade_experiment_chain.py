"""Append-only unknown-blade experiment handoff chain tests."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import biblade_fusion.storage.unknown_blade_experiment as experiment_module
from biblade_fusion.storage.runtime_timing_acceptance import (
    RuntimeTimingAcceptanceAuthority,
)
from biblade_fusion.storage.science_acceptance import ScienceTestEnvelope
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.stop_scan_run import StopScanRunWriter
from biblade_fusion.storage.unknown_blade_experiment import (
    UnknownBladeExperimentFormatError,
    UnknownBladeExperimentWriter,
    read_unknown_blade_experiment,
)
from biblade_fusion.workflows.unknown_blade_runtime import (
    UnknownBladeResumePhase,
    UnknownBladeRuntimeError,
    load_unknown_blade_resume_plan,
)


def _run(root: Path, run_id: str, *, event_type: str) -> StopScanRunWriter:
    writer = StopScanRunWriter.create(root, run_id=run_id)
    writer.append_event(
        phase="motion_blocked",
        cycle_index=0,
        event_type=event_type,
        payload={"motion_authorized": False},
    )
    return writer


def _fine_run(root: Path, run_id: str) -> StopScanRunWriter:
    """Create the exact initial event emitted by StopScanCoordinator.start()."""

    writer = StopScanRunWriter.create(root, run_id=run_id)
    writer.append_event(
        phase="bootstrap_map_required",
        cycle_index=0,
        event_type="run_started",
        payload={
            "depth_backend": "foundation_stereo",
            "bootstrap_mode": "operator_guided",
            "minimum_source_views": 3,
        },
    )
    return writer


def _production_authorities(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ScienceAcceptanceAuthority, RuntimeTimingAcceptanceAuthority]:
    authority_root = root.resolve()
    science_path = authority_root / "science"
    timing_path = authority_root / "timing"
    science_path.mkdir(parents=True)
    timing_path.mkdir(parents=True)
    monkeypatch.setattr(
        ScienceAcceptanceAuthority,
        "assert_acceptance_asset_current",
        lambda _self: None,
    )
    monkeypatch.setattr(
        RuntimeTimingAcceptanceAuthority,
        "assert_acceptance_asset_current",
        lambda _self: None,
    )
    science = ScienceAcceptanceAuthority(
        science_path,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        ScienceTestEnvelope(0.15, 1.5, 0.0, 85.0),
        {
            "foundation_stereo_source_sha256": "4" * 64,
            "foundation_stereo_checkpoint_sha256": "5" * 64,
            "foundation_stereo_model_config_sha256": "6" * 64,
            "stereo_calibration_sha256": "7" * 64,
            "flange_primary_hand_eye_sha256": "8" * 64,
        },
    )
    timing = RuntimeTimingAcceptanceAuthority(
        timing_path,
        "9" * 64,
        "a" * 64,
        "b" * 64,
        {
            "maximum_perception_cycle_duration_s": 4.0,
            "maximum_operator_reposition_interval_s": 30.0,
            "maximum_segment_execution_duration_s": 12.0,
            "maximum_schema5_handoff_duration_s": 20.0,
        },
    )
    return science, timing


def test_init_chain_binds_science_authority_and_rejects_event_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse = _run(tmp_path / "coarse", "experiment-001", event_type="coarse_stopped")
    authority, timing = _production_authorities(
        tmp_path / "authorities",
        monkeypatch,
    )
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id="experiment-001",
        coarse_run_root=coarse.root,
        science_authority=authority,
        runtime_timing_authority=timing,
    )
    assert read_unknown_blade_experiment(writer.root).science_authority == authority

    event = writer.root / "events" / "00000000.json"
    payload = json.loads(event.read_text(encoding="utf-8"))
    payload["payload"]["science_acceptance_authority"][
        "runtime_contract_sha256"
    ] = "9" * 64
    event.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnknownBladeExperimentFormatError, match="event SHA-256"):
        read_unknown_blade_experiment(writer.root)


def test_reader_rejects_duplicate_json_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse = _run(tmp_path / "coarse", "strict-json-001", event_type="coarse_stopped")
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id="strict-json-001",
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    event = writer.root / "events" / "00000000.json"
    text = event.read_text(encoding="utf-8")
    event.write_text(
        text.replace(
            "{\n",
            '{\n  "schema_version": 1,\n',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnknownBladeExperimentFormatError, match="duplicate JSON object key"):
        read_unknown_blade_experiment(writer.root)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sequence", "0", "sequence must be an integer"),
        ("schema_version", True, "schema_version must be an integer"),
    ],
)
def test_reader_rejects_implicit_top_level_type_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    coarse = _run(tmp_path / "coarse", "strict-types-001", event_type="coarse_stopped")
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id="strict-types-001",
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    event = writer.root / "events" / "00000000.json"
    raw = json.loads(event.read_text(encoding="utf-8"))
    raw[field] = replacement
    event.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(UnknownBladeExperimentFormatError, match=message):
        read_unknown_blade_experiment(writer.root)


def test_production_writer_rejects_authorityless_and_science_only_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse = _run(tmp_path / "coarse", "writer-gate-001", event_type="coarse_stopped")
    science, _timing = _production_authorities(tmp_path / "authorities", monkeypatch)

    with pytest.raises(ValueError, match="require science and runtime timing"):
        UnknownBladeExperimentWriter.create(
            tmp_path / "authorityless",
            experiment_id="writer-gate-001",
            coarse_run_root=coarse.root,
        )
    with pytest.raises(ValueError, match="require science and runtime timing"):
        UnknownBladeExperimentWriter.create(
            tmp_path / "science-only",
            experiment_id="writer-gate-001",
            coarse_run_root=coarse.root,
            science_authority=science,
        )

    assert not (tmp_path / "authorityless").exists()
    assert not (tmp_path / "science-only").exists()


def test_experimental_coarse_writer_is_authorityless_and_audit_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse = _run(
        tmp_path / "coarse",
        "experimental-coarse-001",
        event_type="coarse_stopped",
    )

    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "experimental-chain",
        experiment_id="experimental-coarse-001",
        coarse_run_root=coarse.root,
        production=False,
    )
    generation, reference = _sources(tmp_path / "assets", monkeypatch)
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
    )
    fine = _fine_run(tmp_path / "fine", "experimental-coarse-001")
    writer.append_unaccepted_fine_started(fine_run_root=fine.root)
    stored = read_unknown_blade_experiment(writer.root)

    assert stored.science_authority is None
    assert stored.runtime_timing_authority is None
    assert stored.fine_start_protocol is None
    assert stored.latest_event.event_type == "fine_started"


def test_experiment_writer_binds_new_eventless_coarse_run_reservation(
    tmp_path: Path,
) -> None:
    coarse = StopScanRunWriter.create(
        tmp_path / "coarse",
        run_id="eventless-coarse-001",
    )

    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "experimental-chain",
        experiment_id="eventless-coarse-001",
        coarse_run_root=coarse.root,
        coarse_run_id=coarse.run_id,
        production=False,
    )

    with pytest.raises(UnknownBladeExperimentFormatError, match="contains no events"):
        read_unknown_blade_experiment(writer.root)

    coarse.append_event(
        phase="bootstrap_map_required",
        cycle_index=0,
        event_type="run_started",
        payload={"motion_authorized": False},
    )
    stored = read_unknown_blade_experiment(writer.root)

    assert stored.experiment_id == "eventless-coarse-001"
    assert stored.events[0].payload["coarse_run_id"] == coarse.run_id
    assert stored.events[0].payload["coarse_run_root"] == str(coarse.root)


def test_eventless_coarse_run_reservation_requires_matching_explicit_identity(
    tmp_path: Path,
) -> None:
    coarse = StopScanRunWriter.create(
        tmp_path / "coarse",
        run_id="eventless-coarse-002",
    )

    with pytest.raises(ValueError, match="coarse run ID differs"):
        UnknownBladeExperimentWriter.create(
            tmp_path / "experimental-chain",
            experiment_id="different-experiment",
            coarse_run_root=coarse.root,
            coarse_run_id=coarse.run_id,
            production=False,
        )


def test_legacy_chain_is_audit_readable_but_writer_cannot_resume_it(
    tmp_path: Path,
) -> None:
    coarse = _run(tmp_path / "coarse", "legacy-audit-001", event_type="coarse_stopped")
    root = (tmp_path / "legacy-chain").resolve()
    (root / "events").mkdir(parents=True)
    event = experiment_module.UnknownBladeExperimentEvent.build(
        experiment_id="legacy-audit-001",
        sequence=0,
        event_type="experiment_initialized",
        payload={
            "experiment_id": "legacy-audit-001",
            "coarse_run_id": coarse.run_id,
            "coarse_run_root": str(coarse.root),
        },
        previous_event_sha256=None,
    )
    experiment_module._write_new_json(
        root / "events" / "00000000.json",
        event.to_payload(),
    )

    stored = read_unknown_blade_experiment(root)
    assert stored.science_authority is None
    assert stored.runtime_timing_authority is None
    with pytest.raises(ValueError, match="audit-readable only"):
        UnknownBladeExperimentWriter.resume(root)


def test_production_chain_binds_timing_authority_and_handoff_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse = _run(tmp_path / "coarse", "timing-chain-001", event_type="coarse_stopped")
    science_path = (tmp_path / "science-acceptance").resolve()
    timing_path = (tmp_path / "timing-acceptance").resolve()
    science_path.mkdir()
    timing_path.mkdir()
    monkeypatch.setattr(
        ScienceAcceptanceAuthority,
        "assert_acceptance_asset_current",
        lambda _self: None,
    )
    monkeypatch.setattr(
        RuntimeTimingAcceptanceAuthority,
        "assert_acceptance_asset_current",
        lambda _self: None,
    )
    science = ScienceAcceptanceAuthority(
        science_path,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        ScienceTestEnvelope(0.15, 1.5, 0.0, 85.0),
        {
            "foundation_stereo_source_sha256": "4" * 64,
            "foundation_stereo_checkpoint_sha256": "5" * 64,
            "foundation_stereo_model_config_sha256": "6" * 64,
            "stereo_calibration_sha256": "7" * 64,
            "flange_primary_hand_eye_sha256": "8" * 64,
        },
    )
    timing = RuntimeTimingAcceptanceAuthority(
        timing_path,
        "9" * 64,
        "a" * 64,
        "b" * 64,
        {
            "maximum_perception_cycle_duration_s": 4.0,
            "maximum_operator_reposition_interval_s": 30.0,
            "maximum_segment_execution_duration_s": 12.0,
            "maximum_schema5_handoff_duration_s": 20.0,
        },
    )
    generation, reference = _sources(tmp_path / "assets", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id="timing-chain-001",
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    with pytest.raises(ValueError, match="exceeds accepted limit"):
        writer.prepare_handoff(
            schema5_generation=generation,
            reference_coarse_model=reference,
            schema5_prepare_duration_s=20.1,
        )
    writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=10.0,
    )
    fine = _fine_run(tmp_path / "fine", "timing-chain-001")
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    with pytest.raises(ValueError, match="exceeds accepted limit"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=lambda: 20.1,
        )
    publication_samples = [19.0, 19.5]
    started = writer.append_fine_started(
        timing_scope="uninterrupted_total",
        budget_check=lambda: publication_samples.pop(0),
    )

    stored = read_unknown_blade_experiment(writer.root)

    assert stored.runtime_timing_authority == timing
    assert writer.events[2].payload["schema5_prepare_timing"]["actual_duration_s"] == 10.0
    assert started.payload["schema5_handoff_timing"] == {
        "prepublication_check_duration_s": 19.0,
        "prepublication_duration_semantics": (
            "lower_bound_sample_before_final_event_serialization"
        ),
        "accepted_limit_s": 20.0,
        "runtime_timing_acceptance_id": "9" * 64,
        "runtime_timing_metadata_sha256": "a" * 64,
        "measurement_scope": "uninterrupted_total",
        "publication_deadline_contract": (
            "final_budget_check_after_event_fsync_before_atomic_publish"
        ),
    }
    assert publication_samples == []

    crash_writer = UnknownBladeExperimentWriter.create(
        tmp_path / "crash-chain",
        experiment_id="timing-chain-001",
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    crash_writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    crash_writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=11.0,
    )
    resumed = UnknownBladeExperimentWriter.resume(crash_writer.root)
    resumed.append_fine_start_candidate(fine_run_root=fine.root)
    resumed_started = resumed.append_fine_started(
        timing_scope="resume_fine_start",
        budget_check=lambda: 3.0,
    )
    assert resumed_started.payload["schema5_handoff_timing"]["measurement_scope"] == (
        "resume_fine_start"
    )
    assert read_unknown_blade_experiment(crash_writer.root).runtime_timing_authority == timing

    event_path = writer.root / "events" / "00000004.json"
    tampered = json.loads(event_path.read_text(encoding="utf-8"))
    tampered["payload"]["schema5_handoff_timing"][
        "prepublication_check_duration_s"
    ] = 1.0
    event_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(UnknownBladeExperimentFormatError, match="event SHA-256"):
        read_unknown_blade_experiment(writer.root)


def _sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    generation = (tmp_path / "schema5-generation").resolve()
    source_generation = _source_for(generation)
    reference = (tmp_path / "schema5-reference").resolve()
    generation.mkdir(parents=True)
    source_generation.mkdir(parents=True)
    reference.mkdir(parents=True)
    (generation / "generation.json").write_text('{"schema": 5}\n', encoding="utf-8")
    (source_generation / "generation.json").write_text(
        '{"schema": "source"}\n',
        encoding="utf-8",
    )
    (reference / "metadata.json").write_text('{"model": "schema5"}\n', encoding="utf-8")
    monkeypatch.setattr(
        experiment_module,
        "read_coarse_scan_generation",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            coarse_model_path=reference,
            previous_generation_path=_source_for(Path(path).resolve()),
        ),
    )
    return generation, reference


def _source_for(generation: Path) -> Path:
    return generation.with_name(f"{generation.name}-source")


def _prepared_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_id: str,
) -> tuple[Path, UnknownBladeExperimentWriter]:
    experiment_root = (tmp_path / "experiment").resolve()
    coarse = _run(tmp_path / "coarse", experiment_id, event_type="coarse_stopped")
    generation, reference = _sources(tmp_path / "assets", monkeypatch)
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        experiment_root / "experiment_handoff",
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=0.0,
    )
    return experiment_root, writer


def test_candidate_crash_recovers_as_prepared_and_commits_only_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="candidate-recovery-001",
    )
    orphan = _fine_run(tmp_path / "fine-orphan", "candidate-recovery-001")
    orphan_candidate = writer.append_fine_start_candidate(fine_run_root=orphan.root)

    plan = load_unknown_blade_resume_plan(experiment_root)

    assert plan.phase is UnknownBladeResumePhase.PREPARED
    assert plan.fine_run_root is None
    assert read_unknown_blade_experiment(writer.root).latest_event == orphan_candidate

    replacement = _fine_run(tmp_path / "fine-recovery", "candidate-recovery-001")
    resumed = UnknownBladeExperimentWriter.resume(writer.root)
    replacement_candidate = resumed.append_fine_start_candidate(
        fine_run_root=replacement.root
    )
    committed = resumed.append_fine_started(
        timing_scope="resume_fine_start",
        budget_check=lambda: 1.0,
    )

    recovered = load_unknown_blade_resume_plan(experiment_root)
    assert recovered.phase is UnknownBladeResumePhase.FINE
    assert recovered.fine_run_root == replacement.root
    assert committed.payload["fine_start_candidate_event_sha256"] == (
        replacement_candidate.event_sha256
    )
    assert committed.payload["fine_run_root"] != orphan_candidate.payload["fine_run_root"]


def test_fine_start_candidate_rejects_an_already_used_stop_scan_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="used-fine-run-001",
    )
    fine = _fine_run(tmp_path / "fine", "used-fine-run-001")
    fine.append_event(
        phase="awaiting_capture",
        cycle_index=0,
        event_type="capture_requested",
        payload={"motion_authorized": False},
    )

    with pytest.raises(ValueError, match="exactly one fine-run event"):
        writer.append_fine_start_candidate(fine_run_root=fine.root)

    assert writer.events[-1].event_type == "handoff_prepared"


def test_fine_start_candidate_requires_real_stop_scan_bootstrap_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="fine-bootstrap-001",
    )
    fine = _run(
        tmp_path / "fine",
        "fine-bootstrap-001",
        event_type="run_started",
    )

    with pytest.raises(
        ValueError,
        match="cycle-0 bootstrap_map_required/run_started",
    ):
        writer.append_fine_start_candidate(fine_run_root=fine.root)

    assert writer.events[-1].event_type == "handoff_prepared"


def test_candidate_replay_rejects_capture_before_fine_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="candidate-replay-001",
    )
    fine = _fine_run(tmp_path / "fine", "candidate-replay-001")
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    fine.append_event(
        phase="awaiting_capture",
        cycle_index=0,
        event_type="capture_requested",
        payload={"motion_authorized": False},
    )

    with pytest.raises(
        UnknownBladeExperimentFormatError,
        match="exactly one fine-run event",
    ):
        read_unknown_blade_experiment(writer.root)


def test_fine_started_rechecks_single_event_candidate_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="candidate-race-001",
    )
    fine = _fine_run(tmp_path / "fine", "candidate-race-001")
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    fine.append_event(
        phase="awaiting_capture",
        cycle_index=0,
        event_type="capture_requested",
        payload={"motion_authorized": False},
    )
    # Bypass the ordinary full-chain replay gate so this test independently
    # exercises append_fine_started's own readback immediately before commit.
    monkeypatch.setattr(
        UnknownBladeExperimentWriter,
        "_require_current_chain",
        lambda _self: None,
    )

    with pytest.raises(ValueError, match="exactly one fine-run event"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=lambda: 1.0,
        )

    assert writer.events[-1].event_type == "fine_start_candidate"


def test_slow_candidate_persistence_cannot_create_committed_fine_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="candidate-timeout-001",
    )
    fine = _fine_run(tmp_path / "fine", "candidate-timeout-001")
    candidate = writer.append_fine_start_candidate(fine_run_root=fine.root)
    elapsed_after_candidate_persistence = 20.000001

    with pytest.raises(ValueError, match="exceeds accepted limit"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=lambda: elapsed_after_candidate_persistence,
        )

    stored = read_unknown_blade_experiment(writer.root)
    assert stored.latest_event == candidate
    assert all(event.event_type != "fine_started" for event in stored.events)
    assert load_unknown_blade_resume_plan(experiment_root).phase is (
        UnknownBladeResumePhase.PREPARED
    )


def test_final_deadline_check_runs_after_fsync_before_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="commit-deadline-001",
    )
    fine = _fine_run(tmp_path / "fine", "commit-deadline-001")
    candidate = writer.append_fine_start_candidate(fine_run_root=fine.root)
    final_path = writer.root / "events" / "00000004.json"
    checks = 0

    def budget_check() -> float:
        nonlocal checks
        checks += 1
        if checks == 1:
            return 1.0
        assert not final_path.exists()
        assert tuple(final_path.parent.glob(f".{final_path.name}.*.partial"))
        raise RuntimeError("synthetic deadline before atomic publication")

    with pytest.raises(RuntimeError, match="deadline before atomic publication"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=budget_check,
        )

    assert checks == 2
    assert not final_path.exists()
    assert not tuple(final_path.parent.glob(f".{final_path.name}.*.partial"))
    assert writer.events[-1] == candidate
    assert read_unknown_blade_experiment(writer.root).latest_event == candidate
    assert load_unknown_blade_resume_plan(experiment_root).phase is (
        UnknownBladeResumePhase.PREPARED
    )


def test_before_publish_concurrent_append_waits_for_fine_started_linearization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="commit-lock-001",
    )
    fine = _fine_run(tmp_path / "fine", "commit-lock-001")
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    competing_writer = StopScanRunWriter.resume(fine.root)
    final_path = writer.root / "events" / "00000004.json"
    attempted = threading.Event()
    completed = threading.Event()
    outer_visible_after_append: list[bool] = []
    worker_failures: list[BaseException] = []
    worker: threading.Thread | None = None
    checks = 0

    def competing_append() -> None:
        attempted.set()
        try:
            competing_writer.append_event(
                phase="awaiting_capture",
                cycle_index=0,
                event_type="capture_requested",
                payload={"motion_authorized": False},
            )
            outer_visible_after_append.append(final_path.is_file())
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_failures.append(exc)
        finally:
            completed.set()

    def budget_check() -> float:
        nonlocal checks, worker
        checks += 1
        if checks == 2:
            worker = threading.Thread(target=competing_append, daemon=True)
            worker.start()
            assert attempted.wait(timeout=1.0)
            assert not completed.wait(timeout=0.1)
            assert not final_path.exists()
        return 1.0

    started = writer.append_fine_started(
        timing_scope="uninterrupted_total",
        budget_check=budget_check,
    )
    assert worker is not None
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert worker_failures == []
    assert completed.is_set()
    assert outer_visible_after_append == [True]
    assert final_path.is_file()
    assert started.event_type == "fine_started"
    assert len(experiment_module.read_stop_scan_run(fine.root).events) == 2
    assert read_unknown_blade_experiment(writer.root).latest_event == started


def test_before_publish_reentrant_append_invalidates_fine_started_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, writer = _prepared_experiment(
        tmp_path,
        monkeypatch,
        experiment_id="commit-reentrant-001",
    )
    fine = _fine_run(tmp_path / "fine", "commit-reentrant-001")
    candidate = writer.append_fine_start_candidate(fine_run_root=fine.root)
    final_path = writer.root / "events" / "00000004.json"
    checks = 0

    def malicious_budget_check() -> float:
        nonlocal checks
        checks += 1
        if checks == 2:
            fine.append_event(
                phase="awaiting_capture",
                cycle_index=0,
                event_type="capture_requested",
                payload={"motion_authorized": False},
            )
        return 1.0

    with pytest.raises(ValueError, match="exactly one fine-run event"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=malicious_budget_check,
        )

    assert checks == 2
    assert not final_path.exists()
    assert writer.events[-1] == candidate
    assert len(experiment_module.read_stop_scan_run(fine.root).events) == 2


def test_legacy_single_event_fine_started_is_audit_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _experiment_root, current = _prepared_experiment(
        tmp_path / "current",
        monkeypatch,
        experiment_id="legacy-fine-start-001",
    )
    fine = _run(
        tmp_path / "legacy-fine",
        "legacy-fine-start-001",
        event_type="run_started",
    )
    legacy_experiment_root = (tmp_path / "legacy-experiment").resolve()
    legacy_root = legacy_experiment_root / "experiment_handoff"
    (legacy_root / "events").mkdir(parents=True)

    init_payload = dict(current.events[0].payload)
    init_payload.pop("fine_start_protocol")
    legacy_init = experiment_module.UnknownBladeExperimentEvent.build(
        experiment_id=current.experiment_id,
        sequence=0,
        event_type="experiment_initialized",
        payload=init_payload,
        previous_event_sha256=None,
    )
    coarse = experiment_module.UnknownBladeExperimentEvent.build(
        experiment_id=current.experiment_id,
        sequence=1,
        event_type="coarse_checkpoint",
        payload=dict(current.events[1].payload),
        previous_event_sha256=legacy_init.event_sha256,
    )
    prepared_payload = dict(current.events[2].payload)
    prepared_payload["coarse_checkpoint_event_sha256"] = coarse.event_sha256
    prepared = experiment_module.UnknownBladeExperimentEvent.build(
        experiment_id=current.experiment_id,
        sequence=2,
        event_type="handoff_prepared",
        payload=prepared_payload,
        previous_event_sha256=coarse.event_sha256,
    )
    started = experiment_module.UnknownBladeExperimentEvent.build(
        experiment_id=current.experiment_id,
        sequence=3,
        event_type="fine_started",
        payload={
            "prepared_event_sha256": prepared.event_sha256,
            "fine_run_id": fine.run_id,
            "fine_run_root": str(fine.root),
            "fine_first_event_sha256": fine.events[0].event_sha256,
            "schema5_handoff_timing": {
                "actual_duration_s": 1.0,
                "accepted_limit_s": 20.0,
                "runtime_timing_acceptance_id": "9" * 64,
                "runtime_timing_metadata_sha256": "a" * 64,
                "measurement_scope": "uninterrupted_total",
            },
        },
        previous_event_sha256=prepared.event_sha256,
    )
    for event in (legacy_init, coarse, prepared, started):
        experiment_module._write_new_json(
            legacy_root / "events" / f"{event.sequence:08d}.json",
            event.to_payload(),
        )

    stored = read_unknown_blade_experiment(legacy_root)
    assert stored.latest_event.event_type == "fine_started"
    assert stored.fine_start_protocol is None
    with pytest.raises(ValueError, match="audit-readable only"):
        UnknownBladeExperimentWriter.resume(legacy_root)
    with pytest.raises(UnknownBladeRuntimeError, match="audit-readable only"):
        load_unknown_blade_resume_plan(legacy_experiment_root)


def _complete_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_id: str = "experiment-001",
    output_name: str = "experiment-chain",
):
    coarse = _run(tmp_path / f"{output_name}-coarse", experiment_id, event_type="coarse_stopped")
    generation, reference = _sources(tmp_path / f"{output_name}-assets", monkeypatch)
    science, timing = _production_authorities(
        tmp_path / f"{output_name}-authorities",
        monkeypatch,
    )
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / output_name,
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    prepared = writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=0.0,
    )
    fine = _fine_run(tmp_path / f"{output_name}-fine", experiment_id)
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    started = writer.append_fine_started(
        timing_scope="uninterrupted_total",
        budget_check=lambda: 0.0,
    )
    return writer, coarse, fine, generation, reference, prepared, started


def _final_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    coverage = (tmp_path / "final-coverage").resolve()
    reconstruction = (tmp_path / "final-reconstruction").resolve()
    coverage.mkdir(parents=True)
    reconstruction.mkdir(parents=True)
    (coverage / "coverage.json").write_text('{"coverage": "final"}\n', encoding="utf-8")
    (reconstruction / "final_reconstruction.json").write_text(
        '{"reconstruction": "final"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiment_module,
        "read_surface_coverage_generation",
        lambda path, *, require_foreground_bound_science: SimpleNamespace(
            root=Path(path).resolve(),
            generation_id="a" * 64,
            scientific=require_foreground_bound_science,
        ),
    )
    monkeypatch.setattr(
        experiment_module,
        "replay_final_fine_reconstruction",
        lambda path, *, expected_science_authority=None: SimpleNamespace(
            root=Path(path).resolve(),
            artifact_id="d" * 64,
            metadata_sha256="e" * 64,
            science_authority=expected_science_authority,
            result=SimpleNamespace(
                coverage=SimpleNamespace(
                    root=coverage,
                    generation_id="a" * 64,
                )
            ),
        ),
    )
    return coverage, reconstruction


def _append_terminal_event(
    fine: StopScanRunWriter,
    coverage: Path,
    reconstruction: Path,
) -> None:
    fine.append_event(
        phase="complete",
        cycle_index=1,
        event_type="coverage_complete",
        payload={
            "surface_generation_id": "a" * 64,
            "final_reconstruction": {
                "path": str(reconstruction),
                "artifact_id": "d" * 64,
                "metadata_sha256": "e" * 64,
            },
        },
    )


def _sealed_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    writer, coarse, fine, generation, reference, prepared, started = _complete_chain(
        tmp_path,
        monkeypatch,
    )
    coverage, reconstruction = _final_sources(tmp_path / "final-assets", monkeypatch)
    _append_terminal_event(fine, coverage, reconstruction)
    writer.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage,
    )
    completed = writer.append_fine_completed(
        final_surface_coverage_generation=coverage,
        final_reconstruction_product=reconstruction,
    )
    return (
        writer,
        coarse,
        fine,
        generation,
        reference,
        coverage,
        reconstruction,
        prepared,
        started,
        completed,
    )


def test_legacy_outer_chain_cannot_launder_schema2_final_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine = _fine_run(tmp_path / "fine", "laundering-001")
    coverage, reconstruction = _final_sources(tmp_path / "assets", monkeypatch)
    _append_terminal_event(fine, coverage, reconstruction)
    science, _timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    monkeypatch.setattr(
        experiment_module,
        "replay_final_fine_reconstruction",
        lambda path: SimpleNamespace(
            root=Path(path).resolve(),
            artifact_id="d" * 64,
            metadata_sha256="e" * 64,
            science_authority=science,
            result=SimpleNamespace(
                coverage=SimpleNamespace(
                    root=coverage,
                    generation_id="a" * 64,
                )
            ),
        ),
    )

    with pytest.raises(ValueError, match="replayed reconstruction disagree"):
        experiment_module._verify_fine_terminal_assets(
            experiment_module.read_stop_scan_run(fine.root),
            coverage_root=coverage,
            reconstruction_root=reconstruction,
            science_authority=None,
        )


def test_complete_handoff_chain_binds_both_runs_and_schema5_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, coarse, fine, generation, reference, prepared, started = _complete_chain(
        tmp_path,
        monkeypatch,
    )

    stored = read_unknown_blade_experiment(writer.root)

    assert stored.experiment_id == "experiment-001"
    assert tuple(event.event_type for event in stored.events) == (
        "experiment_initialized",
        "coarse_checkpoint",
        "handoff_prepared",
        "fine_start_candidate",
        "fine_started",
    )
    assert prepared.payload["coarse_run_root"] == str(coarse.root)
    assert prepared.payload["coarse_last_event_sha256"] == coarse.events[-1].event_sha256
    assert writer.events[1].payload["coarse_generation"]["root"] == str(
        _source_for(generation)
    )
    assert prepared.payload["schema5_generation"]["root"] == str(generation)
    assert prepared.payload["reference_coarse_model"]["root"] == str(reference)
    candidate = writer.events[3]
    assert candidate.previous_event_sha256 == prepared.event_sha256
    assert started.previous_event_sha256 == candidate.event_sha256
    assert started.payload["fine_start_candidate_event_sha256"] == candidate.event_sha256
    assert started.payload["prepared_event_sha256"] == prepared.event_sha256
    assert started.payload["fine_run_root"] == str(fine.root)
    assert started.payload["fine_first_event_sha256"] == fine.events[0].event_sha256


def test_resume_recomputes_prepared_chain_before_appending_fine_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = "resume-001"
    coarse = _run(tmp_path / "coarse", experiment_id, event_type="coarse_stopped")
    generation, reference = _sources(tmp_path / "assets", monkeypatch)
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    prepared = writer.prepare_handoff(
        schema5_generation=generation,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=0.0,
    )
    fine = _fine_run(tmp_path / "fine", experiment_id)

    resumed = UnknownBladeExperimentWriter.resume(writer.root)
    resumed.append_fine_start_candidate(fine_run_root=fine.root)
    started = resumed.append_fine_started(
        timing_scope="resume_fine_start",
        budget_check=lambda: 0.0,
    )

    assert started.payload["prepared_event_sha256"] == prepared.event_sha256
    assert len(read_unknown_blade_experiment(writer.root).events) == 5


def test_prepared_requires_ready_generation_derived_from_latest_coarse_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = "coarse-checkpoint-001"
    coarse = _run(tmp_path / "coarse", experiment_id, event_type="coarse_stopped")
    generation, reference = _sources(tmp_path / "assets", monkeypatch)
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )

    with pytest.raises(ValueError, match="COARSE_CHECKPOINT"):
        writer.prepare_handoff(
            schema5_generation=generation,
            reference_coarse_model=reference,
        )

    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))
    other = (tmp_path / "other-generation").resolve()
    other.mkdir()
    (other / "generation.json").write_text('{"schema": 5}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="final coarse checkpoint"):
        writer.prepare_handoff(
            schema5_generation=other,
            reference_coarse_model=reference,
            schema5_prepare_duration_s=0.0,
        )


def test_duplicate_coarse_checkpoint_without_run_or_generation_advance_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = "coarse-checkpoint-duplicate"
    coarse = _run(tmp_path / "coarse", experiment_id, event_type="coarse_stopped")
    generation, _reference = _sources(tmp_path / "assets", monkeypatch)
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))

    with pytest.raises(ValueError, match="must advance"):
        writer.append_coarse_checkpoint(coarse_generation=_source_for(generation))


def test_multiple_coarse_and_fine_checkpoints_replay_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = "multiple-checkpoints"
    coarse = _run(tmp_path / "coarse", experiment_id, event_type="coarse_view_000")
    generation1, reference = _sources(tmp_path / "assets", monkeypatch)
    science, timing = _production_authorities(tmp_path / "authorities", monkeypatch)
    writer = UnknownBladeExperimentWriter.create(
        tmp_path / "chain",
        experiment_id=experiment_id,
        coarse_run_root=coarse.root,
        science_authority=science,
        runtime_timing_authority=timing,
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation1))
    coarse.append_event(
        phase="map_ready",
        cycle_index=1,
        event_type="coarse_view_001",
        payload={"motion_authorized": False},
    )
    writer.append_coarse_checkpoint(coarse_generation=_source_for(generation1))
    generation2 = (tmp_path / "schema5-generation-002").resolve()
    source2 = _source_for(generation2)
    generation2.mkdir()
    source2.mkdir()
    (generation2 / "generation.json").write_text('{"schema": 5}\n', encoding="utf-8")
    (source2 / "generation.json").write_text('{"schema": "source-2"}\n', encoding="utf-8")
    writer.append_coarse_checkpoint(coarse_generation=source2)
    writer.prepare_handoff(
        schema5_generation=generation2,
        reference_coarse_model=reference,
        schema5_prepare_duration_s=0.0,
    )
    fine = _fine_run(tmp_path / "fine", experiment_id)
    writer.append_fine_start_candidate(fine_run_root=fine.root)
    writer.append_fine_started(
        timing_scope="uninterrupted_total",
        budget_check=lambda: 0.0,
    )
    coverage1, _ = _final_sources(tmp_path / "fine-assets-001", monkeypatch)
    writer.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage1,
    )
    fine.append_event(
        phase="map_ready",
        cycle_index=1,
        event_type="fine_view_001",
        payload={"motion_authorized": False},
    )
    writer.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage1,
    )
    coverage2 = (tmp_path / "fine-coverage-002").resolve()
    coverage2.mkdir()
    (coverage2 / "coverage.json").write_text('{"coverage": 2}\n', encoding="utf-8")
    writer.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage2,
    )

    stored = read_unknown_blade_experiment(writer.root)

    assert tuple(event.event_type for event in stored.events) == (
        "experiment_initialized",
        "coarse_checkpoint",
        "coarse_checkpoint",
        "coarse_checkpoint",
        "handoff_prepared",
        "fine_start_candidate",
        "fine_started",
        "fine_checkpoint",
        "fine_checkpoint",
        "fine_checkpoint",
    )


def test_fine_completed_seals_terminal_run_and_final_product_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        writer,
        _coarse,
        fine,
        _generation,
        _reference,
        coverage,
        reconstruction,
        _prepared,
        started,
        completed,
    ) = _sealed_chain(tmp_path, monkeypatch)

    stored = read_unknown_blade_experiment(writer.root)

    assert tuple(event.event_type for event in stored.events) == (
        "experiment_initialized",
        "coarse_checkpoint",
        "handoff_prepared",
        "fine_start_candidate",
        "fine_started",
        "fine_checkpoint",
        "fine_completed",
    )
    assert completed.previous_event_sha256 == stored.events[-2].event_sha256
    assert stored.events[-2].previous_event_sha256 == started.event_sha256
    assert completed.payload["fine_event_count"] == len(fine.events)
    assert completed.payload["fine_last_event_sha256"] == fine.events[-1].event_sha256
    assert completed.payload["final_surface_coverage_generation"]["root"] == str(coverage)
    assert completed.payload["final_reconstruction_product"]["root"] == str(reconstruction)


def test_resume_recomputes_full_chain_before_fine_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, _coarse, fine, *_ = _complete_chain(tmp_path, monkeypatch)
    coverage, reconstruction = _final_sources(tmp_path / "final", monkeypatch)
    _append_terminal_event(fine, coverage, reconstruction)

    resumed = UnknownBladeExperimentWriter.resume(writer.root)
    resumed.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage,
    )
    completed = resumed.append_fine_completed(
        final_surface_coverage_generation=coverage,
        final_reconstruction_product=reconstruction,
    )

    assert completed.sequence == 6
    assert read_unknown_blade_experiment(writer.root).latest_event == completed


def test_fine_checkpoint_refuses_duplicate_and_reader_rechecks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, _coarse, _fine, *_ = _complete_chain(tmp_path, monkeypatch)
    coverage, _reconstruction = _final_sources(tmp_path / "fine-assets", monkeypatch)
    writer.append_fine_checkpoint(
        accepted_surface_coverage_generation=coverage,
    )

    with pytest.raises(ValueError, match="must advance"):
        writer.append_fine_checkpoint(
            accepted_surface_coverage_generation=coverage,
        )

    (coverage / "coverage.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(UnknownBladeExperimentFormatError, match="authority content changed"):
        read_unknown_blade_experiment(writer.root)


def test_chain_rejects_tampering_and_missing_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, *_ = _complete_chain(tmp_path, monkeypatch)
    prepared_path = writer.root / "events" / "00000002.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    prepared["payload"]["coarse_last_event_sha256"] = "f" * 64
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")

    with pytest.raises(UnknownBladeExperimentFormatError, match="canonical content"):
        read_unknown_blade_experiment(writer.root)

    prepared_path.write_text(json.dumps(writer.events[2].to_payload()), encoding="utf-8")
    (writer.root / "events" / "00000003.json").rename(
        writer.root / "events" / "00000004.json"
    )
    with pytest.raises(UnknownBladeExperimentFormatError, match="non-contiguous"):
        read_unknown_blade_experiment(writer.root)


def test_chain_rejects_event_spliced_from_another_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, *_ = _complete_chain(
        tmp_path,
        monkeypatch,
        experiment_id="first-001",
        output_name="first-chain",
    )
    second, *_ = _complete_chain(
        tmp_path,
        monkeypatch,
        experiment_id="second-001",
        output_name="second-chain",
    )
    shutil.copyfile(
        second.root / "events" / "00000001.json",
        first.root / "events" / "00000001.json",
    )

    with pytest.raises(UnknownBladeExperimentFormatError, match="predecessor|ID changed"):
        read_unknown_blade_experiment(first.root)


def test_chain_rejects_changed_schema5_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, _coarse, _fine, _generation, reference, *_ = _complete_chain(
        tmp_path,
        monkeypatch,
    )
    (reference / "metadata.json").write_text('{"model": "tampered"}\n', encoding="utf-8")

    with pytest.raises(UnknownBladeExperimentFormatError, match="authority content changed"):
        read_unknown_blade_experiment(writer.root)


def test_completed_chain_rejects_fine_run_appended_after_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, _coarse, fine, *_ = _sealed_chain(tmp_path, monkeypatch)
    fine.append_event(
        phase="complete",
        cycle_index=2,
        event_type="late_event",
        payload={"motion_authorized": False},
    )

    with pytest.raises(UnknownBladeExperimentFormatError, match="terminal binding changed"):
        read_unknown_blade_experiment(writer.root)


@pytest.mark.parametrize(
    ("source_index", "filename"),
    [
        (5, "coverage.json"),
        (6, "final_reconstruction.json"),
    ],
)
def test_completed_chain_rejects_changed_final_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_index: int,
    filename: str,
) -> None:
    sealed = _sealed_chain(tmp_path, monkeypatch)
    writer = sealed[0]
    source = sealed[source_index]
    (source / filename).write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(UnknownBladeExperimentFormatError, match="authority content changed"):
        read_unknown_blade_experiment(writer.root)


def test_writer_refuses_duplicate_or_out_of_order_phase_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, _coarse, fine, generation, reference, *_ = _complete_chain(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(ValueError, match="preceding COARSE_CHECKPOINT"):
        writer.prepare_handoff(
            schema5_generation=generation,
            reference_coarse_model=reference,
        )
    with pytest.raises(ValueError, match="latest FINE_START_CANDIDATE"):
        writer.append_fine_started(
            timing_scope="uninterrupted_total",
            budget_check=lambda: 0.0,
        )


def test_writer_refuses_duplicate_fine_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = _sealed_chain(tmp_path, monkeypatch)
    writer = sealed[0]
    coverage = sealed[5]
    reconstruction = sealed[6]

    with pytest.raises(ValueError, match="preceding fine phase"):
        writer.append_fine_completed(
            final_surface_coverage_generation=coverage,
            final_reconstruction_product=reconstruction,
        )
