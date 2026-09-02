from __future__ import annotations

import json
from pathlib import Path

import pytest

from biblade_fusion.diagnostics.performance_timing import (
    PerformanceTimingRecorder,
    activate_performance_timing,
    performance_span,
)


class _Clock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def test_nested_spans_report_inclusive_and_exclusive_time() -> None:
    wall = _Clock([0, 10, 40, 100])
    cpu = _Clock([0, 5, 25, 60])
    recorder = PerformanceTimingRecorder(
        transaction_kind="unit",
        monotonic_ns_clock=wall,
        process_time_ns_clock=cpu,
    )

    with recorder.span("outer"), recorder.span("inner"):
        pass

    spans = recorder.payload(status="completed")["spans"]
    assert spans["inner"] == {
        "count": 1,
        "failure_count": 0,
        "inclusive_wall_ns": 30,
        "exclusive_wall_ns": 30,
        "inclusive_cpu_ns": 20,
        "exclusive_cpu_ns": 20,
        "maximum_wall_ns": 30,
        "maximum_cpu_ns": 20,
    }
    assert spans["outer"]["inclusive_wall_ns"] == 100
    assert spans["outer"]["exclusive_wall_ns"] == 70
    assert spans["outer"]["inclusive_cpu_ns"] == 60
    assert spans["outer"]["exclusive_cpu_ns"] == 40


def test_span_aggregation_is_bounded_and_records_failures() -> None:
    clock = _Clock([0, 0, 1, 1, 2, 2, 3, 3])
    recorder = PerformanceTimingRecorder(
        transaction_kind="unit",
        maximum_spans=1,
        monotonic_ns_clock=clock,
        process_time_ns_clock=clock,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom"), recorder.span("fixed"):
            raise RuntimeError("boom")

    span = recorder.payload(status="failed")["spans"]["fixed"]
    assert span["count"] == 2
    assert span["failure_count"] == 2
    with pytest.raises(RuntimeError, match="fixed span-name bound"), recorder.span("second"):
        pass


def test_context_span_is_noop_without_recorder_and_aggregates_when_active() -> None:
    with performance_span("not-recorded"):
        pass

    wall = _Clock([0, 4])
    cpu = _Clock([0, 2])
    recorder = PerformanceTimingRecorder(
        transaction_kind="unit",
        monotonic_ns_clock=wall,
        process_time_ns_clock=cpu,
    )
    with activate_performance_timing(recorder), performance_span("recorded"):
        pass

    spans = recorder.payload(status="completed")["spans"]
    assert set(spans) == {"recorded"}


def test_best_effort_file_is_explicitly_non_authoritative(tmp_path: Path) -> None:
    recorder = PerformanceTimingRecorder(
        transaction_kind="perception_cycle",
        identity={"view_id": "view-1", "sequence_index": 3},
    )
    with recorder.span("small"):
        pass

    path = tmp_path / "performance_timing.json"
    assert recorder.write_best_effort(path, status="completed") is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "biblade_fusion.performance_timing_diagnostic"
    assert payload["authority"] == "diagnostic_only_not_safety_or_science_authority"
    assert payload["bounded_storage"]["per_call_trace_retained"] is False
    assert payload["identity"]["sequence_index"] == 3

    original = path.read_bytes()
    assert recorder.write_best_effort(path, status="completed") is False
    assert path.read_bytes() == original


def test_best_effort_write_failure_does_not_raise(tmp_path: Path) -> None:
    recorder = PerformanceTimingRecorder(transaction_kind="unit")
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    assert (
        recorder.write_best_effort(
            blocker / "performance_timing.json",
            status="failed",
            error="original transaction error",
        )
        is False
    )


def test_context_span_never_changes_observed_operation_result() -> None:
    def broken_clock() -> int:
        raise RuntimeError("diagnostic clock failed")

    recorder = PerformanceTimingRecorder(
        transaction_kind="unit",
        monotonic_ns_clock=broken_clock,
    )
    with activate_performance_timing(recorder), performance_span("broken-enter"):
        answer = 42
    assert answer == 42

    with (
        activate_performance_timing(recorder),
        pytest.raises(ValueError, match="operation failed"),
        performance_span("broken-operation"),
    ):
        raise ValueError("operation failed")


def test_context_span_does_not_swallow_process_control_exceptions() -> None:
    def stopping_clock() -> int:
        raise SystemExit(17)

    recorder = PerformanceTimingRecorder(
        transaction_kind="unit",
        monotonic_ns_clock=stopping_clock,
    )
    with (
        activate_performance_timing(recorder),
        pytest.raises(SystemExit) as captured,
        performance_span("interrupted-enter"),
    ):
        pytest.fail("operation must not run after a process-control exception")
    assert captured.value.code == 17
