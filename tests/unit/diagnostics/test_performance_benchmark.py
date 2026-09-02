from __future__ import annotations

import json
from pathlib import Path

import pytest

from biblade_fusion.diagnostics.performance_benchmark import (
    _read_content_tree,
    discover_attempt_assets,
    rank_hotspots,
    run_attempt_benchmark,
    summarize_resource_trials,
    summarize_trials,
)


def _minimal_attempt(tmp_path: Path) -> Path:
    root = tmp_path / "attempt-09"
    cycle = root / "perception" / "coarse" / "cycles" / "000000_view"
    accepted = cycle / "attempt_local"
    accepted.mkdir(parents=True)
    (cycle / "committed.json").write_text(
        json.dumps(
            {
                "accepted_attempt": {
                    "attempt_id": "attempt_local",
                    "root": "/foreign/host/attempt-09/cycles/attempt_local",
                }
            }
        ),
        encoding="utf-8",
    )
    generation = root / "coarse_science" / "generations" / "000000"
    generation.mkdir(parents=True)
    (generation / "generation.json").write_text("{}\n", encoding="utf-8")
    return root


def test_discovery_uses_local_attempt_id_not_foreign_absolute_root(tmp_path: Path) -> None:
    root = _minimal_attempt(tmp_path)
    assets = discover_attempt_assets(root)
    assert assets.cycle_roots == (
        (
            root
            / "perception"
            / "coarse"
            / "cycles"
            / "000000_view"
            / "attempt_local"
        ).resolve(),
    )
    assert assets.generation_roots == (
        (root / "coarse_science" / "generations" / "000000").resolve(),
    )


def test_benchmark_refuses_to_write_inside_immutable_input(tmp_path: Path) -> None:
    root = _minimal_attempt(tmp_path)
    with pytest.raises(ValueError, match="outside the immutable experiment tree"):
        run_attempt_benchmark(root, root / "benchmark", cold_runs=1, warm_runs=1)


def test_content_readback_supports_single_json_file(tmp_path: Path) -> None:
    source = tmp_path / "trace.json"
    source.write_text('{"samples": [1, 2, 3]}\n', encoding="utf-8")
    byte_count, digest = _read_content_tree(source)
    assert byte_count == source.stat().st_size
    assert len(digest) == 64


def test_trial_summary_reports_nearest_rank_p95() -> None:
    trials = []
    for duration in (10, 20, 30, 40, 50):
        trials.append(
            {
                "spans": {
                    "work": {
                        "inclusive_wall_ns": duration,
                        "inclusive_cpu_ns": duration - 1,
                        "exclusive_wall_ns": duration - 2,
                        "exclusive_cpu_ns": duration - 3,
                    }
                }
            }
        )
    summary = summarize_trials(trials)["work"]
    assert summary["inclusive_wall_ns_p50"] == 30
    assert summary["inclusive_wall_ns_p95"] == 50
    assert summary["exclusive_cpu_ns_p50"] == 27
    ranked = rank_hotspots({"slow": summary, "fast": {**summary, "inclusive_cpu_ns_p50": 1}})
    assert [item["span"] for item in ranked["inclusive_cpu_ns_p50"]] == ["slow", "fast"]


def test_resource_summary_reports_peak_rss_and_io_deltas() -> None:
    trials = []
    for index in range(1, 6):
        trials.append(
            {
                "resource_start": {
                    "process_cpu_ns": 10,
                    "maximum_resident_set_size_native": 100,
                    "minor_page_faults": 2,
                    "major_page_faults": 0,
                    "voluntary_context_switches": 1,
                    "involuntary_context_switches": 1,
                    "proc_io": {
                        "read_bytes": 20,
                        "write_bytes": 30,
                        "rchar": 40,
                        "wchar": 50,
                    },
                },
                "resource_end": {
                    "process_cpu_ns": 10 + index,
                    "maximum_resident_set_size_native": 100 + index,
                    "minor_page_faults": 2 + index,
                    "major_page_faults": index,
                    "voluntary_context_switches": 1 + index,
                    "involuntary_context_switches": 1 + index,
                    "proc_io": {
                        "read_bytes": 20 + index,
                        "write_bytes": 30 + index,
                        "rchar": 40 + index,
                        "wchar": 50 + index,
                    },
                },
            }
        )

    summary = summarize_resource_trials(trials)
    assert summary["process_cpu_ns_delta"]["p50"] == 3
    assert summary["process_cpu_ns_delta"]["p95"] == 5
    assert summary["peak_rss_native_end"]["p50"] == 103
    assert summary["proc_io_read_bytes_delta"]["p95"] == 5
