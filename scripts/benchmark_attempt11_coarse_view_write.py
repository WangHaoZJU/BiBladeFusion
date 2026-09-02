#!/usr/bin/env python3
"""Benchmark the attempt-11 ``write_coarse_scan_view`` validation boundary.

The experiment input is immutable.  Every trial writes one new coarse-scan view
below an output directory that must be outside the experiment tree.  Fixture load
and the strict post-write ``read_coarse_scan_view`` oracle are deliberately outside
the measured target; only the production writer is timed.

Run this on the acquisition host, where all absolute, hash-bound FoundationStereo,
calibration, raw-session, and robot-model sources recorded by attempt-11 exist::

    /usr/bin/env -u PYTHONPATH .venv/bin/python -B \
      scripts/benchmark_attempt11_coarse_view_write.py \
      data/experiments/blade-placement-20260901-01-attempt-11 \
      /tmp/bbf-attempt11-coarse-writer-baseline \
      --expected-dda-count 3

Cold trials use a fresh spawned Python process.  This does not evict or control the
kernel page cache.  Warm trials reuse one process and one strictly loaded fixture.
The defaults are the required three cold and five warm trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import socket
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biblade_fusion.diagnostics.performance_benchmark import (
    rank_hotspots,
    summarize_resource_trials,
    summarize_trials,
)
from biblade_fusion.diagnostics.performance_timing import (
    PerformanceTimingRecorder,
    activate_performance_timing,
)
from biblade_fusion.perception.bootstrap_foreground import (
    array_content_sha256,
    bootstrap_seed_payload,
)
from biblade_fusion.storage.coarse_scan import (
    StoredCoarseScanView,
    read_coarse_scan_view,
    write_coarse_scan_view,
)

DEFAULT_COLD_RUNS = 3
DEFAULT_WARM_RUNS = 5
_TARGET_SPAN = "benchmark.write_coarse_scan_view"
_DDA_SPAN = "occupancy.depth_ray_integrator"


class CoarseViewWriteBenchmarkError(RuntimeError):
    """The benchmark could not prove strict input or output equivalence."""


@dataclass(frozen=True, slots=True)
class _TreeFingerprint:
    file_count: int
    size_bytes: int
    sha256: str

    def payload(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _WriterFixture:
    cycle_attempt_root: Path
    reconstructed_root: Path
    stereo_root: Path
    occupancy_root: Path
    oracle: StoredCoarseScanView
    normalized_semantic_payload: dict[str, Any]
    normalized_semantic_sha256: str


def _git_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    raise CoarseViewWriteBenchmarkError(
        f"Benchmark production module is not inside a Git worktree: {resolved}"
    )


def _git_text(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CoarseViewWriteBenchmarkError(
            f"Cannot record benchmark Git provenance: git {' '.join(arguments)}"
        ) from exc
    return completed.stdout.strip()


def _code_authority(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CoarseViewWriteBenchmarkError(f"Code authority is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _runtime_provenance() -> dict[str, Any]:
    production_module = Path(write_coarse_scan_view.__code__.co_filename).resolve()
    benchmark_script = Path(__file__).resolve()
    root = _git_root(production_module)
    production_relative = production_module.relative_to(root).as_posix()
    status_paths = [production_relative]
    benchmark_relative: str | None = None
    if benchmark_script.is_relative_to(root):
        benchmark_relative = benchmark_script.relative_to(root).as_posix()
        status_paths.append(benchmark_relative)
    return {
        "hostname": socket.gethostname(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "git_root": str(root),
        "git_head": _git_text(root, "rev-parse", "HEAD"),
        "git_status_short": _git_text(
            root,
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            *status_paths,
        ),
        "production_module_git_path": production_relative,
        "benchmark_script_git_path": benchmark_relative,
        "code_authorities": {
            "production_coarse_scan": _code_authority(production_module),
            "benchmark_script": _code_authority(benchmark_script),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_fingerprint(root: Path) -> _TreeFingerprint:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Fingerprint root is not a directory: {resolved}")
    digest = hashlib.sha256()
    size_bytes = 0
    file_count = 0
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        size = path.stat().st_size
        content_sha256 = _sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
        size_bytes += size
        file_count += 1
    return _TreeFingerprint(file_count, size_bytes, digest.hexdigest())


def _array_fingerprint(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "content_sha256": array_content_sha256(array),
    }


def _normalized_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    # A JSON round trip both detaches the source object and proves that the payload
    # remains deterministic JSON.  Creation time is the sole expected difference
    # between two semantically equivalent append-only writer outputs.
    normalized = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    if not isinstance(normalized, dict):
        raise ValueError("Coarse-scan metadata must be an object")
    created_at_utc = normalized.pop("created_at_utc", None)
    if not isinstance(created_at_utc, str) or not created_at_utc.strip():
        raise ValueError("Coarse-scan metadata creation time is missing")
    return normalized


def _semantic_payload(stored: StoredCoarseScanView) -> dict[str, Any]:
    foreground = stored.foreground
    reconstructed = stored.reconstructed
    view = reconstructed.view
    support_cloud = stored.support_cloud
    side = stored.target_side.value
    return {
        "normalized_metadata": _normalized_metadata(stored.metadata),
        "target": {
            "view_id": stored.target_view_id,
            "kind": stored.target_kind,
            "side": side,
        },
        "reconstructed_identity": {
            "view_id": view.source_view_id,
            "sequence_index": view.source_sequence_index,
            "frame_number": view.source_frame_number,
            "blade_mask": _array_fingerprint(reconstructed.blade_mask),
        },
        "foreground": {
            "algorithm": foreground.algorithm,
            "config": asdict(foreground.config),
            "seed": bootstrap_seed_payload(foreground.seed),
            "diagnostics": asdict(foreground.diagnostics),
            "policy_sha256": foreground.policy_sha256,
            "left_image_content_sha256": foreground.left_image_content_sha256,
            "depth_content_sha256": foreground.depth_content_sha256,
            "valid_mask_content_sha256": foreground.valid_mask_content_sha256,
            "mask": _array_fingerprint(foreground.mask),
            "seed_mask": _array_fingerprint(foreground.seed_mask),
        },
        "proxy": {
            "configuration": stored.proxy_config.model_dump(mode="json"),
            "diagnostics": stored.proxy_support.metadata_payload(),
            "support_mask": _array_fingerprint(stored.proxy_support.mask),
        },
        "support_cloud": {
            "frame": support_cloud.frame,
            "source_image_shape": list(support_cloud.source_image_shape),
            "points_m": _array_fingerprint(support_cloud.points_m),
            "pixel_uv": _array_fingerprint(support_cloud.pixel_uv),
        },
    }


def _semantic_digest(stored: StoredCoarseScanView) -> tuple[dict[str, Any], str]:
    payload = _semantic_payload(stored)
    return payload, _canonical_sha256(payload)


def _discover_cycle_attempt_root(input_root: Path) -> Path:
    root = input_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark input is not a directory: {root}")
    if (root / "coarse_scan_view").is_dir():
        return root

    markers = sorted((root / "perception/coarse/cycles").glob("*/committed.json"))
    if len(markers) != 1:
        raise ValueError(
            "This benchmark requires exactly one committed coarse cycle; "
            f"found {len(markers)} in {root}"
        )
    marker = markers[0]
    payload = json.loads(marker.read_text(encoding="utf-8"))
    accepted = payload.get("accepted_attempt")
    if not isinstance(accepted, dict):
        raise ValueError(f"Committed marker has no accepted_attempt object: {marker}")
    attempt_id = str(accepted.get("attempt_id", "")).strip()
    if not attempt_id or Path(attempt_id).name != attempt_id:
        raise ValueError(f"Committed marker has an invalid attempt identity: {marker}")
    cycle_attempt = (marker.parent / attempt_id).resolve()
    if cycle_attempt.parent != marker.parent.resolve() or not cycle_attempt.is_dir():
        raise ValueError(f"Committed cycle attempt is unavailable: {cycle_attempt}")
    return cycle_attempt


def _source_root(metadata: Mapping[str, Any], name: str) -> Path:
    sources = metadata.get("sources")
    if not isinstance(sources, Mapping):
        raise CoarseViewWriteBenchmarkError("Coarse-scan metadata sources are missing")
    record = sources.get(name)
    if not isinstance(record, Mapping):
        raise CoarseViewWriteBenchmarkError(f"Coarse-scan source record is missing: {name}")
    raw = Path(str(record.get("root", "")))
    root = raw.resolve()
    if not raw.is_absolute() or raw != root or not root.is_dir():
        raise CoarseViewWriteBenchmarkError(
            f"Strict coarse-scan source is unavailable at its recorded path: {raw}"
        )
    return root


def _load_fixture(cycle_attempt_root: Path) -> _WriterFixture:
    coarse_root = cycle_attempt_root / "coarse_scan_view"
    try:
        oracle = read_coarse_scan_view(coarse_root)
    except Exception as exc:
        raise CoarseViewWriteBenchmarkError(
            "Strict fixture readback failed. Run on the acquisition host with every "
            "recorded FoundationStereo checkpoint/source, calibration, raw session, "
            "and robot-model asset present at its exact hash-bound path."
        ) from exc

    reconstructed_root = _source_root(oracle.metadata, "reconstructed_view")
    stereo_root = _source_root(oracle.metadata, "stereo_inference")
    occupancy_root = _source_root(oracle.metadata, "occupancy_mapping")
    expected = {
        "reconstructed_view": (cycle_attempt_root / "coarse_reconstructed_view").resolve(),
        "stereo_inference": (cycle_attempt_root / "stereo_inference").resolve(),
        "occupancy_mapping": (cycle_attempt_root / "occupancy_mapping").resolve(),
    }
    actual = {
        "reconstructed_view": reconstructed_root,
        "stereo_inference": stereo_root,
        "occupancy_mapping": occupancy_root,
    }
    if actual != expected:
        raise CoarseViewWriteBenchmarkError(
            "Attempt-11 coarse source roots are not the expected direct immutable children"
        )
    payload, digest = _semantic_digest(oracle)
    return _WriterFixture(
        cycle_attempt_root=cycle_attempt_root,
        reconstructed_root=reconstructed_root,
        stereo_root=stereo_root,
        occupancy_root=occupancy_root,
        oracle=oracle,
        normalized_semantic_payload=payload,
        normalized_semantic_sha256=digest,
    )


def _run_one_trial(
    cycle_attempt_root: Path,
    trial_root: Path,
    *,
    expected_dda_count: int,
    fixture: _WriterFixture | None = None,
) -> dict[str, Any]:
    # Fixture construction is intentionally before recorder construction so none
    # of its strict reader cost or process-resource activity enters the target.
    loaded = fixture if fixture is not None else _load_fixture(cycle_attempt_root)
    destination = trial_root.resolve() / "coarse_scan_view"
    if destination.exists():
        raise FileExistsError(f"Trial output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=False)

    recorder = PerformanceTimingRecorder(
        transaction_kind="attempt11_coarse_view_write_benchmark",
        identity={
            "input_cycle_attempt": str(cycle_attempt_root.resolve()),
            "output": str(destination),
        },
    )
    with activate_performance_timing(recorder), recorder.span(_TARGET_SPAN):
        written = Path(
            write_coarse_scan_view(
                destination,
                loaded.oracle.foreground,
                reconstructed_view=loaded.reconstructed_root,
                source_stereo_inference=loaded.stereo_root,
                source_occupancy_mapping=loaded.occupancy_root,
                target_view_id=loaded.oracle.target_view_id,
                target_kind=loaded.oracle.target_kind,
                target_side=loaded.oracle.target_side,
                proxy_config=loaded.oracle.proxy_config,
            )
        ).resolve()
    timing = recorder.payload(status="completed")
    if written != destination:
        raise CoarseViewWriteBenchmarkError(
            f"Writer returned an unexpected output: {written} != {destination}"
        )
    spans = timing["spans"]
    if not isinstance(spans, dict) or _TARGET_SPAN not in spans:
        raise CoarseViewWriteBenchmarkError("Writer target timing span is missing")
    dda = spans.get(_DDA_SPAN)
    if not isinstance(dda, dict):
        raise CoarseViewWriteBenchmarkError(
            "Timed production writer did not execute a full DDA-backed occupancy verification"
        )
    dda_call_count = int(dda.get("count", 0))
    if dda_call_count != expected_dda_count:
        raise CoarseViewWriteBenchmarkError(
            "Timed production writer DDA count differs from the explicit oracle: "
            f"observed={dda_call_count}, expected={expected_dda_count}"
        )

    # This strict readback and all semantic comparison work are outside both the
    # activation context and the already materialized timing/resource payload.
    try:
        stored = read_coarse_scan_view(written)
    except Exception as exc:
        raise CoarseViewWriteBenchmarkError(
            f"Strict post-write readback failed for {written}"
        ) from exc
    semantic_payload, semantic_sha256 = _semantic_digest(stored)
    if (
        semantic_sha256 != loaded.normalized_semantic_sha256
        or semantic_payload != loaded.normalized_semantic_payload
    ):
        raise CoarseViewWriteBenchmarkError(
            "Writer output differs from the immutable attempt-11 semantic oracle"
        )

    output_fingerprint = _tree_fingerprint(written)
    result = dict(timing)
    result.update(
        {
            "dda_call_count": dda_call_count,
            "normalized_semantic_sha256": semantic_sha256,
            "output_tree": output_fingerprint.payload(),
        }
    )
    return result


def _cold_worker(arguments: tuple[str, str, int]) -> dict[str, Any]:
    if multiprocessing.current_process().daemon:
        raise CoarseViewWriteBenchmarkError(
            "Cold benchmark workers must be non-daemonic because the production "
            "occupancy verifier creates its own process pool"
        )
    cycle_attempt, trial_root, expected_dda_count = arguments
    return _run_one_trial(
        Path(cycle_attempt),
        Path(trial_root),
        expected_dda_count=expected_dda_count,
    )


def _validate_run_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not 1 <= value <= 100:
        raise ValueError(f"{name} must lie in [1, 100]")


def _validate_benchmark_paths(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    source = input_root.resolve()
    output = output_root.resolve()
    if not source.is_dir():
        raise ValueError(f"Benchmark input is not a directory: {source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Benchmark output must be outside the immutable input tree")
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    return source, output


def run_benchmark(
    input_root: str | Path,
    output_root: str | Path,
    *,
    expected_dda_count: int,
    cold_runs: int = DEFAULT_COLD_RUNS,
    warm_runs: int = DEFAULT_WARM_RUNS,
) -> Path:
    """Run strict cold/warm trials and publish one diagnostic JSON report."""

    _validate_run_count("cold_runs", cold_runs)
    _validate_run_count("warm_runs", warm_runs)
    _validate_run_count("expected_dda_count", expected_dda_count)
    source, output = _validate_benchmark_paths(Path(input_root), Path(output_root))
    runtime_before = _runtime_provenance()
    cycle_attempt = _discover_cycle_attempt_root(source)
    if not cycle_attempt.is_relative_to(source):
        raise CoarseViewWriteBenchmarkError("Discovered cycle attempt escapes the input tree")

    input_before = _tree_fingerprint(source)
    # The main-process fixture becomes the warm fixture and independently proves
    # that every cold worker must reproduce the same semantic oracle digest.
    warm_fixture = _load_fixture(cycle_attempt)
    output.mkdir(parents=True, exist_ok=False)

    context = multiprocessing.get_context("spawn")
    cold_trials: list[dict[str, Any]] = []
    for index in range(cold_runs):
        trial_root = output / f"cold_{index:02d}"
        arguments = (str(cycle_attempt), str(trial_root), expected_dda_count)
        # A new executor per trial guarantees a fresh, non-daemonic Python process.
        # Non-daemonic is required because the production occupancy verifier starts
        # its own ProcessPoolExecutor.  The OS page cache is intentionally neither
        # evicted nor claimed as cold.
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            cold_trials.append(executor.submit(_cold_worker, arguments).result())

    warm_trials = [
        _run_one_trial(
            cycle_attempt,
            output / f"warm_{index:02d}",
            expected_dda_count=expected_dda_count,
            fixture=warm_fixture,
        )
        for index in range(warm_runs)
    ]
    all_trials = [*cold_trials, *warm_trials]
    if any(
        trial["normalized_semantic_sha256"]
        != warm_fixture.normalized_semantic_sha256
        for trial in all_trials
    ):
        raise CoarseViewWriteBenchmarkError(
            "Cold/warm trials do not share one normalized semantic oracle"
        )
    output_fingerprints = [trial["output_tree"] for trial in all_trials]
    output_layouts = [
        {
            "file_count": int(item["file_count"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in output_fingerprints
    ]
    if any(item != output_layouts[0] for item in output_layouts[1:]):
        raise CoarseViewWriteBenchmarkError(
            "Cold/warm writer output file counts or sizes differ"
        )

    input_after = _tree_fingerprint(source)
    if input_after != input_before:
        raise CoarseViewWriteBenchmarkError(
            "Immutable benchmark input content changed during the run"
        )
    runtime_after = _runtime_provenance()
    if runtime_after != runtime_before:
        raise CoarseViewWriteBenchmarkError(
            "Benchmark host, revision, status, or code authority changed during the run"
        )

    cold_summary = summarize_trials(cold_trials)
    warm_summary = summarize_trials(warm_trials)
    report = {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.attempt11_coarse_view_write_benchmark",
        "authority": "diagnostic_only_not_safety_or_science_authority",
        "verification_scope": (
            "production write_coarse_scan_view plus production strict post-write "
            "read_coarse_scan_view; no verifier substitution or relocation bypass"
        ),
        "runtime_provenance_before": runtime_before,
        "runtime_provenance_after": runtime_after,
        "input": {
            "root": str(source),
            "cycle_attempt_root": str(cycle_attempt),
            "tree_before": input_before.payload(),
            "tree_after": input_after.payload(),
            "write_policy": "read_only_immutable_experiment",
        },
        "output": str(output),
        "target_span": _TARGET_SPAN,
        "dda_span": _DDA_SPAN,
        "dda_count_oracle": expected_dda_count,
        "cold_definition": (
            "fresh_spawned_python_process; fixture_load_outside_target; "
            "kernel_page_cache_not_evicted_or_controlled"
        ),
        "warm_definition": (
            "same_python_process_and_strict_fixture; fixture_load_outside_target"
        ),
        "post_readback_definition": (
            "production strict read_coarse_scan_view outside timed/resource target"
        ),
        "requested_runs": {"cold": cold_runs, "warm": warm_runs},
        "normalized_semantic_sha256": warm_fixture.normalized_semantic_sha256,
        "output_layout_oracle": output_layouts[0],
        "cold_trials": cold_trials,
        "warm_trials": warm_trials,
        "cold_summary": cold_summary,
        "warm_summary": warm_summary,
        "resource_summary": {
            "scope": (
                "process-only RUSAGE_SELF and /proc/self/io; production child-process "
                "CPU and RSS are excluded, so wall time is the authoritative DDA metric"
            ),
            "cold": summarize_resource_trials(cold_trials),
            "warm": summarize_resource_trials(warm_trials),
            "warm_peak_rss_note": (
                "process-lifetime high-water mark; warm trials share one process"
            ),
            "gpu_memory": "not_applicable_no_gpu_workload",
        },
        "ranked_hotspots": {
            "cold": rank_hotspots(cold_summary),
            "warm": rank_hotspots(warm_summary),
        },
    }
    report_path = output / "coarse_view_write_benchmark.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--cold-runs", type=int, default=DEFAULT_COLD_RUNS)
    parser.add_argument("--warm-runs", type=int, default=DEFAULT_WARM_RUNS)
    parser.add_argument(
        "--expected-dda-count",
        type=int,
        required=True,
        help="Exact DDA invocation count required in every cold and warm trial",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    report = run_benchmark(
        arguments.input_root,
        arguments.output_root,
        expected_dda_count=arguments.expected_dda_count,
        cold_runs=arguments.cold_runs,
        warm_runs=arguments.warm_runs,
    )
    print(report)


if __name__ == "__main__":
    main()
