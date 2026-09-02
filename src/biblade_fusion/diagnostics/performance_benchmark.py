"""Read-only Phase-0 benchmark for an immutable unknown-blade experiment.

Run this module directly, for example::

    python -m biblade_fusion.diagnostics.performance_benchmark \
      data/experiments/blade-placement-20260901-01-attempt-09 \
      /tmp/bbf-attempt-09-phase0

The input tree is never written.  ``artifacts`` uses the production stereo reader
and relocation-safe content/integrity checks for artifacts whose copied metadata
contains absolute paths from another host.  It is diagnostic, not a substitute for
the production full-semantic readers.  ``ray-replay`` additionally replays stored
rays through the unchanged Python DDA integrator and checks every replayed snapshot
against the persisted oracle.  Cold trials use a fresh spawned process; OS
page-cache state is reported as uncontrolled rather than falsely claiming eviction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AcquisitionConfig, OccupancyConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.diagnostics.performance_timing import (
    PerformanceTimingRecorder,
    activate_performance_timing,
)
from biblade_fusion.mapping import DepthIntegrationConfig, DepthRayIntegrator
from biblade_fusion.storage.occupancy_mapping import (
    _evidence_from_payload,
    _load_array,
    _load_snapshot_record,
    _validate_array_dtypes,
)
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.workflows.occupancy_mapping import (
    OccupancyFrameUpdate,
    OccupancyMappingContext,
)

BenchmarkSuite = Literal["artifacts", "ray-replay"]


@dataclass(frozen=True, slots=True)
class AttemptAssets:
    root: Path
    cycle_roots: tuple[Path, ...]
    generation_roots: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _RelocatedOccupancy:
    """Motion-ineligible local replay data with external paths intentionally omitted."""

    context: OccupancyMappingContext
    updates: tuple[OccupancyFrameUpdate, ...]
    metadata: dict[str, Any]


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def discover_attempt_assets(input_root: str | Path) -> AttemptAssets:
    """Resolve accepted attempts locally instead of trusting foreign absolute paths."""

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark input is not a directory: {root}")
    markers = sorted((root / "perception" / "coarse" / "cycles").glob("*/committed.json"))
    cycle_roots: list[Path] = []
    for marker in markers:
        payload = _mapping(json.loads(marker.read_text(encoding="utf-8")), label=str(marker))
        accepted = _mapping(payload.get("accepted_attempt"), label="accepted_attempt")
        attempt_id = str(accepted.get("attempt_id", "")).strip()
        if not attempt_id or Path(attempt_id).name != attempt_id:
            raise ValueError(f"Invalid accepted attempt identity in {marker}")
        local_root = (marker.parent / attempt_id).resolve()
        if local_root.parent != marker.parent.resolve() or not local_root.is_dir():
            raise ValueError(f"Accepted attempt is unavailable beside {marker}")
        cycle_roots.append(local_root)
    if not cycle_roots:
        raise ValueError("Benchmark input has no committed coarse perception cycles")
    generations = tuple(
        path.parent.resolve()
        for path in sorted((root / "coarse_science" / "generations").glob("*/generation.json"))
    )
    if not generations:
        raise ValueError("Benchmark input has no coarse generations")
    return AttemptAssets(root, tuple(cycle_roots), generations)


def _artifact_tree_size_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _read_content_tree(root: Path) -> tuple[int, str]:
    """Read, parse and hash one immutable tree without following recorded host paths."""

    digest = hashlib.sha256()
    byte_count = 0
    paths = (root,) if root.is_file() else tuple(
        sorted(item for item in root.rglob("*") if item.is_file())
    )
    for path in paths:
        relative_path = Path(path.name) if root.is_file() else path.relative_to(root)
        relative = relative_path.as_posix().encode("utf-8")
        payload = path.read_bytes()
        byte_count += len(payload)
        digest.update(relative)
        digest.update(b"\x00")
        digest.update(payload)
        if path.suffix == ".json":
            json.loads(payload)
        elif path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if array.dtype.hasobject:
                raise ValueError(f"Object array is forbidden in benchmark input: {path}")
    return byte_count, digest.hexdigest()


def _read_relocated_occupancy(root: Path) -> _RelocatedOccupancy:
    """Read stored arrays while omitting foreign-host source path checks.

    This relocation-safe path checks local file checksums, dtypes/shapes, snapshot
    identities and typed per-frame constructor invariants.  It does not claim the
    complete production evidence/hash-chain or foreign source-path validation.
    ``ray-replay`` separately runs the unchanged integrator against selected stored
    snapshots, but still is not a motion attestation or production semantic reader.
    """

    metadata = _mapping(json.loads((root / "metadata.json").read_bytes()), label="metadata")
    configuration = _mapping(metadata["configuration"], label="configuration")
    OccupancyConfig.model_validate(
        _mapping(configuration["occupancy"], label="occupancy configuration")
    )
    AcquisitionConfig.model_validate(
        _mapping(configuration["acquisition"], label="acquisition configuration")
    )
    context_record = _mapping(metadata["mapping_context"], label="mapping_context")
    context = OccupancyMappingContext(
        json.dumps(
            _mapping(context_record["payload"], label="mapping context payload"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        str(context_record["content_hash"]),
    )
    updates: list[OccupancyFrameUpdate] = []
    for raw_frame in metadata["frames"]:
        frame = _mapping(raw_frame, label="occupancy frame")
        files = _mapping(frame["files"], label="occupancy frame files")
        source_depth = _load_array(root, _mapping(files["source_depth_m"], label="source depth"))
        stereo_valid = _load_array(
            root,
            _mapping(files["stereo_valid_mask"], label="stereo valid mask"),
        )
        stereo_confidence = _load_array(
            root,
            _mapping(files["stereo_confidence"], label="stereo confidence"),
        )
        predicted_depth = _load_array(
            root,
            _mapping(files["predicted_robot_depth_m"], label="predicted robot depth"),
        )
        robot_mask = _load_array(root, _mapping(files["robot_mask"], label="robot mask"))
        integration_mask = _load_array(
            root,
            _mapping(files["integration_valid_mask"], label="integration valid mask"),
        )
        _validate_array_dtypes(
            source_depth=source_depth,
            stereo_valid=stereo_valid,
            stereo_confidence=stereo_confidence,
            predicted_depth=predicted_depth,
            robot_mask=robot_mask,
            integration_mask=integration_mask,
        )
        updates.append(
            OccupancyFrameUpdate(
                _load_snapshot_record(
                    root,
                    _mapping(frame["result_snapshot"], label="result snapshot"),
                ),
                _load_snapshot_record(
                    root,
                    _mapping(frame["mapping_snapshot"], label="mapping snapshot"),
                ),
                context,
                source_depth,
                stereo_valid,
                stereo_confidence,
                predicted_depth,
                robot_mask,
                integration_mask,
                _evidence_from_payload(_mapping(frame["evidence"], label="frame evidence")),
            )
        )
    final = _load_snapshot_record(root, _mapping(metadata["snapshot"], label="final snapshot"))
    if not updates or final != updates[-1].snapshot:
        raise ValueError("Relocated occupancy final snapshot differs from its frame chain")
    return _RelocatedOccupancy(context, tuple(updates), dict(metadata))


def _run_artifact_readbacks(assets: AttemptAssets, recorder: PerformanceTimingRecorder) -> None:
    with recorder.span("stereo.artifact_readback"):
        for cycle in assets.cycle_roots:
            read_stereo_inference(cycle / "stereo_inference")
    with recorder.span("occupancy.relocated_content_readback"):
        for cycle in assets.cycle_roots:
            _read_relocated_occupancy(cycle / "occupancy_mapping")
    with recorder.span("coarse.scan_view_content_readback"):
        for cycle in assets.cycle_roots:
            _read_content_tree(cycle / "coarse_scan_view")
    with recorder.span("stationarity.authority_content_readback"):
        for cycle in assets.cycle_roots:
            _read_content_tree(cycle / "inference_stationarity.json")
    with recorder.span("stationarity.trace_content_readback"):
        for cycle in assets.cycle_roots:
            _read_content_tree(cycle / "inference_stationarity_trace.json")
    with recorder.span("coarse.generation_content_readback"):
        for generation in assets.generation_roots:
            _read_content_tree(generation)


def _camera_intrinsics(payload: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        int(payload["width"]),
        int(payload["height"]),
        float(payload["fx"]),
        float(payload["fy"]),
        float(payload["cx"]),
        float(payload["cy"]),
        str(payload["distortion_model"]),
        tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _replay_latest_rays(
    assets: AttemptAssets,
    recorder: PerformanceTimingRecorder,
    *,
    source_limit: int | None,
) -> None:
    latest_mapping = assets.cycle_roots[-1] / "occupancy_mapping"
    with recorder.span("occupancy.ray_replay_fixture_read"):
        decoded = _read_relocated_occupancy(latest_mapping)
    configuration = _mapping(decoded.metadata["configuration"], label="configuration")
    occupancy = OccupancyConfig.model_validate(
        _mapping(configuration["occupancy"], label="occupancy configuration")
    )
    context = decoded.context.to_payload()
    rectified = _mapping(context["rectified_stereo"], label="rectified_stereo")
    intrinsics = _camera_intrinsics(_mapping(rectified["left"], label="left intrinsics"))
    updates = decoded.updates if source_limit is None else decoded.updates[:source_limit]
    previous = None
    with recorder.span("occupancy.ray_replay"):
        for update in updates:
            evidence = update.evidence
            integrator = DepthRayIntegrator(
                update.mapping_snapshot.geometry_spec(),
                DepthIntegrationConfig(
                    minimum_depth_m=occupancy.minimum_depth_m,
                    maximum_depth_m=occupancy.maximum_depth_m,
                    pixel_stride=occupancy.integration_stride,
                    minimum_valid_rays=1,
                    free_space_margin_m=occupancy.free_space_margin_m,
                    minimum_free_observations=occupancy.minimum_free_observations,
                    minimum_free_view_translation_m=(occupancy.minimum_free_view_translation_m),
                    minimum_free_view_direction_deg=(occupancy.minimum_free_view_direction_deg),
                    ray_integration_backend=occupancy.ray_integration_backend,
                ),
                mapping_context_hash=decoded.context.content_hash,
            )
            replayed = integrator.integrate(
                previous,
                update.source_depth_m,
                intrinsics,
                PoseSE3("base", "left_rectified", evidence.base_t_camera_matrix),
                valid_mask=update.integration_valid_mask,
                source_view_id=evidence.physical_source_id,
                observed_at_utc=datetime.fromisoformat(evidence.captured_at_utc),
            )
            if replayed != update.mapping_snapshot:
                raise ValueError(
                    "Offline ray replay differs from the immutable occupancy snapshot "
                    f"for {evidence.source_view_id}"
                )
            # Actual source-window rebuilding carries the quality-bound result
            # snapshot into the next integration, not the pre-quality map snapshot.
            previous = update.snapshot


def _one_trial(
    input_root: str,
    suite: BenchmarkSuite,
    ray_source_limit: int | None,
) -> dict[str, object]:
    gc.collect()
    assets = discover_attempt_assets(input_root)
    recorder = PerformanceTimingRecorder(
        transaction_kind=f"phase0_offline_{suite}",
        identity={
            "input_root": str(assets.root),
            "cycle_count": len(assets.cycle_roots),
            "generation_count": len(assets.generation_roots),
            "ray_source_limit": ray_source_limit,
        },
    )
    with activate_performance_timing(recorder), recorder.span("benchmark.total"):
        _run_artifact_readbacks(assets, recorder)
        if suite == "ray-replay":
            _replay_latest_rays(assets, recorder, source_limit=ray_source_limit)
    return recorder.payload(status="completed")


def _cold_trial_worker(arguments: tuple[str, BenchmarkSuite, int | None]) -> dict[str, object]:
    return _one_trial(*arguments)


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty timing series")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize_trials(trials: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize fixed span aggregates across otherwise independent trials."""

    names = sorted(
        {str(name) for trial in trials for name in _mapping(trial["spans"], label="trial spans")}
    )
    summary: dict[str, object] = {}
    for name in names:
        records = [
            _mapping(_mapping(trial["spans"], label="trial spans")[name], label=name)
            for trial in trials
            if name in _mapping(trial["spans"], label="trial spans")
        ]
        wall = [int(record["inclusive_wall_ns"]) for record in records]
        cpu = [int(record["inclusive_cpu_ns"]) for record in records]
        exclusive_wall = [int(record["exclusive_wall_ns"]) for record in records]
        exclusive_cpu = [int(record["exclusive_cpu_ns"]) for record in records]
        summary[name] = {
            "trial_count": len(records),
            "inclusive_wall_ns_p50": int(statistics.median(wall)),
            "inclusive_wall_ns_p95": _nearest_rank(wall, 0.95),
            "inclusive_cpu_ns_p50": int(statistics.median(cpu)),
            "inclusive_cpu_ns_p95": _nearest_rank(cpu, 0.95),
            "exclusive_wall_ns_p50": int(statistics.median(exclusive_wall)),
            "exclusive_wall_ns_p95": _nearest_rank(exclusive_wall, 0.95),
            "exclusive_cpu_ns_p50": int(statistics.median(exclusive_cpu)),
            "exclusive_cpu_ns_p95": _nearest_rank(exclusive_cpu, 0.95),
        }
    return summary


def summarize_resource_trials(
    trials: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize per-trial process resource deltas and peak RSS.

    ``ru_maxrss`` is a process-lifetime high-water mark.  It is therefore an
    independent peak for each fresh-process cold trial and a cumulative
    high-water mark for warm trials; the report labels that distinction instead
    of presenting it as instantaneous resident memory.
    """

    values: dict[str, list[int]] = {
        "process_cpu_ns_delta": [],
        "peak_rss_native_end": [],
        "minor_page_faults_delta": [],
        "major_page_faults_delta": [],
        "voluntary_context_switches_delta": [],
        "involuntary_context_switches_delta": [],
        "proc_io_read_bytes_delta": [],
        "proc_io_write_bytes_delta": [],
        "proc_io_rchar_delta": [],
        "proc_io_wchar_delta": [],
    }
    for trial in trials:
        start = _mapping(trial["resource_start"], label="resource_start")
        end = _mapping(trial["resource_end"], label="resource_end")
        values["process_cpu_ns_delta"].append(
            max(0, int(end["process_cpu_ns"]) - int(start["process_cpu_ns"]))
        )
        values["peak_rss_native_end"].append(
            int(end["maximum_resident_set_size_native"])
        )
        for name in (
            "minor_page_faults",
            "major_page_faults",
            "voluntary_context_switches",
            "involuntary_context_switches",
        ):
            values[f"{name}_delta"].append(max(0, int(end[name]) - int(start[name])))
        start_io = start.get("proc_io")
        end_io = end.get("proc_io")
        if isinstance(start_io, Mapping) and isinstance(end_io, Mapping):
            for name in ("read_bytes", "write_bytes", "rchar", "wchar"):
                values[f"proc_io_{name}_delta"].append(
                    max(0, int(end_io.get(name, 0)) - int(start_io.get(name, 0)))
                )

    units = {
        "process_cpu_ns_delta": "ns",
        "peak_rss_native_end": "KiB_on_linux_bytes_on_macos",
        "minor_page_faults_delta": "count",
        "major_page_faults_delta": "count",
        "voluntary_context_switches_delta": "count",
        "involuntary_context_switches_delta": "count",
        "proc_io_read_bytes_delta": "bytes_from_storage_layer",
        "proc_io_write_bytes_delta": "bytes_to_storage_layer",
        "proc_io_rchar_delta": "bytes_returned_by_read_like_syscalls",
        "proc_io_wchar_delta": "bytes_supplied_to_write_like_syscalls",
    }
    return {
        name: {
            "trial_count": len(series),
            "p50": int(statistics.median(series)),
            "p95": _nearest_rank(series, 0.95),
            "unit": units[name],
        }
        for name, series in values.items()
        if series
    }


def rank_hotspots(summary: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    """Rank p50 spans by inclusive/exclusive wall and CPU time."""

    rankings: dict[str, list[dict[str, object]]] = {}
    for metric in (
        "inclusive_wall_ns_p50",
        "exclusive_wall_ns_p50",
        "inclusive_cpu_ns_p50",
        "exclusive_cpu_ns_p50",
    ):
        ordered = sorted(
            (
                (name, int(_mapping(record, label=name)[metric]))
                for name, record in summary.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        rankings[metric] = [
            {"rank": index, "span": name, "value_ns": value}
            for index, (name, value) in enumerate(ordered, start=1)
        ]
    return rankings


def run_attempt_benchmark(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    suite: BenchmarkSuite = "artifacts",
    cold_runs: int = 3,
    warm_runs: int = 5,
    ray_source_limit: int | None = 3,
) -> Path:
    """Run cold-process and warm-process trials and write one bounded report."""

    source = Path(input_root).resolve()
    output = Path(output_dir).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Benchmark output must be outside the immutable experiment tree")
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if suite not in {"artifacts", "ray-replay"}:
        raise ValueError(f"Unsupported benchmark suite: {suite}")
    for name, value in (("cold_runs", cold_runs), ("warm_runs", warm_runs)):
        if isinstance(value, bool) or not 1 <= value <= 100:
            raise ValueError(f"{name} must lie in [1, 100]")
    if ray_source_limit is not None and not 1 <= ray_source_limit <= 3:
        raise ValueError("ray_source_limit must be null or lie in [1, 3]")
    assets = discover_attempt_assets(source)

    arguments = (str(source), suite, ray_source_limit)
    cold_trials: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    for _ in range(cold_runs):
        # Recreate the pool for every trial so Python/module/object caches start
        # empty.  The kernel page cache remains intentionally uncontrolled.
        with context.Pool(processes=1) as pool:
            cold_trials.append(pool.apply(_cold_trial_worker, (arguments,)))
    warm_trials = [_one_trial(*arguments) for _ in range(warm_runs)]

    output.mkdir(parents=True)
    cold_summary = summarize_trials(cold_trials)
    warm_summary = summarize_trials(warm_trials)
    cold_resource_summary = summarize_resource_trials(cold_trials)
    warm_resource_summary = summarize_resource_trials(warm_trials)
    report = {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.phase0_offline_performance_benchmark",
        "authority": "diagnostic_only_not_safety_or_science_authority",
        "verification_scope": {
            "stereo": "production stereo artifact reader",
            "occupancy": (
                "relocation-safe local checksum/dtype/shape/snapshot checks and typed "
                "per-frame construction; excludes foreign source paths and complete "
                "production evidence/hash-chain validation"
            ),
            "coarse_stationarity_generation": (
                "content parse/hash only; not production semantic readback"
            ),
            "ray_replay": (
                "unchanged DDA replay for selected latest-map sources compared with "
                "stored snapshots; not full production-chain authority"
                if suite == "ray-replay"
                else "not run in artifacts suite"
            ),
        },
        "input": {
            "root": str(source),
            "cycle_count": len(assets.cycle_roots),
            "generation_count": len(assets.generation_roots),
            "artifact_size_bytes": _artifact_tree_size_bytes(source),
            "write_policy": "read_only_append_only_experiment",
        },
        "output": str(output),
        "suite": suite,
        "cold_definition": (
            "fresh_spawned_python_process; kernel_page_cache_not_evicted_or_controlled"
        ),
        "warm_definition": "repeated_trials_in_one_python_process",
        "requested_runs": {"cold": cold_runs, "warm": warm_runs},
        "cold_trials": cold_trials,
        "warm_trials": warm_trials,
        "cold_summary": cold_summary,
        "warm_summary": warm_summary,
        "resource_summary": {
            "cold": cold_resource_summary,
            "warm": warm_resource_summary,
            "warm_peak_rss_note": (
                "process-lifetime high-water mark; warm trials share one process"
            ),
            "gpu_memory": "not_applicable_no_gpu_workload",
        },
        "ranked_hotspots": {
            "cold": rank_hotspots(cold_summary),
            "warm": rank_hotspots(warm_summary),
        },
        "ray_source_limit": ray_source_limit if suite == "ray-replay" else None,
        "environment": {
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
        },
    }
    path = output / "phase0_benchmark.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--suite", choices=("artifacts", "ray-replay"), default="artifacts")
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument(
        "--ray-source-limit",
        type=int,
        default=3,
        help="Replay the first N stored sources from the latest map (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    path = run_attempt_benchmark(
        arguments.input_root,
        arguments.output_dir,
        suite=arguments.suite,
        cold_runs=arguments.cold_runs,
        warm_runs=arguments.warm_runs,
        ray_source_limit=arguments.ray_source_limit,
    )
    print(path)


if __name__ == "__main__":
    main()
