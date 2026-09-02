from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.diagnostics.performance_timing import performance_span
from biblade_fusion.perception.bootstrap_foreground import BootstrapForegroundConfig
from biblade_fusion.planning import BladeSide
from scripts import benchmark_attempt11_coarse_view_write as benchmark


@dataclass(frozen=True)
class _Diagnostics:
    mask_pixel_count: int
    seed_pixel_count: int


def _nested_process_pool_probe(value: int) -> int:
    context = benchmark.multiprocessing.get_context("spawn")
    with benchmark.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        return executor.submit(abs, value).result(timeout=10)


def _stored_view(*, created_at: str, target_view_id: str = "front:r0:c0") -> SimpleNamespace:
    mask = np.array([[True, False], [True, False]], dtype=np.bool_)
    seed_mask = np.array([[True, False], [False, False]], dtype=np.bool_)
    points = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.3, 0.2, 0.3], [0.4, 0.2, 0.3]],
        dtype=np.float64,
    )
    pixel_uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    full_cloud = SimpleNamespace(
        frame="base",
        points_m=points,
        pixel_uv=pixel_uv,
        source_image_shape=(2, 2),
    )
    support_mask = np.array([True, False, True, False], dtype=np.bool_)
    support_cloud = SimpleNamespace(
        frame="base",
        points_m=points[support_mask],
        pixel_uv=pixel_uv[support_mask],
        source_image_shape=(2, 2),
    )
    foreground = SimpleNamespace(
        algorithm="bootstrap_blade_foreground_v2",
        config=BootstrapForegroundConfig(),
        seed=None,
        diagnostics=_Diagnostics(mask_pixel_count=2, seed_pixel_count=1),
        policy_sha256="1" * 64,
        left_image_content_sha256="2" * 64,
        depth_content_sha256="3" * 64,
        valid_mask_content_sha256="4" * 64,
        mask=mask,
        seed_mask=seed_mask,
    )
    proxy_support = SimpleNamespace(
        mask=support_mask,
        metadata_payload=lambda: {
            "algorithm": "base_frame_blade_envelope_aabb_v1",
            "retained_point_count": 2,
        },
    )
    metadata = {
        "schema_version": 2,
        "artifact_kind": "biblade_fusion.coarse_scan_view",
        "created_at_utc": created_at,
        "motion_authorized": False,
        "target": {"view_id": target_view_id, "kind": "proxy_normal", "side": "front"},
        "identity": {"view_id": "source-0", "sequence_index": 0, "frame_number": 7},
        "files": {
            "mask": {"path": "mask.npy", "sha256": "5" * 64},
            "seed_mask": {"path": "seed_mask.npy", "sha256": "6" * 64},
            "proxy_support_mask": {
                "path": "proxy_support_mask.npy",
                "sha256": "7" * 64,
            },
        },
        "sources": {
            "reconstructed_view": {"root": "/source/reconstructed"},
            "stereo_inference": {"root": "/source/stereo"},
            "occupancy_mapping": {"root": "/source/occupancy"},
        },
    }
    return SimpleNamespace(
        metadata=metadata,
        target_view_id=target_view_id,
        target_kind="proxy_normal",
        target_side=BladeSide.FRONT,
        foreground=foreground,
        reconstructed=SimpleNamespace(
            blade_mask=mask,
            view=SimpleNamespace(
                source_view_id="source-0",
                source_sequence_index=0,
                source_frame_number=7,
                base_cloud=full_cloud,
            ),
        ),
        proxy_config=ProxyModelConfig(),
        proxy_support=proxy_support,
        support_cloud=support_cloud,
    )


def _fixture(cycle_attempt: Path, stored: SimpleNamespace) -> benchmark._WriterFixture:
    for name in ("coarse_reconstructed_view", "stereo_inference", "occupancy_mapping"):
        (cycle_attempt / name).mkdir(parents=True)
    payload, digest = benchmark._semantic_digest(stored)
    return benchmark._WriterFixture(
        cycle_attempt_root=cycle_attempt,
        reconstructed_root=cycle_attempt / "coarse_reconstructed_view",
        stereo_root=cycle_attempt / "stereo_inference",
        occupancy_root=cycle_attempt / "occupancy_mapping",
        oracle=stored,
        normalized_semantic_payload=payload,
        normalized_semantic_sha256=digest,
    )


def test_semantic_digest_ignores_only_creation_time() -> None:
    first = _stored_view(created_at="2026-09-02T00:00:00+00:00")
    second = _stored_view(created_at="2026-09-02T00:01:00+00:00")

    first_payload, first_digest = benchmark._semantic_digest(first)
    second_payload, second_digest = benchmark._semantic_digest(second)

    assert first_payload == second_payload
    assert first_digest == second_digest
    changed = _stored_view(
        created_at="2026-09-02T00:01:00+00:00",
        target_view_id="front:r0:c1",
    )
    assert benchmark._semantic_digest(changed)[1] != first_digest


def test_one_trial_times_only_writer_and_requires_strict_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "immutable.txt").write_text("authority\n", encoding="utf-8")
    cycle_attempt = input_root / "cycle-attempt"
    cycle_attempt.mkdir()
    stored = _stored_view(created_at="2026-09-02T00:00:00+00:00")
    fixture = _fixture(cycle_attempt, stored)
    before = benchmark._tree_fingerprint(input_root)
    calls: list[str] = []

    def write(destination: Path, _foreground: object, **_kwargs: object) -> Path:
        calls.append("write")
        with performance_span("occupancy.depth_ray_integrator"):
            destination.mkdir()
            (destination / "metadata.json").write_text("{}\n", encoding="utf-8")
        return destination.resolve()

    def read(_path: Path) -> SimpleNamespace:
        calls.append("read")
        with performance_span("test.post_readback"):
            return stored

    monkeypatch.setattr(benchmark, "write_coarse_scan_view", write)
    monkeypatch.setattr(benchmark, "read_coarse_scan_view", read)
    result = benchmark._run_one_trial(
        cycle_attempt,
        tmp_path / "output" / "warm_00",
        expected_dda_count=1,
        fixture=fixture,
    )

    assert calls == ["write", "read"]
    assert result["dda_call_count"] == 1
    assert benchmark._TARGET_SPAN in result["spans"]
    assert "test.post_readback" not in result["spans"]
    assert result["normalized_semantic_sha256"] == fixture.normalized_semantic_sha256
    assert result["output_tree"]["file_count"] == 1
    assert benchmark._tree_fingerprint(input_root) == before


def test_one_trial_rejects_post_readback_semantic_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle_attempt = tmp_path / "input" / "cycle-attempt"
    cycle_attempt.mkdir(parents=True)
    oracle = _stored_view(created_at="2026-09-02T00:00:00+00:00")
    fixture = _fixture(cycle_attempt, oracle)

    def write(destination: Path, _foreground: object, **_kwargs: object) -> Path:
        with performance_span("occupancy.depth_ray_integrator"):
            destination.mkdir()
            (destination / "metadata.json").write_text("{}\n", encoding="utf-8")
        return destination.resolve()

    monkeypatch.setattr(benchmark, "write_coarse_scan_view", write)
    monkeypatch.setattr(
        benchmark,
        "read_coarse_scan_view",
        lambda _path: _stored_view(
            created_at="2026-09-02T00:01:00+00:00",
            target_view_id="changed",
        ),
    )

    with pytest.raises(
        benchmark.CoarseViewWriteBenchmarkError,
        match="semantic oracle",
    ):
        benchmark._run_one_trial(
            cycle_attempt,
            tmp_path / "output" / "cold_00",
            expected_dda_count=1,
            fixture=fixture,
        )


def test_one_trial_requires_exact_dda_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle_attempt = tmp_path / "input" / "cycle-attempt"
    cycle_attempt.mkdir(parents=True)
    oracle = _stored_view(created_at="2026-09-02T00:00:00+00:00")
    fixture = _fixture(cycle_attempt, oracle)

    def write(destination: Path, _foreground: object, **_kwargs: object) -> Path:
        with performance_span("occupancy.depth_ray_integrator"):
            destination.mkdir()
            (destination / "metadata.json").write_text("{}\n", encoding="utf-8")
        return destination.resolve()

    monkeypatch.setattr(benchmark, "write_coarse_scan_view", write)

    with pytest.raises(
        benchmark.CoarseViewWriteBenchmarkError,
        match=r"observed=1, expected=2",
    ):
        benchmark._run_one_trial(
            cycle_attempt,
            tmp_path / "output" / "warm_00",
            expected_dda_count=2,
            fixture=fixture,
        )


def test_benchmark_paths_forbid_input_descendant_and_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "attempt-11"
    source.mkdir()

    with pytest.raises(ValueError, match="outside"):
        benchmark._validate_benchmark_paths(source, source / "benchmark")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark._validate_benchmark_paths(source, existing)


def test_cold_worker_rejects_daemonic_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(daemon=True),
    )

    with pytest.raises(
        benchmark.CoarseViewWriteBenchmarkError,
        match="non-daemonic",
    ):
        benchmark._cold_worker(("/unused/input", "/unused/output", 1))


def test_spawned_cold_worker_can_create_nested_process_pool() -> None:
    context = benchmark.multiprocessing.get_context("spawn")
    with benchmark.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        assert executor.submit(_nested_process_pool_probe, -7).result(timeout=20) == 7


def test_discovery_uses_local_committed_attempt_identity(tmp_path: Path) -> None:
    source = tmp_path / "attempt-11"
    cycle = source / "perception/coarse/cycles/000000_operator_bootstrap_000"
    accepted = cycle / "attempt_local"
    accepted.mkdir(parents=True)
    (cycle / "committed.json").write_text(
        json.dumps(
            {
                "accepted_attempt": {
                    "attempt_id": "attempt_local",
                    "root": "/foreign/host/path/that/must/not/be/trusted",
                }
            }
        ),
        encoding="utf-8",
    )

    assert benchmark._discover_cycle_attempt_root(source) == accepted.resolve()
    assert benchmark.DEFAULT_COLD_RUNS == 3
    assert benchmark.DEFAULT_WARM_RUNS == 5


def test_runtime_provenance_binds_host_revision_and_code_files() -> None:
    provenance = benchmark._runtime_provenance()

    assert provenance["hostname"]
    assert len(provenance["git_head"]) == 40
    assert provenance["production_module_git_path"] == (
        "src/biblade_fusion/storage/coarse_scan.py"
    )
    authorities = provenance["code_authorities"]
    for authority in authorities.values():
        path = Path(authority["path"])
        assert authority["size_bytes"] == path.stat().st_size
        assert authority["sha256"] == benchmark._sha256(path)
