from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.planning.coverage as coverage_module
import biblade_fusion.storage.coarse_scan as coarse_scan_module
import biblade_fusion.storage.initialization as initialization_module
import biblade_fusion.storage.view_plan as view_plan_module
from biblade_fusion.planning import BladeSide, coverage_observation_id
from biblade_fusion.storage.coarse_scan import (
    read_coarse_scan_generation,
    write_coarse_scan_generation,
)


def test_coarse_generation_rejects_any_motion_authorization(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    root.mkdir()
    (root / "generation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.coarse_scan_generation",
                "motion_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="motion-authorized"):
        read_coarse_scan_generation(root)


def _fake_stored_view(tmp_path: Path) -> SimpleNamespace:
    session = (tmp_path / "session").resolve()
    view = SimpleNamespace(
        source_view_id="coarse_00",
        source_sequence_index=2,
        source_frame_number=17,
        base_cloud=object(),
        base_t_projection_camera=object(),
    )
    return SimpleNamespace(
        root=(tmp_path / "coarse-view").resolve(),
        reconstructed=SimpleNamespace(
            view=view,
            metadata={"source": {"session": str(session)}},
        ),
        target_side=BladeSide.FRONT,
        metadata={"sources": {"reconstructed_view": {"root": str(tmp_path / "rv")}}},
    )


def _fake_coverage(
    tmp_path: Path,
    *,
    observation_ids: tuple[str, ...],
    bin_count: int = 1,
) -> SimpleNamespace:
    patch = SimpleNamespace(
        patch_id="front:r0:c0",
        side=BladeSide.FRONT,
        row=0,
        column=0,
        observation_ids=observation_ids,
        bin_point_counts=np.asarray([[bin_count]], dtype=np.int64),
    )
    return SimpleNamespace(
        metadata={
            "source_plan": str((tmp_path / "view-plan").resolve()),
            "source_initialization": str((tmp_path / "initialization").resolve()),
            "previous_ledger": None,
        },
        ledger=SimpleNamespace(
            rows=1,
            columns=1,
            config=object(),
            observation_ids=observation_ids,
            completed_patch_ids=("front:r0:c0",),
            patches=(patch,),
        ),
    )


def test_generation_writer_rejects_same_count_wrong_physical_observation_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_view = _fake_stored_view(tmp_path)
    coverage = _fake_coverage(tmp_path, observation_ids=("same-count-but-wrong",))
    monkeypatch.setattr(coarse_scan_module, "read_coarse_scan_view", lambda _path: stored_view)
    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: coverage)

    with pytest.raises(ValueError, match="physical observation identities"):
        write_coarse_scan_generation(
            tmp_path / "generation",
            views=(stored_view.root,),
            coverage=tmp_path / "coverage",
            source_initialization=tmp_path / "initialization",
            source_view_plan=tmp_path / "view-plan",
            source_discovery_plan=tmp_path / "discovery",
        )


def test_generation_reader_rejects_same_count_wrong_physical_observation_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_view = _fake_stored_view(tmp_path)
    coverage = _fake_coverage(tmp_path, observation_ids=("same-count-but-wrong",))
    authorities = {
        "initialization": (tmp_path / "initialization", "initialization.json"),
        "view_plan": (tmp_path / "view-plan", "view_plan.json"),
        "discovery_plan": (tmp_path / "discovery", "discovery.json"),
        "coverage": (tmp_path / "coverage", "coverage.json"),
        "view": (stored_view.root, "metadata.json"),
    }
    for root, filename in authorities.values():
        root.mkdir(parents=True, exist_ok=True)
        (root / filename).write_text("{}\n", encoding="utf-8")
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.coarse_scan_generation",
                "motion_authorized": False,
                "generation_index": 0,
                "previous_generation": None,
                "sources": {
                    "initialization": coarse_scan_module._directory_record(
                        *authorities["initialization"]
                    ),
                    "view_plan": coarse_scan_module._directory_record(*authorities["view_plan"]),
                    "discovery_plan": coarse_scan_module._directory_record(
                        *authorities["discovery_plan"]
                    ),
                    "coverage": coarse_scan_module._directory_record(*authorities["coverage"]),
                    "coarse_model": None,
                },
                "views": [coarse_scan_module._directory_record(*authorities["view"])],
                "summary": {
                    "view_count": 1,
                    "front_view_count": 1,
                    "back_view_count": 0,
                    "schema5_ready": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(coarse_scan_module, "read_coarse_scan_view", lambda _path: stored_view)
    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: coverage)

    with pytest.raises(ValueError, match="physical observation identities"):
        read_coarse_scan_generation(generation)


def test_generation_writer_rejects_coverage_bins_that_do_not_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_view = _fake_stored_view(tmp_path)
    source = stored_view.reconstructed.metadata["source"]
    view = stored_view.reconstructed.view
    observation_id = coverage_observation_id(
        source["session"],
        view.source_view_id,
        view.source_sequence_index,
        view.source_frame_number,
    )
    stored = _fake_coverage(tmp_path, observation_ids=(observation_id,), bin_count=9)
    replayed = _fake_coverage(tmp_path, observation_ids=(observation_id,), bin_count=1).ledger
    replayed.config = stored.ledger.config
    monkeypatch.setattr(coarse_scan_module, "read_coarse_scan_view", lambda _path: stored_view)
    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: stored)
    monkeypatch.setattr(
        initialization_module,
        "read_initialization",
        lambda _path: SimpleNamespace(observation=SimpleNamespace(proxy=object())),
    )
    monkeypatch.setattr(
        view_plan_module,
        "read_view_plan",
        lambda _path: SimpleNamespace(result=SimpleNamespace(geometric_plan=object())),
    )
    monkeypatch.setattr(coverage_module, "create_coverage_ledger", lambda *_args: object())
    monkeypatch.setattr(coverage_module, "update_coverage", lambda *_args: replayed)

    with pytest.raises(ValueError, match="patch differs from deterministic replay"):
        write_coarse_scan_generation(
            tmp_path / "generation",
            views=(stored_view.root,),
            coverage=tmp_path / "coverage",
            source_initialization=tmp_path / "initialization",
            source_view_plan=tmp_path / "view-plan",
            source_discovery_plan=tmp_path / "discovery",
        )
