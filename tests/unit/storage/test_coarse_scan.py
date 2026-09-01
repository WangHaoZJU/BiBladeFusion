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
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import ProxyModelConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapSeed,
    bootstrap_blade_foreground,
)
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.planning import BladeSide, coverage_observation_id
from biblade_fusion.storage.coarse_scan import (
    read_coarse_scan_generation,
    read_coarse_scan_view,
    write_coarse_scan_generation,
    write_coarse_scan_view,
)
from biblade_fusion.storage.reconstructed_view import StoredReconstructedBladeView
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash
from biblade_fusion.workflows.reconstruction import ReconstructedBladeView


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


def test_coarse_view_persists_and_replays_per_view_proxy_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (6, 6)
    left = np.arange(36, dtype=np.float64).reshape(shape)
    depth = np.full(shape, 0.5, dtype=np.float64)
    integration_valid = np.ones(shape, dtype=np.bool_)
    foreground = bootstrap_blade_foreground(
        left,
        depth,
        integration_valid,
        BootstrapForegroundConfig(
            boundary_margin_px=1,
            minimum_valid_pixels=1,
            minimum_component_pixels=1,
            minimum_mask_pixels=1,
            minimum_mask_fraction=0.0,
            maximum_mask_fraction=1.0,
            minimum_seed_valid_pixels=1,
            minimum_seed_valid_fraction=0.0,
        ),
        BootstrapSeed.rectangle(1, 1, 4, 4, mode="hard_roi"),
    )
    pixels = np.argwhere(foreground.mask)[:, ::-1]
    points = np.column_stack(
        (
            np.where(np.arange(len(pixels)) < len(pixels) // 2, 0.5, 1.5),
            pixels[:, 0] * 0.01,
            np.full(len(pixels), 0.2),
        )
    )
    cloud = PointCloud("base", points, pixels, shape)
    view = ReconstructedBladeView(
        "coarse_00",
        2,
        17,
        CameraIntrinsics(6, 6, 100.0, 100.0, 2.5, 2.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "left_rectified"),
        cloud,
        "foundation_stereo",
    )
    reconstructed_root = (tmp_path / "reconstructed").resolve()
    stereo_root = (tmp_path / "stereo").resolve()
    occupancy_root = (tmp_path / "occupancy").resolve()
    for root in (reconstructed_root, stereo_root, occupancy_root):
        root.mkdir()
        (root / "metadata.json").write_text("{}\n", encoding="utf-8")
    reconstructed = StoredReconstructedBladeView(
        view,
        foreground.mask,
        {
            "source": {
                "session": str((tmp_path / "session").resolve()),
                "stereo_inference": str(stereo_root),
                "view_id": view.source_view_id,
            }
        },
    )
    evidence = SimpleNamespace(
        source_view_id=view.source_view_id,
        source_sequence_index=view.source_sequence_index,
        frame_number=view.source_frame_number,
        integration_valid_mask_content_hash=occupancy_array_content_hash(
            integration_valid
        ),
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "read_reconstructed_view",
        lambda _path: reconstructed,
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "_replay_foreground",
        lambda **_kwargs: foreground,
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "read_occupancy_mapping",
        lambda _path: SimpleNamespace(frame_evidence=(evidence,)),
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "_load_final_integration_mask",
        lambda _path: integration_valid,
    )
    proxy_config = ProxyModelConfig(
        estimated_thickness_m=0.01,
        minimum_points=6,
        blade_envelope_min_m=(0.0, 0.0, 0.0),
        blade_envelope_max_m=(1.0, 1.0, 1.0),
        minimum_envelope_retained_fraction=0.4,
    )

    output = write_coarse_scan_view(
        tmp_path / "coarse-view",
        foreground,
        reconstructed_view=reconstructed_root,
        source_stereo_inference=stereo_root,
        source_occupancy_mapping=occupancy_root,
        target_view_id="front:r0:c0",
        target_kind="proxy_normal",
        target_side=BladeSide.FRONT,
        proxy_config=proxy_config,
    )
    stored = read_coarse_scan_view(output)

    assert stored.proxy_support.retained_point_count == len(points) // 2
    assert stored.support_cloud.points_m.shape == (len(points) // 2, 3)
    assert np.all(stored.support_cloud.points_m[:, 0] == 0.5)
    assert stored.metadata["proxy_support"]["configuration"] == (
        proxy_config.model_dump(mode="json")
    )


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
        proxy_config=ProxyModelConfig(),
        support_cloud=object(),
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
        "initialization": (tmp_path / "initialization", "metadata.json"),
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
        lambda _path: SimpleNamespace(
            observation=SimpleNamespace(proxy=object()),
            metadata={
                "processing": {"proxy_model": ProxyModelConfig().model_dump(mode="json")}
            },
        ),
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
