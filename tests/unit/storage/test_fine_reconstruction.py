from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.settings import (
    FineFinalizationConfig,
    MultiViewFusionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
)
from biblade_fusion.perception.fusion import FusedBladeCloud, RegisteredCloudView
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.perception.tsdf import (
    BilateralTSDFResult,
    SparseTSDFVolume,
    TriangleMesh,
)
from biblade_fusion.planning.surface_coverage import (
    SurfacePatchQuality,
    SurfaceQualityReport,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.storage import fine_reconstruction as module
from biblade_fusion.storage.science_acceptance import ScienceTestEnvelope
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.workflows.fine_completion import finalize_fine_science
from biblade_fusion.workflows.fine_finalization import (
    FinalFineReconstruction,
    FineFinalizationGateReport,
)


def _result(tmp_path, monkeypatch):
    coverage_root = (tmp_path / "coverage").resolve()
    reference_root = (tmp_path / "reference").resolve()
    source_roots = tuple((tmp_path / name).resolve() for name in ("front", "back"))
    for root, metadata_name in (
        (coverage_root, "coverage.json"),
        (reference_root, "metadata.json"),
        (source_roots[0], "metadata.json"),
        (source_roots[1], "metadata.json"),
    ):
        root.mkdir()
        (root / metadata_name).write_text(f"{root.name}\n", encoding="utf-8")
    quality_config = SurfaceQualityConfig(minimum_observed_points=3)
    coverage = SimpleNamespace(
        root=coverage_root,
        generation_id="a" * 64,
        metadata_sha256="b" * 64,
        reference=SimpleNamespace(root=reference_root),
        ledger=SimpleNamespace(observation_ids=("front", "back")),
        quality_config=quality_config,
    )
    points = np.array(
        [
            [0.0, 0.0, 0.01],
            [0.01, 0.0, 0.01],
            [0.0, 0.01, 0.01],
            [0.0, 0.0, -0.01],
            [0.01, 0.0, -0.01],
            [0.0, 0.01, -0.01],
        ]
    )
    fused = FusedBladeCloud(
        points,
        np.vstack(
            (
                np.tile([0.0, 0.0, 1.0], (3, 1)),
                np.tile([0.0, 0.0, -1.0], (3, 1)),
            )
        ),
        np.array([1, 1, 1, -1, -1, -1], dtype=np.int8),
        np.zeros(3),
        np.eye(3),
        0.02,
        (),
    )
    tetra = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    )
    vertices = np.vstack((tetra, tetra + [0.03, 0.0, 0.0]))
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
            [4, 5, 6],
            [4, 7, 5],
            [5, 7, 6],
            [6, 7, 4],
        ],
        dtype=np.int32,
    )
    mesh = TriangleMesh(
        vertices,
        triangles,
        np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8),
    )
    indices = np.array([[0, 0, 0]], dtype=np.int32)
    front_volume = SparseTSDFVolume(
        1, np.zeros(3), 0.001, 0.002, indices, np.zeros(1), np.ones(1)
    )
    back_volume = SparseTSDFVolume(
        -1, np.zeros(3), 0.001, 0.002, indices, np.zeros(1), np.ones(1)
    )
    tsdf = BilateralTSDFResult(front_volume, back_volume, mesh, 0.002)
    patches = tuple(
        SurfacePatchQuality(
            side.value,
            side,
            SurfaceRegion.SURFACE,
            3,
            3,
            1.0,
            0.0,
            1.0,
            0.0,
            True,
            (),
        )
        for side in (BladeSide.FRONT, BladeSide.BACK)
    )
    quality = SurfaceQualityReport(patches, 1.0, {}, 8, 0, 0, True)
    gates = FineFinalizationGateReport(2, 2, 1, 1, 4, 4, 1, 1, 0, 0, True, ())
    registered = (
        RegisteredCloudView("front", points[:3], np.array([0.0, 0.0, 0.3])),
        RegisteredCloudView("back", points[3:], np.array([0.0, 0.0, -0.3])),
    )
    result = FinalFineReconstruction(
        coverage, source_roots, registered, fused, tsdf, quality, gates
    )

    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        lambda *_args, **_kwargs: coverage,
    )
    by_id = {item.view_id: item for item in registered}

    def read_view(root):
        view_id = root.name
        return SimpleNamespace(view=SimpleNamespace(source_view_id=view_id))

    monkeypatch.setattr(module, "read_reconstructed_view", read_view)
    monkeypatch.setattr(
        module,
        "registered_cloud_view",
        lambda view: by_id[view.source_view_id],
    )
    return result, quality_config


def _write(tmp_path, monkeypatch):
    result, quality_config = _result(tmp_path, monkeypatch)
    output = tmp_path / "final"
    module.write_unaccepted_legacy_fine_reconstruction(
        output,
        result,
        fusion_config=MultiViewFusionConfig(),
        tsdf_config=TSDFConfig(),
        surface_quality_config=quality_config,
        finalization_config=FineFinalizationConfig(),
    )
    return output, result


def test_production_terminal_writer_requires_science_authority(
    tmp_path,
    monkeypatch,
) -> None:
    result, quality_config = _result(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires a science acceptance authority"):
        module.write_final_fine_reconstruction(
            tmp_path / "must-not-exist",
            result,
            fusion_config=MultiViewFusionConfig(),
            tsdf_config=TSDFConfig(),
            surface_quality_config=quality_config,
            finalization_config=FineFinalizationConfig(),
            science_authority=None,  # type: ignore[arg-type]
        )

    assert not (tmp_path / "must-not-exist").exists()


def test_production_fine_completion_requires_science_authority() -> None:
    with pytest.raises(ValueError, match="requires a science acceptance authority"):
        finalize_fine_science(
            None,  # type: ignore[arg-type]
            fusion_config=MultiViewFusionConfig(),
            tsdf_config=TSDFConfig(),
            surface_quality_config=SurfaceQualityConfig(),
            finalization_config=FineFinalizationConfig(),
            science_authority=None,  # type: ignore[arg-type]
        )


def _science_authority(tmp_path, monkeypatch) -> ScienceAcceptanceAuthority:
    acceptance = (tmp_path / "science-acceptance").resolve()
    acceptance.mkdir()
    monkeypatch.setattr(
        ScienceAcceptanceAuthority,
        "assert_acceptance_asset_current",
        lambda _self: None,
    )
    return ScienceAcceptanceAuthority(
        acceptance_path=acceptance,
        acceptance_id="1" * 64,
        acceptance_metadata_sha256="2" * 64,
        runtime_contract_sha256="3" * 64,
        test_envelope=ScienceTestEnvelope(0.15, 1.5, 0.0, 85.0),
        inference_identity={
            "foundation_stereo_source_sha256": "4" * 64,
            "foundation_stereo_checkpoint_sha256": "5" * 64,
            "foundation_stereo_model_config_sha256": "6" * 64,
            "stereo_calibration_sha256": "7" * 64,
            "flange_primary_hand_eye_sha256": "8" * 64,
        },
    )


def test_terminal_asset_strictly_reads_and_replays(tmp_path, monkeypatch) -> None:
    output, result = _write(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "build_final_fine_reconstruction",
        lambda *_args, **_kwargs: result,
    )

    stored = module.replay_final_fine_reconstruction(output)

    assert stored.result.coverage.generation_id == "a" * 64
    assert stored.result.gates.passed
    assert len(stored.result.tsdf.mesh.triangles) == 8


def test_terminal_asset_rejects_array_tampering(tmp_path, monkeypatch) -> None:
    output, _ = _write(tmp_path, monkeypatch)
    path = output / "mesh_triangles.npy"
    array = np.load(path, allow_pickle=False)
    array[0] = array[0, ::-1]
    np.save(path, array, allow_pickle=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        module.read_final_fine_reconstruction(output)


def test_terminal_asset_rejects_source_tampering(tmp_path, monkeypatch) -> None:
    output, _ = _write(tmp_path, monkeypatch)
    (tmp_path / "front" / "metadata.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable source changed"):
        module.read_final_fine_reconstruction(output)


def test_terminal_asset_schema2_binds_and_replays_exact_science_authority(
    tmp_path,
    monkeypatch,
) -> None:
    result, quality_config = _result(tmp_path, monkeypatch)
    authority = _science_authority(tmp_path, monkeypatch)
    output = tmp_path / "final"
    module.write_final_fine_reconstruction(
        output,
        result,
        fusion_config=MultiViewFusionConfig(),
        tsdf_config=TSDFConfig(),
        surface_quality_config=quality_config,
        finalization_config=FineFinalizationConfig(),
        science_authority=authority,
    )
    monkeypatch.setattr(
        module,
        "build_final_fine_reconstruction",
        lambda *_args, **_kwargs: result,
    )

    stored = module.replay_final_fine_reconstruction(
        output,
        expected_science_authority=authority,
    )

    assert stored.science_authority == authority
    changed = replace(authority, runtime_contract_sha256="9" * 64)
    with pytest.raises(ValueError, match="science authority changed"):
        module.replay_final_fine_reconstruction(
            output,
            expected_science_authority=changed,
        )

    metadata = output / "final_reconstruction.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["science_acceptance_authority"]["runtime_contract_sha256"] = "9" * 64
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        module.read_final_fine_reconstruction(output)


def test_terminal_replay_rejects_fusion_semantic_drift(tmp_path, monkeypatch) -> None:
    output, result = _write(tmp_path, monkeypatch)
    changed = replace(
        result,
        fused_cloud=replace(
            result.fused_cloud,
            center_m=result.fused_cloud.center_m + np.array([0.001, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_final_fine_reconstruction",
        lambda *_args, **_kwargs: changed,
    )

    with pytest.raises(ValueError, match="changed fusion semantics"):
        module.replay_final_fine_reconstruction(output)


def test_terminal_replay_rejects_tsdf_semantic_drift(tmp_path, monkeypatch) -> None:
    output, result = _write(tmp_path, monkeypatch)
    changed = replace(
        result,
        tsdf=replace(result.tsdf, backend="different_replay_backend"),
    )
    monkeypatch.setattr(
        module,
        "build_final_fine_reconstruction",
        lambda *_args, **_kwargs: changed,
    )

    with pytest.raises(ValueError, match="changed TSDF semantics"):
        module.replay_final_fine_reconstruction(output)
