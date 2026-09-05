from __future__ import annotations

import json
from dataclasses import asdict, replace
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
from biblade_fusion.perception.coarse_foreground import (
    PROJECTED_COARSE_FOREGROUND_ALGORITHM,
    ProjectedCoarseForegroundDiagnostics,
    ProjectedCoarseForegroundGuide,
    ProjectedCoarseForegroundResult,
    projected_coarse_foreground_policy_sha256,
)
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.planning import BladeSide, coverage_observation_id
from biblade_fusion.storage.coarse_scan import (
    StoredCoarseScanView,
    read_coarse_scan_generation,
    read_coarse_scan_view,
    write_coarse_scan_generation,
    write_coarse_scan_view,
)
from biblade_fusion.storage.reconstructed_view import StoredReconstructedBladeView
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash
from biblade_fusion.workflows.reconstruction import ReconstructedBladeView


def _write_fake_verified_occupancy(
    root: Path,
    mask: np.ndarray,
    evidence: SimpleNamespace,
) -> SimpleNamespace:
    source_depth_path = root / "0000_source_depth_m.npy"
    np.save(source_depth_path, np.full(mask.shape, 0.5), allow_pickle=False)
    mask_path = root / "0000_integration_valid_mask.npy"
    np.save(mask_path, mask, allow_pickle=False)
    mapping_snapshot_path = root / "0000_mapping_snapshot.json"
    mapping_snapshot_path.write_text('{"snapshot": "mapping"}\n', encoding="utf-8")
    result_snapshot_path = root / "0000_result_snapshot.json"
    result_snapshot_path.write_text('{"snapshot": "result"}\n', encoding="utf-8")
    final_snapshot_path = root / "occupancy.json"
    final_snapshot_path.write_text('{"snapshot": "final"}\n', encoding="utf-8")
    stereo_root = root.parent / "stereo"
    stereo_metadata_path = stereo_root / "metadata.json"
    stereo_metadata_path.write_text('{"schema_version": 2}\n', encoding="utf-8")
    session_root = root.parent / "session"
    session_root.mkdir(exist_ok=True)
    session_manifest_path = session_root / "manifest.json"
    session_manifest_path.write_text('{"schema_version": 3}\n', encoding="utf-8")
    hand_eye_path = root.parent / "hand_eye.yaml"
    hand_eye_path.write_text("schema_version: 2\n", encoding="utf-8")

    def array_record(path: Path, value: np.ndarray) -> dict[str, object]:
        return {
            "path": path.name,
            "sha256": coarse_scan_module._sha256(path),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    def snapshot_record(path: Path) -> dict[str, object]:
        return {"path": path.name, "sha256": coarse_scan_module._sha256(path)}

    metadata = {
        "schema_version": 7,
        "artifact_kind": "biblade_fusion.occupancy_mapping",
        "snapshot": snapshot_record(final_snapshot_path),
        "sources": {
            "hand_eye": {
                "root": str(hand_eye_path.parent.resolve()),
                "file": hand_eye_path.name,
                "sha256": coarse_scan_module._sha256(hand_eye_path),
            }
        },
        "frames": [
            {
                "files": {
                    "source_depth_m": array_record(
                        source_depth_path,
                        np.full(mask.shape, 0.5),
                    ),
                    "integration_valid_mask": array_record(mask_path, mask),
                },
                "mapping_snapshot": snapshot_record(mapping_snapshot_path),
                "result_snapshot": snapshot_record(result_snapshot_path),
                "sources": {
                    "stereo_inference": {
                        "root": str(stereo_root.resolve()),
                        "file": "metadata.json",
                        "sha256": coarse_scan_module._sha256(stereo_metadata_path),
                    },
                    "session": {
                        "root": str(session_root.resolve()),
                        "file": "manifest.json",
                        "sha256": coarse_scan_module._sha256(session_manifest_path),
                    },
                },
            }
        ],
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    stored = SimpleNamespace(
        motion_eligible=True,
        verification_status="full_semantic_verified_for_motion_preflight",
        frame_evidence=(evidence,),
        semantic_attestation=SimpleNamespace(
            occupancy_metadata_sha256=coarse_scan_module._sha256(metadata_path),
            attestation_hash="a" * 64,
        ),
    )
    authority_paths = {
        "metadata": metadata_path,
        "mask": mask_path,
        "nonfinal_array": source_depth_path,
        "snapshot": mapping_snapshot_path,
        "stereo_metadata": stereo_metadata_path,
        "hand_eye": hand_eye_path,
    }
    stored.authority_paths = authority_paths
    stored.authority_fingerprints = {
        path: (coarse_scan_module._sha256(path), path.stat().st_size)
        for path in authority_paths.values()
    }
    return stored


def _strict_fake_occupancy_reader(
    stored: SimpleNamespace,
    reads: list[Path],
):
    def read_occupancy(path: str | Path) -> SimpleNamespace:
        reads.append(Path(path).resolve())
        for authority, expected in stored.authority_fingerprints.items():
            actual = (coarse_scan_module._sha256(authority), authority.stat().st_size)
            if actual != expected:
                raise ValueError(f"Full occupancy authority changed: {authority}")
        return stored

    return read_occupancy


def _coarse_writer_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
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
            np.full(len(pixels), 0.5),
            pixels[:, 0] * 0.01,
            np.full(len(pixels), 0.2),
        )
    )
    view = ReconstructedBladeView(
        "coarse_00",
        2,
        17,
        CameraIntrinsics(6, 6, 100.0, 100.0, 2.5, 2.5, "none", ()),
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "left_rectified"),
        PointCloud("base", points, pixels, shape),
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
    stored_occupancy = _write_fake_verified_occupancy(
        occupancy_root,
        integration_valid,
        evidence,
    )
    occupancy_reads: list[Path] = []

    monkeypatch.setattr(coarse_scan_module, "read_reconstructed_view", lambda _path: reconstructed)
    monkeypatch.setattr(coarse_scan_module, "_replay_foreground", lambda **_kwargs: foreground)
    monkeypatch.setattr(
        coarse_scan_module,
        "read_occupancy_mapping",
        _strict_fake_occupancy_reader(stored_occupancy, occupancy_reads),
    )
    return SimpleNamespace(
        output=tmp_path / "coarse-view",
        foreground=foreground,
        reconstructed_root=reconstructed_root,
        stereo_root=stereo_root,
        occupancy_root=occupancy_root,
        integration_mask_path=occupancy_root / "0000_integration_valid_mask.npy",
        integration_valid=integration_valid,
        authority_paths=stored_occupancy.authority_paths,
        occupancy_reads=occupancy_reads,
        proxy_config=ProxyModelConfig(
            estimated_thickness_m=0.01,
            minimum_points=6,
            blade_envelope_min_m=(0.0, 0.0, 0.0),
            blade_envelope_max_m=(1.0, 1.0, 1.0),
            minimum_envelope_retained_fraction=0.4,
        ),
    )


def _write_coarse_writer_case(case: SimpleNamespace) -> Path:
    return write_coarse_scan_view(
        case.output,
        case.foreground,
        reconstructed_view=case.reconstructed_root,
        source_stereo_inference=case.stereo_root,
        source_occupancy_mapping=case.occupancy_root,
        target_view_id="front:r0:c0",
        target_kind="proxy_normal",
        target_side=BladeSide.FRONT,
        proxy_config=case.proxy_config,
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
    replay_sources: list[object | None] = []

    def replay_foreground(**kwargs: object):
        replay_sources.append(kwargs.get("verified_integration"))
        return foreground

    monkeypatch.setattr(coarse_scan_module, "_replay_foreground", replay_foreground)
    stored_occupancy = _write_fake_verified_occupancy(
        occupancy_root,
        integration_valid,
        evidence,
    )
    occupancy_reads: list[Path] = []

    monkeypatch.setattr(
        coarse_scan_module,
        "read_occupancy_mapping",
        _strict_fake_occupancy_reader(stored_occupancy, occupancy_reads),
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

    assert occupancy_reads == [occupancy_root, occupancy_root]
    assert replay_sources[0] is not None
    assert replay_sources[1] is None
    assert stored.proxy_support.retained_point_count == len(points) // 2
    assert stored.support_cloud.points_m.shape == (len(points) // 2, 3)
    assert np.all(stored.support_cloud.points_m[:, 0] == 0.5)
    assert stored.metadata["proxy_support"]["configuration"] == (
        proxy_config.model_dump(mode="json")
    )


def test_projected_coarse_view_binds_exact_predecessor_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _coarse_writer_case(tmp_path, monkeypatch)
    generation = tmp_path / "accepted-generation"
    generation.mkdir()
    generation_metadata = generation / "generation.json"
    generation_metadata.write_text(
        '{"artifact_kind":"biblade_fusion.coarse_scan_generation"}\n',
        encoding="utf-8",
    )
    mask = case.foreground.mask
    projection = case.foreground.seed_mask
    mask_count = int(np.count_nonzero(mask))
    projected_count = int(np.count_nonzero(projection))
    guide = ProjectedCoarseForegroundGuide(
        source_generation_path=generation,
        source_generation_metadata_sha256=coarse_scan_module._sha256(
            generation_metadata
        ),
        reference_points_content_sha256="b" * 64,
        blade_envelope_min_m=(0.0, 0.0, 0.0),
        blade_envelope_max_m=(1.0, 1.0, 1.0),
    )
    projected = ProjectedCoarseForegroundResult(
        mask=mask,
        projected_reference_mask=projection,
        diagnostics=ProjectedCoarseForegroundDiagnostics(
            image_pixel_count=mask.size,
            supplied_valid_pixel_count=mask.size,
            depth_valid_pixel_count=mask.size,
            reference_point_count=100,
            projected_reference_pixel_count=projected_count,
            eligible_projected_pixel_count=projected_count,
            predicted_depth_consistent_pixel_count=mask_count,
            base_envelope_pixel_count=mask_count,
            mask_pixel_count=mask_count,
            mask_fraction=mask_count / mask.size,
            projected_match_fraction=mask_count / projected_count,
            minimum_mask_depth_m=0.5,
            median_mask_depth_m=0.5,
            maximum_mask_depth_m=0.5,
        ),
        config=case.foreground.config,
        guide=guide,
        algorithm=PROJECTED_COARSE_FOREGROUND_ALGORITHM,
        policy_sha256=projected_coarse_foreground_policy_sha256(
            case.foreground.config
        ),
        left_image_content_sha256=case.foreground.left_image_content_sha256,
        depth_content_sha256=case.foreground.depth_content_sha256,
        valid_mask_content_sha256=case.foreground.valid_mask_content_sha256,
    )
    case.foreground = projected
    monkeypatch.setattr(
        coarse_scan_module,
        "_replay_foreground",
        lambda **_kwargs: projected,
    )

    output = _write_coarse_writer_case(case)
    stored = read_coarse_scan_view(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert stored.foreground.guide == guide
    assert metadata["foreground"]["algorithm"] == (
        PROJECTED_COARSE_FOREGROUND_ALGORITHM
    )
    assert metadata["foreground"]["guide"] == guide.payload()
    assert metadata["sources"]["foreground_reference_generation"]["authority"] == (
        "generation.json"
    )

    generation_metadata.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="directory source changed"):
        read_coarse_scan_view(output)


def test_generation_reader_memoizes_recursive_projected_prefix_per_top_level_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four generations replay four unique views, not all 15 prefix occurrences."""

    shape = (6, 6)
    left = np.arange(36, dtype=np.float64).reshape(shape)
    depth = np.full(shape, 0.5, dtype=np.float64)
    valid = np.ones(shape, dtype=np.bool_)
    config = BootstrapForegroundConfig(
        boundary_margin_px=1,
        minimum_valid_pixels=1,
        minimum_component_pixels=1,
        minimum_mask_pixels=1,
        minimum_mask_fraction=0.0,
        maximum_mask_fraction=1.0,
        minimum_seed_valid_pixels=1,
        minimum_seed_valid_fraction=0.0,
    )
    bootstrap = bootstrap_blade_foreground(
        left,
        depth,
        valid,
        config,
        BootstrapSeed.rectangle(1, 1, 4, 4, mode="hard_roi"),
    )
    pixels = np.argwhere(bootstrap.mask)[:, ::-1]
    points = np.column_stack(
        (
            np.full(len(pixels), 0.5),
            pixels[:, 0] * 0.01,
            np.full(len(pixels), 0.2),
        )
    )
    proxy_config = ProxyModelConfig(
        estimated_thickness_m=0.01,
        minimum_points=6,
        blade_envelope_min_m=(0.0, 0.0, 0.0),
        blade_envelope_max_m=(1.0, 1.0, 1.0),
        minimum_envelope_retained_fraction=0.4,
    )
    proxy_support = coarse_scan_module.select_proxy_support(
        points,
        proxy_config,
        frame="base",
    )

    common_roots: dict[str, Path] = {}
    for name, filename in (
        ("initialization", "metadata.json"),
        ("view_plan", "view_plan.json"),
    ):
        root = tmp_path / name
        root.mkdir()
        (root / filename).write_text("{}\n", encoding="utf-8")
        common_roots[name] = root

    reconstructed_by_root: dict[Path, StoredReconstructedBladeView] = {}
    foreground_by_view_id: dict[str, object] = {}
    view_roots: list[Path] = []
    generation_roots: list[Path] = []
    coverage_by_root: dict[Path, SimpleNamespace] = {}
    occupancy_integrity_sha256: dict[Path, str] = {}

    def array_record(path: Path) -> dict[str, object]:
        value = np.load(path, allow_pickle=False)
        return {
            "path": path.name,
            "sha256": coarse_scan_module._sha256(path),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    for index in range(4):
        view_id = f"coarse_{index:02d}"
        reconstructed_root = tmp_path / f"reconstructed-{index}"
        stereo_root = tmp_path / f"stereo-{index}"
        occupancy_root = tmp_path / f"occupancy-{index}"
        view_root = tmp_path / f"coarse-view-{index}"
        for root in (reconstructed_root, stereo_root, occupancy_root, view_root):
            root.mkdir()
            (root / "metadata.json").write_text("{}\n", encoding="utf-8")
        np.save(
            occupancy_root / "integrity.npy",
            np.full(shape, index, dtype=np.int16),
            allow_pickle=False,
        )
        np.save(
            reconstructed_root / "integrity.npy",
            np.full(shape, index, dtype=np.int32),
            allow_pickle=False,
        )
        np.save(
            stereo_root / "integrity.npy",
            np.full(shape, index, dtype=np.float32),
            allow_pickle=False,
        )
        (reconstructed_root / "metadata.json").write_text(
            json.dumps(
                {
                    "files": {
                        "integrity": array_record(reconstructed_root / "integrity.npy")
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (stereo_root / "metadata.json").write_text(
            json.dumps(
                {"files": {"integrity": array_record(stereo_root / "integrity.npy")}}
            )
            + "\n",
            encoding="utf-8",
        )
        occupancy_integrity_sha256[occupancy_root.resolve()] = coarse_scan_module._sha256(
            occupancy_root / "integrity.npy"
        )
        view = ReconstructedBladeView(
            view_id,
            index,
            index,
            CameraIntrinsics(6, 6, 100.0, 100.0, 2.5, 2.5, "none", ()),
            np.zeros(6),
            PoseSE3.identity("base", "left_ir"),
            PoseSE3.from_rotation_translation(
                "base",
                "left_rectified",
                np.eye(3),
                (0.0, 0.0, 0.5),
            ),
            PointCloud("base", points, pixels, shape),
            "foundation_stereo",
        )
        reconstructed_by_root[reconstructed_root.resolve()] = StoredReconstructedBladeView(
            view,
            bootstrap.mask,
            {
                "source": {
                    "session": str((tmp_path / "session").resolve()),
                    "stereo_inference": str(stereo_root.resolve()),
                    "view_id": view_id,
                }
            },
        )
        if index == 0:
            foreground = bootstrap
            guide_payload = None
            reference_record = None
        else:
            predecessor = generation_roots[index - 1]
            predecessor_metadata = predecessor / "generation.json"
            guide = ProjectedCoarseForegroundGuide(
                source_generation_path=predecessor.resolve(),
                source_generation_metadata_sha256=coarse_scan_module._sha256(
                    predecessor_metadata
                ),
                reference_points_content_sha256=f"{index}" * 64,
                blade_envelope_min_m=(0.0, 0.0, 0.0),
                blade_envelope_max_m=(1.0, 1.0, 1.0),
            )
            mask_count = int(np.count_nonzero(bootstrap.mask))
            seed_count = int(np.count_nonzero(bootstrap.seed_mask))
            foreground = ProjectedCoarseForegroundResult(
                mask=bootstrap.mask,
                projected_reference_mask=bootstrap.seed_mask,
                diagnostics=ProjectedCoarseForegroundDiagnostics(
                    image_pixel_count=bootstrap.mask.size,
                    supplied_valid_pixel_count=bootstrap.mask.size,
                    depth_valid_pixel_count=bootstrap.mask.size,
                    reference_point_count=len(points),
                    projected_reference_pixel_count=seed_count,
                    eligible_projected_pixel_count=seed_count,
                    predicted_depth_consistent_pixel_count=mask_count,
                    base_envelope_pixel_count=mask_count,
                    mask_pixel_count=mask_count,
                    mask_fraction=mask_count / bootstrap.mask.size,
                    projected_match_fraction=mask_count / seed_count,
                    minimum_mask_depth_m=0.5,
                    median_mask_depth_m=0.5,
                    maximum_mask_depth_m=0.5,
                ),
                config=config,
                guide=guide,
                algorithm=PROJECTED_COARSE_FOREGROUND_ALGORITHM,
                policy_sha256=projected_coarse_foreground_policy_sha256(config),
                left_image_content_sha256=bootstrap.left_image_content_sha256,
                depth_content_sha256=bootstrap.depth_content_sha256,
                valid_mask_content_sha256=bootstrap.valid_mask_content_sha256,
            )
            guide_payload = guide.payload()
            reference_record = coarse_scan_module._directory_record(
                predecessor,
                "generation.json",
            )
        foreground_by_view_id[view_id] = foreground
        np.save(view_root / "mask.npy", foreground.mask, allow_pickle=False)
        np.save(view_root / "seed_mask.npy", foreground.seed_mask, allow_pickle=False)
        np.save(
            view_root / "proxy_support_mask.npy",
            proxy_support.mask,
            allow_pickle=False,
        )
        view_payload = {
            "schema_version": 3,
            "artifact_kind": "biblade_fusion.coarse_scan_view",
            "motion_authorized": False,
            "target": {"view_id": view_id, "kind": "proxy_normal", "side": "front"},
            "identity": {"view_id": view_id, "sequence_index": index, "frame_number": index},
            "foreground": {
                "algorithm": foreground.algorithm,
                "config": asdict(foreground.config),
                "seed": coarse_scan_module.bootstrap_seed_payload(foreground.seed),
                "guide": guide_payload,
                "policy_sha256": foreground.policy_sha256,
                "diagnostics": asdict(foreground.diagnostics),
                "input_content_sha256": {
                    "left_rectified": foreground.left_image_content_sha256,
                    "depth_m": foreground.depth_content_sha256,
                    "integration_valid_mask": foreground.valid_mask_content_sha256,
                },
            },
            "proxy_support": {
                "configuration": proxy_config.model_dump(mode="json"),
                "diagnostics": proxy_support.metadata_payload(),
            },
            "files": {
                name: array_record(view_root / filename)
                for name, filename in (
                    ("mask", "mask.npy"),
                    ("seed_mask", "seed_mask.npy"),
                    ("proxy_support_mask", "proxy_support_mask.npy"),
                )
            },
            "sources": {
                "reconstructed_view": coarse_scan_module._directory_record(
                    reconstructed_root,
                    "metadata.json",
                ),
                "stereo_inference": coarse_scan_module._directory_record(
                    stereo_root,
                    "metadata.json",
                ),
                "occupancy_mapping": coarse_scan_module._directory_record(
                    occupancy_root,
                    "metadata.json",
                ),
                **(
                    {"foreground_reference_generation": reference_record}
                    if reference_record is not None
                    else {}
                ),
            },
        }
        (view_root / "metadata.json").write_text(
            json.dumps(view_payload) + "\n",
            encoding="utf-8",
        )
        view_roots.append(view_root)

        discovery_root = tmp_path / f"discovery-{index}"
        coverage_root = tmp_path / f"coverage-{index}"
        discovery_root.mkdir()
        coverage_root.mkdir()
        (discovery_root / "discovery.json").write_text("{}\n", encoding="utf-8")
        (coverage_root / "coverage.json").write_text("{}\n", encoding="utf-8")
        coverage_by_root[coverage_root.resolve()] = SimpleNamespace(
            metadata={
                "previous_ledger": (
                    str((tmp_path / f"coverage-{index - 1}").resolve())
                    if index
                    else None
                )
            }
        )
        generation_root = tmp_path / f"generation-{index}"
        generation_root.mkdir()
        generation_payload = {
            "schema_version": 1,
            "artifact_kind": "biblade_fusion.coarse_scan_generation",
            "motion_authorized": False,
            "generation_index": index,
            "previous_generation": (
                coarse_scan_module._directory_record(
                    generation_roots[index - 1],
                    "generation.json",
                )
                if index
                else None
            ),
            "sources": {
                "initialization": coarse_scan_module._directory_record(
                    common_roots["initialization"],
                    "metadata.json",
                ),
                "view_plan": coarse_scan_module._directory_record(
                    common_roots["view_plan"],
                    "view_plan.json",
                ),
                "discovery_plan": coarse_scan_module._directory_record(
                    discovery_root,
                    "discovery.json",
                ),
                "coverage": coarse_scan_module._directory_record(
                    coverage_root,
                    "coverage.json",
                ),
                "coarse_model": None,
            },
            "views": [
                coarse_scan_module._directory_record(root, "metadata.json")
                for root in view_roots
            ],
            "summary": {
                "view_count": index + 1,
                "front_view_count": index + 1,
                "back_view_count": 0,
                "schema5_ready": False,
            },
        }
        (generation_root / "generation.json").write_text(
            json.dumps(generation_payload) + "\n",
            encoding="utf-8",
        )
        generation_roots.append(generation_root)

    reconstructed_reads: list[Path] = []
    replayed_view_ids: list[str] = []

    def read_reconstructed(path: str | Path) -> StoredReconstructedBladeView:
        root = Path(path).resolve()
        reconstructed_reads.append(root)
        return reconstructed_by_root[root]

    def replay_foreground(**kwargs: object) -> object:
        reconstructed = kwargs["reconstructed"]
        assert isinstance(reconstructed, StoredReconstructedBladeView)
        view_id = reconstructed.view.source_view_id
        replayed_view_ids.append(view_id)
        guide = kwargs["guide"]
        if isinstance(guide, ProjectedCoarseForegroundGuide):
            predecessor = read_coarse_scan_generation(guide.source_generation_path)
            assert predecessor.metadata_sha256 == guide.source_generation_metadata_sha256
        return foreground_by_view_id[view_id]

    def read_occupancy_integrity(path: str | Path) -> SimpleNamespace:
        root = Path(path).resolve()
        if (
            coarse_scan_module._sha256(root / "integrity.npy")
            != occupancy_integrity_sha256[root]
        ):
            raise ValueError("occupancy array checksum mismatch")
        return SimpleNamespace()

    monkeypatch.setattr(coarse_scan_module, "read_reconstructed_view", read_reconstructed)
    monkeypatch.setattr(coarse_scan_module, "_replay_foreground", replay_foreground)
    monkeypatch.setattr(
        coarse_scan_module,
        "read_occupancy_mapping_for_replay",
        read_occupancy_integrity,
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "read_coverage_ledger",
        lambda path: coverage_by_root[Path(path).resolve()],
    )
    monkeypatch.setattr(coarse_scan_module, "_assert_coverage_replays", lambda **_kwargs: None)
    monkeypatch.setattr(
        initialization_module,
        "read_initialization",
        lambda _path: SimpleNamespace(
            observation=SimpleNamespace(
                proxy=SimpleNamespace(frame_T_proxy=PoseSE3.identity("base", "proxy"))
            )
        ),
    )

    with coarse_scan_module._strict_coarse_read_transaction():
        first = read_coarse_scan_generation(generation_roots[-1])

        assert replayed_view_ids == [f"coarse_{index:02d}" for index in range(4)]
        assert len(reconstructed_reads) == 4

        cached = read_coarse_scan_generation(generation_roots[-1])

        assert cached is first
        assert replayed_view_ids == [f"coarse_{index:02d}" for index in range(4)]

        np.save(view_roots[0] / "mask.npy", ~bootstrap.mask, allow_pickle=False)
        with pytest.raises(ValueError, match="checksum mismatch"):
            read_coarse_scan_generation(generation_roots[-1])
        np.save(view_roots[0] / "mask.npy", bootstrap.mask, allow_pickle=False)

        reconstructed_integrity = tmp_path / "reconstructed-0" / "integrity.npy"
        np.save(
            reconstructed_integrity,
            np.full(shape, 99, dtype=np.int32),
            allow_pickle=False,
        )
        with pytest.raises(ValueError, match="checksum mismatch"):
            read_coarse_scan_generation(generation_roots[-1])
        np.save(
            reconstructed_integrity,
            np.full(shape, 0, dtype=np.int32),
            allow_pickle=False,
        )

        stereo_integrity = tmp_path / "stereo-0" / "integrity.npy"
        np.save(
            stereo_integrity,
            np.full(shape, 99, dtype=np.float32),
            allow_pickle=False,
        )
        with pytest.raises(ValueError, match="checksum mismatch"):
            read_coarse_scan_generation(generation_roots[-1])
        np.save(
            stereo_integrity,
            np.full(shape, 0, dtype=np.float32),
            allow_pickle=False,
        )

        occupancy_integrity = tmp_path / "occupancy-0" / "integrity.npy"
        np.save(
            occupancy_integrity,
            np.full(shape, 99, dtype=np.int16),
            allow_pickle=False,
        )
        with pytest.raises(ValueError, match="occupancy array checksum mismatch"):
            read_coarse_scan_generation(generation_roots[-1])
        np.save(
            occupancy_integrity,
            np.full(shape, 0, dtype=np.int16),
            allow_pickle=False,
        )

        initialization_authority = common_roots["initialization"] / "metadata.json"
        original_initialization = initialization_authority.read_bytes()
        initialization_authority.write_text('{"changed":true}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="directory source changed"):
            read_coarse_scan_generation(generation_roots[-1])
        initialization_authority.write_bytes(original_initialization)

    assert first.generation_index == 3
    assert tuple(item.target_view_id for item in first.views) == tuple(
        f"coarse_{index:02d}" for index in range(4)
    )
    assert replayed_view_ids == [f"coarse_{index:02d}" for index in range(4)]
    assert len(reconstructed_reads) == 4
    assert coarse_scan_module._STRICT_READ_CONTEXT.get() is None

    replayed_view_ids.clear()
    reconstructed_reads.clear()
    second = read_coarse_scan_generation(generation_roots[-1])

    assert second.metadata_sha256 == first.metadata_sha256
    assert len(replayed_view_ids) == 4
    assert len(reconstructed_reads) == 4

    np.save(view_roots[0] / "mask.npy", ~bootstrap.mask, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_coarse_scan_generation(generation_roots[-1])


def test_strict_read_context_rejects_conflicting_authority_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "view"
    root.mkdir()
    authority = root / "metadata.json"
    authority.write_text('{"revision": 1}\n', encoding="utf-8")
    first_record = coarse_scan_module._directory_record(root, "metadata.json")
    context = coarse_scan_module._StrictReadContext()
    resolved, first_identity = coarse_scan_module._resolve_bound_directory_record(
        first_record,
        expected_authority="metadata.json",
    )
    context.expected_views[resolved] = first_identity

    authority.write_text('{"revision": 2}\n', encoding="utf-8")
    conflicting_record = coarse_scan_module._directory_record(root, "metadata.json")

    with pytest.raises(ValueError, match="conflicting authority identities"):
        coarse_scan_module._read_bound_coarse_scan_view(conflicting_record, context)


def test_strict_read_context_rejects_recursive_generation_cycle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generation"
    root.mkdir()
    authority = root / "generation.json"
    authority.write_text("{}\n", encoding="utf-8")
    content = authority.read_bytes()
    identity = coarse_scan_module._authority_identity_from_bytes(
        root,
        authority="generation.json",
        content=content,
    )
    context = coarse_scan_module._StrictReadContext()
    context.generations_in_progress.add(identity)
    token = coarse_scan_module._STRICT_READ_CONTEXT.set(context)
    try:
        with pytest.raises(ValueError, match="authority graph is cyclic"):
            read_coarse_scan_generation(root)
    finally:
        coarse_scan_module._STRICT_READ_CONTEXT.reset(token)


def test_strict_read_context_binds_full_occupancy_storage_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "occupancy"
    root.mkdir()
    mask = np.asarray(((True, False), (False, True)), dtype=np.bool_)
    mask_path = root / "integration_valid_mask.npy"
    np.save(mask_path, mask, allow_pickle=False)
    metadata = {
        "frames": [
            {
                "files": {
                    "integration_valid_mask": {
                        "path": mask_path.name,
                        "sha256": coarse_scan_module._sha256(mask_path),
                        "dtype": str(mask.dtype),
                        "shape": list(mask.shape),
                    }
                }
            }
        ]
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    authority = SimpleNamespace(
        root=root.resolve(),
        metadata_sha256=coarse_scan_module._sha256(metadata_path),
        metadata_size_bytes=metadata_path.stat().st_size,
    )
    monkeypatch.setattr(coarse_scan_module, "read_occupancy_mapping", lambda _path: object())
    monkeypatch.setattr(
        coarse_scan_module,
        "_bind_occupancy_storage_authority",
        lambda _path, _stored: authority,
    )

    with coarse_scan_module._strict_coarse_read_transaction():
        replayed = coarse_scan_module._load_final_integration_mask(root)
        context = coarse_scan_module._STRICT_READ_CONTEXT.get()
        assert context is not None
        assert tuple(context.occupancy_authorities.values()) == (authority,)

    np.testing.assert_array_equal(replayed, mask)
    assert coarse_scan_module._STRICT_READ_CONTEXT.get() is None


@pytest.mark.parametrize(
    "tamper_target",
    [
        "metadata",
        "mask",
        "nonfinal_array",
        "snapshot",
        "stereo_metadata",
        "hand_eye",
    ],
)
def test_coarse_writer_rechecks_occupancy_before_publish_and_cleans_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    case = _coarse_writer_case(tmp_path, monkeypatch)
    original_array_record = coarse_scan_module._array_record
    tampered = False

    def array_record(path: Path) -> dict[str, object]:
        nonlocal tampered
        if not tampered:
            tampered = True
            if tamper_target == "metadata":
                metadata_path = case.occupancy_root / "metadata.json"
                content = metadata_path.read_text(encoding="utf-8")
                assert '"schema_version": 7' in content
                metadata_path.write_text(
                    content.replace('"schema_version": 7', '"schema_version": 8', 1),
                    encoding="utf-8",
                )
            elif tamper_target == "mask":
                np.save(
                    case.integration_mask_path,
                    ~case.integration_valid,
                    allow_pickle=False,
                )
            else:
                case.authority_paths[tamper_target].write_bytes(
                    f"tampered {tamper_target}\n".encode()
                )
        return original_array_record(path)

    monkeypatch.setattr(coarse_scan_module, "_array_record", array_record)

    with pytest.raises((OSError, ValueError)):
        _write_coarse_writer_case(case)

    assert case.occupancy_reads == [case.occupancy_root, case.occupancy_root]
    assert not case.output.exists()
    assert list(tmp_path.glob(".coarse-view.*.partial")) == []


def _fake_stored_view(tmp_path: Path) -> StoredCoarseScanView:
    session = (tmp_path / "session").resolve()
    source_records: dict[str, dict[str, object]] = {}
    for name, root_name in (
        ("reconstructed_view", "rv"),
        ("stereo_inference", "stereo"),
        ("occupancy_mapping", "occupancy"),
    ):
        root = (tmp_path / root_name).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "metadata.json").write_text("{}\n", encoding="utf-8")
        source_records[name] = coarse_scan_module._directory_record(
            root,
            "metadata.json",
        )
    cloud = PointCloud(
        "base",
        np.asarray(((0.0, 0.0, 0.5),), dtype=np.float64),
        np.asarray(((0, 0),), dtype=np.int64),
        (1, 1),
    )
    view = SimpleNamespace(
        source_view_id="coarse_00",
        source_sequence_index=2,
        source_frame_number=17,
        base_cloud=cloud,
        base_t_projection_camera=object(),
    )
    root = (tmp_path / "coarse-view").resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"sources": source_records}
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return StoredCoarseScanView(
        root=root,
        reconstructed=SimpleNamespace(
            view=view,
            metadata={"source": {"session": str(session)}},
        ),
        foreground=SimpleNamespace(),
        target_view_id="coarse_00",
        target_kind="operator_seed",
        target_side=BladeSide.FRONT,
        proxy_support=SimpleNamespace(mask=np.asarray((True,), dtype=np.bool_)),
        proxy_config=ProxyModelConfig(),
        metadata=metadata,
        metadata_sha256=coarse_scan_module._sha256(metadata_path),
        metadata_size_bytes=metadata_path.stat().st_size,
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


def test_coarse_view_transaction_readback_rechecks_exact_authorities(
    tmp_path: Path,
) -> None:
    stored_view = _fake_stored_view(tmp_path)

    readback = coarse_scan_module._bind_coarse_scan_view_readback(stored_view)
    verified = coarse_scan_module._revalidate_coarse_scan_view_readback(
        readback,
        expected_root=stored_view.root,
    )

    assert verified is stored_view


@pytest.mark.parametrize("tamper_target", ("view", "occupancy_source"))
def test_coarse_view_transaction_readback_rejects_late_mutation(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    stored_view = _fake_stored_view(tmp_path)
    readback = coarse_scan_module._bind_coarse_scan_view_readback(stored_view)
    target = (
        stored_view.root / "metadata.json"
        if tamper_target == "view"
        else Path(stored_view.metadata["sources"]["occupancy_mapping"]["root"])
        / "metadata.json"
    )
    target.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="changed"):
        coarse_scan_module._revalidate_coarse_scan_view_readback(
            readback,
            expected_root=stored_view.root,
        )


def test_generation_writer_rejects_same_count_wrong_physical_observation_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_view = _fake_stored_view(tmp_path)
    coverage = _fake_coverage(tmp_path, observation_ids=("same-count-but-wrong",))
    for root, filename in (
        (tmp_path / "initialization", "metadata.json"),
        (tmp_path / "view-plan", "view_plan.json"),
        (tmp_path / "discovery", "discovery.json"),
        (tmp_path / "coverage", "coverage.json"),
    ):
        root.mkdir(parents=True, exist_ok=True)
        (root / filename).write_text("{}\n", encoding="utf-8")
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


def test_generation_writer_reuses_transaction_verified_views_without_full_read(
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
    coverage = _fake_coverage(tmp_path, observation_ids=(observation_id,))
    authorities = (
        (tmp_path / "initialization", "metadata.json"),
        (tmp_path / "view-plan", "view_plan.json"),
        (tmp_path / "discovery", "discovery.json"),
        (tmp_path / "coverage", "coverage.json"),
        (stored_view.root, "metadata.json"),
    )
    for root, filename in authorities:
        root.mkdir(parents=True, exist_ok=True)
        authority = root / filename
        if not authority.exists():
            authority.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: coverage)
    monkeypatch.setattr(coarse_scan_module, "_assert_coverage_replays", lambda **_kwargs: None)
    monkeypatch.setattr(
        coarse_scan_module,
        "read_coarse_scan_view",
        lambda _path: pytest.fail("verified views must not be read again"),
    )
    monkeypatch.setattr(
        coarse_scan_module,
        "read_coarse_scan_generation",
        lambda _path: pytest.fail("a verified predecessor must not be read again"),
    )

    output, stored_generation = (
        coarse_scan_module._write_coarse_scan_generation_from_verified(
            tmp_path / "generation",
            views=(stored_view.root,),
            verified_views=(stored_view,),
            coverage=tmp_path / "coverage",
            source_initialization=tmp_path / "initialization",
            source_view_plan=tmp_path / "view-plan",
            source_discovery_plan=tmp_path / "discovery",
            previous_generation=None,
            verified_previous_generation=None,
        )
    )

    payload = json.loads((output / "generation.json").read_text(encoding="utf-8"))
    assert payload["summary"]["view_count"] == 1
    assert payload["views"][0]["root"] == str(stored_view.root)
    assert stored_generation.root == output
    assert stored_generation.views == (stored_view,)
    assert stored_generation.metadata == payload


@pytest.mark.parametrize("tamper_target", ("view", "occupancy_source"))
def test_generation_writer_rechecks_verified_view_authority_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
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
    coverage = _fake_coverage(tmp_path, observation_ids=(observation_id,))
    authorities = (
        (tmp_path / "initialization", "metadata.json"),
        (tmp_path / "view-plan", "view_plan.json"),
        (tmp_path / "discovery", "discovery.json"),
        (tmp_path / "coverage", "coverage.json"),
        (stored_view.root, "metadata.json"),
    )
    for root, filename in authorities:
        root.mkdir(parents=True, exist_ok=True)
        authority = root / filename
        if not authority.exists():
            authority.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: coverage)
    monkeypatch.setattr(coarse_scan_module, "_assert_coverage_replays", lambda **_kwargs: None)
    original_resolve = coarse_scan_module._resolve_directory_record
    tampered = False

    def resolve_record(record: dict[str, object]) -> Path:
        nonlocal tampered
        resolved = original_resolve(record)
        if not tampered and resolved == stored_view.root:
            tampered = True
            target = (
                stored_view.root / "metadata.json"
                if tamper_target == "view"
                else Path(
                    stored_view.metadata["sources"]["occupancy_mapping"]["root"]
                )
                / "metadata.json"
            )
            target.write_text('{"tampered": true}\n', encoding="utf-8")
        return resolved

    monkeypatch.setattr(coarse_scan_module, "_resolve_directory_record", resolve_record)
    output = tmp_path / "generation"

    with pytest.raises(ValueError, match="changed|differs"):
        coarse_scan_module._write_coarse_scan_generation_from_verified(
            output,
            views=(stored_view.root,),
            verified_views=(stored_view,),
            coverage=tmp_path / "coverage",
            source_initialization=tmp_path / "initialization",
            source_view_plan=tmp_path / "view-plan",
            source_discovery_plan=tmp_path / "discovery",
            previous_generation=None,
            verified_previous_generation=None,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".generation.*.partial")) == []


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


def test_generation_reader_allows_new_discovery_revision_for_appended_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _fake_stored_view(tmp_path)
    first_view = SimpleNamespace(
        **{
            **vars(first.reconstructed.view),
            "base_t_projection_camera": PoseSE3.from_rotation_translation(
                "base", "camera-0", np.eye(3), (0.0, 0.0, 0.5)
            ),
        }
    )
    first = replace(
        first,
        reconstructed=SimpleNamespace(
            view=first_view,
            metadata=first.reconstructed.metadata,
        ),
    )
    second_root = (tmp_path / "coarse-view-2").resolve()
    second_root.mkdir()
    second_metadata_path = second_root / "metadata.json"
    second_metadata_path.write_text(json.dumps(first.metadata) + "\n", encoding="utf-8")
    second_view = SimpleNamespace(
        **{
            **vars(first.reconstructed.view),
            "source_view_id": "coarse_01",
            "source_sequence_index": 3,
            "source_frame_number": 18,
            "base_t_projection_camera": PoseSE3.from_rotation_translation(
                "base", "camera-1", np.eye(3), (0.0, 0.1, 0.5)
            ),
        }
    )
    second = replace(
        first,
        root=second_root,
        reconstructed=SimpleNamespace(
            view=second_view,
            metadata=first.reconstructed.metadata,
        ),
        target_view_id="coarse_01",
        metadata_sha256=coarse_scan_module._sha256(second_metadata_path),
        metadata_size_bytes=second_metadata_path.stat().st_size,
    )
    authorities = {
        "initialization": (tmp_path / "initialization", "metadata.json"),
        "view_plan": (tmp_path / "view-plan", "view_plan.json"),
        "discovery_0": (tmp_path / "discovery-0", "discovery.json"),
        "discovery_1": (tmp_path / "discovery-1", "discovery.json"),
        "coverage_0": (tmp_path / "coverage-0", "coverage.json"),
        "coverage_1": (tmp_path / "coverage-1", "coverage.json"),
    }
    for root, filename in authorities.values():
        root.mkdir()
        (root / filename).write_text("{}\n", encoding="utf-8")

    def directory(name: str) -> dict[str, object]:
        return coarse_scan_module._directory_record(*authorities[name])

    first_generation = tmp_path / "generation-0"
    first_generation.mkdir()
    (first_generation / "generation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.coarse_scan_generation",
                "motion_authorized": False,
                "generation_index": 0,
                "previous_generation": None,
                "sources": {
                    "initialization": directory("initialization"),
                    "view_plan": directory("view_plan"),
                    "discovery_plan": directory("discovery_0"),
                    "coverage": directory("coverage_0"),
                    "coarse_model": None,
                },
                "views": [coarse_scan_module._directory_record(first.root, "metadata.json")],
                "summary": {
                    "view_count": 1,
                    "front_view_count": 1,
                    "back_view_count": 0,
                    "schema5_ready": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    generation = tmp_path / "generation-1"
    generation.mkdir()
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.coarse_scan_generation",
                "motion_authorized": False,
                "generation_index": 1,
                "previous_generation": coarse_scan_module._directory_record(
                    first_generation, "generation.json"
                ),
                "sources": {
                    "initialization": directory("initialization"),
                    "view_plan": directory("view_plan"),
                    "discovery_plan": directory("discovery_1"),
                    "coverage": directory("coverage_1"),
                    "coarse_model": None,
                },
                "views": [
                    coarse_scan_module._directory_record(first.root, "metadata.json"),
                    coarse_scan_module._directory_record(second.root, "metadata.json"),
                ],
                "summary": {
                    "view_count": 2,
                    "front_view_count": 2,
                    "back_view_count": 0,
                    "schema5_ready": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation_ids = tuple(
        coverage_observation_id(
            item.reconstructed.metadata["source"]["session"],
            item.reconstructed.view.source_view_id,
            item.reconstructed.view.source_sequence_index,
            item.reconstructed.view.source_frame_number,
        )
        for item in (first, second)
    )
    coverage = _fake_coverage(tmp_path, observation_ids=observation_ids)
    coverage.metadata["previous_ledger"] = str(authorities["coverage_0"][0].resolve())
    monkeypatch.setattr(
        coarse_scan_module,
        "read_coarse_scan_view",
        lambda path: first if Path(path).resolve() == first.root else second,
    )
    monkeypatch.setattr(coarse_scan_module, "read_coverage_ledger", lambda _path: coverage)
    monkeypatch.setattr(coarse_scan_module, "_assert_coverage_replays", lambda **_kwargs: None)
    monkeypatch.setattr(
        initialization_module,
        "read_initialization",
        lambda _path: SimpleNamespace(
            observation=SimpleNamespace(
                proxy=SimpleNamespace(frame_T_proxy=PoseSE3.identity("base", "proxy"))
            )
        ),
    )

    stored = read_coarse_scan_generation(generation)

    assert stored.generation_index == 1
    assert stored.metadata["sources"]["discovery_plan"]["root"] == str(
        authorities["discovery_1"][0].resolve()
    )


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
    for root, filename in (
        (tmp_path / "initialization", "metadata.json"),
        (tmp_path / "view-plan", "view_plan.json"),
        (tmp_path / "discovery", "discovery.json"),
        (tmp_path / "coverage", "coverage.json"),
    ):
        root.mkdir(parents=True, exist_ok=True)
        (root / filename).write_text("{}\n", encoding="utf-8")
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
