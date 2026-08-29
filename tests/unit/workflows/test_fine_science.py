from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.settings import BladeForegroundConfig, load_settings
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping import OccupancyMapState
from biblade_fusion.workflows import fine_science as module
from biblade_fusion.workflows.stop_scan_coordinator import CapturePurpose


def _settings():
    settings = load_settings("configs/default.yaml")
    return settings.model_copy(
        update={
            "blade_foreground": BladeForegroundConfig(
                enabled=True,
                minimum_reference_pixels=1,
                minimum_target_reference_pixels=1,
                minimum_mask_pixels=1,
                minimum_target_mask_pixels=1,
            )
        }
    )


def _captured(tmp_path: Path, purpose: CapturePurpose):
    cycle = (tmp_path / "cycle").resolve()
    raw = cycle / "raw"
    raw.mkdir(parents=True)
    bundle = SimpleNamespace(
        view_id="candidate-001",
        sequence_index=7,
        stereo=SimpleNamespace(frame_number=19),
    )
    return SimpleNamespace(
        purpose=purpose,
        bundle=bundle,
        cycle_root=cycle,
        raw_session_path=raw,
    )


def _occupancy(state: OccupancyMapState):
    eligible = np.ones((3, 4), dtype=np.bool_)
    return SimpleNamespace(
        snapshot=SimpleNamespace(map_state=state),
        integration_valid_mask=eligible,
        evidence=SimpleNamespace(
            base_t_camera_matrix=np.eye(4),
            integration_valid_mask_content_hash="1" * 64,
        ),
    )


def _reference(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        module,
        "read_coarse_model_summary",
        lambda path: SimpleNamespace(root=Path(path).resolve(), metadata={"schema_version": 5}),
    )
    monkeypatch.setattr(
        module,
        "production_selection_policy_payload",
        lambda _settings, _hand_eye, *, reference_coarse_model: {
            "test_reference": str(Path(reference_coarse_model).resolve())
        },
    )
    root.mkdir()


def _policy_metadata(root: Path) -> dict[str, object]:
    return {
        "reacquisition_policy": module._selection_policy_record(
            {"test_reference": str(root.resolve())}
        )
    }


def test_startup_rejects_accepted_lineage_without_schema_3_science_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "coarse"
    accepted = tmp_path / "accepted"
    _reference(monkeypatch, reference)
    accepted.mkdir()

    def reject_legacy(path: Path, **kwargs):
        assert Path(path).resolve() == accepted.resolve()
        assert kwargs == {"require_foreground_bound_science": True}
        raise ValueError("legacy schema-2 observation")

    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        reject_legacy,
    )

    with pytest.raises(
        module.FineSciencePreparationError,
        match="foreground-bound schema-3 verification",
    ):
        module.validate_fine_science_startup(
            _settings(),
            SimpleNamespace(),
            reference_coarse_model=reference,
            accepted_coverage_path=accepted,
        )


def test_bootstrap_waits_for_map_ready_before_writing_empty_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "coarse"
    _reference(monkeypatch, reference)
    captured = _captured(tmp_path, CapturePurpose.BOOTSTRAP)
    writes: list[Path] = []
    monkeypatch.setattr(
        module,
        "write_surface_coverage_generation",
        lambda path, **kwargs: writes.append(Path(path)),
    )

    pending = module.prepare_fine_science_assets(
        purpose=CapturePurpose.BOOTSTRAP,
        captured=captured,
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(OccupancyMapState.MAPPING),
        occupancy_path=tmp_path / "occupancy",
        settings=_settings(),
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=None,
    )

    assert pending.coverage_path is None
    assert pending.advances_coverage is False
    assert writes == []


def test_first_map_ready_bootstrap_stages_local_empty_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    reference = tmp_path / "coarse"
    _reference(monkeypatch, reference)
    captured = _captured(tmp_path, CapturePurpose.BOOTSTRAP)
    coverage_path = captured.cycle_root / "surface_coverage"

    def write_coverage(path, **kwargs):
        assert Path(path).resolve() == coverage_path.resolve()
        assert kwargs["reference_coarse_model"] == reference.resolve()
        assert "previous_generation" not in kwargs
        Path(path).mkdir()

    initial = SimpleNamespace(
        root=coverage_path.resolve(),
        reference=SimpleNamespace(root=reference.resolve()),
        quality_config=settings.surface_quality,
        ledger=SimpleNamespace(observation_ids=()),
        current_reconstructed_view_path=None,
        metadata=_policy_metadata(reference),
    )
    monkeypatch.setattr(module, "write_surface_coverage_generation", write_coverage)
    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        lambda path, **kwargs: initial,
    )

    pending = module.prepare_fine_science_assets(
        purpose=CapturePurpose.BOOTSTRAP,
        captured=captured,
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(OccupancyMapState.MAP_READY),
        occupancy_path=tmp_path / "occupancy",
        settings=settings,
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=None,
    )

    assert pending.coverage_path == coverage_path.resolve()
    assert pending.advances_coverage is True


def test_transit_carries_exact_accepted_generation_without_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "coarse"
    accepted = (tmp_path / "accepted").resolve()
    _reference(monkeypatch, reference)
    accepted.mkdir()
    state = SimpleNamespace(
        root=accepted,
        reference=SimpleNamespace(root=reference.resolve()),
        quality_config=_settings().surface_quality,
        metadata=_policy_metadata(reference),
    )
    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        lambda path, **kwargs: state,
    )
    monkeypatch.setattr(
        module,
        "reference_guided_blade_mask",
        lambda *args, **kwargs: pytest.fail("transit must not call the science masker"),
    )

    pending = module.prepare_fine_science_assets(
        purpose=CapturePurpose.TRANSIT,
        captured=_captured(tmp_path, CapturePurpose.TRANSIT),
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(OccupancyMapState.MAP_READY),
        occupancy_path=tmp_path / "occupancy",
        settings=_settings(),
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=accepted,
    )

    assert pending.coverage_path == accepted
    assert pending.blade_foreground_path is None
    assert pending.reconstructed_view_path is None
    assert pending.advances_coverage is False


@pytest.mark.parametrize(
    "purpose",
    [CapturePurpose.BOOTSTRAP, CapturePurpose.SAFETY_REFRESH],
)
def test_bootstrap_or_safety_refresh_carries_existing_generation_without_advancing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: CapturePurpose,
) -> None:
    settings = _settings()
    reference = tmp_path / "coarse"
    accepted = (tmp_path / "accepted").resolve()
    _reference(monkeypatch, reference)
    accepted.mkdir()
    state = SimpleNamespace(
        root=accepted,
        reference=SimpleNamespace(root=reference.resolve()),
        quality_config=settings.surface_quality,
        metadata=_policy_metadata(reference),
    )
    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        lambda path, **kwargs: state,
    )
    monkeypatch.setattr(
        module,
        "write_surface_coverage_generation",
        lambda *args, **kwargs: pytest.fail("an accepted generation must not be replaced"),
    )

    pending = module.prepare_fine_science_assets(
        purpose=purpose,
        captured=_captured(tmp_path, purpose),
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(OccupancyMapState.MAP_READY),
        occupancy_path=tmp_path / "occupancy",
        settings=settings,
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=accepted,
    )

    assert pending.coverage_path == accepted
    assert pending.blade_foreground_path is None
    assert pending.reconstructed_view_path is None
    assert pending.advances_coverage is False


@pytest.mark.parametrize(
    ("purpose", "map_state"),
    [
        (CapturePurpose.BOOTSTRAP, OccupancyMapState.MAPPING),
        (CapturePurpose.BOOTSTRAP, OccupancyMapState.STALE),
        (CapturePurpose.SAFETY_REFRESH, OccupancyMapState.MAPPING),
        (CapturePurpose.SAFETY_REFRESH, OccupancyMapState.STALE),
    ],
)
def test_preinitialization_nonready_safety_cycles_produce_no_science_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: CapturePurpose,
    map_state: OccupancyMapState,
) -> None:
    reference = tmp_path / "coarse"
    _reference(monkeypatch, reference)
    monkeypatch.setattr(
        module,
        "write_surface_coverage_generation",
        lambda *args, **kwargs: pytest.fail("non-ready safety map must not initialize coverage"),
    )

    pending = module.prepare_fine_science_assets(
        purpose=purpose,
        captured=_captured(tmp_path, purpose),
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(map_state),
        occupancy_path=tmp_path / "occupancy",
        settings=_settings(),
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=None,
    )

    assert pending == module.PreparedFineScienceAssets(None, None, None, False)


@pytest.mark.parametrize(
    "purpose",
    [CapturePurpose.BOOTSTRAP, CapturePurpose.SAFETY_REFRESH],
)
def test_first_map_ready_safety_cycle_stages_generation_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: CapturePurpose,
) -> None:
    settings = _settings()
    reference = tmp_path / "coarse"
    _reference(monkeypatch, reference)
    captured = _captured(tmp_path, purpose)
    coverage_path = (captured.cycle_root / "surface_coverage").resolve()
    writes: list[dict[str, object]] = []

    def write_coverage(path, **kwargs):
        assert Path(path).resolve() == coverage_path
        writes.append(kwargs)
        Path(path).mkdir()

    initial = SimpleNamespace(
        root=coverage_path,
        reference=SimpleNamespace(root=reference.resolve()),
        quality_config=settings.surface_quality,
        ledger=SimpleNamespace(observation_ids=()),
        current_reconstructed_view_path=None,
        metadata=_policy_metadata(reference),
    )
    monkeypatch.setattr(module, "write_surface_coverage_generation", write_coverage)
    monkeypatch.setattr(
        module,
        "read_surface_coverage_generation",
        lambda path, **kwargs: initial,
    )

    pending = module.prepare_fine_science_assets(
        purpose=purpose,
        captured=captured,
        stereo=SimpleNamespace(),
        stereo_path=tmp_path / "stereo",
        occupancy_update=_occupancy(OccupancyMapState.MAP_READY),
        occupancy_path=tmp_path / "occupancy",
        settings=settings,
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=None,
    )

    assert pending.coverage_path == coverage_path
    assert pending.blade_foreground_path is None
    assert pending.reconstructed_view_path is None
    assert pending.advances_coverage is True
    assert len(writes) == 1
    assert writes[0]["reference_coarse_model"] == reference.resolve()
    assert "previous_generation" not in writes[0]


def test_candidate_stages_mask_reconstruction_and_one_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    reference = tmp_path / "coarse"
    accepted = (tmp_path / "accepted").resolve()
    stereo_root = (tmp_path / "stereo").resolve()
    occupancy_root = (tmp_path / "occupancy").resolve()
    _reference(monkeypatch, reference)
    accepted.mkdir()
    stereo_root.mkdir()
    occupancy_root.mkdir()
    captured = _captured(tmp_path, CapturePurpose.CANDIDATE)
    intrinsics = CameraIntrinsics(4, 3, 100.0, 100.0, 1.5, 1.0, "none", ())
    candidate = SimpleNamespace(
        view_id=captured.bundle.view_id,
        patch=SimpleNamespace(patch_id="patch-1"),
    )
    previous = SimpleNamespace(
        root=accepted,
        reference=SimpleNamespace(root=reference.resolve()),
        quality_config=settings.surface_quality,
        view_plan=SimpleNamespace(candidates=(candidate,)),
        ledger=SimpleNamespace(observation_ids=()),
        surface=SimpleNamespace(),
        metadata=_policy_metadata(reference),
    )
    mask_array = np.ones((3, 4), dtype=np.bool_)
    mask_result = SimpleNamespace(mask=mask_array)
    reconstructed = SimpleNamespace()
    successor_path = captured.cycle_root / "surface_coverage"
    reconstructed_path = captured.cycle_root / "reconstructed_view"
    mask_path = captured.cycle_root / "blade_foreground"
    successor = SimpleNamespace(
        root=successor_path.resolve(),
        reference=previous.reference,
        quality_config=settings.surface_quality,
        previous_generation_path=accepted,
        current_reconstructed_view_path=reconstructed_path.resolve(),
        ledger=SimpleNamespace(observation_ids=(captured.bundle.view_id,)),
        current_reacquisition=None,
    )
    read_calls = 0

    def read_coverage(path, **kwargs):
        nonlocal read_calls
        read_calls += 1
        assert kwargs == {"require_foreground_bound_science": True}
        return previous if Path(path).resolve() == accepted else successor

    monkeypatch.setattr(module, "read_surface_coverage_generation", read_coverage)
    monkeypatch.setattr(module, "reference_guided_blade_mask", lambda *args: mask_result)

    def write_mask(path, result, **kwargs):
        assert result is mask_result
        assert kwargs["target_patch_id"] == "patch-1"
        Path(path).mkdir()

    monkeypatch.setattr(module, "write_blade_foreground_mask", write_mask)
    monkeypatch.setattr(
        module,
        "reconstruct_foundation_stereo_view",
        lambda *args, **kwargs: reconstructed,
    )

    def write_reconstruction(path, view, mask, *args, **kwargs):
        assert view is reconstructed
        np.testing.assert_array_equal(mask, mask_array)
        assert Path(kwargs["source_blade_foreground_mask"]).resolve() == mask_path.resolve()
        Path(path).mkdir()

    monkeypatch.setattr(module, "write_reconstructed_view", write_reconstruction)

    def write_coverage(path, **kwargs):
        assert Path(kwargs["previous_generation"]).resolve() == accepted
        assert Path(kwargs["current_reconstructed_view"]).resolve() == reconstructed_path.resolve()
        assert kwargs["observation_id"] == captured.bundle.view_id
        Path(path).mkdir()

    monkeypatch.setattr(module, "write_surface_coverage_generation", write_coverage)
    stored_mask = SimpleNamespace(
        result=SimpleNamespace(mask=mask_array),
        metadata={
            "identity": {
                "view_id": captured.bundle.view_id,
                "sequence_index": captured.bundle.sequence_index,
                "frame_number": captured.bundle.stereo.frame_number,
            }
        },
    )
    stored_view = SimpleNamespace(
        blade_mask=mask_array,
        view=SimpleNamespace(
            source_view_id=captured.bundle.view_id,
            source_sequence_index=captured.bundle.sequence_index,
            source_frame_number=captured.bundle.stereo.frame_number,
        ),
        metadata={
            "schema_version": 3,
            "source": {
                "session": str(captured.raw_session_path),
                "stereo_inference": str(stereo_root),
            },
        },
    )
    monkeypatch.setattr(module, "read_blade_foreground_mask", lambda path: stored_mask)
    monkeypatch.setattr(module, "read_reconstructed_view", lambda path: stored_view)
    stereo = SimpleNamespace(
        depth_m=np.full((3, 4), 0.5),
        rectified=SimpleNamespace(calibration=SimpleNamespace(left=intrinsics)),
    )

    pending = module.prepare_fine_science_assets(
        purpose=CapturePurpose.CANDIDATE,
        captured=captured,
        stereo=stereo,
        stereo_path=stereo_root,
        occupancy_update=_occupancy(OccupancyMapState.MAP_READY),
        occupancy_path=occupancy_root,
        settings=settings,
        hand_eye=SimpleNamespace(),
        reference_coarse_model=reference,
        accepted_coverage_path=accepted,
    )

    assert pending.blade_foreground_path == mask_path.resolve()
    assert pending.reconstructed_view_path == reconstructed_path.resolve()
    assert pending.coverage_path == successor_path.resolve()
    assert pending.advances_coverage is True
    assert read_calls >= 2
