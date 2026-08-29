from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import biblade_fusion.storage.blade_foreground as blade_foreground_storage
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import BladeForegroundConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.blade_foreground import (
    REFERENCE_PROJECTED_ALGORITHM,
    BladeForegroundDiagnostics,
    BladeForegroundMaskResult,
)
from biblade_fusion.storage.blade_foreground import (
    read_blade_foreground_mask,
    write_blade_foreground_mask,
)
from biblade_fusion.workflows.occupancy_mapping import occupancy_array_content_hash

_REAL_REPLAY_SOURCE_RESULT = blade_foreground_storage._replay_source_result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(config: BladeForegroundConfig) -> str:
    payload = {
        "algorithm": REFERENCE_PROJECTED_ALGORITHM,
        "configuration": config.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _result() -> BladeForegroundMaskResult:
    config = BladeForegroundConfig(
        enabled=True,
        minimum_reference_pixels=1,
        minimum_target_reference_pixels=1,
        minimum_mask_pixels=1,
        minimum_target_mask_pixels=1,
    )
    eligible = np.array([[True, True, True], [True, False, True]], dtype=np.bool_)
    reference = np.array([[0.50, 0.51, 0.52], [0.53, np.nan, 0.55]], dtype=np.float64)
    target = np.array([[0.50, np.nan, np.nan], [0.53, np.nan, np.nan]], dtype=np.float64)
    mask = np.array([[True, True, False], [True, False, False]], dtype=np.bool_)
    diagnostics = BladeForegroundDiagnostics(
        target_patch_id="front-section-000",
        target_incidence_cosine=1.0,
        image_pixel_count=6,
        eligible_pixel_count=5,
        valid_eligible_depth_pixel_count=5,
        reference_pixel_count=5,
        eligible_reference_pixel_count=5,
        target_reference_pixel_count=2,
        eligible_target_reference_pixel_count=2,
        mask_pixel_count=3,
        target_mask_pixel_count=2,
        mask_fraction=0.5,
        reference_match_fraction=0.6,
        target_match_fraction=1.0,
    )
    return BladeForegroundMaskResult(
        mask,
        reference,
        target,
        eligible,
        diagnostics,
        config,
        REFERENCE_PROJECTED_ALGORITHM,
        _policy(config),
    )


def _sources(tmp_path: Path, result: BladeForegroundMaskResult) -> dict[str, Path]:
    session = tmp_path / "session"
    view_metadata = session / "views" / "0000_candidate" / "metadata.json"
    _write_json(
        view_metadata,
        {
            "view_id": "candidate",
            "sequence_index": 0,
            "stereo": {"frame_number": 7},
        },
    )
    _write_json(
        session / "manifest.json",
        {
            "schema_version": 3,
            "views": [
                {
                    "view_id": "candidate",
                    "sequence_index": 0,
                    "path": "views/0000_candidate",
                }
            ],
        },
    )

    intrinsics = _intrinsics()
    stereo = tmp_path / "stereo"
    _write_json(
        stereo / "metadata.json",
        {
            "schema_version": 2,
            "source": {
                "session": str(session.resolve()),
                "view_id": "candidate",
                "sequence_index": 0,
                "frame_number": 7,
            },
            "calibration": {
                "left": {
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                    "distortion_model": intrinsics.distortion_model,
                    "distortion_coefficients": [],
                }
            },
        },
    )

    occupancy = tmp_path / "occupancy"
    occupancy.mkdir()
    integration_path = occupancy / "0000_candidate_integration_valid_mask.npy"
    np.save(integration_path, result.eligible_mask, allow_pickle=False)
    integration_hash = occupancy_array_content_hash(result.eligible_mask)
    _write_json(
        occupancy / "metadata.json",
        {
            "schema_version": 6,
            "artifact_kind": "biblade_fusion.occupancy_mapping",
            "motion_authorized": False,
            "frames": [
                {
                    "evidence": {
                        "source_view_id": "candidate",
                        "source_sequence_index": 0,
                        "frame_number": 7,
                        "base_t_camera_matrix": np.eye(4).tolist(),
                        "source_stereo_metadata_sha256": _sha256(stereo / "metadata.json"),
                        "source_session_manifest_sha256": _sha256(session / "manifest.json"),
                        "source_session_view_metadata_sha256": _sha256(view_metadata),
                        "integration_valid_mask_content_hash": integration_hash,
                    },
                    "files": {
                        "integration_valid_mask": {
                            "path": integration_path.name,
                            "sha256": _sha256(integration_path),
                        }
                    },
                    "sources": {
                        "stereo_inference": {
                            "root": str(stereo.resolve()),
                            "file": "metadata.json",
                            "sha256": _sha256(stereo / "metadata.json"),
                        },
                        "session": {
                            "root": str(session.resolve()),
                            "file": "manifest.json",
                            "sha256": _sha256(session / "manifest.json"),
                        },
                    },
                }
            ],
        },
    )
    coarse = tmp_path / "coarse"
    _write_json(
        coarse / "metadata.json",
        {
            "schema_version": 5,
            "motion_authorized": False,
            "surface": {"patches": [{"patch_id": "front-section-000"}]},
        },
    )
    return {
        "session": session,
        "stereo": stereo,
        "occupancy": occupancy,
        "coarse": coarse,
    }


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(3, 2, 10.0, 10.0, 1.0, 0.5, "none", ())


@pytest.fixture(autouse=True)
def _replay_minimal_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compact source fixture models replay without an official FS checkout."""

    monkeypatch.setattr(
        blade_foreground_storage,
        "_replay_source_result",
        lambda *args, **kwargs: _result(),
    )


def _write_asset(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    result = _result()
    sources = _sources(tmp_path, result)
    output = write_blade_foreground_mask(
        tmp_path / "foreground",
        result,
        view_id="candidate",
        sequence_index=0,
        frame_number=7,
        base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
        intrinsics=_intrinsics(),
        source_session=sources["session"],
        source_stereo_inference=sources["stereo"],
        source_occupancy_mapping=sources["occupancy"],
        reference_coarse_model=sources["coarse"],
        source_integration_valid_mask_hash=occupancy_array_content_hash(result.eligible_mask),
        target_patch_id="front-section-000",
    )
    return output, sources


def test_blade_foreground_round_trip_binds_sources_and_arrays(tmp_path: Path) -> None:
    output, sources = _write_asset(tmp_path)

    stored = read_blade_foreground_mask(output)

    assert stored.root == output.resolve()
    assert stored.metadata["sources"]["session"]["root"] == str(sources["session"].resolve())
    assert stored.result.policy_sha256 == _result().policy_sha256
    np.testing.assert_array_equal(stored.result.mask, _result().mask)
    assert not stored.result.mask.flags.writeable


def test_blade_foreground_rejects_tampered_array(tmp_path: Path) -> None:
    output, _ = _write_asset(tmp_path)
    np.save(output / "mask.npy", np.zeros((2, 3), dtype=np.bool_), allow_pickle=False)

    with pytest.raises(ValueError, match="array checksum mismatch"):
        read_blade_foreground_mask(output)


def test_blade_foreground_rejects_tampered_source(tmp_path: Path) -> None:
    output, sources = _write_asset(tmp_path)
    metadata = json.loads((sources["coarse"] / "metadata.json").read_text())
    metadata["tampered"] = True
    _write_json(sources["coarse"] / "metadata.json", metadata)

    with pytest.raises(ValueError, match="reference coarse model source changed"):
        read_blade_foreground_mask(output)


def test_blade_foreground_rejects_tampered_integration_mask(tmp_path: Path) -> None:
    output, sources = _write_asset(tmp_path)
    np.save(
        sources["occupancy"] / "0000_candidate_integration_valid_mask.npy",
        np.ones((2, 3), dtype=np.bool_),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="integration-valid-mask file checksum"):
        read_blade_foreground_mask(output)


def test_blade_foreground_rejects_noncanonical_source_path(tmp_path: Path) -> None:
    output, _ = _write_asset(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    root = metadata["sources"]["session"]["root"]
    metadata["sources"]["session"]["root"] = f"{root}/../{Path(root).name}"
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="absolute and canonical"):
        read_blade_foreground_mask(output)


def test_blade_foreground_rejects_mask_outside_eligible() -> None:
    result = _result()
    invalid_mask = result.mask.copy()
    invalid_mask[1, 1] = True

    with pytest.raises(ValueError, match="subset"):
        BladeForegroundMaskResult(
            invalid_mask,
            result.reference_depth_m,
            result.target_reference_depth_m,
            result.eligible_mask,
            result.diagnostics,
            result.config,
            result.algorithm,
            result.policy_sha256,
        )


def test_blade_foreground_result_rejects_support_below_its_policy() -> None:
    result = _result()
    strict = result.config.model_copy(update={"minimum_mask_pixels": 4})

    with pytest.raises(ValueError, match="policy|minimum_mask_pixels|support"):
        replace(result, config=strict, policy_sha256=_policy(strict))


def test_blade_foreground_asset_rejects_tampered_threshold_policy(
    tmp_path: Path,
) -> None:
    output, _ = _write_asset(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    strict = BladeForegroundConfig.model_validate(
        {
            **metadata["processing"]["config"],
            "minimum_mask_pixels": 4,
        }
    )
    metadata["processing"]["config"] = strict.model_dump(mode="json")
    metadata["processing"]["policy_sha256"] = _policy(strict)
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="policy|minimum_mask_pixels|support"):
        read_blade_foreground_mask(output)


@pytest.mark.parametrize("changed_source", ["stereo_depth", "coarse_surface"])
def test_blade_foreground_rejects_replay_changed_by_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_source: str,
) -> None:
    output, _ = _write_asset(tmp_path)
    replayed = _result()
    if changed_source == "stereo_depth":
        changed_mask = replayed.mask.copy()
        changed_mask[0, 0] = False
        changed_mask[0, 2] = True
        changed = replace(replayed, mask=changed_mask)
    else:
        changed_reference = replayed.reference_depth_m.copy()
        changed_reference[0, 0] += 0.001
        changed = replace(replayed, reference_depth_m=changed_reference)
    monkeypatch.setattr(
        blade_foreground_storage,
        "_replay_source_result",
        lambda *args, **kwargs: changed,
    )

    with pytest.raises(ValueError, match="do not reproduce from bound sources"):
        read_blade_foreground_mask(output)


def test_real_replay_path_uses_typed_stereo_identity_and_all_source_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biblade_fusion.storage import stereo_inference as stereo_storage
    from biblade_fusion.storage import surface_coverage as coverage_storage

    result = _result()
    paths = _sources(tmp_path, result)
    records = {
        "session": blade_foreground_storage._directory_source_record(
            paths["session"], "manifest.json"
        ),
        "stereo_inference": blade_foreground_storage._directory_source_record(
            paths["stereo"], "metadata.json"
        ),
        "reference_coarse_model": blade_foreground_storage._directory_source_record(
            paths["coarse"], "metadata.json"
        ),
    }
    observation = SimpleNamespace(
        source_view_id="candidate",
        source_sequence_index=0,
        rectified=SimpleNamespace(
            source_frame_number=7,
            calibration=SimpleNamespace(left=_intrinsics()),
        ),
        depth_m=np.full((2, 3), 0.5, dtype=np.float32),
    )
    stored_stereo = SimpleNamespace(observation=observation)
    verified_sessions: list[Path] = []
    surface = object()

    monkeypatch.setattr(
        stereo_storage,
        "read_stereo_inference",
        lambda path: stored_stereo,
    )

    def verify_source(stored, *, expected_session):
        assert stored is stored_stereo
        verified_sessions.append(Path(expected_session).resolve())

    monkeypatch.setattr(
        stereo_storage,
        "verify_stereo_inference_source",
        verify_source,
    )
    monkeypatch.setattr(
        coverage_storage,
        "read_coarse_surface_reference",
        lambda path: surface,
    )

    def replay_mask(
        depth_m,
        eligible_mask,
        intrinsics,
        base_t_left_rectified,
        replay_surface,
        target_patch_id,
        config,
    ):
        np.testing.assert_array_equal(depth_m, observation.depth_m)
        np.testing.assert_array_equal(eligible_mask, result.eligible_mask)
        assert intrinsics == _intrinsics()
        np.testing.assert_allclose(
            base_t_left_rectified.matrix,
            PoseSE3.identity("base", "left_rectified").matrix,
        )
        assert replay_surface is surface
        assert target_patch_id == "front-section-000"
        assert config == result.config
        return result

    monkeypatch.setattr(
        blade_foreground_storage,
        "reference_guided_blade_mask",
        replay_mask,
    )

    replayed = _REAL_REPLAY_SOURCE_RESULT(
        records,
        view_id="candidate",
        sequence_index=0,
        frame_number=7,
        base_t_left_rectified=PoseSE3.identity("base", "left_rectified"),
        intrinsics=_intrinsics(),
        eligible_mask=result.eligible_mask,
        target_patch_id="front-section-000",
        config=result.config,
    )

    assert replayed is result
    assert verified_sessions == [paths["session"].resolve()]
