from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import biblade_fusion.storage.science_authority as authority_module
from biblade_fusion.storage.science_acceptance import ScienceTestEnvelope
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority


def _authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ScienceAcceptanceAuthority:
    acceptance = (tmp_path / "science-acceptance").resolve()
    acceptance.mkdir()
    metadata = acceptance / "metadata.json"
    metadata.write_text('{"accepted":true}\n', encoding="utf-8")
    envelope = ScienceTestEnvelope(0.15, 1.5, 0.0, 85.0)
    acceptance_id = "a" * 64
    monkeypatch.setattr(
        authority_module,
        "read_science_acceptance",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            acceptance_id=acceptance_id,
            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            test_envelope=envelope,
        ),
    )
    identity = {
        "foundation_stereo_source_sha256": "1" * 64,
        "foundation_stereo_checkpoint_sha256": "2" * 64,
        "foundation_stereo_model_config_sha256": "3" * 64,
        "stereo_calibration_sha256": "4" * 64,
        "flange_primary_hand_eye_sha256": "5" * 64,
    }
    return ScienceAcceptanceAuthority(
        acceptance_path=acceptance,
        acceptance_id=acceptance_id,
        acceptance_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        runtime_contract_sha256="a" * 64,
        test_envelope=envelope,
        inference_identity=identity,
    )


def test_science_authority_round_trip_and_actual_inference_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path, monkeypatch)
    restored = ScienceAcceptanceAuthority.from_payload(authority.to_payload())
    observation = SimpleNamespace(
        result=SimpleNamespace(
            metadata={
                "source_sha256": "1" * 64,
                "checkpoint_sha256": "2" * 64,
                "model_config_sha256": "3" * 64,
            }
        )
    )

    assert restored == authority
    restored.assert_acceptance_asset_current()
    restored.assert_inference_observation(observation)
    observation.result.metadata["checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="output identity"):
        restored.assert_inference_observation(observation)


def test_science_authority_rejects_acceptance_metadata_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path, monkeypatch)
    metadata = authority.acceptance_path / "metadata.json"
    metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="acceptance asset changed"):
        authority.assert_acceptance_asset_current()
