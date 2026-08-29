"""Strict runtime authority derived from one immutable science acceptance."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biblade_fusion.storage.science_acceptance import (
    ScienceTestEnvelope,
    read_science_acceptance,
    required_science_test_envelope_for_settings,
    science_runtime_contract_for_settings,
    science_runtime_contract_payload,
)

if TYPE_CHECKING:
    from biblade_fusion.core.settings import AppSettings
    from biblade_fusion.storage.stereo_inference import StoredStereoInference
    from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation

SCIENCE_AUTHORITY_SCHEMA_VERSION = 1
_SHA_KEYS = (
    "foundation_stereo_source_sha256",
    "foundation_stereo_checkpoint_sha256",
    "foundation_stereo_model_config_sha256",
    "stereo_calibration_sha256",
    "flange_primary_hand_eye_sha256",
)
_SOURCE_LABELS = {
    "foundation_stereo_source": "foundation_stereo_source_sha256",
    "foundation_stereo_checkpoint": "foundation_stereo_checkpoint_sha256",
    "foundation_stereo_model_config": "foundation_stereo_model_config_sha256",
    "stereo_calibration": "stereo_calibration_sha256",
    "flange_primary_hand_eye": "flange_primary_hand_eye_sha256",
}


def _digest(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inference_identity(settings: AppSettings) -> dict[str, str]:
    payload = science_runtime_contract_payload(settings)
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("science runtime contract lacks its source records")
    found: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("science runtime source record is not an object")
        key = _SOURCE_LABELS.get(str(source.get("label")))
        if key is not None:
            found[key] = _digest(source.get("sha256"), label=key)
    if set(found) != set(_SHA_KEYS):
        raise ValueError("science runtime contract lacks inference identity sources")
    return found


def _test_envelope_payload(value: ScienceTestEnvelope) -> dict[str, float]:
    return {
        "minimum_distance_m": value.minimum_distance_m,
        "maximum_distance_m": value.maximum_distance_m,
        "minimum_incidence_deg": value.minimum_incidence_deg,
        "maximum_incidence_deg": value.maximum_incidence_deg,
    }


def _test_envelope_from_payload(payload: object) -> ScienceTestEnvelope:
    if not isinstance(payload, Mapping) or set(payload) != {
        "minimum_distance_m",
        "maximum_distance_m",
        "minimum_incidence_deg",
        "maximum_incidence_deg",
    }:
        raise ValueError("science authority test envelope fields changed")
    return ScienceTestEnvelope(
        minimum_distance_m=float(payload["minimum_distance_m"]),
        maximum_distance_m=float(payload["maximum_distance_m"]),
        minimum_incidence_deg=float(payload["minimum_incidence_deg"]),
        maximum_incidence_deg=float(payload["maximum_incidence_deg"]),
    )


@dataclass(frozen=True, slots=True)
class ScienceAcceptanceAuthority:
    """Exact accepted runtime, physical envelope, and inference-source identity."""

    acceptance_path: Path
    acceptance_id: str
    acceptance_metadata_sha256: str
    runtime_contract_sha256: str
    test_envelope: ScienceTestEnvelope
    inference_identity: dict[str, str]

    def __post_init__(self) -> None:
        raw = Path(self.acceptance_path)
        resolved = raw.resolve()
        if not raw.is_absolute() or raw != resolved:
            raise ValueError("science authority acceptance path must be absolute and canonical")
        object.__setattr__(self, "acceptance_path", resolved)
        for label, value in (
            ("acceptance_id", self.acceptance_id),
            ("acceptance_metadata_sha256", self.acceptance_metadata_sha256),
            ("runtime_contract_sha256", self.runtime_contract_sha256),
        ):
            _digest(value, label=label)
        if set(self.inference_identity) != set(_SHA_KEYS):
            raise ValueError("science authority inference identity fields changed")
        object.__setattr__(
            self,
            "inference_identity",
            {
                key: _digest(self.inference_identity[key], label=key)
                for key in _SHA_KEYS
            },
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENCE_AUTHORITY_SCHEMA_VERSION,
            "acceptance_path": str(self.acceptance_path),
            "acceptance_id": self.acceptance_id,
            "acceptance_metadata_sha256": self.acceptance_metadata_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "test_envelope": _test_envelope_payload(self.test_envelope),
            "inference_identity": dict(self.inference_identity),
        }

    @classmethod
    def from_payload(cls, payload: object) -> ScienceAcceptanceAuthority:
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "acceptance_path",
            "acceptance_id",
            "acceptance_metadata_sha256",
            "runtime_contract_sha256",
            "test_envelope",
            "inference_identity",
        }:
            raise ValueError("science authority fields changed")
        if payload["schema_version"] != SCIENCE_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("science authority schema changed")
        identity = payload["inference_identity"]
        if not isinstance(identity, Mapping):
            raise ValueError("science authority inference identity is not an object")
        return cls(
            acceptance_path=Path(str(payload["acceptance_path"])),
            acceptance_id=str(payload["acceptance_id"]),
            acceptance_metadata_sha256=str(payload["acceptance_metadata_sha256"]),
            runtime_contract_sha256=str(payload["runtime_contract_sha256"]),
            test_envelope=_test_envelope_from_payload(payload["test_envelope"]),
            inference_identity={str(key): str(value) for key, value in identity.items()},
        )

    def assert_acceptance_asset_current(self) -> None:
        stored = read_science_acceptance(self.acceptance_path)
        if (
            stored.path != self.acceptance_path
            or stored.acceptance_id != self.acceptance_id
            or stored.metadata_sha256 != self.acceptance_metadata_sha256
            or stored.test_envelope != self.test_envelope
        ):
            raise ValueError("science acceptance asset changed after authority creation")

    def assert_current(self, settings: AppSettings) -> None:
        configured_path = settings.science_acceptance.path
        configured_id = settings.science_acceptance.acceptance_id
        if configured_path is None or configured_id is None:
            raise ValueError("science acceptance configuration is absent")
        if Path(configured_path).resolve() != self.acceptance_path:
            raise ValueError("science acceptance path changed during the run")
        required = required_science_test_envelope_for_settings(settings)
        stored = read_science_acceptance(self.acceptance_path)
        current_contract = science_runtime_contract_for_settings(settings)
        stored.assert_matches(
            acceptance_id=configured_id,
            runtime_contract_sha256=current_contract,
            required_test_envelope=required,
        )
        if (
            stored.acceptance_id != self.acceptance_id
            or stored.metadata_sha256 != self.acceptance_metadata_sha256
            or stored.test_envelope != self.test_envelope
            or current_contract != self.runtime_contract_sha256
            or _inference_identity(settings) != self.inference_identity
        ):
            raise ValueError("science runtime authority changed during the run")

    def assert_inference_observation(
        self,
        observation: StereoInferenceObservation,
    ) -> None:
        metadata = observation.result.metadata
        expected = {
            "source_sha256": self.inference_identity[
                "foundation_stereo_source_sha256"
            ],
            "checkpoint_sha256": self.inference_identity[
                "foundation_stereo_checkpoint_sha256"
            ],
            "model_config_sha256": self.inference_identity[
                "foundation_stereo_model_config_sha256"
            ],
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "FoundationStereo output identity differs from science authority"
            )

    def assert_stereo_artifact(self, stored: StoredStereoInference) -> None:
        self.assert_inference_observation(stored.observation)
        source = stored.metadata.get("source")
        calibration = (
            source.get("stereo_calibration_asset")
            if isinstance(source, Mapping)
            else None
        )
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("sha256")
            != self.inference_identity["stereo_calibration_sha256"]
        ):
            raise ValueError("stereo inference calibration differs from science authority")
        path = Path(str(calibration.get("path")))
        if (
            not path.is_absolute()
            or path != path.resolve()
            or not path.is_file()
            or _sha256(path) != calibration.get("sha256")
        ):
            raise ValueError("stereo inference calibration authority is no longer current")


def load_science_acceptance_authority(settings: AppSettings) -> ScienceAcceptanceAuthority:
    path = settings.science_acceptance.path
    acceptance_id = settings.science_acceptance.acceptance_id
    if path is None or acceptance_id is None:
        raise ValueError("science acceptance path and identity are required")
    stored = read_science_acceptance(path)
    runtime_contract = science_runtime_contract_for_settings(settings)
    stored.assert_matches(
        acceptance_id=acceptance_id,
        runtime_contract_sha256=runtime_contract,
        required_test_envelope=required_science_test_envelope_for_settings(settings),
    )
    authority = ScienceAcceptanceAuthority(
        acceptance_path=stored.path,
        acceptance_id=stored.acceptance_id,
        acceptance_metadata_sha256=stored.metadata_sha256,
        runtime_contract_sha256=runtime_contract,
        test_envelope=stored.test_envelope,
        inference_identity=_inference_identity(settings),
    )
    authority.assert_current(settings)
    return authority


__all__ = [
    "SCIENCE_AUTHORITY_SCHEMA_VERSION",
    "ScienceAcceptanceAuthority",
    "load_science_acceptance_authority",
]
