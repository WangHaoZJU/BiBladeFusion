import json

import pytest

from biblade_fusion.core.settings import OccupancyConfig
from biblade_fusion.storage.motion_preflight import (
    MOTION_PREFLIGHT_SCHEMA_VERSION,
    _offline_occupancy_configuration_matches,
    read_motion_preflight,
)


def test_motion_preflight_schema_five_rejects_legacy_schema_four(tmp_path) -> None:
    artifact = tmp_path / "legacy_motion_preflight"
    artifact.mkdir()
    (artifact / "motion_preflight.json").write_text(
        json.dumps({"schema_version": 4}),
        encoding="utf-8",
    )

    assert MOTION_PREFLIGHT_SCHEMA_VERSION == 5
    with pytest.raises(ValueError, match="unsupported schema 4"):
        read_motion_preflight(artifact)


def test_offline_preflight_allows_only_the_occupancy_enabled_flag_to_differ() -> None:
    active = OccupancyConfig(enabled=False)
    replay = active.model_copy(update={"enabled": True})

    assert _offline_occupancy_configuration_matches(active, replay)
    assert not _offline_occupancy_configuration_matches(
        active,
        replay.model_copy(update={"voxel_size_m": 0.02}),
    )
    assert not _offline_occupancy_configuration_matches(active, active)
