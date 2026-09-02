from __future__ import annotations

from pathlib import Path

import pytest

from biblade_fusion.core.settings import OccupancyConfig
from scripts import validate_cuda_ray_integration as validation


def test_latest_occupancy_root_selects_latest_committed_attempt_path(
    tmp_path: Path,
) -> None:
    first = (
        tmp_path
        / "perception/coarse/cycles/000000_view/attempt_a/occupancy_mapping"
    )
    latest = (
        tmp_path
        / "perception/coarse/cycles/000001_view/attempt_b/occupancy_mapping"
    )
    for root in (first, latest):
        root.mkdir(parents=True)
        (root / "metadata.json").write_text("{}\n", encoding="utf-8")

    assert validation._latest_occupancy_root(tmp_path) == latest.resolve()


def test_latest_occupancy_root_requires_evidence(tmp_path: Path) -> None:
    with pytest.raises(validation.CudaRayValidationError, match="No coarse occupancy"):
        validation._latest_occupancy_root(tmp_path)


def test_validation_backend_override_preserves_semantic_configuration() -> None:
    occupancy = OccupancyConfig(
        minimum_depth_m=0.2,
        maximum_depth_m=1.2,
        integration_stride=3,
        free_space_margin_m=0.02,
        minimum_free_observations=4,
        maximum_source_views=4,
        minimum_free_view_translation_m=0.03,
        minimum_free_view_direction_deg=6.0,
    )

    cpu = validation._integration_config(occupancy, "cpu")
    cuda = validation._integration_config(occupancy, "cuda")

    assert cpu.ray_integration_backend == "cpu"
    assert cuda.ray_integration_backend == "cuda"
    assert cpu.minimum_depth_m == cuda.minimum_depth_m == 0.2
    assert cpu.maximum_depth_m == cuda.maximum_depth_m == 1.2
    assert cpu.pixel_stride == cuda.pixel_stride == 3
    assert cpu.free_space_margin_m == cuda.free_space_margin_m == 0.02
    assert cpu.minimum_free_observations == cuda.minimum_free_observations == 4
