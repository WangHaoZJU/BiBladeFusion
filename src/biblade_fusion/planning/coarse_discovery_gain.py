"""Single-initial-view coarse discovery gain inside the blade proxy ROI."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from biblade_fusion.planning.coverage import CoverageLedger
from biblade_fusion.planning.filtering import EvaluatedCandidate


@dataclass(frozen=True, slots=True)
class CoarseDiscoveryGain:
    """Auditable proxy-stage expected gain for one endpoint-feasible candidate."""

    view_id: str
    patch_id: str
    side: str
    proxy_coverage_deficit: float
    side_observation_deficit: float
    fin_pair_evidence: float
    predicted_visibility: float
    predicted_projection: float
    incidence_quality: float
    measurement_quality: float
    expected_gain: float

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _proxy_coverage_deficit(
    candidate: EvaluatedCandidate,
    coverage: CoverageLedger,
) -> float:
    patch_id = candidate.candidate.patch.patch_id
    direct = tuple(item for item in coverage.patches if item.patch_id == patch_id)
    selected = direct or tuple(
        item
        for item in coverage.patches
        if item.side is candidate.candidate.patch.side
    )
    if not selected:
        raise ValueError("Coarse gain has no proxy coverage evidence for candidate side")
    observed = tuple(
        item.occupied_fraction(coverage.config.minimum_points_per_bin)
        for item in selected
    )
    return 1.0 - float(np.mean(observed))


def expected_coarse_discovery_gain(
    candidate: EvaluatedCandidate,
    coverage: CoverageLedger,
    *,
    side_observation_count: int,
    minimum_views_per_side: int,
    fin_pair_evidence: float,
    surface_weight: float,
    side_balance_weight: float,
    fin_pair_weight: float,
) -> CoarseDiscoveryGain:
    """Estimate useful proxy/fin evidence without counting background unknown voxels."""

    weights = (surface_weight, side_balance_weight, fin_pair_weight)
    if (
        side_observation_count < 0
        or minimum_views_per_side < 1
        or not np.isfinite((*weights, fin_pair_evidence)).all()
        or any(not 0.0 <= value <= 1.0 for value in weights)
        or not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-9)
        or not 0.0 <= fin_pair_evidence <= 1.0
    ):
        raise ValueError("Coarse discovery gain inputs are invalid")
    proxy_deficit = _proxy_coverage_deficit(candidate, coverage)
    side_deficit = 1.0 - min(1.0, side_observation_count / minimum_views_per_side)
    visibility = float(np.clip(candidate.candidate.visibility_fraction, 0.0, 1.0))
    projection = float(np.clip(candidate.candidate.projection_fraction, 0.0, 1.0))
    incidence = float(np.clip(candidate.metrics.incidence_cosine, 0.0, 1.0))
    measurement_quality = float(np.cbrt(visibility * projection * incidence))
    expected_gain = measurement_quality * (
        surface_weight * proxy_deficit
        + side_balance_weight * side_deficit
        + fin_pair_weight * fin_pair_evidence
    )
    return CoarseDiscoveryGain(
        candidate.candidate.view_id,
        candidate.candidate.patch.patch_id,
        candidate.candidate.patch.side.value,
        proxy_deficit,
        side_deficit,
        fin_pair_evidence,
        visibility,
        projection,
        incidence,
        measurement_quality,
        float(expected_gain),
    )
