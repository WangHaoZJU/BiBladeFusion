"""Blade-ROI expected scientific gain for online next-best-view selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from biblade_fusion.core.settings import NextViewSelectionConfig, ScientificGainConfig
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.planning.filtering import EvaluatedCandidate
from biblade_fusion.planning.surface_coverage import SurfacePatchQuality


@dataclass(frozen=True, slots=True)
class ExpectedScientificGain:
    """Normalized, reproducible gain components for one feasible blade view."""

    view_id: str
    patch_id: str
    region: SurfaceRegion
    coverage_deficit: float
    quality_deficit: float
    predicted_visibility: float
    predicted_projection: float
    incidence_quality: float
    measurement_quality: float
    coverage_novelty: float
    quality_recovery: float
    semantic_priority: float
    fin_face_bonus: float
    expected_gain: float

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["region"] = self.region.value
        return payload


def _quality_deficit(
    quality: SurfacePatchQuality,
    maximum_rmse_m: float,
) -> float:
    normal_deficit = 1.0 - float(np.clip(quality.normal_consistency, 0.0, 1.0))
    rmse_deficit = (
        1.0
        if not np.isfinite(quality.rmse_m)
        else float(np.clip(quality.rmse_m / maximum_rmse_m, 0.0, 1.0))
    )
    return max(normal_deficit, rmse_deficit)


def _semantic_priority(
    region: SurfaceRegion,
    selection: NextViewSelectionConfig,
    gain: ScientificGainConfig,
) -> float:
    priorities = tuple(SurfaceRegion(value) for value in selection.region_priority)
    index = priorities.index(region)
    normalized = (
        1.0 if len(priorities) == 1 else 1.0 - index / (len(priorities) - 1)
    )
    return 1.0 + gain.region_priority_multiplier * normalized


def expected_scientific_gain(
    candidate: EvaluatedCandidate,
    quality: SurfacePatchQuality,
    selection: NextViewSelectionConfig,
    *,
    maximum_rmse_m: float,
) -> ExpectedScientificGain:
    """Estimate useful new blade evidence; safety occupancy is intentionally absent."""

    if candidate.candidate.patch.patch_id != quality.patch_id:
        raise ValueError("Scientific gain candidate and quality patch identities differ")
    if maximum_rmse_m <= 0.0 or not np.isfinite(maximum_rmse_m):
        raise ValueError("Scientific gain requires a finite positive RMSE scale")

    policy = selection.scientific_gain
    coverage = float(np.clip(quality.coverage_fraction, 0.0, 1.0))
    coverage_deficit = 1.0 - coverage
    quality_deficit = _quality_deficit(quality, maximum_rmse_m)
    visibility = float(np.clip(candidate.candidate.visibility_fraction, 0.0, 1.0))
    projection = float(np.clip(candidate.candidate.projection_fraction, 0.0, 1.0))
    incidence = float(np.clip(candidate.metrics.incidence_cosine, 0.0, 1.0))
    measurement_quality = float(np.cbrt(visibility * projection * incidence))
    coverage_novelty = coverage_deficit * measurement_quality
    quality_recovery = coverage * quality_deficit * measurement_quality
    semantic_priority = _semantic_priority(quality.region, selection, policy)
    fin_face_bonus = (
        policy.unobserved_fin_face_bonus * measurement_quality
        if quality.region is SurfaceRegion.FIN_FACE and quality.observed_point_count == 0
        else 0.0
    )
    expected_gain = (
        semantic_priority
        * (
            policy.coverage_weight * coverage_novelty
            + policy.quality_recovery_weight * quality_recovery
        )
        + fin_face_bonus
    )
    if not policy.enabled:
        expected_gain = 0.0
    return ExpectedScientificGain(
        candidate.candidate.view_id,
        quality.patch_id,
        quality.region,
        coverage_deficit,
        quality_deficit,
        visibility,
        projection,
        incidence,
        measurement_quality,
        coverage_novelty,
        quality_recovery,
        semantic_priority,
        fin_face_bonus,
        float(expected_gain),
    )
