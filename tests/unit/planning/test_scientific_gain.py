import json

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import NextViewSelectionConfig
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.planning.filtering import (
    CandidateMetrics,
    CandidateStatus,
    EvaluatedCandidate,
)
from biblade_fusion.planning.scientific_gain import expected_scientific_gain
from biblade_fusion.planning.surface_coverage import SurfacePatchQuality
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch


def _candidate(
    patch_id: str,
    *,
    visibility: float = 1.0,
    projection: float = 1.0,
    incidence: float = 1.0,
) -> EvaluatedCandidate:
    patch = SurfacePatch(
        patch_id,
        BladeSide.FRONT,
        0,
        0,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        (0.1, 0.1),
    )
    pose = PoseSE3.from_rotation_translation(
        "base",
        "left_ir",
        np.diag([1.0, -1.0, -1.0]),
        np.array([0.0, 0.0, 0.3]),
    )
    candidate = CandidateView(
        patch_id,
        patch,
        pose,
        0.3,
        (0.1, 0.1),
        projection,
        visibility,
        "unit_test",
    )
    metrics = CandidateMetrics(1.0, incidence, 1.0, 0.3, 0.0, 0.2, 1.0)
    return EvaluatedCandidate(
        candidate,
        CandidateStatus.ENDPOINT_FEASIBLE,
        metrics,
        (),
        np.zeros(6),
    )


def _quality(
    patch_id: str,
    region: SurfaceRegion,
    *,
    observed: int = 0,
    coverage: float = 0.0,
    rmse_m: float = float("inf"),
    normal_consistency: float = 0.0,
) -> SurfacePatchQuality:
    return SurfacePatchQuality(
        patch_id,
        BladeSide.FRONT,
        region,
        100,
        observed,
        coverage,
        rmse_m,
        normal_consistency,
        0.0,
        False,
        ("incomplete",),
    )


def _selection() -> NextViewSelectionConfig:
    return NextViewSelectionConfig(
        required_regions=("fin_face", "surface"),
        region_priority=("fin_face", "surface"),
        require_each_region_on_both_blade_sides=False,
        require_two_observed_fin_faces_per_side=False,
    )


def test_unobserved_fin_face_has_explicit_semantic_gain() -> None:
    selection = _selection()
    fin = expected_scientific_gain(
        _candidate("fin"),
        _quality("fin", SurfaceRegion.FIN_FACE),
        selection,
        maximum_rmse_m=0.003,
    )
    surface = expected_scientific_gain(
        _candidate("surface"),
        _quality("surface", SurfaceRegion.SURFACE),
        selection,
        maximum_rmse_m=0.003,
    )

    assert fin.fin_face_bonus > 0.0
    assert fin.semantic_priority > surface.semantic_priority
    assert fin.expected_gain > surface.expected_gain


def test_visibility_projection_and_incidence_reduce_expected_gain() -> None:
    selection = _selection()
    quality = _quality("surface", SurfaceRegion.SURFACE)
    clear = expected_scientific_gain(
        _candidate("surface"),
        quality,
        selection,
        maximum_rmse_m=0.003,
    )
    oblique = expected_scientific_gain(
        _candidate("surface", visibility=0.25, projection=0.5, incidence=0.5),
        quality,
        selection,
        maximum_rmse_m=0.003,
    )

    assert oblique.measurement_quality < clear.measurement_quality
    assert oblique.expected_gain < clear.expected_gain


def test_partial_bad_measurement_produces_quality_recovery_gain() -> None:
    gain = expected_scientific_gain(
        _candidate("surface"),
        _quality(
            "surface",
            SurfaceRegion.SURFACE,
            observed=50,
            coverage=0.5,
            rmse_m=0.004,
            normal_consistency=0.4,
        ),
        _selection(),
        maximum_rmse_m=0.003,
    )

    assert gain.coverage_novelty == 0.5
    assert gain.quality_recovery == 0.5
    json.dumps(gain.as_payload())
