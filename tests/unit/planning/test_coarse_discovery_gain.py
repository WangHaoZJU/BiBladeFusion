import json

import numpy as np

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CoverageConfig
from biblade_fusion.planning.coarse_discovery_gain import (
    expected_coarse_discovery_gain,
)
from biblade_fusion.planning.coverage import CoverageLedger, PatchCoverage
from biblade_fusion.planning.filtering import (
    CandidateMetrics,
    CandidateStatus,
    EvaluatedCandidate,
)
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch


def _coverage() -> CoverageLedger:
    config = CoverageConfig(bins_per_axis=2, minimum_points_per_bin=1)
    return CoverageLedger(
        (
            PatchCoverage(
                "front_patch",
                BladeSide.FRONT,
                0,
                0,
                np.ones((2, 2), dtype=np.int64),
            ),
            PatchCoverage(
                "back_patch",
                BladeSide.BACK,
                0,
                0,
                np.zeros((2, 2), dtype=np.int64),
            ),
        ),
        (),
        config,
        1,
        1,
    )


def _candidate(view_id: str, side: BladeSide) -> EvaluatedCandidate:
    normal = np.array([0.0, 0.0, 1.0 if side is BladeSide.FRONT else -1.0])
    pose = PoseSE3.from_rotation_translation(
        "base",
        "left_ir",
        np.diag([1.0, -1.0, -1.0]),
        0.3 * normal,
    )
    patch = SurfacePatch(view_id, side, 0, 0, np.zeros(3), normal, (0.1, 0.1))
    candidate = CandidateView(
        view_id,
        patch,
        pose,
        0.3,
        (0.1, 0.1),
        1.0,
        1.0,
        "unit_test",
    )
    return EvaluatedCandidate(
        candidate,
        CandidateStatus.ENDPOINT_FEASIBLE,
        CandidateMetrics(1.0, 1.0, 1.0, 0.3, 0.0, 0.2, 1.0),
        (),
        np.zeros(6),
    )


def _gain(candidate: EvaluatedCandidate, *, side_count: int, fin: float):
    return expected_coarse_discovery_gain(
        candidate,
        _coverage(),
        side_observation_count=side_count,
        minimum_views_per_side=3,
        fin_pair_evidence=fin,
        surface_weight=0.45,
        side_balance_weight=0.25,
        fin_pair_weight=0.30,
    )


def test_unseen_back_side_has_more_coarse_gain_than_covered_front() -> None:
    front = _gain(_candidate("front_candidate", BladeSide.FRONT), side_count=1, fin=0.0)
    back = _gain(_candidate("back_candidate", BladeSide.BACK), side_count=0, fin=0.0)

    assert front.proxy_coverage_deficit == 0.0
    assert back.proxy_coverage_deficit == 1.0
    assert back.side_observation_deficit == 1.0
    assert back.expected_gain > front.expected_gain


def test_opposite_fin_member_completion_has_more_gain_than_pair_seed() -> None:
    candidate = _candidate("back_fin", BladeSide.BACK)
    seed = _gain(candidate, side_count=1, fin=0.6)
    completion = _gain(candidate, side_count=1, fin=1.0)

    assert completion.expected_gain > seed.expected_gain
    json.dumps(completion.as_payload())
