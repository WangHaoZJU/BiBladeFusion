import numpy as np
import pytest

from biblade_fusion.planning import BladeSide
from biblade_fusion.workflows import (
    DepthAggregationError,
    DepthComparisonMetrics,
    LabeledDepthComparison,
    PairedDepthComparison,
    aggregate_depth_comparisons,
)


def comparison(view_id: str, frame: int, errors: list[float]) -> PairedDepthComparison:
    error = np.asarray(errors, dtype=np.float32).reshape(1, -1)
    mask = np.ones_like(error, dtype=bool)
    native = np.full_like(error, 0.5)
    absolute = np.abs(error)
    metrics = DepthComparisonMetrics(
        blade_pixel_count=len(errors),
        native_valid_pixel_count=len(errors),
        stereo_valid_pixel_count=len(errors),
        overlap_pixel_count=len(errors),
        native_coverage_fraction=1.0,
        stereo_coverage_fraction=1.0,
        overlap_fraction=1.0,
        signed_mean_error_m=float(np.mean(error)),
        signed_median_error_m=float(np.median(error)),
        mean_absolute_error_m=float(np.mean(absolute)),
        root_mean_square_error_m=float(np.sqrt(np.mean(np.square(error)))),
        p95_absolute_error_m=float(np.percentile(absolute, 95)),
        median_stereo_to_native_ratio=1.0,
        agreement_fractions=((0.01, float(np.mean(absolute <= 0.01))),),
    )
    return PairedDepthComparison(view_id, frame, frame, native, mask, error, metrics)


def test_aggregate_retains_side_and_incidence_groups() -> None:
    report = aggregate_depth_comparisons(
        (
            LabeledDepthComparison(
                comparison("front", 1, [0.0, 0.01]), BladeSide.FRONT, 5.0
            ),
            LabeledDepthComparison(
                comparison("back", 2, [0.02]), BladeSide.BACK, 25.0
            ),
        ),
        (0.0, 15.0, 30.0, 90.0),
    )

    groups = {group.group_id: group.metrics for group in report.groups}
    assert groups["all"].view_count == 2
    assert groups["all"].overlap_pixel_count == 3
    assert groups["all"].pooled_mean_absolute_error_m == pytest.approx(0.01)
    assert groups["side:front"].view_count == 1
    assert groups["side:back"].view_count == 1
    assert groups["incidence:[0,15]deg"].view_count == 1
    assert groups["incidence:[15,30]deg"].view_count == 1


def test_aggregate_rejects_duplicate_physical_view() -> None:
    item = LabeledDepthComparison(
        comparison("front", 1, [0.0]), BladeSide.FRONT, 0.0
    )

    with pytest.raises(DepthAggregationError, match="duplicate"):
        aggregate_depth_comparisons((item, item), (0.0, 90.0))
