"""View-balanced and pixel-pooled summaries for paired depth experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biblade_fusion.planning import BladeSide
from biblade_fusion.workflows.depth_evaluation import PairedDepthComparison


class DepthAggregationError(ValueError):
    """Paired comparison artifacts cannot form a rigorous aggregate."""


@dataclass(frozen=True, slots=True)
class LabeledDepthComparison:
    comparison: PairedDepthComparison
    side: BladeSide
    incidence_angle_deg: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.incidence_angle_deg) or not (
            0.0 <= self.incidence_angle_deg <= 90.0
        ):
            raise ValueError("Incidence angle must be finite and in [0, 90] degrees")


@dataclass(frozen=True, slots=True)
class DepthAggregateMetrics:
    view_count: int
    overlap_pixel_count: int
    mean_native_coverage_fraction: float
    mean_stereo_coverage_fraction: float
    mean_overlap_fraction: float
    view_mean_absolute_error_m: float
    pooled_signed_mean_error_m: float
    pooled_signed_median_error_m: float
    pooled_mean_absolute_error_m: float
    pooled_root_mean_square_error_m: float
    pooled_p95_absolute_error_m: float
    pooled_agreement_fractions: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class DepthAggregateGroup:
    group_id: str
    metrics: DepthAggregateMetrics


@dataclass(frozen=True, slots=True)
class DepthAggregateReport:
    groups: tuple[DepthAggregateGroup, ...]
    incidence_bin_edges_deg: tuple[float, ...]
    source_view_keys: tuple[str, ...]


def _source_key(item: LabeledDepthComparison) -> str:
    comparison = item.comparison
    return (
        f"{comparison.source_sequence_index}:"
        f"{comparison.source_frame_number}:{comparison.source_view_id}"
    )


def _aggregate(
    group_id: str,
    items: tuple[LabeledDepthComparison, ...],
) -> DepthAggregateGroup:
    errors = np.concatenate(
        [item.comparison.signed_error_m[item.comparison.comparison_mask] for item in items]
    ).astype(np.float64)
    absolute = np.abs(errors)
    thresholds = tuple(
        threshold
        for threshold, _ in items[0].comparison.metrics.agreement_fractions
    )
    for item in items[1:]:
        candidate = tuple(
            threshold for threshold, _ in item.comparison.metrics.agreement_fractions
        )
        if candidate != thresholds:
            raise DepthAggregationError(
                "All comparisons must use identical agreement thresholds"
            )
    metrics = DepthAggregateMetrics(
        view_count=len(items),
        overlap_pixel_count=len(errors),
        mean_native_coverage_fraction=float(
            np.mean([item.comparison.metrics.native_coverage_fraction for item in items])
        ),
        mean_stereo_coverage_fraction=float(
            np.mean([item.comparison.metrics.stereo_coverage_fraction for item in items])
        ),
        mean_overlap_fraction=float(
            np.mean([item.comparison.metrics.overlap_fraction for item in items])
        ),
        view_mean_absolute_error_m=float(
            np.mean([item.comparison.metrics.mean_absolute_error_m for item in items])
        ),
        pooled_signed_mean_error_m=float(np.mean(errors)),
        pooled_signed_median_error_m=float(np.median(errors)),
        pooled_mean_absolute_error_m=float(np.mean(absolute)),
        pooled_root_mean_square_error_m=float(np.sqrt(np.mean(np.square(errors)))),
        pooled_p95_absolute_error_m=float(np.percentile(absolute, 95)),
        pooled_agreement_fractions=tuple(
            (threshold, float(np.mean(absolute <= threshold)))
            for threshold in thresholds
        ),
    )
    return DepthAggregateGroup(group_id, metrics)


def aggregate_depth_comparisons(
    comparisons: tuple[LabeledDepthComparison, ...],
    incidence_bin_edges_deg: tuple[float, ...],
) -> DepthAggregateReport:
    """Aggregate all views while retaining side and incidence-angle strata."""

    if not comparisons:
        raise DepthAggregationError("At least one depth comparison is required")
    edges = tuple(float(value) for value in incidence_bin_edges_deg)
    if (
        len(edges) < 2
        or edges[0] != 0.0
        or edges[-1] != 90.0
        or not np.isfinite(edges).all()
        or any(first >= second for first, second in zip(edges, edges[1:], strict=False))
    ):
        raise DepthAggregationError(
            "Incidence bin edges must increase strictly from 0 to 90 degrees"
        )
    source_keys = tuple(_source_key(item) for item in comparisons)
    if len(set(source_keys)) != len(source_keys):
        raise DepthAggregationError("Aggregate contains a duplicate physical source view")

    groups = [_aggregate("all", comparisons)]
    for side in BladeSide:
        selected = tuple(item for item in comparisons if item.side is side)
        if selected:
            groups.append(_aggregate(f"side:{side.value}", selected))
    for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False)):
        selected = tuple(
            item
            for item in comparisons
            if item.incidence_angle_deg >= lower
            and (
                item.incidence_angle_deg < upper
                or (index == len(edges) - 2 and item.incidence_angle_deg <= upper)
            )
        )
        if selected:
            groups.append(_aggregate(f"incidence:[{lower:g},{upper:g}]deg", selected))
    return DepthAggregateReport(tuple(groups), edges, source_keys)
