"""Auditable end-to-end project workflows."""

from biblade_fusion.workflows.depth_aggregation import (
    DepthAggregateGroup,
    DepthAggregateMetrics,
    DepthAggregateReport,
    DepthAggregationError,
    LabeledDepthComparison,
    aggregate_depth_comparisons,
)
from biblade_fusion.workflows.depth_evaluation import (
    DepthComparisonError,
    DepthComparisonMetrics,
    PairedDepthComparison,
    compare_paired_depth,
)
from biblade_fusion.workflows.hand_eye_extraction import (
    HandEyeExtractionResult,
    extract_hand_eye_samples,
)
from biblade_fusion.workflows.initialization import (
    InitializationError,
    InitialObservation,
    initialize_foundation_stereo_depth,
    initialize_native_depth,
)
from biblade_fusion.workflows.reconstruction import (
    ReconstructedBladeView,
    ReconstructionError,
    reconstruct_foundation_stereo_view,
    reconstruct_native_depth_view,
)
from biblade_fusion.workflows.stereo_inference import (
    StereoInferenceObservation,
    infer_rectified_stereo,
)
from biblade_fusion.workflows.view_planning import (
    OfflineViewPlanningResult,
    plan_initial_observation,
)

__all__ = [
    "DepthComparisonError",
    "DepthComparisonMetrics",
    "PairedDepthComparison",
    "compare_paired_depth",
    "DepthAggregateGroup",
    "DepthAggregateMetrics",
    "DepthAggregateReport",
    "DepthAggregationError",
    "LabeledDepthComparison",
    "aggregate_depth_comparisons",
    "InitialObservation",
    "HandEyeExtractionResult",
    "InitializationError",
    "OfflineViewPlanningResult",
    "ReconstructedBladeView",
    "ReconstructionError",
    "StereoInferenceObservation",
    "initialize_native_depth",
    "initialize_foundation_stereo_depth",
    "extract_hand_eye_samples",
    "infer_rectified_stereo",
    "plan_initial_observation",
    "reconstruct_foundation_stereo_view",
    "reconstruct_native_depth_view",
]
