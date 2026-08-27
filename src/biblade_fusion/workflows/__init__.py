"""Auditable end-to-end project workflows."""

from biblade_fusion.workflows.coarse_model import (
    CoarseModelResult,
    build_coarse_blade_model,
    registered_cloud_view,
)
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
    DepthViewGeometry,
    PairedDepthComparison,
    classify_depth_view_geometry,
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
from biblade_fusion.workflows.motion_preflight import (
    MotionSequenceCost,
    PreflightedMotionLeg,
    ViewSequenceMotionPreflight,
    preflight_view_sequence_motion,
)
from biblade_fusion.workflows.path_validation import (
    PathSequenceError,
    ValidatedPathLeg,
    ViewSequenceCollisionReport,
    validate_view_sequence_collision,
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
    "CoarseModelResult",
    "DepthComparisonMetrics",
    "DepthViewGeometry",
    "PairedDepthComparison",
    "compare_paired_depth",
    "classify_depth_view_geometry",
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
    "MotionSequenceCost",
    "PathSequenceError",
    "PreflightedMotionLeg",
    "ReconstructedBladeView",
    "ReconstructionError",
    "ValidatedPathLeg",
    "ViewSequenceCollisionReport",
    "ViewSequenceMotionPreflight",
    "StereoInferenceObservation",
    "initialize_native_depth",
    "initialize_foundation_stereo_depth",
    "extract_hand_eye_samples",
    "infer_rectified_stereo",
    "plan_initial_observation",
    "preflight_view_sequence_motion",
    "validate_view_sequence_collision",
    "reconstruct_foundation_stereo_view",
    "reconstruct_native_depth_view",
    "build_coarse_blade_model",
    "registered_cloud_view",
]
