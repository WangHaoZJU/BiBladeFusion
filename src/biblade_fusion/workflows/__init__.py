"""Auditable end-to-end project workflows."""

from biblade_fusion.workflows.coarse_model import (
    CoarseModelResult,
    build_coarse_blade_model,
    derive_consistent_left_rectified_t_left_ir,
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
    EndpointPoseConsistency,
    LiveJointSegmentPreflight,
    MotionSequenceCost,
    PreflightedMotionLeg,
    ViewSequenceMotionPreflight,
    evaluate_endpoint_pose_consistency,
    preflight_live_joint_segment,
    preflight_view_sequence_motion,
)
from biblade_fusion.workflows.native_overlap import (
    IcpCorrectionDiagnostic,
    NativeOverlapPairMetrics,
    NativeOverlapPairResult,
    NativeOverlapReport,
    NativeOverlapValidationError,
    evaluate_native_overlap,
)
from biblade_fusion.workflows.occupancy_mapping import (
    OccupancyFrameEvidence,
    OccupancyFrameUpdate,
    OccupancyMappingError,
    integrate_foundation_stereo_occupancy,
    mark_snapshot_stale_if_expired,
    occupancy_physical_source_id,
)
from biblade_fusion.workflows.path_validation import (
    PathSequenceError,
    ValidatedPathLeg,
    ViewSequenceCollisionReport,
    validate_view_sequence_collision,
)
from biblade_fusion.workflows.reconstruction import (
    AuthoritativeRobotPose,
    ReconstructedBladeView,
    ReconstructionError,
    reconstruct_foundation_stereo_view,
    reconstruct_native_depth_view,
    resolve_authoritative_robot_pose,
)
from biblade_fusion.workflows.stereo_inference import (
    StereoInferenceObservation,
    infer_rectified_stereo,
)
from biblade_fusion.workflows.view_planning import (
    AdaptiveViewPlanTrace,
    OfflineViewPlanningResult,
    plan_initial_observation,
)

__all__ = [
    "AdaptiveViewPlanTrace",
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
    "LiveJointSegmentPreflight",
    "EndpointPoseConsistency",
    "IcpCorrectionDiagnostic",
    "NativeOverlapPairMetrics",
    "NativeOverlapPairResult",
    "NativeOverlapReport",
    "NativeOverlapValidationError",
    "OccupancyFrameEvidence",
    "OccupancyFrameUpdate",
    "OccupancyMappingError",
    "occupancy_physical_source_id",
    "PathSequenceError",
    "PreflightedMotionLeg",
    "AuthoritativeRobotPose",
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
    "evaluate_native_overlap",
    "integrate_foundation_stereo_occupancy",
    "mark_snapshot_stale_if_expired",
    "plan_initial_observation",
    "preflight_view_sequence_motion",
    "preflight_live_joint_segment",
    "evaluate_endpoint_pose_consistency",
    "validate_view_sequence_collision",
    "reconstruct_foundation_stereo_view",
    "reconstruct_native_depth_view",
    "resolve_authoritative_robot_pose",
    "build_coarse_blade_model",
    "derive_consistent_left_rectified_t_left_ir",
    "registered_cloud_view",
]
