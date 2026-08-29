"""Workflow boundary for the unknown-blade foreground bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapForegroundError,
    BootstrapForegroundResult,
    BootstrapSeed,
    bootstrap_blade_foreground,
)
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


@dataclass(frozen=True, slots=True)
class BootstrapForegroundObservation:
    """Source-identified result ready for immutable persistence."""

    source_view_id: str
    source_sequence_index: int
    source_frame_number: int
    result: BootstrapForegroundResult

    def __post_init__(self) -> None:
        if not self.source_view_id:
            raise ValueError("Bootstrap foreground source view must be non-empty")
        if self.source_sequence_index < 0 or self.source_frame_number < 0:
            raise ValueError("Bootstrap foreground source indices must be non-negative")


def bootstrap_foundation_stereo_foreground(
    observation: StereoInferenceObservation,
    config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None = None,
) -> BootstrapForegroundObservation:
    """Create an initial mask from an official FoundationStereo observation.

    Requiring the backend identity prevents native depth or a test disparity source
    from silently entering the scientific reconstruction chain.
    """

    metadata = observation.result.metadata
    if metadata.get("backend") != "foundation_stereo":
        raise BootstrapForegroundError(
            "Bootstrap foreground requires a FoundationStereo depth source"
        )
    if metadata.get("runtime") != "official_nvidia_foundation_stereo":
        raise BootstrapForegroundError(
            "Bootstrap foreground requires the official NVIDIA FoundationStereo runtime"
        )
    result = bootstrap_blade_foreground(
        observation.rectified.left_ir,
        observation.depth_m,
        observation.result.valid_mask,
        config,
        seed,
    )
    return BootstrapForegroundObservation(
        source_view_id=observation.source_view_id,
        source_sequence_index=observation.source_sequence_index,
        source_frame_number=observation.rectified.source_frame_number,
        result=result,
    )
