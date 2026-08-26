"""Build hand-eye samples from already synchronized stored observations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import (
    CharucoDetectionError,
    CharucoTargetDetector,
    HandEyeSample,
    HandEyeSampleRejection,
)
from biblade_fusion.core.settings import CharucoTargetConfig


@dataclass(frozen=True, slots=True)
class HandEyeExtractionResult:
    samples: tuple[HandEyeSample, ...]
    rejected: tuple[HandEyeSampleRejection, ...]


def extract_hand_eye_samples(
    observations: Sequence[tuple[str | Path, SynchronizedFrameBundle]],
    target_config: CharucoTargetConfig,
) -> HandEyeExtractionResult:
    """Detect the target per view and pair it with the synchronized selected robot pose."""

    samples: list[HandEyeSample] = []
    rejected: list[HandEyeSampleRejection] = []
    for source_session, bundle in observations:
        session = str(Path(source_session).resolve())
        session_key = hashlib.sha256(session.encode("utf-8")).hexdigest()[:8]
        sample_id = (
            f"{Path(source_session).name}-{session_key}:"
            f"{bundle.sequence_index:04d}:{bundle.view_id}"
        )
        try:
            detection = CharucoTargetDetector(
                target_config,
                bundle.stereo.calibration.left,
            ).detect(bundle.stereo.left_ir)
            samples.append(
                HandEyeSample(
                    sample_id,
                    bundle.selected_robot_state.base_t_tcp,
                    detection.left_ir_t_target,
                    session,
                    len(detection.charuco_ids),
                    detection.reprojection_rmse_px,
                    detection.pose_ambiguity_ratio,
                )
            )
        except CharucoDetectionError as exc:
            rejected.append(HandEyeSampleRejection(sample_id, str(exc)))
    return HandEyeExtractionResult(tuple(samples), tuple(rejected))
