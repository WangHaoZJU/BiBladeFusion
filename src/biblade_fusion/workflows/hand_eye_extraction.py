"""Build hand-eye samples from already synchronized stored observations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import (
    CharucoDetectionError,
    CharucoTargetDetector,
    HandEyeSample,
    HandEyeSampleRejection,
)
from biblade_fusion.core.settings import CharucoTargetConfig
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp


@dataclass(frozen=True, slots=True)
class HandEyeExtractionResult:
    samples: tuple[HandEyeSample, ...]
    rejected: tuple[HandEyeSampleRejection, ...]


def extract_hand_eye_samples(
    observations: Sequence[tuple[str | Path, SynchronizedFrameBundle]],
    target_config: CharucoTargetConfig,
    kinematics: Es68KinematicModel | None = None,
) -> HandEyeExtractionResult:
    """Pair left-IR ChArUco observations with calibrated ES68 flange FK poses."""

    model = kinematics or Es68KinematicModel.from_resources()
    flange_t_tcp = load_es68_flange_t_tcp()
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
            base_t_flange = model.base_t_flange(bundle.selected_robot_state.joint_positions_rad)
            predicted_tcp = base_t_flange.compose(flange_t_tcp)
            observed_tcp = bundle.selected_robot_state.base_t_tcp
            tcp_rotation_delta = predicted_tcp.rotation.T @ observed_tcp.rotation
            rotation_cosine = np.clip((np.trace(tcp_rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
            samples.append(
                HandEyeSample(
                    sample_id=sample_id,
                    base_t_flange=base_t_flange,
                    left_ir_t_target=detection.left_ir_t_target,
                    source_session=session,
                    charuco_corner_count=len(detection.charuco_ids),
                    reprojection_rmse_px=detection.reprojection_rmse_px,
                    pose_ambiguity_ratio=detection.pose_ambiguity_ratio,
                    joint_positions_rad=bundle.selected_robot_state.joint_positions_rad,
                    base_t_tcp_observed=bundle.selected_robot_state.base_t_tcp,
                    charuco_ids=detection.charuco_ids,
                    image_points_px=detection.image_points_px,
                    object_points_m=detection.object_points_m,
                    frame_number=bundle.stereo.frame_number,
                    bracket_ms=bundle.metrics.bracket_ms,
                    selected_robot_state_offset_ms=(bundle.metrics.selected_robot_state_offset_ms),
                    controller_time_s=bundle.selected_robot_state.controller_time_s,
                    robot_mode=bundle.selected_robot_state.robot_mode,
                    safety_status=bundle.selected_robot_state.safety_status,
                    fk_tcp_translation_error_m=float(
                        np.linalg.norm(predicted_tcp.translation_m - observed_tcp.translation_m)
                    ),
                    fk_tcp_rotation_error_deg=float(np.degrees(np.arccos(rotation_cosine))),
                )
            )
        except CharucoDetectionError as exc:
            rejected.append(HandEyeSampleRejection(sample_id, str(exc)))
    return HandEyeExtractionResult(tuple(samples), tuple(rejected))
