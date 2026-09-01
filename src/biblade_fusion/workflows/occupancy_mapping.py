"""Stop-and-capture safety occupancy updates from FoundationStereo depth."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AcquisitionConfig, OccupancyConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping import (
    DepthIntegrationConfig,
    DepthRayIntegrator,
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancySnapshot,
)
from biblade_fusion.mapping.self_mask import (
    RobotSelfMaskConfig,
    RobotSelfMaskReport,
    RobotSelfMaskResult,
    depth_consistent_robot_self_mask,
)
from biblade_fusion.robotics.es68_model import (
    Es68ModelResources,
    load_es68_flange_t_tcp,
)
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation


class OccupancyMappingError(ValueError):
    """A synchronized depth observation cannot become safety-map evidence."""


class RobotDepthRenderer(Protocol):
    model_content_hash: str
    self_mask_excluded_link_names: tuple[str, ...]
    self_mask_render_backend: str
    joint_zero_offsets_rad: tuple[float, ...]

    def base_t_flange_matrix(
        self,
        joint_positions_rad: tuple[float, ...] | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def render_robot_depth(
        self,
        intrinsics: CameraIntrinsics,
        joint_positions_rad: tuple[float, ...] | NDArray[np.float64],
        base_t_camera: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MAPPING_CONTEXT_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class OccupancyMappingContext:
    """Canonical contract shared by every observation in one occupancy map."""

    canonical_json: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Mapping context must be canonical JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Mapping context payload must be an object")
        if payload.get("schema_version") != MAPPING_CONTEXT_SCHEMA_VERSION:
            raise ValueError("Unsupported mapping context schema")
        canonical = _canonical_json(dict(payload))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.content_hash:
            if self.content_hash != expected:
                raise ValueError("Mapping context content_hash mismatch")
        else:
            object.__setattr__(self, "content_hash", expected)
        object.__setattr__(self, "canonical_json", canonical)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> OccupancyMappingContext:
        return cls(_canonical_json(dict(payload)))

    def to_payload(self) -> dict[str, object]:
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # defensive; validated in __post_init__
            raise RuntimeError("Mapping context payload is no longer an object")
        return payload


@dataclass(frozen=True, slots=True)
class OccupancyFrameEvidence:
    source_view_id: str
    physical_source_id: str
    source_sequence_index: int
    frame_number: int
    captured_at_utc: str
    robot_model_hash: str
    hand_eye_hash: str
    source_stereo_metadata_sha256: str
    source_session_manifest_sha256: str
    source_session_view_metadata_sha256: str
    base_t_flange_matrix: tuple[tuple[float, float, float, float], ...]
    predicted_base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...]
    observed_base_t_tcp_matrix: tuple[tuple[float, float, float, float], ...]
    base_t_camera_matrix: tuple[tuple[float, float, float, float], ...]
    joint_positions_rad: tuple[float, float, float, float, float, float]
    fk_tcp_translation_error_m: float
    fk_tcp_rotation_error_deg: float
    valid_depth_fraction: float
    stereo_valid_fraction: float
    confidence_accepted_fraction: float
    mean_accepted_confidence: float
    lr_consistency_threshold_px: float
    self_mask: RobotSelfMaskReport
    mapping_context_hash: str
    previous_evidence_hash: str | None
    mapping_snapshot_content_hash: str
    mapping_snapshot_sequence: int
    mapping_source_view_ids: tuple[str, ...]
    source_depth_content_hash: str
    stereo_valid_mask_content_hash: str
    stereo_confidence_content_hash: str
    predicted_robot_depth_content_hash: str
    robot_mask_content_hash: str
    integration_valid_mask_content_hash: str
    quality_evidence_hash: str = ""

    def __post_init__(self) -> None:
        source_view_id = str(self.source_view_id).strip()
        if not source_view_id:
            raise ValueError("Occupancy evidence source_view_id must be non-empty")
        physical_source_id = str(self.physical_source_id).strip()
        if _SHA256_PATTERN.fullmatch(physical_source_id) is None:
            raise ValueError("Occupancy evidence physical_source_id must be a SHA-256 digest")
        if (
            isinstance(self.source_sequence_index, bool)
            or not isinstance(self.source_sequence_index, (int, np.integer))
            or self.source_sequence_index < 0
            or isinstance(self.frame_number, bool)
            or not isinstance(self.frame_number, (int, np.integer))
            or self.frame_number < 0
        ):
            raise ValueError("Occupancy evidence source indices must be non-negative")
        captured = datetime.fromisoformat(str(self.captured_at_utc))
        if captured.tzinfo is None:
            raise ValueError("Occupancy evidence captured_at_utc must be timezone-aware")
        captured_text = captured.astimezone(UTC).isoformat()
        for name, digest in (
            ("robot_model_hash", self.robot_model_hash),
            ("hand_eye_hash", self.hand_eye_hash),
            ("source_stereo_metadata_sha256", self.source_stereo_metadata_sha256),
            ("source_session_manifest_sha256", self.source_session_manifest_sha256),
            (
                "source_session_view_metadata_sha256",
                self.source_session_view_metadata_sha256,
            ),
            ("mapping_context_hash", self.mapping_context_hash),
            ("previous_evidence_hash", self.previous_evidence_hash),
            ("mapping_snapshot_content_hash", self.mapping_snapshot_content_hash),
            ("source_depth_content_hash", self.source_depth_content_hash),
            ("stereo_valid_mask_content_hash", self.stereo_valid_mask_content_hash),
            ("stereo_confidence_content_hash", self.stereo_confidence_content_hash),
            (
                "predicted_robot_depth_content_hash",
                self.predicted_robot_depth_content_hash,
            ),
            ("robot_mask_content_hash", self.robot_mask_content_hash),
            (
                "integration_valid_mask_content_hash",
                self.integration_valid_mask_content_hash,
            ),
        ):
            if digest is not None and _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"Occupancy evidence {name} must be a SHA-256 digest")
        matrices = {
            "base_t_flange_matrix": _validated_transform_matrix(
                self.base_t_flange_matrix,
                name="Occupancy evidence base_t_flange_matrix",
            ),
            "predicted_base_t_tcp_matrix": _validated_transform_matrix(
                self.predicted_base_t_tcp_matrix,
                name="Occupancy evidence predicted_base_t_tcp_matrix",
            ),
            "observed_base_t_tcp_matrix": _validated_transform_matrix(
                self.observed_base_t_tcp_matrix,
                name="Occupancy evidence observed_base_t_tcp_matrix",
            ),
            "base_t_camera_matrix": _validated_transform_matrix(
                self.base_t_camera_matrix,
                name="Occupancy evidence base_t_camera_matrix",
            ),
        }
        joints = tuple(float(value) for value in self.joint_positions_rad)
        if len(joints) != 6 or not np.isfinite(joints).all():
            raise ValueError("Occupancy evidence joint_positions_rad must be a finite six-vector")
        pose_errors = (
            float(self.fk_tcp_translation_error_m),
            float(self.fk_tcp_rotation_error_deg),
        )
        if not np.isfinite(pose_errors).all() or any(value < 0.0 for value in pose_errors):
            raise ValueError("Occupancy FK/TCP errors must be finite and non-negative")
        reproduced_errors = (
            float(
                np.linalg.norm(
                    matrices["predicted_base_t_tcp_matrix"][:3, 3]
                    - matrices["observed_base_t_tcp_matrix"][:3, 3]
                )
            ),
            _rotation_error_deg(
                matrices["predicted_base_t_tcp_matrix"][:3, :3],
                matrices["observed_base_t_tcp_matrix"][:3, :3],
            ),
        )
        if not np.allclose(pose_errors, reproduced_errors, rtol=0.0, atol=1e-12):
            raise ValueError("Occupancy FK/TCP errors do not reproduce from pose evidence")
        fractions = (
            self.valid_depth_fraction,
            self.stereo_valid_fraction,
            self.confidence_accepted_fraction,
            self.mean_accepted_confidence,
        )
        if not np.isfinite(fractions).all() or any(
            value < 0.0 or value > 1.0 for value in fractions
        ):
            raise ValueError("Occupancy evidence quality fractions must lie in [0, 1]")
        threshold = float(self.lr_consistency_threshold_px)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Occupancy evidence LR consistency threshold must be positive")
        source_views = tuple(str(value).strip() for value in self.mapping_source_view_ids)
        if (
            not source_views
            or len(set(source_views)) != len(source_views)
            or source_views[-1] != physical_source_id
        ):
            raise ValueError("Occupancy evidence mapping_source_view_ids are invalid")
        expected_physical_source_id = occupancy_physical_source_id(
            source_session_manifest_sha256=self.source_session_manifest_sha256,
            source_session_view_metadata_sha256=self.source_session_view_metadata_sha256,
            source_sequence_index=int(self.source_sequence_index),
            frame_number=int(self.frame_number),
            source_view_id=source_view_id,
        )
        if physical_source_id != expected_physical_source_id:
            raise ValueError("Occupancy evidence physical_source_id does not reproduce")
        if len(source_views) == 1 and self.previous_evidence_hash is not None:
            raise ValueError("First occupancy evidence must not have a parent")
        if len(source_views) > 1 and self.previous_evidence_hash is None:
            raise ValueError("Multi-view occupancy evidence requires a parent hash")
        if (
            isinstance(self.mapping_snapshot_sequence, bool)
            or not isinstance(self.mapping_snapshot_sequence, (int, np.integer))
            or self.mapping_snapshot_sequence <= 0
        ):
            raise ValueError("Occupancy evidence mapping snapshot sequence must be positive")

        object.__setattr__(self, "source_view_id", source_view_id)
        object.__setattr__(self, "physical_source_id", physical_source_id)
        object.__setattr__(self, "source_sequence_index", int(self.source_sequence_index))
        object.__setattr__(self, "frame_number", int(self.frame_number))
        object.__setattr__(self, "captured_at_utc", captured_text)
        for name, matrix in matrices.items():
            object.__setattr__(
                self,
                name,
                tuple(tuple(float(value) for value in row) for row in matrix),
            )
        object.__setattr__(self, "joint_positions_rad", joints)
        object.__setattr__(self, "fk_tcp_translation_error_m", pose_errors[0])
        object.__setattr__(self, "fk_tcp_rotation_error_deg", pose_errors[1])
        object.__setattr__(self, "mapping_source_view_ids", source_views)
        object.__setattr__(
            self,
            "mapping_snapshot_sequence",
            int(self.mapping_snapshot_sequence),
        )
        object.__setattr__(self, "lr_consistency_threshold_px", threshold)
        expected = compute_occupancy_evidence_hash(self)
        if self.quality_evidence_hash:
            if self.quality_evidence_hash != expected:
                raise ValueError("Occupancy frame quality_evidence_hash mismatch")
        else:
            object.__setattr__(self, "quality_evidence_hash", expected)


@dataclass(frozen=True, slots=True)
class OccupancyFrameUpdate:
    snapshot: OccupancySnapshot
    mapping_snapshot: OccupancySnapshot
    mapping_context: OccupancyMappingContext
    source_depth_m: NDArray[np.float64]
    stereo_valid_mask: NDArray[np.bool_]
    stereo_confidence: NDArray[np.float64]
    predicted_robot_depth_m: NDArray[np.float64]
    robot_mask: NDArray[np.bool_]
    integration_valid_mask: NDArray[np.bool_]
    evidence: OccupancyFrameEvidence

    def __post_init__(self) -> None:
        arrays = {
            "source_depth_m": np.float64,
            "stereo_valid_mask": np.bool_,
            "stereo_confidence": np.float64,
            "predicted_robot_depth_m": np.float64,
            "robot_mask": np.bool_,
            "integration_valid_mask": np.bool_,
        }
        for name, dtype in arrays.items():
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        depth = self.source_depth_m
        stereo_valid = self.stereo_valid_mask
        confidence = self.stereo_confidence
        predicted = self.predicted_robot_depth_m
        robot_mask = self.robot_mask
        integration_mask = self.integration_valid_mask
        if (
            depth.ndim != 2
            or stereo_valid.shape != depth.shape
            or confidence.shape != depth.shape
            or predicted.shape != depth.shape
            or robot_mask.shape != depth.shape
            or integration_mask.shape != depth.shape
        ):
            raise ValueError("Occupancy frame depth, confidence and masks must share one HxW shape")
        if np.isfinite(depth[~stereo_valid]).any():
            raise ValueError("Stereo-invalid occupancy depth pixels must be non-finite")
        if not np.isfinite(depth[stereo_valid]).all() or np.any(depth[stereo_valid] <= 0.0):
            raise ValueError("Stereo-valid occupancy depth pixels must be positive")
        if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("Occupancy stereo confidence must be finite in [0, 1]")
        if (
            np.any(np.isnan(predicted))
            or np.any(np.isneginf(predicted))
            or np.any(np.isfinite(predicted) & (predicted <= 0.0))
        ):
            raise ValueError("Predicted robot depth must contain positive metres or +inf")
        if np.any(robot_mask & integration_mask):
            raise ValueError("Robot self mask and integration valid mask must not overlap")

        try:
            occupancy_contract = OccupancyConfig.model_validate(
                self.mapping_context.to_payload()["occupancy_contract"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Mapping context has no valid occupancy contract") from exc
        metadata = self.mapping_context.to_payload().get("foundation_stereo")
        if not isinstance(metadata, Mapping):
            raise ValueError("Mapping context has no FoundationStereo contract")
        threshold = metadata.get("left_right_consistency_threshold_px")
        if (
            metadata.get("backend") != "foundation_stereo"
            or metadata.get("left_right_consistency_applied") is not True
            or metadata.get("confidence_semantic")
            != "exp_negative_left_right_disparity_error_not_calibrated_probability"
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isclose(float(threshold), self.evidence.lr_consistency_threshold_px)
            or float(threshold) > occupancy_contract.maximum_lr_consistency_error_px
        ):
            raise ValueError("Occupancy update has an invalid FoundationStereo quality contract")
        accepted = stereo_valid & (confidence >= occupancy_contract.minimum_stereo_confidence)
        range_valid = (
            accepted
            & np.isfinite(depth)
            & (depth >= occupancy_contract.minimum_depth_m)
            & (depth <= occupancy_contract.maximum_depth_m)
        )
        recomputed = depth_consistent_robot_self_mask(
            depth,
            predicted,
            valid_mask=range_valid,
            config=RobotSelfMaskConfig(
                front_tolerance_m=occupancy_contract.self_mask_front_tolerance_m,
                back_tolerance_m=occupancy_contract.self_mask_back_tolerance_m,
                dilation_px=occupancy_contract.self_mask_dilation_px,
            ),
        )
        if not np.array_equal(robot_mask, recomputed.robot_mask) or not np.array_equal(
            integration_mask,
            recomputed.integration_valid_mask,
        ):
            raise ValueError("Occupancy self-mask arrays do not reproduce from stored evidence")
        if self.evidence.self_mask != recomputed.report:
            raise ValueError("Occupancy self-mask report does not reproduce from stored evidence")
        pixel_count = depth.size
        expected_metrics = (
            float(np.count_nonzero(integration_mask) / pixel_count),
            float(np.count_nonzero(stereo_valid) / pixel_count),
            float(np.count_nonzero(accepted) / pixel_count),
            float(np.mean(confidence[accepted])) if np.any(accepted) else 0.0,
        )
        recorded_metrics = (
            self.evidence.valid_depth_fraction,
            self.evidence.stereo_valid_fraction,
            self.evidence.confidence_accepted_fraction,
            self.evidence.mean_accepted_confidence,
        )
        if not np.allclose(expected_metrics, recorded_metrics, rtol=0.0, atol=1e-12):
            raise ValueError("Occupancy quality metrics do not reproduce from stored evidence")
        if expected_metrics[0] < occupancy_contract.minimum_valid_depth_fraction:
            raise ValueError("Occupancy valid depth fraction is below its quality contract")
        array_hashes = (
            occupancy_array_content_hash(depth),
            occupancy_array_content_hash(stereo_valid),
            occupancy_array_content_hash(confidence),
            occupancy_array_content_hash(predicted),
            occupancy_array_content_hash(robot_mask),
            occupancy_array_content_hash(integration_mask),
        )
        recorded_hashes = (
            self.evidence.source_depth_content_hash,
            self.evidence.stereo_valid_mask_content_hash,
            self.evidence.stereo_confidence_content_hash,
            self.evidence.predicted_robot_depth_content_hash,
            self.evidence.robot_mask_content_hash,
            self.evidence.integration_valid_mask_content_hash,
        )
        if array_hashes != recorded_hashes:
            raise ValueError("Occupancy array content hashes do not match stored evidence")
        if self.mapping_context.content_hash != self.evidence.mapping_context_hash:
            raise ValueError("Occupancy update evidence does not match its mapping context")
        if self.mapping_snapshot.content_hash != self.evidence.mapping_snapshot_content_hash:
            raise ValueError("Occupancy update evidence does not match its MAPPING snapshot")
        if self.mapping_snapshot.sequence != self.evidence.mapping_snapshot_sequence:
            raise ValueError("Occupancy update evidence has the wrong MAPPING sequence")
        if self.mapping_snapshot.map_state is not OccupancyMapState.MAPPING:
            raise ValueError("Occupancy update mapping_snapshot must be MAPPING")
        if self.mapping_snapshot.quality_evidence_hash is not None:
            raise ValueError("Intermediate MAPPING snapshot must not yet bind current evidence")
        if self.mapping_snapshot.parent_evidence_hash != self.evidence.previous_evidence_hash:
            raise ValueError("Occupancy update has a broken parent evidence link")
        if self.mapping_snapshot.mapping_context_hash != self.mapping_context.content_hash:
            raise ValueError("Intermediate snapshot has the wrong mapping context")
        if self.mapping_snapshot.source_view_ids != self.evidence.mapping_source_view_ids:
            raise ValueError("Occupancy update source-view evidence is inconsistent")
        evidence_pose = np.asarray(self.evidence.base_t_camera_matrix, dtype=np.float64)
        if not np.allclose(
            self.mapping_snapshot.source_camera_centres_base_m[-1],
            evidence_pose[:3, 3],
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            self.mapping_snapshot.source_camera_axes_base[-1],
            evidence_pose[:3, 2],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Intermediate snapshot camera-view evidence does not match base_T_camera"
            )
        if self.snapshot.quality_evidence_hash != self.evidence.quality_evidence_hash:
            raise ValueError("Result snapshot does not bind current frame evidence")
        if self.snapshot.mapping_context_hash != self.mapping_context.content_hash:
            raise ValueError("Result snapshot has the wrong mapping context")
        if (
            self.snapshot.free_indices != self.mapping_snapshot.free_indices
            or self.snapshot.free_observation_counts
            != self.mapping_snapshot.free_observation_counts
            or self.snapshot.minimum_free_observations
            != self.mapping_snapshot.minimum_free_observations
            or self.snapshot.minimum_free_view_translation_m
            != self.mapping_snapshot.minimum_free_view_translation_m
            or self.snapshot.minimum_free_view_direction_deg
            != self.mapping_snapshot.minimum_free_view_direction_deg
            or self.snapshot.occupied_indices != self.mapping_snapshot.occupied_indices
            or self.snapshot.source_view_ids != self.mapping_snapshot.source_view_ids
            or self.snapshot.source_camera_centres_base_m
            != self.mapping_snapshot.source_camera_centres_base_m
            or self.snapshot.source_camera_axes_base
            != self.mapping_snapshot.source_camera_axes_base
            or self.snapshot.rebuild_started_at_utc != self.mapping_snapshot.rebuild_started_at_utc
            or self.snapshot.parent_evidence_hash != self.mapping_snapshot.parent_evidence_hash
            or self.snapshot.created_at_utc != self.mapping_snapshot.created_at_utc
        ):
            raise ValueError("Result snapshot changed authoritative MAPPING geometry")
        expected_delta = 2 if self.snapshot.map_state is OccupancyMapState.STALE else 1
        if self.snapshot.sequence != self.mapping_snapshot.sequence + expected_delta:
            raise ValueError("Result snapshot lifecycle sequence is inconsistent")


@dataclass(frozen=True, slots=True)
class PreparedOccupancyFrame:
    """Current-frame safety inputs prepared before any occupancy ray integration."""

    bundle: SynchronizedFrameBundle
    stereo: StereoInferenceObservation
    captured_at_utc: datetime
    grid: OccupancyGridSpec
    mapping_context: OccupancyMappingContext
    predicted_base_t_flange: PoseSE3
    predicted_base_t_tcp: PoseSE3
    observed_base_t_tcp: PoseSE3
    base_t_camera: PoseSE3
    hand_eye_hash: str
    physical_source_id: str
    source_stereo_metadata_sha256: str
    source_session_manifest_sha256: str
    source_session_view_metadata_sha256: str
    fk_tcp_translation_error_m: float
    fk_tcp_rotation_error_deg: float
    valid_depth_fraction: float
    stereo_valid_fraction: float
    confidence_accepted_fraction: float
    mean_accepted_confidence: float
    lr_consistency_threshold_px: float
    source_depth_m: NDArray[np.float64]
    stereo_valid_mask: NDArray[np.bool_]
    stereo_confidence: NDArray[np.float64]
    predicted_robot_depth_m: NDArray[np.float64]
    self_mask: RobotSelfMaskResult

    def __post_init__(self) -> None:
        arrays = {
            "source_depth_m": np.float64,
            "stereo_valid_mask": np.bool_,
            "stereo_confidence": np.float64,
            "predicted_robot_depth_m": np.float64,
        }
        for name, dtype in arrays.items():
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        shape = self.source_depth_m.shape
        if (
            self.source_depth_m.ndim != 2
            or self.stereo_valid_mask.shape != shape
            or self.stereo_confidence.shape != shape
            or self.predicted_robot_depth_m.shape != shape
            or self.self_mask.integration_valid_mask.shape != shape
        ):
            raise ValueError("Prepared occupancy arrays must share one HxW shape")


def occupancy_frame_evidence_payload(evidence: OccupancyFrameEvidence) -> dict[str, object]:
    """Return the canonical, independently reproducible per-frame evidence payload."""

    return {
        "source_view_id": evidence.source_view_id,
        "physical_source_id": evidence.physical_source_id,
        "source_sequence_index": evidence.source_sequence_index,
        "frame_number": evidence.frame_number,
        "captured_at_utc": evidence.captured_at_utc,
        "robot_model_hash": evidence.robot_model_hash,
        "hand_eye_hash": evidence.hand_eye_hash,
        "source_stereo_metadata_sha256": evidence.source_stereo_metadata_sha256,
        "source_session_manifest_sha256": evidence.source_session_manifest_sha256,
        "source_session_view_metadata_sha256": (evidence.source_session_view_metadata_sha256),
        "base_t_flange_matrix": [list(row) for row in evidence.base_t_flange_matrix],
        "predicted_base_t_tcp_matrix": [list(row) for row in evidence.predicted_base_t_tcp_matrix],
        "observed_base_t_tcp_matrix": [list(row) for row in evidence.observed_base_t_tcp_matrix],
        "base_t_camera_matrix": [list(row) for row in evidence.base_t_camera_matrix],
        "joint_positions_rad": list(evidence.joint_positions_rad),
        "fk_tcp_translation_error_m": evidence.fk_tcp_translation_error_m,
        "fk_tcp_rotation_error_deg": evidence.fk_tcp_rotation_error_deg,
        "valid_depth_fraction": evidence.valid_depth_fraction,
        "stereo_valid_fraction": evidence.stereo_valid_fraction,
        "confidence_accepted_fraction": evidence.confidence_accepted_fraction,
        "mean_accepted_confidence": evidence.mean_accepted_confidence,
        "lr_consistency_threshold_px": evidence.lr_consistency_threshold_px,
        "self_mask": asdict(evidence.self_mask),
        "mapping_context_hash": evidence.mapping_context_hash,
        "previous_evidence_hash": evidence.previous_evidence_hash,
        "mapping_snapshot_content_hash": evidence.mapping_snapshot_content_hash,
        "mapping_snapshot_sequence": evidence.mapping_snapshot_sequence,
        "mapping_source_view_ids": list(evidence.mapping_source_view_ids),
        "source_depth_content_hash": evidence.source_depth_content_hash,
        "stereo_valid_mask_content_hash": evidence.stereo_valid_mask_content_hash,
        "stereo_confidence_content_hash": evidence.stereo_confidence_content_hash,
        "predicted_robot_depth_content_hash": (evidence.predicted_robot_depth_content_hash),
        "robot_mask_content_hash": evidence.robot_mask_content_hash,
        "integration_valid_mask_content_hash": (evidence.integration_valid_mask_content_hash),
    }


def compute_occupancy_evidence_hash(evidence: OccupancyFrameEvidence) -> str:
    return hashlib.sha256(
        _canonical_json(occupancy_frame_evidence_payload(evidence)).encode("utf-8")
    ).hexdigest()


def occupancy_physical_source_id(
    *,
    source_session_manifest_sha256: str,
    source_session_view_metadata_sha256: str,
    source_sequence_index: int,
    frame_number: int,
    source_view_id: str,
) -> str:
    """Identify one immutable raw observation independently of its UI label."""

    logical_id = str(source_view_id).strip()
    if not logical_id:
        raise ValueError("Occupancy physical source logical view ID must be non-empty")
    for name, digest in (
        ("source_session_manifest_sha256", source_session_manifest_sha256),
        ("source_session_view_metadata_sha256", source_session_view_metadata_sha256),
    ):
        if _SHA256_PATTERN.fullmatch(str(digest)) is None:
            raise ValueError(f"Occupancy physical source {name} must be a SHA-256 digest")
    for name, value in (
        ("source_sequence_index", source_sequence_index),
        ("frame_number", frame_number),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"Occupancy physical source {name} must be non-negative")
    payload = {
        "schema_version": 1,
        "source_session_manifest_sha256": str(source_session_manifest_sha256),
        "source_session_view_metadata_sha256": str(source_session_view_metadata_sha256),
        "source_sequence_index": int(source_sequence_index),
        "frame_number": int(frame_number),
        "source_view_id": logical_id,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def occupancy_array_content_hash(array: NDArray[np.generic]) -> str:
    """Hash an array's exact dtype, shape and C-order bytes deterministically."""

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise ValueError("Occupancy evidence arrays must not contain Python objects")
    contiguous = np.ascontiguousarray(value)
    descriptor = _canonical_json(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def prepare_foundation_stereo_occupancy_frame(
    bundle: SynchronizedFrameBundle,
    stereo: StereoInferenceObservation,
    hand_eye: HandEyeCalibration,
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    renderer: RobotDepthRenderer,
    *,
    captured_at_utc: datetime,
    source_stereo_metadata_sha256: str,
    source_session_manifest_sha256: str,
    source_session_view_metadata_sha256: str,
) -> PreparedOccupancyFrame:
    """Prepare exact current-frame masks without performing expensive ray integration."""

    if not occupancy_config.enabled:
        raise OccupancyMappingError("Occupancy mapping is disabled in the active configuration")
    if occupancy_config.mapping_mode != "stop_and_capture":
        raise OccupancyMappingError("Only stop_and_capture occupancy mapping is supported")
    _validate_source_identity(bundle, stereo)
    _validate_settled_capture(bundle, acquisition_config)
    captured = _utc(captured_at_utc)
    bounds_min = occupancy_config.workspace_bounds_min_m
    bounds_max = occupancy_config.workspace_bounds_max_m
    if bounds_min is None or bounds_max is None:
        raise OccupancyMappingError("Measured occupancy workspace bounds are missing")
    extents = np.asarray(bounds_max, dtype=np.float64) - np.asarray(bounds_min, dtype=np.float64)
    shape = tuple(
        int(value) for value in np.ceil(extents / occupancy_config.voxel_size_m).astype(np.int64)
    )
    grid = OccupancyGridSpec(
        frame_id=occupancy_config.frame_id,
        voxel_size_m=occupancy_config.voxel_size_m,
        origin_m=bounds_min,
        grid_shape=shape,
    )
    rectified_calibration = stereo.rectified.calibration
    joints = tuple(float(value) for value in bundle.selected_robot_state.joint_positions_rad)
    es68_resources = Es68ModelResources.packaged()
    flange_t_tcp = load_es68_flange_t_tcp(es68_resources)
    if hand_eye.flange_t_left_ir is None:
        raise OccupancyMappingError(
            "Safety occupancy requires schema-2 flange-primary hand-eye calibration"
        )
    predicted_base_t_flange = PoseSE3(
        "base",
        "flange",
        renderer.base_t_flange_matrix(joints),
    )
    predicted_base_t_tcp = predicted_base_t_flange.compose(flange_t_tcp)
    observed_base_t_tcp = bundle.selected_robot_state.base_t_tcp
    fk_tcp_translation_error_m = float(
        np.linalg.norm(predicted_base_t_tcp.translation_m - observed_base_t_tcp.translation_m)
    )
    fk_tcp_rotation_error_deg = _rotation_error_deg(
        predicted_base_t_tcp.rotation,
        observed_base_t_tcp.rotation,
    )
    pose_violations = []
    if fk_tcp_translation_error_m > occupancy_config.maximum_fk_tcp_translation_error_m:
        pose_violations.append("translation")
    if fk_tcp_rotation_error_deg > occupancy_config.maximum_fk_tcp_rotation_error_deg:
        pose_violations.append("rotation")
    if pose_violations:
        raise OccupancyMappingError(
            "ES68 FK and observed controller TCP disagree for safety mapping: "
            + ", ".join(pose_violations)
            + f" (translation={fk_tcp_translation_error_m:.6f} m, "
            f"rotation={fk_tcp_rotation_error_deg:.6f} deg)"
        )
    base_t_left_ir = predicted_base_t_flange.compose(hand_eye.flange_t_left_ir)
    base_t_camera = base_t_left_ir.compose(rectified_calibration.left_rectified_t_left_ir.inverse())
    hand_eye_hash = _file_sha256(hand_eye.source_path)
    context = _build_mapping_context(
        grid,
        occupancy_config,
        acquisition_config,
        stereo,
        hand_eye_hash=hand_eye_hash,
        flange_t_left_ir=hand_eye.flange_t_left_ir.matrix,
        robot_model_hash=renderer.model_content_hash,
        self_mask_excluded_link_names=renderer.self_mask_excluded_link_names,
        self_mask_render_backend=renderer.self_mask_render_backend,
        joint_zero_offsets_rad=renderer.joint_zero_offsets_rad,
        flange_t_tcp=flange_t_tcp.matrix,
        flange_tcp_asset_hash=_file_sha256(es68_resources.tcp_offset_json),
    )
    predicted_depth = renderer.render_robot_depth(
        rectified_calibration.left,
        bundle.selected_robot_state.joint_positions_rad,
        base_t_camera.matrix,
    )
    depth = np.asarray(stereo.depth_m, dtype=np.float64)
    (
        quality_valid,
        stereo_valid_fraction,
        confidence_accepted_fraction,
        mean_accepted_confidence,
        lr_threshold_px,
    ) = _validated_stereo_quality_mask(stereo, occupancy_config)
    range_valid = (
        quality_valid
        & np.isfinite(depth)
        & (depth >= occupancy_config.minimum_depth_m)
        & (depth <= occupancy_config.maximum_depth_m)
    )
    self_mask = depth_consistent_robot_self_mask(
        depth,
        predicted_depth,
        valid_mask=range_valid,
        config=RobotSelfMaskConfig(
            front_tolerance_m=occupancy_config.self_mask_front_tolerance_m,
            back_tolerance_m=occupancy_config.self_mask_back_tolerance_m,
            dilation_px=occupancy_config.self_mask_dilation_px,
        ),
    )
    valid_fraction = float(np.count_nonzero(self_mask.integration_valid_mask) / depth.size)
    if valid_fraction < occupancy_config.minimum_valid_depth_fraction:
        raise OccupancyMappingError(
            "Post-self-mask valid depth fraction is below the configured gate "
            f"({valid_fraction:.6f} < {occupancy_config.minimum_valid_depth_fraction:.6f})"
        )
    physical_source_id = occupancy_physical_source_id(
        source_session_manifest_sha256=source_session_manifest_sha256,
        source_session_view_metadata_sha256=source_session_view_metadata_sha256,
        source_sequence_index=bundle.sequence_index,
        frame_number=stereo.rectified.source_frame_number,
        source_view_id=bundle.view_id,
    )
    return PreparedOccupancyFrame(
        bundle=bundle,
        stereo=stereo,
        captured_at_utc=captured,
        grid=grid,
        mapping_context=context,
        predicted_base_t_flange=predicted_base_t_flange,
        predicted_base_t_tcp=predicted_base_t_tcp,
        observed_base_t_tcp=observed_base_t_tcp,
        base_t_camera=base_t_camera,
        hand_eye_hash=hand_eye_hash,
        physical_source_id=physical_source_id,
        source_stereo_metadata_sha256=source_stereo_metadata_sha256,
        source_session_manifest_sha256=source_session_manifest_sha256,
        source_session_view_metadata_sha256=source_session_view_metadata_sha256,
        fk_tcp_translation_error_m=fk_tcp_translation_error_m,
        fk_tcp_rotation_error_deg=fk_tcp_rotation_error_deg,
        valid_depth_fraction=valid_fraction,
        stereo_valid_fraction=stereo_valid_fraction,
        confidence_accepted_fraction=confidence_accepted_fraction,
        mean_accepted_confidence=mean_accepted_confidence,
        lr_consistency_threshold_px=lr_threshold_px,
        source_depth_m=depth,
        stereo_valid_mask=np.asarray(stereo.result.valid_mask, dtype=np.bool_),
        stereo_confidence=np.asarray(stereo.result.confidence, dtype=np.float64),
        predicted_robot_depth_m=predicted_depth,
        self_mask=self_mask,
    )


def integrate_prepared_foundation_stereo_occupancy(
    previous_snapshot: OccupancySnapshot | None,
    prepared: PreparedOccupancyFrame,
    occupancy_config: OccupancyConfig,
    renderer: RobotDepthRenderer,
    *,
    previous_evidence_hash: str | None = None,
) -> OccupancyFrameUpdate:
    """Integrate one already-validated current frame into the safety map."""

    integrator = DepthRayIntegrator(
        prepared.grid,
        DepthIntegrationConfig(
            minimum_depth_m=occupancy_config.minimum_depth_m,
            maximum_depth_m=occupancy_config.maximum_depth_m,
            pixel_stride=occupancy_config.integration_stride,
            minimum_valid_rays=1,
            free_space_margin_m=occupancy_config.free_space_margin_m,
            minimum_free_observations=occupancy_config.minimum_free_observations,
            minimum_free_view_translation_m=(occupancy_config.minimum_free_view_translation_m),
            minimum_free_view_direction_deg=(occupancy_config.minimum_free_view_direction_deg),
        ),
        mapping_context_hash=prepared.mapping_context.content_hash,
    )
    mapping_snapshot = integrator.integrate(
        previous_snapshot,
        prepared.source_depth_m,
        prepared.stereo.rectified.calibration.left,
        prepared.base_t_camera,
        valid_mask=prepared.self_mask.integration_valid_mask,
        source_view_id=prepared.physical_source_id,
        observed_at_utc=prepared.captured_at_utc,
    )
    inherited_evidence_hash = mapping_snapshot.parent_evidence_hash
    if previous_evidence_hash is not None:
        if _SHA256_PATTERN.fullmatch(previous_evidence_hash) is None:
            raise OccupancyMappingError("previous_evidence_hash must be a SHA-256 digest")
        if previous_evidence_hash != inherited_evidence_hash:
            raise OccupancyMappingError(
                "previous_evidence_hash is not bound to the previous occupancy snapshot"
            )
    evidence = OccupancyFrameEvidence(
        source_view_id=prepared.bundle.view_id,
        physical_source_id=prepared.physical_source_id,
        source_sequence_index=prepared.bundle.sequence_index,
        frame_number=prepared.stereo.rectified.source_frame_number,
        captured_at_utc=prepared.captured_at_utc.isoformat(),
        robot_model_hash=renderer.model_content_hash,
        hand_eye_hash=prepared.hand_eye_hash,
        source_stereo_metadata_sha256=prepared.source_stereo_metadata_sha256,
        source_session_manifest_sha256=prepared.source_session_manifest_sha256,
        source_session_view_metadata_sha256=(prepared.source_session_view_metadata_sha256),
        base_t_flange_matrix=tuple(
            tuple(float(value) for value in row)
            for row in prepared.predicted_base_t_flange.matrix
        ),
        predicted_base_t_tcp_matrix=tuple(
            tuple(float(value) for value in row) for row in prepared.predicted_base_t_tcp.matrix
        ),
        observed_base_t_tcp_matrix=tuple(
            tuple(float(value) for value in row) for row in prepared.observed_base_t_tcp.matrix
        ),
        base_t_camera_matrix=tuple(
            tuple(float(value) for value in row) for row in prepared.base_t_camera.matrix
        ),
        joint_positions_rad=tuple(
            float(value)
            for value in prepared.bundle.selected_robot_state.joint_positions_rad
        ),
        fk_tcp_translation_error_m=prepared.fk_tcp_translation_error_m,
        fk_tcp_rotation_error_deg=prepared.fk_tcp_rotation_error_deg,
        valid_depth_fraction=prepared.valid_depth_fraction,
        stereo_valid_fraction=prepared.stereo_valid_fraction,
        confidence_accepted_fraction=prepared.confidence_accepted_fraction,
        mean_accepted_confidence=prepared.mean_accepted_confidence,
        lr_consistency_threshold_px=prepared.lr_consistency_threshold_px,
        self_mask=prepared.self_mask.report,
        mapping_context_hash=prepared.mapping_context.content_hash,
        previous_evidence_hash=inherited_evidence_hash,
        mapping_snapshot_content_hash=mapping_snapshot.content_hash,
        mapping_snapshot_sequence=mapping_snapshot.sequence,
        mapping_source_view_ids=mapping_snapshot.source_view_ids,
        source_depth_content_hash=occupancy_array_content_hash(prepared.source_depth_m),
        stereo_valid_mask_content_hash=occupancy_array_content_hash(prepared.stereo_valid_mask),
        stereo_confidence_content_hash=occupancy_array_content_hash(prepared.stereo_confidence),
        predicted_robot_depth_content_hash=occupancy_array_content_hash(
            prepared.predicted_robot_depth_m
        ),
        robot_mask_content_hash=occupancy_array_content_hash(prepared.self_mask.robot_mask),
        integration_valid_mask_content_hash=occupancy_array_content_hash(
            prepared.self_mask.integration_valid_mask
        ),
    )
    if len(mapping_snapshot.source_view_ids) >= occupancy_config.minimum_source_views:
        snapshot = mapping_snapshot.promote_to_ready(evidence.quality_evidence_hash)
    else:
        snapshot = mapping_snapshot.bind_mapping_evidence(evidence.quality_evidence_hash)
    return OccupancyFrameUpdate(
        snapshot=snapshot,
        mapping_snapshot=mapping_snapshot,
        mapping_context=prepared.mapping_context,
        source_depth_m=prepared.source_depth_m,
        stereo_valid_mask=prepared.stereo_valid_mask,
        stereo_confidence=prepared.stereo_confidence,
        predicted_robot_depth_m=prepared.predicted_robot_depth_m,
        robot_mask=prepared.self_mask.robot_mask,
        integration_valid_mask=prepared.self_mask.integration_valid_mask,
        evidence=evidence,
    )


def integrate_foundation_stereo_occupancy(
    previous_snapshot: OccupancySnapshot | None,
    bundle: SynchronizedFrameBundle,
    stereo: StereoInferenceObservation,
    hand_eye: HandEyeCalibration,
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    renderer: RobotDepthRenderer,
    *,
    captured_at_utc: datetime,
    source_stereo_metadata_sha256: str,
    source_session_manifest_sha256: str,
    source_session_view_metadata_sha256: str,
    previous_evidence_hash: str | None = None,
) -> OccupancyFrameUpdate:
    """Prepare and integrate one settled FoundationStereo observation."""

    prepared = prepare_foundation_stereo_occupancy_frame(
        bundle,
        stereo,
        hand_eye,
        occupancy_config,
        acquisition_config,
        renderer,
        captured_at_utc=captured_at_utc,
        source_stereo_metadata_sha256=source_stereo_metadata_sha256,
        source_session_manifest_sha256=source_session_manifest_sha256,
        source_session_view_metadata_sha256=source_session_view_metadata_sha256,
    )
    return integrate_prepared_foundation_stereo_occupancy(
        previous_snapshot,
        prepared,
        occupancy_config,
        renderer,
        previous_evidence_hash=previous_evidence_hash,
    )


def mark_snapshot_stale_if_expired(
    snapshot: OccupancySnapshot,
    occupancy_config: OccupancyConfig,
    *,
    now_utc: datetime,
) -> OccupancySnapshot:
    """Materialise expiry as a new immutable snapshot version for storage/UI."""

    maximum_age_s = occupancy_config.maximum_map_age_s
    if (
        maximum_age_s is not None
        and snapshot.map_state is OccupancyMapState.MAP_READY
        and snapshot.is_stale(_utc(now_utc), maximum_age_s)
    ):
        return snapshot.mark_stale("capture age exceeded maximum_map_age_s")
    return snapshot


def _validate_source_identity(
    bundle: SynchronizedFrameBundle,
    stereo: StereoInferenceObservation,
) -> None:
    if (
        stereo.source_view_id != bundle.view_id
        or stereo.source_sequence_index != bundle.sequence_index
        or stereo.rectified.source_frame_number != bundle.stereo.frame_number
    ):
        raise OccupancyMappingError("FoundationStereo artifact does not match the stored view")


def _validate_settled_capture(
    bundle: SynchronizedFrameBundle,
    config: AcquisitionConfig,
) -> None:
    metrics = bundle.metrics
    violations = []
    if metrics.bracket_ms > config.max_bracket_ms:
        violations.append("robot/camera bracket")
    if metrics.max_joint_delta_rad > config.max_joint_delta_rad:
        violations.append("joint motion")
    if metrics.tcp_translation_delta_m > config.max_tcp_translation_delta_m:
        violations.append("TCP translation")
    if metrics.tcp_rotation_delta_rad > config.max_tcp_rotation_delta_rad:
        violations.append("TCP rotation")
    if violations:
        raise OccupancyMappingError(
            "Depth frame was not captured at a settled robot pose: " + ", ".join(violations)
        )


def _validated_stereo_quality_mask(
    stereo: StereoInferenceObservation,
    config: OccupancyConfig,
) -> tuple[NDArray[np.bool_], float, float, float, float]:
    confidence = stereo.result.confidence
    if confidence is None:
        raise OccupancyMappingError(
            "Safety occupancy requires FoundationStereo left-right confidence"
        )
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if (
        confidence_array.shape != stereo.depth_m.shape
        or not np.isfinite(confidence_array).all()
        or np.any((confidence_array < 0.0) | (confidence_array > 1.0))
    ):
        raise OccupancyMappingError("Stereo confidence must be finite and lie in [0, 1]")
    metadata = stereo.result.metadata
    if metadata.get("backend") != "foundation_stereo":
        raise OccupancyMappingError("Safety occupancy requires the FoundationStereo backend")
    if metadata.get("left_right_consistency_applied") is not True:
        raise OccupancyMappingError("Stereo left-right consistency was not applied")
    if (
        metadata.get("confidence_semantic")
        != "exp_negative_left_right_disparity_error_not_calibrated_probability"
    ):
        raise OccupancyMappingError("Stereo confidence semantic is not the required LR score")
    raw_threshold = metadata.get("left_right_consistency_threshold_px")
    if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
        raise OccupancyMappingError("Stereo LR consistency threshold metadata is missing")
    threshold = float(raw_threshold)
    if (
        not math.isfinite(threshold)
        or threshold <= 0.0
        or threshold > config.maximum_lr_consistency_error_px
    ):
        raise OccupancyMappingError(
            "Stereo LR consistency threshold exceeds the occupancy quality contract"
        )
    stereo_valid = np.asarray(stereo.result.valid_mask, dtype=np.bool_)
    accepted = stereo_valid & (confidence_array >= config.minimum_stereo_confidence)
    pixel_count = int(stereo_valid.size)
    stereo_fraction = float(np.count_nonzero(stereo_valid) / pixel_count)
    accepted_fraction = float(np.count_nonzero(accepted) / pixel_count)
    mean_confidence = float(np.mean(confidence_array[accepted])) if np.any(accepted) else 0.0
    return accepted, stereo_fraction, accepted_fraction, mean_confidence, threshold


def _build_mapping_context(
    grid: OccupancyGridSpec,
    occupancy_config: OccupancyConfig,
    acquisition_config: AcquisitionConfig,
    stereo: StereoInferenceObservation,
    *,
    hand_eye_hash: str,
    flange_t_left_ir: NDArray[np.float64],
    robot_model_hash: str,
    self_mask_excluded_link_names: tuple[str, ...],
    self_mask_render_backend: str,
    joint_zero_offsets_rad: tuple[float, ...],
    flange_t_tcp: NDArray[np.float64],
    flange_tcp_asset_hash: str,
) -> OccupancyMappingContext:
    for name, digest in (
        ("hand_eye_hash", hand_eye_hash),
        ("robot_model_hash", robot_model_hash),
        ("flange_tcp_asset_hash", flange_tcp_asset_hash),
    ):
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise OccupancyMappingError(f"{name} must be a SHA-256 digest")
    calibration = stereo.rectified.calibration
    offsets = np.asarray(joint_zero_offsets_rad, dtype=np.float64)
    if offsets.shape != (6,) or not np.isfinite(offsets).all():
        raise OccupancyMappingError(
            "Robot renderer joint_zero_offsets_rad must be a finite six-vector"
        )
    excluded_link_names = tuple(str(name).strip() for name in self_mask_excluded_link_names)
    if (
        any(not name for name in excluded_link_names)
        or len(set(excluded_link_names)) != len(excluded_link_names)
    ):
        raise OccupancyMappingError(
            "Robot renderer self-mask exclusions must be unique non-empty link names"
        )
    render_backend = str(self_mask_render_backend).strip()
    if not render_backend:
        raise OccupancyMappingError("Robot renderer self-mask backend must be non-empty")
    try:
        flange_tcp_matrix = _validated_transform_matrix(
            flange_t_tcp,
            name="ES68 flange_T_tcp",
        )
        flange_left_ir_matrix = _validated_transform_matrix(
            flange_t_left_ir,
            name="hand-eye flange_T_left_ir",
        )
    except ValueError as exc:
        raise OccupancyMappingError(str(exc)) from exc
    payload: dict[str, object] = {
        "schema_version": MAPPING_CONTEXT_SCHEMA_VERSION,
        "grid": {
            "frame_id": grid.frame_id,
            "voxel_size_m": grid.voxel_size_m,
            "origin_m": list(grid.origin_m),
            "grid_shape": list(grid.grid_shape),
        },
        "occupancy_contract": occupancy_config.model_dump(mode="json"),
        "acquisition_contract": acquisition_config.model_dump(mode="json"),
        "robot": {
            "model_content_hash": robot_model_hash,
            "self_mask_excluded_link_names": list(excluded_link_names),
            "self_mask_render_backend": render_backend,
            "joint_zero_offsets_rad": offsets.tolist(),
            "flange_T_tcp": flange_tcp_matrix.tolist(),
            "flange_tcp_asset_sha256": flange_tcp_asset_hash,
        },
        "hand_eye": {
            "artifact_sha256": hand_eye_hash,
            "flange_T_left_ir": flange_left_ir_matrix.tolist(),
        },
        "rectified_stereo": {
            "left": _intrinsics_payload(calibration.left),
            "right": _intrinsics_payload(calibration.right),
            "right_rectified_T_left_rectified": (
                calibration.right_rectified_t_left_rectified.matrix.tolist()
            ),
            "left_rectified_T_left_ir": (calibration.left_rectified_t_left_ir.matrix.tolist()),
            "right_rectified_T_right_ir": (calibration.right_rectified_t_right_ir.matrix.tolist()),
            "disparity_to_depth_q": calibration.disparity_to_depth_q.tolist(),
            "left_valid_roi": list(calibration.left_valid_roi),
            "right_valid_roi": list(calibration.right_valid_roi),
        },
        "foundation_stereo": dict(stereo.result.metadata),
    }
    try:
        return OccupancyMappingContext.from_payload(payload)
    except (TypeError, ValueError) as exc:
        raise OccupancyMappingError(f"Mapping context is not canonical: {exc}") from exc


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, object]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion_model": intrinsics.distortion_model,
        "distortion_coefficients": list(intrinsics.distortion_coefficients),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation_error_deg(
    predicted_rotation: NDArray[np.float64],
    observed_rotation: NDArray[np.float64],
) -> float:
    relative = np.asarray(predicted_rotation, dtype=np.float64).T @ np.asarray(
        observed_rotation,
        dtype=np.float64,
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _validated_transform_matrix(
    value: object,
    *,
    name: str,
) -> NDArray[np.float64]:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-7)
    ):
        raise ValueError(f"{name} is not a finite rigid transform")
    return matrix


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise OccupancyMappingError("captured_at_utc must be timezone-aware")
    return value.astimezone(UTC)
