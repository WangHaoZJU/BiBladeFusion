"""Versioned, immutable input contract for the read-only supervisory console."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPERVISORY_SNAPSHOT_SCHEMA_VERSION = 2
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArrayReference(_FrozenModel):
    """A checksummed, non-pickle NumPy array relative to the snapshot root."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    dtype: str = Field(min_length=1)
    shape: tuple[Annotated[int, Field(ge=0)], ...]
    semantic: str = Field(min_length=1)
    allow_nonfinite: bool = False

    @field_validator("dtype")
    @classmethod
    def _valid_numpy_dtype(cls, value: str) -> str:
        try:
            dtype = np.dtype(value)
        except TypeError as exc:
            raise ValueError(f"invalid NumPy dtype: {value}") from exc
        if dtype.hasobject:
            raise ValueError("object arrays are forbidden in supervision snapshots")
        return dtype.str


class TransformSnapshot(_FrozenModel):
    parent_frame: str = Field(min_length=1)
    child_frame: str = Field(min_length=1)
    matrix: tuple[tuple[float, float, float, float], ...]

    @field_validator("matrix")
    @classmethod
    def _valid_homogeneous_matrix(
        cls, value: tuple[tuple[float, float, float, float], ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("transform matrix must be 4x4")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("transform matrix must be finite")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("transform matrix must have homogeneous final row [0,0,0,1]")
        return value


PlanningGateState = Literal[
    "UNKNOWN",
    "PENDING",
    "RUNNING",
    "CLEAR",
    "BLOCKED",
    "NOT_REQUIRED",
    "UNAVAILABLE",
    "ERROR",
]


class CandidatePlanningSnapshot(_FrozenModel):
    """Read-only status of one already-generated, science-ranked candidate."""

    candidate_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    science_rank: int = Field(ge=1)
    science_score: float | None = None
    selected: bool = False
    active: bool = False
    ik_status: PlanningGateState = "UNKNOWN"
    endpoint_status: PlanningGateState = "UNKNOWN"
    straight_path_status: PlanningGateState = "UNKNOWN"
    rrt_status: PlanningGateState = "UNKNOWN"
    ik_duration_s: float | None = Field(default=None, ge=0.0)
    endpoint_duration_s: float | None = Field(default=None, ge=0.0)
    straight_path_duration_s: float | None = Field(default=None, ge=0.0)
    total_duration_s: float | None = Field(default=None, ge=0.0)
    rrt_duration_s: float | None = Field(default=None, ge=0.0)
    target_camera_pose: TransformSnapshot | None = None
    target_tcp_pose: TransformSnapshot | None = None
    blocking_reasons: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_candidate(self) -> CandidatePlanningSnapshot:
        if self.science_score is not None and not np.isfinite(self.science_score):
            raise ValueError("candidate science_score must be finite")
        return self


class PlanningProgressSnapshot(_FrozenModel):
    """Small planning telemetry contract; never an approval or command surface."""

    phase: str = "unknown"
    disposition: str = "unknown"
    cycle_index: int = Field(default=0, ge=0)
    phase_started_at_utc: datetime | None = None
    phase_elapsed_s: float = Field(default=0.0, ge=0.0)
    selection_duration_s: float | None = Field(default=None, ge=0.0)
    candidate_count: int = Field(default=0, ge=0)
    active_candidate_id: str | None = None
    selected_candidate_id: str | None = None
    selected_path_kind: str | None = None
    planned_motion_duration_s: float | None = Field(default=None, ge=0.0)
    planning_waypoint_count: int | None = Field(default=None, ge=0)
    latest_event_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    candidates: tuple[CandidatePlanningSnapshot, ...] = ()

    @field_validator("phase_started_at_utc")
    @classmethod
    def _phase_start_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.utcoffset() is None or value.utcoffset().total_seconds() != 0.0
        ):
            raise ValueError("phase_started_at_utc must include a UTC offset")
        return value

    @model_validator(mode="after")
    def _valid_queue(self) -> PlanningProgressSnapshot:
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must equal the candidate queue length")
        if len({item.rank for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate ranks must be unique")
        selected = tuple(item.candidate_id for item in self.candidates if item.selected)
        active = tuple(item.candidate_id for item in self.candidates if item.active)
        if len(selected) > 1 or len(active) > 1:
            raise ValueError("at most one planning candidate may be selected or active")
        if self.selected_candidate_id is not None and selected != (
            self.selected_candidate_id,
        ):
            raise ValueError("selected_candidate_id must identify the selected candidate")
        if self.active_candidate_id is not None and active != (self.active_candidate_id,):
            raise ValueError("active_candidate_id must identify the active candidate")
        return self


class LivePlanningUpdate(_FrozenModel):
    """Atomically replaced, best-effort progress sidecar for follow-mode viewers."""

    schema_version: Literal[1] = 1
    generated_at_utc: datetime
    source_session_id: str = Field(min_length=1)
    latest_event_sha256: str = Field(pattern=_SHA256_PATTERN)
    planning: PlanningProgressSnapshot

    @field_validator("generated_at_utc")
    @classmethod
    def _generated_at_is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0.0:
            raise ValueError("generated_at_utc must include a UTC offset")
        return value


def _point_array(reference: ArrayReference | None, field_name: str) -> None:
    if reference is None:
        return
    if len(reference.shape) != 2 or reference.shape[1] != 3:
        raise ValueError(f"{field_name} must reference an Nx3 array")


class RobotSceneSnapshot(_FrozenModel):
    model_id: str = Field(min_length=1)
    base_frame: str = Field(min_length=1)
    robot_mode: str = "UNKNOWN"
    safety_status: str = "UNKNOWN"
    joint_names: tuple[str, ...] = ()
    joint_positions_rad: tuple[float, ...] = ()
    link_origins_base_m: ArrayReference | None = None
    collision_mesh_vertices_base_m: ArrayReference | None = None
    collision_mesh_triangles: ArrayReference | None = None
    planned_tcp_path_base_m: ArrayReference | None = None
    actual_tcp_path_base_m: ArrayReference | None = None
    camera_pose: TransformSnapshot | None = None

    @model_validator(mode="after")
    def _consistent_robot_scene(self) -> RobotSceneSnapshot:
        if len(self.joint_names) != len(self.joint_positions_rad):
            raise ValueError("joint_names and joint_positions_rad must have equal length")
        if not np.all(np.isfinite(self.joint_positions_rad)):
            raise ValueError("joint positions must be finite")
        for field_name in (
            "link_origins_base_m",
            "collision_mesh_vertices_base_m",
            "planned_tcp_path_base_m",
            "actual_tcp_path_base_m",
        ):
            _point_array(getattr(self, field_name), field_name)
        if (self.collision_mesh_vertices_base_m is None) != (
            self.collision_mesh_triangles is None
        ):
            raise ValueError("collision mesh vertices and triangles must be provided together")
        if self.collision_mesh_triangles is not None:
            triangles = self.collision_mesh_triangles
            if len(triangles.shape) != 2 or triangles.shape[1] != 3:
                raise ValueError("collision_mesh_triangles must reference an Mx3 array")
            if np.dtype(triangles.dtype).kind not in {"i", "u"}:
                raise ValueError("collision_mesh_triangles must use an integer dtype")
        if self.camera_pose is not None and self.camera_pose.parent_frame != self.base_frame:
            raise ValueError("camera_pose must be expressed in robot base_frame")
        return self


class OccupancySnapshot(_FrozenModel):
    frame_id: str = Field(min_length=1)
    state: Literal["UNREADY", "BOOTSTRAP_READY", "READY", "STALE", "FAILED"]
    version: str = Field(min_length=1)
    content_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    voxel_size_m: float = Field(gt=0.0)
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    age_s: float = Field(ge=0.0)
    integrated_frame_count: int = Field(ge=0)
    occupied_centres_m: ArrayReference | None = None
    inflated_centres_m: ArrayReference | None = None
    free_centres_m: ArrayReference | None = None
    frontier_centres_m: ArrayReference | None = None
    unknown_centres_m: ArrayReference | None = None

    @model_validator(mode="after")
    def _consistent_occupancy(self) -> OccupancySnapshot:
        lower = np.asarray(self.bounds_min_m, dtype=np.float64)
        upper = np.asarray(self.bounds_max_m, dtype=np.float64)
        if not np.all(np.isfinite(np.concatenate((lower, upper)))):
            raise ValueError("occupancy bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("occupancy bounds_max_m must exceed bounds_min_m")
        if self.state in {"BOOTSTRAP_READY", "READY", "STALE"} and self.content_sha256 is None:
            raise ValueError(
                "BOOTSTRAP_READY/READY/STALE occupancy requires content_sha256"
            )
        for field_name in (
            "occupied_centres_m",
            "inflated_centres_m",
            "free_centres_m",
            "frontier_centres_m",
            "unknown_centres_m",
        ):
            _point_array(getattr(self, field_name), field_name)
        return self


class ReconstructionSnapshot(_FrozenModel):
    frame_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    current_points_m: ArrayReference | None = None
    fused_points_m: ArrayReference | None = None
    fused_colors_rgb: ArrayReference | None = None
    surface_normals: ArrayReference | None = None
    front_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    back_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    fin_front_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    fin_back_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    registered_view_count: int = Field(default=0, ge=0)
    registration_rmse_m: float | None = Field(default=None, ge=0.0)
    provenance_status: Literal[
        "UNAVAILABLE", "CURRENT_RUN_VERIFIED", "INDEPENDENT_REFERENCE"
    ] = "UNAVAILABLE"
    provenance_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent_reconstruction(self) -> ReconstructionSnapshot:
        _point_array(self.current_points_m, "current_points_m")
        _point_array(self.fused_points_m, "fused_points_m")
        _point_array(self.surface_normals, "surface_normals")
        if self.fused_colors_rgb is not None:
            _point_array(self.fused_colors_rgb, "fused_colors_rgb")
            if self.fused_points_m is None:
                raise ValueError("fused_colors_rgb requires fused_points_m")
            if self.fused_colors_rgb.shape[0] != self.fused_points_m.shape[0]:
                raise ValueError("fused colors and points must have equal length")
        return self


class SensorSnapshot(_FrozenModel):
    source: Literal["FOUNDATION_STEREO", "D435I_NATIVE", "NONE"] = "NONE"
    frame_number: int | None = Field(default=None, ge=0)
    inference_latency_ms: float | None = Field(default=None, ge=0.0)
    dropped_frame_count: int | None = Field(default=None, ge=0)
    left_ir: ArrayReference | None = None
    right_ir: ArrayReference | None = None
    depth_m: ArrayReference | None = None
    confidence: ArrayReference | None = None
    robot_self_mask: ArrayReference | None = None
    captured_at_utc: datetime | None = None
    occupancy_quality_evidence_sha256: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    valid_depth_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    stereo_valid_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_accepted_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_accepted_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    lr_consistency_threshold_px: float | None = Field(default=None, gt=0.0)
    fk_tcp_translation_error_m: float | None = Field(default=None, ge=0.0)
    fk_tcp_rotation_error_deg: float | None = Field(default=None, ge=0.0)
    projected_robot_pixel_count: int | None = Field(default=None, ge=0)
    measured_valid_pixel_count: int | None = Field(default=None, ge=0)
    depth_matched_pixel_count: int | None = Field(default=None, ge=0)
    masked_valid_pixel_count: int | None = Field(default=None, ge=0)
    retained_valid_pixel_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent_sensor_images(self) -> SensorSnapshot:
        references = {
            "left_ir": self.left_ir,
            "right_ir": self.right_ir,
            "depth_m": self.depth_m,
            "confidence": self.confidence,
            "robot_self_mask": self.robot_self_mask,
        }
        for field_name, reference in references.items():
            if reference is not None and len(reference.shape) != 2:
                raise ValueError(f"{field_name} must reference an HxW array")
        shapes = {reference.shape for reference in references.values() if reference is not None}
        if len(shapes) > 1:
            raise ValueError("sensor image arrays must share one rectified image shape")
        quality_fields = (
            self.captured_at_utc,
            self.occupancy_quality_evidence_sha256,
            self.valid_depth_fraction,
            self.stereo_valid_fraction,
            self.confidence_accepted_fraction,
            self.mean_accepted_confidence,
            self.lr_consistency_threshold_px,
            self.fk_tcp_translation_error_m,
            self.fk_tcp_rotation_error_deg,
            self.projected_robot_pixel_count,
            self.measured_valid_pixel_count,
            self.depth_matched_pixel_count,
            self.masked_valid_pixel_count,
            self.retained_valid_pixel_count,
        )
        if any(value is not None for value in quality_fields) and not all(
            value is not None for value in quality_fields
        ):
            raise ValueError("occupancy sensor-quality evidence must be complete")
        if self.captured_at_utc is not None and (
            self.captured_at_utc.utcoffset() is None
            or self.captured_at_utc.utcoffset().total_seconds() != 0.0
        ):
            raise ValueError("sensor captured_at_utc must include a UTC offset")
        if self.measured_valid_pixel_count is not None:
            if (
                self.masked_valid_pixel_count + self.retained_valid_pixel_count
                != self.measured_valid_pixel_count
            ):
                raise ValueError(
                    "masked and retained pixels must partition measured-valid pixels"
                )
            if self.depth_matched_pixel_count > self.masked_valid_pixel_count:
                raise ValueError("depth-matched robot pixels cannot exceed masked pixels")
        return self


class PlanSnapshot(_FrozenModel):
    plan_id: str = "none"
    state: Literal[
        "NONE",
        "PLANNED",
        "PREFLIGHT_FAILED",
        "READY_FOR_EXTERNAL_APPROVAL",
        "EXECUTING",
        "PAUSED",
        "COMPLETED",
        "ABORTED",
    ] = "NONE"
    current_view_index: int | None = Field(default=None, ge=0)
    total_view_count: int = Field(default=0, ge=0)
    current_view_id: str | None = None
    next_view_id: str | None = None
    minimum_clearance_m: float | None = Field(default=None, ge=0.0)
    blocking_reasons: tuple[str, ...] = ()
    planning: PlanningProgressSnapshot = PlanningProgressSnapshot()

    @model_validator(mode="after")
    def _valid_progress(self) -> PlanSnapshot:
        if self.current_view_index is not None and self.current_view_index >= self.total_view_count:
            raise ValueError("current_view_index must be less than total_view_count")
        return self


class SafetySnapshot(_FrozenModel):
    system_state: Literal[
        "BLOCKED", "READY_FOR_EXTERNAL_APPROVAL", "EXECUTING", "STOPPED", "FAULT"
    ]
    viewer_mode: Literal["READ_ONLY", "REPLAY"]
    viewer_motion_command_capable: Literal[False] = False
    unknown_occupancy_policy: Literal["BLOCK"] = "BLOCK"
    stale_occupancy_policy: Literal["BLOCK"] = "BLOCK"
    feedback_age_ms: float | None = Field(default=None, ge=0.0)
    blocking_reasons: tuple[str, ...] = ()
    calibration_ids: tuple[str, ...] = ()


class AssetRecord(_FrozenModel):
    logical_name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    version: str | None = None


class EventRecord(_FrozenModel):
    timestamp_utc: datetime
    severity: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("timestamp_utc")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0.0:
            raise ValueError("timestamp_utc must include a UTC offset")
        return value


class SupervisorySnapshot(_FrozenModel):
    """One immutable observation of the supervisory state.

    The schema can describe externally executing motion, but it cannot request,
    authorize, or issue motion.  This is enforced by the literal-false capability.
    """

    schema_version: Literal[SUPERVISORY_SNAPSHOT_SCHEMA_VERSION]
    snapshot_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    created_at_utc: datetime
    source_session_id: str | None = None
    safety: SafetySnapshot
    robot: RobotSceneSnapshot
    occupancy: OccupancySnapshot
    reconstruction: ReconstructionSnapshot
    sensor: SensorSnapshot = SensorSnapshot()
    plan: PlanSnapshot = PlanSnapshot()
    assets: tuple[AssetRecord, ...] = ()
    events: tuple[EventRecord, ...] = ()

    @field_validator("created_at_utc")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0.0:
            raise ValueError("created_at_utc must include a UTC offset")
        return value

    @model_validator(mode="after")
    def _cross_component_invariants(self) -> SupervisorySnapshot:
        if self.occupancy.frame_id != self.robot.base_frame:
            raise ValueError("occupancy must be expressed in robot base_frame")
        if self.reconstruction.frame_id != self.robot.base_frame:
            raise ValueError("reconstruction must be expressed in robot base_frame")
        if (
            self.occupancy.state not in {"BOOTSTRAP_READY", "READY"}
            and self.system_can_be_ready
        ):
            raise ValueError(
                "non-motion-eligible occupancy cannot accompany a ready/executing system"
            )
        return self

    @property
    def system_can_be_ready(self) -> bool:
        return self.safety.system_state in {"READY_FOR_EXTERNAL_APPROVAL", "EXECUTING"}


@dataclass(frozen=True, slots=True)
class StoredSupervisorySnapshot:
    path: Path
    root: Path
    content_sha256: str
    snapshot: SupervisorySnapshot


@dataclass(frozen=True, slots=True)
class SupervisoryTimeline:
    snapshots: tuple[StoredSupervisorySnapshot, ...]

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("supervisory timeline cannot be empty")
        sequences = [item.snapshot.sequence for item in self.snapshots]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("supervisory snapshot sequences must be unique and ordered")
        identifiers = [item.snapshot.snapshot_id for item in self.snapshots]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("supervisory snapshot IDs must be unique")
        timestamps = [item.snapshot.created_at_utc for item in self.snapshots]
        if timestamps != sorted(timestamps):
            raise ValueError("supervisory snapshot timestamps must follow sequence order")
        sessions = {
            item.snapshot.source_session_id
            for item in self.snapshots
            if item.snapshot.source_session_id is not None
        }
        if len(sessions) > 1:
            raise ValueError("one supervisory timeline cannot mix source sessions")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_snapshot_file(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    if candidate.is_dir():
        candidate = candidate / "snapshot.json"
    if not candidate.is_file():
        raise ValueError(f"supervisory snapshot does not exist: {candidate}")
    return candidate


def _resolve_array(root: Path, reference: ArrayReference) -> Path:
    relative = Path(reference.path)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"supervision array escapes snapshot root: {reference.path}")
    if path.suffix != ".npy":
        raise ValueError(f"supervision arrays must use .npy: {reference.path}")
    if not path.is_file():
        raise ValueError(f"supervision array does not exist: {path}")
    return path


def _resolve_asset(root: Path, asset: AssetRecord) -> Path:
    candidate = Path(asset.path)
    path = (root / candidate).resolve()
    if candidate.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"supervision asset escapes snapshot root: {asset.path}")
    if not path.is_file():
        raise ValueError(f"supervision asset does not exist: {path}")
    return path


def _verify_asset(root: Path, asset: AssetRecord) -> None:
    path = _resolve_asset(root, asset)
    if _sha256(path) != asset.sha256:
        raise ValueError(f"supervision asset checksum mismatch: {path}")


def snapshot_array_references(
    snapshot: SupervisorySnapshot,
) -> tuple[ArrayReference, ...]:
    """Return every checksummed display-array dependency in schema order."""

    candidates = (
        snapshot.robot.link_origins_base_m,
        snapshot.robot.collision_mesh_vertices_base_m,
        snapshot.robot.collision_mesh_triangles,
        snapshot.robot.planned_tcp_path_base_m,
        snapshot.robot.actual_tcp_path_base_m,
        snapshot.occupancy.occupied_centres_m,
        snapshot.occupancy.inflated_centres_m,
        snapshot.occupancy.free_centres_m,
        snapshot.occupancy.frontier_centres_m,
        snapshot.occupancy.unknown_centres_m,
        snapshot.reconstruction.current_points_m,
        snapshot.reconstruction.fused_points_m,
        snapshot.reconstruction.fused_colors_rgb,
        snapshot.reconstruction.surface_normals,
        snapshot.sensor.left_ir,
        snapshot.sensor.right_ir,
        snapshot.sensor.depth_m,
        snapshot.sensor.confidence,
        snapshot.sensor.robot_self_mask,
    )
    return tuple(item for item in candidates if item is not None)


def _verify_array(root: Path, reference: ArrayReference) -> None:
    path = _resolve_array(root, reference)
    if _sha256(path) != reference.sha256:
        raise ValueError(f"supervision array checksum mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype.hasobject:
        raise ValueError(f"object array is forbidden: {path}")
    if array.dtype.str != reference.dtype:
        raise ValueError(
            f"supervision array dtype mismatch for {path}: {array.dtype.str} != {reference.dtype}"
        )
    if tuple(array.shape) != reference.shape:
        raise ValueError(
            f"supervision array shape mismatch for {path}: {array.shape} != {reference.shape}"
        )
    if (
        not reference.allow_nonfinite
        and np.issubdtype(array.dtype, np.number)
        and not bool(np.all(np.isfinite(array)))
    ):
        raise ValueError(f"supervision array contains non-finite values: {path}")


def _verify_robot_mesh_indices(root: Path, snapshot: SupervisorySnapshot) -> None:
    vertices_reference = snapshot.robot.collision_mesh_vertices_base_m
    triangles_reference = snapshot.robot.collision_mesh_triangles
    if vertices_reference is None or triangles_reference is None:
        return
    triangles = np.load(
        _resolve_array(root, triangles_reference), mmap_mode="r", allow_pickle=False
    )
    if triangles.size and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= vertices_reference.shape[0]
    ):
        raise ValueError("collision mesh triangle index is outside the vertex array")


def read_supervisory_snapshot(path: str | Path) -> StoredSupervisorySnapshot:
    """Read and verify a snapshot plus every display-array dependency."""

    snapshot_path = _resolve_snapshot_file(path)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = SupervisorySnapshot.model_validate(payload)
        root = snapshot_path.parent.resolve()
        seen: dict[str, ArrayReference] = {}
        for reference in snapshot_array_references(snapshot):
            previous = seen.get(reference.path)
            if previous is not None and previous != reference:
                raise ValueError(f"conflicting metadata for array: {reference.path}")
            seen[reference.path] = reference
            _verify_array(root, reference)
        seen_assets: dict[str, AssetRecord] = {}
        for asset in snapshot.assets:
            previous_asset = seen_assets.get(asset.path)
            if previous_asset is not None and previous_asset != asset:
                raise ValueError(f"conflicting metadata for asset: {asset.path}")
            seen_assets[asset.path] = asset
            _verify_asset(root, asset)
        _verify_robot_mesh_indices(root, snapshot)
        return StoredSupervisorySnapshot(
            path=snapshot_path,
            root=root,
            content_sha256=_sha256(snapshot_path),
            snapshot=snapshot,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid supervisory snapshot {snapshot_path}: {exc}") from exc


def read_live_planning_update(path: str | Path) -> LivePlanningUpdate:
    """Read the optional progress sidecar without granting it scientific authority."""

    candidate = Path(path).resolve()
    if candidate.is_dir():
        candidate = candidate / "live_planning.json"
    if not candidate.is_file():
        raise ValueError(f"live planning update does not exist: {candidate}")
    try:
        return LivePlanningUpdate.model_validate_json(candidate.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid live planning update {candidate}: {exc}") from exc


def write_live_planning_update(path: str | Path, update: LivePlanningUpdate) -> Path:
    """Atomically replace one diagnostic-only planning update."""

    if type(update) is not LivePlanningUpdate:
        raise TypeError("live planning update must use the typed read-only contract")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            update.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_snapshot_array(
    stored: StoredSupervisorySnapshot,
    reference: ArrayReference | None,
) -> np.ndarray | None:
    """Load one already-verified display array without allowing pickle payloads."""

    if reference is None:
        return None
    path = _resolve_array(stored.root, reference)
    array = np.load(path, allow_pickle=False)
    array.setflags(write=False)
    return array


def discover_supervisory_snapshots(path: str | Path) -> SupervisoryTimeline:
    """Discover one snapshot or an ordered, direct-child replay timeline."""

    root = Path(path).resolve()
    if root.is_file() or (root.is_dir() and (root / "snapshot.json").is_file()):
        return SupervisoryTimeline((read_supervisory_snapshot(root),))
    if not root.is_dir():
        raise ValueError(f"supervisory replay source does not exist: {root}")

    candidates = set(root.glob("*.supervision.json"))
    candidates.update(root.glob("*/snapshot.json"))
    if not candidates:
        raise ValueError(
            "supervisory replay directory must contain *.supervision.json or */snapshot.json"
        )
    snapshots = tuple(
        sorted(
            (read_supervisory_snapshot(candidate) for candidate in candidates),
            key=lambda item: (item.snapshot.sequence, item.snapshot.created_at_utc),
        )
    )
    return SupervisoryTimeline(snapshots)
