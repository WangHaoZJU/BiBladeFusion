"""Atomic, read-only live snapshots for the supervised stop-scan runner.

The bridge is an observer only.  It accepts already materialized perception and
preflight summaries, never a robot driver, approval object, executor, or command
port.  Every publication is a self-contained immutable snapshot that the existing
``supervise replay --follow`` console can discover without seeing partial files.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.diagnostics.performance_timing import (
    activate_performance_timing,
    performance_span,
    performance_timed,
    try_create_performance_timing,
)
from biblade_fusion.mapping import OccupancyMapState
from biblade_fusion.mapping import OccupancySnapshot as MapSnapshot
from biblade_fusion.robotics import (
    ES68_JOINT_NAMES,
    Es68KinematicModel,
    load_es68_flange_t_tcp,
)
from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    es68_d435i_collision_content_hash,
    es68_d435i_robot_geometry_hash,
)
from biblade_fusion.storage.coarse_scan import (
    _CoarseScanViewReadback,
    _revalidate_coarse_scan_view_readback,
    read_coarse_scan_view,
)
from biblade_fusion.storage.display_source_registry import AppendOnlyDisplaySourceRegistry
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.storage.stop_scan_run import StopScanRunEvent
from biblade_fusion.storage.surface_coverage import read_surface_coverage_generation
from biblade_fusion.supervision.experiment import (
    ExperimentDisposition,
    ExperimentStatusSnapshot,
)
from biblade_fusion.supervision.snapshot import (
    SUPERVISORY_SNAPSHOT_SCHEMA_VERSION,
    EventRecord,
    OccupancySnapshot,
    PlanSnapshot,
    ReconstructionSnapshot,
    RobotSceneSnapshot,
    SafetySnapshot,
    SensorSnapshot,
    StoredSupervisorySnapshot,
    SupervisorySnapshot,
    TransformSnapshot,
    discover_supervisory_snapshots,
)
from biblade_fusion.supervision.storage import AtomicSupervisorySnapshotWriter
from biblade_fusion.workflows.stop_scan_coordinator import (
    PerceptionCycleResult,
    PreparedSegment,
)


class LiveSupervisionError(RuntimeError):
    """A live observation could not be published without overstating readiness."""


_DISPLAY_VOXEL_SIZE_M = 0.002
_MAXIMUM_CURRENT_DISPLAY_POINTS = 50_000
_MAXIMUM_FUSED_DISPLAY_POINTS = 200_000
_DISPLAY_UNION_ALGORITHM = "deterministic_bounded_display_voxel_union_v1"


@dataclass(frozen=True, slots=True)
class LiveSupervisionLayout:
    """Static scene identity and fallback occupancy bounds for an empty run."""

    model_id: str
    occupancy_bounds_min_m: tuple[float, float, float]
    occupancy_bounds_max_m: tuple[float, float, float]
    occupancy_voxel_size_m: float
    base_frame: Literal["base"] = "base"
    joint_names: tuple[str, ...] = ES68_JOINT_NAMES

    def __post_init__(self) -> None:
        lower = np.asarray(self.occupancy_bounds_min_m, dtype=np.float64)
        upper = np.asarray(self.occupancy_bounds_max_m, dtype=np.float64)
        if not self.model_id.strip():
            raise ValueError("Live supervision model_id must be non-empty")
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(np.concatenate((lower, upper))).all()
            or np.any(upper <= lower)
        ):
            raise ValueError("Live supervision occupancy bounds are invalid")
        if not math.isfinite(self.occupancy_voxel_size_m) or self.occupancy_voxel_size_m <= 0:
            raise ValueError("Live supervision voxel size must be finite and positive")
        if len(self.joint_names) != 6 or any(not value.strip() for value in self.joint_names):
            raise ValueError("Live supervision requires six named ES68 joints")


@dataclass(frozen=True, slots=True)
class _CollisionMeshPart:
    link_name: str
    parent_link: str
    vertices_m: NDArray[np.float64]
    triangles: NDArray[np.int64]
    parent_t_mesh: NDArray[np.float64]
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        vertices = _readonly_array(self.vertices_m, dtype=np.float64)
        triangles = _readonly_array(self.triangles, dtype=np.int64)
        transform = _readonly_array(self.parent_t_mesh, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or transform.shape != (4, 4)
            or not np.isfinite(vertices).all()
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12)
            or np.any(triangles < 0)
            or (len(vertices) and np.any(triangles >= len(vertices)))
        ):
            raise ValueError("Live collision mesh part is malformed")
        if not self.link_name.strip() or not self.parent_link.strip():
            raise ValueError("Live collision mesh link identity must be non-empty")
        source = self.source_path.resolve()
        if not source.is_file() or _sha256(source) != self.source_sha256:
            raise ValueError(f"Live collision mesh source changed: {source}")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "parent_t_mesh", transform)
        object.__setattr__(self, "source_path", source)


@dataclass(frozen=True, slots=True)
class LiveCollisionGeometry:
    """Detached display copy of the exact active collision STL assembly.

    The object contains no robot, checker, permit, approval, or command reference.
    Its constructor recomputes the active manifest identities and refuses a mismatch
    with the collision checker that supplied the expected hashes.
    """

    model_id: str
    collision_model_hash: str
    robot_geometry_hash: str
    manifest_path: Path
    manifest_sha256: str
    parts: tuple[_CollisionMeshPart, ...]
    kinematics: Es68KinematicModel

    @classmethod
    def from_active_resources(
        cls,
        resources: Es68D435iCollisionResources,
        *,
        joint_zero_offsets_rad: Sequence[float],
        expected_model_id: str,
        expected_collision_model_hash: str,
        expected_robot_geometry_hash: str,
    ) -> LiveCollisionGeometry:
        template = resources.load_active()
        collision_hash = es68_d435i_collision_content_hash(template)
        geometry_hash = es68_d435i_robot_geometry_hash(
            template,
            joint_zero_offsets_rad=joint_zero_offsets_rad,
        )
        if template.model_id != expected_model_id:
            raise LiveSupervisionError("Live collision display model ID differs from checker")
        if collision_hash != expected_collision_model_hash:
            raise LiveSupervisionError("Live collision display manifest/STLs differ from checker")
        if geometry_hash != expected_robot_geometry_hash:
            raise LiveSupervisionError("Live collision display geometry differs from checker")
        import trimesh

        parts: list[_CollisionMeshPart] = []
        for spec in (*template.links, template.attachment):
            raw = trimesh.load_mesh(spec.mesh_path, process=False)
            if isinstance(raw, trimesh.Scene):
                geometries = tuple(raw.geometry.values())
                if not geometries:
                    raise LiveSupervisionError(f"Empty collision mesh scene: {spec.mesh_path}")
                raw = trimesh.util.concatenate(geometries)
            vertices = np.asarray(raw.vertices, dtype=np.float64) * template.mesh_scale
            triangles = np.asarray(raw.faces, dtype=np.int64)
            if spec is template.attachment:
                parent_link = template.attachment.parent_link
                parent_t_mesh = _xyz_rpy_matrix(
                    template.attachment.joint_xyz_m,
                    template.attachment.joint_rpy_rad,
                ) @ _xyz_rpy_matrix(spec.origin_xyz_m, spec.origin_rpy_rad)
            else:
                parent_link = spec.link_name
                parent_t_mesh = _xyz_rpy_matrix(spec.origin_xyz_m, spec.origin_rpy_rad)
            parts.append(
                _CollisionMeshPart(
                    link_name=spec.link_name,
                    parent_link=parent_link,
                    vertices_m=vertices,
                    triangles=triangles,
                    parent_t_mesh=parent_t_mesh,
                    source_path=spec.mesh_path,
                    source_sha256=_sha256(spec.mesh_path),
                )
            )
        return cls(
            model_id=template.model_id,
            collision_model_hash=collision_hash,
            robot_geometry_hash=geometry_hash,
            manifest_path=template.manifest_path.resolve(),
            manifest_sha256=_sha256(template.manifest_path),
            parts=tuple(parts),
            kinematics=Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=joint_zero_offsets_rad
            ),
        )

    def __post_init__(self) -> None:
        for label, value in (
            ("collision_model_hash", self.collision_model_hash),
            ("robot_geometry_hash", self.robot_geometry_hash),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"Live collision {label} must be a lowercase SHA-256 digest")
        manifest = self.manifest_path.resolve()
        if not manifest.is_file() or _sha256(manifest) != self.manifest_sha256:
            raise ValueError(f"Live collision manifest changed: {manifest}")
        if not self.model_id.strip() or not self.parts:
            raise ValueError("Live collision geometry identity/parts are incomplete")
        object.__setattr__(self, "manifest_path", manifest)

    @performance_timed("live.collision_geometry")
    def base_mesh(
        self,
        joint_positions_rad: Sequence[float],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, object]]:
        """Return one checksummed base-frame triangle soup for display only."""

        self._assert_sources_unchanged()
        joints = np.asarray(joint_positions_rad, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise LiveSupervisionError("Live collision mesh requires six finite joints")
        link_transforms = self.kinematics.link_transforms(joints)
        base_t_flange = self.kinematics.forward_kinematics(joints)
        vertices: list[NDArray[np.float64]] = []
        triangles: list[NDArray[np.int64]] = []
        vertex_offset = 0
        for part in self.parts:
            if part.parent_link == "flange":
                base_t_parent = base_t_flange
            else:
                base_t_parent = link_transforms.get(part.parent_link)
                if base_t_parent is None:
                    raise LiveSupervisionError(
                        "Collision display parent link is absent from active FK: "
                        f"{part.parent_link}"
                    )
            base_t_mesh = base_t_parent @ part.parent_t_mesh
            homogeneous = np.column_stack((part.vertices_m, np.ones(len(part.vertices_m))))
            transformed = (base_t_mesh @ homogeneous.T).T[:, :3]
            vertices.append(transformed)
            triangles.append(part.triangles + vertex_offset)
            vertex_offset += len(part.vertices_m)
        vertex_array = np.concatenate(vertices, axis=0).astype(np.float64, copy=False)
        triangle_array = np.concatenate(triangles, axis=0).astype(np.int64, copy=False)
        binding: dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": "biblade_fusion.live_collision_mesh_binding",
            "motion_authorized": False,
            "model_id": self.model_id,
            "collision_model_sha256": self.collision_model_hash,
            "robot_geometry_sha256": self.robot_geometry_hash,
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
            },
            "joint_positions_rad": joints.tolist(),
            "vertices_f64le_sha256": _array_content_sha256(vertex_array, dtype="<f8"),
            "triangles_i64le_sha256": _array_content_sha256(triangle_array, dtype="<i8"),
            "vertex_count": int(len(vertex_array)),
            "triangle_count": int(len(triangle_array)),
            "parts": [
                {
                    "link_name": part.link_name,
                    "parent_link": part.parent_link,
                    "path": str(part.source_path),
                    "sha256": part.source_sha256,
                    "vertex_count": int(len(part.vertices_m)),
                    "triangle_count": int(len(part.triangles)),
                }
                for part in self.parts
            ],
        }
        return (
            _readonly_array(vertex_array, dtype=np.float64),
            _readonly_array(triangle_array, dtype=np.int64),
            binding,
        )

    def _assert_sources_unchanged(self) -> None:
        if _sha256(self.manifest_path) != self.manifest_sha256:
            raise LiveSupervisionError("Active collision manifest changed after display binding")
        for part in self.parts:
            if not part.source_path.is_file() or _sha256(part.source_path) != part.source_sha256:
                raise LiveSupervisionError(
                    f"Active collision STL changed after display binding: {part.source_path}"
                )


@dataclass(frozen=True, slots=True)
class _AssetSource:
    name: str
    logical_name: str
    kind: str
    path: Path
    sha256: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class _SensorState:
    frame_number: int
    captured_at_utc: datetime
    left_ir: NDArray[np.uint8]
    right_ir: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    confidence: NDArray[np.float32] | None
    occupancy_quality_evidence_sha256: str
    valid_depth_fraction: float
    stereo_valid_fraction: float
    confidence_accepted_fraction: float
    mean_accepted_confidence: float
    lr_consistency_threshold_px: float
    fk_tcp_translation_error_m: float
    fk_tcp_rotation_error_deg: float
    projected_robot_pixel_count: int
    measured_valid_pixel_count: int
    depth_matched_pixel_count: int
    masked_valid_pixel_count: int
    retained_valid_pixel_count: int


@dataclass(frozen=True, slots=True)
class _ScienceState:
    current_points_m: NDArray[np.float64] | None = None
    fused_points_m: NDArray[np.float64] | None = None
    front_coverage: float = 0.0
    back_coverage: float = 0.0
    fin_front_coverage: float = 0.0
    fin_back_coverage: float = 0.0
    registered_view_count: int = 0
    model_version: str = "unavailable"


@dataclass(frozen=True, slots=True)
class _PerceptionState:
    robot_joint_positions_rad: tuple[float, ...]
    robot_mode: str
    safety_status: str
    camera_pose_matrix: tuple[tuple[float, float, float, float], ...]
    captured_at_utc: datetime
    occupancy: MapSnapshot
    sensor: _SensorState
    science: _ScienceState
    assets: tuple[_AssetSource, ...]


@dataclass(frozen=True, slots=True)
class _CurrentScienceView:
    points_m: NDArray[np.float64] | None
    asset: _AssetSource | None
    source_kind: str | None = None
    source_view_id: str | None = None
    source_sequence_index: int | None = None
    source_frame_number: int | None = None
    metadata_path: Path | None = None
    metadata_sha256: str | None = None
    point_array_path: Path | None = None
    point_array_file_sha256: str | None = None
    points_f64le_sha256: str | None = None


def _display_key_rank(key: tuple[int, int, int]) -> bytes:
    return hashlib.sha256(f"{key[0]},{key[1]},{key[2]}".encode("ascii")).digest()


@performance_timed("live.voxel_conversion")
def _display_voxel_representatives(
    points: NDArray[np.float64],
    *,
    maximum_points: int,
) -> tuple[NDArray[np.float64], tuple[tuple[int, int, int], ...]]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise LiveSupervisionError("Display point cloud must be finite Nx3 base-frame points")
    best: dict[tuple[int, int, int], tuple[float, tuple[float, float, float]]] = {}
    for row in array:
        point = tuple(float(value) for value in row)
        key_array = np.floor(row / _DISPLAY_VOXEL_SIZE_M).astype(np.int64)
        key = tuple(int(value) for value in key_array)
        centre = (key_array.astype(np.float64) + 0.5) * _DISPLAY_VOXEL_SIZE_M
        candidate = (float(np.sum((row - centre) ** 2)), point)
        prior = best.get(key)
        if prior is None or candidate < prior:
            best[key] = candidate
    retained_keys = tuple(
        sorted(best, key=lambda key: (_display_key_rank(key), key))[:maximum_points]
    )
    points_out = _readonly_array([best[key][1] for key in retained_keys], dtype=np.float64)
    return points_out.reshape((-1, 3)), retained_keys


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[object] | type[object] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise ValueError("Live supervision arrays cannot use object dtype")
    array.setflags(write=False)
    return array


def _array_content_sha256(
    value: object,
    *,
    dtype: str,
) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _xyz_rpy_matrix(
    xyz_m: Sequence[float],
    rpy_rad: Sequence[float],
) -> NDArray[np.float64]:
    translation = np.asarray(xyz_m, dtype=np.float64)
    angles = np.asarray(rpy_rad, dtype=np.float64)
    if translation.shape != (3,) or angles.shape != (3,) or not np.isfinite(
        np.concatenate((translation, angles))
    ).all():
        raise ValueError("Collision mesh xyz/rpy must be finite three-vectors")
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    transform[:3, 3] = translation
    return transform


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@performance_timed("live.asset_binding")
def _asset(
    name: str,
    logical_name: str,
    kind: str,
    path: str | Path,
    *,
    version: str | None = None,
) -> _AssetSource:
    source = Path(path).resolve()
    if not source.is_file():
        raise LiveSupervisionError(f"Live supervision source asset is missing: {source}")
    return _AssetSource(name, logical_name, kind, source, _sha256(source), version)


@performance_timed("live.voxel_conversion")
def _voxel_centres(
    indices: frozenset[tuple[int, int, int]],
    snapshot: MapSnapshot,
) -> NDArray[np.float64]:
    if not indices:
        return np.empty((0, 3), dtype=np.float64)
    ordered = np.asarray(sorted(indices), dtype=np.float64)
    return np.asarray(snapshot.origin_m, dtype=np.float64) + (ordered + 0.5) * snapshot.voxel_size_m


def _tcp_path(
    model: Es68KinematicModel,
    joint_path: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    if joint_path is None:
        return None
    flange_t_tcp = load_es68_flange_t_tcp().matrix
    points = np.empty((len(joint_path), 3), dtype=np.float64)
    for index, joints in enumerate(joint_path):
        points[index] = (model.forward_kinematics(joints) @ flange_t_tcp)[:3, 3]
    return points


def _link_origins(
    model: Es68KinematicModel,
    joints: tuple[float, ...],
) -> NDArray[np.float64]:
    transforms = model.link_transforms(joints)
    return np.asarray([matrix[:3, 3] for matrix in transforms.values()], dtype=np.float64)


def _joint_path(prepared: PreparedSegment) -> NDArray[np.float64]:
    stream = prepared.preflight.servoj_stream
    if stream is not None:
        stream.validate()
        commands = np.asarray(stream.commands, dtype=np.float64)
        start = np.asarray(prepared.preflight.start_joint_positions_rad, dtype=np.float64)
        if not np.allclose(commands[0], start, rtol=0.0, atol=1e-12):
            commands = np.vstack((start, commands))
        return commands
    return np.asarray(
        (
            prepared.preflight.start_joint_positions_rad,
            prepared.preflight.goal_joint_positions_rad,
        ),
        dtype=np.float64,
    )


def _coverage_values(path: Path | None) -> tuple[float, float, float, float, int]:
    if path is None:
        return 0.0, 0.0, 0.0, 0.0, 0
    stored = read_surface_coverage_generation(
        path,
        require_foreground_bound_science=True,
    )

    def fraction(*, side: str, fin: bool) -> float:
        selected = tuple(
            patch
            for patch in stored.quality.patches
            if patch.side.value == side and (patch.region.value.startswith("fin_")) is fin
        )
        weights = np.asarray(
            [max(0, patch.reference_point_count) for patch in selected],
            dtype=np.float64,
        )
        if not len(weights) or float(np.sum(weights)) <= 0.0:
            return 0.0
        return float(
            np.average(
                [patch.coverage_fraction for patch in selected],
                weights=weights,
            )
        )

    return (
        fraction(side="front", fin=False),
        fraction(side="back", fin=False),
        fraction(side="front", fin=True),
        fraction(side="back", fin=True),
        len(stored.ledger.observation_ids),
    )


def _read_current_science_view(
    result: PerceptionCycleResult,
    *,
    coarse_readback: _CoarseScanViewReadback | None = None,
) -> _CurrentScienceView:
    """Resolve exactly one fine or coarse reconstructed view for display."""

    if result.reconstructed_view_path is not None and result.coarse_scan_view_path is not None:
        raise LiveSupervisionError(
            "One perception cycle cannot publish fine and coarse reconstructions together"
        )
    reconstructed = None
    asset = None
    source_kind = None
    reconstructed_root = None
    if result.coarse_scan_view_path is not None:
        try:
            coarse = (
                read_coarse_scan_view(result.coarse_scan_view_path)
                if coarse_readback is None
                else _revalidate_coarse_scan_view_readback(
                    coarse_readback,
                    expected_root=result.coarse_scan_view_path,
                )
            )
        except ValueError as exc:
            raise LiveSupervisionError(
                f"Coarse live readback authority changed: {exc}"
            ) from exc
        sources = coarse.metadata["sources"]
        if (
            Path(str(sources["stereo_inference"]["root"])).resolve()
            != result.stereo_inference_path.resolve()
            or Path(str(sources["occupancy_mapping"]["root"])).resolve()
            != result.occupancy_mapping_path.resolve()
        ):
            raise LiveSupervisionError(
                "Coarse live readback sources differ from the perception result"
            )
        reconstructed = coarse.reconstructed
        source_kind = "coarse_scan_view"
        reconstructed_root = Path(
            str(coarse.metadata["sources"]["reconstructed_view"]["root"])
        ).resolve()
        asset = _asset(
            "coarse_scan_view",
            "current_coarse_scan_view_metadata",
            "biblade_fusion.coarse_scan_view",
            result.coarse_scan_view_path / "metadata.json",
        )
    elif result.reconstructed_view_path is not None:
        if coarse_readback is not None:
            raise LiveSupervisionError(
                "A coarse transaction readback cannot authorize a fine reconstruction"
            )
        reconstructed = read_reconstructed_view(result.reconstructed_view_path)
        source_kind = "fine_reconstructed_view"
        reconstructed_root = result.reconstructed_view_path.resolve()
        asset = _asset(
            "reconstructed_view",
            "current_reconstructed_view_metadata",
            "biblade_fusion.reconstructed_view",
            result.reconstructed_view_path / "metadata.json",
        )
    if reconstructed is None or reconstructed_root is None or source_kind is None:
        return _CurrentScienceView(None, None)
    bundle = result.bundle
    if (
        reconstructed.view.source_view_id != bundle.view_id
        or reconstructed.view.source_sequence_index != bundle.sequence_index
        or reconstructed.view.source_frame_number != bundle.stereo.frame_number
    ):
        raise LiveSupervisionError(
            "Live reconstructed science view and captured view identities differ"
        )
    metadata_path = reconstructed_root / "metadata.json"
    point_record = reconstructed.metadata["files"]["base_points_m"]
    relative = Path(str(point_record["path"]))
    point_path = (reconstructed_root / relative).resolve()
    if relative.is_absolute() or not point_path.is_relative_to(reconstructed_root):
        raise LiveSupervisionError("Reconstructed display point path escapes its artifact")
    point_file_sha256 = str(point_record["sha256"])
    if not point_path.is_file() or _sha256(point_path) != point_file_sha256:
        raise LiveSupervisionError("Reconstructed display point file changed")
    points = _readonly_array(reconstructed.view.base_cloud.points_m, dtype=np.float64)
    points_f64le_sha256 = _array_content_sha256(points, dtype="<f8")
    try:
        persisted_points = np.load(point_path, allow_pickle=False)
        persisted_f64le_sha256 = _array_content_sha256(
            persisted_points,
            dtype="<f8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LiveSupervisionError(
            "Reconstructed display point array cannot be verified"
        ) from exc
    if persisted_f64le_sha256 != points_f64le_sha256:
        raise LiveSupervisionError(
            "Typed reconstructed display points differ from their immutable array"
        )
    return _CurrentScienceView(
        points_m=points,
        asset=asset,
        source_kind=source_kind,
        source_view_id=reconstructed.view.source_view_id,
        source_sequence_index=reconstructed.view.source_sequence_index,
        source_frame_number=reconstructed.view.source_frame_number,
        metadata_path=metadata_path,
        metadata_sha256=_sha256(metadata_path),
        point_array_path=point_path,
        point_array_file_sha256=point_file_sha256,
        points_f64le_sha256=points_f64le_sha256,
    )


class LiveSupervisionBridge:
    """Runner callbacks that publish only immutable, command-incapable snapshots.

    Register :meth:`observe_perception` as a ``perception_callback``,
    :meth:`observe_prepared_segment` as a ``prepared_segment_callback``,
    :meth:`observe_event` as an ``event_callback``, and the bridge itself as a
    ``status_callback``.  All callbacks consume detached evidence; none can reach a
    robot or an approval/execution boundary.
    """

    motion_command_capable: Literal[False] = False

    def __init__(
        self,
        timeline_root: str | Path,
        *,
        layout: LiveSupervisionLayout,
        kinematics: Es68KinematicModel,
        collision_geometry: LiveCollisionGeometry | None = None,
        utc_clock=lambda: datetime.now(UTC),
        maximum_event_records: int = 32,
    ) -> None:
        if maximum_event_records < 1:
            raise ValueError("maximum_event_records must be positive")
        self._root = Path(timeline_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._layout = layout
        self._kinematics = kinematics
        self._collision_geometry = collision_geometry
        if collision_geometry is not None and collision_geometry.model_id != layout.model_id:
            raise LiveSupervisionError(
                "Live collision geometry and supervision layout model IDs differ"
            )
        self._utc_clock = utc_clock
        self._maximum_event_records = maximum_event_records
        self._lock = threading.RLock()
        self._perception: _PerceptionState | None = None
        self._prepared: PreparedSegment | None = None
        self._planned_joint_path: NDArray[np.float64] | None = None
        self._sampled_actual_joints: list[tuple[float, ...]] = []
        self._display_union: dict[
            tuple[int, int, int], tuple[float, tuple[float, float, float]]
        ] = {}
        try:
            registry_root = self._root / "display_source_registry"
            timeline_exists = bool(
                tuple(self._root.glob("*/snapshot.json"))
                or tuple(self._root.glob("*.supervision.json"))
            )
            if timeline_exists and not registry_root.is_dir():
                raise ValueError("existing live timeline has no display-source registry")
            self._display_registry = AppendOnlyDisplaySourceRegistry(
                registry_root
            )
            self._registered_physical_source_ids: set[str] = set()
            self._restore_display_registry()
        except (OSError, TypeError, ValueError) as exc:
            raise LiveSupervisionError(
                f"Live display-source registry recovery failed: {exc}"
            ) from exc
        self._events: list[StopScanRunEvent] = []
        self._next_sequence, self._existing_session_id = self._discover_timeline_state()
        self._verify_latest_timeline_bindings()

    @property
    def timeline_root(self) -> Path:
        return self._root

    def _restore_display_registry(self) -> None:
        for entry in self._display_registry.entries:
            if (
                entry.display_algorithm != _DISPLAY_UNION_ALGORITHM
                or entry.display_voxel_size_m != _DISPLAY_VOXEL_SIZE_M
                or entry.maximum_current_points != _MAXIMUM_CURRENT_DISPLAY_POINTS
            ):
                raise LiveSupervisionError(
                    "Stored display-source policy differs from the active deterministic policy"
                )
            points = self._display_registry.load_points(entry)
            representatives, keys = _display_voxel_representatives(
                points,
                maximum_points=_MAXIMUM_CURRENT_DISPLAY_POINTS,
            )
            if len(representatives) != entry.voxel_point_count:
                raise LiveSupervisionError(
                    "Stored display-source voxel count does not replay"
                )
            self._merge_display_points(representatives, keys)
            self._registered_physical_source_ids.add(entry.physical_source_id)

    def _merge_display_points(
        self,
        points: NDArray[np.float64],
        keys: tuple[tuple[int, int, int], ...],
    ) -> None:
        for point, key in zip(points, keys, strict=True):
            key_array = np.asarray(key, dtype=np.float64)
            centre = (key_array + 0.5) * _DISPLAY_VOXEL_SIZE_M
            value = tuple(float(item) for item in point)
            candidate = (float(np.sum((point - centre) ** 2)), value)
            prior = self._display_union.get(key)
            if prior is None or candidate < prior:
                self._display_union[key] = candidate
        if len(self._display_union) > _MAXIMUM_FUSED_DISPLAY_POINTS:
            retained = set(
                sorted(
                    self._display_union,
                    key=lambda item: (_display_key_rank(item), item),
                )[:_MAXIMUM_FUSED_DISPLAY_POINTS]
            )
            self._display_union = {
                key: value for key, value in self._display_union.items() if key in retained
            }

    def _discover_timeline_state(self) -> tuple[int, str | None]:
        candidates = tuple(self._root.glob("*/snapshot.json"))
        legacy_candidates = tuple(self._root.glob("*.supervision.json"))
        if not candidates and not legacy_candidates:
            return 0, None
        timeline = discover_supervisory_snapshots(self._root)
        sessions = {
            item.snapshot.source_session_id
            for item in timeline.snapshots
            if item.snapshot.source_session_id is not None
        }
        return (
            timeline.snapshots[-1].snapshot.sequence + 1,
            next(iter(sessions)) if sessions else None,
        )

    def _verify_latest_timeline_bindings(self) -> None:
        candidates = tuple(self._root.glob("*/snapshot.json"))
        legacy_candidates = tuple(self._root.glob("*.supervision.json"))
        if not candidates and not legacy_candidates:
            return
        latest = discover_supervisory_snapshots(self._root).snapshots[-1]
        registry_assets = tuple(
            asset
            for asset in latest.snapshot.assets
            if asset.logical_name == "append_only_display_source_registry_head"
        )
        if len(registry_assets) != 1:
            raise LiveSupervisionError(
                "Existing live timeline lacks one display-source registry head binding"
            )
        registry_payload = json.loads(
            (latest.root / registry_assets[0].path).read_text(encoding="utf-8")
        )
        snapshot_count = int(registry_payload["entry_count"])
        entries = self._display_registry.entries
        if (
            Path(str(registry_payload["registry_root"])).resolve()
            != self._display_registry.root
            or snapshot_count < 0
            or snapshot_count > len(entries)
        ):
            raise LiveSupervisionError(
                "Existing timeline and display-source registry bounds differ"
            )
        expected_head = entries[snapshot_count - 1].entry_sha256 if snapshot_count else None
        expected_path = str(entries[snapshot_count - 1].path) if snapshot_count else None
        expected_file_hash = _sha256(entries[snapshot_count - 1].path) if snapshot_count else None
        if (
            registry_payload["head_entry_sha256"] != expected_head
            or registry_payload["head_entry_path"] != expected_path
            or registry_payload["head_entry_file_sha256"] != expected_file_hash
        ):
            raise LiveSupervisionError(
                "Existing timeline display-source chain head is not a registry ancestor"
            )
        collision_assets = tuple(
            asset
            for asset in latest.snapshot.assets
            if asset.logical_name == "active_live_collision_mesh_binding"
        )
        if len(collision_assets) > 1:
            raise LiveSupervisionError("Existing live timeline has ambiguous collision binding")
        if collision_assets:
            if self._collision_geometry is None:
                raise LiveSupervisionError(
                    "Existing live timeline collision binding has no active display geometry"
                )
            collision_payload = json.loads(
                (latest.root / collision_assets[0].path).read_text(encoding="utf-8")
            )
            if (
                collision_payload["model_id"] != self._collision_geometry.model_id
                or collision_payload["collision_model_sha256"]
                != self._collision_geometry.collision_model_hash
                or collision_payload["robot_geometry_sha256"]
                != self._collision_geometry.robot_geometry_hash
                or collision_payload["manifest"]["sha256"]
                != self._collision_geometry.manifest_sha256
            ):
                raise LiveSupervisionError(
                    "Existing live timeline collision geometry differs from the active model"
                )

    def observe_event(self, event: StopScanRunEvent) -> None:
        """Remember a verified event for the next immutable UI publication."""

        if type(event) is not StopScanRunEvent:
            raise LiveSupervisionError("Live supervision requires a typed run event")
        with self._lock:
            if self._events:
                previous = self._events[-1]
                if event.run_id != previous.run_id or event.sequence != previous.sequence + 1:
                    raise LiveSupervisionError("Live supervision event stream is not contiguous")
                if event.previous_event_sha256 != previous.event_sha256:
                    raise LiveSupervisionError("Live supervision event hash chain changed")
            self._events.append(event)
            if len(self._events) > self._maximum_event_records:
                self._events = self._events[-self._maximum_event_records :]

    def begin_new_event_stream(self, *, run_id: str) -> None:
        """Reset only phase-local run events while preserving read-only scene state.

        The coarse and fine coordinators use separate append-only event stores whose
        sequence counters each begin at zero.  A schema-5 handoff may reuse this
        observer only when both stores carry the same durable run identity.  The
        method clears the in-memory event suffix and obsolete prepared trajectory,
        but deliberately retains copied perception, accumulated coarse point clouds,
        stopped joint samples and the next append-only snapshot sequence.

        This method exposes no command, approval, executor, or robot reference.
        """

        identity = run_id.strip()
        if not identity:
            raise ValueError("Live event-stream run identity must be non-empty")
        with self._lock:
            if self._existing_session_id not in {None, identity}:
                raise LiveSupervisionError(
                    "Live timeline already belongs to another supervised run"
                )
            if self._events and any(event.run_id != identity for event in self._events):
                raise LiveSupervisionError("Cannot reset a live event stream for a different run")
            self._events.clear()
            self._prepared = None
            self._planned_joint_path = None
            self._existing_session_id = identity

    def observe_perception(
        self,
        result: PerceptionCycleResult,
        *,
        coarse_readback: _CoarseScanViewReadback | None = None,
    ) -> None:
        """Time one ingest without changing the callback's typed contract."""

        if type(result) is not PerceptionCycleResult:
            # Preserve the original validation path without inspecting an
            # untrusted protocol substitute for diagnostic identity.
            self._observe_perception_transaction(
                result,
                coarse_readback=coarse_readback,
            )
            return
        recorder = try_create_performance_timing(
            transaction_kind="live_perception_ingest",
            identity={
                "view_id": result.bundle.view_id,
                "sequence_index": result.bundle.sequence_index,
                "frame_number": result.bundle.stereo.frame_number,
            },
        )
        if recorder is None:
            self._observe_perception_transaction(
                result,
                coarse_readback=coarse_readback,
            )
            return
        status = "failed"
        error: str | None = None
        try:
            with activate_performance_timing(recorder), performance_span(
                "live.perception_ingest"
            ):
                self._observe_perception_transaction(
                    result,
                    coarse_readback=coarse_readback,
                )
            status = "completed"
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            recorder.write_best_effort(
                self._root
                / "performance_diagnostics"
                / f"perception_{result.bundle.sequence_index:08d}.json",
                status=status,
                error=error,
            )

    def _observe_perception_transaction(
        self,
        result: PerceptionCycleResult,
        *,
        coarse_readback: _CoarseScanViewReadback | None = None,
    ) -> None:
        """Verify and copy one coordinator-accepted stopped perception result."""

        if type(result) is not PerceptionCycleResult:
            raise LiveSupervisionError("Live supervision requires a typed perception result")
        mapping = result.stored_occupancy
        if (
            mapping.motion_eligible is not True
            or mapping.verification_status != "full_semantic_verified_for_motion_preflight"
            or not mapping.frame_evidence
        ):
            raise LiveSupervisionError("Live occupancy lacks full semantic verification")
        evidence = mapping.frame_evidence[-1]
        bundle = result.bundle
        if (
            evidence.source_view_id != bundle.view_id
            or evidence.source_sequence_index != bundle.sequence_index
            or evidence.frame_number != bundle.stereo.frame_number
        ):
            raise LiveSupervisionError("Live occupancy and captured view identities differ")
        stereo = read_stereo_inference(result.stereo_inference_path)
        observation = stereo.observation
        if (
            observation.source_view_id != bundle.view_id
            or observation.source_sequence_index != bundle.sequence_index
            or observation.rectified.source_frame_number != bundle.stereo.frame_number
        ):
            raise LiveSupervisionError("Live stereo and captured view identities differ")

        current_science = _read_current_science_view(
            result,
            coarse_readback=coarse_readback,
        )
        current_points = current_science.points_m
        with self._lock:
            if current_points is not None:
                required_source_values = (
                    current_science.asset,
                    current_science.source_kind,
                    current_science.source_view_id,
                    current_science.source_sequence_index,
                    current_science.source_frame_number,
                    current_science.metadata_path,
                    current_science.metadata_sha256,
                    current_science.point_array_path,
                    current_science.point_array_file_sha256,
                    current_science.points_f64le_sha256,
                )
                if any(value is None for value in required_source_values):
                    raise LiveSupervisionError(
                        "Display points lack a complete immutable physical-source binding"
                    )
                current_points, current_keys = _display_voxel_representatives(
                    current_points,
                    maximum_points=_MAXIMUM_CURRENT_DISPLAY_POINTS,
                )
                entry = self._display_registry.append(
                    source_kind=str(current_science.source_kind),
                    view_id=str(current_science.source_view_id),
                    source_sequence_index=int(current_science.source_sequence_index),
                    source_frame_number=int(current_science.source_frame_number),
                    metadata_path=Path(current_science.metadata_path),
                    metadata_sha256=str(current_science.metadata_sha256),
                    point_array_path=Path(current_science.point_array_path),
                    point_array_file_sha256=str(current_science.point_array_file_sha256),
                    points_f64le_sha256=str(current_science.points_f64le_sha256),
                    raw_point_count=int(len(current_science.points_m)),
                    voxel_point_count=int(len(current_points)),
                    display_algorithm=_DISPLAY_UNION_ALGORITHM,
                    display_voxel_size_m=_DISPLAY_VOXEL_SIZE_M,
                    maximum_current_points=_MAXIMUM_CURRENT_DISPLAY_POINTS,
                    created_at_utc=datetime.fromisoformat(evidence.captured_at_utc).astimezone(UTC),
                )
                if entry.physical_source_id not in self._registered_physical_source_ids:
                    self._merge_display_points(current_points, current_keys)
                    self._registered_physical_source_ids.add(entry.physical_source_id)
            registered_view_count = len(self._registered_physical_source_ids)
            ordered_union = tuple(
                self._display_union[key][1] for key in sorted(self._display_union)
            )

        fused = (
            _readonly_array(ordered_union, dtype=np.float64).reshape((-1, 3))
            if ordered_union
            else None
        )
        front, back, fin_front, fin_back, registered_count = _coverage_values(result.coverage_path)
        report = evidence.self_mask
        confidence = observation.result.confidence
        sensor = _SensorState(
            frame_number=bundle.stereo.frame_number,
            captured_at_utc=datetime.fromisoformat(evidence.captured_at_utc).astimezone(UTC),
            left_ir=_readonly_array(observation.rectified.left_ir, dtype=np.uint8),
            right_ir=_readonly_array(observation.rectified.right_ir, dtype=np.uint8),
            depth_m=_readonly_array(observation.depth_m, dtype=np.float32),
            confidence=(
                None if confidence is None else _readonly_array(confidence, dtype=np.float32)
            ),
            occupancy_quality_evidence_sha256=evidence.quality_evidence_hash,
            valid_depth_fraction=evidence.valid_depth_fraction,
            stereo_valid_fraction=evidence.stereo_valid_fraction,
            confidence_accepted_fraction=evidence.confidence_accepted_fraction,
            mean_accepted_confidence=evidence.mean_accepted_confidence,
            lr_consistency_threshold_px=evidence.lr_consistency_threshold_px,
            fk_tcp_translation_error_m=evidence.fk_tcp_translation_error_m,
            fk_tcp_rotation_error_deg=evidence.fk_tcp_rotation_error_deg,
            projected_robot_pixel_count=report.projected_robot_pixels,
            measured_valid_pixel_count=report.measured_valid_pixels,
            depth_matched_pixel_count=report.depth_matched_pixels,
            masked_valid_pixel_count=report.masked_valid_pixels,
            retained_valid_pixel_count=report.retained_valid_pixels,
        )
        assets = [
            _asset(
                "raw_session",
                "current_raw_session_manifest",
                "biblade_fusion.session_manifest",
                result.raw_session_path / "manifest.json",
            ),
            _asset(
                "stereo_inference",
                "current_foundation_stereo_metadata",
                "biblade_fusion.stereo_inference",
                result.stereo_inference_path / "metadata.json",
            ),
            _asset(
                "occupancy_mapping",
                "current_occupancy_mapping_metadata",
                "biblade_fusion.occupancy_mapping",
                result.occupancy_mapping_path / "metadata.json",
            ),
            _asset(
                "inference_stationarity",
                "current_inference_stationarity",
                "biblade_fusion.inference_stationarity",
                result.inference_stationarity_path,
            ),
        ]
        if result.blade_foreground_path is not None:
            assets.append(
                _asset(
                    "blade_foreground",
                    "current_blade_foreground_metadata",
                    "biblade_fusion.blade_foreground",
                    result.blade_foreground_path / "metadata.json",
                )
            )
        if current_science.asset is not None:
            assets.append(current_science.asset)
        if result.coverage_path is not None:
            assets.append(
                _asset(
                    "surface_coverage",
                    "current_surface_coverage_generation",
                    "biblade_fusion.surface_coverage",
                    result.coverage_path / "coverage.json",
                )
            )

        selected = bundle.selected_robot_state
        state = _PerceptionState(
            robot_joint_positions_rad=tuple(float(value) for value in selected.joint_positions_rad),
            robot_mode=selected.robot_mode,
            safety_status=selected.safety_status,
            camera_pose_matrix=tuple(
                tuple(float(value) for value in row) for row in evidence.base_t_camera_matrix
            ),
            captured_at_utc=sensor.captured_at_utc,
            occupancy=mapping.snapshot,
            sensor=sensor,
            science=_ScienceState(
                current_points_m=current_points,
                fused_points_m=fused,
                front_coverage=front,
                back_coverage=back,
                fin_front_coverage=fin_front,
                fin_back_coverage=fin_back,
                registered_view_count=max(registered_count, registered_view_count),
                model_version=(
                    f"{_DISPLAY_UNION_ALGORITHM}:{registered_view_count}"
                    if fused is not None
                    else "unavailable"
                ),
            ),
            assets=tuple(assets),
        )
        with self._lock:
            self._perception = state
            actual = state.robot_joint_positions_rad
            if not self._sampled_actual_joints or not np.allclose(
                actual,
                self._sampled_actual_joints[-1],
                rtol=0.0,
                atol=1e-12,
            ):
                self._sampled_actual_joints.append(actual)

    def observe_prepared_segment(self, prepared: PreparedSegment | None) -> None:
        """Copy a detached plan summary; this method cannot approve or execute it."""

        if prepared is not None and type(prepared) is not PreparedSegment:
            raise LiveSupervisionError("Live supervision requires a typed prepared segment")
        with self._lock:
            self._prepared = prepared
            self._planned_joint_path = (
                None
                if prepared is None
                else _readonly_array(_joint_path(prepared), dtype=np.float64)
            )

    def __call__(self, status: ExperimentStatusSnapshot) -> None:
        self.publish_status(status)

    def publish_status(
        self,
        status: ExperimentStatusSnapshot,
    ) -> StoredSupervisorySnapshot:
        """Time one synchronous publication without changing fail-closed behavior."""

        if type(status) is not ExperimentStatusSnapshot:
            return self._publish_status_transaction(status)
        # Capture the append-only sequence before publication advances it so the
        # best-effort diagnostic remains bound to the snapshot it timed.
        diagnostic_sequence = self._next_sequence
        recorder = try_create_performance_timing(
            transaction_kind="live_snapshot_publication",
            identity={
                "run_id": status.run_id,
                "phase": status.phase,
                "disposition": status.disposition.value,
                "snapshot_sequence": diagnostic_sequence,
            },
        )
        if recorder is None:
            return self._publish_status_transaction(status)
        timing_status = "failed"
        error: str | None = None
        try:
            with activate_performance_timing(recorder), performance_span(
                "live.snapshot_publication"
            ):
                stored = self._publish_status_transaction(status)
            timing_status = "completed"
            return stored
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            recorder.write_best_effort(
                self._root
                / "performance_diagnostics"
                / (
                    f"snapshot_{diagnostic_sequence:08d}_"
                    f"{_safe_token(status.phase)}.json"
                ),
                status=timing_status,
                error=error,
            )

    def _publish_status_transaction(
        self,
        status: ExperimentStatusSnapshot,
    ) -> StoredSupervisorySnapshot:
        """Atomically append one snapshot suitable for GUI follow mode."""

        if type(status) is not ExperimentStatusSnapshot:
            raise LiveSupervisionError("Live supervision requires a typed experiment status")
        if status.motion_command_capable is not False:
            raise LiveSupervisionError("Experiment status unexpectedly exposes motion commands")
        if self._existing_session_id not in {None, status.run_id}:
            raise LiveSupervisionError("Live timeline already belongs to another supervised run")
        if any(event.run_id != status.run_id for event in self._events):
            raise LiveSupervisionError("Live events and experiment status use different runs")
        with self._lock:
            stored, fail_closed_reason = self._publish_locked(status)
        if fail_closed_reason is not None:
            raise LiveSupervisionError(fail_closed_reason)
        return stored

    def _publish_locked(
        self,
        status: ExperimentStatusSnapshot,
    ) -> tuple[StoredSupervisorySnapshot, str | None]:
        now = self._utc_now()
        state = self._perception
        prepared = self._prepared
        planned_joints = self._planned_joint_path
        actual_joints = (
            np.asarray(self._sampled_actual_joints, dtype=np.float64)
            if self._sampled_actual_joints
            else None
        )
        requirements: list[str] = []
        approval_phase = status.disposition is ExperimentDisposition.WAITING_APPROVAL
        bootstrap_mapping_ready = bool(
            approval_phase
            and status.phase == "bootstrap_motion_ready"
            and state is not None
            and state.occupancy.map_state is OccupancyMapState.MAPPING
            and prepared is not None
            and prepared.ready_for_approval
            and prepared.proposal.bootstrap_mapping_prefix
        )
        motion_relevant_phase = approval_phase or status.phase in {
            "map_ready",
            "planning",
            "preflighting",
            "executing",
        }
        if motion_relevant_phase:
            if state is None:
                requirements.append("live_perception_unavailable")
            elif (
                state.occupancy.map_state is not OccupancyMapState.MAP_READY
                and not bootstrap_mapping_ready
            ):
                requirements.append("live_occupancy_not_map_ready")
            if self._collision_geometry is None:
                requirements.append("live_collision_mesh_unavailable")
        if approval_phase:
            if prepared is None or not prepared.ready_for_approval:
                requirements.append("live_prepared_trajectory_unavailable")
            if planned_joints is None:
                requirements.append("live_planned_joint_trajectory_unavailable")

        if state is None:
            occupancy_state = "UNREADY"
            occupancy_version = "unavailable"
            occupancy_content_hash = None
            voxel_size = self._layout.occupancy_voxel_size_m
            bounds_min = self._layout.occupancy_bounds_min_m
            bounds_max = self._layout.occupancy_bounds_max_m
            map_age_s = 0.0
            frame_count = 0
            occupied = free = None
            joints: tuple[float, ...] = ()
            joint_names: tuple[str, ...] = ()
            robot_mode = "UNKNOWN"
            safety_status = "UNKNOWN"
            camera_pose = None
            link_origins = None
            sensor_state = None
            science = _ScienceState()
            external_assets: tuple[_AssetSource, ...] = ()
        else:
            map_snapshot = state.occupancy
            occupancy_state = {
                OccupancyMapState.UNMAPPED: "UNREADY",
                OccupancyMapState.MAPPING: (
                    "BOOTSTRAP_READY" if bootstrap_mapping_ready else "UNREADY"
                ),
                OccupancyMapState.MAP_READY: "READY",
                OccupancyMapState.STALE: "STALE",
            }[map_snapshot.map_state]
            occupancy_version = map_snapshot.version
            occupancy_content_hash = (
                map_snapshot.content_hash
                if occupancy_state in {"BOOTSTRAP_READY", "READY", "STALE"}
                else None
            )
            voxel_size = map_snapshot.voxel_size_m
            bounds_min = map_snapshot.origin_m
            bounds_max = map_snapshot.bounds_max_m
            freshness_origin = map_snapshot.rebuild_started_at_utc or map_snapshot.created_at_utc
            map_age_s = max(0.0, (now - freshness_origin).total_seconds())
            frame_count = len(map_snapshot.source_view_ids)
            occupied = _voxel_centres(map_snapshot.occupied_indices, map_snapshot)
            free = _voxel_centres(map_snapshot.free_indices, map_snapshot)
            joints = state.robot_joint_positions_rad
            joint_names = self._layout.joint_names
            robot_mode = state.robot_mode
            safety_status = state.safety_status
            camera_pose = TransformSnapshot(
                parent_frame="base",
                child_frame="left_rectified",
                matrix=state.camera_pose_matrix,
            )
            link_origins = _link_origins(self._kinematics, joints)
            sensor_state = state.sensor
            science = state.science
            external_assets = state.assets

        collision_vertices = None
        collision_triangles = None
        collision_binding: dict[str, object] | None = None
        if state is not None and self._collision_geometry is not None:
            collision_vertices, collision_triangles, collision_binding = (
                self._collision_geometry.base_mesh(joints)
            )

        try:
            self._display_registry.verify()
        except (OSError, TypeError, ValueError) as exc:
            raise LiveSupervisionError(
                f"Live display-source registry changed before publication: {exc}"
            ) from exc
        registry_head = self._display_registry.head

        planned_tcp = _tcp_path(self._kinematics, planned_joints)
        actual_tcp = _tcp_path(self._kinematics, actual_joints)
        system_state = _system_state(status, requirements)
        blocking_reasons = tuple(dict.fromkeys((*status.blocking_reasons, *requirements)))
        if system_state == "BLOCKED" and not blocking_reasons:
            blocking_reasons = (f"experiment_phase:{status.phase}",)
        plan_state = _plan_state(status, requirements)
        fail_closed_reason = (
            ";".join(requirements) if motion_relevant_phase and requirements else None
        )

        output = self._root / (
            f"{self._next_sequence:08d}_{_safe_token(status.phase)}_{uuid4().hex[:8]}"
        )
        with TemporaryDirectory(prefix=".live-supervision-assets-", dir=self._root) as temp:
            trajectory_path = Path(temp) / "joint_trajectory.json"
            display_manifest_path = Path(temp) / "display_union_manifest.json"
            registry_head_path = Path(temp) / "display_source_registry_head.json"
            collision_binding_path = Path(temp) / "live_collision_mesh_binding.json"
            registry_head_payload = {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.display_source_registry_head",
                "motion_authorized": False,
                "scientific_fusion": False,
                "registry_root": str(registry_head.root),
                "entry_count": registry_head.entry_count,
                "head_entry_sha256": registry_head.head_entry_sha256,
                "head_entry_path": (
                    str(registry_head.head_entry_path)
                    if registry_head.head_entry_path is not None
                    else None
                ),
                "head_entry_file_sha256": registry_head.head_entry_file_sha256,
            }
            registry_head_path.write_text(
                json.dumps(
                    registry_head_payload,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if collision_binding is not None:
                collision_binding_path.write_text(
                    json.dumps(
                        collision_binding,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            display_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_kind": "biblade_fusion.bounded_display_union",
                        "motion_authorized": False,
                        "scientific_fusion": False,
                        "semantic": "bounded_display_union_not_scientific_fusion",
                        "algorithm": _DISPLAY_UNION_ALGORITHM,
                        "display_voxel_size_m": _DISPLAY_VOXEL_SIZE_M,
                        "maximum_current_points": _MAXIMUM_CURRENT_DISPLAY_POINTS,
                        "maximum_fused_points": _MAXIMUM_FUSED_DISPLAY_POINTS,
                        "retained_fused_point_count": len(self._display_union),
                        "source_registry": registry_head_payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            trajectory_payload = {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.read_only_joint_trajectory",
                "motion_authorized": False,
                "joint_names": list(self._layout.joint_names),
                "current_joint_positions_rad": list(joints) if joints else None,
                "planned_servoj_joint_path_rad": (
                    planned_joints.tolist() if planned_joints is not None else None
                ),
                "status_sampled_actual_joint_path_rad": (
                    actual_joints.tolist() if actual_joints is not None else None
                ),
                "actual_path_semantic": (
                    "stopped_perception_samples_only_not_high_rate_servoj_tracking"
                ),
                "phase": status.phase,
                "run_id": status.run_id,
                "latest_event_sha256": status.latest_event_sha256,
            }
            trajectory_path.write_text(
                json.dumps(
                    trajectory_payload,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with AtomicSupervisorySnapshotWriter(output) as writer:
                link_ref = _write_array(
                    writer,
                    "robot_link_origins_base_m",
                    link_origins,
                    semantic="calibrated_es68_link_origins_base_m",
                )
                planned_tcp_ref = _write_array(
                    writer,
                    "planned_tcp_path_base_m",
                    planned_tcp,
                    semantic="prepared_servoj_tcp_path_base_m",
                )
                actual_tcp_ref = _write_array(
                    writer,
                    "sampled_actual_tcp_path_base_m",
                    actual_tcp,
                    semantic="stopped_perception_sampled_tcp_path_base_m_not_servoj_tracking",
                )
                collision_vertices_ref = _write_array(
                    writer,
                    "robot_collision_mesh_vertices_base_m",
                    collision_vertices,
                    semantic="active_hash_bound_collision_stl_vertices_base_m",
                )
                collision_triangles_ref = _write_array(
                    writer,
                    "robot_collision_mesh_triangles",
                    collision_triangles,
                    semantic="active_hash_bound_collision_stl_triangle_indices",
                )
                occupied_ref = _write_array(
                    writer,
                    "occupied_voxel_centres_base_m",
                    occupied,
                    semantic="occupied_voxel_centres_base_m",
                )
                free_ref = _write_array(
                    writer,
                    "free_voxel_centres_base_m",
                    free,
                    semantic="multi_view_confirmed_free_voxel_centres_base_m",
                )
                current_ref = _write_array(
                    writer,
                    "current_registered_blade_points_base_m",
                    science.current_points_m,
                    semantic="current_registered_blade_points_base_m",
                )
                fused_ref = _write_array(
                    writer,
                    "registered_blade_view_union_base_m",
                    science.fused_points_m,
                    semantic="display_registered_view_union_base_m_not_tsdf_fusion",
                )
                left_ref = right_ref = depth_ref = confidence_ref = None
                if sensor_state is not None:
                    left_ref = _write_array(
                        writer,
                        "left_rectified_ir",
                        sensor_state.left_ir,
                        semantic="foundation_stereo_left_rectified_ir",
                    )
                    right_ref = _write_array(
                        writer,
                        "right_rectified_ir",
                        sensor_state.right_ir,
                        semantic="foundation_stereo_right_rectified_ir",
                    )
                    depth_ref = _write_array(
                        writer,
                        "foundation_stereo_depth_m",
                        sensor_state.depth_m,
                        semantic="foundation_stereo_depth_m_nan_invalid",
                        allow_nonfinite=True,
                    )
                    confidence_ref = _write_array(
                        writer,
                        "foundation_stereo_confidence",
                        sensor_state.confidence,
                        semantic="foundation_stereo_confidence",
                        allow_nonfinite=True,
                    )

                assets = [
                    writer.write_asset(
                        "joint_trajectory",
                        trajectory_path,
                        logical_name="read_only_joint_trajectory",
                        kind="biblade_fusion.read_only_joint_trajectory",
                        version="1",
                        expected_sha256=_sha256(trajectory_path),
                    ),
                    writer.write_asset(
                        "display_union_manifest",
                        display_manifest_path,
                        logical_name="bounded_display_union_manifest",
                        kind="biblade_fusion.bounded_display_union",
                        version="1",
                        expected_sha256=_sha256(display_manifest_path),
                    ),
                    writer.write_asset(
                        "display_source_registry_head",
                        registry_head_path,
                        logical_name="append_only_display_source_registry_head",
                        kind="biblade_fusion.display_source_registry_head",
                        version="1",
                        expected_sha256=_sha256(registry_head_path),
                    ),
                ]
                if registry_head.head_entry_path is not None:
                    assets.append(
                        writer.write_asset(
                            "display_source_registry_entry",
                            registry_head.head_entry_path,
                            logical_name="append_only_display_source_registry_head_entry",
                            kind="biblade_fusion.display_source_registry_entry",
                            version="1",
                            expected_sha256=registry_head.head_entry_file_sha256,
                        )
                    )
                if collision_binding is not None and self._collision_geometry is not None:
                    assets.extend(
                        (
                            writer.write_asset(
                                "live_collision_mesh_binding",
                                collision_binding_path,
                                logical_name="active_live_collision_mesh_binding",
                                kind="biblade_fusion.live_collision_mesh_binding",
                                version="1",
                                expected_sha256=_sha256(collision_binding_path),
                            ),
                            writer.write_asset(
                                "active_collision_manifest",
                                self._collision_geometry.manifest_path,
                                logical_name="active_es68_d435i_collision_manifest",
                                kind="biblade_fusion.es68_d435i_collision.v1",
                                version="1",
                                expected_sha256=self._collision_geometry.manifest_sha256,
                            ),
                        )
                    )
                for source in external_assets:
                    assets.append(
                        writer.write_asset(
                            source.name,
                            source.path,
                            logical_name=source.logical_name,
                            kind=source.kind,
                            version=source.version,
                            expected_sha256=source.sha256,
                        )
                    )

                feedback_age_ms = (
                    None
                    if state is None
                    else max(0.0, (now - state.captured_at_utc).total_seconds() * 1000.0)
                )
                snapshot = SupervisorySnapshot(
                    schema_version=SUPERVISORY_SNAPSHOT_SCHEMA_VERSION,
                    snapshot_id=f"{status.run_id}:{self._next_sequence}:{uuid4().hex}",
                    sequence=self._next_sequence,
                    created_at_utc=now,
                    source_session_id=status.run_id,
                    safety=SafetySnapshot(
                        system_state=system_state,
                        viewer_mode="READ_ONLY",
                        viewer_motion_command_capable=False,
                        blocking_reasons=blocking_reasons,
                        feedback_age_ms=feedback_age_ms,
                    ),
                    robot=RobotSceneSnapshot(
                        model_id=self._layout.model_id,
                        base_frame="base",
                        robot_mode=robot_mode,
                        safety_status=safety_status,
                        joint_names=joint_names,
                        joint_positions_rad=joints,
                        link_origins_base_m=link_ref,
                        collision_mesh_vertices_base_m=collision_vertices_ref,
                        collision_mesh_triangles=collision_triangles_ref,
                        planned_tcp_path_base_m=planned_tcp_ref,
                        actual_tcp_path_base_m=actual_tcp_ref,
                        camera_pose=camera_pose,
                    ),
                    occupancy=OccupancySnapshot(
                        frame_id="base",
                        state=occupancy_state,
                        version=occupancy_version,
                        content_sha256=occupancy_content_hash,
                        voxel_size_m=voxel_size,
                        bounds_min_m=bounds_min,
                        bounds_max_m=bounds_max,
                        age_s=map_age_s,
                        integrated_frame_count=frame_count,
                        occupied_centres_m=occupied_ref,
                        free_centres_m=free_ref,
                    ),
                    reconstruction=ReconstructionSnapshot(
                        frame_id="base",
                        model_version=science.model_version,
                        current_points_m=current_ref,
                        fused_points_m=fused_ref,
                        front_coverage=science.front_coverage,
                        back_coverage=science.back_coverage,
                        fin_front_coverage=science.fin_front_coverage,
                        fin_back_coverage=science.fin_back_coverage,
                        registered_view_count=science.registered_view_count,
                        provenance_status=(
                            "CURRENT_RUN_VERIFIED"
                            if science.current_points_m is not None
                            else "UNAVAILABLE"
                        ),
                        provenance_reasons=(
                            ("bounded_display_union_is_not_scientific_fusion",)
                            if science.fused_points_m is not None
                            else ()
                        ),
                    ),
                    sensor=_sensor_snapshot(
                        sensor_state,
                        left_ref=left_ref,
                        right_ref=right_ref,
                        depth_ref=depth_ref,
                        confidence_ref=confidence_ref,
                    ),
                    plan=PlanSnapshot(
                        plan_id=(prepared.proposal.proposal_id if prepared is not None else "none"),
                        state=plan_state,
                        current_view_index=(
                            max(0, status.cycle_index - 1)
                            if status.current_view_id is not None
                            else None
                        ),
                        total_view_count=max(
                            status.cycle_index + (1 if status.proposed_view_id else 0),
                            1 if status.current_view_id is not None else 0,
                        ),
                        current_view_id=status.current_view_id,
                        next_view_id=status.proposed_view_id,
                        blocking_reasons=blocking_reasons,
                    ),
                    assets=tuple(assets),
                    events=self._event_records(status, now),
                )
                stored = writer.commit(snapshot)
        self._next_sequence += 1
        self._existing_session_id = status.run_id
        return stored, fail_closed_reason

    def _event_records(
        self,
        status: ExperimentStatusSnapshot,
        now: datetime,
    ) -> tuple[EventRecord, ...]:
        records = [
            EventRecord(
                timestamp_utc=datetime.fromisoformat(event.created_at_utc).astimezone(UTC),
                severity=(
                    "ERROR"
                    if event.phase in {"failed", "aborted"}
                    else "WARNING"
                    if event.phase == "motion_blocked"
                    else "INFO"
                ),
                category=event.event_type,
                message=(
                    f"phase={event.phase}; cycle={event.cycle_index}; "
                    f"event={event.event_sha256[:12]}"
                ),
            )
            for event in self._events
        ]
        records.append(
            EventRecord(
                timestamp_utc=now,
                severity=(
                    "ERROR" if status.disposition is ExperimentDisposition.BLOCKED else "INFO"
                ),
                category="experiment_status",
                message=(
                    f"phase={status.phase}; disposition={status.disposition.value}; "
                    "motion_command_capable=false"
                ),
            )
        )
        return tuple(records)

    def _utc_now(self) -> datetime:
        value = self._utc_clock()
        if value.tzinfo is None:
            raise LiveSupervisionError("Live supervision UTC clock must be timezone-aware")
        return value.astimezone(UTC)


def _write_array(
    writer: AtomicSupervisorySnapshotWriter,
    name: str,
    array: np.ndarray | None,
    *,
    semantic: str,
    allow_nonfinite: bool = False,
):
    if array is None:
        return None
    return writer.write_array(
        name,
        array,
        semantic=semantic,
        allow_nonfinite=allow_nonfinite,
    )


def _sensor_snapshot(
    state: _SensorState | None,
    *,
    left_ref,
    right_ref,
    depth_ref,
    confidence_ref,
) -> SensorSnapshot:
    if state is None:
        return SensorSnapshot()
    return SensorSnapshot(
        source="FOUNDATION_STEREO",
        frame_number=state.frame_number,
        left_ir=left_ref,
        right_ir=right_ref,
        depth_m=depth_ref,
        confidence=confidence_ref,
        captured_at_utc=state.captured_at_utc,
        occupancy_quality_evidence_sha256=(state.occupancy_quality_evidence_sha256),
        valid_depth_fraction=state.valid_depth_fraction,
        stereo_valid_fraction=state.stereo_valid_fraction,
        confidence_accepted_fraction=state.confidence_accepted_fraction,
        mean_accepted_confidence=state.mean_accepted_confidence,
        lr_consistency_threshold_px=state.lr_consistency_threshold_px,
        fk_tcp_translation_error_m=state.fk_tcp_translation_error_m,
        fk_tcp_rotation_error_deg=state.fk_tcp_rotation_error_deg,
        projected_robot_pixel_count=state.projected_robot_pixel_count,
        measured_valid_pixel_count=state.measured_valid_pixel_count,
        depth_matched_pixel_count=state.depth_matched_pixel_count,
        masked_valid_pixel_count=state.masked_valid_pixel_count,
        retained_valid_pixel_count=state.retained_valid_pixel_count,
    )


def _system_state(
    status: ExperimentStatusSnapshot,
    requirements: list[str],
) -> Literal["BLOCKED", "READY_FOR_EXTERNAL_APPROVAL", "EXECUTING", "STOPPED", "FAULT"]:
    if status.phase in {"failed", "aborted"}:
        return "FAULT"
    if status.disposition is ExperimentDisposition.COMPLETE:
        return "STOPPED"
    if status.phase == "executing" and not requirements:
        return "EXECUTING"
    if status.disposition is ExperimentDisposition.WAITING_APPROVAL and not requirements:
        return "READY_FOR_EXTERNAL_APPROVAL"
    return "BLOCKED"


def _plan_state(
    status: ExperimentStatusSnapshot,
    requirements: list[str],
) -> Literal[
    "NONE",
    "PLANNED",
    "PREFLIGHT_FAILED",
    "READY_FOR_EXTERNAL_APPROVAL",
    "EXECUTING",
    "PAUSED",
    "COMPLETED",
    "ABORTED",
]:
    if status.phase in {"failed", "aborted"}:
        return "ABORTED"
    if status.disposition is ExperimentDisposition.COMPLETE:
        return "COMPLETED"
    if status.phase == "executing":
        return "EXECUTING" if not requirements else "PREFLIGHT_FAILED"
    if status.disposition is ExperimentDisposition.WAITING_APPROVAL:
        return "READY_FOR_EXTERNAL_APPROVAL" if not requirements else "PREFLIGHT_FAILED"
    if status.disposition is ExperimentDisposition.BLOCKED:
        return "PREFLIGHT_FAILED"
    if status.proposed_view_id is not None:
        return "PLANNED"
    return "NONE"


def _safe_token(value: str) -> str:
    token = "".join(character if character.isalnum() else "_" for character in value)
    token = token.strip("_")[:48]
    return token or "status"


__all__ = [
    "LiveCollisionGeometry",
    "LiveSupervisionBridge",
    "LiveSupervisionError",
    "LiveSupervisionLayout",
]
