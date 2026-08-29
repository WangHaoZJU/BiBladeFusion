"""Bridge verified project assets into an offline supervisory replay snapshot.

This workflow is deliberately read-only.  It has no device or executor imports and
cannot turn historical evidence into a live motion authorization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from biblade_fusion.core.settings import OccupancyConfig
from biblade_fusion.mapping import OccupancyMapState
from biblade_fusion.mapping.robot_depth_renderer import Es68D435iRobotDepthRenderer
from biblade_fusion.robotics import (
    CS68_COLLISION_LINK_NAMES,
    ES68_JOINT_NAMES,
    Es68KinematicModel,
    Es68ModelResources,
)
from biblade_fusion.storage.coarse_model import read_coarse_model_summary
from biblade_fusion.storage.motion_preflight import read_motion_preflight
from biblade_fusion.storage.occupancy_mapping import read_occupancy_mapping_for_replay
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
from biblade_fusion.storage.stereo_inference import read_stereo_inference
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
)
from biblade_fusion.supervision.storage import AtomicSupervisorySnapshotWriter
from biblade_fusion.workflows.occupancy_mapping import occupancy_physical_source_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: datetime | str, *, label: str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _verified_array(root: Path, record: dict[str, Any], *, label: str) -> np.ndarray:
    resolved_root = root.resolve()
    relative = Path(str(record["path"]))
    path = (resolved_root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(resolved_root):
        raise ValueError(f"{label} array escapes its artifact root: {relative}")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"{label} array checksum mismatch: {relative}")
    array = np.load(path, allow_pickle=False)
    if array.dtype.hasobject:
        raise ValueError(f"{label} object array is forbidden: {relative}")
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"{label} array manifest mismatch: {relative}")
    return array


def _source_root(record: dict[str, Any], *, label: str) -> Path:
    root = Path(str(record["root"])).resolve()
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"{label} source escapes its artifact root")
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"{label} source checksum mismatch: {path}")
    return root


def _voxel_centres(
    indices: frozenset[tuple[int, int, int]],
    *,
    origin_m: tuple[float, float, float],
    voxel_size_m: float,
) -> np.ndarray:
    """Convert sparse integer cells to metric cell centres, never cell corners."""

    if not indices:
        return np.empty((0, 3), dtype=np.float64)
    ordered = np.asarray(sorted(indices), dtype=np.float64).reshape(-1, 3)
    return np.asarray(origin_m, dtype=np.float64) + (ordered + 0.5) * voxel_size_m


def _coverage(
    patches: list[dict[str, Any]],
    *,
    side: Literal["front", "back"],
    fin: bool,
) -> float:
    selected = [
        item
        for item in patches
        if str(item.get("side")) == side and str(item.get("region", "")).startswith("fin_") is fin
    ]
    weights = np.asarray(
        [max(0, int(item.get("reference_point_count", 0))) for item in selected],
        dtype=np.float64,
    )
    if not len(weights) or float(np.sum(weights)) <= 0.0:
        return 0.0
    fractions = np.asarray(
        [float(item.get("coverage_fraction", 0.0)) for item in selected],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(fractions)) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("Coarse-model coverage contains an invalid fraction")
    return float(np.average(fractions, weights=weights))


def _registration_rmse(refinements: list[dict[str, Any]]) -> float | None:
    accepted = [
        item
        for item in refinements
        if bool(item.get("accepted"))
        and item.get("rmse_after_m") is not None
        and int(item.get("correspondence_count", 0)) > 0
    ]
    if not accepted:
        return None
    weights = np.asarray([int(item["correspondence_count"]) for item in accepted], dtype=np.float64)
    values = np.asarray([float(item["rmse_after_m"]) for item in accepted])
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Coarse-model accepted registration RMSE is invalid")
    return float(np.average(values, weights=weights))


@dataclass(frozen=True, slots=True)
class _SourceAsset:
    logical_name: str
    kind: str
    source_path: Path
    sha256: str
    version: str | None


def _asset(
    logical_name: str,
    kind: str,
    root: Path,
    filename: str,
    version: str | int | None,
) -> _SourceAsset:
    path = (root.resolve() / filename).resolve()
    if not path.is_file():
        raise ValueError(f"Supervisory source asset does not exist: {path}")
    return _SourceAsset(
        logical_name,
        kind,
        path,
        _sha256(path),
        str(version) if version is not None else None,
    )


@dataclass(frozen=True, slots=True)
class _OccupancySourceBinding:
    evidence: Any
    session_root: Path
    stereo_root: Path


def _occupancy_source_bindings(
    stored_occupancy: Any,
) -> dict[str, _OccupancySourceBinding]:
    records = stored_occupancy.metadata["frames"]
    evidence_items = stored_occupancy.frame_evidence
    if len(records) != len(evidence_items):
        raise ValueError("Occupancy frame records and evidence have different lengths")
    bindings: dict[str, _OccupancySourceBinding] = {}
    for index, (record, evidence) in enumerate(zip(records, evidence_items, strict=True)):
        session_root = _source_root(
            record["sources"]["session"], label=f"occupancy session {index}"
        )
        stereo_root = _source_root(
            record["sources"]["stereo_inference"],
            label=f"occupancy stereo inference {index}",
        )
        key = evidence.physical_source_id
        if key in bindings:
            raise ValueError(f"Ambiguous occupancy source identity: {key}")
        bindings[key] = _OccupancySourceBinding(evidence, session_root, stereo_root)
    return bindings


def _matrix(value: object, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
    ):
        raise ValueError(f"{label} must be a finite homogeneous 4x4 transform")
    return matrix


def _validate_reconstructed_view_binding(
    stored_view: Any,
    binding: _OccupancySourceBinding,
    *,
    mapping_context: Mapping[str, Any],
) -> None:
    """Prove one reconstructed view came from the matching occupancy frame."""

    view = stored_view.view
    evidence = binding.evidence
    key = (view.source_view_id, view.source_sequence_index, view.source_frame_number)
    expected_key = (
        evidence.source_view_id,
        evidence.source_sequence_index,
        evidence.frame_number,
    )
    if key != expected_key:
        raise ValueError("reconstructed view and occupancy evidence identify different frames")
    if view.depth_source != "foundation_stereo":
        raise ValueError("reconstructed view is not derived from FoundationStereo depth")

    source = stored_view.metadata.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("reconstructed view has no source contract")
    if Path(str(source.get("session"))).resolve() != binding.session_root:
        raise ValueError("reconstructed view belongs to a different acquisition session")
    stereo_source = source.get("stereo_inference")
    if stereo_source is None or Path(str(stereo_source)).resolve() != binding.stereo_root:
        raise ValueError("reconstructed view belongs to a different stereo artifact")
    if not np.allclose(
        view.joint_positions_rad,
        evidence.joint_positions_rad,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("reconstructed view joints differ from occupancy evidence")

    stored_stereo = read_stereo_inference(binding.stereo_root)
    stereo = stored_stereo.observation
    if (
        stereo.source_view_id != evidence.source_view_id
        or stereo.source_sequence_index != evidence.source_sequence_index
        or stereo.rectified.source_frame_number != evidence.frame_number
        or Path(str(stored_stereo.metadata["source"]["session"])).resolve() != binding.session_root
    ):
        raise ValueError("bound stereo artifact disagrees with occupancy evidence")
    if view.planning_intrinsics != stereo.rectified.calibration.left:
        raise ValueError("reconstructed view intrinsics differ from occupancy stereo")

    hand_eye = stored_view.metadata.get("hand_eye")
    if not isinstance(hand_eye, Mapping):
        raise ValueError("reconstructed view has no hand-eye provenance")
    hand_eye_source = hand_eye.get("source")
    if not isinstance(hand_eye_source, Mapping):
        raise ValueError("reconstructed view has no hand-eye source record")
    hand_eye_path = Path(str(hand_eye_source.get("path"))).resolve()
    if (
        not hand_eye_path.is_file()
        or _sha256(hand_eye_path) != str(hand_eye_source.get("sha256"))
        or hand_eye_path.stat().st_size != int(hand_eye_source.get("size_bytes", -1))
        or _sha256(hand_eye_path) != evidence.hand_eye_hash
    ):
        raise ValueError("reconstructed view hand-eye asset differs from occupancy evidence")

    robot_context = mapping_context.get("robot")
    hand_eye_context = mapping_context.get("hand_eye")
    rectified_context = mapping_context.get("rectified_stereo")
    occupancy_contract = mapping_context.get("occupancy_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (
            robot_context,
            hand_eye_context,
            rectified_context,
            occupancy_contract,
        )
    ):
        raise ValueError("occupancy mapping context lacks pose-chain evidence")
    flange_t_tcp = _matrix(robot_context["flange_T_tcp"], label="flange_T_tcp")
    flange_t_left_ir = _matrix(hand_eye_context["flange_T_left_ir"], label="flange_T_left_ir")
    left_rectified_t_left_ir = _matrix(
        rectified_context["left_rectified_T_left_ir"],
        label="left_rectified_T_left_ir",
    )
    tcp_t_left_ir = np.linalg.inv(flange_t_tcp) @ flange_t_left_ir
    stored_tcp_t_left_ir = _matrix(
        hand_eye.get("tcp_T_left_ir"), label="reconstructed tcp_T_left_ir"
    )
    stored_flange_t_left_ir = _matrix(
        hand_eye.get("flange_T_left_ir"),
        label="reconstructed flange_T_left_ir",
    )
    stored_flange_t_tcp = _matrix(hand_eye.get("flange_T_tcp"), label="reconstructed flange_T_tcp")
    if not np.allclose(
        stored_flange_t_left_ir,
        flange_t_left_ir,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("reconstructed flange-primary hand-eye differs from occupancy")
    if not np.allclose(stored_flange_t_tcp, flange_t_tcp, atol=1e-9, rtol=0.0):
        raise ValueError("reconstructed flange_T_tcp differs from occupancy context")
    if not np.allclose(stored_tcp_t_left_ir, tcp_t_left_ir, atol=1e-9, rtol=0.0):
        raise ValueError("reconstructed view hand-eye matrix differs from occupancy context")

    authority = view.pose_authority
    if authority is None:
        raise ValueError("reconstructed view lacks authoritative joints-to-FK pose evidence")
    expected_offsets = tuple(float(value) for value in robot_context["joint_zero_offsets_rad"])
    if authority.joint_zero_offsets_rad != expected_offsets:
        raise ValueError("reconstructed view and occupancy use different joint offsets")
    authority_pairs = (
        (
            authority.base_t_flange.matrix,
            evidence.base_t_flange_matrix,
            "base_T_flange",
        ),
        (
            authority.predicted_base_t_tcp.matrix,
            evidence.predicted_base_t_tcp_matrix,
            "predicted base_T_tcp",
        ),
        (
            authority.observed_base_t_tcp.matrix,
            evidence.observed_base_t_tcp_matrix,
            "observed base_T_tcp",
        ),
    )
    for reconstructed, mapped, label in authority_pairs:
        if not np.allclose(reconstructed, mapped, atol=1e-9, rtol=0.0):
            raise ValueError(f"reconstructed {label} differs from occupancy pose evidence")
    if not np.allclose(
        (
            authority.fk_tcp_translation_error_m,
            authority.fk_tcp_rotation_error_deg,
        ),
        (
            evidence.fk_tcp_translation_error_m,
            evidence.fk_tcp_rotation_error_deg,
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("reconstructed FK/TCP residual differs from occupancy evidence")
    expected_limits = (
        float(occupancy_contract["maximum_fk_tcp_translation_error_m"]),
        float(occupancy_contract["maximum_fk_tcp_rotation_error_deg"]),
    )
    if not np.allclose(
        (
            authority.maximum_fk_tcp_translation_error_m,
            authority.maximum_fk_tcp_rotation_error_deg,
        ),
        expected_limits,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("reconstructed and occupancy FK/TCP gates differ")

    expected_base_t_left_ir = (
        _matrix(evidence.base_t_flange_matrix, label="base_T_flange") @ flange_t_left_ir
    )
    expected_base_t_camera = expected_base_t_left_ir @ np.linalg.inv(left_rectified_t_left_ir)
    if not np.allclose(view.base_t_left_ir.matrix, expected_base_t_left_ir, atol=1e-9, rtol=0.0):
        raise ValueError("reconstructed base_T_left_ir differs from occupancy pose chain")
    if not np.allclose(
        view.base_t_projection_camera.matrix,
        expected_base_t_camera,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("reconstructed base_T_camera differs from authoritative FK chain")
    if not np.allclose(
        view.base_t_projection_camera.matrix,
        evidence.base_t_camera_matrix,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("reconstructed base_T_camera differs from the mapped camera pose")

    expected_mapping_camera = (
        _matrix(evidence.base_t_flange_matrix, label="base_T_flange")
        @ flange_t_left_ir
        @ np.linalg.inv(left_rectified_t_left_ir)
    )
    if not np.allclose(
        evidence.base_t_camera_matrix,
        expected_mapping_camera,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("occupancy base_T_camera does not reproduce from its context")


def _coarse_model_provenance(
    metadata: Mapping[str, Any],
    bindings: Mapping[str, _OccupancySourceBinding],
    *,
    mapping_context: Mapping[str, Any],
) -> tuple[Literal["CURRENT_RUN_VERIFIED", "INDEPENDENT_REFERENCE"], tuple[str, ...]]:
    reasons: list[str] = []
    sources = metadata.get("source_views")
    if not isinstance(sources, list) or not sources:
        return "INDEPENDENT_REFERENCE", ("coarse_model_has_no_source_views",)
    for index, source in enumerate(sources):
        try:
            if not isinstance(source, Mapping):
                raise ValueError("source record is not a mapping")
            stored_view = read_reconstructed_view(Path(str(source["path"])).resolve())
            view = stored_view.view
            reconstructed_source = stored_view.metadata["source"]
            session_root = Path(str(reconstructed_source["session"])).resolve()
            session_reader = SessionReader(session_root)
            descriptor = session_reader.descriptor(view.source_sequence_index)
            view_metadata = (
                session_root / descriptor.relative_path / "metadata.json"
            ).resolve()
            key = occupancy_physical_source_id(
                source_session_manifest_sha256=_sha256(session_root / "manifest.json"),
                source_session_view_metadata_sha256=_sha256(view_metadata),
                source_sequence_index=view.source_sequence_index,
                frame_number=view.source_frame_number,
                source_view_id=view.source_view_id,
            )
            binding = bindings.get(key)
            if binding is None:
                raise ValueError("source frame is absent from the occupancy evidence chain")
            _validate_reconstructed_view_binding(
                stored_view,
                binding,
                mapping_context=mapping_context,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            reasons.append(f"source_view_{index}_unbound:{exc}")
    if reasons:
        return "INDEPENDENT_REFERENCE", tuple(reasons)
    return "CURRENT_RUN_VERIFIED", ()


def _robot_scene_geometry(
    evidence: Any,
    offsets: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str, str | None, Path | None]:
    """Use the exact active renderer when its complete geometry hash matches evidence."""

    try:
        renderer = Es68D435iRobotDepthRenderer.from_active_resources(joint_zero_offsets_rad=offsets)
        if renderer.model_content_hash != evidence.robot_model_hash:
            raise ValueError("robot_geometry_hash_mismatch")
        import pinocchio as pin

        model = renderer.pinocchio_model
        configuration = model._to_configuration(evidence.joint_positions_rad)
        pin.forwardKinematics(model.model, model.data, configuration)
        pin.updateFramePlacements(model.model, model.data)
        link_origins = np.stack(
            [
                np.asarray(
                    model.data.oMf[int(model.model.getFrameId(name))].translation,
                    dtype=np.float64,
                )
                for name in CS68_COLLISION_LINK_NAMES
            ]
        )
        vertices: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        vertex_offset = 0
        for mesh in renderer.meshes:
            frame_id = int(model.model.getFrameId(mesh.link_name))
            if frame_id >= len(model.model.frames) or mesh.faces.shape[1] != 3:
                raise ValueError("renderer_mesh_contract_invalid")
            base_t_link = np.asarray(model.data.oMf[frame_id].homogeneous, dtype=np.float64)
            base_t_mesh = base_t_link @ mesh.link_t_mesh
            homogeneous = np.column_stack(
                (mesh.vertices_m, np.ones(len(mesh.vertices_m), dtype=np.float64))
            )
            vertices.append((base_t_mesh @ homogeneous.T).T[:, :3])
            triangles.append(np.asarray(mesh.faces, dtype=np.int64) + vertex_offset)
            vertex_offset += len(mesh.vertices_m)
        return (
            link_origins,
            np.vstack(vertices),
            np.vstack(triangles),
            f"es68-d435i:verified:{evidence.robot_model_hash[:16]}",
            None,
            renderer.template.manifest_path,
        )
    except (
        FileNotFoundError,
        ImportError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        resources = Es68ModelResources.packaged()
        fallback = Es68KinematicModel.from_resources(
            resources,
            joint_zero_offsets_rad=offsets,
        )
        transforms = fallback.link_transforms(evidence.joint_positions_rad)
        link_origins = np.stack([transforms[name][:3, 3] for name in CS68_COLLISION_LINK_NAMES])
        reason = (
            "robot_visualization_geometry_hash_mismatch"
            if isinstance(exc, ValueError) and str(exc) == "robot_geometry_hash_mismatch"
            else f"robot_visualization_model_unavailable:{type(exc).__name__}"
        )
        return (
            link_origins,
            None,
            None,
            f"es68-d435i:UNVERIFIED:historical-{evidence.robot_model_hash[:12]}",
            reason,
            None,
        )


def _historical_preflight(
    root: Path,
    *,
    occupancy_root: Path,
) -> tuple[PlanSnapshot, _SourceAsset, EventRecord, np.ndarray]:
    """Canonically re-derive a preflight, then expose it only as expired history."""

    resolved = root.resolve()
    path = resolved / "motion_preflight.json"
    try:
        stored = read_motion_preflight(resolved)
        payload = stored.metadata
        sources = payload["sources"]
        occupancy_record = sources.get("occupancy")
        if occupancy_record is None:
            raise ValueError("motion preflight has no bound occupancy source")
        bound_occupancy = _source_root(occupancy_record, label="motion-preflight occupancy")
        if bound_occupancy != occupancy_root.resolve():
            raise ValueError("motion preflight belongs to a different occupancy artifact")
        report = stored.report
        ordered = report.ordered_view_ids
        legs = report.legs
        reasons = tuple(
            dict.fromkeys(
                str(reason)
                for leg in legs
                for reason in (
                    *leg.preflight.blocking_reasons,
                    *leg.endpoint_consistency.blocking_reasons,
                )
            )
        )
        ready = report.ready_for_approval
        historical_reason = "historical_preflight_expired_requires_live_revalidation"
        plan = PlanSnapshot(
            plan_id=f"preflight:{_sha256(path)[:16]}",
            state="PREFLIGHT_FAILED",
            total_view_count=len(ordered),
            next_view_id=ordered[0] if ordered else None,
            blocking_reasons=tuple(dict.fromkeys((*reasons, historical_reason))),
        )
        asset = _asset(
            "motion_preflight",
            "biblade_fusion.motion_preflight",
            resolved,
            "motion_preflight.json",
            payload["schema_version"],
        )
        event = EventRecord(
            timestamp_utc=_utc(payload["evaluated_at_utc"], label="preflight timestamp"),
            severity="WARNING",
            category="motion_preflight",
            message=(
                "Canonically re-derived historical preflight for replay "
                f"(originally_ready={ready}); it is expired and blocked until live "
                "revalidation plus external approval"
            ),
        )
        planned_path = np.asarray(
            [np.asarray(leg.goal_base_t_tcp_matrix, dtype=np.float64)[:3, 3] for leg in legs],
            dtype=np.float64,
        ).reshape(-1, 3)
        return plan, asset, event, planned_path
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid historical motion preflight {resolved}: {exc}") from exc


def build_supervisory_replay_snapshot(
    output_dir: str | Path,
    *,
    source_occupancy: str | Path,
    source_stereo_inference: str | Path | None = None,
    source_reconstructed_view: str | Path | None = None,
    source_coarse_model: str | Path | None = None,
    source_motion_preflight: str | Path | None = None,
    created_at_utc: datetime | None = None,
) -> StoredSupervisorySnapshot:
    """Build one verified, self-contained and permanently non-executable replay."""

    created = _utc(created_at_utc or datetime.now(UTC), label="snapshot timestamp")
    occupancy_root = Path(source_occupancy).resolve()
    stored_occupancy = read_occupancy_mapping_for_replay(occupancy_root)
    if (
        stored_occupancy.motion_eligible
        or stored_occupancy.verification_status != "integrity_only_unverified_for_motion"
    ):
        raise ValueError("Supervisory replay requires permanently motion-ineligible occupancy")
    map_data = stored_occupancy.snapshot
    if not stored_occupancy.frame_evidence:
        raise ValueError("Occupancy artifact has no frame evidence")
    if not map_data.source_view_ids:
        raise ValueError("Occupancy artifact has no source-view identity")

    occupancy_config = OccupancyConfig.model_validate(
        stored_occupancy.metadata["configuration"]["occupancy"]
    )
    mapping_context = stored_occupancy.mapping_context.to_payload()
    bindings = _occupancy_source_bindings(stored_occupancy)
    frame_record = stored_occupancy.metadata["frames"][-1]
    evidence = stored_occupancy.frame_evidence[-1]
    latest_key = evidence.physical_source_id
    latest_binding = bindings.get(latest_key)
    if latest_binding is None or evidence.physical_source_id != map_data.source_view_ids[-1]:
        raise ValueError("Latest occupancy evidence does not match map source-view order")

    session_root = latest_binding.session_root
    latest_stereo_root = latest_binding.stereo_root
    session_reader = SessionReader(session_root)
    bundle = session_reader.load_bundle(evidence.source_sequence_index)
    if (
        bundle.sequence_index != evidence.source_sequence_index
        or bundle.stereo.frame_number != evidence.frame_number
        or not np.allclose(
            bundle.selected_robot_state.joint_positions_rad,
            evidence.joint_positions_rad,
            atol=1e-10,
            rtol=0.0,
        )
    ):
        raise ValueError("Latest occupancy evidence disagrees with its source session")

    age_s = (created - map_data.created_at_utc).total_seconds()
    if age_s < 0.0:
        raise ValueError("Snapshot timestamp predates its occupancy map")
    state_by_map = {
        OccupancyMapState.UNMAPPED: "UNREADY",
        OccupancyMapState.MAPPING: "UNREADY",
        OccupancyMapState.MAP_READY: "READY",
        OccupancyMapState.STALE: "STALE",
    }
    occupancy_state = (
        "STALE"
        if map_data.is_stale(created, occupancy_config.maximum_map_age_s)
        else state_by_map[map_data.map_state]
    )

    if source_stereo_inference is not None and (
        Path(source_stereo_inference).resolve() != latest_stereo_root
    ):
        raise ValueError("Explicit stereo artifact is not the occupancy map's latest source")
    stored_stereo = read_stereo_inference(latest_stereo_root)
    stereo = stored_stereo.observation
    if (
        stereo.source_view_id != evidence.source_view_id
        or stereo.source_sequence_index != evidence.source_sequence_index
        or stereo.rectified.source_frame_number != evidence.frame_number
        or Path(str(stored_stereo.metadata["source"]["session"])).resolve() != session_root
    ):
        raise ValueError("Stereo artifact and occupancy evidence identify different sources")
    self_mask = _verified_array(
        occupancy_root,
        frame_record["files"]["robot_mask"],
        label="occupancy robot self mask",
    )
    if self_mask.shape != stereo.depth_m.shape:
        raise ValueError("Occupancy robot mask and latest stereo image shape differ")

    robot_state = bundle.selected_robot_state
    robot_context = mapping_context.get("robot")
    if not isinstance(robot_context, Mapping):
        raise ValueError("Occupancy mapping context has no robot contract")
    offsets_array = np.asarray(robot_context.get("joint_zero_offsets_rad"), dtype=np.float64)
    if offsets_array.shape != (6,) or not np.isfinite(offsets_array).all():
        raise ValueError("Occupancy mapping context has invalid ES68 joint-zero offsets")
    offsets = tuple(float(value) for value in offsets_array)
    kinematic_resources = Es68ModelResources.packaged()
    kinematic_sha = _sha256(kinematic_resources.kinematics_yaml)
    (
        link_origins,
        mesh_vertices,
        mesh_triangles,
        robot_model_id,
        robot_geometry_reason,
        collision_manifest,
    ) = _robot_scene_geometry(evidence, offsets)

    source_assets = [
        _asset(
            "occupancy_mapping",
            "biblade_fusion.occupancy_mapping",
            occupancy_root,
            "metadata.json",
            stored_occupancy.metadata["schema_version"],
        ),
        _asset(
            "source_session",
            "biblade_fusion.acquisition_session",
            session_root,
            "manifest.json",
            session_reader.schema_version,
        ),
        _asset(
            "latest_stereo_inference",
            "biblade_fusion.stereo_inference",
            latest_stereo_root,
            "metadata.json",
            stored_stereo.metadata["schema_version"],
        ),
        _asset(
            "es68_visualization_kinematics",
            "biblade_fusion.es68_kinematics",
            kinematic_resources.kinematics_yaml.parent,
            kinematic_resources.kinematics_yaml.name,
            kinematic_sha[:16],
        ),
    ]
    if collision_manifest is not None:
        source_assets.append(
            _asset(
                "es68_d435i_collision_manifest",
                "biblade_fusion.es68_d435i_collision_manifest",
                collision_manifest.parent,
                collision_manifest.name,
                evidence.robot_model_hash[:16],
            )
        )

    events = [
        EventRecord(
            timestamp_utc=_utc(evidence.captured_at_utc, label="occupancy capture timestamp"),
            severity="INFO",
            category="occupancy",
            message=(
                f"Integrated source view {evidence.source_view_id}; persisted map "
                f"state={map_data.map_state.value}, replay state={occupancy_state}"
            ),
        )
    ]
    blocking_reasons = [
        "offline_replay_never_authorizes_motion",
        "occupancy_replay_integrity_only_unverified_for_motion",
    ]
    if occupancy_state != "READY":
        if occupancy_state == "STALE" and map_data.map_state is not OccupancyMapState.STALE:
            blocking_reasons.append("occupancy_stale_by_active_maximum_age")
        else:
            blocking_reasons.append(f"occupancy_{map_data.map_state.value}")
    if robot_geometry_reason is not None:
        blocking_reasons.append(robot_geometry_reason)
        events.append(
            EventRecord(
                timestamp_utc=created,
                severity="WARNING",
                category="robot_model",
                message=(
                    "Displayed only an UNVERIFIED ES68 kinematic skeleton because the "
                    f"active renderer identity could not be proven ({robot_geometry_reason})"
                ),
            )
        )

    plan = PlanSnapshot()
    planned_tcp_path: np.ndarray | None = None
    if source_motion_preflight is not None:
        plan, preflight_asset, preflight_event, planned_tcp_path = _historical_preflight(
            Path(source_motion_preflight), occupancy_root=occupancy_root
        )
        source_assets.append(preflight_asset)
        events.append(preflight_event)
        blocking_reasons.append("historical_preflight_requires_fresh_recheck")

    current_view = None
    current_root: Path | None = None
    if source_reconstructed_view is not None:
        current_root = Path(source_reconstructed_view).resolve()
        current_view = read_reconstructed_view(current_root)
        _validate_reconstructed_view_binding(
            current_view,
            latest_binding,
            mapping_context=mapping_context,
        )
        source_assets.append(
            _asset(
                "current_reconstructed_view",
                "biblade_fusion.reconstructed_view",
                current_root,
                "metadata.json",
                current_view.metadata["schema_version"],
            )
        )
        events.append(
            EventRecord(
                timestamp_utc=created,
                severity="INFO",
                category="reconstruction",
                message="Current view is bound to the latest occupancy acquisition chain",
            )
        )

    coarse_metadata: dict[str, Any] | None = None
    coarse_root: Path | None = None
    coarse_provenance: Literal["CURRENT_RUN_VERIFIED", "INDEPENDENT_REFERENCE"] | None = None
    coarse_provenance_reasons: tuple[str, ...] = ()
    if source_coarse_model is not None:
        stored_coarse = read_coarse_model_summary(source_coarse_model)
        coarse_root = stored_coarse.root
        coarse_metadata = stored_coarse.metadata
        coarse_provenance, coarse_provenance_reasons = _coarse_model_provenance(
            coarse_metadata,
            bindings,
            mapping_context=mapping_context,
        )
        source_assets.append(
            _asset(
                "coarse_blade_model",
                "biblade_fusion.coarse_model",
                coarse_root,
                "metadata.json",
                coarse_metadata["schema_version"],
            )
        )
        if coarse_provenance == "INDEPENDENT_REFERENCE":
            blocking_reasons.append("coarse_model_not_bound_to_occupancy_chain")
            events.append(
                EventRecord(
                    timestamp_utc=created,
                    severity="WARNING",
                    category="reconstruction",
                    message=(
                        "Loaded coarse model only as an independent reference; it is not "
                        "proven to share this occupancy acquisition chain: "
                        + "; ".join(coarse_provenance_reasons)
                    ),
                )
            )
        else:
            events.append(
                EventRecord(
                    timestamp_utc=_utc(
                        coarse_metadata["created_at_utc"], label="coarse-model timestamp"
                    ),
                    severity="INFO",
                    category="reconstruction",
                    message="Coarse model is bound to the occupancy acquisition chain",
                )
            )

    with AtomicSupervisorySnapshotWriter(output_dir) as writer:
        assets = tuple(
            writer.write_asset(
                f"{index:02d}_{source.logical_name}",
                source.source_path,
                logical_name=source.logical_name,
                kind=source.kind,
                version=source.version,
                expected_sha256=source.sha256,
            )
            for index, source in enumerate(source_assets)
        )
        occupied_reference = writer.write_array(
            "occupancy_occupied_centres_m",
            _voxel_centres(
                map_data.occupied_indices,
                origin_m=map_data.origin_m,
                voxel_size_m=map_data.voxel_size_m,
            ),
            semantic="occupied voxel centres in robot base frame, metres",
        )
        free_reference = writer.write_array(
            "occupancy_free_centres_m",
            _voxel_centres(
                map_data.free_indices,
                origin_m=map_data.origin_m,
                voxel_size_m=map_data.voxel_size_m,
            ),
            semantic="free voxel centres in robot base frame, metres",
        )
        link_reference = writer.write_array(
            "robot_link_origins_base_m",
            link_origins,
            semantic="ES68 link-frame origins in chain order, robot base frame, metres",
        )
        mesh_vertices_reference = None
        mesh_triangles_reference = None
        if mesh_vertices is not None and mesh_triangles is not None:
            mesh_vertices_reference = writer.write_array(
                "robot_collision_mesh_vertices_base_m",
                mesh_vertices,
                semantic=("verified active ES68-D435i collision-mesh vertices in base, metres"),
            )
            mesh_triangles_reference = writer.write_array(
                "robot_collision_mesh_triangles",
                mesh_triangles,
                semantic="triangle indices for verified active ES68-D435i collision mesh",
            )
        planned_path_reference = None
        if planned_tcp_path is not None:
            planned_path_reference = writer.write_array(
                "historical_planned_tcp_path_base_m",
                planned_tcp_path,
                semantic=(
                    "historical preflight leg endpoint TCP translations in base; "
                    "not an executable trajectory"
                ),
            )

        left_reference = writer.write_array(
            "left_ir_rectified",
            stereo.rectified.left_ir,
            semantic="latest FoundationStereo rectified left infrared image",
        )
        right_reference = writer.write_array(
            "right_ir_rectified",
            stereo.rectified.right_ir,
            semantic="latest FoundationStereo rectified right infrared image",
        )
        depth_reference = writer.write_array(
            "foundation_stereo_depth_m",
            stereo.depth_m,
            semantic="latest FoundationStereo depth in metres; invalid pixels are non-finite",
            allow_nonfinite=True,
        )
        mask_reference = writer.write_array(
            "robot_self_mask",
            self_mask,
            semantic="depth-consistent projected ES68 and camera self mask",
        )
        confidence_reference = None
        if stereo.result.confidence is not None:
            confidence_reference = writer.write_array(
                "foundation_stereo_confidence",
                stereo.result.confidence,
                semantic="FoundationStereo confidence supplied by the inference artifact",
            )

        current_reference = None
        if current_view is not None:
            current_reference = writer.write_array(
                "current_reconstructed_points_base_m",
                current_view.view.base_cloud.points_m,
                semantic="latest verified pose-registered blade points in base, metres",
            )

        fused_reference = None
        colors_reference = None
        normals_reference = None
        model_version = "unavailable"
        front_coverage = back_coverage = 0.0
        fin_front_coverage = fin_back_coverage = 0.0
        registered_view_count = 0
        registration_rmse_m = None
        if coarse_metadata is not None and coarse_root is not None:
            files = coarse_metadata["files"]
            fused_points = _verified_array(
                coarse_root, files["fused_points_m"], label="coarse model"
            )
            fused_normals = _verified_array(
                coarse_root, files["fused_normals"], label="coarse model"
            )
            side_labels = _verified_array(
                coarse_root, files["fused_side_labels"], label="coarse model"
            )
            if (
                fused_points.ndim != 2
                or fused_points.shape[1] != 3
                or fused_normals.shape != fused_points.shape
                or side_labels.shape != (len(fused_points),)
            ):
                raise ValueError("Coarse-model fused arrays have inconsistent shapes")
            if not np.all(np.isin(side_labels, (-1, 1))):
                raise ValueError("Coarse-model side labels must contain only -1/+1")
            colors = np.empty((len(side_labels), 3), dtype=np.uint8)
            colors[side_labels > 0] = (68, 196, 235)
            colors[side_labels < 0] = (183, 119, 237)
            fused_reference = writer.write_array(
                "coarse_fused_points_base_m",
                fused_points,
                semantic="multi-view fused blade points in robot base frame, metres",
            )
            normals_reference = writer.write_array(
                "coarse_fused_normals_base",
                fused_normals,
                semantic="unit surface normals of fused blade points in robot base frame",
            )
            colors_reference = writer.write_array(
                "coarse_fused_side_colors_rgb",
                colors,
                semantic="front/back display colors derived from authoritative side labels",
            )
            coarse_sha = _sha256(coarse_root / "metadata.json")
            model_prefix = (
                "coarse" if coarse_provenance == "CURRENT_RUN_VERIFIED" else "reference-unbound"
            )
            model_version = f"{model_prefix}:{coarse_sha[:16]}"
            patches = list(coarse_metadata.get("quality", {}).get("patches", ()))
            front_coverage = _coverage(patches, side="front", fin=False)
            back_coverage = _coverage(patches, side="back", fin=False)
            fin_front_coverage = _coverage(patches, side="front", fin=True)
            fin_back_coverage = _coverage(patches, side="back", fin=True)
            registered_view_count = len(coarse_metadata["source_views"])
            registration_rmse_m = _registration_rmse(
                list(coarse_metadata.get("fusion", {}).get("refinements", ()))
            )

        if coarse_provenance is not None:
            reconstruction_provenance = coarse_provenance
            reconstruction_reasons = coarse_provenance_reasons
        elif current_view is not None:
            reconstruction_provenance = "CURRENT_RUN_VERIFIED"
            reconstruction_reasons = ()
        else:
            reconstruction_provenance = "UNAVAILABLE"
            reconstruction_reasons = ("no_reconstruction_asset_supplied",)

        report = evidence.self_mask
        sensor = SensorSnapshot(
            source="FOUNDATION_STEREO",
            frame_number=evidence.frame_number,
            inference_latency_ms=None,
            dropped_frame_count=None,
            left_ir=left_reference,
            right_ir=right_reference,
            depth_m=depth_reference,
            confidence=confidence_reference,
            robot_self_mask=mask_reference,
            captured_at_utc=_utc(evidence.captured_at_utc, label="occupancy capture timestamp"),
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
        reconstruction = ReconstructionSnapshot(
            frame_id=map_data.frame_id,
            model_version=model_version,
            current_points_m=current_reference,
            fused_points_m=fused_reference,
            fused_colors_rgb=colors_reference,
            surface_normals=normals_reference,
            front_coverage=front_coverage,
            back_coverage=back_coverage,
            fin_front_coverage=fin_front_coverage,
            fin_back_coverage=fin_back_coverage,
            registered_view_count=registered_view_count,
            registration_rmse_m=registration_rmse_m,
            provenance_status=reconstruction_provenance,
            provenance_reasons=reconstruction_reasons,
        )
        occupancy = OccupancySnapshot(
            frame_id=map_data.frame_id,
            state=occupancy_state,
            version=map_data.version,
            content_sha256=map_data.content_hash,
            voxel_size_m=map_data.voxel_size_m,
            bounds_min_m=map_data.origin_m,
            bounds_max_m=map_data.bounds_max_m,
            age_s=age_s,
            integrated_frame_count=len(map_data.source_view_ids),
            occupied_centres_m=occupied_reference,
            free_centres_m=free_reference,
        )
        robot = RobotSceneSnapshot(
            model_id=robot_model_id,
            base_frame=map_data.frame_id,
            robot_mode=f"HISTORICAL:{robot_state.robot_mode}",
            safety_status=f"HISTORICAL:{robot_state.safety_status}",
            joint_names=ES68_JOINT_NAMES,
            joint_positions_rad=tuple(float(value) for value in evidence.joint_positions_rad),
            link_origins_base_m=link_reference,
            collision_mesh_vertices_base_m=mesh_vertices_reference,
            collision_mesh_triangles=mesh_triangles_reference,
            planned_tcp_path_base_m=planned_path_reference,
            actual_tcp_path_base_m=None,
            camera_pose=TransformSnapshot(
                parent_frame=map_data.frame_id,
                child_frame="left_rectified",
                matrix=evidence.base_t_camera_matrix,
            ),
        )
        safety = SafetySnapshot(
            system_state="BLOCKED",
            viewer_mode="REPLAY",
            viewer_motion_command_capable=False,
            unknown_occupancy_policy="BLOCK",
            stale_occupancy_policy="BLOCK",
            feedback_age_ms=None,
            blocking_reasons=tuple(dict.fromkeys(blocking_reasons)),
            calibration_ids=(
                f"hand_eye:{evidence.hand_eye_hash}",
                f"robot_model:{evidence.robot_model_hash}",
            ),
        )
        events.append(
            EventRecord(
                timestamp_utc=created,
                severity="WARNING",
                category="supervision",
                message=(
                    "Built offline replay snapshot; no live freshness, actual trajectory, "
                    "or motion capability is asserted"
                ),
            )
        )
        identity_seed = "|".join(
            (
                map_data.content_hash,
                created.isoformat(),
                *(asset.sha256 for asset in assets),
            )
        ).encode("utf-8")
        snapshot = SupervisorySnapshot(
            schema_version=SUPERVISORY_SNAPSHOT_SCHEMA_VERSION,
            snapshot_id=f"replay:{hashlib.sha256(identity_seed).hexdigest()[:20]}",
            sequence=map_data.sequence,
            created_at_utc=created,
            source_session_id=str(session_reader.manifest.get("session_id") or session_root.name),
            safety=safety,
            robot=robot,
            occupancy=occupancy,
            reconstruction=reconstruction,
            sensor=sensor,
            plan=plan,
            assets=assets,
            events=tuple(sorted(events, key=lambda event: event.timestamp_utc)),
        )
        return writer.commit(snapshot)
