from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AcquisitionConfig, OccupancyConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.mapping import (
    DepthIntegrationConfig,
    DepthRayIntegrator,
    OccupancyGridSpec,
    OccupancyMapState,
)
from biblade_fusion.mapping.self_mask import RobotSelfMaskReport
from biblade_fusion.robotics import (
    OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH,
    Es68KinematicModel,
    Es68ModelResources,
    load_es68_flange_t_tcp,
)
from biblade_fusion.storage.occupancy_mapping import (
    OCCUPANCY_MAPPING_SCHEMA_VERSION,
    LegacyReplayOccupancyMapping,
    OccupancyMappingValidationDependencies,
    ReplayOccupancyMapping,
    VerifiedHandEyeSource,
    _read_occupancy_mapping_with_dependencies,
    _write_occupancy_mapping_with_dependencies,
    read_legacy_occupancy_mapping_for_replay,
    read_occupancy_mapping,
    read_occupancy_mapping_for_replay,
    write_occupancy_mapping,
)
from biblade_fusion.workflows.occupancy_mapping import (
    MAPPING_CONTEXT_SCHEMA_VERSION,
    OccupancyFrameEvidence,
    OccupancyFrameUpdate,
    OccupancyMappingContext,
    occupancy_array_content_hash,
    occupancy_physical_source_id,
)

_EMPTY_JSON_SHA256 = hashlib.sha256(b"{}\n").hexdigest()
_HAND_EYE_BYTES = b"schema_version: 2\n"
_HAND_EYE_SHA256 = hashlib.sha256(_HAND_EYE_BYTES).hexdigest()


def _source(root: Path, filename: str) -> Path:
    root.mkdir()
    (root / filename).write_text("{}\n", encoding="utf-8")
    return root


def _add_session_view_source(root: Path, view_id: str) -> None:
    view = root / "views" / view_id
    view.mkdir(parents=True)
    (view / "metadata.json").write_text("{}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> OccupancyConfig:
    return OccupancyConfig(
        enabled=True,
        voxel_size_m=0.05,
        workspace_bounds_min_m=(0.0, 0.0, 0.0),
        workspace_bounds_max_m=(0.1, 0.1, 1.0),
        minimum_depth_m=0.1,
        maximum_depth_m=2.0,
        integration_stride=1,
        free_space_margin_m=0.0,
    )


def _context(
    config: OccupancyConfig,
    acquisition: AcquisitionConfig,
    flange_t_left_ir: PoseSE3,
) -> OccupancyMappingContext:
    resources = Es68ModelResources.packaged()
    flange_t_tcp = load_es68_flange_t_tcp(resources)
    return OccupancyMappingContext.from_payload(
        {
            "schema_version": MAPPING_CONTEXT_SCHEMA_VERSION,
            "grid": {
                "frame_id": "base",
                "voxel_size_m": 0.05,
                "origin_m": [0.0, 0.0, 0.0],
                "grid_shape": [2, 2, 20],
            },
            "occupancy_contract": config.model_dump(mode="json"),
            "acquisition_contract": acquisition.model_dump(mode="json"),
            "robot": {
                "model_content_hash": "a" * 64,
                "self_mask_excluded_link_names": [],
                "self_mask_render_backend": "test_fake:v1",
                "joint_zero_offsets_rad": [0.0] * 6,
                "flange_T_tcp": flange_t_tcp.matrix.tolist(),
                "flange_tcp_asset_sha256": _sha256(resources.tcp_offset_json),
            },
            "hand_eye": {
                "artifact_sha256": _HAND_EYE_SHA256,
                "flange_T_left_ir": flange_t_left_ir.matrix.tolist(),
            },
            "foundation_stereo": {
                "backend": "foundation_stereo",
                "left_right_consistency_applied": True,
                "left_right_consistency_threshold_px": 1.0,
                "confidence_semantic": (
                    "exp_negative_left_right_disparity_error_not_calibrated_probability"
                ),
            },
            "rectified_stereo": {
                "left": {
                    "width": 1,
                    "height": 1,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "distortion_model": "none",
                    "distortion_coefficients": [],
                },
                "right": {
                    "width": 1,
                    "height": 1,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "distortion_model": "none",
                    "distortion_coefficients": [],
                },
                "right_rectified_T_left_rectified": PoseSE3.from_rotation_translation(
                    "right_rectified",
                    "left_rectified",
                    np.eye(3),
                    [0.1, 0.0, 0.0],
                ).matrix.tolist(),
                "left_rectified_T_left_ir": np.eye(4).tolist(),
                "right_rectified_T_right_ir": np.eye(4).tolist(),
                "disparity_to_depth_q": np.eye(4).tolist(),
                "left_valid_roi": [0, 0, 1, 1],
                "right_valid_roi": [0, 0, 1, 1],
            },
        }
    )


def _updates() -> tuple[
    tuple[OccupancyFrameUpdate, ...],
    OccupancyConfig,
    AcquisitionConfig,
]:
    config = _config()
    acquisition = AcquisitionConfig()
    intrinsics = CameraIntrinsics(1, 1, 1.0, 1.0, 0.0, 0.0, "none", ())
    initial_pose = PoseSE3.from_rotation_translation(
        "base",
        "left_rectified",
        np.eye(3),
        [0.025, 0.025, 0.025],
    )
    kinematics = Es68KinematicModel.from_resources()
    initial_joints = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    initial_base_t_flange = kinematics.base_t_flange(initial_joints)
    flange_t_tcp = load_es68_flange_t_tcp()
    base_t_left_ir = PoseSE3("base", "left_ir", initial_pose.matrix)
    flange_t_left_ir = initial_base_t_flange.inverse().compose(base_t_left_ir)
    context = _context(config, acquisition, flange_t_left_ir)
    integrator = DepthRayIntegrator(
        OccupancyGridSpec(0.05, (0.0, 0.0, 0.0), (2, 2, 20)),
        DepthIntegrationConfig(0.1, 2.0, 1, 1, 0.0),
        mapping_context_hash=context.content_hash,
    )
    depth = np.array([[0.75]], dtype=np.float64)
    valid = np.array([[True]], dtype=np.bool_)
    confidence = np.array([[1.0]], dtype=np.float64)
    predicted = np.array([[np.inf]], dtype=np.float64)
    robot_mask = np.array([[False]], dtype=np.bool_)
    integration_mask = np.array([[True]], dtype=np.bool_)
    report = RobotSelfMaskReport(0, 1, 0, 0, 1, 0.01, 0.02, 1)
    snapshot = None
    result: list[OccupancyFrameUpdate] = []
    observed = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
    for index in range(3):
        source_id = f"view-{index}"
        physical_source_id = occupancy_physical_source_id(
            source_session_manifest_sha256=_EMPTY_JSON_SHA256,
            source_session_view_metadata_sha256=_EMPTY_JSON_SHA256,
            source_sequence_index=index,
            frame_number=index,
            source_view_id=source_id,
        )
        joints_array = np.zeros(6, dtype=np.float64)
        joints_array[2] = 0.1 * index
        joints = tuple(float(value) for value in joints_array)
        base_t_flange = kinematics.base_t_flange(joints)
        predicted_base_t_tcp = base_t_flange.compose(flange_t_tcp)
        relative_tcp_rotation = (
            predicted_base_t_tcp.rotation.T @ predicted_base_t_tcp.rotation
        )
        zero_pose_rotation_error_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(relative_tcp_rotation) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        pose = PoseSE3(
            "base",
            "left_rectified",
            base_t_flange.compose(flange_t_left_ir).matrix,
        )
        mapping = integrator.integrate(
            snapshot,
            depth,
            intrinsics,
            pose,
            valid_mask=integration_mask,
            source_view_id=physical_source_id,
            observed_at_utc=observed + timedelta(milliseconds=index),
        )
        evidence = OccupancyFrameEvidence(
            source_view_id=source_id,
            physical_source_id=physical_source_id,
            source_sequence_index=index,
            frame_number=index,
            captured_at_utc=mapping.created_at_utc.isoformat(),
            robot_model_hash="a" * 64,
            hand_eye_hash=_HAND_EYE_SHA256,
            source_stereo_metadata_sha256=_EMPTY_JSON_SHA256,
            source_session_manifest_sha256=_EMPTY_JSON_SHA256,
            source_session_view_metadata_sha256=_EMPTY_JSON_SHA256,
            base_t_flange_matrix=tuple(
                tuple(float(value) for value in row) for row in base_t_flange.matrix
            ),
            predicted_base_t_tcp_matrix=tuple(
                tuple(float(value) for value in row) for row in predicted_base_t_tcp.matrix
            ),
            observed_base_t_tcp_matrix=tuple(
                tuple(float(value) for value in row) for row in predicted_base_t_tcp.matrix
            ),
            base_t_camera_matrix=tuple(tuple(float(value) for value in row) for row in pose.matrix),
            joint_positions_rad=joints,
            fk_tcp_translation_error_m=0.0,
            fk_tcp_rotation_error_deg=zero_pose_rotation_error_deg,
            valid_depth_fraction=1.0,
            stereo_valid_fraction=1.0,
            confidence_accepted_fraction=1.0,
            mean_accepted_confidence=1.0,
            lr_consistency_threshold_px=1.0,
            self_mask=report,
            mapping_context_hash=context.content_hash,
            previous_evidence_hash=mapping.parent_evidence_hash,
            mapping_snapshot_content_hash=mapping.content_hash,
            mapping_snapshot_sequence=mapping.sequence,
            mapping_source_view_ids=mapping.source_view_ids,
            source_depth_content_hash=occupancy_array_content_hash(depth),
            stereo_valid_mask_content_hash=occupancy_array_content_hash(valid),
            stereo_confidence_content_hash=occupancy_array_content_hash(confidence),
            predicted_robot_depth_content_hash=occupancy_array_content_hash(predicted),
            robot_mask_content_hash=occupancy_array_content_hash(robot_mask),
            integration_valid_mask_content_hash=occupancy_array_content_hash(integration_mask),
        )
        snapshot = (
            mapping.promote_to_ready(evidence.quality_evidence_hash)
            if index == 2
            else mapping.bind_mapping_evidence(evidence.quality_evidence_hash)
        )
        result.append(
            OccupancyFrameUpdate(
                snapshot,
                mapping,
                context,
                depth,
                valid,
                confidence,
                predicted,
                robot_mask,
                integration_mask,
                evidence,
            )
        )
    return tuple(result), config, acquisition


class _FakeRenderer:
    def __init__(
        self,
        *,
        model_content_hash: str,
        joint_zero_offsets_rad: tuple[float, ...],
        predicted_depth_m: np.ndarray,
    ) -> None:
        self.model_content_hash = model_content_hash
        self.self_mask_excluded_link_names: tuple[str, ...] = ()
        self.self_mask_render_backend = "test_fake:v1"
        self.joint_zero_offsets_rad = joint_zero_offsets_rad
        self._predicted_depth_m = predicted_depth_m

    def render_robot_depth(self, intrinsics, joint_positions_rad, base_t_camera):
        del intrinsics, joint_positions_rad, base_t_camera
        return self._predicted_depth_m.copy()


class _FakeStereoResult:
    def __init__(self, update: OccupancyFrameUpdate, depth_m: np.ndarray) -> None:
        self.valid_mask = update.stereo_valid_mask.copy()
        self.confidence = update.stereo_confidence.astype(np.float32)
        self.metadata = dict(update.mapping_context.to_payload()["foundation_stereo"])
        self._depth_m = depth_m

    def depth_m(self, calibration):
        del calibration
        return self._depth_m.copy()


class _FakeSessionReader:
    def __init__(
        self,
        root: Path,
        update: OccupancyFrameUpdate,
        *,
        joint_delta: float = 0.0,
        capture_joint_delta: float = 0.0,
    ):
        self.path = root.resolve()
        self.manifest = {"created_at_utc": update.evidence.captured_at_utc}
        joints = np.asarray(update.evidence.joint_positions_rad, dtype=np.float64).copy()
        joints[0] += joint_delta
        after_joints = joints.copy()
        after_joints[0] += capture_joint_delta
        monotonic_time_ns = 100 + update.evidence.frame_number
        selected_state = SimpleNamespace(
            monotonic_time_ns=monotonic_time_ns,
            joint_positions_rad=joints,
            base_t_tcp=PoseSE3(
                "base",
                "tcp",
                update.evidence.observed_base_t_tcp_matrix,
            ),
        )
        after_state = SimpleNamespace(
            monotonic_time_ns=monotonic_time_ns,
            joint_positions_rad=after_joints,
            base_t_tcp=selected_state.base_t_tcp,
        )
        relative_tcp_rotation = (
            selected_state.base_t_tcp.rotation.T
            @ after_state.base_t_tcp.rotation
        )
        tcp_rotation_delta_rad = float(
            np.arccos(
                np.clip(
                    (np.trace(relative_tcp_rotation) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        self._bundle = SimpleNamespace(
            view_id=update.evidence.source_view_id,
            sequence_index=update.evidence.source_sequence_index,
            stereo=SimpleNamespace(
                frame_number=update.evidence.frame_number,
                monotonic_time_ns=monotonic_time_ns,
                left_device_time_ms=1.0,
                right_device_time_ms=1.0,
            ),
            robot_state_before=selected_state,
            robot_state_after=after_state,
            selected_robot_state=selected_state,
            metrics=SimpleNamespace(
                bracket_ms=0.0,
                max_joint_delta_rad=abs(capture_joint_delta),
                tcp_translation_delta_m=0.0,
                tcp_rotation_delta_rad=tcp_rotation_delta_rad,
                selected_robot_state_offset_ms=0.0,
            ),
        )

    def load_bundle(self, selector: int | str):
        if selector not in {self._bundle.view_id, self._bundle.sequence_index}:
            raise KeyError(selector)
        return self._bundle

    def descriptor(self, selector: int | str):
        if selector not in {self._bundle.view_id, self._bundle.sequence_index}:
            raise KeyError(selector)
        return SimpleNamespace(relative_path=f"views/{self._bundle.view_id}")


def _fake_stereo_source(
    update: OccupancyFrameUpdate,
    session_root: Path,
    *,
    depth_delta_m: float = 0.0,
):
    payload = update.mapping_context.to_payload()
    rectified = payload["rectified_stereo"]
    left = CameraIntrinsics(**{
        **rectified["left"],
        "distortion_coefficients": tuple(rectified["left"]["distortion_coefficients"]),
    })
    right = CameraIntrinsics(**{
        **rectified["right"],
        "distortion_coefficients": tuple(rectified["right"]["distortion_coefficients"]),
    })
    calibration = SimpleNamespace(
        left=left,
        right=right,
        right_rectified_t_left_rectified=PoseSE3(
            "right_rectified",
            "left_rectified",
            rectified["right_rectified_T_left_rectified"],
        ),
        left_rectified_t_left_ir=PoseSE3(
            "left_rectified",
            "left_ir",
            rectified["left_rectified_T_left_ir"],
        ),
        right_rectified_t_right_ir=PoseSE3(
            "right_rectified",
            "right_ir",
            rectified["right_rectified_T_right_ir"],
        ),
        disparity_to_depth_q=np.asarray(rectified["disparity_to_depth_q"], dtype=np.float64),
        left_valid_roi=tuple(rectified["left_valid_roi"]),
        right_valid_roi=tuple(rectified["right_valid_roi"]),
    )
    depth = update.source_depth_m.astype(np.float32) + np.float32(depth_delta_m)
    observation = SimpleNamespace(
        source_view_id=update.evidence.source_view_id,
        source_sequence_index=update.evidence.source_sequence_index,
        rectified=SimpleNamespace(
            source_frame_number=update.evidence.frame_number,
            source_monotonic_time_ns=100 + update.evidence.frame_number,
            calibration=calibration,
        ),
        depth_m=depth,
        result=_FakeStereoResult(update, depth),
    )
    return SimpleNamespace(
        observation=observation,
        metadata={
            "source": {
                "session": str(session_root.resolve()),
                "view_id": update.evidence.source_view_id,
                "sequence_index": update.evidence.source_sequence_index,
                "frame_number": update.evidence.frame_number,
                "monotonic_time_ns": 100 + update.evidence.frame_number,
            }
        },
    )


def _validation_dependencies(
    updates: tuple[OccupancyFrameUpdate, ...],
    stereo_sources: tuple[Path, ...],
    session_sources: tuple[Path, ...],
    *,
    hand_eye_pose: PoseSE3 | None = None,
    renderer_hash: str = "a" * 64,
    rendered_depth_m: np.ndarray | None = None,
    session_joint_delta: float = 0.0,
    capture_joint_delta: float = 0.0,
) -> OccupancyMappingValidationDependencies:
    stereo_by_root = {
        root.resolve(): _fake_stereo_source(update, session_root)
        for update, root, session_root in zip(
            updates,
            stereo_sources,
            session_sources,
            strict=True,
        )
    }
    session_by_root = {
        root.resolve(): _FakeSessionReader(
            root,
            update,
            joint_delta=session_joint_delta,
            capture_joint_delta=capture_joint_delta,
        )
        for update, root in zip(updates, session_sources, strict=True)
    }
    context_hand_eye = PoseSE3(
        "flange",
        "left_ir",
        updates[0].mapping_context.to_payload()["hand_eye"]["flange_T_left_ir"],
    )
    predicted = (
        updates[0].predicted_robot_depth_m
        if rendered_depth_m is None
        else np.asarray(rendered_depth_m, dtype=np.float64)
    )
    return OccupancyMappingValidationDependencies(
        stereo_reader=lambda path: stereo_by_root[Path(path).resolve()],
        stereo_source_verifier=lambda stored, session: None,
        session_reader_factory=lambda path: session_by_root[Path(path).resolve()],
        hand_eye_reader=lambda path: VerifiedHandEyeSource(
            artifact_sha256=_HAND_EYE_SHA256,
            flange_t_left_ir=hand_eye_pose or context_hand_eye,
        ),
        renderer_factory=lambda offsets: _FakeRenderer(
            model_content_hash=renderer_hash,
            joint_zero_offsets_rad=tuple(offsets),
            predicted_depth_m=predicted,
        ),
    )


def _write_asset(
    tmp_path: Path,
) -> tuple[Path, tuple[Path, ...], OccupancyMappingValidationDependencies]:
    updates, config, acquisition = _updates()
    stereo_sources = tuple(
        _source(tmp_path / f"stereo-{index}", "metadata.json") for index in range(3)
    )
    session_sources = tuple(
        _source(tmp_path / f"session-{index}", "manifest.json") for index in range(3)
    )
    for update, session_root in zip(updates, session_sources, strict=True):
        _add_session_view_source(session_root, update.evidence.source_view_id)
    hand_eye_source = tmp_path / "hand-eye.yaml"
    hand_eye_source.write_bytes(_HAND_EYE_BYTES)
    dependencies = _validation_dependencies(updates, stereo_sources, session_sources)
    destination = _write_occupancy_mapping_with_dependencies(
        tmp_path / "occupancy",
        updates,
        config,
        acquisition,
        source_stereo_inferences=stereo_sources,
        source_sessions=session_sources,
        source_hand_eye=hand_eye_source,
        validation_dependencies=dependencies,
    )
    return destination, stereo_sources, dependencies


def _source_inputs(tmp_path: Path):
    updates, config, acquisition = _updates()
    stereo_sources = tuple(
        _source(tmp_path / f"attack-stereo-{index}", "metadata.json")
        for index in range(3)
    )
    session_sources = tuple(
        _source(tmp_path / f"attack-session-{index}", "manifest.json")
        for index in range(3)
    )
    for update, session_root in zip(updates, session_sources, strict=True):
        _add_session_view_source(session_root, update.evidence.source_view_id)
    hand_eye_source = tmp_path / "attack-hand-eye.yaml"
    hand_eye_source.write_bytes(_HAND_EYE_BYTES)
    return (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    )


def _rebind_first_update(
    update: OccupancyFrameUpdate,
    evidence: OccupancyFrameEvidence,
) -> OccupancyFrameUpdate:
    snapshot = update.mapping_snapshot.bind_mapping_evidence(evidence.quality_evidence_hash)
    return replace(update, snapshot=snapshot, evidence=evidence)


def _write_single_update(
    tmp_path: Path,
    update: OccupancyFrameUpdate,
    config: OccupancyConfig,
    acquisition: AcquisitionConfig,
) -> Path:
    stereo = _source(tmp_path / "single-stereo", "metadata.json")
    session = _source(tmp_path / "single-session", "manifest.json")
    _add_session_view_source(session, update.evidence.source_view_id)
    hand_eye_source = tmp_path / "single-hand-eye.yaml"
    hand_eye_source.write_bytes(_HAND_EYE_BYTES)
    dependencies = _validation_dependencies((update,), (stereo,), (session,))
    return _write_occupancy_mapping_with_dependencies(
        tmp_path / "single-occupancy",
        (update,),
        config,
        acquisition,
        source_stereo_inferences=(stereo,),
        source_sessions=(session,),
        source_hand_eye=hand_eye_source,
        validation_dependencies=dependencies,
    )


def _rewrite_context_metadata(
    destination: Path,
    mutate,
) -> None:
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = metadata["mapping_context"]["payload"]
    mutate(payload)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    metadata["mapping_context"]["content_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _snapshot_content_hash(raw: dict[str, object]) -> str:
    payload = dict(raw)
    payload.pop("content_hash", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_snapshot_payload(path: Path, raw: dict[str, object]) -> str:
    raw["content_hash"] = _snapshot_content_hash(raw)
    payload = {
        "format_version": 4,
        "artifact_kind": "biblade_fusion.safety_occupancy",
        "units": "m",
        "snapshot": raw,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return str(raw["content_hash"])


def _downgrade_fixture_to_genuine_schema6(destination: Path) -> None:
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = 6
    logical_ids: list[str] = []
    previous_evidence_hash: str | None = None
    final_result_payload: dict[str, object] | None = None
    for frame in metadata["frames"]:
        evidence = frame["evidence"]
        logical_ids.append(str(evidence["source_view_id"]))
        evidence.pop("physical_source_id")
        evidence["mapping_source_view_ids"] = list(logical_ids)
        evidence["previous_evidence_hash"] = previous_evidence_hash

        mapping_record = frame["mapping_snapshot"]
        mapping_path = destination / mapping_record["path"]
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))["snapshot"]
        mapping_payload["source_view_ids"] = list(logical_ids)
        mapping_payload["parent_evidence_hash"] = previous_evidence_hash
        mapping_payload["quality_evidence_hash"] = None
        mapping_payload["state_reason"] = (
            f"integrated {logical_ids[-1]}; awaiting self-mask and depth quality gates"
        )
        mapping_hash = _write_snapshot_payload(mapping_path, mapping_payload)
        mapping_record["sha256"] = _sha256(mapping_path)
        mapping_record["content_hash"] = mapping_hash
        evidence["mapping_snapshot_content_hash"] = mapping_hash

        evidence_without_hash = dict(evidence)
        evidence_without_hash.pop("quality_evidence_hash")
        evidence_hash = hashlib.sha256(
            json.dumps(
                evidence_without_hash,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        evidence["quality_evidence_hash"] = evidence_hash

        result_record = frame["result_snapshot"]
        result_path = destination / result_record["path"]
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))["snapshot"]
        result_payload["source_view_ids"] = list(logical_ids)
        result_payload["parent_evidence_hash"] = previous_evidence_hash
        result_payload["quality_evidence_hash"] = evidence_hash
        result_payload["state_reason"] = mapping_payload["state_reason"]
        result_hash = _write_snapshot_payload(result_path, result_payload)
        result_record["sha256"] = _sha256(result_path)
        result_record["content_hash"] = result_hash
        previous_evidence_hash = evidence_hash
        final_result_payload = result_payload

    assert final_result_payload is not None
    final_record = metadata["snapshot"]
    final_path = destination / final_record["path"]
    final_hash = _write_snapshot_payload(final_path, final_result_payload)
    final_record["sha256"] = _sha256(final_path)
    final_record["content_hash"] = final_hash
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_occupancy_mapping_asset_round_trip_verifies_full_chain(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    stored = _read_occupancy_mapping_with_dependencies(
        destination,
        validation_dependencies=dependencies,
    )

    assert stored.snapshot.map_state is OccupancyMapState.MAP_READY
    assert stored.snapshot.source_view_ids == tuple(
        evidence.physical_source_id for evidence in stored.frame_evidence
    )
    assert len(stored.mapping_snapshots) == 3
    assert len(stored.result_snapshots) == 3
    assert stored.frame_evidence[1].previous_evidence_hash == (
        stored.frame_evidence[0].quality_evidence_hash
    )
    assert stored.mapping_context.content_hash == stored.snapshot.mapping_context_hash
    assert stored.metadata["motion_authorized"] is False
    assert stored.metadata["schema_version"] == OCCUPANCY_MAPPING_SCHEMA_VERSION == 7
    assert stored.motion_eligible is True
    assert stored.verification_status == "full_semantic_verified_for_motion_preflight"
    assert stored.semantic_attestation.snapshot_content_hash == stored.snapshot.content_hash
    assert (
        stored.semantic_attestation.mapping_context_hash
        == stored.snapshot.mapping_context_hash
    )
    assert (
        stored.semantic_attestation.quality_evidence_hash
        == stored.snapshot.quality_evidence_hash
    )
    assert stored.semantic_attestation.robot_geometry_hash == "a" * 64
    assert (
        stored.semantic_attestation.semantic_verifier_contract_hash
        == OCCUPANCY_SEMANTIC_VERIFIER_CONTRACT_HASH
    )
    assert stored.semantic_attestation.occupancy_metadata_sha256 == _sha256(
        destination / "metadata.json"
    )
    np.testing.assert_allclose(
        stored.frame_evidence[0].base_t_flange_matrix,
        Es68KinematicModel.from_resources().base_t_flange((0.0,) * 6).matrix,
    )
    np.testing.assert_allclose(
        stored.frame_evidence[0].predicted_base_t_tcp_matrix,
        stored.frame_evidence[0].observed_base_t_tcp_matrix,
    )


def test_replay_reader_is_explicitly_unverified_and_never_motion_eligible(
    tmp_path: Path,
) -> None:
    destination, _, _ = _write_asset(tmp_path)

    replay = read_occupancy_mapping_for_replay(destination)

    assert isinstance(replay, ReplayOccupancyMapping)
    assert replay.verification_status == "integrity_only_unverified_for_motion"
    assert replay.motion_eligible is False
    assert isinstance(replay.metadata, dict)
    assert not hasattr(replay, "semantic_attestation")


def test_schema7_rejects_tampered_physical_source_identity(tmp_path: Path) -> None:
    destination, _, _ = _write_asset(tmp_path)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["evidence"]["physical_source_id"] = "f" * 64
    metadata["frames"][0]["evidence"]["mapping_source_view_ids"][0] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="physical_source_id does not reproduce"):
        read_occupancy_mapping_for_replay(destination)


def test_schema6_requires_explicit_legacy_reader_and_is_never_motion_eligible(
    tmp_path: Path,
) -> None:
    destination, _, _ = _write_asset(tmp_path)
    _downgrade_fixture_to_genuine_schema6(destination)

    with pytest.raises(ValueError, match="unsupported schema 6"):
        read_occupancy_mapping_for_replay(destination)
    with pytest.raises(ValueError, match="unsupported schema 6"):
        read_occupancy_mapping(destination)
    legacy = read_legacy_occupancy_mapping_for_replay(destination)

    assert isinstance(legacy, LegacyReplayOccupancyMapping)
    assert legacy.legacy_schema_version == 6
    assert legacy.motion_eligible is False
    assert legacy.verification_status == "legacy_integrity_only_unverified_for_motion"


def test_full_reader_rejects_metadata_change_during_semantic_verification(
    tmp_path: Path,
) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    metadata_path = destination / "metadata.json"
    original_verifier = dependencies.stereo_source_verifier
    changed = {"done": False}

    def mutate_metadata(stored, session) -> None:
        original_verifier(stored, session)
        if not changed["done"]:
            metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
            changed["done"] = True

    racing_dependencies = replace(
        dependencies,
        stereo_source_verifier=mutate_metadata,
    )

    with pytest.raises(ValueError, match="changed during full semantic verification"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=racing_dependencies,
        )


def test_public_full_validation_api_has_no_dependency_injection_escape_hatch() -> None:
    assert "validation_dependencies" not in inspect.signature(
        read_occupancy_mapping
    ).parameters
    assert "validation_dependencies" not in inspect.signature(
        write_occupancy_mapping
    ).parameters


def test_writer_rejects_semantically_swapped_stereo_source(tmp_path: Path) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    dependencies = _validation_dependencies(updates, stereo_sources, session_sources)

    with pytest.raises(ValueError, match="stereo source identity"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "swapped-source",
            updates,
            config,
            acquisition,
            source_stereo_inferences=(
                stereo_sources[1],
                stereo_sources[0],
                stereo_sources[2],
            ),
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_claimed_hand_eye_hash_with_different_primary_pose(
    tmp_path: Path,
) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    context_pose = PoseSE3(
        "flange",
        "left_ir",
        updates[0].mapping_context.to_payload()["hand_eye"]["flange_T_left_ir"],
    )
    wrong_matrix = context_pose.matrix.copy()
    wrong_matrix[0, 3] += 0.01
    dependencies = _validation_dependencies(
        updates,
        stereo_sources,
        session_sources,
        hand_eye_pose=PoseSE3("flange", "left_ir", wrong_matrix),
    )

    with pytest.raises(ValueError, match="hand-eye.*flange_T_left_ir"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "wrong-hand-eye",
            updates,
            config,
            acquisition,
            source_stereo_inferences=stereo_sources,
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_claimed_robot_hash_not_recomputed_from_active_model(
    tmp_path: Path,
) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    dependencies = _validation_dependencies(
        updates,
        stereo_sources,
        session_sources,
        renderer_hash="c" * 64,
    )

    with pytest.raises(ValueError, match="active robot geometry hash"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "wrong-robot-hash",
            updates,
            config,
            acquisition,
            source_stereo_inferences=stereo_sources,
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_robot_depth_that_does_not_rerender(tmp_path: Path) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    dependencies = _validation_dependencies(
        updates,
        stereo_sources,
        session_sources,
        rendered_depth_m=np.array([[0.5]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="robot depth does not reproduce"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "wrong-robot-depth",
            updates,
            config,
            acquisition,
            source_stereo_inferences=stereo_sources,
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_session_joints_not_bound_to_frame_evidence(tmp_path: Path) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    dependencies = _validation_dependencies(
        updates,
        stereo_sources,
        session_sources,
        session_joint_delta=0.01,
    )

    with pytest.raises(ValueError, match="session joints"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "wrong-session-joints",
            updates,
            config,
            acquisition,
            source_stereo_inferences=stereo_sources,
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_source_captured_while_robot_was_moving(tmp_path: Path) -> None:
    (
        updates,
        config,
        acquisition,
        stereo_sources,
        session_sources,
        hand_eye_source,
    ) = _source_inputs(tmp_path)
    dependencies = _validation_dependencies(
        updates,
        stereo_sources,
        session_sources,
        capture_joint_delta=0.02,
    )

    with pytest.raises(ValueError, match="settled robot pose.*joint motion"):
        _write_occupancy_mapping_with_dependencies(
            tmp_path / "moving-source",
            updates,
            config,
            acquisition,
            source_stereo_inferences=stereo_sources,
            source_sessions=session_sources,
            source_hand_eye=hand_eye_source,
            validation_dependencies=dependencies,
        )


def test_occupancy_mapping_asset_rejects_tampered_source(tmp_path: Path) -> None:
    destination, stereo_sources, dependencies = _write_asset(tmp_path)
    (stereo_sources[0] / "metadata.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source checksum mismatch"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_evidence_hash_is_recomputed_when_metadata_is_tampered(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["evidence"]["valid_depth_fraction"] = 0.5
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="quality_evidence_hash mismatch"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_array_hash_catches_tampering_even_if_file_checksum_is_rewritten(
    tmp_path: Path,
) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record = metadata["frames"][0]["files"]["integration_valid_mask"]
    array_path = destination / record["path"]
    np.save(array_path, np.array([[False]], dtype=np.bool_), allow_pickle=False)
    record["sha256"] = _sha256(array_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="self-mask arrays|array content hashes"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_mapping_context_rejects_configuration_tampering(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["configuration"]["occupancy"]["minimum_depth_m"] = 0.2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration mismatch"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_mapping_context_requires_complete_robot_contract(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    _rewrite_context_metadata(
        destination,
        lambda payload: payload["robot"].pop("joint_zero_offsets_rad"),
    )

    with pytest.raises(ValueError, match="robot fields are invalid"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_mapping_context_rejects_non_packaged_flange_tcp_hash(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)
    _rewrite_context_metadata(
        destination,
        lambda payload: payload["robot"].__setitem__(
            "flange_tcp_asset_sha256",
            "c" * 64,
        ),
    )

    with pytest.raises(ValueError, match="flange/TCP asset hash"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_mapping_context_rejects_invalid_hand_eye_transform(tmp_path: Path) -> None:
    destination, _, dependencies = _write_asset(tmp_path)

    def corrupt_rotation(payload) -> None:
        payload["hand_eye"]["flange_T_left_ir"][0][0] = 2.0

    _rewrite_context_metadata(destination, corrupt_rotation)

    with pytest.raises(ValueError, match=r"flange_T_left_ir.*SE\(3\)"):
        _read_occupancy_mapping_with_dependencies(
            destination,
            validation_dependencies=dependencies,
        )


def test_writer_rejects_evidence_robot_hash_not_bound_to_context(tmp_path: Path) -> None:
    updates, config, acquisition = _updates()
    evidence = replace(
        updates[0].evidence,
        robot_model_hash="c" * 64,
        quality_evidence_hash="",
    )
    update = _rebind_first_update(updates[0], evidence)

    with pytest.raises(ValueError, match="robot model hash"):
        _write_single_update(tmp_path, update, config, acquisition)


def test_writer_recomputes_fk_instead_of_trusting_recorded_matrix(tmp_path: Path) -> None:
    updates, config, acquisition = _updates()
    matrix = np.asarray(updates[0].evidence.base_t_flange_matrix).copy()
    matrix[0, 3] += 0.0001
    evidence = replace(
        updates[0].evidence,
        base_t_flange_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        quality_evidence_hash="",
    )
    update = _rebind_first_update(updates[0], evidence)

    with pytest.raises(ValueError, match="base_T_flange FK does not reproduce"):
        _write_single_update(tmp_path, update, config, acquisition)


def test_writer_recomputes_predicted_tcp_instead_of_trusting_pose_evidence(
    tmp_path: Path,
) -> None:
    updates, config, acquisition = _updates()
    matrix = np.asarray(updates[0].evidence.predicted_base_t_tcp_matrix).copy()
    matrix[0, 3] += 0.001
    evidence = replace(
        updates[0].evidence,
        predicted_base_t_tcp_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        observed_base_t_tcp_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        quality_evidence_hash="",
    )
    update = _rebind_first_update(updates[0], evidence)

    with pytest.raises(ValueError, match="predicted base_T_tcp does not reproduce"):
        _write_single_update(tmp_path, update, config, acquisition)


def test_update_rejects_tampered_camera_pose_evidence(tmp_path: Path) -> None:
    del tmp_path
    updates, config, acquisition = _updates()
    del config, acquisition
    matrix = np.asarray(updates[0].evidence.base_t_camera_matrix).copy()
    matrix[2, 3] += 0.0001
    evidence = replace(
        updates[0].evidence,
        base_t_camera_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        quality_evidence_hash="",
    )
    with pytest.raises(ValueError, match="camera-view evidence does not match"):
        _rebind_first_update(updates[0], evidence)
