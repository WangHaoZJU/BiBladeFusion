from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.robotics.stationarity import validate_stationary_trace
from biblade_fusion.storage.inference_stationarity import (
    read_inference_stationarity,
    write_inference_stationarity,
)


def _state(time_s: float) -> RobotState:
    return RobotState(
        monotonic_time_ns=round(time_s * 1e9),
        controller_time_s=time_s,
        joint_positions_rad=np.zeros(6),
        base_t_tcp=PoseSE3.identity("base", "tcp"),
        robot_mode="IDLE",
        safety_status="NORMAL",
        speed_scaling=0.1,
    )


def _write_asset(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
    reference = _state(1.0)
    trace = (_state(1.1), _state(1.2))
    evidence = validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        maximum_robot_state_staleness_s=0.25,
    )
    stored = write_inference_stationarity(
        tmp_path / "inference_stationarity.json",
        view_id="front-001",
        sequence_index=3,
        reference=reference,
        trace=trace,
        evidence=evidence,
        source_session_manifest=manifest,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        maximum_robot_state_staleness_s=0.25,
    )
    return stored, manifest


def test_round_trip_recomputes_trace_and_source_binding(tmp_path: Path) -> None:
    stored, manifest = _write_asset(tmp_path)

    reread = read_inference_stationarity(stored.path)

    assert reread.view_id == "front-001"
    assert reread.sequence_index == 3
    assert reread.source_session_manifest_path == manifest.resolve()
    assert reread.thresholds == (0.001, 0.001, 0.001, 0.25)
    assert reread.file_sha256 == stored.file_sha256
    assert reread.content_sha256 == stored.content_sha256
    assert reread.evidence.sample_count == 3


def test_round_trip_preserves_powered_stopped_runtime_state(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
    reference = RobotState(
        monotonic_time_ns=1_000_000_000,
        controller_time_s=1.0,
        joint_positions_rad=np.zeros(6),
        base_t_tcp=PoseSE3.identity("base", "tcp"),
        robot_mode="RUNNING",
        safety_status="NORMAL",
        speed_scaling=1.0,
        runtime_state="STOPPED",
    )
    trace = tuple(
        RobotState(
            monotonic_time_ns=round(time_s * 1e9),
            controller_time_s=time_s,
            joint_positions_rad=np.zeros(6),
            base_t_tcp=PoseSE3.identity("base", "tcp"),
            robot_mode="RUNNING",
            safety_status="NORMAL",
            speed_scaling=1.0,
            runtime_state="STOPPED",
        )
        for time_s in (1.1, 1.2)
    )
    evidence = validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
    )

    stored = write_inference_stationarity(
        tmp_path / "powered_stationarity.json",
        view_id="front-powered",
        sequence_index=0,
        reference=reference,
        trace=trace,
        evidence=evidence,
        source_session_manifest=manifest,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
        maximum_robot_state_staleness_s=0.25,
    )

    reread = read_inference_stationarity(stored.path)

    assert reread.reference.robot_mode == "RUNNING"
    assert reread.reference.runtime_state == "STOPPED"
    assert all(state.runtime_state == "STOPPED" for state in reread.trace)


def test_reader_keeps_legacy_schema_one_assets_compatible(tmp_path: Path) -> None:
    stored, _ = _write_asset(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload["reference_robot_state"].pop("runtime_state")
    for state in payload["inference_robot_state_trace"]:
        state.pop("runtime_state")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    canonical = json.dumps(
        body,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    stored.path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reread = read_inference_stationarity(stored.path)

    assert reread.reference.robot_mode == "IDLE"
    assert reread.reference.runtime_state is None


def test_content_tamper_is_rejected(tmp_path: Path) -> None:
    stored, _ = _write_asset(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["view_id"] = "forged-view"
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        read_inference_stationarity(stored.path)


def test_source_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    stored, manifest = _write_asset(tmp_path)
    manifest.write_text('{"status":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source session manifest changed"):
        read_inference_stationarity(stored.path)


def test_asset_is_write_once(tmp_path: Path) -> None:
    stored, manifest = _write_asset(tmp_path)
    reference = _state(2.0)
    trace = (_state(2.1), _state(2.2))
    evidence = validate_stationary_trace(
        reference,
        trace,
        max_joint_delta_rad=0.001,
        max_tcp_translation_delta_m=0.001,
        max_tcp_rotation_delta_rad=0.001,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_inference_stationarity(
            stored.path,
            view_id="front-002",
            sequence_index=4,
            reference=reference,
            trace=trace,
            evidence=evidence,
            source_session_manifest=manifest,
            max_joint_delta_rad=0.001,
            max_tcp_translation_delta_m=0.001,
            max_tcp_rotation_delta_rad=0.001,
            maximum_robot_state_staleness_s=0.25,
        )
