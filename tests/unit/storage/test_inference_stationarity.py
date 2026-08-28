from __future__ import annotations

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
