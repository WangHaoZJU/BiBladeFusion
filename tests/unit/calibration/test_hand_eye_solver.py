from pathlib import Path

import numpy as np
import pytest

from biblade_fusion.calibration import (
    HandEyeSample,
    HandEyeSolveError,
    load_hand_eye_calibration,
    read_hand_eye_samples,
    solve_hand_eye,
    write_hand_eye_calibration,
    write_hand_eye_samples,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import HandEyeConfig
from biblade_fusion.devices.robot.conversions import rotation_vector_to_matrix


def pose(parent: str, child: str, rotation_vector, translation) -> PoseSE3:
    return PoseSE3.from_rotation_translation(
        parent,
        child,
        rotation_vector_to_matrix(rotation_vector),
        translation,
    )


def synthetic_samples() -> tuple[tuple[HandEyeSample, ...], PoseSE3]:
    tcp_t_left_ir = pose("tcp", "left_ir", [0.10, -0.05, 0.08], [0.04, 0.01, 0.08])
    base_t_target = pose("base", "target", [0.20, 0.10, -0.10], [0.55, -0.05, 0.20])
    motions = (
        ([0.00, 0.00, 0.00], [0.30, -0.20, 0.35]),
        ([0.30, 0.00, 0.00], [0.35, -0.15, 0.40]),
        ([0.00, -0.35, 0.00], [0.28, -0.10, 0.45]),
        ([0.00, 0.00, 0.40], [0.40, -0.25, 0.30]),
        ([0.25, -0.20, 0.10], [0.32, -0.05, 0.38]),
        ([-0.20, 0.15, 0.30], [0.45, -0.10, 0.42]),
        ([0.10, 0.30, -0.25], [0.38, -0.18, 0.33]),
        ([-0.25, -0.20, -0.15], [0.25, -0.22, 0.48]),
    )
    samples = []
    for index, (rotation_vector, translation) in enumerate(motions):
        base_t_tcp = pose("base", "tcp", rotation_vector, translation)
        left_ir_t_target = (
            tcp_t_left_ir.inverse().compose(base_t_tcp.inverse()).compose(base_t_target)
        )
        samples.append(HandEyeSample(f"sample-{index:02d}", base_t_tcp, left_ir_t_target))
    return tuple(samples), tcp_t_left_ir


def solve_config(**overrides) -> HandEyeConfig:
    values = {
        "minimum_samples": 6,
        "maximum_translation_rmse_m": 1e-5,
        "maximum_rotation_rmse_deg": 1e-3,
        "minimum_rotation_span_deg": 10.0,
        "minimum_translation_span_m": 0.01,
        "minimum_rotation_axis_diversity": 0.01,
    }
    values.update(overrides)
    return HandEyeConfig.model_validate(values)


def test_park_solver_recovers_synthetic_eye_in_hand_transform() -> None:
    samples, expected = synthetic_samples()

    solution = solve_hand_eye(samples, solve_config())

    np.testing.assert_allclose(solution.tcp_t_left_ir.matrix, expected.matrix, atol=1e-7)
    assert solution.translation_rmse_m < 1e-8
    assert solution.rotation_rmse_deg < 3e-6
    assert solution.observability.rotation_axis_diversity > 0.1


def test_solver_rejects_rotationally_degenerate_samples() -> None:
    samples = tuple(
        HandEyeSample(
            f"sample-{index}",
            PoseSE3.from_rotation_translation("base", "tcp", np.eye(3), [index * 0.1, 0, 0]),
            PoseSE3.identity("left_ir", "target"),
        )
        for index in range(6)
    )

    with pytest.raises(HandEyeSolveError, match="rotation span"):
        solve_hand_eye(samples, solve_config())


def test_sample_and_solution_artifacts_round_trip(tmp_path: Path) -> None:
    samples, _ = synthetic_samples()
    samples_path = write_hand_eye_samples(tmp_path / "samples.yaml", samples)
    loaded_samples = read_hand_eye_samples(samples_path)
    solution = solve_hand_eye(loaded_samples, solve_config())
    calibration_path = write_hand_eye_calibration(tmp_path / "hand_eye.yaml", solution)

    loaded = load_hand_eye_calibration(
        solve_config(
            calibration_path=calibration_path,
            minimum_samples=8,
        )
    )

    assert len(loaded_samples) == 8
    np.testing.assert_allclose(loaded.tcp_t_left_ir.matrix, solution.tcp_t_left_ir.matrix)
    assert loaded.sample_count == 8
