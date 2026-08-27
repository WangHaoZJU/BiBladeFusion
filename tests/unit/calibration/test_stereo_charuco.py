from pathlib import Path

import cv2
import numpy as np
import pytest

from biblade_fusion.calibration import (
    CharucoImageDetection,
    DistortionModel,
    StereoCharucoBoard,
    StereoCharucoSample,
    compare_and_solve_stereo_charuco,
    load_stereo_calibration,
    publish_runtime_stereo_calibration,
    solve_stereo_charuco,
    write_stereo_calibration,
)

TARGET = Path("configs/charuco_dict5x5_14x9_20mm_15mm.yaml")


def _synthetic_samples(target: StereoCharucoBoard) -> list[StereoCharucoSample]:
    left_k = np.array([[910.0, 0, 638.0], [0, 905.0, 361.0], [0, 0, 1.0]])
    right_k = np.array([[914.0, 0, 642.0], [0, 908.0, 358.0], [0, 0, 1.0]])
    distortion = np.zeros(5)
    right_t_left_rvec = np.array([0.001, -0.002, 0.0005])
    right_t_left_r, _ = cv2.Rodrigues(right_t_left_rvec)
    right_t_left_t = np.array([-0.0502, 0.0002, -0.0001])
    object_points = np.array(
        [[x * 0.02, y * 0.02, 0.0] for y in range(8) for x in range(13)],
        dtype=np.float32,
    )
    ids = np.arange(len(object_points), dtype=np.int32)
    samples = []
    for index in range(20):
        rvec = np.array(
            [0.08 * np.sin(index), 0.15 * np.cos(index * 0.7), 0.04 * np.sin(index * 0.4)]
        )
        left_r, _ = cv2.Rodrigues(rvec)
        left_t = np.array(
            [-0.13 + 0.012 * (index % 5), -0.08 + 0.015 * (index % 4), 0.55 + 0.025 * (index % 6)]
        )
        right_r = right_t_left_r @ left_r
        right_t = right_t_left_r @ left_t + right_t_left_t
        right_rvec, _ = cv2.Rodrigues(right_r)
        left_points, _ = cv2.projectPoints(object_points, rvec, left_t, left_k, distortion)
        right_points, _ = cv2.projectPoints(object_points, right_rvec, right_t, right_k, distortion)
        left_detection = CharucoImageDetection(ids, left_points.reshape(-1, 2), object_points, 50)
        right_detection = CharucoImageDetection(ids, right_points.reshape(-1, 2), object_points, 50)
        samples.append(StereoCharucoSample(f"sample-{index}", left_detection, right_detection))
    return samples


def test_board_configuration_is_preserved() -> None:
    target = StereoCharucoBoard.read(TARGET)
    assert (target.squares_x, target.squares_y) == (14, 9)
    assert target.square_length_m == 0.020
    assert target.marker_length_m == 0.015
    assert target.dictionary_name == "DICT_5X5_100"


def test_joint_solver_and_artifact_round_trip(tmp_path: Path) -> None:
    target = StereoCharucoBoard.read(TARGET)
    samples = _synthetic_samples(target)
    solved = solve_stereo_charuco(samples, (1280, 720), target)
    assert solved.metrics.sample_count == 20
    assert solved.calibration.baseline_m == pytest.approx(0.0502, abs=2e-4)
    assert solved.metrics.epipolar_rmse_px < 0.01
    artifact = write_stereo_calibration(
        tmp_path / "stereo.yaml", solved, [item.sample_id for item in samples]
    )
    loaded = load_stereo_calibration(artifact)
    np.testing.assert_allclose(loaded.right_t_left.matrix, solved.calibration.right_t_left.matrix)
    assert loaded.left.fx == pytest.approx(solved.calibration.left.fx)
    runtime = publish_runtime_stereo_calibration(artifact, tmp_path / "runtime/active.yaml")
    assert runtime.read_bytes() == artifact.read_bytes()
    assert load_stereo_calibration(runtime).left.fx == pytest.approx(solved.calibration.left.fx)


@pytest.mark.parametrize(
    ("model", "coefficient_count"),
    [
        (DistortionModel.RADIAL2, 5),
        (DistortionModel.BROWN5, 5),
        (DistortionModel.RATIONAL8, 8),
    ],
)
def test_selectable_distortion_models(model: DistortionModel, coefficient_count: int) -> None:
    target = StereoCharucoBoard.read(TARGET)
    solved = solve_stereo_charuco(
        _synthetic_samples(target),
        (1280, 720),
        target,
        distortion_model=model,
    )
    assert solved.distortion_model is model
    assert len(solved.calibration.left.distortion_coefficients) == coefficient_count
    assert solved.calibration.baseline_m == pytest.approx(0.0502, abs=2e-4)


def test_auto_model_comparison_uses_held_out_views() -> None:
    target = StereoCharucoBoard.read(TARGET)
    solved = compare_and_solve_stereo_charuco(_synthetic_samples(target), (1280, 720), target)
    assert solved.distortion_model is DistortionModel.RADIAL2
    assert {item.model for item in solved.model_comparison} == set(DistortionModel)
    assert all(item.validation_sample_count == 4 for item in solved.model_comparison)
