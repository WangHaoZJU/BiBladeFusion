"""D435i infrared stereo calibration from ChArUco observations.

No factory camera intrinsics or extrinsics enter this solver.  Monocular Zhang
calibration supplies the initial values and ``stereoCalibrate`` performs the joint
non-linear least-squares refinement (bundle adjustment) over both cameras and every
accepted board observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.devices.depth_camera import CameraIntrinsics, StereoCalibrationSnapshot


class StereoCharucoCalibrationError(ValueError):
    """The observations cannot produce a trustworthy stereo calibration."""


class DistortionModel(StrEnum):
    """Supported OpenCV pinhole distortion parameterizations."""

    RADIAL2 = "radial2"
    BROWN5 = "brown5"
    RATIONAL8 = "rational8"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return {
            self.RADIAL2: ("k1", "k2"),
            self.BROWN5: ("k1", "k2", "p1", "p2", "k3"),
            self.RATIONAL8: ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
        }[self]

    @property
    def calibration_flags(self) -> int:
        if self is self.RADIAL2:
            return (
                cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K3
                | cv2.CALIB_FIX_K4
                | cv2.CALIB_FIX_K5
                | cv2.CALIB_FIX_K6
            )
        if self is self.BROWN5:
            return cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        return cv2.CALIB_RATIONAL_MODEL


@dataclass(frozen=True, slots=True)
class StereoCharucoBoard:
    name: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary_name: str
    legacy_pattern: bool
    minimum_corners_per_camera: int
    detector_params: dict[str, int | float | bool]

    @classmethod
    def read(cls, path: str | Path) -> StereoCharucoBoard:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "charuco_board":
            raise StereoCharucoCalibrationError("target configuration is not a ChArUco board")
        board = cls(
            name=str(payload["name"]),
            squares_x=int(payload["squares_x"]),
            squares_y=int(payload["squares_y"]),
            square_length_m=float(payload["square_length_m"]),
            marker_length_m=float(payload["marker_length_m"]),
            dictionary_name=str(payload["dictionary_name"]),
            legacy_pattern=bool(payload.get("legacy_pattern", False)),
            minimum_corners_per_camera=int(payload.get("minimum_corners_per_camera", 20)),
            detector_params=dict(payload.get("detector_params", {})),
        )
        if board.squares_x < 3 or board.squares_y < 3:
            raise StereoCharucoCalibrationError("ChArUco board must contain at least 3x3 squares")
        if not 0 < board.marker_length_m < board.square_length_m:
            raise StereoCharucoCalibrationError("marker length must be below square length")
        return board

    def create_opencv(self) -> tuple[Any, Any]:
        try:
            dictionary_id = getattr(cv2.aruco, self.dictionary_name)
        except AttributeError as exc:
            raise StereoCharucoCalibrationError(
                f"OpenCV has no ArUco dictionary {self.dictionary_name}"
            ) from exc
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )
        board.setLegacyPattern(self.legacy_pattern)
        parameters = cv2.aruco.DetectorParameters()
        for name, value in self.detector_params.items():
            if not hasattr(parameters, name):
                raise StereoCharucoCalibrationError(f"unknown ArUco detector parameter: {name}")
            setattr(parameters, name, value)
        return board, cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            parameters,
        )


@dataclass(frozen=True, slots=True)
class CharucoImageDetection:
    ids: NDArray[np.int32]
    image_points_px: NDArray[np.float32]
    object_points_m: NDArray[np.float32]
    marker_count: int

    @property
    def corner_count(self) -> int:
        return len(self.ids)


@dataclass(frozen=True, slots=True)
class StereoCharucoSample:
    sample_id: str
    left: CharucoImageDetection
    right: CharucoImageDetection


@dataclass(frozen=True, slots=True)
class StereoCalibrationMetrics:
    sample_count: int
    left_monocular_rms_px: float
    right_monocular_rms_px: float
    joint_stereo_rms_px: float
    epipolar_rmse_px: float
    epipolar_p95_px: float


@dataclass(frozen=True, slots=True)
class DistortionModelComparison:
    model: DistortionModel
    calibration_sample_count: int
    validation_sample_count: int
    validation_reprojection_rmse_px: float
    validation_epipolar_rmse_px: float


@dataclass(frozen=True, slots=True)
class SolvedStereoCalibration:
    calibration: StereoCalibrationSnapshot
    metrics: StereoCalibrationMetrics
    image_size: tuple[int, int]
    board_name: str
    distortion_model: DistortionModel
    model_comparison: tuple[DistortionModelComparison, ...] = ()


class StereoCharucoDetector:
    def __init__(self, target: StereoCharucoBoard) -> None:
        self.target = target
        self.board, self.detector = target.create_opencv()

    def detect(self, image: NDArray[np.uint8]) -> CharucoImageDetection | None:
        source = np.asarray(image)
        if source.dtype != np.uint8 or source.ndim != 2:
            raise StereoCharucoCalibrationError("calibration images must be uint8 grayscale")
        corners, ids, _marker_corners, marker_ids = self.detector.detectBoard(source)
        if ids is None or len(ids) < 4:
            return None
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        return CharucoImageDetection(
            np.asarray(ids, dtype=np.int32).reshape(-1),
            np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
            0 if marker_ids is None else len(marker_ids),
        )

    def detect_pair(
        self,
        sample_id: str,
        left: NDArray[np.uint8],
        right: NDArray[np.uint8],
    ) -> StereoCharucoSample | None:
        left_detection = self.detect(left)
        right_detection = self.detect(right)
        if left_detection is None or right_detection is None:
            return None
        if min(left_detection.corner_count, right_detection.corner_count) < (
            self.target.minimum_corners_per_camera
        ):
            return None
        return StereoCharucoSample(sample_id, left_detection, right_detection)


def _common_points(
    sample: StereoCharucoSample,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    left_by_id = {
        int(identifier): point
        for identifier, point in zip(sample.left.ids, sample.left.image_points_px, strict=True)
    }
    right_by_id = {
        int(identifier): point
        for identifier, point in zip(sample.right.ids, sample.right.image_points_px, strict=True)
    }
    object_by_id = {
        int(identifier): point
        for identifier, point in zip(sample.left.ids, sample.left.object_points_m, strict=True)
    }
    common = sorted(set(left_by_id) & set(right_by_id))
    if len(common) < 6:
        raise StereoCharucoCalibrationError(
            f"sample {sample.sample_id} has fewer than six common stereo corners"
        )
    return (
        np.asarray([object_by_id[item] for item in common], dtype=np.float32),
        np.asarray([left_by_id[item] for item in common], dtype=np.float32),
        np.asarray([right_by_id[item] for item in common], dtype=np.float32),
    )


def _camera_intrinsics(
    matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    size: tuple[int, int],
    model: DistortionModel,
) -> CameraIntrinsics:
    coefficient_count = 8 if model is DistortionModel.RATIONAL8 else 5
    return CameraIntrinsics(
        size[0],
        size[1],
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        "brown_conrady",
        tuple(float(value) for value in distortion.reshape(-1)[:coefficient_count]),
    )


def solve_stereo_charuco(
    samples: list[StereoCharucoSample],
    image_size: tuple[int, int],
    target: StereoCharucoBoard,
    *,
    minimum_samples: int = 15,
    distortion_model: DistortionModel = DistortionModel.BROWN5,
) -> SolvedStereoCalibration:
    """Run independent Zhang initialization followed by joint stereo bundle adjustment."""

    if len(samples) < minimum_samples:
        raise StereoCharucoCalibrationError(
            f"need at least {minimum_samples} accepted views, received {len(samples)}"
        )
    left_objects = [item.left.object_points_m for item in samples]
    left_images = [item.left.image_points_px for item in samples]
    right_objects = [item.right.object_points_m for item in samples]
    right_images = [item.right.image_points_px for item in samples]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-10)
    mono_flags = distortion_model.calibration_flags
    left_rms, left_k, left_d, *_ = cv2.calibrateCamera(
        left_objects, left_images, image_size, None, None, flags=mono_flags, criteria=criteria
    )
    right_rms, right_k, right_d, *_ = cv2.calibrateCamera(
        right_objects, right_images, image_size, None, None, flags=mono_flags, criteria=criteria
    )

    common = [_common_points(sample) for sample in samples]
    stereo_objects = [item[0] for item in common]
    stereo_left = [item[1] for item in common]
    stereo_right = [item[2] for item in common]
    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS | distortion_model.calibration_flags
    stereo_rms, left_k, left_d, right_k, right_d, rotation, translation, essential, _ = (
        cv2.stereoCalibrate(
            stereo_objects,
            stereo_left,
            stereo_right,
            left_k,
            left_d,
            right_k,
            right_d,
            image_size,
            flags=stereo_flags,
            criteria=criteria,
        )
    )
    if not np.isfinite((left_rms, right_rms, stereo_rms, *translation.reshape(-1))).all():
        raise StereoCharucoCalibrationError("calibration produced non-finite parameters")
    baseline = float(np.linalg.norm(translation))
    if not 0.02 <= baseline <= 0.15:
        raise StereoCharucoCalibrationError(
            f"estimated baseline {baseline:.6f} m is outside a plausible D435i range"
        )

    distances: list[float] = []
    fundamental = np.linalg.inv(right_k).T @ essential @ np.linalg.inv(left_k)
    for _, left_points, right_points in common:
        left_h = np.column_stack((left_points, np.ones(len(left_points))))
        right_h = np.column_stack((right_points, np.ones(len(right_points))))
        lines_right = (fundamental @ left_h.T).T
        numerator = np.abs(np.sum(right_h * lines_right, axis=1))
        denominator = np.linalg.norm(lines_right[:, :2], axis=1)
        distances.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
    epipolar = np.asarray(distances)
    metrics = StereoCalibrationMetrics(
        len(samples),
        float(left_rms),
        float(right_rms),
        float(stereo_rms),
        float(np.sqrt(np.mean(epipolar**2))),
        float(np.percentile(epipolar, 95)),
    )
    calibration = StereoCalibrationSnapshot(
        _camera_intrinsics(left_k, left_d, image_size, distortion_model),
        _camera_intrinsics(right_k, right_d, image_size, distortion_model),
        PoseSE3.from_rotation_translation("right_ir", "left_ir", rotation, translation.reshape(3)),
        None,
    )
    return SolvedStereoCalibration(calibration, metrics, image_size, target.name, distortion_model)


def _skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _validation_metrics(
    result: SolvedStereoCalibration,
    samples: list[StereoCharucoSample],
) -> tuple[float, float]:
    calibration = result.calibration

    def matrix(value: CameraIntrinsics) -> NDArray[np.float64]:
        return np.array([[value.fx, 0.0, value.cx], [0.0, value.fy, value.cy], [0.0, 0.0, 1.0]])

    left_k = matrix(calibration.left)
    right_k = matrix(calibration.right)
    left_d = np.asarray(calibration.left.distortion_coefficients)
    right_d = np.asarray(calibration.right.distortion_coefficients)
    residuals: list[NDArray[np.float64]] = []
    epipolar_distances: list[float] = []
    essential = _skew(calibration.right_t_left.translation_m) @ calibration.right_t_left.rotation
    fundamental = np.linalg.inv(right_k).T @ essential @ np.linalg.inv(left_k)
    for sample in samples:
        for detection, camera_matrix, distortion in (
            (sample.left, left_k, left_d),
            (sample.right, right_k, right_d),
        ):
            solved, rotation, translation = cv2.solvePnP(
                detection.object_points_m,
                detection.image_points_px,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not solved:
                raise StereoCharucoCalibrationError(
                    f"validation pose failed for sample {sample.sample_id}"
                )
            projected, _ = cv2.projectPoints(
                detection.object_points_m,
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            residuals.append(projected.reshape(-1, 2) - detection.image_points_px)
        _, left_points, right_points = _common_points(sample)
        left_h = np.column_stack((left_points, np.ones(len(left_points))))
        right_h = np.column_stack((right_points, np.ones(len(right_points))))
        lines_right = (fundamental @ left_h.T).T
        numerator = np.abs(np.sum(right_h * lines_right, axis=1))
        denominator = np.linalg.norm(lines_right[:, :2], axis=1)
        epipolar_distances.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
    reprojection = np.concatenate(residuals, axis=0)
    epipolar = np.asarray(epipolar_distances)
    return (
        float(np.sqrt(np.mean(np.sum(reprojection**2, axis=1)))),
        float(np.sqrt(np.mean(epipolar**2))),
    )


def compare_and_solve_stereo_charuco(
    samples: list[StereoCharucoSample],
    image_size: tuple[int, int],
    target: StereoCharucoBoard,
    *,
    minimum_samples: int = 20,
) -> SolvedStereoCalibration:
    """Compare all models on held-out views, then refit the selected model on all data."""

    if len(samples) < minimum_samples:
        raise StereoCharucoCalibrationError(
            f"automatic comparison needs at least {minimum_samples} views"
        )
    validation_indices = set(range(4, len(samples), 5))
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    training = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    if len(validation) < 3 or len(training) < 12:
        raise StereoCharucoCalibrationError(
            "automatic comparison needs at least three validation and twelve calibration views"
        )
    comparisons: list[DistortionModelComparison] = []
    for model in DistortionModel:
        candidate = solve_stereo_charuco(
            training,
            image_size,
            target,
            minimum_samples=12,
            distortion_model=model,
        )
        reprojection, epipolar = _validation_metrics(candidate, validation)
        comparisons.append(
            DistortionModelComparison(
                model,
                len(training),
                len(validation),
                reprojection,
                epipolar,
            )
        )
    best_error = min(item.validation_reprojection_rmse_px for item in comparisons)
    eligible = [
        item
        for item in comparisons
        if item.validation_reprojection_rmse_px <= best_error * 1.02 + 0.005
    ]
    selected = min(eligible, key=lambda item: len(item.model.parameter_names)).model
    final = solve_stereo_charuco(
        samples,
        image_size,
        target,
        minimum_samples=minimum_samples,
        distortion_model=selected,
    )
    return SolvedStereoCalibration(
        final.calibration,
        final.metrics,
        final.image_size,
        final.board_name,
        final.distortion_model,
        tuple(comparisons),
    )


def write_stereo_calibration(
    path: str | Path,
    result: SolvedStereoCalibration,
    sample_ids: list[str],
) -> Path:
    """Write an auditable YAML artifact for later rectification and reconstruction."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    calibration = result.calibration

    def intrinsics_payload(value: CameraIntrinsics) -> dict[str, object]:
        coefficients = list(value.distortion_coefficients)
        named = {
            name: coefficients[index]
            for index, name in enumerate(result.distortion_model.parameter_names)
        }
        return {
            "width": value.width,
            "height": value.height,
            "camera_matrix": [
                [value.fx, 0.0, value.cx],
                [0.0, value.fy, value.cy],
                [0.0, 0.0, 1.0],
            ],
            "distortion_model": result.distortion_model.value,
            "opencv_distortion_model": value.distortion_model,
            "active_distortion_parameters": named,
            "distortion_coefficients": coefficients,
        }

    payload = {
        "schema_version": 1,
        "calibration_type": "d435i_ir_stereo_charuco",
        "created_utc": datetime.now(UTC).isoformat(),
        "factory_intrinsics_used": False,
        "initialization": "independent_zhang",
        "optimization": "joint_stereo_bundle_adjustment",
        "selected_distortion_model": result.distortion_model.value,
        "target_name": result.board_name,
        "accepted_sample_ids": sample_ids,
        "left_ir": intrinsics_payload(calibration.left),
        "right_ir": intrinsics_payload(calibration.right),
        "right_ir_T_left_ir": calibration.right_t_left.matrix.tolist(),
        "baseline_m": calibration.baseline_m,
        "metrics": {
            "sample_count": result.metrics.sample_count,
            "left_monocular_rms_px": result.metrics.left_monocular_rms_px,
            "right_monocular_rms_px": result.metrics.right_monocular_rms_px,
            "joint_stereo_rms_px": result.metrics.joint_stereo_rms_px,
            "epipolar_rmse_px": result.metrics.epipolar_rmse_px,
            "epipolar_p95_px": result.metrics.epipolar_p95_px,
        },
        "model_comparison": [
            {
                "model": item.model.value,
                "calibration_sample_count": item.calibration_sample_count,
                "validation_sample_count": item.validation_sample_count,
                "validation_reprojection_rmse_px": item.validation_reprojection_rmse_px,
                "validation_epipolar_rmse_px": item.validation_epipolar_rmse_px,
            }
            for item in result.model_comparison
        ],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_stereo_calibration(path: str | Path) -> StereoCalibrationSnapshot:
    """Load the user-calibrated IR stereo parameters; factory values are not consulted."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload.get("calibration_type") != "d435i_ir_stereo_charuco":
        raise StereoCharucoCalibrationError("unsupported stereo calibration artifact")
    if payload.get("factory_intrinsics_used") is not False:
        raise StereoCharucoCalibrationError("stereo artifact does not certify user calibration")

    def read_intrinsics(key: str) -> CameraIntrinsics:
        item = payload[key]
        matrix = np.asarray(item["camera_matrix"], dtype=np.float64)
        return CameraIntrinsics(
            int(item["width"]),
            int(item["height"]),
            float(matrix[0, 0]),
            float(matrix[1, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
            str(item.get("opencv_distortion_model", "brown_conrady")),
            tuple(float(value) for value in item["distortion_coefficients"]),
        )

    transform = np.asarray(payload["right_ir_T_left_ir"], dtype=np.float64)
    return StereoCalibrationSnapshot(
        read_intrinsics("left_ir"),
        read_intrinsics("right_ir"),
        PoseSE3.from_rotation_translation(
            "right_ir", "left_ir", transform[:3, :3], transform[:3, 3]
        ),
        None,
    )
