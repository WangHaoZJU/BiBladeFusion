"""Identified ChArUco target pose estimation in a calibrated left-IR image."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import CharucoTargetConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics


class CharucoDetectionError(ValueError):
    """The configured target could not be estimated with sufficient quality."""


@dataclass(frozen=True, slots=True)
class CharucoDetection:
    left_ir_t_target: PoseSE3
    charuco_ids: NDArray[np.int32]
    image_points_px: NDArray[np.float32]
    object_points_m: NDArray[np.float32]
    marker_count: int
    reprojection_rmse_px: float
    pose_ambiguity_ratio: float | None

    def __post_init__(self) -> None:
        if (self.left_ir_t_target.parent_frame, self.left_ir_t_target.child_frame) != (
            "left_ir",
            "target",
        ):
            raise ValueError("ChArUco detection requires left_ir_T_target")
        ids = np.array(self.charuco_ids, dtype=np.int32, copy=True).reshape(-1)
        image_points = np.array(self.image_points_px, dtype=np.float32, copy=True).reshape(-1, 2)
        object_points = np.array(self.object_points_m, dtype=np.float32, copy=True).reshape(-1, 3)
        if len(ids) != len(image_points) or len(ids) != len(object_points):
            raise ValueError("ChArUco IDs, image points, and object points must align")
        if len(ids) < 4 or len(set(ids.tolist())) != len(ids):
            raise ValueError("ChArUco detection needs at least four unique corner IDs")
        if self.marker_count < 1 or self.reprojection_rmse_px < 0.0:
            raise ValueError("ChArUco detection metrics are invalid")
        if self.pose_ambiguity_ratio is not None and self.pose_ambiguity_ratio < 1.0:
            raise ValueError("ChArUco pose ambiguity ratio must be at least one")
        for array in (ids, image_points, object_points):
            array.setflags(write=False)
        object.__setattr__(self, "charuco_ids", ids)
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "object_points_m", object_points)


def _camera_matrix(intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _distortion(intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    model = intrinsics.distortion_model.lower()
    if model in {"none", "distortion.none"}:
        return np.zeros(5, dtype=np.float64)
    if model not in {
        "brown_conrady",
        "distortion.brown_conrady",
        "modified_brown_conrady",
        "distortion.modified_brown_conrady",
    }:
        raise CharucoDetectionError(
            f"left-IR distortion model is not supported by solvePnP: {intrinsics.distortion_model}"
        )
    return np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)


class CharucoTargetDetector:
    """Detect an ID-stable planar target and solve its pose using IPPE."""

    def __init__(
        self,
        config: CharucoTargetConfig,
        intrinsics: CameraIntrinsics,
        cv2_module: Any | None = None,
    ) -> None:
        if config.square_length_m is None or config.marker_length_m is None:
            raise CharucoDetectionError(
                "ChArUco square_length_m and marker_length_m must match the printed board"
            )
        self._config = config
        self._intrinsics = intrinsics
        self._cv2 = cv2_module or import_module("cv2")
        try:
            dictionary_id = getattr(self._cv2.aruco, config.dictionary)
            dictionary = self._cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.board = self._cv2.aruco.CharucoBoard(
                (config.squares_x, config.squares_y),
                config.square_length_m,
                config.marker_length_m,
                dictionary,
            )
            self.board.setLegacyPattern(config.legacy_pattern)
            self._detector = self._cv2.aruco.CharucoDetector(self.board)
        except (AttributeError, TypeError, self._cv2.error) as exc:
            raise CharucoDetectionError(f"OpenCV ChArUco initialization failed: {exc}") from exc
        self._camera_matrix = _camera_matrix(intrinsics)
        self._distortion = _distortion(intrinsics)

    def detect(self, image: NDArray[np.uint8]) -> CharucoDetection:
        source = np.asarray(image)
        expected_shape = (self._intrinsics.height, self._intrinsics.width)
        if source.dtype != np.uint8 or source.shape != expected_shape:
            raise CharucoDetectionError(
                f"left-IR image must be uint8 with shape {expected_shape}, got "
                f"{source.dtype} {source.shape}"
            )
        try:
            corners, ids, marker_corners, marker_ids = self._detector.detectBoard(source)
        except self._cv2.error as exc:
            raise CharucoDetectionError(f"OpenCV ChArUco detection failed: {exc}") from exc
        corner_count = 0 if ids is None else int(len(ids))
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        if corner_count < self._config.minimum_corners:
            raise CharucoDetectionError(
                f"detected {corner_count} ChArUco corners; "
                f"at least {self._config.minimum_corners} are required"
            )
        object_points, image_points = self.board.matchImagePoints(corners, ids)
        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        if len(object_points) != corner_count:
            raise CharucoDetectionError("ChArUco board point matching returned an invalid count")

        try:
            solved, rotation_vectors, translation_vectors, *_ = self._cv2.solvePnPGeneric(
                object_points,
                image_points,
                self._camera_matrix,
                self._distortion,
                flags=self._cv2.SOLVEPNP_IPPE,
            )
        except self._cv2.error as exc:
            raise CharucoDetectionError(f"ChArUco IPPE pose solve failed: {exc}") from exc
        if not solved:
            raise CharucoDetectionError("ChArUco IPPE pose solve returned no solution")

        candidates: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
        for rotation_vector, translation_vector in zip(
            rotation_vectors, translation_vectors, strict=True
        ):
            rotation, _ = self._cv2.Rodrigues(rotation_vector)
            translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
            depths = object_points @ rotation[2, :].T + translation[2]
            if np.any(depths <= 0.0):
                continue
            projected, _ = self._cv2.projectPoints(
                object_points,
                rotation_vector,
                translation,
                self._camera_matrix,
                self._distortion,
            )
            residual = projected.reshape(-1, 2) - image_points
            rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            candidates.append((rmse, rotation, translation))
        if not candidates:
            raise CharucoDetectionError("ChArUco pose has no positive-depth IPPE solution")
        candidates.sort(key=lambda item: item[0])
        rmse, rotation, translation = candidates[0]
        if rmse > self._config.maximum_reprojection_rmse_px:
            raise CharucoDetectionError(
                f"ChArUco reprojection RMSE {rmse:.3f} px exceeds "
                f"{self._config.maximum_reprojection_rmse_px:.3f} px"
            )
        ambiguity_ratio = None
        if len(candidates) > 1:
            ambiguity_ratio = candidates[1][0] / max(rmse, 1e-9)
            if ambiguity_ratio < self._config.minimum_pose_ambiguity_ratio:
                raise CharucoDetectionError(
                    f"ChArUco planar pose is ambiguous: secondary/primary RMSE ratio "
                    f"{ambiguity_ratio:.3f} is below "
                    f"{self._config.minimum_pose_ambiguity_ratio:.3f}"
                )
        return CharucoDetection(
            PoseSE3.from_rotation_translation(
                "left_ir",
                "target",
                rotation,
                translation,
            ),
            np.asarray(ids, dtype=np.int32),
            image_points,
            object_points,
            marker_count,
            rmse,
            ambiguity_ratio,
        )
