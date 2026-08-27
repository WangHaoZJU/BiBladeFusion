"""Independent, append-only validation of a fixed D435i IR stereo calibration.

This module deliberately has no calibration solver.  It consumes a previously
generated user calibration, evaluates newly acquired ChArUco observations, and
stores enough raw and derived evidence to reproduce every reported metric.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from biblade_fusion.calibration.stereo_assets import RawInfraredStereoFrame
from biblade_fusion.calibration.stereo_charuco import (
    CharucoImageDetection,
    StereoCharucoBoard,
    StereoCharucoDetector,
    StereoCharucoSample,
    load_stereo_calibration,
)
from biblade_fusion.core.settings import StereoRectificationConfig
from biblade_fusion.devices.depth_camera import CameraIntrinsics, StereoFrame
from biblade_fusion.perception.stereo import StereoRectifier


class StereoValidationError(RuntimeError):
    """A validation session or one of its immutable assets is invalid."""


@dataclass(frozen=True, slots=True)
class StereoValidationThresholds:
    """Explicit pass/fail gates recorded with every validation report."""

    minimum_accepted_pairs: int = 8
    maximum_vertical_disparity_rmse_px: float = 0.5
    maximum_vertical_disparity_p95_px: float = 1.0
    maximum_monocular_reprojection_rmse_px: float = 0.5
    maximum_stereo_transfer_rmse_px: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_accepted_pairs < 3:
            raise ValueError("stereo validation requires at least three accepted pairs")
        limits = (
            self.maximum_vertical_disparity_rmse_px,
            self.maximum_vertical_disparity_p95_px,
            self.maximum_monocular_reprojection_rmse_px,
            self.maximum_stereo_transfer_rmse_px,
        )
        if not np.isfinite(limits).all() or any(value <= 0.0 for value in limits):
            raise ValueError("stereo validation limits must be finite and positive")


@dataclass(frozen=True, slots=True)
class StereoValidationMetrics:
    accepted_pair_count: int
    rejected_pair_count: int
    common_corner_count: int
    vertical_disparity_signed_mean_px: float
    vertical_disparity_rmse_px: float
    vertical_disparity_p95_px: float
    vertical_disparity_max_px: float
    left_reprojection_rmse_px: float
    right_reprojection_rmse_px: float
    stereo_transfer_rmse_px: float
    passed: bool


@dataclass(frozen=True, slots=True)
class StereoValidationResult:
    session_root: Path
    analysis_root: Path
    report_json: Path
    report_text: Path
    metrics: StereoValidationMetrics
    accepted_pair_ids: tuple[str, ...]
    rejected_pair_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PairEvaluation:
    vertical_signed_px: NDArray[np.float64]
    left_reprojection_px: NDArray[np.float64]
    right_reprojection_px: NDArray[np.float64]
    stereo_transfer_px: NDArray[np.float64]
    rectified_left_points: NDArray[np.float64]
    rectified_right_points: NDArray[np.float64]
    common_ids: NDArray[np.int32]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_asset(root: Path, supplied: object) -> Path:
    relative = Path(str(supplied))
    if relative.is_absolute():
        raise StereoValidationError("validation manifest paths must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise StereoValidationError("validation manifest path escapes the session root")
    return resolved


def _write_image(path: Path, image: NDArray[np.uint8]) -> None:
    if not cv2.imwrite(str(path), image):
        raise StereoValidationError(f"failed to write validation image: {path}")


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
    return np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)


def _common_observations(
    sample: StereoCharucoSample,
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    left = {
        int(identifier): (point, object_point)
        for identifier, point, object_point in zip(
            sample.left.ids,
            sample.left.image_points_px,
            sample.left.object_points_m,
            strict=True,
        )
    }
    right = {
        int(identifier): point
        for identifier, point in zip(
            sample.right.ids, sample.right.image_points_px, strict=True
        )
    }
    common = np.asarray(sorted(set(left) & set(right)), dtype=np.int32)
    if len(common) < 6:
        raise StereoValidationError(
            f"validation pair {sample.sample_id} has fewer than six common corners"
        )
    return (
        common,
        np.asarray([left[int(item)][1] for item in common], dtype=np.float32),
        np.asarray([left[int(item)][0] for item in common], dtype=np.float32),
        np.asarray([right[int(item)] for item in common], dtype=np.float32),
    )


def _solve_reprojection(
    detection: CharucoImageDetection,
    intrinsics: CameraIntrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrix = _camera_matrix(intrinsics)
    distortion = _distortion(intrinsics)
    solved, rotation, translation = cv2.solvePnP(
        detection.object_points_m,
        detection.image_points_px,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise StereoValidationError("could not solve independent ChArUco validation pose")
    projected, _ = cv2.projectPoints(
        detection.object_points_m,
        rotation,
        translation,
        matrix,
        distortion,
    )
    residual = projected.reshape(-1, 2) - detection.image_points_px
    return rotation.reshape(3), translation.reshape(3), np.linalg.norm(residual, axis=1)


def _rectify_points(
    points: NDArray[np.float32],
    source: CameraIntrinsics,
    rotation: NDArray[np.float64],
    rectified: CameraIntrinsics,
) -> NDArray[np.float64]:
    return cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        _camera_matrix(source),
        _distortion(source),
        R=rotation,
        P=_camera_matrix(rectified),
    ).reshape(-1, 2)


def _evaluate_pair(
    sample: StereoCharucoSample,
    calibration: Any,
    rectifier: StereoRectifier,
) -> _PairEvaluation:
    common_ids, object_points, left_points, right_points = _common_observations(sample)
    left_rvec, left_tvec, left_residual = _solve_reprojection(
        sample.left, calibration.left
    )
    _, _, right_residual = _solve_reprojection(sample.right, calibration.right)

    left_rotation, _ = cv2.Rodrigues(left_rvec)
    right_rotation = calibration.right_t_left.rotation @ left_rotation
    right_translation = (
        calibration.right_t_left.rotation @ left_tvec
        + calibration.right_t_left.translation_m
    )
    right_rvec, _ = cv2.Rodrigues(right_rotation)
    transferred, _ = cv2.projectPoints(
        object_points,
        right_rvec,
        right_translation,
        _camera_matrix(calibration.right),
        _distortion(calibration.right),
    )
    transfer_residual = np.linalg.norm(
        transferred.reshape(-1, 2) - right_points, axis=1
    )

    rectified = rectifier.calibration
    left_rectified = _rectify_points(
        left_points,
        calibration.left,
        rectified.left_rectified_t_left_ir.rotation,
        rectified.left,
    )
    right_rectified = _rectify_points(
        right_points,
        calibration.right,
        rectified.right_rectified_t_right_ir.rotation,
        rectified.right,
    )
    return _PairEvaluation(
        right_rectified[:, 1] - left_rectified[:, 1],
        np.asarray(left_residual, dtype=np.float64),
        np.asarray(right_residual, dtype=np.float64),
        np.asarray(transfer_residual, dtype=np.float64),
        left_rectified,
        right_rectified,
        common_ids,
    )


def _rmse(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _annotated_detection(
    image: NDArray[np.uint8], detection: CharucoImageDetection | None
) -> NDArray[np.uint8]:
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if detection is None:
        cv2.putText(
            output,
            "REJECTED: NO CHARUCO BOARD",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 30, 240),
            2,
        )
        return output
    for point in detection.image_points_px:
        cv2.circle(output, tuple(np.rint(point).astype(int)), 3, (30, 220, 30), -1)
    cv2.putText(
        output,
        f"corners={detection.corner_count}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 220, 30),
        2,
    )
    return output


def _corner_color(identifier: int) -> tuple[int, int, int]:
    return (
        40 + (identifier * 67) % 200,
        40 + (identifier * 97) % 200,
        40 + (identifier * 137) % 200,
    )


def _rectified_visualizations(
    left: NDArray[np.uint8],
    right: NDArray[np.uint8],
    evaluation: _PairEvaluation,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    left_color = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_color = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    for identifier, left_point, right_point in zip(
        evaluation.common_ids,
        evaluation.rectified_left_points,
        evaluation.rectified_right_points,
        strict=True,
    ):
        color = _corner_color(int(identifier))
        cv2.circle(left_color, tuple(np.rint(left_point).astype(int)), 4, color, -1)
        cv2.circle(right_color, tuple(np.rint(right_point).astype(int)), 4, color, -1)
    overlay = np.hstack((left_color, right_color))
    height, width = left.shape
    spacing = max(30, height // 12)
    for y in range(spacing // 2, height, spacing):
        cv2.line(overlay, (0, y), (2 * width - 1, y), (255, 210, 30), 1)
    cv2.line(overlay, (width, 0), (width, height - 1), (255, 255, 255), 2)
    cv2.putText(
        overlay,
        f"rectified vertical RMSE={_rmse(evaluation.vertical_signed_px):.3f}px",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 255, 40),
        2,
    )
    return left_color, right_color, overlay


class StereoValidationAssetSession:
    """Append raw hold-out pairs and bind one independent validation report."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest_path = root / "validation_manifest.json"
        self._manifest = manifest
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        collection_root: str | Path,
        *,
        target_path: str | Path,
        calibration_path: str | Path,
        image_size: tuple[int, int],
        frames_per_second: int,
        serial_number: str | None,
        emitter_enabled: bool,
        rectification: StereoRectificationConfig,
        thresholds: StereoValidationThresholds,
    ) -> StereoValidationAssetSession:
        source_target = Path(target_path).resolve()
        source_calibration = Path(calibration_path).resolve()
        if not source_target.is_file():
            raise FileNotFoundError(source_target)
        if not source_calibration.is_file():
            raise FileNotFoundError(source_calibration)
        calibration = load_stereo_calibration(calibration_path)
        left_size = (calibration.left.width, calibration.left.height)
        right_size = (calibration.right.width, calibration.right.height)
        if left_size != image_size or right_size != image_size:
            raise StereoValidationError(
                f"configured stream {image_size} does not match calibration "
                f"left={left_size}, right={right_size}"
            )
        collection = Path(collection_root)
        collection.mkdir(parents=True, exist_ok=True)
        created = _utc_now()
        session_id = created.strftime("validation_%Y%m%dT%H%M%S_%fZ")
        root = collection / session_id
        root.mkdir(exist_ok=False)
        configuration = root / "configuration"
        configuration.mkdir()
        (root / "raw_pairs").mkdir()
        (root / "analyses").mkdir()

        copied_target = configuration / "charuco_target.yaml"
        copied_calibration = configuration / "fixed_stereo_calibration.yaml"
        copied_target.write_bytes(source_target.read_bytes())
        copied_calibration.write_bytes(source_calibration.read_bytes())

        manifest: dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "asset_type": "d435i_ir_stereo_independent_validation_session",
            "session_id": session_id,
            "status": "capturing",
            "created_at_utc": _utc_text(created),
            "calibration_refit_performed": False,
            "stream": {
                "serial_number": serial_number,
                "width": image_size[0],
                "height": image_size[1],
                "frames_per_second": frames_per_second,
                "pixel_format": "Y8",
                "infrared_emitter_enabled": emitter_enabled,
            },
            "target": {
                **_file_record(copied_target, root),
                "source_path": str(source_target),
            },
            "fixed_calibration": {
                **_file_record(copied_calibration, root),
                "source_path": str(source_calibration),
                "baseline_m": calibration.baseline_m,
            },
            "rectification": rectification.model_dump(mode="json"),
            "thresholds": asdict(thresholds),
            "raw_pairs": [],
            "analyses": [],
            "result": None,
        }
        session = cls(root, manifest)
        session._write_manifest()
        return session

    @classmethod
    def open(cls, root: str | Path) -> StereoValidationAssetSession:
        session_root = Path(root).resolve()
        manifest_path = session_root / "validation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise StereoValidationError("unsupported stereo validation schema")
        if manifest.get("asset_type") != "d435i_ir_stereo_independent_validation_session":
            raise StereoValidationError("not a D435i IR stereo validation session")
        session = cls(session_root, manifest)
        session._verified_configuration("target")
        session._verified_configuration("fixed_calibration")
        return session

    @property
    def raw_pair_count(self) -> int:
        return len(self._manifest["raw_pairs"])

    @property
    def image_size(self) -> tuple[int, int]:
        stream = self._manifest["stream"]
        return int(stream["width"]), int(stream["height"])

    @property
    def target_path(self) -> Path:
        return self._verified_configuration("target")

    @property
    def calibration_path(self) -> Path:
        return self._verified_configuration("fixed_calibration")

    @property
    def thresholds(self) -> StereoValidationThresholds:
        return StereoValidationThresholds(**self._manifest["thresholds"])

    @property
    def rectification(self) -> StereoRectificationConfig:
        return StereoRectificationConfig.model_validate(self._manifest["rectification"])

    def _verified_configuration(self, key: str) -> Path:
        record = self._manifest[key]
        path = _resolve_asset(self.root, record["path"])
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise StereoValidationError(f"{key} checksum mismatch")
        return path

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self._manifest)

    def record_device_info(self, information: dict[str, str]) -> None:
        with self._lock:
            if self.raw_pair_count:
                raise StereoValidationError(
                    "device identity cannot change after validation capture begins"
                )
            normalized = {str(key): str(value) for key, value in information.items()}
            configured = self._manifest["stream"].get("serial_number")
            detected = normalized.get("serial_number")
            if configured is not None and detected is not None and configured != detected:
                raise StereoValidationError(
                    f"configured D435i serial {configured} does not match detected {detected}"
                )
            if detected is not None:
                self._manifest["stream"]["serial_number"] = detected
            self._manifest["device"] = normalized
            self._write_manifest()

    def mark_capture_failed(self, message: str) -> None:
        with self._lock:
            if self._manifest["status"] == "capturing":
                self._manifest["status"] = "capture_failed"
                self._manifest["capture_error"] = message
                self._manifest["capture_failed_at_utc"] = _utc_text()
                self._write_manifest()

    def mark_capture_closed(self) -> None:
        with self._lock:
            if self._manifest["status"] == "capturing":
                self._manifest["status"] = "capture_closed"
                self._manifest["capture_closed_at_utc"] = _utc_text()
                self._write_manifest()

    def record_pair(self, frame: RawInfraredStereoFrame) -> str:
        with self._lock:
            if self._manifest["status"] == "completed":
                raise StereoValidationError("completed validation sessions are immutable")
            expected = (self.image_size[1], self.image_size[0])
            if frame.left.shape != expected:
                raise StereoValidationError(
                    f"raw pair shape {frame.left.shape} does not match validation {expected}"
                )
            keys = {
                (int(item["left_frame_number"]), int(item["right_frame_number"]))
                for item in self._manifest["raw_pairs"]
            }
            if frame.key in keys:
                raise StereoValidationError(f"raw stereo frame {frame.key} is already stored")
            pair_id = f"pair_{self.raw_pair_count:04d}"
            pair_root = self.root / "raw_pairs" / pair_id
            partial = pair_root.with_name(pair_root.name + ".partial")
            if pair_root.exists() or partial.exists():
                raise FileExistsError(pair_root)
            partial.mkdir()
            left_path = partial / "left_ir.png"
            right_path = partial / "right_ir.png"
            _write_image(left_path, frame.left)
            _write_image(right_path, frame.right)
            metadata = {
                "schema_version": 1,
                "pair_id": pair_id,
                "captured_at_utc": frame.captured_at_utc,
                "left_frame_number": frame.left_frame_number,
                "right_frame_number": frame.right_frame_number,
                "left_timestamp_ms": frame.left_timestamp_ms,
                "right_timestamp_ms": frame.right_timestamp_ms,
                "timestamp_domain": frame.timestamp_domain,
                "synchronization_delta_ms": frame.synchronization_delta_ms,
                "shape": list(frame.left.shape),
                "dtype": str(frame.left.dtype),
                "left_ir": _file_record(left_path, partial),
                "right_ir": _file_record(right_path, partial),
            }
            metadata_path = partial / "frame_metadata.json"
            _atomic_json(metadata_path, metadata)
            partial.rename(pair_root)
            self._manifest["raw_pairs"].append(
                {
                    "pair_id": pair_id,
                    "path": pair_root.relative_to(self.root).as_posix(),
                    "captured_at_utc": frame.captured_at_utc,
                    "left_frame_number": frame.left_frame_number,
                    "right_frame_number": frame.right_frame_number,
                    "synchronization_delta_ms": frame.synchronization_delta_ms,
                    "metadata_sha256": _sha256(pair_root / "frame_metadata.json"),
                }
            )
            self._manifest["status"] = "capturing"
            self._write_manifest()
            return pair_id

    def _load_pair(
        self, record: dict[str, Any]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        pair_root = _resolve_asset(self.root, record["path"])
        metadata_path = pair_root / "frame_metadata.json"
        if _sha256(metadata_path) != str(record["metadata_sha256"]):
            raise StereoValidationError(
                f"raw-pair metadata checksum mismatch: {record['pair_id']}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        images: list[NDArray[np.uint8]] = []
        for field in ("left_ir", "right_ir"):
            file_record = metadata[field]
            path = _resolve_asset(pair_root, file_record["path"])
            if _sha256(path) != str(file_record["sha256"]):
                raise StereoValidationError(
                    f"raw image checksum mismatch: {record['pair_id']}/{field}"
                )
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise StereoValidationError(f"cannot decode raw validation image: {path}")
            images.append(image)
        return images[0], images[1]

    def analyze(self) -> StereoValidationResult:
        """Evaluate hold-out observations without modifying the fixed calibration."""

        with self._lock:
            if self._manifest["status"] == "completed":
                raise StereoValidationError("completed validation sessions are immutable")
            run_id = f"analysis_{len(self._manifest['analyses']) + 1:03d}"
            analysis_root = self.root / "analyses" / run_id
            analysis_root.mkdir(exist_ok=False)
            (analysis_root / "pairs").mkdir()
            entry: dict[str, Any] = {
                "run_id": run_id,
                "status": "running",
                "started_at_utc": _utc_text(),
                "path": analysis_root.relative_to(self.root).as_posix(),
                "calibration_refit_performed": False,
            }
            self._manifest["analyses"].append(entry)
            self._manifest["status"] = "analyzing"
            self._write_manifest()

        try:
            result = self._run_analysis(analysis_root)
        except Exception as exc:
            with self._lock:
                entry["status"] = "failed"
                entry["failed_at_utc"] = _utc_text()
                entry["error"] = str(exc)
                self._manifest["status"] = "analysis_failed"
                self._write_manifest()
            raise

        with self._lock:
            entry["status"] = "completed"
            entry["completed_at_utc"] = _utc_text()
            entry["report"] = _file_record(result.report_json, self.root)
            entry["passed"] = result.metrics.passed
            self._manifest["result"] = {
                "analysis_run_id": run_id,
                "report": _file_record(result.report_json, self.root),
                "report_text": _file_record(result.report_text, self.root),
                "metrics": asdict(result.metrics),
            }
            self._manifest["status"] = "completed"
            self._manifest["completed_at_utc"] = _utc_text()
            self._write_manifest()
        return result

    def _run_analysis(self, analysis_root: Path) -> StereoValidationResult:
        calibration = load_stereo_calibration(self.calibration_path)
        target = StereoCharucoBoard.read(self.target_path)
        detector = StereoCharucoDetector(target)
        rectifier = StereoRectifier(calibration, self.rectification)
        evaluations: list[_PairEvaluation] = []
        pair_reports: list[dict[str, Any]] = []
        accepted_ids: list[str] = []
        rejected_ids: list[str] = []

        for record in self._manifest["raw_pairs"]:
            pair_id = str(record["pair_id"])
            left, right = self._load_pair(record)
            left_detection = detector.detect(left)
            right_detection = detector.detect(right)
            common_ids = (
                sorted(set(left_detection.ids.tolist()) & set(right_detection.ids.tolist()))
                if left_detection is not None and right_detection is not None
                else []
            )
            reasons: list[str] = []
            if left_detection is None:
                reasons.append("left_board_not_detected")
            if right_detection is None:
                reasons.append("right_board_not_detected")
            if left_detection is not None and left_detection.corner_count < (
                target.minimum_corners_per_camera
            ):
                reasons.append("left_corner_count_below_threshold")
            if right_detection is not None and right_detection.corner_count < (
                target.minimum_corners_per_camera
            ):
                reasons.append("right_corner_count_below_threshold")
            if left_detection is not None and right_detection is not None and len(common_ids) < 6:
                reasons.append("fewer_than_six_common_corners")

            pair_root = analysis_root / "pairs" / pair_id
            pair_root.mkdir()
            left_detection_path = pair_root / "left_detection.png"
            right_detection_path = pair_root / "right_detection.png"
            _write_image(left_detection_path, _annotated_detection(left, left_detection))
            _write_image(right_detection_path, _annotated_detection(right, right_detection))
            report: dict[str, Any] = {
                "pair_id": pair_id,
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "left_corner_count": 0 if left_detection is None else left_detection.corner_count,
                "right_corner_count": 0
                if right_detection is None
                else right_detection.corner_count,
                "common_corner_count": len(common_ids),
                "files": {
                    "left_detection": _file_record(left_detection_path, analysis_root),
                    "right_detection": _file_record(right_detection_path, analysis_root),
                },
            }
            if reasons:
                rejected_ids.append(pair_id)
                validation_path = pair_root / "validation.json"
                _atomic_json(validation_path, report)
                report["files"]["validation_record"] = _file_record(
                    validation_path, analysis_root
                )
                pair_reports.append(report)
                continue

            sample = StereoCharucoSample(pair_id, left_detection, right_detection)  # type: ignore[arg-type]
            evaluation = _evaluate_pair(sample, calibration, rectifier)
            stereo_frame = StereoFrame(
                0,
                int(record["left_frame_number"]),
                0.0,
                0.0,
                left,
                right,
                None,
                calibration,
            )
            rectified_frame = rectifier.rectify(stereo_frame)
            left_visual, right_visual, overlay = _rectified_visualizations(
                rectified_frame.left_ir,
                rectified_frame.right_ir,
                evaluation,
            )
            left_rectified_path = pair_root / "left_rectified.png"
            right_rectified_path = pair_root / "right_rectified.png"
            overlay_path = pair_root / "rectified_epipolar_overlay.png"
            _write_image(left_rectified_path, left_visual)
            _write_image(right_rectified_path, right_visual)
            _write_image(overlay_path, overlay)
            vertical_abs = np.abs(evaluation.vertical_signed_px)
            pair_metrics = {
                "vertical_disparity_signed_mean_px": float(
                    np.mean(evaluation.vertical_signed_px)
                ),
                "vertical_disparity_rmse_px": _rmse(evaluation.vertical_signed_px),
                "vertical_disparity_p95_px": float(np.percentile(vertical_abs, 95)),
                "vertical_disparity_max_px": float(np.max(vertical_abs)),
                "left_reprojection_rmse_px": _rmse(evaluation.left_reprojection_px),
                "right_reprojection_rmse_px": _rmse(evaluation.right_reprojection_px),
                "stereo_transfer_rmse_px": _rmse(evaluation.stereo_transfer_px),
            }
            report["metrics"] = pair_metrics
            report["files"].update(
                {
                    "left_rectified": _file_record(left_rectified_path, analysis_root),
                    "right_rectified": _file_record(right_rectified_path, analysis_root),
                    "rectified_epipolar_overlay": _file_record(overlay_path, analysis_root),
                }
            )
            validation_path = pair_root / "validation.json"
            _atomic_json(validation_path, report)
            report["files"]["validation_record"] = _file_record(
                validation_path, analysis_root
            )
            evaluations.append(evaluation)
            accepted_ids.append(pair_id)
            pair_reports.append(report)

        thresholds = self.thresholds
        if not evaluations:
            raise StereoValidationError("no validation pair passed ChArUco detection")
        vertical = np.concatenate([item.vertical_signed_px for item in evaluations])
        left_reprojection = np.concatenate(
            [item.left_reprojection_px for item in evaluations]
        )
        right_reprojection = np.concatenate(
            [item.right_reprojection_px for item in evaluations]
        )
        transfer = np.concatenate([item.stereo_transfer_px for item in evaluations])
        vertical_abs = np.abs(vertical)
        vertical_rmse = _rmse(vertical)
        vertical_p95 = float(np.percentile(vertical_abs, 95))
        left_rmse = _rmse(left_reprojection)
        right_rmse = _rmse(right_reprojection)
        transfer_rmse = _rmse(transfer)
        passed = (
            len(evaluations) >= thresholds.minimum_accepted_pairs
            and vertical_rmse <= thresholds.maximum_vertical_disparity_rmse_px
            and vertical_p95 <= thresholds.maximum_vertical_disparity_p95_px
            and max(left_rmse, right_rmse)
            <= thresholds.maximum_monocular_reprojection_rmse_px
            and transfer_rmse <= thresholds.maximum_stereo_transfer_rmse_px
        )
        metrics = StereoValidationMetrics(
            len(evaluations),
            len(rejected_ids),
            len(vertical),
            float(np.mean(vertical)),
            vertical_rmse,
            vertical_p95,
            float(np.max(vertical_abs)),
            left_rmse,
            right_rmse,
            transfer_rmse,
            passed,
        )
        report_payload = {
            "schema_version": 1,
            "report_type": "d435i_ir_stereo_independent_validation",
            "created_at_utc": _utc_text(),
            "session_id": self._manifest["session_id"],
            "calibration_refit_performed": False,
            "fixed_calibration": self._manifest["fixed_calibration"],
            "target": self._manifest["target"],
            "rectification": self._manifest["rectification"],
            "thresholds": asdict(thresholds),
            "metrics": asdict(metrics),
            "accepted_pair_ids": accepted_ids,
            "rejected_pair_ids": rejected_ids,
            "pairs": pair_reports,
        }
        report_json = analysis_root / "validation_report.json"
        _atomic_json(report_json, report_payload)
        status = "PASS" if passed else "FAIL"
        report_text = analysis_root / "validation_summary.txt"
        report_text.write_text(
            "\n".join(
                (
                    "BiBladeFusion D435i IR stereo independent validation",
                    f"result: {status}",
                    "calibration refit performed: false",
                    f"accepted/rejected pairs: {len(evaluations)}/{len(rejected_ids)}",
                    f"common corners: {len(vertical)}",
                    f"vertical disparity mean: {np.mean(vertical):.6f} px",
                    f"vertical disparity RMSE/P95/max: "
                    f"{vertical_rmse:.6f}/{vertical_p95:.6f}/{np.max(vertical_abs):.6f} px",
                    f"left/right reprojection RMSE: {left_rmse:.6f}/{right_rmse:.6f} px",
                    f"left-to-right stereo transfer RMSE: {transfer_rmse:.6f} px",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return StereoValidationResult(
            self.root,
            analysis_root,
            report_json,
            report_text,
            metrics,
            tuple(accepted_ids),
            tuple(rejected_ids),
        )


def validate_stereo_asset_session(
    session: StereoValidationAssetSession,
) -> StereoValidationResult:
    """Run one immutable validation analysis; calibration parameters remain fixed."""

    if session.raw_pair_count < session.thresholds.minimum_accepted_pairs:
        raise StereoValidationError(
            f"need at least {session.thresholds.minimum_accepted_pairs} raw validation pairs, "
            f"received {session.raw_pair_count}"
        )
    return session.analyze()
