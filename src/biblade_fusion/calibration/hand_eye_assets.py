"""Append-only ES68/D435i hand-eye sessions and held-out validation."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from biblade_fusion.acquisition import SynchronizedFrameBundle
from biblade_fusion.calibration.hand_eye import load_hand_eye_calibration
from biblade_fusion.calibration.hand_eye_solver import (
    HandEyeSample,
    HandEyeSolution,
    write_hand_eye_calibration,
    write_hand_eye_samples,
)
from biblade_fusion.calibration.stereo_charuco import load_stereo_calibration
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    RealSenseConfig,
    RobotConfig,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.robotics import Es68ModelResources


class HandEyeAssetError(RuntimeError):
    """A hand-eye session asset is invalid or cannot be written safely."""


@dataclass(frozen=True, slots=True)
class HandEyeValidationMetrics:
    sample_count: int
    translation_rmse_m: float
    translation_p95_m: float
    translation_max_m: float
    rotation_rmse_deg: float
    rotation_p95_deg: float
    rotation_max_deg: float
    reprojection_rmse_px: float
    reprojection_p95_px: float
    reprojection_max_px: float
    passed: bool


@dataclass(frozen=True, slots=True)
class HandEyeValidationResult:
    metrics: HandEyeValidationMetrics
    per_sample: tuple[dict[str, float | str], ...]


class LatestHandEyeBundleMailbox:
    """Retain only the newest synchronized bundle and bound GUI notifications to one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: SynchronizedFrameBundle | None = None
        self._notification_pending = False

    def publish(self, bundle: SynchronizedFrameBundle) -> bool:
        with self._lock:
            self._latest = bundle
            if self._notification_pending:
                return False
            self._notification_pending = True
            return True

    def take_for_preview(self) -> SynchronizedFrameBundle | None:
        with self._lock:
            bundle = self._latest
            self._notification_pending = False
            return bundle

    def snapshot(self) -> SynchronizedFrameBundle | None:
        with self._lock:
            return self._latest


def _utc_text() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _copy_bound(source: Path, destination: Path, root: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.write_bytes(source.read_bytes())
    return {**_file_record(destination, root), "source_path": str(source.resolve())}


def _write_image(path: Path, image: NDArray[np.uint8]) -> None:
    if not cv2.imwrite(str(path), image):
        raise HandEyeAssetError(f"failed to write hand-eye image: {path}")


def _rotation_error_deg(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _camera_matrix(intrinsics: CameraIntrinsics) -> NDArray[np.float64]:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def evaluate_hand_eye_validation(
    solution: HandEyeSolution,
    samples: Sequence[HandEyeSample],
    intrinsics: CameraIntrinsics,
    config: HandEyeConfig,
) -> HandEyeValidationResult:
    """Evaluate new fixed-board poses without changing either optimized transform."""

    if len(samples) < config.validation_minimum_samples:
        raise HandEyeAssetError(
            f"need at least {config.validation_minimum_samples} independent validation poses, "
            f"received {len(samples)}"
        )
    matrix = _camera_matrix(intrinsics)
    distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    pixel_errors: list[NDArray[np.float64]] = []
    per_sample: list[dict[str, float | str]] = []
    left_t_flange = solution.flange_t_left_ir.inverse()
    for sample in samples:
        observed_base_t_target = (
            sample.base_t_flange
            .compose(solution.flange_t_left_ir)
            .compose(sample.left_ir_t_target)
        )
        translation = float(
            np.linalg.norm(
                observed_base_t_target.translation_m - solution.base_t_target.translation_m
            )
        )
        rotation = _rotation_error_deg(
            solution.base_t_target.rotation, observed_base_t_target.rotation
        )
        if sample.object_points_m is None or sample.image_points_px is None:
            raise HandEyeAssetError(
                f"validation sample {sample.sample_id} has no stored corner observations"
            )
        predicted_left_t_target = (
            left_t_flange
            .compose(sample.base_t_flange.inverse())
            .compose(solution.base_t_target)
        )
        rotation_vector, _ = cv2.Rodrigues(predicted_left_t_target.rotation)
        projected, _ = cv2.projectPoints(
            sample.object_points_m,
            rotation_vector,
            predicted_left_t_target.translation_m,
            matrix,
            distortion,
        )
        pixels = np.linalg.norm(
            projected.reshape(-1, 2) - sample.image_points_px, axis=1
        )
        translation_errors.append(translation)
        rotation_errors.append(rotation)
        pixel_errors.append(pixels)
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(pixels)))),
                "reprojection_max_px": float(np.max(pixels)),
            }
        )

    translations = np.asarray(translation_errors, dtype=np.float64)
    rotations = np.asarray(rotation_errors, dtype=np.float64)
    pixels = np.concatenate(pixel_errors)
    translation_rmse = float(np.sqrt(np.mean(np.square(translations))))
    rotation_rmse = float(np.sqrt(np.mean(np.square(rotations))))
    reprojection_rmse = float(np.sqrt(np.mean(np.square(pixels))))
    metrics = HandEyeValidationMetrics(
        sample_count=len(samples),
        translation_rmse_m=translation_rmse,
        translation_p95_m=float(np.percentile(translations, 95)),
        translation_max_m=float(np.max(translations)),
        rotation_rmse_deg=rotation_rmse,
        rotation_p95_deg=float(np.percentile(rotations, 95)),
        rotation_max_deg=float(np.max(rotations)),
        reprojection_rmse_px=reprojection_rmse,
        reprojection_p95_px=float(np.percentile(pixels, 95)),
        reprojection_max_px=float(np.max(pixels)),
        passed=(
            translation_rmse <= config.validation_maximum_translation_rmse_m
            and rotation_rmse <= config.validation_maximum_rotation_rmse_deg
            and reprojection_rmse <= config.validation_maximum_reprojection_rmse_px
        ),
    )
    return HandEyeValidationResult(metrics, tuple(per_sample))


class HandEyeAssetSession:
    """One unique, append-only hand-eye training and validation session."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest_path = root / "session_manifest.json"
        self._manifest = manifest
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        collection_root: str | Path,
        *,
        target_path: str | Path,
        stereo_calibration_path: str | Path,
        robot_config: RobotConfig,
        realsense_config: RealSenseConfig,
        hand_eye_config: HandEyeConfig,
        kinematics_config: KinematicsConfig,
    ) -> HandEyeAssetSession:
        target_source = Path(target_path).resolve()
        stereo_source = Path(stereo_calibration_path).resolve()
        load_stereo_calibration(stereo_source)
        resources = Es68ModelResources.packaged()
        created = datetime.now(UTC)
        session_id = created.strftime("session_%Y%m%dT%H%M%S_%fZ")
        root = Path(collection_root) / session_id
        root.mkdir(parents=True, exist_ok=False)
        configuration = root / "configuration"
        configuration.mkdir()
        (root / "training_samples").mkdir()
        (root / "validation_samples").mkdir()
        (root / "result").mkdir()

        target_record = _copy_bound(
            target_source, configuration / "charuco_target.yaml", root
        )
        stereo_record = _copy_bound(
            stereo_source, configuration / "fixed_stereo_calibration.yaml", root
        )
        kinematics_record = _copy_bound(
            resources.kinematics_yaml,
            configuration / "es68_holorobot_kinematics.yaml",
            root,
        )
        tcp_record = _copy_bound(
            resources.tcp_offset_json,
            configuration / "es68_flange_tcp_offset.json",
            root,
        )
        runtime_path = configuration / "runtime_configuration.json"
        _atomic_json(
            runtime_path,
            {
                "robot": robot_config.model_dump(mode="json"),
                "realsense": realsense_config.model_dump(mode="json"),
                "hand_eye": hand_eye_config.model_dump(mode="json"),
                "kinematics": kinematics_config.model_dump(mode="json"),
            },
        )
        manifest: dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "asset_type": "es68_d435i_left_ir_hand_eye_session",
            "session_id": session_id,
            "status": "connecting",
            "created_at_utc": created.isoformat(),
            "camera_stream": "infrared/1",
            "robot_model": "es68",
            "motion_commanded": False,
            "solver": "OpenCV Park-Martin + joint LM bundle adjustment",
            "configuration": {
                "target": target_record,
                "stereo_calibration": stereo_record,
                "es68_kinematics": kinematics_record,
                "flange_tcp_offset": tcp_record,
                "runtime": _file_record(runtime_path, root),
            },
            "training_samples": [],
            "validation_samples": [],
            "candidate": None,
            "validation": None,
            "result": None,
        }
        session = cls(root, manifest)
        session._write_manifest()
        return session

    @property
    def training_count(self) -> int:
        return self._active_count("training_samples")

    @property
    def validation_count(self) -> int:
        return self._active_count("validation_samples")

    def _active_count(self, key: str) -> int:
        return sum(bool(item.get("active", True)) for item in self._manifest[key])

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self._manifest)

    def record_connection_info(self, information: dict[str, object]) -> None:
        with self._lock:
            self._manifest["connection"] = information
            self._manifest["status"] = "capturing_training"
            self._manifest["connected_at_utc"] = _utc_text()
            self._write_manifest()

    def mark_failed(self, message: str) -> None:
        with self._lock:
            self._manifest["status"] = "failed"
            self._manifest["error"] = message
            self._manifest["failed_at_utc"] = _utc_text()
            self._write_manifest()

    def mark_closed(self) -> None:
        with self._lock:
            if self._manifest["status"] not in {"completed", "failed"}:
                self._manifest["status"] = "closed"
                self._manifest["closed_at_utc"] = _utc_text()
                self._write_manifest()

    def record_sample(
        self,
        phase: Literal["training", "validation"],
        sample: HandEyeSample,
        bundle: SynchronizedFrameBundle,
        annotated_left: NDArray[np.uint8],
    ) -> Path:
        key = f"{phase}_samples"
        with self._lock:
            records = self._manifest[key]
            index = len(records)
            sample_root = self.root / key / f"sample_{index:04d}_{sample.sample_id}"
            partial = sample_root.with_name(sample_root.name + ".partial")
            if sample_root.exists() or partial.exists():
                raise FileExistsError(sample_root)
            partial.mkdir()
            left_path = partial / "left_ir.png"
            right_path = partial / "right_ir_audit.png"
            detection_path = partial / "left_detection.png"
            sample_path = partial / "sample.yaml"
            _write_image(left_path, bundle.stereo.left_ir)
            _write_image(right_path, bundle.stereo.right_ir)
            _write_image(detection_path, cv2.cvtColor(annotated_left, cv2.COLOR_RGB2BGR))
            write_hand_eye_samples(sample_path, [sample])
            metadata = {
                "schema_version": 1,
                "phase": phase,
                "sample_id": sample.sample_id,
                "captured_at_utc": _utc_text(),
                "frame_number": sample.frame_number,
                "files": {
                    "left_ir": _file_record(left_path, partial),
                    "right_ir_audit": _file_record(right_path, partial),
                    "left_detection": _file_record(detection_path, partial),
                    "sample": _file_record(sample_path, partial),
                },
            }
            metadata_path = partial / "metadata.json"
            _atomic_json(metadata_path, metadata)
            partial.rename(sample_root)
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "active": True,
                    "path": sample_root.relative_to(self.root).as_posix(),
                    "metadata_sha256": _sha256(sample_root / "metadata.json"),
                    "frame_number": sample.frame_number,
                    "charuco_corner_count": sample.charuco_corner_count,
                    "reprojection_rmse_px": sample.reprojection_rmse_px,
                    "bracket_ms": sample.bracket_ms,
                }
            )
            self._manifest["status"] = f"capturing_{phase}"
            self._write_manifest()
            return sample_root

    def exclude_last_sample(self, phase: Literal["training", "validation"]) -> str:
        key = f"{phase}_samples"
        with self._lock:
            for record in reversed(self._manifest[key]):
                if bool(record.get("active", True)):
                    record["active"] = False
                    record["excluded_at_utc"] = _utc_text()
                    record["exclusion_reason"] = "operator_undo"
                    self._write_manifest()
                    return str(record["sample_id"])
        raise HandEyeAssetError(f"no active {phase} sample can be undone")

    def record_candidate(
        self,
        solution: HandEyeSolution,
        training_samples: Sequence[HandEyeSample],
        intrinsics: CameraIntrinsics,
    ) -> Path:
        with self._lock:
            if self._manifest["candidate"] is not None:
                raise HandEyeAssetError("this session already contains a hand-eye candidate")
            samples_path = self.root / "result" / "training_samples.yaml"
            candidate_path = self.root / "result" / "hand_eye_candidate.yaml"
            write_hand_eye_samples(samples_path, training_samples)
            write_hand_eye_calibration(
                candidate_path,
                solution,
                intrinsics=intrinsics,
                stereo_calibration_path=(
                    self.root / "configuration" / "fixed_stereo_calibration.yaml"
                ),
                target_path=self.root / "configuration" / "charuco_target.yaml",
            )
            self._manifest["candidate"] = {
                "calibration": _file_record(candidate_path, self.root),
                "training_samples": _file_record(samples_path, self.root),
                "method": solution.method,
                "quality": {
                    "sample_count": solution.sample_count,
                    "translation_rmse_m": solution.translation_rmse_m,
                    "rotation_rmse_deg": solution.rotation_rmse_deg,
                    "rotation_max_deg": solution.rotation_max_deg,
                    "rotation_span_deg": solution.observability.rotation_span_deg,
                    "translation_span_m": solution.observability.translation_span_m,
                    "rotation_axis_diversity": (
                        solution.observability.rotation_axis_diversity
                    ),
                    "bundle_adjustment": asdict(solution.bundle_adjustment),
                },
            }
            self._manifest["status"] = "capturing_validation"
            self._write_manifest()
            return candidate_path

    def finalize_validation(
        self,
        result: HandEyeValidationResult,
        validation_samples: Sequence[HandEyeSample],
        runtime_path: str | Path,
        config: HandEyeConfig,
    ) -> tuple[Path, Path | None, Path]:
        """Store fixed-parameter evidence and publish only when every gate passes."""

        with self._lock:
            candidate_record = self._manifest.get("candidate")
            if candidate_record is None:
                raise HandEyeAssetError("solve a training candidate before validation")
            result_root = self.root / "result"
            attempt = 1 + len(list(result_root.glob("validation_report_*.json")))
            validation_samples_path = result_root / f"validation_samples_{attempt:03d}.yaml"
            report_path = result_root / f"validation_report_{attempt:03d}.json"
            write_hand_eye_samples(validation_samples_path, validation_samples)
            report_payload = {
                "schema_version": 1,
                "report_type": "es68_d435i_left_ir_hand_eye_independent_validation",
                "created_at_utc": _utc_text(),
                "calibration_refit_performed": False,
                "thresholds": {
                    "minimum_samples": config.validation_minimum_samples,
                    "maximum_translation_rmse_m": (
                        config.validation_maximum_translation_rmse_m
                    ),
                    "maximum_rotation_rmse_deg": (
                        config.validation_maximum_rotation_rmse_deg
                    ),
                    "maximum_reprojection_rmse_px": (
                        config.validation_maximum_reprojection_rmse_px
                    ),
                },
                "metrics": asdict(result.metrics),
                "samples": list(result.per_sample),
            }
            _atomic_json(report_path, report_payload)
            self._manifest["validation"] = {
                "report": _file_record(report_path, self.root),
                "samples": _file_record(validation_samples_path, self.root),
                "metrics": asdict(result.metrics),
            }
            final_path: Path | None = None
            published_path: Path | None = None
            if result.metrics.passed:
                candidate_path = self.root / candidate_record["calibration"]["path"]
                payload = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
                payload["independent_validation"] = {
                    "required": True,
                    "passed": True,
                    "calibration_refit_performed": False,
                    "report": _file_record(report_path, self.root),
                    "metrics": asdict(result.metrics),
                }
                final_path = result_root / "es68_d435i_left_ir_hand_eye.yaml"
                temporary = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
                temporary.write_text(
                    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                temporary.replace(final_path)
                published_path = publish_runtime_hand_eye_calibration(
                    final_path, runtime_path, config
                )
                self._manifest["result"] = {
                    "calibration": _file_record(final_path, self.root),
                    "runtime_path": str(published_path.resolve()),
                    "runtime_sha256": _sha256(published_path),
                }
                self._manifest["status"] = "completed"
                self._manifest["completed_at_utc"] = _utc_text()
            else:
                self._manifest["status"] = "validation_failed"
            self._write_manifest()
            return report_path, published_path, validation_samples_path


def publish_runtime_hand_eye_calibration(
    source: str | Path,
    destination: str | Path,
    config: HandEyeConfig,
) -> Path:
    """Verify and atomically publish one independently validated hand-eye result."""

    source_path = Path(source)
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    validation = payload.get("independent_validation", {})
    if validation.get("passed") is not True:
        raise HandEyeAssetError("hand-eye result has no passing independent validation")
    load_hand_eye_calibration(config.model_copy(update={"calibration_path": source_path}))
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(source_path.read_bytes())
        load_hand_eye_calibration(config.model_copy(update={"calibration_path": temporary}))
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path
