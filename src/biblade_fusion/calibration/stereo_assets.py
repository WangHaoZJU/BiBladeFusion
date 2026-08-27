"""Append-only digital assets for D435i infrared stereo calibration sessions."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from biblade_fusion.calibration.stereo_charuco import (
    CharucoImageDetection,
    DistortionModel,
    SolvedStereoCalibration,
    StereoCharucoBoard,
    StereoCharucoDetector,
    StereoCharucoSample,
    compare_and_solve_stereo_charuco,
    load_stereo_calibration,
    solve_stereo_charuco,
    write_stereo_calibration,
)


class StereoCalibrationAssetError(RuntimeError):
    """A calibration asset cannot be created or its integrity check failed."""


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


def _write_png(path: Path, image: NDArray[np.uint8]) -> None:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise StereoCalibrationAssetError("raw infrared assets must be uint8 grayscale images")
    if not cv2.imwrite(str(path), image):
        raise StereoCalibrationAssetError(f"failed to write infrared image: {path}")


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_asset(root: Path, relative: object) -> Path:
    supplied = Path(str(relative))
    if supplied.is_absolute():
        raise StereoCalibrationAssetError("asset manifest paths must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / supplied).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise StereoCalibrationAssetError("asset manifest path escapes the session root")
    return resolved


@dataclass(frozen=True, slots=True)
class RawInfraredStereoFrame:
    """One synchronized raw-Y8 pair with acquisition provenance."""

    left: NDArray[np.uint8]
    right: NDArray[np.uint8]
    left_frame_number: int
    right_frame_number: int
    left_timestamp_ms: float
    right_timestamp_ms: float
    timestamp_domain: str
    captured_at_utc: str

    def __post_init__(self) -> None:
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("raw infrared frames must use uint8 pixels")
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("raw infrared frames must be equal-size grayscale images")
        if self.left_frame_number < 0 or self.right_frame_number < 0:
            raise ValueError("frame numbers must be non-negative")
        if not np.isfinite((self.left_timestamp_ms, self.right_timestamp_ms)).all():
            raise ValueError("frame timestamps must be finite")

    @property
    def synchronization_delta_ms(self) -> float:
        return abs(self.left_timestamp_ms - self.right_timestamp_ms)

    @property
    def key(self) -> tuple[int, int]:
        return self.left_frame_number, self.right_frame_number


class LatestStereoFrameMailbox:
    """Bound a live preview to one notification while retaining only the newest pair."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: RawInfraredStereoFrame | None = None
        self._notification_pending = False

    def publish(self, frame: RawInfraredStereoFrame) -> bool:
        """Store ``frame`` and report whether the GUI needs one new notification."""

        with self._lock:
            self._latest = frame
            if self._notification_pending:
                return False
            self._notification_pending = True
            return True

    def take_for_preview(self) -> RawInfraredStereoFrame | None:
        """Return the newest pair and allow the producer to queue one later notification."""

        with self._lock:
            frame = self._latest
            self._notification_pending = False
            return frame

    def snapshot(self) -> RawInfraredStereoFrame | None:
        """Return the newest synchronized pair without affecting preview flow control."""

        with self._lock:
            return self._latest


@dataclass(frozen=True, slots=True)
class StereoDetectionRun:
    run_id: str
    samples: tuple[StereoCharucoSample, ...]
    accepted_pair_ids: tuple[str, ...]
    rejected_pair_ids: tuple[str, ...]
    summary_path: Path


class StereoCalibrationAssetSession:
    """Create and verify one append-only stereo-calibration asset session."""

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
        image_size: tuple[int, int],
        frames_per_second: int,
        serial_number: str | None,
        emitter_enabled: bool,
    ) -> StereoCalibrationAssetSession:
        collection = Path(collection_root)
        collection.mkdir(parents=True, exist_ok=True)
        created = _utc_now()
        prefix = created.strftime("session_%Y%m%dT%H%M%S_%fZ")
        root = collection / prefix
        root.mkdir(exist_ok=False)
        (root / "configuration").mkdir()
        (root / "raw_pairs").mkdir()
        (root / "analyses").mkdir()

        source_target = Path(target_path).resolve()
        if not source_target.is_file():
            raise FileNotFoundError(source_target)
        copied_target = root / "configuration" / "charuco_target.yaml"
        copied_target.write_bytes(source_target.read_bytes())

        manifest: dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "asset_type": "d435i_raw_ir_stereo_calibration_session",
            "session_id": prefix,
            "status": "capturing",
            "created_at_utc": _utc_text(created),
            "factory_intrinsics_used": False,
            "factory_stereo_extrinsics_used": False,
            "stream": {
                "serial_number": serial_number,
                "width": int(image_size[0]),
                "height": int(image_size[1]),
                "frames_per_second": int(frames_per_second),
                "pixel_format": "Y8",
                "infrared_emitter_enabled": bool(emitter_enabled),
            },
            "target": {
                **_file_record(copied_target, root),
                "source_path": str(source_target),
            },
            "raw_pairs": [],
            "analyses": [],
            "result": None,
        }
        session = cls(root, manifest)
        session._write_manifest()
        return session

    @classmethod
    def open(cls, root: str | Path) -> StereoCalibrationAssetSession:
        session_root = Path(root).resolve()
        manifest_path = session_root / "session_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise StereoCalibrationAssetError("unsupported stereo asset-session schema")
        if manifest.get("asset_type") != "d435i_raw_ir_stereo_calibration_session":
            raise StereoCalibrationAssetError("not a D435i IR stereo calibration session")
        session = cls(session_root, manifest)
        _ = session.target_path
        return session

    @property
    def raw_pair_count(self) -> int:
        return len(self._manifest["raw_pairs"])

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self._manifest["stream"]["width"]), int(self._manifest["stream"]["height"])

    @property
    def target_path(self) -> Path:
        record = self._manifest["target"]
        path = _resolve_asset(self.root, record["path"])
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise StereoCalibrationAssetError("copied ChArUco target checksum mismatch")
        return path

    def record_device_info(self, information: dict[str, str]) -> None:
        """Bind non-calibration device identity to the session before acquisition."""

        with self._lock:
            if self.raw_pair_count:
                raise StereoCalibrationAssetError(
                    "device identity cannot be changed after raw acquisition has started"
                )
            normalized = {str(key): str(value) for key, value in information.items()}
            configured = self._manifest["stream"].get("serial_number")
            detected = normalized.get("serial_number")
            if configured is not None and detected is not None and configured != detected:
                raise StereoCalibrationAssetError(
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

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self._manifest)

    def record_pair(self, frame: RawInfraredStereoFrame) -> str:
        """Atomically append one raw pair; existing pair assets are never overwritten."""

        with self._lock:
            if self._manifest["status"] == "completed":
                raise StereoCalibrationAssetError("completed calibration sessions are immutable")
            expected_shape = (
                int(self._manifest["stream"]["height"]),
                int(self._manifest["stream"]["width"]),
            )
            if frame.left.shape != expected_shape:
                raise StereoCalibrationAssetError(
                    f"raw pair shape {frame.left.shape} does not match session {expected_shape}"
                )
            existing_keys = {
                (int(item["left_frame_number"]), int(item["right_frame_number"]))
                for item in self._manifest["raw_pairs"]
            }
            if frame.key in existing_keys:
                raise StereoCalibrationAssetError(
                    f"raw stereo frame {frame.key} is already part of this session"
                )
            pair_id = f"pair_{self.raw_pair_count:04d}"
            pair_root = self.root / "raw_pairs" / pair_id
            partial = pair_root.with_name(pair_root.name + ".partial")
            if pair_root.exists() or partial.exists():
                raise FileExistsError(f"raw-pair destination already exists: {pair_root}")
            partial.mkdir()
            left_path = partial / "left_ir.png"
            right_path = partial / "right_ir.png"
            _write_png(left_path, frame.left)
            _write_png(right_path, frame.right)
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
            _atomic_json(partial / "frame_metadata.json", metadata)
            partial.rename(pair_root)
            record = {
                "pair_id": pair_id,
                "path": pair_root.relative_to(self.root).as_posix(),
                "captured_at_utc": frame.captured_at_utc,
                "left_frame_number": frame.left_frame_number,
                "right_frame_number": frame.right_frame_number,
                "synchronization_delta_ms": frame.synchronization_delta_ms,
                "metadata_sha256": _sha256(pair_root / "frame_metadata.json"),
            }
            self._manifest["raw_pairs"].append(record)
            self._manifest["status"] = "capturing"
            self._write_manifest()
            return pair_id

    def _load_verified_pair(
        self, record: dict[str, Any]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        pair_root = _resolve_asset(self.root, record["path"])
        metadata_path = pair_root / "frame_metadata.json"
        if _sha256(metadata_path) != str(record["metadata_sha256"]):
            raise StereoCalibrationAssetError(
                f"raw-pair metadata checksum mismatch: {record['pair_id']}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        images: list[NDArray[np.uint8]] = []
        for field in ("left_ir", "right_ir"):
            file_record = metadata[field]
            path = _resolve_asset(pair_root, file_record["path"])
            if _sha256(path) != str(file_record["sha256"]):
                raise StereoCalibrationAssetError(
                    f"raw image checksum mismatch: {record['pair_id']}/{field}"
                )
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise StereoCalibrationAssetError(f"cannot decode raw image: {path}")
            images.append(image)
        return images[0], images[1]

    @staticmethod
    def _annotated(
        image: NDArray[np.uint8], detection: CharucoImageDetection | None
    ) -> NDArray[np.uint8]:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if detection is None:
            cv2.putText(
                output,
                "REJECTED: NO BOARD",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 240),
                2,
            )
            return output
        for point in detection.image_points_px:
            cv2.circle(output, tuple(np.rint(point).astype(int)), 3, (30, 220, 30), -1)
        cv2.putText(
            output,
            f"corners={detection.corner_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 220, 30),
            2,
        )
        return output

    @staticmethod
    def _detection_payload(
        detection: CharucoImageDetection | None,
    ) -> dict[str, object] | None:
        if detection is None:
            return None
        return {
            "corner_count": detection.corner_count,
            "marker_count": detection.marker_count,
            "ids": detection.ids.tolist(),
            "image_points_px": detection.image_points_px.tolist(),
            "object_points_m": detection.object_points_m.tolist(),
        }

    def detect_offline(self, detector: StereoCharucoDetector) -> StereoDetectionRun:
        """Verify every raw asset, detect ChArUco corners, and persist an auditable run."""

        with self._lock:
            if self._manifest["status"] == "completed":
                raise StereoCalibrationAssetError("completed calibration sessions are immutable")
            run_id = f"analysis_{len(self._manifest['analyses']) + 1:03d}"
            run_root = self.root / "analyses" / run_id
            run_root.mkdir(exist_ok=False)
            (run_root / "pairs").mkdir()
            entry: dict[str, Any] = {
                "run_id": run_id,
                "status": "running",
                "started_at_utc": _utc_text(),
                "path": run_root.relative_to(self.root).as_posix(),
            }
            self._manifest["analyses"].append(entry)
            self._manifest["status"] = "analyzing"
            self._write_manifest()

        samples: list[StereoCharucoSample] = []
        pair_summaries: list[dict[str, object]] = []
        try:
            for record in self._manifest["raw_pairs"]:
                pair_id = str(record["pair_id"])
                left, right = self._load_verified_pair(record)
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
                    detector.target.minimum_corners_per_camera
                ):
                    reasons.append("left_corner_count_below_threshold")
                if right_detection is not None and right_detection.corner_count < (
                    detector.target.minimum_corners_per_camera
                ):
                    reasons.append("right_corner_count_below_threshold")
                if (
                    left_detection is not None
                    and right_detection is not None
                    and len(common_ids) < 6
                ):
                    reasons.append("fewer_than_six_common_corners")
                accepted = not reasons
                if accepted:
                    samples.append(
                        StereoCharucoSample(pair_id, left_detection, right_detection)  # type: ignore[arg-type]
                    )

                pair_output = run_root / "pairs" / pair_id
                pair_output.mkdir()
                left_annotated = pair_output / "left_detection.png"
                right_annotated = pair_output / "right_detection.png"
                if not cv2.imwrite(str(left_annotated), self._annotated(left, left_detection)):
                    raise StereoCalibrationAssetError(f"failed to write {left_annotated}")
                if not cv2.imwrite(str(right_annotated), self._annotated(right, right_detection)):
                    raise StereoCalibrationAssetError(f"failed to write {right_annotated}")
                detection_payload = {
                    "schema_version": 1,
                    "pair_id": pair_id,
                    "accepted": accepted,
                    "rejection_reasons": reasons,
                    "common_corner_count": len(common_ids),
                    "common_ids": common_ids,
                    "left_laplacian_variance": float(cv2.Laplacian(left, cv2.CV_64F).var()),
                    "right_laplacian_variance": float(cv2.Laplacian(right, cv2.CV_64F).var()),
                    "left": self._detection_payload(left_detection),
                    "right": self._detection_payload(right_detection),
                    "files": {
                        "left_annotated": _file_record(left_annotated, run_root),
                        "right_annotated": _file_record(right_annotated, run_root),
                    },
                }
                detection_path = pair_output / "detection.json"
                _atomic_json(detection_path, detection_payload)
                pair_summaries.append(
                    {
                        "pair_id": pair_id,
                        "accepted": accepted,
                        "rejection_reasons": reasons,
                        "common_corner_count": len(common_ids),
                        "detection": _file_record(detection_path, run_root),
                    }
                )

            accepted_ids = tuple(item.sample_id for item in samples)
            rejected_ids = tuple(
                str(item["pair_id"]) for item in pair_summaries if not bool(item["accepted"])
            )
            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "completed_at_utc": _utc_text(),
                "raw_pair_count": len(pair_summaries),
                "accepted_pair_count": len(accepted_ids),
                "rejected_pair_count": len(rejected_ids),
                "minimum_corners_per_camera": detector.target.minimum_corners_per_camera,
                "accepted_pair_ids": list(accepted_ids),
                "rejected_pair_ids": list(rejected_ids),
                "pairs": pair_summaries,
            }
            summary_path = run_root / "detection_summary.json"
            _atomic_json(summary_path, summary)
        except Exception as exc:
            self.mark_analysis_failed(run_id, str(exc))
            raise

        with self._lock:
            entry = self._analysis_entry(run_id)
            entry.update(
                {
                    "status": "detected",
                    "completed_at_utc": _utc_text(),
                    "raw_pair_count": len(pair_summaries),
                    "accepted_pair_count": len(samples),
                    "rejected_pair_count": len(pair_summaries) - len(samples),
                    "summary": _file_record(summary_path, self.root),
                }
            )
            self._write_manifest()
        return StereoDetectionRun(
            run_id,
            tuple(samples),
            accepted_ids,
            rejected_ids,
            summary_path,
        )

    def _analysis_entry(self, run_id: str) -> dict[str, Any]:
        for entry in self._manifest["analyses"]:
            if entry["run_id"] == run_id:
                return entry
        raise KeyError(run_id)

    def solution_path(self, run_id: str) -> Path:
        result_root = self.root / "analyses" / run_id / "result"
        result_root.mkdir(exist_ok=False)
        return result_root / "d435i_ir_stereo_calibration.yaml"

    def record_solution(
        self,
        run_id: str,
        path: str | Path,
        result: SolvedStereoCalibration,
    ) -> None:
        calibration_path = Path(path)
        with self._lock:
            entry = self._analysis_entry(run_id)
            metrics = result.metrics
            result_record = {
                **_file_record(calibration_path, self.root),
                "analysis_run_id": run_id,
                "distortion_model": result.distortion_model.value,
                "sample_count": metrics.sample_count,
                "left_monocular_rms_px": metrics.left_monocular_rms_px,
                "right_monocular_rms_px": metrics.right_monocular_rms_px,
                "joint_stereo_rms_px": metrics.joint_stereo_rms_px,
                "epipolar_rmse_px": metrics.epipolar_rmse_px,
                "epipolar_p95_px": metrics.epipolar_p95_px,
                "baseline_m": result.calibration.baseline_m,
            }
            entry["status"] = "completed"
            entry["result"] = result_record
            self._manifest["result"] = result_record
            self._manifest["status"] = "completed"
            self._manifest["completed_at_utc"] = _utc_text()
            self._write_manifest()

    def mark_analysis_failed(self, run_id: str, message: str) -> None:
        with self._lock:
            entry = self._analysis_entry(run_id)
            entry["status"] = "failed"
            entry["failed_at_utc"] = _utc_text()
            entry["error"] = message
            self._manifest["status"] = "analysis_failed"
            self._write_manifest()


def solve_stereo_asset_session(
    session: StereoCalibrationAssetSession,
    *,
    minimum_samples: int,
    distortion_model: str | DistortionModel,
    runtime_calibration_path: str | Path | None = None,
) -> tuple[StereoDetectionRun, SolvedStereoCalibration, Path]:
    """Run a new immutable offline detection/solution attempt for an asset session."""

    if minimum_samples < 10:
        raise ValueError("minimum_samples must be at least 10")
    selected = (
        distortion_model.value
        if isinstance(distortion_model, DistortionModel)
        else str(distortion_model)
    )
    if selected not in {"auto", *(item.value for item in DistortionModel)}:
        raise ValueError(f"unsupported distortion model: {selected}")
    if selected == "auto" and minimum_samples < 20:
        raise ValueError("automatic distortion comparison requires at least 20 samples")

    target = StereoCharucoBoard.read(session.target_path)
    detector = StereoCharucoDetector(target)
    detection_run: StereoDetectionRun | None = None
    try:
        detection_run = session.detect_offline(detector)
        samples = list(detection_run.samples)
        if len(samples) < minimum_samples:
            raise StereoCalibrationAssetError(
                f"offline detection accepted {len(samples)} pairs, below required "
                f"{minimum_samples}; raw assets remain intact"
            )
        if selected == "auto":
            result = compare_and_solve_stereo_charuco(
                samples,
                session.image_size,
                target,
                minimum_samples=max(20, minimum_samples),
            )
        else:
            result = solve_stereo_charuco(
                samples,
                session.image_size,
                target,
                minimum_samples=minimum_samples,
                distortion_model=DistortionModel(selected),
            )
        output = session.solution_path(detection_run.run_id)
        write_stereo_calibration(output, result, list(detection_run.accepted_pair_ids))
        if runtime_calibration_path is not None:
            publish_runtime_stereo_calibration(output, runtime_calibration_path)
        session.record_solution(detection_run.run_id, output, result)
        return detection_run, result, output
    except Exception as exc:
        if detection_run is not None:
            session.mark_analysis_failed(detection_run.run_id, str(exc))
        raise


def publish_runtime_stereo_calibration(
    source: str | Path,
    destination: str | Path,
) -> Path:
    """Atomically publish one verified asset as the calibration used by later workflows."""

    source_path = Path(source)
    load_stereo_calibration(source_path)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.write_bytes(source_path.read_bytes())
    load_stereo_calibration(temporary)
    temporary.replace(destination_path)
    return destination_path
