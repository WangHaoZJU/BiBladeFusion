"""Immutable static native-depth overlap validation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from biblade_fusion.calibration import HandEyeCalibration, load_hand_eye_calibration
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    NativeOverlapValidationConfig,
    PointCloudConfig,
)
from biblade_fusion.robotics import Es68ModelResources, load_es68_flange_t_tcp
from biblade_fusion.storage.reader import SessionReader
from biblade_fusion.workflows import NativeOverlapReport, evaluate_native_overlap

NATIVE_OVERLAP_SCHEMA_VERSION = 2

_PALETTE_RGB = np.asarray(
    [
        (255, 255, 255),
        (239, 71, 111),
        (6, 214, 160),
        (17, 138, 178),
        (255, 209, 102),
        (131, 56, 236),
        (255, 127, 80),
        (77, 201, 246),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True, slots=True)
class StoredNativeOverlapReport:
    report: NativeOverlapReport
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LegacyNativeOverlapReplay:
    """Integrity-verified schema-1 result with no FK-authority eligibility."""

    metadata: dict[str, Any]
    overlay_points_m: np.ndarray
    overlay_view_indices: np.ndarray
    pair_residuals_m: tuple[np.ndarray, ...]
    verification_status: str = "legacy_tcp_primary_integrity_only"
    current_fk_authority_eligible: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Native-overlap source file is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _array_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {
            **_file_record(path, relative_to=relative_to),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    finally:
        del array


def _safe(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not result:
        raise ValueError("Native-overlap view ID has no safe filename characters")
    return result


def _session_source(session: str | Path, view_id: str) -> dict[str, Any]:
    reader = SessionReader(session)
    descriptor = reader.descriptor(view_id)
    view_root = (reader.path / descriptor.relative_path).resolve()
    metadata_path = view_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    native_name = metadata["stereo"].get("native_depth_file")
    if native_name is None:
        raise ValueError(f"Native-overlap source {view_id} has no native depth")
    config_path = reader.path / "config_snapshot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "root": str(reader.path),
        "view_id": view_id,
        "sequence_index": descriptor.sequence_index,
        "frame_number": int(metadata["stereo"]["frame_number"]),
        "device_time_ms": float(metadata["stereo"]["left_device_time_ms"]),
        "emitter_enabled": bool(config["realsense"]["infrared_emitter_enabled"]),
        "manifest": _source_file(reader.path / "manifest.json"),
        "config_snapshot": _source_file(config_path),
        "view_metadata": _source_file(metadata_path),
        "native_depth": _source_file(view_root / str(native_name)),
    }


def _pair_payload(pair) -> dict[str, Any]:
    diagnostic = pair.icp_diagnostic
    metrics = asdict(pair.metrics)
    metrics["agreement_fractions"] = [list(item) for item in pair.metrics.agreement_fractions]
    metrics["failure_reasons"] = list(pair.metrics.failure_reasons)
    return {
        "reference_view_id": pair.reference_view_id,
        "comparison_view_id": pair.comparison_view_id,
        "metrics": metrics,
        "icp_diagnostic": (
            {
                **asdict(diagnostic),
                "correction_matrix": diagnostic.correction_matrix.tolist(),
            }
            if diagnostic is not None
            else None
        ),
    }


def _report_payload(report: NativeOverlapReport) -> dict[str, Any]:
    return {
        "reference_view_id": report.reference_view_id,
        "view_ids": list(report.view_ids),
        "translation_span_m": report.translation_span_m,
        "rotation_span_deg": report.rotation_span_deg,
        "passed": report.passed,
        "failure_reasons": list(report.failure_reasons),
        "pairs": [_pair_payload(pair) for pair in report.pairs],
    }


def _write_binary_ply(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    colors = _PALETTE_RGB[labels % len(_PALETTE_RGB)]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def _draw_projection(
    image: np.ndarray,
    points_2d: np.ndarray,
    labels: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
) -> None:
    x0, y0 = origin
    width, height = size
    cv2.rectangle(image, (x0, y0), (x0 + width - 1, y0 + height - 1), (70, 70, 70), 1)
    cv2.putText(
        image,
        title,
        (x0 + 10, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    lower = np.percentile(points_2d, 1, axis=0)
    upper = np.percentile(points_2d, 99, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    scale = min((width - 30) / span[0], (height - 50) / span[1])
    center = (lower + upper) / 2.0
    pixels = (points_2d - center) * scale
    pixels[:, 0] += x0 + width / 2.0
    pixels[:, 1] = y0 + height / 2.0 - pixels[:, 1]
    pixels = np.rint(pixels).astype(int)
    inside = (
        (pixels[:, 0] >= x0 + 2)
        & (pixels[:, 0] < x0 + width - 2)
        & (pixels[:, 1] >= y0 + 30)
        & (pixels[:, 1] < y0 + height - 2)
    )
    for view_index in range(int(labels.max()) + 1):
        selected = pixels[inside & (labels == view_index)]
        color_rgb = _PALETTE_RGB[view_index % len(_PALETTE_RGB)]
        color_bgr = tuple(int(value) for value in color_rgb[::-1])
        image[selected[:, 1], selected[:, 0]] = color_bgr


def _write_overview(path: Path, report: NativeOverlapReport) -> None:
    points = report.overlay_points_m
    centered = points - np.median(points, axis=0)
    _, _, axes_t = np.linalg.svd(centered, full_matrices=False)
    local = centered @ axes_t.T
    maximum = 12000
    if len(local) > maximum:
        indices = np.linspace(0, len(local) - 1, maximum, dtype=np.int64)
        local = local[indices]
        labels = report.overlay_view_indices[indices]
    else:
        labels = report.overlay_view_indices
    image = np.full((600, 1800, 3), 18, dtype=np.uint8)
    _draw_projection(image, local[:, [0, 1]], labels, (0, 0), (600, 600), "PCA major/minor")
    _draw_projection(image, local[:, [0, 2]], labels, (600, 0), (600, 600), "PCA major/normal")
    _draw_projection(image, local[:, [1, 2]], labels, (1200, 0), (600, 600), "PCA minor/normal")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write native-overlap overview: {path}")


def _write_metrics_csv(path: Path, report: NativeOverlapReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reference_view_id",
                "comparison_view_id",
                "surface_inlier_fraction",
                "median_absolute_error_mm",
                "rmse_mm",
                "p95_absolute_error_mm",
                "agreement_2mm_fraction",
                "agreement_5mm_fraction",
                "icp_translation_correction_mm",
                "icp_rotation_correction_deg",
                "passed",
                "failure_reasons",
            ]
        )
        for pair in report.pairs:
            agreements = dict(pair.metrics.agreement_fractions)
            diagnostic = pair.icp_diagnostic
            writer.writerow(
                [
                    pair.reference_view_id,
                    pair.comparison_view_id,
                    f"{pair.metrics.surface_inlier_fraction:.9f}",
                    f"{pair.metrics.median_absolute_error_m * 1000:.6f}",
                    f"{pair.metrics.root_mean_square_error_m * 1000:.6f}",
                    f"{pair.metrics.p95_absolute_error_m * 1000:.6f}",
                    f"{agreements.get(0.002, float('nan')):.9f}",
                    f"{agreements.get(0.005, float('nan')):.9f}",
                    (
                        f"{diagnostic.translation_correction_m * 1000:.6f}"
                        if diagnostic is not None
                        and diagnostic.translation_correction_m is not None
                        else ""
                    ),
                    (
                        f"{diagnostic.rotation_correction_deg:.6f}"
                        if diagnostic is not None and diagnostic.rotation_correction_deg is not None
                        else ""
                    ),
                    str(pair.metrics.passed).lower(),
                    "; ".join(pair.metrics.failure_reasons),
                ]
            )


def write_native_overlap_report(
    output_dir: str | Path,
    report: NativeOverlapReport,
    source_sessions: tuple[str | Path, ...],
    hand_eye: HandEyeCalibration,
    hand_eye_config: HandEyeConfig,
    kinematics_config: KinematicsConfig,
    point_cloud_config: PointCloudConfig,
    validation_config: NativeOverlapValidationConfig,
) -> Path:
    """Persist checksummed overlap evidence without modifying source observations."""

    if len(source_sessions) != len(report.view_ids):
        raise ValueError("Native-overlap source-session count does not match report views")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Native-overlap output already exists: {output}")
    sources = tuple(
        _session_source(session, view_id)
        for session, view_id in zip(source_sessions, report.view_ids, strict=True)
    )
    if len({source["root"] for source in sources}) != len(sources):
        raise ValueError("Native-overlap source sessions must be unique")
    if len({source["device_time_ms"] for source in sources}) != len(sources):
        raise ValueError("Native-overlap source frames must be physically distinct")
    if len({source["emitter_enabled"] for source in sources}) != 1:
        raise ValueError("Native-overlap sessions use mixed projector states")
    flange_t_left_ir = hand_eye.require_flange_primary()
    flange_t_tcp = load_es68_flange_t_tcp()
    hand_eye_path = hand_eye.source_path.resolve()
    resources = Es68ModelResources.packaged()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    arrays_dir = temporary / "arrays"
    arrays_dir.mkdir()
    residual_records: list[dict[str, Any]] = []
    try:
        overlay_points_path = arrays_dir / "overlay_points_m.npy"
        overlay_labels_path = arrays_dir / "overlay_view_indices.npy"
        np.save(overlay_points_path, report.overlay_points_m, allow_pickle=False)
        np.save(overlay_labels_path, report.overlay_view_indices, allow_pickle=False)
        for index, pair in enumerate(report.pairs):
            residual_path = arrays_dir / (
                f"residual_{index:03d}_{_safe(pair.comparison_view_id)}.npy"
            )
            np.save(
                residual_path,
                pair.symmetric_signed_residuals_m,
                allow_pickle=False,
            )
            residual_records.append(
                {
                    "comparison_view_id": pair.comparison_view_id,
                    **_array_record(residual_path, relative_to=temporary),
                }
            )
        ply_path = temporary / "overlay_base_frame.ply"
        overview_path = temporary / "overview.png"
        csv_path = temporary / "metrics.csv"
        _write_binary_ply(ply_path, report.overlay_points_m, report.overlay_view_indices)
        _write_overview(overview_path, report)
        _write_metrics_csv(csv_path, report)
        payload: dict[str, Any] = {
            "schema_version": NATIVE_OVERLAP_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "evaluation": _report_payload(report),
            "sources": {
                "sessions": list(sources),
                "hand_eye": {
                    **_source_file(hand_eye_path),
                    "flange_T_left_ir": flange_t_left_ir.matrix.tolist(),
                    "tcp_T_left_ir": hand_eye.tcp_t_left_ir.matrix.tolist(),
                    "flange_T_tcp": flange_t_tcp.matrix.tolist(),
                    "method": hand_eye.method,
                },
                "kinematics_assets": {
                    "model": _source_file(resources.kinematics_yaml),
                    "joint_limits": _source_file(resources.joint_limits_yaml),
                    "flange_tcp": _source_file(resources.tcp_offset_json),
                },
            },
            "files": {
                "overlay_points_m": _array_record(overlay_points_path, relative_to=temporary),
                "overlay_view_indices": _array_record(overlay_labels_path, relative_to=temporary),
                "pair_residuals": residual_records,
                "overlay_ply": _file_record(ply_path, relative_to=temporary),
                "overview_png": _file_record(overview_path, relative_to=temporary),
                "metrics_csv": _file_record(csv_path, relative_to=temporary),
            },
            "processing": {
                "hand_eye": hand_eye_config.model_dump(mode="json"),
                "kinematics": kinematics_config.model_dump(mode="json"),
                "point_cloud": point_cloud_config.model_dump(mode="json"),
                "native_overlap_validation": validation_config.model_dump(mode="json"),
            },
            "interpretation": {
                "primary": (
                    "symmetric projective native-depth residuals use the unmodified "
                    "joints -> packaged ES68 FK -> base_T_flange · "
                    "flange_T_left_ir · left_ir_T_depth chain; controller TCP is "
                    "validation-only"
                ),
                "icp": (
                    "ICP corrections are diagnostic only and were not applied to primary "
                    "metrics, exported overlay points, or pass/fail decisions"
                ),
                "scope": (
                    "This validates internal static-scene consistency, not traceable "
                    "absolute dimensional accuracy"
                ),
            },
        }
        (temporary / "native_overlap_report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _verify_source(record: dict[str, Any]) -> None:
    path = Path(str(record["path"])).resolve()
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"native-overlap source checksum mismatch: {path}")


def _contained(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    path = (root.resolve() / relative_path).resolve()
    if relative_path.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"native-overlap file escapes artifact: {relative}")
    return path


def _load_array(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _contained(root, str(record["path"]))
    if _sha256(path) != str(record["sha256"]):
        raise ValueError(f"native-overlap checksum mismatch: {record['path']}")
    array = np.load(path, allow_pickle=False)
    if str(array.dtype) != str(record["dtype"]) or list(array.shape) != record["shape"]:
        raise ValueError(f"native-overlap array manifest mismatch: {record['path']}")
    return array


def read_native_overlap_report(path: str | Path) -> StoredNativeOverlapReport:
    """Verify sources, derived arrays, metrics, and pass/fail by full recomputation."""

    root = Path(path)
    try:
        payload = json.loads((root / "native_overlap_report.json").read_text(encoding="utf-8"))
        if int(payload["schema_version"]) != NATIVE_OVERLAP_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {payload['schema_version']}")
        sources = payload["sources"]
        session_sources = sources["sessions"]
        for source in session_sources:
            for field in (
                "manifest",
                "config_snapshot",
                "view_metadata",
                "native_depth",
            ):
                _verify_source(source[field])
        _verify_source(sources["hand_eye"])
        for record in sources["kinematics_assets"].values():
            _verify_source(record)
        files = payload["files"]
        overlay_points = _load_array(root, files["overlay_points_m"])
        overlay_labels = _load_array(root, files["overlay_view_indices"])
        residuals = tuple(_load_array(root, record) for record in files["pair_residuals"])
        for field in ("overlay_ply", "overview_png", "metrics_csv"):
            record = files[field]
            derived_path = _contained(root, str(record["path"]))
            if _sha256(derived_path) != str(record["sha256"]) or derived_path.stat().st_size != int(
                record["size_bytes"]
            ):
                raise ValueError(f"native-overlap derived file mismatch: {record['path']}")
        processing = payload["processing"]
        hand_eye_config = HandEyeConfig.model_validate(processing["hand_eye"])
        hand_eye_config = hand_eye_config.model_copy(
            update={"calibration_path": Path(sources["hand_eye"]["path"])}
        )
        hand_eye = load_hand_eye_calibration(hand_eye_config)
        if hand_eye.flange_t_left_ir is None:
            raise ValueError("native-overlap requires flange-primary hand-eye")
        if not np.allclose(
            hand_eye.flange_t_left_ir.matrix,
            sources["hand_eye"]["flange_T_left_ir"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("native-overlap flange-primary hand-eye matrix changed")
        if not np.allclose(
            hand_eye.tcp_t_left_ir.matrix,
            sources["hand_eye"]["tcp_T_left_ir"],
            atol=1e-12,
        ):
            raise ValueError("native-overlap hand-eye matrix changed")
        if not np.allclose(
            load_es68_flange_t_tcp().matrix,
            sources["hand_eye"]["flange_T_tcp"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("native-overlap flange_T_tcp changed")
        bundles = tuple(
            SessionReader(source["root"]).load_bundle(str(source["view_id"]))
            for source in session_sources
        )
        expected = evaluate_native_overlap(
            bundles,
            hand_eye,
            PointCloudConfig.model_validate(processing["point_cloud"]),
            NativeOverlapValidationConfig.model_validate(processing["native_overlap_validation"]),
            kinematics_config=KinematicsConfig.model_validate(processing["kinematics"]),
            hand_eye_config=hand_eye_config,
        )
        if _report_payload(expected) != payload["evaluation"]:
            raise ValueError("native-overlap metrics do not match recomputed sources")
        if not np.allclose(expected.overlay_points_m, overlay_points, atol=1e-12):
            raise ValueError("native-overlap overlay points do not match sources")
        if not np.array_equal(expected.overlay_view_indices, overlay_labels):
            raise ValueError("native-overlap overlay labels do not match sources")
        if len(residuals) != len(expected.pairs):
            raise ValueError("native-overlap residual count does not match pairs")
        for pair, residual in zip(expected.pairs, residuals, strict=True):
            if not np.allclose(pair.symmetric_signed_residuals_m, residual, atol=1e-12):
                raise ValueError(f"native-overlap residuals changed for {pair.comparison_view_id}")
        return StoredNativeOverlapReport(expected, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid native-overlap artifact {root}: {exc}") from exc


def read_legacy_native_overlap_for_replay(
    path: str | Path,
) -> LegacyNativeOverlapReplay:
    """Read schema-1 TCP-primary assets without reinterpreting them as schema 2.

    This API verifies every still-addressable source and every derived file.  It
    deliberately does not return :class:`StoredNativeOverlapReport`, because the
    legacy coordinate chain cannot satisfy the current FK-authority experiment.
    """

    root = Path(path)
    try:
        payload = json.loads(
            (root / "native_overlap_report.json").read_text(encoding="utf-8")
        )
        if int(payload["schema_version"]) != 1:
            raise ValueError(
                f"legacy replay requires schema 1, got {payload['schema_version']}"
            )
        primary = str(payload["interpretation"]["primary"])
        if "base_T_tcp" not in primary or "tcp_T_left_ir" not in primary:
            raise ValueError("legacy report does not declare its TCP-primary chain")
        sources = payload["sources"]
        for source in sources["sessions"]:
            for field in (
                "manifest",
                "config_snapshot",
                "view_metadata",
                "native_depth",
            ):
                _verify_source(source[field])
        _verify_source(sources["hand_eye"])
        files = payload["files"]
        overlay_points = _load_array(root, files["overlay_points_m"])
        overlay_labels = _load_array(root, files["overlay_view_indices"])
        residuals = tuple(
            _load_array(root, record) for record in files["pair_residuals"]
        )
        if (
            overlay_points.ndim != 2
            or overlay_points.shape[1] != 3
            or overlay_labels.shape != (len(overlay_points),)
        ):
            raise ValueError("legacy native-overlap overlay arrays are invalid")
        pairs = payload["evaluation"]["pairs"]
        if len(residuals) != len(pairs):
            raise ValueError("legacy native-overlap residual count differs from metrics")
        for residual in residuals:
            if residual.ndim != 1 or not np.isfinite(residual).all():
                raise ValueError("legacy native-overlap residual array is invalid")
        for field in ("overlay_ply", "overview_png", "metrics_csv"):
            record = files[field]
            derived = _contained(root, str(record["path"]))
            if (
                _sha256(derived) != str(record["sha256"])
                or derived.stat().st_size != int(record["size_bytes"])
            ):
                raise ValueError(
                    f"legacy native-overlap derived file mismatch: {record['path']}"
                )
        for array in (overlay_points, overlay_labels, *residuals):
            array.setflags(write=False)
        return LegacyNativeOverlapReplay(
            payload,
            overlay_points,
            overlay_labels,
            residuals,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid legacy native-overlap artifact {root}: {exc}"
        ) from exc
