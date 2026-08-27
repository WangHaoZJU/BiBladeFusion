"""Atomic reports and portable geometry for fine-view inspection."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from biblade_fusion.workflows.fine_plan_inspection import FinePlanInspection

FINE_PLAN_INSPECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StoredFinePlanInspection:
    root: Path
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": _sha256(path), "size": path.stat().st_size}


def _write_csv(path: Path, inspection: FinePlanInspection) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "view_id",
                "patch_id",
                "side",
                "region",
                "accepted",
                "standoff_distance_m",
                "footprint_width_m",
                "footprint_height_m",
                "projection_fraction",
                "visibility_fraction",
                "distance_policy",
                "reasons",
            )
        )
        for item in inspection.views:
            writer.writerow(
                (
                    item.view_id,
                    item.patch_id,
                    item.side,
                    item.region,
                    item.accepted,
                    f"{item.standoff_distance_m:.9f}",
                    f"{item.footprint_m[0]:.9f}",
                    f"{item.footprint_m[1]:.9f}",
                    f"{item.projection_fraction:.6f}",
                    f"{item.visibility_fraction:.6f}",
                    item.distance_policy,
                    " | ".join(item.reasons),
                )
            )


def _write_ply(path: Path, inspection: FinePlanInspection) -> None:
    points = inspection.scene_points_m
    colors = inspection.scene_colors_rgb
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _frustum_vertices(item: Any) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    matrix = item.base_t_left_ir
    camera = matrix[:3, 3]
    centre = item.target_m
    x_axis = matrix[:3, 0]
    y_axis = matrix[:3, 1]
    half_width = item.footprint_m[0] / 2.0
    half_height = item.footprint_m[1] / 2.0
    corners = np.asarray(
        [
            centre - half_width * x_axis - half_height * y_axis,
            centre + half_width * x_axis - half_height * y_axis,
            centre + half_width * x_axis + half_height * y_axis,
            centre - half_width * x_axis + half_height * y_axis,
        ]
    )
    normal_end = centre + item.outward_normal * max(item.footprint_m) * 0.25
    vertices = np.vstack((camera, centre, corners, normal_end))
    lines = (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 2),
        (1, 6),
    )
    return vertices, lines


def _write_obj(path: Path, inspection: FinePlanInspection) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# BiBladeFusion non-executable fine-view frusta\n")
        offset = 1
        for item in inspection.views:
            vertices, lines = _frustum_vertices(item)
            stream.write(
                f"\no {item.view_id}\n# side={item.side} region={item.region} "
                f"accepted={str(item.accepted).lower()}\n"
            )
            for vertex in vertices:
                stream.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
            for first, second in lines:
                stream.write(f"l {offset + first} {offset + second}\n")
            offset += len(vertices)


def _orthographic_project(
    point: np.ndarray,
    first_axis: int,
    second_axis: int,
    minimum: np.ndarray,
    origin_x: float,
    origin_y: float,
    scale: float,
) -> tuple[float, float]:
    return (
        origin_x + (point[first_axis] - minimum[0]) * scale,
        origin_y - (point[second_axis] - minimum[1]) * scale,
    )


def _write_svg(path: Path, inspection: FinePlanInspection) -> None:
    points = inspection.scene_points_m
    candidate_points = np.vstack(
        [np.vstack((item.camera_position_m, item.target_m)) for item in inspection.views]
    )
    all_points = np.vstack((points, candidate_points))
    axes = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    panel_width, panel_height, padding = 480, 380, 34
    sample = np.linspace(0, len(points) - 1, min(len(points), 7000), dtype=np.int64)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width * 3}" '
        f'height="{panel_height}" viewBox="0 0 {panel_width * 3} {panel_height}">',
        '<rect width="100%" height="100%" fill="#11151b"/>',
    ]
    for panel, (first_axis, second_axis, label) in enumerate(axes):
        minimum = np.min(all_points[:, (first_axis, second_axis)], axis=0)
        maximum = np.max(all_points[:, (first_axis, second_axis)], axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        scale = min(
            (panel_width - 2 * padding) / span[0],
            (panel_height - 2 * padding) / span[1],
        )
        origin_x = panel * panel_width + (panel_width - span[0] * scale) / 2.0
        origin_y = (panel_height + span[1] * scale) / 2.0

        parts.append(
            f'<text x="{panel * panel_width + 16}" y="24" fill="#e7edf4" '
            f'font-family="sans-serif" font-size="16">{label}</text>'
        )
        for index in sample:
            x, y = _orthographic_project(
                points[index], first_axis, second_axis, minimum, origin_x, origin_y, scale
            )
            red, green, blue = inspection.scene_colors_rgb[index]
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.15" '
                f'fill="rgb({red},{green},{blue})" fill-opacity="0.72"/>'
            )
        for item in inspection.views:
            camera_x, camera_y = _orthographic_project(
                item.camera_position_m,
                first_axis,
                second_axis,
                minimum,
                origin_x,
                origin_y,
                scale,
            )
            target_x, target_y = _orthographic_project(
                item.target_m,
                first_axis,
                second_axis,
                minimum,
                origin_x,
                origin_y,
                scale,
            )
            color = "#65d98b" if item.accepted else "#ff5d67"
            parts.append(
                f'<line x1="{camera_x:.2f}" y1="{camera_y:.2f}" '
                f'x2="{target_x:.2f}" y2="{target_y:.2f}" '
                f'stroke="{color}" stroke-width="0.8" stroke-opacity="0.55"/>'
            )
            parts.append(
                f'<circle cx="{camera_x:.2f}" cy="{camera_y:.2f}" r="2.0" fill="{color}"/>'
            )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_fine_plan_inspection(
    output_dir: str | Path,
    inspection: FinePlanInspection,
) -> Path:
    """Write immutable inspection evidence plus portable point/frustum geometry."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Fine-plan inspection output already exists: {output}")
    source_metadata = Path(inspection.source_root) / "metadata.json"
    if not source_metadata.is_file():
        raise ValueError(f"Source coarse-model metadata is missing: {source_metadata}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        _write_csv(temporary / "views.csv", inspection)
        _write_ply(temporary / "patches.ply", inspection)
        _write_obj(temporary / "view_frusta.obj", inspection)
        _write_svg(temporary / "overview.svg", inspection)
        files = {
            name: _record(temporary / name)
            for name in ("views.csv", "patches.ply", "view_frusta.obj", "overview.svg")
        }
        payload = {
            "schema_version": FINE_PLAN_INSPECTION_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "source": {
                "coarse_model": inspection.source_root,
                "coarse_model_schema_version": inspection.source_schema_version,
                "metadata_sha256": _sha256(source_metadata),
            },
            "geometry_passed": inspection.geometry_passed,
            "robot_feasibility": inspection.robot_feasibility,
            "global_reasons": list(inspection.global_reasons),
            "warnings": list(inspection.warnings),
            "region_counts": inspection.region_counts,
            "inspection_configuration": inspection.inspection_configuration,
            "files": files,
            "views": [
                {
                    "view_id": item.view_id,
                    "patch_id": item.patch_id,
                    "side": item.side,
                    "region": item.region,
                    "accepted": item.accepted,
                    "standoff_distance_m": item.standoff_distance_m,
                    "footprint_m": list(item.footprint_m),
                    "projection_fraction": item.projection_fraction,
                    "visibility_fraction": item.visibility_fraction,
                    "distance_policy": item.distance_policy,
                    "reasons": list(item.reasons),
                    "base_T_left_ir": item.base_t_left_ir.tolist(),
                    "target_m": item.target_m.tolist(),
                    "outward_normal": item.outward_normal.tolist(),
                }
                for item in inspection.views
            ],
        }
        (temporary / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output.resolve()


def read_fine_plan_inspection(path: str | Path) -> StoredFinePlanInspection:
    """Verify an inspection artifact and its source binding."""

    root = Path(path).resolve()
    try:
        payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if int(payload["schema_version"]) != FINE_PLAN_INSPECTION_SCHEMA_VERSION:
            raise ValueError("unsupported fine-plan inspection schema")
        if payload.get("motion_authorized") is not False:
            raise ValueError("inspection artifact must explicitly forbid motion")
        source_metadata = Path(str(payload["source"]["coarse_model"])) / "metadata.json"
        if _sha256(source_metadata) != str(payload["source"]["metadata_sha256"]):
            raise ValueError("source coarse-model metadata checksum mismatch")
        for record in payload["files"].values():
            relative = Path(str(record["path"]))
            resolved = (root / relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(root):
                raise ValueError(f"inspection file escapes output: {relative}")
            if _sha256(resolved) != str(record["sha256"]):
                raise ValueError(f"inspection file checksum mismatch: {relative}")
            if resolved.stat().st_size != int(record["size"]):
                raise ValueError(f"inspection file size mismatch: {relative}")
        return StoredFinePlanInspection(root, payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid fine-plan inspection artifact {root}: {exc}") from exc
