#!/usr/bin/env python3
"""Prove CPU/CUDA occupancy equivalence on one immutable experiment source window.

This diagnostic never opens a robot or camera and never writes inside the input
experiment.  It first integrity-decodes the latest occupancy mapping, then
reintegrates every stored source with the CPU and CUDA DDA backends.  Any snapshot
difference, missing CUDA runtime, or backend error is fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import OccupancyConfig
from biblade_fusion.devices.depth_camera.base import CameraIntrinsics
from biblade_fusion.mapping.integrator import DepthIntegrationConfig, DepthRayIntegrator
from biblade_fusion.storage.occupancy_mapping import _read_occupancy_mapping_integrity


class CudaRayValidationError(RuntimeError):
    """CPU/CUDA replay could not be proven exact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_occupancy_root(experiment_root: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in experiment_root.glob(
            "perception/coarse/cycles/*/attempt_*/occupancy_mapping/metadata.json"
        )
    )
    if not candidates:
        raise CudaRayValidationError(
            f"No coarse occupancy mapping exists below {experiment_root}"
        )
    return candidates[-1].resolve()


def _intrinsics(payload: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        int(payload["width"]),
        int(payload["height"]),
        float(payload["fx"]),
        float(payload["fy"]),
        float(payload["cx"]),
        float(payload["cy"]),
        str(payload["distortion_model"]),
        tuple(float(value) for value in payload["distortion_coefficients"]),
    )


def _integration_config(
    occupancy: OccupancyConfig,
    backend: str,
) -> DepthIntegrationConfig:
    return DepthIntegrationConfig(
        minimum_depth_m=occupancy.minimum_depth_m,
        maximum_depth_m=occupancy.maximum_depth_m,
        pixel_stride=occupancy.integration_stride,
        minimum_valid_rays=1,
        free_space_margin_m=occupancy.free_space_margin_m,
        minimum_free_observations=occupancy.minimum_free_observations,
        minimum_free_view_translation_m=occupancy.minimum_free_view_translation_m,
        minimum_free_view_direction_deg=occupancy.minimum_free_view_direction_deg,
        ray_integration_backend=backend,  # type: ignore[arg-type]
    )


def _git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate(experiment_root: Path, output_path: Path) -> Path:
    experiment = experiment_root.resolve()
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CUDA validation report: {output}")
    occupancy_root = _latest_occupancy_root(experiment)
    metadata_path = occupancy_root / "metadata.json"
    decoded = _read_occupancy_mapping_integrity(occupancy_root)
    context = decoded.context.to_payload()
    occupancy_payload = context.get("occupancy_contract")
    stereo_payload = context.get("rectified_stereo")
    if not isinstance(occupancy_payload, dict) or not isinstance(stereo_payload, dict):
        raise CudaRayValidationError("Occupancy context lacks required contracts")
    left_payload = stereo_payload.get("left")
    if not isinstance(left_payload, dict):
        raise CudaRayValidationError("Occupancy context lacks left rectified intrinsics")
    occupancy = OccupancyConfig.model_validate(occupancy_payload)
    intrinsics = _intrinsics(left_payload)
    try:
        torch = import_module("torch")
        if not bool(torch.cuda.is_available()):
            raise CudaRayValidationError(
                "CUDA validation requires torch.cuda.is_available() to be true"
            )
        device_index = int(torch.cuda.current_device())
        device = torch.device("cuda", device_index)
        cuda_runtime = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "device_index": device_index,
            "device_name": str(torch.cuda.get_device_name(device_index)),
            "device_capability": list(torch.cuda.get_device_capability(device_index)),
        }
    except CudaRayValidationError:
        raise
    except Exception as exc:
        raise CudaRayValidationError(
            f"CUDA runtime provenance probe failed: {type(exc).__name__}: {exc}"
        ) from exc

    previous = None
    comparisons: list[dict[str, Any]] = []
    for index, update in enumerate(decoded.updates):
        common = {
            "snapshot": previous,
            "depth_m": update.source_depth_m,
            "intrinsics": intrinsics,
            "base_t_camera": PoseSE3(
                "base",
                "left_rectified",
                update.evidence.base_t_camera_matrix,
            ),
            "valid_mask": update.integration_valid_mask,
            "source_view_id": update.evidence.physical_source_id,
            "observed_at_utc": datetime.fromisoformat(update.evidence.captured_at_utc),
        }
        cpu_integrator = DepthRayIntegrator(
            update.mapping_snapshot.geometry_spec(),
            _integration_config(occupancy, "cpu"),
            mapping_context_hash=decoded.context.content_hash,
        )
        cuda_integrator = DepthRayIntegrator(
            update.mapping_snapshot.geometry_spec(),
            _integration_config(occupancy, "cuda"),
            mapping_context_hash=decoded.context.content_hash,
        )
        cpu_started = time.perf_counter()
        cpu = cpu_integrator.integrate(**common)
        cpu_wall_s = time.perf_counter() - cpu_started
        torch.cuda.synchronize(device)
        cuda_memory_before = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        cuda_started = time.perf_counter()
        cuda = cuda_integrator.integrate(**common)
        cuda_wall_s = time.perf_counter() - cuda_started
        end_event.record()
        end_event.synchronize()
        cuda_device_s = float(start_event.elapsed_time(end_event)) / 1000.0
        cuda_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        if cpu != update.mapping_snapshot:
            raise CudaRayValidationError(
                f"CPU replay differs from stored mapping snapshot at source {index}"
            )
        if cuda != cpu:
            raise CudaRayValidationError(
                f"CUDA replay differs from CPU mapping snapshot at source {index}"
            )
        comparisons.append(
            {
                "source_index": index,
                "physical_source_id": update.evidence.physical_source_id,
                "snapshot_content_hash": cpu.content_hash,
                "cpu_wall_s": cpu_wall_s,
                "cuda_wall_s": cuda_wall_s,
                "cuda_device_s": cuda_device_s,
                "speedup": cpu_wall_s / cuda_wall_s,
                "cuda_memory_allocated_before_bytes": cuda_memory_before,
                "cuda_peak_allocated_bytes": cuda_peak_allocated,
                "cuda_additional_peak_allocated_bytes": max(
                    0,
                    cuda_peak_allocated - cuda_memory_before,
                ),
                "exact_snapshot_equal": True,
            }
        )
        previous = update.snapshot

    project_root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.cuda_ray_integration_validation",
        "authority": "diagnostic_only_not_motion_or_science_authority",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_head(project_root),
        "cuda_runtime": cuda_runtime,
        "experiment_root": str(experiment),
        "occupancy_root": str(occupancy_root),
        "occupancy_metadata_sha256": _sha256(metadata_path),
        "source_count": len(decoded.updates),
        "comparisons": comparisons,
        "all_exact": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(validate(arguments.experiment_root, arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
