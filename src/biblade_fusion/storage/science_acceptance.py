"""Immutable physical acceptance for the geometry-science pipeline.

This asset is deliberately separate from motion acceptance.  It records that the
exact FoundationStereo, foreground, fusion and final-quality contract was measured
against traceable references over the deployed distance/incidence envelope.  The
record is a release gate only and never authorizes robot motion.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import acos, degrees
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

SCIENCE_ACCEPTANCE_SCHEMA_VERSION = 2
SCIENCE_ACCEPTANCE_DECLARATION_SCHEMA_VERSION = 2
_ASSET_TYPE = "biblade_fusion.geometry_science_acceptance"
_DECLARATION_TYPE = "biblade_fusion.geometry_science_acceptance_declaration"
_RUNTIME_SCHEMA = "biblade_fusion.geometry_science_runtime_contract.v3"
_RAW_MANIFEST_TYPE = "biblade_fusion.raw_science_acceptance_asset_manifest"
_EVALUATION_TYPE = "biblade_fusion.geometry_science_evaluation_report"
_REVIEW_TYPE = "biblade_fusion.geometry_science_independent_review_report"
_EVIDENCE_PATHS = {
    "geometry_evaluation_report": "evidence/geometry_evaluation_report.json",
    "raw_acceptance_asset_manifest": "evidence/raw_acceptance_asset_manifest.json",
    "independent_review_report": "evidence/independent_review_report.json",
}
_ASSET_ROLES = {"depth_reference", "annotation", "specimen"}
_DECLARATION = (
    "The exact bound stereo, foreground, fusion and final-surface pipeline passed "
    "traceable physical tests across the declared distance and incidence envelope."
)
_METRICS = (
    "depth_rmse_m",
    "depth_p95_m",
    "depth_absolute_bias_m",
    "bootstrap_mask_precision",
    "bootstrap_mask_recall",
    "fine_mask_precision",
    "fine_mask_recall",
    "final_surface_rmse_m",
    "final_surface_p95_m",
    "final_hole_fraction",
    "thickness_absolute_error_m",
)
_UPPER_BOUNDED = {
    "depth_rmse_m",
    "depth_p95_m",
    "depth_absolute_bias_m",
    "final_surface_rmse_m",
    "final_surface_p95_m",
    "final_hole_fraction",
    "thickness_absolute_error_m",
}
_UNIT_INTERVAL = {
    "bootstrap_mask_precision",
    "bootstrap_mask_recall",
    "fine_mask_precision",
    "fine_mask_recall",
    "final_hole_fraction",
}
_CHECKS = (
    "traceable_depth_reference_verified",
    "distance_and_incidence_envelope_verified",
    "front_back_and_both_fins_annotated",
    "bootstrap_and_reference_masks_reviewed",
    "final_mesh_holes_and_thickness_reviewed",
    "raw_acceptance_assets_archived",
    "independent_result_review_completed",
)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _source_record(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"geometry-science contract source is missing ({label}): {resolved}")
    return {
        "label": label,
        "sha256": _sha256_path(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _relative_file_record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"runtime source is outside its declared tree: {resolved}")
    return {
        "relative_path": resolved.relative_to(resolved_root).as_posix(),
        "sha256": _sha256_path(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _python_source_tree(root: Path, *, label: str) -> dict[str, Any]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} source tree is missing: {resolved}")
    files = tuple(
        sorted(
            (item for item in resolved.rglob("*.py") if item.is_file()),
            key=lambda item: item.relative_to(resolved).as_posix(),
        )
    )
    if not files:
        raise ValueError(f"{label} source tree contains no Python sources")
    return {
        "label": label,
        "files": [_relative_file_record(item, root=resolved) for item in files],
    }


def _canonical_distribution_name(value: str) -> str:
    name = re.sub(r"[-_.]+", "-", value).lower()
    if name != value or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
        raise ValueError(f"Python distribution name is not canonical: {value!r}")
    return name


def _installed_distributions() -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("installed Python distribution has no Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name.strip()).lower()
        version = str(distribution.version).strip()
        if not name or not version:
            raise ValueError("installed Python distribution identity is incomplete")
        previous = records.get(name)
        if previous is not None and previous["version"] != version:
            raise ValueError(f"ambiguous installed Python distribution version: {name}")
        records[name] = {"name": name, "version": version}
    return [records[key] for key in sorted(records)]


def _nvidia_driver_identity() -> dict[str, Any]:
    candidates = (
        ("sys_module_nvidia_version", Path("/sys/module/nvidia/version")),
        ("proc_nvidia_driver_version", Path("/proc/driver/nvidia/version")),
    )
    for source, path in candidates:
        try:
            content = path.read_bytes()
        except (OSError, PermissionError):
            continue
        text = content.decode("utf-8", errors="replace").strip()
        match = re.search(r"(?:Kernel Module\s+|NVRM version:\s+NVIDIA[^\d]*)([0-9][\w.-]*)", text)
        if match is None:
            match = re.search(r"\b[0-9]+(?:\.[0-9]+){1,3}\b", text)
        return {
            "readable": True,
            "source": source,
            "version": match.group(1) if match else None,
            "content_sha256": _sha256_bytes(content),
        }
    return {
        "readable": False,
        "source": None,
        "version": None,
        "content_sha256": None,
    }


def _torch_runtime() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "importable": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "devices": [],
        "probe_error_type": None,
    }
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        devices = []
        if available:
            for index in range(int(torch.cuda.device_count())):
                properties = torch.cuda.get_device_properties(index)
                capability = torch.cuda.get_device_capability(index)
                devices.append(
                    {
                        "index": index,
                        "name": str(torch.cuda.get_device_name(index)),
                        "capability": [int(capability[0]), int(capability[1])],
                        "total_memory_bytes": int(properties.total_memory),
                        "multiprocessor_count": int(properties.multi_processor_count),
                    }
                )
        runtime = {
            "importable": True,
            "torch_version": str(torch.__version__),
            "cuda_available": available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": devices,
            "probe_error_type": None,
        }
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        runtime["probe_error_type"] = type(exc).__name__
    return runtime


def _runtime_environment() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "os_name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "kernel_version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "libc": {"name": libc_name or None, "version": libc_version or None},
        },
        "python_distributions": _installed_distributions(),
        "torch_runtime": _torch_runtime(),
        "nvidia_driver": _nvidia_driver_identity(),
        "visibility": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
    }


def _strict_json_payload(
    content: bytes,
    *,
    label: str,
    require_canonical: bool = False,
) -> dict[str, Any]:
    def object_from_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    payload = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("geometry-science acceptance metadata must be an object")
    if require_canonical and content != _canonical_json(payload) + b"\n":
        raise ValueError(f"JSON evidence is not canonical: {label}")
    return payload


def _strict_load(path: Path, *, require_canonical: bool = False) -> dict[str, Any]:
    return _strict_json_payload(
        path.read_bytes(),
        label=path.name,
        require_canonical=require_canonical,
    )


def _without_navigation_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_navigation_paths(item)
            for key, item in value.items()
            if not (
                str(key) != "relative_path"
                and (
                    str(key) == "path"
                    or str(key).endswith("_path")
                    or str(key).endswith("_directory")
                    or str(key).endswith("_root")
                )
            )
        }
    if isinstance(value, list):
        return [_without_navigation_paths(item) for item in value]
    return value


def _reject_absolute_identity_strings(value: Any, *, label: str = "runtime contract") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_identity_strings(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_identity_strings(item, label=f"{label}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"absolute navigation path is forbidden in {label}")


def _validate_runtime_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema",
        "sources",
        "source_trees",
        "project_runtime_files",
        "policies",
        "runtime_environment",
    }
    if set(payload) != expected or payload.get("schema") != _RUNTIME_SCHEMA:
        raise ValueError("geometry-science runtime contract schema is invalid")
    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("runtime contract sources must be a non-empty list")
    source_labels: list[str] = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"label", "sha256", "size_bytes"}:
            raise ValueError("runtime source record is invalid")
        label = str(source["label"])
        if not isinstance(source["label"], str) or not label:
            raise ValueError("runtime source label must be a non-empty string")
        source_labels.append(label)
        _digest(source["sha256"], label=f"runtime source {label}")
        if (
            isinstance(source["size_bytes"], bool)
            or not isinstance(source["size_bytes"], int)
            or source["size_bytes"] <= 0
        ):
            raise ValueError("runtime source size must be positive")
    if source_labels != sorted(source_labels) or len(source_labels) != len(set(source_labels)):
        raise ValueError("runtime sources must be uniquely sorted by logical label")

    files = payload["project_runtime_files"]
    if not isinstance(files, list) or [item.get("relative_path") for item in files] != [
        "pyproject.toml",
        "uv.lock",
    ]:
        raise ValueError("runtime contract must bind pyproject.toml and uv.lock")
    for item in files:
        if set(item) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError("project runtime file record is invalid")
        _digest(item["sha256"], label=str(item["relative_path"]))
        if (
            isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
        ):
            raise ValueError("project runtime file size must be positive")

    trees = payload["source_trees"]
    if not isinstance(trees, list) or not trees:
        raise ValueError("runtime source trees must be a non-empty list")
    tree_labels = [str(tree.get("label")) for tree in trees if isinstance(tree, dict)]
    if len(tree_labels) != len(trees) or tree_labels != sorted(tree_labels):
        raise ValueError("runtime source trees must be sorted by label")
    for tree in trees:
        if set(tree) != {"label", "files"} or not isinstance(tree["files"], list):
            raise ValueError("runtime source tree record is invalid")
        relative_paths = [item.get("relative_path") for item in tree["files"]]
        if relative_paths != sorted(relative_paths) or len(relative_paths) != len(
            set(relative_paths)
        ):
            raise ValueError("runtime source tree files must be uniquely sorted")
        for item in tree["files"]:
            if set(item) != {"relative_path", "sha256", "size_bytes"}:
                raise ValueError("runtime source-tree file record is invalid")
            relative = Path(str(item["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("runtime source-tree paths must be safe and relative")
            _digest(item["sha256"], label=str(relative))
            if (
                isinstance(item["size_bytes"], bool)
                or not isinstance(item["size_bytes"], int)
                or item["size_bytes"] <= 0
            ):
                raise ValueError("runtime source-tree file size must be positive")

    environment = payload["runtime_environment"]
    environment_keys = {
        "python",
        "platform",
        "python_distributions",
        "torch_runtime",
        "nvidia_driver",
        "visibility",
    }
    if not isinstance(environment, dict) or set(environment) != environment_keys:
        raise ValueError("runtime environment identity is incomplete")
    python_identity = environment["python"]
    if not isinstance(python_identity, dict) or set(python_identity) != {
        "implementation",
        "version",
        "compiler",
        "cache_tag",
    }:
        raise ValueError("runtime Python interpreter identity is incomplete")
    platform_identity = environment["platform"]
    if not isinstance(platform_identity, dict) or set(platform_identity) != {
        "os_name",
        "system",
        "release",
        "kernel_version",
        "machine",
        "architecture",
        "libc",
    }:
        raise ValueError("runtime OS/platform identity is incomplete")
    libc_identity = platform_identity["libc"]
    if not isinstance(libc_identity, dict) or set(libc_identity) != {"name", "version"}:
        raise ValueError("runtime libc identity is incomplete")
    distributions = environment["python_distributions"]
    if not isinstance(distributions, list):
        raise ValueError("runtime Python distributions must be a list")
    identities: list[tuple[str, str]] = []
    for distribution in distributions:
        if not isinstance(distribution, dict) or set(distribution) != {"name", "version"}:
            raise ValueError("runtime Python distribution record is invalid")
        name = _canonical_distribution_name(str(distribution["name"]))
        version = distribution["version"]
        if not isinstance(version, str) or not version:
            raise ValueError("runtime Python distribution version is empty")
        identities.append((name, version))
    names = [name for name, _version in identities]
    if identities != sorted(identities) or len(names) != len(set(names)):
        raise ValueError("runtime Python distributions must be uniquely sorted")
    torch_runtime = environment["torch_runtime"]
    if not isinstance(torch_runtime, dict) or set(torch_runtime) != {
        "importable",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "devices",
        "probe_error_type",
    }:
        raise ValueError("runtime Torch/CUDA identity is incomplete")
    if not isinstance(torch_runtime["devices"], list):
        raise ValueError("runtime GPU device identity must be a list")
    for index, device in enumerate(torch_runtime["devices"]):
        if not isinstance(device, dict) or set(device) != {
            "index",
            "name",
            "capability",
            "total_memory_bytes",
            "multiprocessor_count",
        }:
            raise ValueError("runtime GPU device identity is invalid")
        if device["index"] != index:
            raise ValueError("runtime GPU devices must be sorted by index")
    driver = environment["nvidia_driver"]
    if not isinstance(driver, dict) or set(driver) != {
        "readable",
        "source",
        "version",
        "content_sha256",
    }:
        raise ValueError("runtime NVIDIA driver identity is incomplete")
    if driver["readable"] is True:
        _digest(driver["content_sha256"], label="NVIDIA driver content")
    elif driver != {
        "readable": False,
        "source": None,
        "version": None,
        "content_sha256": None,
    }:
        raise ValueError("unreadable NVIDIA driver identity is invalid")
    visibility = environment["visibility"]
    if not isinstance(visibility, dict) or set(visibility) != {
        "cuda_visible_devices",
        "nvidia_visible_devices",
    }:
        raise ValueError("runtime GPU visibility identity is incomplete")
    if not isinstance(payload["policies"], dict):
        raise ValueError("runtime policies must be an object")
    normalized = json.loads(_canonical_json(payload))
    if _without_navigation_paths(normalized) != normalized:
        raise ValueError("runtime identity must not contain navigation paths")
    _reject_absolute_identity_strings(normalized)
    return normalized


def science_runtime_contract_payload(settings: Any) -> dict[str, Any]:
    """Return the exact deployable science contract and all executable sources."""

    stereo = settings.foundation_stereo
    model_config = (
        Path(stereo.model_config_path)
        if stereo.model_config_path is not None
        else Path(stereo.checkpoint_path).parent / "cfg.yaml"
    )
    calibration = settings.realsense.stereo_calibration_path
    hand_eye = settings.hand_eye.calibration_path
    if calibration is None or hand_eye is None:
        raise ValueError("science acceptance requires active stereo and hand-eye assets")
    sources = [
        _source_record(
            Path(stereo.repository_path) / "core" / "foundation_stereo.py",
            label="foundation_stereo_source",
        ),
        _source_record(stereo.checkpoint_path, label="foundation_stereo_checkpoint"),
        _source_record(model_config, label="foundation_stereo_model_config"),
        _source_record(calibration, label="stereo_calibration"),
        _source_record(hand_eye, label="flange_primary_hand_eye"),
    ]
    for label, candidate in (
        ("kinematics_model", settings.kinematics.model_path),
        ("kinematics_plugin", settings.kinematics.plugin_path),
    ):
        if candidate is not None:
            sources.append(_source_record(candidate, label=label))
    policy_names = (
        "foundation_stereo",
        "stereo_rectification",
        "point_cloud",
        "bootstrap_foreground",
        "blade_foreground",
        "coarse_science",
        "multi_view_fusion",
        "surface_partition",
        "tsdf",
        "surface_quality",
        "fine_finalization",
        "next_view_selection",
        "view_planning",
        "realsense",
        "acquisition",
        "proxy_model",
        "view_filter",
        "coverage",
        "hand_eye",
        "kinematics",
        "occupancy",
    )
    policies = {
        name: getattr(settings, name).model_dump(mode="json")
        for name in policy_names
        if hasattr(settings, name)
    }
    policies = _without_navigation_paths(policies)
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent.parent
    foundation_root = Path(stereo.repository_path).resolve()
    return _validate_runtime_contract(
        {
            "schema": _RUNTIME_SCHEMA,
            "sources": sorted(sources, key=lambda item: str(item["label"])),
            "source_trees": sorted(
                [
                    _python_source_tree(package_root, label="biblade_fusion_python"),
                    _python_source_tree(foundation_root, label="foundation_stereo_python"),
                ],
                key=lambda item: str(item["label"]),
            ),
            "project_runtime_files": [
                _relative_file_record(project_root / "pyproject.toml", root=project_root),
                _relative_file_record(project_root / "uv.lock", root=project_root),
            ],
            "policies": policies,
            "runtime_environment": _runtime_environment(),
        }
    )


def science_runtime_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a validated, path-independent runtime identity."""

    return _sha256_bytes(_canonical_json(_validate_runtime_contract(payload)))


def science_runtime_contract_for_settings(settings: Any) -> str:
    return science_runtime_contract_sha256(science_runtime_contract_payload(settings))


def _metric_map(value: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if set(value) != set(_METRICS):
        raise ValueError(f"{label} must contain exactly the geometry-science metrics")
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], (int, float))
        for name in _METRICS
    ):
        raise ValueError(f"{label} metrics must be JSON numbers")
    result = {name: float(value[name]) for name in _METRICS}
    if not np.isfinite(tuple(result.values())).all() or any(item < 0.0 for item in result.values()):
        raise ValueError(f"{label} metrics must be finite and non-negative")
    if any(result[name] > 1.0 for name in _UNIT_INTERVAL):
        raise ValueError(f"{label} ratio metrics must lie in [0, 1]")
    return result


def _utc_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    canonical = parsed.astimezone(UTC).isoformat()
    if canonical != value:
        raise ValueError(f"{label} must be canonical UTC")
    return canonical


def _generator_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"name", "version"}:
        raise ValueError("evidence generator identity is invalid")
    name = value["name"]
    version = value["version"]
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError("evidence generator name is invalid")
    if not isinstance(version, str) or not version.strip() or version != version.strip():
        raise ValueError("evidence generator version is invalid")
    return {"name": name, "version": version}


def _validate_raw_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "report_type",
        "generator",
        "created_at_utc",
        "assets",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("report_type") != _RAW_MANIFEST_TYPE
    ):
        raise ValueError("raw acceptance asset manifest schema is invalid")
    assets = value["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("raw acceptance asset manifest must list assets")
    normalized_assets: list[dict[str, Any]] = []
    asset_ids: set[str] = set()
    physical_assets: set[tuple[str, int]] = set()
    for asset in assets:
        fields = {"asset_id", "role", "archive_path", "sha256", "size_bytes"}
        if not isinstance(asset, dict) or set(asset) != fields:
            raise ValueError("raw acceptance asset record is invalid")
        asset_id = asset["asset_id"]
        role = asset["role"]
        archive_path = asset["archive_path"]
        if not isinstance(asset_id, str) or not asset_id or asset_id != asset_id.strip():
            raise ValueError("raw acceptance asset_id is invalid")
        if role not in _ASSET_ROLES:
            raise ValueError("raw acceptance asset role is invalid")
        if not isinstance(archive_path, str):
            raise ValueError("raw acceptance archive_path must be a string")
        relative = Path(archive_path)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != archive_path:
            raise ValueError("raw acceptance archive_path must be canonical and relative")
        size = asset["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("raw acceptance asset size must be a positive integer")
        digest = _digest(asset["sha256"], label=f"raw asset {asset_id}")
        if asset_id in asset_ids or (digest, size) in physical_assets:
            raise ValueError("raw acceptance asset records must be unique")
        asset_ids.add(asset_id)
        physical_assets.add((digest, size))
        normalized_assets.append(
            {
                "asset_id": asset_id,
                "role": role,
                "archive_path": archive_path,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    expected_order = sorted(
        normalized_assets,
        key=lambda item: (str(item["role"]), str(item["asset_id"])),
    )
    if normalized_assets != expected_order:
        raise ValueError("raw acceptance assets must be sorted by role and asset_id")
    if {str(item["role"]) for item in normalized_assets} != _ASSET_ROLES:
        raise ValueError("raw manifest must contain depth, annotation and specimen assets")
    return {
        "schema_version": 1,
        "report_type": _RAW_MANIFEST_TYPE,
        "generator": _generator_identity(value["generator"]),
        "created_at_utc": _utc_timestamp(value["created_at_utc"], label="created_at_utc"),
        "assets": normalized_assets,
    }


def _sample_counts(value: Any) -> dict[str, int]:
    expected = {"depth_reference", "annotated_frames", "reconstructed_specimens"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("science sample-count record is invalid")
    if any(isinstance(value[name], bool) or not isinstance(value[name], int) for name in expected):
        raise ValueError("science sample counts must be integers")
    return {name: int(value[name]) for name in sorted(expected)}


def _test_envelope(value: Any) -> dict[str, float]:
    expected = {
        "minimum_distance_m",
        "maximum_distance_m",
        "minimum_incidence_deg",
        "maximum_incidence_deg",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("science test envelope record is invalid")
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], (int, float))
        for name in expected
    ):
        raise ValueError("science test envelope values must be JSON numbers")
    return {name: float(value[name]) for name in sorted(expected)}


def _validate_evaluation_report(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "report_type",
        "generator",
        "created_at_utc",
        "science_runtime_contract_sha256",
        "raw_acceptance_asset_manifest_sha256",
        "measurements",
        "test_envelope",
        "sample_counts",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("report_type") != _EVALUATION_TYPE
    ):
        raise ValueError("geometry science evaluation report schema is invalid")
    return {
        "schema_version": 1,
        "report_type": _EVALUATION_TYPE,
        "generator": _generator_identity(value["generator"]),
        "created_at_utc": _utc_timestamp(value["created_at_utc"], label="created_at_utc"),
        "science_runtime_contract_sha256": _digest(
            value["science_runtime_contract_sha256"],
            label="evaluation science_runtime_contract_sha256",
        ),
        "raw_acceptance_asset_manifest_sha256": _digest(
            value["raw_acceptance_asset_manifest_sha256"],
            label="evaluation raw_acceptance_asset_manifest_sha256",
        ),
        "measurements": _metric_map(value["measurements"], label="evaluation measurements"),
        "test_envelope": _test_envelope(value["test_envelope"]),
        "sample_counts": _sample_counts(value["sample_counts"]),
    }


def _validate_review_report(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "report_type",
        "reviewed_at_utc",
        "reviewer_id",
        "operator_id",
        "decision",
        "notes",
        "science_runtime_contract_sha256",
        "geometry_evaluation_report_sha256",
        "raw_acceptance_asset_manifest_sha256",
        "checklist",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("report_type") != _REVIEW_TYPE
    ):
        raise ValueError("independent science review report schema is invalid")
    reviewer = value["reviewer_id"]
    operator = value["operator_id"]
    notes = value["notes"]
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip():
        raise ValueError("independent reviewer_id is invalid")
    if not isinstance(operator, str) or not operator.strip() or operator != operator.strip():
        raise ValueError("review operator_id is invalid")
    if reviewer == operator:
        raise ValueError("independent reviewer must differ from the operator")
    if value["decision"] != "accepted":
        raise ValueError("independent science review decision must be accepted")
    if not isinstance(notes, str) or not notes.strip() or notes != notes.strip():
        raise ValueError("independent science review notes must be non-empty")
    checklist = value["checklist"]
    if not isinstance(checklist, dict) or set(checklist) != set(_CHECKS):
        raise ValueError("independent science review checklist is invalid")
    if not all(checklist[name] is True for name in _CHECKS):
        raise ValueError("independent science review checklist must pass")
    return {
        "schema_version": 1,
        "report_type": _REVIEW_TYPE,
        "reviewed_at_utc": _utc_timestamp(value["reviewed_at_utc"], label="reviewed_at_utc"),
        "reviewer_id": reviewer,
        "operator_id": operator,
        "decision": "accepted",
        "notes": notes,
        "science_runtime_contract_sha256": _digest(
            value["science_runtime_contract_sha256"],
            label="review science_runtime_contract_sha256",
        ),
        "geometry_evaluation_report_sha256": _digest(
            value["geometry_evaluation_report_sha256"],
            label="review geometry_evaluation_report_sha256",
        ),
        "raw_acceptance_asset_manifest_sha256": _digest(
            value["raw_acceptance_asset_manifest_sha256"],
            label="review raw_acceptance_asset_manifest_sha256",
        ),
        "checklist": {name: True for name in _CHECKS},
    }


@dataclass(frozen=True, slots=True)
class _EvidenceSource:
    payload: dict[str, Any]
    content: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CanonicalScienceEvidence:
    """One newly written, semantic-validated canonical evidence document."""

    path: Path
    kind: str
    sha256: str
    size_bytes: int


def canonicalize_science_evidence(
    *,
    kind: str,
    input_path: str | Path,
    output_path: str | Path,
) -> CanonicalScienceEvidence:
    """Validate and encode one evidence JSON without authorizing motion."""

    validators = {
        "raw-manifest": _validate_raw_manifest,
        "evaluation": _validate_evaluation_report,
        "review": _validate_review_report,
    }
    normalized_kind = str(kind).strip().lower()
    validator = validators.get(normalized_kind)
    if validator is None:
        raise ValueError("science evidence kind must be raw-manifest, evaluation, or review")
    source = Path(input_path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("science evidence input must be a regular file")
    source_content = source.read_bytes()
    payload = _strict_json_payload(source_content, label=source.name)
    normalized = validator(payload)
    content = _canonical_json(normalized) + b"\n"
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with destination.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created and destination.is_file():
            destination.unlink()
        raise
    return CanonicalScienceEvidence(
        path=destination,
        kind=normalized_kind,
        sha256=_sha256_bytes(content),
        size_bytes=len(content),
    )


def _evidence_source(path: str | Path) -> _EvidenceSource:
    source = Path(path).resolve()
    content = source.read_bytes()
    payload = _strict_json_payload(content, label=source.name, require_canonical=True)
    return _EvidenceSource(payload, content, _sha256_bytes(content), len(content))


def load_science_acceptance_declaration(path: str | Path) -> dict[str, Any]:
    """Load a strict declaration; evidence navigation remains outside identity."""

    payload = _strict_load(Path(path).resolve())
    expected = {
        "schema_version",
        "declaration_type",
        "workcell_id",
        "operator_id",
        "accepted_at_utc",
        "limits",
        "measurements",
        "test_envelope",
        "sample_counts",
        "checklist",
        "evidence",
    }
    if (
        set(payload) != expected
        or payload.get("schema_version") != SCIENCE_ACCEPTANCE_DECLARATION_SCHEMA_VERSION
        or payload.get("declaration_type") != _DECLARATION_TYPE
    ):
        raise ValueError("geometry-science declaration schema is invalid")
    evidence = payload["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(_EVIDENCE_PATHS):
        raise ValueError("geometry-science declaration evidence paths are incomplete")
    if any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
        raise ValueError("geometry-science declaration evidence paths are invalid")
    return payload


def _validated_payload(
    *,
    workcell_id: str,
    operator_id: str,
    accepted_at_utc: datetime,
    science_runtime_contract_sha256: str,
    limits: Mapping[str, Any],
    measurements: Mapping[str, Any],
    minimum_test_distance_m: float,
    maximum_test_distance_m: float,
    minimum_test_incidence_deg: float,
    maximum_test_incidence_deg: float,
    depth_reference_sample_count: int,
    annotated_frame_count: int,
    reconstructed_specimen_count: int,
    checklist: Mapping[str, bool],
) -> dict[str, Any]:
    workcell = str(workcell_id).strip()
    operator = str(operator_id).strip()
    if not workcell or not operator:
        raise ValueError("workcell_id and operator_id must be non-empty")
    if accepted_at_utc.tzinfo is None or accepted_at_utc.utcoffset() is None:
        raise ValueError("accepted_at_utc must be timezone-aware")
    accepted_limits = _metric_map(limits, label="limits")
    observed = _metric_map(measurements, label="measurements")
    if any(accepted_limits[name] <= 0.0 for name in _UPPER_BOUNDED):
        raise ValueError("upper-bound acceptance limits must be positive")
    failed = tuple(
        name
        for name in _METRICS
        if (
            observed[name] > accepted_limits[name]
            if name in _UPPER_BOUNDED
            else observed[name] < accepted_limits[name]
        )
    )
    if failed:
        raise ValueError(
            "geometry-science measurements exceed acceptance limits: " + ", ".join(failed)
        )
    distances = (float(minimum_test_distance_m), float(maximum_test_distance_m))
    incidence = (float(minimum_test_incidence_deg), float(maximum_test_incidence_deg))
    if (
        not np.isfinite((*distances, *incidence)).all()
        or distances[0] <= 0.0
        or distances[1] <= distances[0]
        or incidence[0] < 0.0
        or incidence[1] <= incidence[0]
        or incidence[1] > 90.0
    ):
        raise ValueError("acceptance distance/incidence envelope is invalid")
    counts = (
        depth_reference_sample_count,
        annotated_frame_count,
        reconstructed_specimen_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise ValueError("acceptance sample counts must be integers")
    if counts[0] < 100 or counts[1] < 10 or counts[2] < 3:
        raise ValueError("acceptance sample counts are below their evidence floors")
    if set(checklist) != set(_CHECKS) or not all(checklist[name] is True for name in _CHECKS):
        raise ValueError("all geometry-science physical checks must be true")
    return {
        "schema_version": SCIENCE_ACCEPTANCE_SCHEMA_VERSION,
        "asset_type": _ASSET_TYPE,
        "workcell_id": workcell,
        "operator_id": operator,
        "accepted_at_utc": accepted_at_utc.astimezone(UTC).isoformat(),
        "science_runtime_contract_sha256": _digest(
            science_runtime_contract_sha256,
            label="science_runtime_contract_sha256",
        ),
        "limits": accepted_limits,
        "measurements": observed,
        "test_envelope": {
            "minimum_distance_m": distances[0],
            "maximum_distance_m": distances[1],
            "minimum_incidence_deg": incidence[0],
            "maximum_incidence_deg": incidence[1],
        },
        "sample_counts": {
            "depth_reference": counts[0],
            "annotated_frames": counts[1],
            "reconstructed_specimens": counts[2],
        },
        "checklist": {name: True for name in _CHECKS},
        "declaration": _DECLARATION,
        "motion_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class ScienceTestEnvelope:
    """Distance and incidence domain covered by one physical acceptance."""

    minimum_distance_m: float
    maximum_distance_m: float
    minimum_incidence_deg: float
    maximum_incidence_deg: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_distance_m,
            self.maximum_distance_m,
            self.minimum_incidence_deg,
            self.maximum_incidence_deg,
        )
        if (
            not np.isfinite(values).all()
            or self.minimum_distance_m <= 0.0
            or self.maximum_distance_m <= self.minimum_distance_m
            or self.minimum_incidence_deg < 0.0
            or self.maximum_incidence_deg <= self.minimum_incidence_deg
            or self.maximum_incidence_deg > 90.0
        ):
            raise ValueError("geometry-science test envelope is invalid")

    def assert_covers(self, required: ScienceTestEnvelope) -> None:
        if self.minimum_distance_m > required.minimum_distance_m:
            raise ValueError("geometry-science acceptance does not cover the minimum runtime depth")
        if self.maximum_distance_m < required.maximum_distance_m:
            raise ValueError("geometry-science acceptance does not cover the maximum runtime depth")
        if self.minimum_incidence_deg > required.minimum_incidence_deg:
            raise ValueError("geometry-science acceptance does not cover normal-incidence views")
        if self.maximum_incidence_deg < required.maximum_incidence_deg:
            raise ValueError(
                "geometry-science acceptance does not cover the maximum runtime incidence"
            )


def required_science_test_envelope_for_settings(settings: Any) -> ScienceTestEnvelope:
    """Return the full distance/incidence envelope reachable by configured science."""

    retry_tilts = tuple(
        float(item.tilt_deg) for item in settings.next_view_selection.reacquisition_perturbations
    )
    maximum_retry_tilt = max(retry_tilts, default=0.0)
    proxy_incidence = degrees(acos(float(settings.proxy_model.minimum_camera_normal_cosine)))
    filtered_incidence = degrees(acos(float(settings.view_filter.minimum_incidence_cosine)))
    return ScienceTestEnvelope(
        minimum_distance_m=float(settings.point_cloud.minimum_depth_m),
        maximum_distance_m=float(settings.point_cloud.maximum_depth_m),
        minimum_incidence_deg=0.0,
        maximum_incidence_deg=max(
            float(settings.coarse_science.discovery_tilt_deg),
            maximum_retry_tilt,
            proxy_incidence,
            filtered_incidence,
        ),
    )


@dataclass(frozen=True, slots=True)
class StoredScienceAcceptance:
    path: Path
    acceptance_id: str
    workcell_id: str
    operator_id: str
    accepted_at_utc: datetime
    science_runtime_contract_sha256: str
    limits: dict[str, float]
    measurements: dict[str, float]
    test_envelope: ScienceTestEnvelope
    science_runtime_contract: dict[str, Any]
    evidence: dict[str, dict[str, Any]]
    metadata_sha256: str

    def assert_matches(
        self,
        *,
        acceptance_id: str,
        runtime_contract_sha256: str,
        required_test_envelope: ScienceTestEnvelope | None = None,
    ) -> None:
        if self.acceptance_id != _digest(acceptance_id, label="acceptance_id"):
            raise ValueError("geometry-science acceptance ID differs from configuration")
        if self.science_runtime_contract_sha256 != _digest(
            runtime_contract_sha256,
            label="runtime_contract_sha256",
        ):
            raise ValueError("geometry-science runtime contract changed after acceptance")
        if required_test_envelope is not None:
            self.test_envelope.assert_covers(required_test_envelope)


def _validate_evidence_bundle(
    *,
    base: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    raw_manifest: _EvidenceSource,
    evaluation: _EvidenceSource,
    review: _EvidenceSource,
) -> None:
    runtime_sha256 = science_runtime_contract_sha256(runtime_contract)
    raw_payload = _validate_raw_manifest(raw_manifest.payload)
    evaluation_payload = _validate_evaluation_report(evaluation.payload)
    review_payload = _validate_review_report(review.payload)
    if raw_payload != raw_manifest.payload:
        raise ValueError("raw acceptance asset manifest is not semantically canonical")
    if evaluation_payload != evaluation.payload:
        raise ValueError("geometry science evaluation report is not semantically canonical")
    if review_payload != review.payload:
        raise ValueError("independent science review report is not semantically canonical")
    if evaluation_payload["science_runtime_contract_sha256"] != runtime_sha256:
        raise ValueError("evaluation report runtime contract does not match acceptance")
    if evaluation_payload["raw_acceptance_asset_manifest_sha256"] != raw_manifest.sha256:
        raise ValueError("evaluation report raw asset manifest binding is invalid")
    for field in ("measurements", "test_envelope", "sample_counts"):
        if evaluation_payload[field] != base[field]:
            raise ValueError(f"evaluation report {field} differs from declaration")
    if review_payload["operator_id"] != base["operator_id"]:
        raise ValueError("independent review operator differs from acceptance operator")
    bindings = {
        "science_runtime_contract_sha256": runtime_sha256,
        "geometry_evaluation_report_sha256": evaluation.sha256,
        "raw_acceptance_asset_manifest_sha256": raw_manifest.sha256,
    }
    if any(review_payload[name] != digest for name, digest in bindings.items()):
        raise ValueError("independent review evidence bindings are invalid")
    if review_payload["checklist"] != base["checklist"]:
        raise ValueError("independent review checklist differs from declaration")


def _evidence_record(source: _EvidenceSource, *, relative_path: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }


def write_science_acceptance(
    path: str | Path,
    *,
    science_runtime_contract: Mapping[str, Any],
    geometry_evaluation_report_path: str | Path,
    raw_acceptance_asset_manifest_path: str | Path,
    independent_review_report_path: str | Path,
    **values: Any,
) -> StoredScienceAcceptance:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"geometry-science acceptance already exists: {destination}")
    runtime_contract = _validate_runtime_contract(science_runtime_contract)
    runtime_sha256 = science_runtime_contract_sha256(runtime_contract)
    payload = _validated_payload(
        science_runtime_contract_sha256=runtime_sha256,
        **values,
    )
    raw_manifest = _evidence_source(raw_acceptance_asset_manifest_path)
    evaluation = _evidence_source(geometry_evaluation_report_path)
    review = _evidence_source(independent_review_report_path)
    _validate_evidence_bundle(
        base=payload,
        runtime_contract=runtime_contract,
        raw_manifest=raw_manifest,
        evaluation=evaluation,
        review=review,
    )
    evidence_sources = {
        "geometry_evaluation_report": evaluation,
        "raw_acceptance_asset_manifest": raw_manifest,
        "independent_review_report": review,
    }
    payload["science_runtime_contract"] = runtime_contract
    payload["evidence"] = {
        name: _evidence_record(source, relative_path=_EVIDENCE_PATHS[name])
        for name, source in sorted(evidence_sources.items())
    }
    payload["acceptance_id"] = _sha256_bytes(_canonical_json(payload))
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        for name, source in evidence_sources.items():
            evidence_path = temporary / _EVIDENCE_PATHS[name]
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(source.content)
            with evidence_path.open("rb") as stream:
                os.fsync(stream.fileno())
        metadata = temporary / "metadata.json"
        metadata.write_bytes(_canonical_json(payload) + b"\n")
        with metadata.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return read_science_acceptance(destination)


def read_science_acceptance(path: str | Path) -> StoredScienceAcceptance:
    root = Path(path).resolve()
    metadata_path = root / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("geometry-science acceptance metadata must be a regular file")
    metadata_content = metadata_path.read_bytes()
    payload = _strict_json_payload(
        metadata_content,
        label=metadata_path.name,
        require_canonical=True,
    )
    expected = {
        "schema_version",
        "asset_type",
        "workcell_id",
        "operator_id",
        "accepted_at_utc",
        "science_runtime_contract_sha256",
        "science_runtime_contract",
        "limits",
        "measurements",
        "test_envelope",
        "sample_counts",
        "checklist",
        "evidence",
        "declaration",
        "motion_authorized",
        "acceptance_id",
    }
    if set(payload) != expected:
        raise ValueError("geometry-science acceptance fields differ from schema")
    acceptance_id = _digest(payload.pop("acceptance_id"), label="acceptance_id")
    if acceptance_id != _sha256_bytes(_canonical_json(payload)):
        raise ValueError("geometry-science acceptance identity mismatch")
    if (
        payload["schema_version"] != SCIENCE_ACCEPTANCE_SCHEMA_VERSION
        or payload["asset_type"] != _ASSET_TYPE
        or payload["declaration"] != _DECLARATION
        or payload["motion_authorized"] is not False
    ):
        raise ValueError("geometry-science acceptance schema contract is invalid")
    runtime_contract = _validate_runtime_contract(payload["science_runtime_contract"])
    if runtime_contract != payload["science_runtime_contract"]:
        raise ValueError("geometry-science runtime contract is not canonical")
    runtime_sha256 = science_runtime_contract_sha256(runtime_contract)
    if runtime_sha256 != payload["science_runtime_contract_sha256"]:
        raise ValueError("geometry-science runtime contract identity mismatch")
    envelope = payload["test_envelope"]
    counts = payload["sample_counts"]
    if not isinstance(envelope, dict) or not isinstance(counts, dict):
        raise ValueError("geometry-science envelope/sample records must be objects")
    reproduced = _validated_payload(
        workcell_id=str(payload["workcell_id"]),
        operator_id=str(payload["operator_id"]),
        accepted_at_utc=datetime.fromisoformat(str(payload["accepted_at_utc"])),
        science_runtime_contract_sha256=str(payload["science_runtime_contract_sha256"]),
        limits=dict(payload["limits"]),
        measurements=dict(payload["measurements"]),
        minimum_test_distance_m=float(envelope["minimum_distance_m"]),
        maximum_test_distance_m=float(envelope["maximum_distance_m"]),
        minimum_test_incidence_deg=float(envelope["minimum_incidence_deg"]),
        maximum_test_incidence_deg=float(envelope["maximum_incidence_deg"]),
        depth_reference_sample_count=int(counts["depth_reference"]),
        annotated_frame_count=int(counts["annotated_frames"]),
        reconstructed_specimen_count=int(counts["reconstructed_specimens"]),
        checklist=dict(payload["checklist"]),
    )
    base_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"science_runtime_contract", "evidence"}
    }
    if reproduced != base_payload:
        raise ValueError("geometry-science acceptance does not reproduce canonically")
    evidence_records = payload["evidence"]
    if not isinstance(evidence_records, dict) or set(evidence_records) != set(_EVIDENCE_PATHS):
        raise ValueError("geometry-science evidence records are incomplete")
    loaded_evidence: dict[str, _EvidenceSource] = {}
    for name, expected_path in _EVIDENCE_PATHS.items():
        record = evidence_records[name]
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("geometry-science evidence record is invalid")
        if record["relative_path"] != expected_path:
            raise ValueError("geometry-science evidence location is not canonical")
        evidence_path = root / expected_path
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise ValueError("geometry-science evidence must be a sealed regular file")
        source = _evidence_source(evidence_path)
        if source.sha256 != _digest(record["sha256"], label=f"{name} evidence"):
            raise ValueError("geometry-science evidence SHA-256 mismatch")
        if (
            isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] != source.size_bytes
        ):
            raise ValueError("geometry-science evidence size mismatch")
        loaded_evidence[name] = source
    _validate_evidence_bundle(
        base=reproduced,
        runtime_contract=runtime_contract,
        raw_manifest=loaded_evidence["raw_acceptance_asset_manifest"],
        evaluation=loaded_evidence["geometry_evaluation_report"],
        review=loaded_evidence["independent_review_report"],
    )
    return StoredScienceAcceptance(
        root,
        acceptance_id,
        str(payload["workcell_id"]),
        str(payload["operator_id"]),
        datetime.fromisoformat(str(payload["accepted_at_utc"])),
        str(payload["science_runtime_contract_sha256"]),
        dict(payload["limits"]),
        dict(payload["measurements"]),
        ScienceTestEnvelope(
            minimum_distance_m=float(envelope["minimum_distance_m"]),
            maximum_distance_m=float(envelope["maximum_distance_m"]),
            minimum_incidence_deg=float(envelope["minimum_incidence_deg"]),
            maximum_incidence_deg=float(envelope["maximum_incidence_deg"]),
        ),
        runtime_contract,
        dict(evidence_records),
        _sha256_bytes(metadata_content),
    )


__all__ = [
    "CanonicalScienceEvidence",
    "SCIENCE_ACCEPTANCE_DECLARATION_SCHEMA_VERSION",
    "SCIENCE_ACCEPTANCE_SCHEMA_VERSION",
    "ScienceTestEnvelope",
    "StoredScienceAcceptance",
    "canonicalize_science_evidence",
    "load_science_acceptance_declaration",
    "read_science_acceptance",
    "required_science_test_envelope_for_settings",
    "science_runtime_contract_for_settings",
    "science_runtime_contract_payload",
    "science_runtime_contract_sha256",
    "write_science_acceptance",
]
