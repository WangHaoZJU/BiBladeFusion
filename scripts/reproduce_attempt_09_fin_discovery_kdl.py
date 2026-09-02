#!/usr/bin/env python3
"""Reproduce the read-only attempt-09 paired fin-discovery KDL diagnostic.

This script never connects to the robot and never writes an artifact.  It loads the
persisted first-view proxy, the copied controller MDH and hand-eye assets, then calls
the native Elite KDL plugin through ``EliteCs68IkChecker`` for every generated endpoint.
The checked-in JSON manifest supplies the proposed (not enabled) paired candidates and
the expected output used for an independent comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from biblade_fusion.calibration import (
    load_cs68_kinematics,
    load_hand_eye_calibration,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    HandEyeConfig,
    KinematicsConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import EliteCs68IkChecker
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarseSciencePolicy,
    generate_fin_discovery_plan,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTEMPT = (
    REPOSITORY_ROOT
    / "data/experiments/blade-placement-20260901-01-attempt-09"
)
DEFAULT_MANIFEST = REPOSITORY_ROOT / "docs/attempt-09-fin-discovery-kdl-dry-run.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _assert_source_hash(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Source hash changed for {path}: expected {expected_sha256}, got {actual}"
        )


def _attempt_proxy(metadata: dict[str, Any]) -> BilateralBladeProxy:
    proxy = metadata["proxy"]
    return BilateralBladeProxy(
        frame_T_proxy=PoseSE3("base", "blade_proxy", proxy["base_T_proxy"]),
        extents_m=proxy["extents_m"],
        observed_surface_centroid_m=proxy["observed_surface_centroid_m"],
        pca_eigenvalues_m2=proxy["pca_eigenvalues_m2"],
        raw_point_count=int(proxy["raw_point_count"]),
        finite_point_count=int(proxy["finite_point_count"]),
        voxel_point_count=int(proxy["voxel_point_count"]),
        camera_normal_cosine=float(proxy["camera_normal_cosine"]),
    )


def _native_paths(kinematics: KinematicsConfig) -> tuple[Any, Path, Path]:
    native = import_module("elite_cs_sdk.elite_cs_sdk_python")
    binding_path = Path(native.__file__).resolve()
    plugin_path = (
        kinematics.plugin_path.resolve()
        if kinematics.plugin_path is not None
        else binding_path.parent / "libelite_kdl_kinematics.so"
    )
    if not plugin_path.is_file():
        raise RuntimeError(f"Native Elite KDL plugin is missing: {plugin_path}")
    return native, binding_path, plugin_path


def _rounded_vector(value: np.ndarray) -> list[float]:
    return np.round(np.asarray(value, dtype=np.float64), 6).tolist()


def _computed_results(
    *,
    attempt: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    initialization_path = attempt / "coarse_science/initialization/metadata.json"
    view_plan_path = attempt / "coarse_science/proxy_view_plan/view_plan.json"
    initialization = _load_json(initialization_path)
    view_plan = _load_json(view_plan_path)

    sources = manifest["sources"]
    _assert_source_hash(
        initialization_path,
        str(sources["attempt_09_initialization_metadata"]["sha256"]),
    )

    kinematics_record = sources["kinematics"]
    kinematics_path = REPOSITORY_ROOT / str(kinematics_record["path"])
    _assert_source_hash(kinematics_path, str(kinematics_record["sha256"]))

    hand_eye_record = sources["hand_eye"]
    hand_eye_path = REPOSITORY_ROOT / str(hand_eye_record["path"])
    _assert_source_hash(hand_eye_path, str(hand_eye_record["sha256"]))

    flange_tcp_record = sources["flange_tcp"]
    flange_tcp_path = REPOSITORY_ROOT / str(flange_tcp_record["path"])
    _assert_source_hash(flange_tcp_path, str(flange_tcp_record["sha256"]))

    processing = initialization["processing"]
    diagnostic_seed = np.asarray(
        manifest["diagnostic_seed_joint_positions_rad"],
        dtype=np.float64,
    )
    if not np.array_equal(
        diagnostic_seed,
        np.asarray(initialization["seed_joint_positions_rad"], dtype=np.float64),
    ):
        raise RuntimeError("Diagnostic IK seed differs from attempt-09 initialization")
    kinematics_config = KinematicsConfig.model_validate(processing["kinematics"])
    hand_eye_config = HandEyeConfig.model_validate(processing["hand_eye_gate"])
    hand_eye = load_hand_eye_calibration(
        hand_eye_config.model_copy(update={"calibration_path": hand_eye_path})
    )
    native, binding_path, plugin_path = _native_paths(kinematics_config)
    checker = EliteCs68IkChecker(
        load_cs68_kinematics(kinematics_path),
        hand_eye,
        diagnostic_seed,
        kinematics_config.model_copy(update={"plugin_path": plugin_path}),
        native_module=native,
    )

    plan_configuration = dict(view_plan["configuration"]["view_planning"])
    plan_configuration["paired_fin_discovery_fallbacks"] = manifest[
        "in_memory_candidate_configuration"
    ]
    planning = ViewPlanningConfig.model_validate(plan_configuration)
    filtering = ViewFilterConfig.model_validate(
        view_plan["configuration"]["view_filter"]
    )
    policy = CoarseSciencePolicy(**manifest["coarse_science_policy"])
    result = generate_fin_discovery_plan(
        _attempt_proxy(initialization),
        tuple(float(value) for value in view_plan["grid"]["footprint_m"]),
        planning,
        filtering,
        policy,
        checker,
    )

    workspace = filtering.workspace
    if workspace is None:
        raise RuntimeError("The diagnostic requires a measured camera workspace")
    lower = np.asarray(workspace.minimum_m) + filtering.camera_clearance_radius_m
    upper = np.asarray(workspace.maximum_m) - filtering.camera_clearance_radius_m
    endpoint_results = []
    evaluations = []
    for item in result.filtered.candidates:
        position = item.candidate.base_t_left_ir.translation_m
        evaluation = {
            "view_id": item.candidate.view_id,
            "status": item.status.value,
            "reasons": list(item.reasons),
            "camera_position_m": _rounded_vector(position),
            "joint_positions_rad": (
                _rounded_vector(item.joint_positions_rad)
                if item.joint_positions_rad is not None
                else None
            ),
        }
        evaluations.append(evaluation)
        if item.status.value == "endpoint_feasible":
            endpoint_results.append(
                {
                    **evaluation,
                    "workspace_margin_m": _rounded_vector(
                        np.minimum(position - lower, upper - position)
                    ),
                }
            )

    rejection_reasons = Counter(
        reason for evaluation in evaluations for reason in evaluation["reasons"]
    )
    return {
        "native_execution": {
            "python_binding_path": _repository_relative(binding_path),
            "python_binding_sha256": _sha256(binding_path),
            "plugin_path": _repository_relative(plugin_path),
            "plugin_sha256": _sha256(plugin_path),
            "class_loader_created": getattr(checker, "_loader", None) is not None,
        },
        "policy_sha256": result.policy_sha256,
        "evaluated_candidate_count": len(evaluations),
        "endpoint_feasible_count": len(endpoint_results),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "results": endpoint_results,
        "evaluations": evaluations,
    }


def _assert_expected(manifest: dict[str, Any], computed: dict[str, Any]) -> None:
    expected_native = manifest["checker"]["native_execution"]
    if computed["native_execution"] != expected_native:
        raise RuntimeError(
            "Native Elite binding/plugin provenance differs from the checked-in manifest"
        )
    if computed["evaluated_candidate_count"] != manifest["evaluated_candidate_count"]:
        raise RuntimeError("Evaluated candidate count differs from the manifest")
    if computed["endpoint_feasible_count"] != manifest["endpoint_feasible_count"]:
        raise RuntimeError("Endpoint-feasible candidate count differs from the manifest")
    if computed["policy_sha256"] != manifest["policy_sha256"]:
        raise RuntimeError("Generated fin-discovery policy hash differs from the manifest")
    if computed["rejection_reason_counts"] != manifest["rejection_reason_counts"]:
        raise RuntimeError("Endpoint rejection diagnostics differ from the manifest")
    expected = {item["view_id"]: item for item in manifest["results"]}
    actual = {item["view_id"]: item for item in computed["results"]}
    if set(actual) != set(expected):
        raise RuntimeError("Endpoint-feasible candidate identities differ from the manifest")
    for view_id, expected_item in expected.items():
        actual_item = actual[view_id]
        if actual_item["status"] != expected_item["status"]:
            raise RuntimeError(f"Endpoint status changed for {view_id}")
        for field in (
            "camera_position_m",
            "workspace_margin_m",
            "joint_positions_rad",
        ):
            if not np.allclose(
                actual_item[field],
                expected_item[field],
                rtol=0.0,
                atol=1e-6,
            ):
                raise RuntimeError(f"{field} changed for {view_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--show-evaluations",
        action="store_true",
        help="Include all rejected baseline endpoints in stdout JSON.",
    )
    args = parser.parse_args()
    manifest = _load_json(args.manifest.resolve())
    computed = _computed_results(
        attempt=args.attempt.resolve(),
        manifest=manifest,
    )
    _assert_expected(manifest, computed)
    if not args.show_evaluations:
        computed.pop("evaluations")
    print(json.dumps(computed, indent=2, ensure_ascii=False, allow_nan=False))
    print("PASS: checked-in attempt-09 endpoint results reproduced with native Elite KDL")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
