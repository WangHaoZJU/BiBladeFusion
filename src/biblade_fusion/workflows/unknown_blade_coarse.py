"""Online coarse-science composition for one unknown bilateral finned blade.

This module is deliberately motion-free.  It prepares one stopped scientific view,
appends it to an immutable proxy-coverage generation, selects only endpoint-feasible
coarse targets, and promotes a generation to a schema-5 reference only after both
blade sides and both faces of the single fin on each side have evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import atan2, cos, degrees, radians, sin, sqrt
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AppSettings,
    PairedFinDiscoveryFallbackConfig,
    PointCloudConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.diagnostics.performance_timing import (
    PerformanceTimingRecorder,
    activate_performance_timing,
    performance_span,
    try_create_performance_timing,
)
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapForegroundResult,
    BootstrapSeed,
    array_content_sha256,
    bootstrap_blade_foreground,
    bootstrap_seed_payload,
)
from biblade_fusion.perception.coarse_foreground import (
    ProjectedCoarseForegroundGuide,
    ProjectedCoarseForegroundResult,
    projected_coarse_blade_foreground,
)
from biblade_fusion.perception.proxy import (
    BilateralBladeProxy,
    build_bilateral_proxy,
)
from biblade_fusion.planning import (
    AdaptiveViewSearchConfig,
    AdaptiveViewSearchResult,
    BladeSide,
    CandidateStatus,
    CandidateView,
    CoverageLedger,
    EndpointCollisionAwareReachabilityChecker,
    EndpointConfigurationValidator,
    EvaluatedCandidate,
    FilteredViewPlan,
    ReachabilityChecker,
    SurfacePatch,
    adaptive_view_search_payload,
    coverage_observation_id,
    create_coverage_ledger,
    filter_candidate_views,
    search_adaptive_candidate_family,
    select_uncovered_candidates,
    update_coverage,
)
from biblade_fusion.planning.coarse_discovery_gain import (
    CoarseDiscoveryGain,
    expected_coarse_discovery_gain,
)
from biblade_fusion.storage.coarse_model import (
    read_coarse_model_summary,
    write_coarse_model,
)
from biblade_fusion.storage.coarse_scan import (
    CoarseForegroundResult,
    CoarseTargetKind,
    StoredCoarseScanGeneration,
    StoredCoarseScanView,
    _bind_coarse_scan_view_readback,
    _CoarseScanViewReadback,
    _write_coarse_scan_generation_from_verified,
    read_coarse_integration_source,
    read_coarse_scan_generation,
    read_coarse_scan_view,
    write_coarse_scan_generation,
    write_coarse_scan_view,
)
from biblade_fusion.storage.coverage import read_coverage_ledger, write_coverage_ledger
from biblade_fusion.storage.initialization import (
    INITIALIZATION_METADATA_FILENAME,
    read_initialization,
    write_initialization,
)
from biblade_fusion.storage.reconstructed_view import write_reconstructed_view
from biblade_fusion.storage.stereo_inference import read_stereo_inference
from biblade_fusion.storage.view_plan import read_view_plan, write_view_plan
from biblade_fusion.workflows.coarse_model import build_coarse_blade_model
from biblade_fusion.workflows.initialization import InitialObservation
from biblade_fusion.workflows.occupancy_mapping import (
    OccupancyFrameUpdate,
    PreparedOccupancyFrame,
    occupancy_array_content_hash,
)
from biblade_fusion.workflows.reconstruction import (
    ReconstructedBladeView,
    reconstruct_foundation_stereo_view,
)
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    NextViewSelection,
    PerceptionCycleResult,
    RankedNextViewCandidate,
    next_view_target_from_candidate,
)
from biblade_fusion.workflows.view_planning import plan_initial_observation


class UnknownBladeCoarseError(RuntimeError):
    """The coarse phase cannot safely prepare, recover, select, or promote."""


BootstrapSeedProvider = Callable[[CapturedStopScanView, Path], BootstrapSeed]


class CoarsePhase(StrEnum):
    COLLECTING = "collecting"
    COLLECTING_FIN_EVIDENCE = "collecting_fin_evidence"
    READY_FOR_FINE = "ready_for_fine"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class CoarseSciencePolicy:
    """Explicit completion and conservative fin-discovery policy."""

    discovery_tilt_deg: float = 15.0
    discovery_tilt_samples_deg: tuple[float, ...] = (
        10.0,
        20.0,
        30.0,
        45.0,
        60.0,
    )
    discovery_gain_surface_weight: float = 0.45
    discovery_gain_side_balance_weight: float = 0.25
    discovery_gain_fin_pair_weight: float = 0.30
    discovery_gain_fin_seed_value: float = 0.60
    discovery_gain_minimum: float = 0.0
    minimum_total_views: int = 6
    minimum_views_per_side: int = 3
    maximum_attempts_per_candidate: int = 2
    require_complete_proxy_coverage: bool = True
    maximum_discovery_translation_error_m: float = 0.020
    maximum_discovery_rotation_error_deg: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.discovery_tilt_deg < 75.0:
            raise ValueError("Initial coarse discovery tilt must lie in (0, 75) degrees")
        samples = tuple(float(value) for value in self.discovery_tilt_samples_deg)
        if (
            not samples
            or not np.isfinite(samples).all()
            or any(not 0.0 < value < 75.0 for value in samples)
            or len(set(samples)) != len(samples)
        ):
            raise ValueError(
                "Coarse discovery tilt samples must be unique finite values in (0, 75)"
            )
        object.__setattr__(self, "discovery_tilt_samples_deg", samples)
        gain_weights = (
            self.discovery_gain_surface_weight,
            self.discovery_gain_side_balance_weight,
            self.discovery_gain_fin_pair_weight,
        )
        if (
            not np.isfinite((*gain_weights, self.discovery_gain_fin_seed_value)).all()
            or any(not 0.0 <= value <= 1.0 for value in gain_weights)
            or not np.isclose(sum(gain_weights), 1.0, rtol=0.0, atol=1e-9)
            or not 0.0 <= self.discovery_gain_fin_seed_value <= 1.0
            or not np.isfinite(self.discovery_gain_minimum)
            or self.discovery_gain_minimum < 0.0
        ):
            raise ValueError("Coarse discovery gain policy is invalid")
        if self.minimum_total_views < 4 or self.minimum_views_per_side < 2:
            raise ValueError("Coarse view gates must include both sides and paired obliques")
        if self.minimum_total_views < 2 * self.minimum_views_per_side:
            raise ValueError("Total coarse-view gate is below the per-side requirement")
        if self.maximum_attempts_per_candidate < 1:
            raise ValueError("Coarse candidate attempt limit must be positive")
        if self.maximum_discovery_translation_error_m <= 0.0:
            raise ValueError("Coarse discovery translation tolerance must be positive")
        if not 0.0 < self.maximum_discovery_rotation_error_deg <= 30.0:
            raise ValueError("Coarse discovery rotation tolerance must lie in (0, 30]")


@dataclass(frozen=True, slots=True)
class AdaptiveFinDiscoverySearch:
    config: AdaptiveViewSearchConfig
    result: AdaptiveViewSearchResult


@dataclass(frozen=True, slots=True)
class CoarseDiscoveryPlan:
    filtered: FilteredViewPlan
    policy_sha256: str
    adaptive_searches: tuple[AdaptiveFinDiscoverySearch, ...] = ()
    current_joint_positions_rad: tuple[float, float, float, float, float, float] | None = None

    @property
    def endpoint_feasible(self) -> tuple[EvaluatedCandidate, ...]:
        return self.filtered.endpoint_feasible

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class PreparedCoarseScienceView:
    coarse_view_path: Path
    reconstructed_view_path: Path
    target_view_id: str
    target_kind: CoarseTargetKind
    target_side: BladeSide

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CoarsePhaseTransition:
    phase: CoarsePhase
    reasons: tuple[str, ...]
    source_generation_path: Path
    ready_generation_path: Path | None = None
    reference_coarse_model_path: Path | None = None

    def __post_init__(self) -> None:
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if not reasons or any(not reason for reason in reasons):
            raise ValueError("Coarse phase transition requires explicit reasons")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "source_generation_path",
            Path(self.source_generation_path).resolve(),
        )
        for name in ("ready_generation_path", "reference_coarse_model_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        if self.phase is CoarsePhase.READY_FOR_FINE and (
            self.ready_generation_path is None or self.reference_coarse_model_path is None
        ):
            raise ValueError("Fine transition requires exact generation and schema-5 paths")

    @property
    def motion_authorized(self) -> bool:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_bootstrap_annotation_request(
    captured: CapturedStopScanView,
    stereo: StereoInferenceObservation,
    stereo_path: Path,
    integration_valid_mask: np.ndarray,
) -> tuple[Path, Path]:
    """Persist the exact formal frame that an operator ROI must annotate."""

    root = captured.cycle_root / "bootstrap_annotation"
    root.mkdir()
    image_path = root / "left_rectified.png"
    image = np.asarray(stereo.rectified.left_ir)
    if not cv2.imwrite(str(image_path), image):
        raise UnknownBladeCoarseError("Cannot write formal bootstrap annotation image")
    request_path = root / "request.json"
    request = {
        "schema_version": 1,
        "artifact_kind": "biblade_fusion.bootstrap_annotation_request",
        "identity": {
            "view_id": captured.bundle.view_id,
            "sequence_index": captured.bundle.sequence_index,
            "frame_number": captured.bundle.stereo.frame_number,
        },
        "sources": {
            "stereo_inference": str(stereo_path.resolve()),
            "stereo_metadata_sha256": _sha256(stereo_path / "metadata.json"),
        },
        "content_sha256": {
            "left_rectified": array_content_sha256(image),
            "integration_valid_mask": array_content_sha256(integration_valid_mask),
            "left_rectified_png": _sha256(image_path),
        },
        "annotation_image": image_path.name,
    }
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return root, image_path


def _write_bootstrap_annotation_response(
    root: Path,
    foreground: BootstrapForegroundResult,
) -> None:
    response_path = root / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "biblade_fusion.bootstrap_annotation_response",
                "seed": bootstrap_seed_payload(foreground.seed),
                "policy_sha256": foreground.policy_sha256,
                "mask_pixel_count": foreground.diagnostics.mask_pixel_count,
                "mask_fraction": foreground.diagnostics.mask_fraction,
                "input_content_sha256": {
                    "left_rectified": foreground.left_image_content_sha256,
                    "depth_m": foreground.depth_content_sha256,
                    "integration_valid_mask": foreground.valid_mask_content_sha256,
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _projected_foreground_from_generation(
    generation_path: str | Path,
    *,
    stereo: StereoInferenceObservation,
    integration_valid_mask: np.ndarray,
    base_t_left_rectified: PoseSE3,
    foreground_config: BootstrapForegroundConfig,
    settings: AppSettings,
) -> ProjectedCoarseForegroundResult:
    """Use only previously accepted blade support to guide a later coarse mask."""

    generation = read_coarse_scan_generation(generation_path)
    reference_points = np.vstack(
        [item.support_cloud.points_m for item in generation.views]
    )
    lower = settings.proxy_model.blade_envelope_min_m
    upper = settings.proxy_model.blade_envelope_max_m
    if lower is None or upper is None:
        raise UnknownBladeCoarseError(
            "Automatic coarse foreground requires a blade-only base-frame envelope"
        )
    guide = ProjectedCoarseForegroundGuide(
        source_generation_path=generation.root,
        source_generation_metadata_sha256=generation.metadata_sha256,
        reference_points_content_sha256=array_content_sha256(reference_points),
        blade_envelope_min_m=lower,
        blade_envelope_max_m=upper,
    )
    try:
        return projected_coarse_blade_foreground(
            stereo.rectified.left_ir,
            stereo.depth_m,
            integration_valid_mask,
            foreground_config,
            intrinsics=stereo.rectified.calibration.left,
            base_t_left_rectified=base_t_left_rectified,
            reference_points_base_m=reference_points,
            guide=guide,
        )
    except ValueError as exc:
        raise UnknownBladeCoarseError(
            "Projected accumulated-blade foreground could not identify this coarse view"
        ) from exc


def _rotation_error_deg(first: PoseSE3, second: PoseSE3) -> float:
    relative = first.rotation.T @ second.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _resolve_operator_bootstrap_side(
    base_t_projection_camera: PoseSE3,
    proxy: BilateralBladeProxy | None,
    explicit_side: BladeSide | None,
) -> BladeSide:
    if explicit_side is not None:
        return explicit_side
    if proxy is None:
        # The proxy's positive normal is defined toward its first observing camera.
        return BladeSide.FRONT
    local_camera = proxy.frame_T_proxy.inverse().transform_points(
        base_t_projection_camera.translation_m
    )
    if abs(float(local_camera[2])) <= 1e-9:
        raise UnknownBladeCoarseError("Operator bootstrap camera lies on the proxy mid-plane")
    return BladeSide.FRONT if local_camera[2] > 0.0 else BladeSide.BACK


def _look_at_pose(
    *,
    view_id: str,
    position_m: np.ndarray,
    target_m: np.ndarray,
    preferred_x: np.ndarray,
) -> PoseSE3:
    camera_z = target_m - position_m
    camera_z /= np.linalg.norm(camera_z)
    camera_x = preferred_x - camera_z * float(preferred_x @ camera_z)
    if np.linalg.norm(camera_x) <= 1e-9:
        raise UnknownBladeCoarseError("Fin-discovery camera basis is singular")
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    return PoseSE3.from_rotation_translation(
        "base",
        f"{view_id}_left_ir",
        np.column_stack((camera_x, camera_y, camera_z)),
        position_m,
    )


_DISCOVERY_VIEW_ID = re.compile(
    r"^(front|back)_fin_discovery_(major|minor)_(negative|positive)"
    r"(?:_adaptive_\d+)?"
    r"(?:_paired_fallback_(\d+))?$"
)


def _discovery_view_identity(
    view_id: str,
) -> tuple[BladeSide, str, str, str] | None:
    match = _DISCOVERY_VIEW_ID.fullmatch(view_id)
    if match is None:
        return None
    side_name, axis, sign_name, fallback_index = match.groups()
    family = "baseline" if fallback_index is None else f"paired_fallback_{fallback_index}"
    return BladeSide(side_name), axis, sign_name, family


def _explicit_fin_discovery_pair(
    *,
    proxy: BilateralBladeProxy,
    axes: np.ndarray,
    entry: PairedFinDiscoveryFallbackConfig,
    entry_index: int,
    baseline_standoff_m: float,
    geometric_footprint_m: tuple[float, float],
) -> tuple[CandidateView, CandidateView]:
    """Materialize exactly one separately configured opposing fallback pair."""

    side = BladeSide(entry.side)
    major, minor, front_normal = axes.T
    signed_axis, common_axis = (major, minor) if entry.axis == "major" else (minor, major)
    normal = front_normal if side is BladeSide.FRONT else -front_normal
    target = proxy.center_m + normal * float(proxy.extents_m[2]) / 2.0
    distance = baseline_standoff_m + entry.distance_offset_m
    total_tilt = radians(entry.total_tilt_deg)
    signed_component = sin(radians(entry.opposing_tilt_deg))
    total_tangent = sin(total_tilt)
    common_component = entry.common_bias_sign * sqrt(
        max(0.0, total_tangent * total_tangent - signed_component * signed_component)
    )
    normal_component = cos(total_tilt)
    footprint_scale = distance / baseline_standoff_m
    footprint = tuple(float(value * footprint_scale) for value in geometric_footprint_m)
    extents = (float(proxy.extents_m[0]), float(proxy.extents_m[1]))
    candidates: list[CandidateView] = []
    for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
        view_id = (
            f"{side.value}_fin_discovery_{entry.axis}_{sign_name}_"
            f"paired_fallback_{entry_index:02d}"
        )
        direction = (
            normal_component * normal
            + sign * signed_component * signed_axis
            + common_component * common_axis
        )
        if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-9):
            raise UnknownBladeCoarseError("Explicit fin-discovery direction is not unit")
        position = target + distance * direction
        patch = SurfacePatch(view_id, side, 0, 0, target, normal, extents)
        candidates.append(
            CandidateView(
                view_id,
                patch,
                _look_at_pose(
                    view_id=view_id,
                    position_m=position,
                    target_m=target,
                    preferred_x=common_axis,
                ),
                distance,
                footprint,
                projection_fraction=normal_component,
                visibility_fraction=normal_component,
                distance_policy="explicit_paired_fin_discovery_fallback_v1",
            )
        )
    return candidates[0], candidates[1]


def _fin_discovery_azimuth_deg(candidate: CandidateView) -> float:
    """Recover the signed nominal oblique direction in the candidate tangent frame."""

    normal = candidate.patch.outward_normal
    direction = candidate.base_t_left_ir.translation_m - candidate.patch.target_m
    direction /= np.linalg.norm(direction)
    tangent = direction - normal * float(direction @ normal)
    tangent /= np.linalg.norm(tangent)
    tangent_x = candidate.base_t_left_ir.rotation[:, 0].copy()
    tangent_x -= normal * float(tangent_x @ normal)
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    return float(degrees(atan2(float(tangent @ tangent_y), float(tangent @ tangent_x))) % 360.0)


def _fin_discovery_azimuth_samples_deg(
    candidate: CandidateView,
    configured_samples_deg: tuple[float, ...],
) -> tuple[float, ...]:
    """Search the signed fin half-plane, including HoloRobot-style common bias."""

    nominal = _fin_discovery_azimuth_deg(candidate)
    configured = tuple(float(value) % 360.0 for value in configured_samples_deg)
    ordered = tuple(sorted(set(configured)))
    midpoints = tuple(
        (first + ((second - first) % 360.0) / 2.0) % 360.0
        for first, second in zip(ordered, (*ordered[1:], ordered[0]), strict=True)
    )
    available = {nominal, *configured, *midpoints}

    def signed_offset(value: float) -> float:
        return (value - nominal + 180.0) % 360.0 - 180.0

    # A strict open half-plane retains the requested negative/positive fin-face
    # identity.  Within it, try the nominal direction first and then the largest
    # common tangential biases: those asymmetric look-at poses are what lets a
    # wrist-limited arm observe opposing fin faces without demanding a symmetric
    # pair of camera positions.
    valid = tuple(
        value for value in available if abs(signed_offset(value)) < 90.0 - 1e-9
    )
    return tuple(
        sorted(
            valid,
            key=lambda value: (
                0 if abs(signed_offset(value)) <= 1e-9 else 1,
                -abs(signed_offset(value)),
                signed_offset(value) > 0.0,
                value,
            ),
        )
    )


def _fin_discovery_search_config(
    candidate: CandidateView,
    planning_config: ViewPlanningConfig,
    point_cloud_config: PointCloudConfig,
    policy: CoarseSciencePolicy,
) -> AdaptiveViewSearchConfig:
    adaptive = planning_config.adaptive_ik_view_search
    tilts = [float(policy.discovery_tilt_deg)]
    tilts.extend(
        float(value)
        for value in policy.discovery_tilt_samples_deg
        if not np.isclose(value, policy.discovery_tilt_deg, rtol=0.0, atol=1e-12)
    )
    # Rank useful incidence first.  sin(t)*cos(t), also used by the science
    # ranking, peaks at 45 degrees; +/-15 degrees remains a seed, not a gate.
    tilts.sort(
        key=lambda value: (
            -(sin(radians(value)) * cos(radians(value))),
            value,
        )
    )
    return AdaptiveViewSearchConfig(
        minimum_optical_distance_m=point_cloud_config.minimum_depth_m,
        maximum_optical_distance_m=point_cloud_config.maximum_depth_m,
        distance_step_m=adaptive.distance_step_m,
        maximum_distance_expansions=adaptive.maximum_distance_expansions,
        tilt_samples_deg=tuple(tilts),
        azimuth_samples_deg=_fin_discovery_azimuth_samples_deg(
            candidate,
            adaptive.azimuth_samples_deg,
        ),
        roll_samples_deg=adaptive.roll_samples_deg,
        maximum_generated_candidates=adaptive.maximum_generated_candidates,
        # Each nominal discovery family contributes one endpoint to the global
        # science ranking; collecting eight alternatives for every one of the
        # eight seed poses only multiplies IK latency.
        maximum_ik_feasible_candidates=min(
            2,
            adaptive.maximum_ik_feasible_candidates,
        ),
        maximum_ik_attempts_per_family=adaptive.maximum_ik_attempts_per_family,
        maximum_search_duration_s=adaptive.maximum_search_duration_s,
        sampling_order="distance_major",
        ranking_mode="fin_discovery",
        require_attempted_per_tilt=False,
    )


def generate_fin_discovery_plan(
    proxy: BilateralBladeProxy,
    geometric_footprint_m: tuple[float, float],
    planning_config: ViewPlanningConfig,
    filter_config: ViewFilterConfig,
    policy: CoarseSciencePolicy,
    reachability_checker: ReachabilityChecker,
    point_cloud_config: PointCloudConfig | None = None,
    current_joint_positions_rad: (
        tuple[float, float, float, float, float, float] | None
    ) = None,
    endpoint_validator: EndpointConfigurationValidator | None = None,
) -> CoarseDiscoveryPlan:
    """Generate paired +/- oblique targets about both unknown fin axes.

    The first pair may reveal both fin faces when the fin extends along the proxy's
    other in-plane axis.  The orthogonal pair is retained as deterministic fallback.
    Every target still passes the ordinary workspace and endpoint IK filter.
    """

    standoff = planning_config.standoff_distance_m
    if standoff is None:
        raise UnknownBladeCoarseError("Fin discovery requires a configured standoff")
    axes = np.asarray(proxy.axes, dtype=np.float64)
    major, minor, front_normal = axes.T
    angle = radians(policy.discovery_tilt_deg)
    candidates: list[CandidateView] = []
    extents = (float(proxy.extents_m[0]), float(proxy.extents_m[1]))
    for side, side_sign in ((BladeSide.FRONT, 1.0), (BladeSide.BACK, -1.0)):
        normal = side_sign * front_normal
        target = proxy.center_m + normal * float(proxy.extents_m[2]) / 2.0
        for axis_name, direction, preferred_x in (
            ("major", major, minor),
            ("minor", minor, major),
        ):
            for sign_name, lateral_sign in (("negative", -1.0), ("positive", 1.0)):
                view_id = f"{side.value}_fin_discovery_{axis_name}_{sign_name}"
                position = (
                    target
                    + normal * standoff * cos(angle)
                    + direction * lateral_sign * standoff * sin(angle)
                )
                patch = SurfacePatch(view_id, side, 0, 0, target, normal, extents)
                candidates.append(
                    CandidateView(
                        view_id,
                        patch,
                        _look_at_pose(
                            view_id=view_id,
                            position_m=position,
                            target_m=target,
                            preferred_x=preferred_x,
                        ),
                        standoff,
                        geometric_footprint_m,
                        projection_fraction=cos(angle),
                        visibility_fraction=cos(angle),
                        distance_policy="proxy_fin_discovery_oblique",
                    )
                )
    adaptive_searches: list[AdaptiveFinDiscoverySearch] = []
    adaptive_enabled = planning_config.adaptive_ik_view_search.enabled
    endpoint_checker = reachability_checker
    if not adaptive_enabled and endpoint_validator is not None:
        if current_joint_positions_rad is None:
            raise UnknownBladeCoarseError(
                "Endpoint collision-aware fin discovery requires current joints"
            )
        endpoint_checker = EndpointCollisionAwareReachabilityChecker(
            reachability_checker,
            current_joint_positions_rad,
            endpoint_validator,
        )
    if adaptive_enabled:
        if point_cloud_config is None or current_joint_positions_rad is None:
            raise UnknownBladeCoarseError(
                "Adaptive fin discovery requires physical depth limits and current joints"
            )
        evaluated = []
        for candidate in candidates:
            search_config = _fin_discovery_search_config(
                candidate,
                planning_config,
                point_cloud_config,
                policy,
            )
            search = search_adaptive_candidate_family(
                candidate,
                proxy,
                filter_config,
                (reachability_checker,),
                current_joint_positions_rad,
                search_config,
                endpoint_validator=endpoint_validator,
            )
            adaptive_searches.append(AdaptiveFinDiscoverySearch(search_config, search))
            if search.ranked_feasible:
                # Preserve bounded same-semantics pose alternatives.  A high-gain
                # endpoint whose path is blocked must not hide a lower-ranked
                # angle/distance/roll pose for the same fin face.
                evaluated.extend(
                    attempt.evaluated
                    for attempt in search.ranked_feasible[
                        : search_config.maximum_ik_feasible_candidates
                    ]
                )
            elif search.attempts:
                evaluated.append(search.attempts[0].evaluated)
    else:
        filtered = filter_candidate_views(
            tuple(candidates),
            proxy,
            filter_config,
            endpoint_checker,
            deduplicate=False,
        )
        evaluated = list(filtered.candidates)

    # Generic normal-view fallbacks have different semantics and are intentionally
    # ignored here. Each entry below names one exact side/axis opposing pair.
    fallbacks = (
        () if adaptive_enabled else planning_config.paired_fin_discovery_fallbacks
    )
    for fallback_index, fallback in enumerate(fallbacks, start=1):
        side = BladeSide(fallback.side)
        interim = CoarseDiscoveryPlan(FilteredViewPlan(tuple(evaluated), ()), "pending")
        if _paired_discovery_ids(interim, side):
            continue
        if fallback.opposing_tilt_deg + 1e-12 < policy.discovery_tilt_deg:
            raise UnknownBladeCoarseError(
                "Paired fin-discovery fallback "
                f"{fallback_index} opposing_tilt_deg={fallback.opposing_tilt_deg} "
                f"is below policy discovery_tilt_deg={policy.discovery_tilt_deg}"
            )
        pair = _explicit_fin_discovery_pair(
            proxy=proxy,
            axes=axes,
            entry=fallback,
            entry_index=fallback_index,
            baseline_standoff_m=standoff,
            geometric_footprint_m=geometric_footprint_m,
        )
        checked = filter_candidate_views(
            pair,
            proxy,
            filter_config,
            endpoint_checker,
            deduplicate=False,
        )
        evaluated.extend(checked.candidates)
    filtered = FilteredViewPlan(tuple(evaluated), ())
    canonical = json.dumps(
        {
            "algorithm": (
                "adaptive_bilateral_paired_oblique_fin_discovery_v4"
                if adaptive_enabled
                else "explicit_bilateral_paired_oblique_fin_discovery_v3"
            ),
            "ik_branch_collision_filter_enabled": endpoint_validator is not None,
            "policy": asdict(policy),
            "view_planning": planning_config.model_dump(mode="json"),
            "view_filter": filter_config.model_dump(mode="json"),
            "candidate_poses": {
                item.candidate.view_id: item.candidate.base_t_left_ir.matrix.tolist()
                for item in filtered.candidates
            },
            "candidate_status": {
                item.candidate.view_id: item.status.value for item in filtered.candidates
            },
            "joint_positions_rad": {
                item.candidate.view_id: (
                    item.joint_positions_rad.tolist()
                    if item.joint_positions_rad is not None
                    else None
                )
                for item in filtered.candidates
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return CoarseDiscoveryPlan(
        filtered,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        tuple(adaptive_searches),
        current_joint_positions_rad if adaptive_enabled else None,
    )


def prepare_unknown_blade_coarse_view(
    *,
    captured: CapturedStopScanView,
    stereo: StereoInferenceObservation,
    stereo_inference_path: str | Path,
    occupancy_update: OccupancyFrameUpdate,
    occupancy_mapping_path: str | Path,
    hand_eye: HandEyeCalibration,
    settings: AppSettings,
    foreground_config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
    target_view_id: str,
    target_kind: CoarseTargetKind,
    target_side: BladeSide | None,
    side_proxy: BilateralBladeProxy | None = None,
    preflight_foreground: CoarseForegroundResult | None = None,
) -> PreparedCoarseScienceView:
    """Prepare one stopped coarse view from the occupancy integration-valid depth."""

    return _prepare_unknown_blade_coarse_view(
        captured=captured,
        stereo=stereo,
        stereo_inference_path=stereo_inference_path,
        integration_valid_mask=occupancy_update.integration_valid_mask,
        integration_valid_mask_content_hash=(
            occupancy_update.evidence.integration_valid_mask_content_hash
        ),
        integration_identity=(
            occupancy_update.evidence.source_view_id,
            occupancy_update.evidence.source_sequence_index,
            occupancy_update.evidence.frame_number,
        ),
        occupancy_mapping_path=occupancy_mapping_path,
        hand_eye=hand_eye,
        settings=settings,
        foreground_config=foreground_config,
        seed=seed,
        target_view_id=target_view_id,
        target_kind=target_kind,
        target_side=target_side,
        side_proxy=side_proxy,
        preflight_foreground=preflight_foreground,
    )


def prepare_unknown_blade_coarse_cycle(
    *,
    captured: CapturedStopScanView,
    result: PerceptionCycleResult,
    hand_eye: HandEyeCalibration,
    settings: AppSettings,
    foreground_config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
    target_view_id: str,
    target_kind: CoarseTargetKind,
    target_side: BladeSide | None,
    side_proxy: BilateralBladeProxy | None = None,
) -> PreparedCoarseScienceView:
    """Adapter from a committed-capable FoundationStereo cycle result.

    It does not mutate the cycle result or its occupancy publication.  The caller
    must still commit/reject that perception transaction through the existing
    engine.  This hook exists because the public result intentionally omits the
    in-memory ``OccupancyFrameUpdate``.
    """

    if result.bundle is not captured.bundle or result.raw_session_path != captured.raw_session_path:
        raise UnknownBladeCoarseError("Coarse adapter received a different capture transaction")
    stored_stereo = read_stereo_inference(result.stereo_inference_path)
    integration = read_coarse_integration_source(result.occupancy_mapping_path)
    return _prepare_unknown_blade_coarse_view(
        captured=captured,
        stereo=stored_stereo.observation,
        stereo_inference_path=result.stereo_inference_path,
        integration_valid_mask=integration.mask,
        integration_valid_mask_content_hash=integration.occupancy_content_sha256,
        integration_identity=(
            integration.source_view_id,
            integration.source_sequence_index,
            integration.frame_number,
        ),
        occupancy_mapping_path=result.occupancy_mapping_path,
        hand_eye=hand_eye,
        settings=settings,
        foreground_config=foreground_config,
        seed=seed,
        target_view_id=target_view_id,
        target_kind=target_kind,
        target_side=target_side,
        side_proxy=side_proxy,
    )


def _prepare_unknown_blade_coarse_view(
    *,
    captured: CapturedStopScanView,
    stereo: StereoInferenceObservation,
    stereo_inference_path: str | Path,
    integration_valid_mask: np.ndarray,
    integration_valid_mask_content_hash: str,
    integration_identity: tuple[str, int, int],
    occupancy_mapping_path: str | Path,
    hand_eye: HandEyeCalibration,
    settings: AppSettings,
    foreground_config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
    target_view_id: str,
    target_kind: CoarseTargetKind,
    target_side: BladeSide | None,
    side_proxy: BilateralBladeProxy | None,
    preflight_foreground: CoarseForegroundResult | None = None,
) -> PreparedCoarseScienceView:

    identity = (
        captured.bundle.view_id,
        captured.bundle.sequence_index,
        captured.bundle.stereo.frame_number,
    )
    if (
        identity
        != (
            stereo.source_view_id,
            stereo.source_sequence_index,
            stereo.rectified.source_frame_number,
        )
        or identity != integration_identity
    ):
        raise UnknownBladeCoarseError("Coarse cycle source identities differ")
    if preflight_foreground is None:
        if target_kind != "operator_seed":
            raise UnknownBladeCoarseError(
                "Automatic coarse view requires accepted-generation projected "
                "foreground preflight"
            )
        if seed is None or seed.mode != "hard_roi":
            raise UnknownBladeCoarseError(
                "Operator bootstrap requires a hard_roi foreground seed"
            )
        with performance_span("coarse.foreground"):
            foreground = bootstrap_blade_foreground(
                stereo.rectified.left_ir,
                stereo.depth_m,
                integration_valid_mask,
                foreground_config,
                seed,
            )
    else:
        foreground = preflight_foreground
    if foreground.config != foreground_config or foreground.seed != seed:
        raise UnknownBladeCoarseError("Coarse foreground differs from its staged policy")
    if foreground.valid_mask_content_sha256 != array_content_sha256(
        integration_valid_mask
    ) or integration_valid_mask_content_hash != occupancy_array_content_hash(
        integration_valid_mask
    ):
        raise UnknownBladeCoarseError("Coarse foreground is not integration-mask bound")
    with performance_span("coarse.reconstructed_view"):
        reconstructed = reconstruct_foundation_stereo_view(
            captured.bundle,
            stereo,
            foreground.mask,
            hand_eye,
            settings.point_cloud,
            kinematics_config=settings.kinematics,
            hand_eye_config=settings.hand_eye,
        )
    effective_side = _resolve_operator_bootstrap_side(
        reconstructed.base_t_projection_camera,
        side_proxy,
        target_side,
    )
    reconstructed_path = captured.cycle_root / "coarse_reconstructed_view"
    with performance_span("coarse.reconstructed_view_write"):
        write_reconstructed_view(
            reconstructed_path,
            reconstructed,
            foreground.mask,
            hand_eye,
            settings.point_cloud,
            settings.kinematics,
            settings.hand_eye,
            source_session=captured.raw_session_path,
            source_stereo_inference=stereo_inference_path,
        )
    coarse_view_path = captured.cycle_root / "coarse_scan_view"
    with performance_span("coarse.scan_view_write"):
        write_coarse_scan_view(
            coarse_view_path,
            foreground,
            reconstructed_view=reconstructed_path,
            source_stereo_inference=stereo_inference_path,
            source_occupancy_mapping=occupancy_mapping_path,
            target_view_id=target_view_id,
            target_kind=target_kind,
            target_side=effective_side,
            proxy_config=settings.proxy_model,
        )
    return PreparedCoarseScienceView(
        coarse_view_path.resolve(),
        reconstructed_path.resolve(),
        target_view_id,
        target_kind,
        effective_side,
    )


def _camera_side(view: StoredCoarseScanView, proxy: BilateralBladeProxy) -> BladeSide:
    local = proxy.frame_T_proxy.inverse().transform_points(
        view.reconstructed.view.base_t_projection_camera.translation_m
    )
    if abs(float(local[2])) <= 1e-9:
        raise UnknownBladeCoarseError("Coarse camera lies on the proxy mid-plane")
    return BladeSide.FRONT if local[2] > 0.0 else BladeSide.BACK


def append_coarse_scan_generation(
    output_dir: str | Path,
    *,
    new_view: str | Path,
    source_initialization: str | Path,
    source_view_plan: str | Path,
    source_discovery_plan: str | Path,
    settings: AppSettings,
    previous_generation: str | Path | None = None,
) -> Path:
    """Append one independently verified view and update measured proxy coverage."""

    with performance_span("coarse.generation_source_read"):
        current = read_coarse_scan_view(new_view)
    previous = None
    if previous_generation is not None:
        with performance_span("coarse.generation_previous_read"):
            previous = read_coarse_scan_generation(previous_generation)
    return _append_coarse_scan_generation_from_verified(
        output_dir,
        current=current,
        source_initialization=source_initialization,
        source_view_plan=source_view_plan,
        source_discovery_plan=source_discovery_plan,
        settings=settings,
        previous_generation=previous_generation,
        verified_previous_generation=previous,
    )


def _append_coarse_scan_generation_from_verified(
    output_dir: str | Path,
    *,
    current: StoredCoarseScanView,
    source_initialization: str | Path,
    source_view_plan: str | Path,
    source_discovery_plan: str | Path,
    settings: AppSettings,
    previous_generation: str | Path | None,
    verified_previous_generation: StoredCoarseScanGeneration | None,
) -> Path:
    """Append using strict current/predecessor reads from this transaction."""

    initialization_root = Path(source_initialization).resolve()
    plan_root = Path(source_view_plan).resolve()
    discovery_root = Path(source_discovery_plan).resolve()
    with performance_span("coarse.generation_planning_source_read"):
        initialization = read_initialization(initialization_root)
        plan = read_view_plan(plan_root)
    if current.proxy_config != settings.proxy_model:
        raise UnknownBladeCoarseError(
            "Coarse view was not filtered with the active blade-envelope policy"
        )
    if initialization.metadata["processing"]["proxy_model"] != (
        settings.proxy_model.model_dump(mode="json")
    ):
        raise UnknownBladeCoarseError(
            "Coarse initialization and active blade-envelope policy differ"
        )
    if _camera_side(current, initialization.observation.proxy) is not current.target_side:
        raise UnknownBladeCoarseError("Coarse target-side label disagrees with camera geometry")
    previous_path = Path(previous_generation).resolve() if previous_generation else None
    if (verified_previous_generation is None) != (previous_path is None):
        raise UnknownBladeCoarseError(
            "Verified coarse predecessor presence differs from append source"
        )
    if (
        verified_previous_generation is not None
        and verified_previous_generation.root.resolve() != previous_path
    ):
        raise UnknownBladeCoarseError(
            "Verified coarse predecessor root differs from append source"
        )
    previous = verified_previous_generation
    if previous is None:
        views = (current.root,)
        previous_coverage = None
        ledger: CoverageLedger = create_coverage_ledger(
            plan.result.geometric_plan,
            settings.coverage,
        )
    else:
        expected_initialization = Path(
            str(previous.metadata["sources"]["initialization"]["root"])
        ).resolve()
        expected_plan = Path(str(previous.metadata["sources"]["view_plan"]["root"])).resolve()
        if expected_initialization != initialization_root or expected_plan != plan_root:
            raise UnknownBladeCoarseError("Coarse generation changed proxy or view plan")
        if any(item.root == current.root for item in previous.views):
            raise UnknownBladeCoarseError("Coarse view was already accepted")
        views = (*tuple(item.root for item in previous.views), current.root)
        previous_coverage = previous.coverage_path
        with performance_span("coarse.coverage_previous_read"):
            ledger = read_coverage_ledger(previous.coverage_path).ledger
    source = current.reconstructed.metadata["source"]
    observation_id = coverage_observation_id(
        source["session"],
        current.reconstructed.view.source_view_id,
        current.reconstructed.view.source_sequence_index,
        current.reconstructed.view.source_frame_number,
    )
    with performance_span("coarse.coverage_update"):
        ledger = update_coverage(
            ledger,
            plan.result.geometric_plan,
            initialization.observation.proxy,
            current.support_cloud,
            current.reconstructed.view.base_t_projection_camera,
            observation_id,
        )
    output = Path(output_dir)
    coverage_path = output.with_name(f"{output.name}_coverage")
    coverage_created = False
    try:
        with performance_span("coarse.coverage_write"):
            write_coverage_ledger(
                coverage_path,
                ledger,
                source_plan=plan_root,
                source_initialization=initialization_root,
                previous_ledger=previous_coverage,
            )
        coverage_created = True
        with performance_span("coarse.generation_write"):
            return _write_coarse_scan_generation_from_verified(
                output,
                views=views,
                verified_views=(
                    (*previous.views, current)
                    if previous is not None
                    else (current,)
                ),
                coverage=coverage_path,
                source_initialization=initialization_root,
                source_view_plan=plan_root,
                source_discovery_plan=discovery_root,
                previous_generation=previous_generation,
                verified_previous_generation=previous,
            )
    except Exception:
        if coverage_created:
            shutil.rmtree(coverage_path, ignore_errors=True)
        raise


def _candidate_kind(candidate_id: str) -> CoarseTargetKind:
    identity = _discovery_view_identity(candidate_id)
    if identity is None:
        if "_fin_discovery_" in candidate_id:
            raise UnknownBladeCoarseError(
                f"Malformed fin-discovery candidate identity: {candidate_id!r}"
            )
        return "proxy_normal"
    _, axis, sign_name, _ = identity
    return f"fin_discovery_{axis}_{sign_name}"  # type: ignore[return-value]


def _candidate_attempts(
    generation: StoredCoarseScanGeneration,
) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for item in generation.views:
        attempts[item.target_view_id] = attempts.get(item.target_view_id, 0) + 1
    return attempts


def _verified_discovery_ids(
    generation: StoredCoarseScanGeneration,
    discovery: CoarseDiscoveryPlan,
    policy: CoarseSciencePolicy,
) -> frozenset[str]:
    candidates = {item.candidate.view_id: item for item in discovery.endpoint_feasible}
    verified: set[str] = set()
    for view in generation.views:
        candidate = candidates.get(view.target_view_id)
        if candidate is None or not view.target_kind.startswith("fin_discovery_"):
            continue
        actual = view.reconstructed.view.base_t_left_ir
        expected = candidate.candidate.base_t_left_ir
        if (
            np.linalg.norm(actual.translation_m - expected.translation_m)
            <= policy.maximum_discovery_translation_error_m
            and _rotation_error_deg(actual, expected) <= policy.maximum_discovery_rotation_error_deg
        ):
            verified.add(view.target_view_id)
    return frozenset(verified)


def _paired_discovery_ids(
    discovery: CoarseDiscoveryPlan,
    side: BladeSide,
) -> tuple[tuple[str, str], ...]:
    families: dict[tuple[str, str], dict[str, list[str]]] = {}
    for item in discovery.endpoint_feasible:
        identity = _discovery_view_identity(item.candidate.view_id)
        if identity is None or identity[0] is not side:
            continue
        _, axis, sign_name, family = identity
        families.setdefault((axis, family), {}).setdefault(sign_name, []).append(
            item.candidate.view_id
        )
    pairs: list[tuple[str, str]] = []
    for members in families.values():
        for negative in members.get("negative", ()):
            for positive in members.get("positive", ()):
                pairs.append((negative, positive))
    return tuple(pairs)


def _missing_discovery_pair_error(
    discovery: CoarseDiscoveryPlan,
    side: BladeSide,
) -> UnknownBladeCoarseError:
    candidates = [
        item
        for item in discovery.filtered.candidates
        if item.candidate.patch.side is side
        and _discovery_view_identity(item.candidate.view_id) is not None
    ]
    reasons = Counter(reason for item in candidates for reason in item.reasons)
    reason_summary = ", ".join(
        f"{reason} ({count})"
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    )
    adaptive = bool(discovery.adaptive_searches)
    if not reason_summary:
        reason_summary = "no endpoint supplied valid geometry and IK evidence"
    evaluation = (
        "adaptive angle/distance/roll evaluation"
        if adaptive
        else "explicit paired fallback evaluation"
    )
    recovery = (
        "change blade placement or the bounded adaptive search policy"
        if adaptive
        else "change blade placement, or add a measured paired_fin_discovery_fallbacks entry"
    )
    attempt_summary = f"tested {len(candidates)} semantic endpoints"
    if adaptive:
        traces = tuple(
            trace.result
            for trace in discovery.adaptive_searches
            if trace.result.nominal_view_id.startswith(f"{side.value}_fin_discovery_")
        )
        attempts = tuple(attempt for trace in traces for attempt in trace.attempts)
        rolls = sorted({attempt.parameters.roll_deg for attempt in attempts})
        azimuths = sorted({round(attempt.parameters.azimuth_deg, 6) for attempt in attempts})
        attempt_summary += (
            f" backed by {len(attempts)} pose samples across {len(traces)} families; "
            f"rolls={rolls}, azimuth_count={len(azimuths)}"
        )
        if any(trace.truncated for trace in traces):
            attempt_summary += "; bounded search prefix was truncated"
    return UnknownBladeCoarseError(
        f"No endpoint-feasible opposing fin-discovery pair exists on {side.value} "
        f"after {evaluation}; {attempt_summary}; "
        f"rejections: {reason_summary}. Automatic motion remains disabled. Start a new "
        f"attempt after you {recovery}; do not bypass IK or collision checks."
    )


def _require_bilateral_discovery_pairs(discovery: CoarseDiscoveryPlan) -> None:
    for side in (BladeSide.FRONT, BladeSide.BACK):
        if not _paired_discovery_ids(discovery, side):
            raise _missing_discovery_pair_error(discovery, side)


def _rank_candidates(
    generation: StoredCoarseScanGeneration,
    discovery: CoarseDiscoveryPlan,
    policy: CoarseSciencePolicy,
    *,
    require_additional_fin_evidence: bool,
) -> tuple[tuple[EvaluatedCandidate, CoarseDiscoveryGain], ...]:
    attempts = _candidate_attempts(generation)
    verified = _verified_discovery_ids(generation, discovery, policy)
    discovery_by_id = {item.candidate.view_id: item for item in discovery.endpoint_feasible}
    coverage = read_coverage_ledger(generation.coverage_path).ledger
    side_view_counts = Counter(item.target_side for item in generation.views)
    eligible: list[tuple[EvaluatedCandidate, float]] = []

    missing_pair_sides: list[BladeSide] = []
    exhausted_pair_sides: list[BladeSide] = []
    # Fin pairs are high-value candidates, not a requirement that must be proved
    # from the initial stopped posture.  Final schema-5 promotion still requires
    # measured bilateral opposing evidence.
    for side in (BladeSide.FRONT, BladeSide.BACK):
        pairs = _paired_discovery_ids(discovery, side)
        if not pairs:
            missing_pair_sides.append(side)
            continue
        complete = any(set(pair) <= verified for pair in pairs)
        if complete and not require_additional_fin_evidence:
            continue
        side_eligible = 0
        for pair in pairs:
            for view_id in pair:
                if (
                    view_id in verified
                    or attempts.get(view_id, 0) >= policy.maximum_attempts_per_candidate
                ):
                    continue
                opposite = pair[1] if view_id == pair[0] else pair[0]
                fin_evidence = (
                    1.0
                    if opposite in verified
                    else policy.discovery_gain_fin_seed_value
                )
                eligible.append((discovery_by_id[view_id], fin_evidence))
                side_eligible += 1
        if not complete and side_eligible == 0:
            exhausted_pair_sides.append(side)

    plan_root = Path(str(generation.metadata["sources"]["view_plan"]["root"])).resolve()
    reduced = select_uncovered_candidates(read_view_plan(plan_root).result.filtered_plan, coverage)
    proxy_endpoint = {
        item.candidate.view_id: item
        for item in reduced.remaining
        if item.status is CandidateStatus.ENDPOINT_FEASIBLE and item.joint_positions_rad is not None
    }
    for view_id in reduced.sequence.ordered_view_ids:
        if attempts.get(view_id, 0) < policy.maximum_attempts_per_candidate:
            eligible.append((proxy_endpoint[view_id], 0.0))

    eligible_by_view_id: dict[str, tuple[EvaluatedCandidate, float]] = {}
    for candidate, fin_evidence in eligible:
        view_id = candidate.candidate.view_id
        previous = eligible_by_view_id.get(view_id)
        if previous is None or fin_evidence > previous[1]:
            eligible_by_view_id[view_id] = (candidate, fin_evidence)
    ranked: list[tuple[EvaluatedCandidate, CoarseDiscoveryGain]] = []
    for candidate, fin_evidence in eligible_by_view_id.values():
        gain = expected_coarse_discovery_gain(
            candidate,
            coverage,
            side_observation_count=side_view_counts[candidate.candidate.patch.side],
            minimum_views_per_side=policy.minimum_views_per_side,
            fin_pair_evidence=fin_evidence,
            surface_weight=policy.discovery_gain_surface_weight,
            side_balance_weight=policy.discovery_gain_side_balance_weight,
            fin_pair_weight=policy.discovery_gain_fin_pair_weight,
        )
        ranked.append((candidate, gain))
    if ranked:
        ordered = tuple(sorted(
            ranked,
            key=lambda item: (
                -item[1].expected_gain,
                -item[1].fin_pair_evidence,
                -item[1].proxy_coverage_deficit,
                -item[1].side_observation_deficit,
                -item[0].metrics.geometric_score,
                item[0].candidate.view_id,
            ),
        ))
        eligible_ranked = tuple(
            item
            for item in ordered
            if item[1].expected_gain + 1e-12 >= policy.discovery_gain_minimum
        )
        if not eligible_ranked:
            raise UnknownBladeCoarseError(
                "Coarse evidence is incomplete, but every feasible view has expected "
                "discovery gain below the configured minimum"
            )
        return eligible_ranked
    if missing_pair_sides:
        raise _missing_discovery_pair_error(discovery, missing_pair_sides[0])
    if exhausted_pair_sides:
        raise UnknownBladeCoarseError(
            "Fin-discovery attempts exhausted without an opposing pair on "
            + ", ".join(side.value for side in exhausted_pair_sides)
        )
    if reduced.blocked_patch_ids:
        raise UnknownBladeCoarseError(
            "Incomplete proxy patches have no endpoint-feasible target: "
            + ", ".join(reduced.blocked_patch_ids)
        )
    raise UnknownBladeCoarseError(
        "Coarse evidence is incomplete but all endpoint-feasible attempts are exhausted"
    )


def _select_candidate(
    generation: StoredCoarseScanGeneration,
    discovery: CoarseDiscoveryPlan,
    policy: CoarseSciencePolicy,
    *,
    require_additional_fin_evidence: bool,
) -> tuple[EvaluatedCandidate, CoarseDiscoveryGain]:
    """Compatibility helper returning the highest science-ranked endpoint."""

    return _rank_candidates(
        generation,
        discovery,
        policy,
        require_additional_fin_evidence=require_additional_fin_evidence,
    )[0]


def select_coarse_next_view(
    generation_path: str | Path,
    discovery: CoarseDiscoveryPlan,
    hand_eye: HandEyeCalibration,
    policy: CoarseSciencePolicy,
    *,
    require_additional_fin_evidence: bool = False,
) -> NextViewSelection:
    """Return a coordinator-compatible, non-authorizing coarse selector result."""

    generation = read_coarse_scan_generation(generation_path)
    if generation.coarse_model_path is not None:
        required = max(1, len(read_coverage_ledger(generation.coverage_path).ledger.patches))
        initialization_root = Path(
            str(generation.metadata["sources"]["initialization"]["root"])
        ).resolve()
        return NextViewSelection(
            None,
            _sha256(generation.root / "generation.json"),
            # The coarse coordinator pins the proxy initialization as its reference
            # on the first selection.  Schema-5 is an output handed to a new fine
            # coordinator, not a mid-run replacement for that reference contract.
            _sha256(initialization_root / INITIALIZATION_METADATA_FILENAME),
            discovery.policy_sha256,
            required,
            0,
            True,
            ("schema-5 coarse reference is committed",),
        )
    ranked = _rank_candidates(
        generation,
        discovery,
        policy,
        require_additional_fin_evidence=require_additional_fin_evidence,
    )
    coverage = read_coverage_ledger(generation.coverage_path).ledger
    required = len(coverage.patches) + 4
    incomplete = max(1, required - len(coverage.completed_patch_ids))
    initialization_root = Path(
        str(generation.metadata["sources"]["initialization"]["root"])
    ).resolve()
    ranked_candidates: list[RankedNextViewCandidate] = []
    for rank, (candidate, gain) in enumerate(ranked, start=1):
        diagnostics = (
            "algorithm=single_initial_view_proxy_fin_gain_nbv_v1",
            f"science_rank={rank}",
            f"coarse target kind={_candidate_kind(candidate.candidate.view_id)}",
            f"expected_discovery_gain={gain.expected_gain:.6f}",
            f"gain_proxy_coverage_deficit={gain.proxy_coverage_deficit:.6f}",
            f"gain_side_observation_deficit={gain.side_observation_deficit:.6f}",
            f"gain_fin_pair_evidence={gain.fin_pair_evidence:.6f}",
            f"gain_measurement_quality={gain.measurement_quality:.6f}",
            "endpoint IK is feasible; trajectory safety remains unproven here",
        )
        ranked_candidates.append(
            RankedNextViewCandidate(
                next_view_target_from_candidate(candidate, hand_eye),
                diagnostics,
            )
        )
    selected = ranked_candidates[0]
    return NextViewSelection(
        selected.target,
        _sha256(generation.root / "generation.json"),
        _sha256(initialization_root / INITIALIZATION_METADATA_FILENAME),
        discovery.policy_sha256,
        required,
        min(incomplete, required),
        False,
        selected.diagnostics,
        ranked_candidates=tuple(ranked_candidates),
    )


def finalize_coarse_generation(
    generation_path: str | Path,
    discovery: CoarseDiscoveryPlan,
    policy: CoarseSciencePolicy,
    settings: AppSettings,
    *,
    output_coarse_model: str | Path,
    output_ready_generation: str | Path,
) -> CoarsePhaseTransition:
    """Build and atomically name the fine reference only after every hard gate."""

    generation = read_coarse_scan_generation(generation_path)
    if generation.coarse_model_path is not None:
        return CoarsePhaseTransition(
            CoarsePhase.READY_FOR_FINE,
            ("schema-5 reference was already committed",),
            generation.root,
            generation.root,
            generation.coarse_model_path,
        )
    counts = {
        side: sum(item.target_side is side for item in generation.views)
        for side in (BladeSide.FRONT, BladeSide.BACK)
    }
    reasons: list[str] = []
    if len(generation.views) < policy.minimum_total_views:
        reasons.append(f"view_count={len(generation.views)}<{policy.minimum_total_views}")
    for side, count in counts.items():
        if count < policy.minimum_views_per_side:
            reasons.append(f"{side.value}_view_count={count}<{policy.minimum_views_per_side}")
    verified = _verified_discovery_ids(generation, discovery, policy)
    for side in (BladeSide.FRONT, BladeSide.BACK):
        pairs = _paired_discovery_ids(discovery, side)
        if not any(set(pair) <= verified for pair in pairs):
            reasons.append(f"{side.value} has no verified opposing oblique pair")
    coverage = read_coverage_ledger(generation.coverage_path).ledger
    if policy.require_complete_proxy_coverage and len(coverage.completed_patch_ids) != len(
        coverage.patches
    ):
        reasons.append(
            f"proxy_coverage={len(coverage.completed_patch_ids)}/{len(coverage.patches)}"
        )
    if reasons:
        return CoarsePhaseTransition(
            CoarsePhase.COLLECTING,
            tuple(reasons),
            generation.root,
        )

    reconstructed_roots = tuple(
        Path(item.metadata["sources"]["reconstructed_view"]["root"]).resolve()
        for item in generation.views
    )
    coarse_output = Path(output_coarse_model).resolve()
    if coarse_output.exists():
        _assert_reusable_coarse_model(
            coarse_output,
            source_views=reconstructed_roots,
            source_coarse_views=tuple(item.root for item in generation.views),
            settings=settings,
        )
        coarse_path = coarse_output
    else:
        views = tuple(
            replace(item.reconstructed.view, base_cloud=item.support_cloud)
            if isinstance(item.reconstructed.view, ReconstructedBladeView)
            else item.reconstructed.view
            for item in generation.views
        )
        result = build_coarse_blade_model(
            views,
            views[0].planning_intrinsics,
            settings.multi_view_fusion,
            settings.surface_partition,
            settings.view_planning,
            settings.tsdf,
            settings.surface_quality,
        )
        missing_fin_sides = tuple(
            side
            for side in (BladeSide.FRONT, BladeSide.BACK)
            if (
                result.surface.fin_component(side) is None
                or not result.surface.fin_component(side).two_faces_observed  # type: ignore[union-attr]
            )
        )
        if missing_fin_sides:
            unused = {
                item.candidate.view_id for item in discovery.endpoint_feasible
            } - _verified_discovery_ids(generation, discovery, policy)
            phase = CoarsePhase.COLLECTING_FIN_EVIDENCE if unused else CoarsePhase.BLOCKED
            return CoarsePhaseTransition(
                phase,
                tuple(
                    f"{side.value} fin lacks two-face coarse evidence" for side in missing_fin_sides
                ),
                generation.root,
            )
        coarse_path = write_coarse_model(
            coarse_output,
            result,
            settings,
            source_views=reconstructed_roots,
            source_coarse_views=tuple(item.root for item in generation.views),
        )
    initialization_root = Path(
        str(generation.metadata["sources"]["initialization"]["root"])
    ).resolve()
    plan_root = Path(str(generation.metadata["sources"]["view_plan"]["root"])).resolve()
    discovery_root = Path(str(generation.metadata["sources"]["discovery_plan"]["root"])).resolve()
    ready_path = write_coarse_scan_generation(
        output_ready_generation,
        views=tuple(item.root for item in generation.views),
        coverage=generation.coverage_path,
        source_initialization=initialization_root,
        source_view_plan=plan_root,
        source_discovery_plan=discovery_root,
        previous_generation=generation.root,
        coarse_model=coarse_path,
    )
    return CoarsePhaseTransition(
        CoarsePhase.READY_FOR_FINE,
        (
            "bilateral proxy coverage complete",
            "front fin has two-face evidence",
            "back fin has two-face evidence",
            "schema-5 coarse model committed",
        ),
        generation.root,
        ready_path,
        coarse_path,
    )


def _assert_reusable_coarse_model(
    path: Path,
    *,
    source_views: tuple[Path, ...],
    source_coarse_views: tuple[Path, ...],
    settings: AppSettings,
) -> None:
    """Accept only the exact immutable model left by an interrupted promotion.

    A model directory is never deleted or overwritten.  Recovery first runs the
    complete schema-5 reader, then binds the model to the exact reconstructed-view
    sequence and every configuration block used by ``write_coarse_model``.
    """

    try:
        stored = read_coarse_model_summary(path)
        metadata = stored.metadata
        if int(metadata["schema_version"]) != 5:
            raise ValueError("existing coarse model is not schema 5")
        actual_sources = tuple(
            Path(str(record["path"])).resolve() for record in metadata["source_views"]
        )
        if actual_sources != source_views:
            raise ValueError("source reconstructed-view sequence differs")
        support = metadata.get("proxy_support")
        if support is None:
            raise ValueError("existing coarse model lacks per-view blade-envelope provenance")
        actual_support_sources = tuple(
            Path(str(record["path"])).resolve() for record in support["source_coarse_views"]
        )
        if actual_support_sources != source_coarse_views:
            raise ValueError("source coarse-view support sequence differs")
        if support["configuration"] != settings.proxy_model.model_dump(mode="json"):
            raise ValueError("proxy-support configuration differs")
        expected_configurations = {
            "fusion": settings.multi_view_fusion.model_dump(mode="json"),
            "surface": settings.surface_partition.model_dump(mode="json"),
            "view_plan": settings.view_planning.model_dump(mode="json"),
            "tsdf": settings.tsdf.model_dump(mode="json"),
            "quality": settings.surface_quality.model_dump(mode="json"),
        }
        for section, expected in expected_configurations.items():
            if metadata[section]["configuration"] != expected:
                raise ValueError(f"{section} configuration differs")
        fin_components = metadata["surface"]["fin_components"]
        observed_fin_sides = tuple(
            str(component["side"])
            for component in fin_components
            if bool(component["two_faces_observed"])
        )
        required_fin_sides = (BladeSide.FRONT.value, BladeSide.BACK.value)
        if len(fin_components) != 2 or sorted(observed_fin_sides) != sorted(required_fin_sides):
            raise ValueError("bilateral fin two-face evidence differs")
    except (KeyError, TypeError, ValueError) as exc:
        raise UnknownBladeCoarseError(
            "Existing coarse-model asset is not the exact recoverable schema-5 output; "
            "refusing to overwrite or reuse it"
        ) from exc


def _file_source_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise UnknownBladeCoarseError(f"Coarse session source is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _write_discovery_plan_asset(
    output_dir: Path,
    discovery: CoarseDiscoveryPlan,
    *,
    source_initialization: Path,
    source_view_plan: Path,
    source_kinematics: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Coarse discovery output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.{uuid4().hex}.partial")
    temporary.mkdir()
    try:
        payload = {
            "schema_version": 4,
            "artifact_kind": "biblade_fusion.coarse_fin_discovery_plan",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "policy_sha256": discovery.policy_sha256,
            "current_joint_positions_rad": (
                list(discovery.current_joint_positions_rad)
                if discovery.current_joint_positions_rad is not None
                else None
            ),
            "sources": {
                "initialization": _file_source_record(
                    source_initialization / INITIALIZATION_METADATA_FILENAME
                ),
                "view_plan": _file_source_record(source_view_plan / "view_plan.json"),
                "kinematics": _file_source_record(source_kinematics),
            },
            "candidates": [
                {
                    "view_id": item.candidate.view_id,
                    "side": item.candidate.patch.side.value,
                    "base_T_left_ir": item.candidate.base_t_left_ir.matrix.tolist(),
                    "status": item.status.value,
                    "reasons": list(item.reasons),
                    "joint_positions_rad": (
                        item.joint_positions_rad.tolist()
                        if item.joint_positions_rad is not None
                        else None
                    ),
                }
                for item in discovery.filtered.candidates
            ],
        }
        if discovery.adaptive_searches:
            assert discovery.current_joint_positions_rad is not None
            targets = [
                adaptive_view_search_payload(
                    trace.result,
                    trace.config,
                    discovery.current_joint_positions_rad,
                    source_initialization=str(source_initialization.resolve()),
                    source_kinematics=str(source_kinematics.resolve()),
                )
                for trace in discovery.adaptive_searches
            ]
            payload["adaptive_ik_fin_discovery"] = {
                "motion_authorized": False,
                "endpoint_collision_checked": bool(targets) and all(
                    target["endpoint_collision_checked"] is True for target in targets
                ),
                "trajectory_checked": False,
                "targets": targets,
            }
        (temporary / "discovery.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir.resolve()


def _verify_discovery_plan_asset(
    path: Path,
    discovery: CoarseDiscoveryPlan,
    *,
    source_initialization: Path,
    source_view_plan: Path,
    source_kinematics: Path,
) -> None:
    payload = json.loads((path / "discovery.json").read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") not in {1, 2, 3, 4}
        or payload.get("artifact_kind") != "biblade_fusion.coarse_fin_discovery_plan"
        or payload.get("motion_authorized") is not False
        or payload.get("policy_sha256") != discovery.policy_sha256
    ):
        raise UnknownBladeCoarseError("Persisted coarse discovery policy changed")
    if payload.get("schema_version") == 4:
        expected_current = (
            list(discovery.current_joint_positions_rad)
            if discovery.current_joint_positions_rad is not None
            else None
        )
        if payload.get("current_joint_positions_rad") != expected_current:
            raise UnknownBladeCoarseError(
                "Persisted coarse discovery current posture changed"
            )
    adaptive_payload = payload.get("adaptive_ik_fin_discovery")
    if adaptive_payload is not None:
        if any(
            adaptive_payload.get(name) is not False
            for name in ("motion_authorized", "trajectory_checked")
        ):
            raise UnknownBladeCoarseError(
                "Adaptive fin discovery must remain explicitly non-authorizing"
            )
        endpoint_checked = adaptive_payload.get("endpoint_collision_checked")
        if type(endpoint_checked) is not bool:
            raise UnknownBladeCoarseError(
                "Adaptive fin discovery endpoint-collision status is invalid"
            )
    expected_sources = {
        "initialization": source_initialization / INITIALIZATION_METADATA_FILENAME,
        "view_plan": source_view_plan / "view_plan.json",
        "kinematics": source_kinematics,
    }
    for name, expected in expected_sources.items():
        record = payload["sources"][name]
        actual = Path(str(record["path"])).resolve()
        if (
            actual != expected.resolve()
            or _sha256(actual) != record["sha256"]
            or actual.stat().st_size != int(record["size_bytes"])
        ):
            raise UnknownBladeCoarseError(f"Coarse discovery {name} source changed")
    expected_candidates = {item.candidate.view_id: item for item in discovery.filtered.candidates}
    records = {str(item["view_id"]): item for item in payload["candidates"]}
    if set(records) != set(expected_candidates):
        raise UnknownBladeCoarseError("Coarse discovery candidate identities changed")
    for view_id, item in expected_candidates.items():
        record = records[view_id]
        joints = item.joint_positions_rad.tolist() if item.joint_positions_rad is not None else None
        if (
            record["side"] != item.candidate.patch.side.value
            or record["status"] != item.status.value
            or record["reasons"] != list(item.reasons)
            or record["joint_positions_rad"] != joints
            or not np.allclose(
                record["base_T_left_ir"],
                item.candidate.base_t_left_ir.matrix,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise UnknownBladeCoarseError(f"Coarse discovery candidate changed: {view_id}")


class CoarseScienceSession:
    """Motion-free production composition for bootstrap-to-schema-5 coarse science.

    The first accepted view creates the proxy initialization, endpoint-filtered
    normal plan and persistent oblique discovery plan.  Every later view appends a
    new immutable generation.  The class can be recovered only from an explicitly
    named generation; it never follows a mutable ``latest`` pointer.
    """

    def __init__(
        self,
        *,
        settings: AppSettings,
        hand_eye: HandEyeCalibration,
        reachability_checker: ReachabilityChecker,
        source_kinematics: str | Path,
        output_root: str | Path,
        foreground_config: BootstrapForegroundConfig | None = None,
        policy: CoarseSciencePolicy | None = None,
        endpoint_validator: EndpointConfigurationValidator | None = None,
        recovered_generation: str | Path | None = None,
    ) -> None:
        source_kinematics_path = Path(source_kinematics).resolve()
        if not source_kinematics_path.is_file():
            raise ValueError("Coarse session requires a persisted kinematics source")
        self._settings = settings.model_copy(deep=True)
        self._hand_eye = hand_eye
        self._reachability = reachability_checker
        self._endpoint_validator = endpoint_validator
        self._source_kinematics = source_kinematics_path
        self._output_root = Path(output_root).resolve()
        self._foreground_config = foreground_config or BootstrapForegroundConfig()
        self._policy = policy or CoarseSciencePolicy()
        self._generation: Path | None = None
        self._initialization: Path | None = None
        self._view_plan: Path | None = None
        self._discovery_path: Path | None = None
        self._discovery: CoarseDiscoveryPlan | None = None
        self._requires_additional_fin_evidence = False
        self._last_transition: CoarsePhaseTransition | None = None
        self._pending_selection: NextViewSelection | None = None
        self._pending_operator_side: BladeSide | None = None
        self._pending_seed: BootstrapSeed | None = None
        self._pending_seed_provider: BootstrapSeedProvider | None = None
        self._pending_foreground: CoarseForegroundResult | None = None
        self._pending_prepared: PreparedCoarseScienceView | None = None
        self._pending_live_readback: _CoarseScanViewReadback | None = None
        self._operator_capture_staged = False
        if recovered_generation is not None:
            self._recover(Path(recovered_generation).resolve())

    @property
    def current_generation_path(self) -> Path | None:
        return self._generation

    @property
    def reference_coarse_model_path(self) -> Path | None:
        if self._generation is None:
            return None
        return read_coarse_scan_generation(self._generation).coarse_model_path

    @property
    def discovery_plan(self) -> CoarseDiscoveryPlan | None:
        return self._discovery

    @property
    def last_transition(self) -> CoarsePhaseTransition | None:
        return self._last_transition

    @property
    def motion_authorized(self) -> bool:
        return False

    def _recover(self, generation_path: Path) -> None:
        generation = read_coarse_scan_generation(generation_path)
        sources = generation.metadata["sources"]
        initialization = Path(str(sources["initialization"]["root"])).resolve()
        view_plan = Path(str(sources["view_plan"]["root"])).resolve()
        discovery_path = Path(str(sources["discovery_plan"]["root"])).resolve()
        stored_initialization = read_initialization(initialization)
        stored_plan = read_view_plan(view_plan)
        discovery_payload = json.loads(
            (discovery_path / "discovery.json").read_text(encoding="utf-8")
        )
        current_joint_positions = discovery_payload.get("current_joint_positions_rad")
        if current_joint_positions is None:
            # Schema <= 3 discovery assets were evaluated only at the operator
            # bootstrap posture.  Preserve their deterministic recovery contract.
            current_joint_positions = (
                stored_initialization.observation.seed_joint_positions_rad
            )
        current = tuple(float(value) for value in current_joint_positions)
        if len(current) != 6 or not np.isfinite(current).all():
            raise UnknownBladeCoarseError(
                "Persisted coarse discovery current posture is invalid"
            )
        discovery = generate_fin_discovery_plan(
            stored_initialization.observation.proxy,
            stored_plan.result.geometric_plan.footprint_m,
            self._settings.view_planning,
            self._settings.view_filter,
            self._policy,
            self._reachability,
            self._settings.point_cloud,
            current,
            self._endpoint_validator,
        )
        _verify_discovery_plan_asset(
            discovery_path,
            discovery,
            source_initialization=initialization,
            source_view_plan=view_plan,
            source_kinematics=self._source_kinematics,
        )
        self._generation = generation.root
        self._initialization = initialization
        self._view_plan = view_plan
        self._discovery_path = discovery_path
        self._discovery = discovery

    def refresh_discovery(
        self,
        *,
        current_joint_positions_rad: tuple[float, float, float, float, float, float],
        reachability_checker: ReachabilityChecker,
    ) -> None:
        """Re-evaluate fin endpoints from the latest stopped robot posture.

        IK feasibility is state dependent.  Keeping the bootstrap posture's
        evaluated candidate set for the whole experiment could permanently label
        a view unreachable even after an earlier safe move made it reachable.
        Each revision is immutable and the accepting generation records the exact
        revision used for its selection.
        """

        if self._pending_selection is not None or self._pending_prepared is not None:
            raise UnknownBladeCoarseError(
                "Cannot refresh fin discovery while a coarse transaction is pending"
            )
        if (
            self._generation is None
            or self._initialization is None
            or self._view_plan is None
            or self._discovery is None
        ):
            raise UnknownBladeCoarseError(
                "Cannot refresh fin discovery before bootstrap initialization"
            )
        current = tuple(float(value) for value in current_joint_positions_rad)
        if len(current) != 6 or not np.isfinite(current).all():
            raise UnknownBladeCoarseError("Current discovery posture is invalid")
        prior = self._discovery.current_joint_positions_rad
        if prior is not None and np.max(np.abs(np.asarray(current) - np.asarray(prior))) <= 1e-3:
            return
        initialization = read_initialization(self._initialization)
        view_plan = read_view_plan(self._view_plan)
        with performance_span("coarse.fin_discovery_refresh"):
            discovery = generate_fin_discovery_plan(
                initialization.observation.proxy,
                view_plan.result.geometric_plan.footprint_m,
                self._settings.view_planning,
                self._settings.view_filter,
                self._policy,
                reachability_checker,
                self._settings.point_cloud,
                current,
                self._endpoint_validator,
            )
        posture_sha256 = hashlib.sha256(
            json.dumps(current, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        generation_index = read_coarse_scan_generation(self._generation).generation_index
        output = (
            self._output_root
            / "fin_discovery_revisions"
            / f"{generation_index:06d}_{posture_sha256[:16]}"
        )
        if output.exists():
            _verify_discovery_plan_asset(
                output,
                discovery,
                source_initialization=self._initialization,
                source_view_plan=self._view_plan,
                source_kinematics=self._source_kinematics,
            )
            discovery_path = output.resolve()
        else:
            discovery_path = _write_discovery_plan_asset(
                output,
                discovery,
                source_initialization=self._initialization,
                source_view_plan=self._view_plan,
                source_kinematics=self._source_kinematics,
            )
        self._reachability = reachability_checker
        self._discovery = discovery
        self._discovery_path = discovery_path

    def prepare_cycle(
        self,
        *,
        captured: CapturedStopScanView,
        result: PerceptionCycleResult,
        seed: BootstrapSeed | None,
        selection: NextViewSelection | None = None,
        operator_side: BladeSide | None = None,
    ) -> PreparedCoarseScienceView:
        """Turn one stopped FoundationStereo cycle into a pending coarse asset."""

        if selection is None:
            target_view_id = captured.bundle.view_id
            target_kind: CoarseTargetKind = "operator_seed"
            target_side = operator_side
            side_proxy = (
                read_initialization(self._initialization).observation.proxy
                if self._initialization is not None
                else None
            )
        else:
            if selection.coverage_complete or selection.target is None:
                raise UnknownBladeCoarseError("Coarse cycle selection has no target")
            if captured.bundle.view_id != selection.target.view_id:
                raise UnknownBladeCoarseError("Coarse capture does not match its selected target")
            target_view_id = selection.target.view_id
            candidate = self._candidate(target_view_id)
            target_kind = _candidate_kind(target_view_id)
            target_side = candidate.candidate.patch.side
            side_proxy = None
        return prepare_unknown_blade_coarse_cycle(
            captured=captured,
            result=result,
            hand_eye=self._hand_eye,
            settings=self._settings,
            foreground_config=self._foreground_config,
            seed=seed,
            target_view_id=target_view_id,
            target_kind=target_kind,
            target_side=target_side,
            side_proxy=side_proxy,
        )

    def stage_operator_capture(
        self,
        *,
        seed: BootstrapSeed | None = None,
        seed_provider: BootstrapSeedProvider | None = None,
        operator_side: BladeSide | None = None,
    ) -> None:
        """Stage one explicitly triggered manual bootstrap for the engine hook."""

        if (
            self._pending_prepared is not None
            or self._pending_selection is not None
            or self._operator_capture_staged
        ):
            raise UnknownBladeCoarseError("A coarse engine transaction is already pending")
        self._pending_seed = seed
        self._pending_seed_provider = seed_provider
        self._pending_foreground = None
        self._pending_operator_side = operator_side
        # ``None`` selection is meaningful, so pending readiness is represented by
        # a sentinel boolean stored on the instance.
        self._operator_capture_staged = True

    def stage_selected_capture(
        self,
        selection: NextViewSelection,
        *,
        seed: BootstrapSeed | None = None,
    ) -> None:
        """Bind the next engine CANDIDATE capture to one selector decision."""

        if (
            self._pending_prepared is not None
            or self._pending_selection is not None
            or self._operator_capture_staged
        ):
            raise UnknownBladeCoarseError("A coarse engine transaction is already pending")
        if selection.coverage_complete or selection.target is None:
            raise UnknownBladeCoarseError("Cannot stage a completed coarse selection")
        self._pending_selection = selection
        self._pending_operator_side = None
        self._pending_seed = seed
        self._pending_seed_provider = None
        self._pending_foreground = None
        self._operator_capture_staged = False

    def preflight_engine_cycle(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        prepared_occupancy: PreparedOccupancyFrame,
    ) -> None:
        """Resolve and validate foreground before occupancy ray integration."""

        if self._pending_prepared is not None or self._pending_foreground is not None:
            raise UnknownBladeCoarseError("Prior coarse preflight awaits acceptance")
        operator_staged = self._operator_capture_staged
        selection = self._pending_selection
        if selection is None and not operator_staged:
            raise UnknownBladeCoarseError(
                "Coarse capture was not explicitly staged by operator or selector"
            )
        identity = (
            captured.bundle.view_id,
            captured.bundle.sequence_index,
            captured.bundle.stereo.frame_number,
        )
        prepared_identity = (
            prepared_occupancy.bundle.view_id,
            prepared_occupancy.bundle.sequence_index,
            prepared_occupancy.stereo.rectified.source_frame_number,
        )
        if identity != prepared_identity:
            raise UnknownBladeCoarseError("Coarse preflight source identities differ")

        annotation_root: Path | None = None
        seed = self._pending_seed
        if operator_staged:
            annotation_root, image_path = _materialize_bootstrap_annotation_request(
                captured,
                stereo,
                stereo_path,
                prepared_occupancy.self_mask.integration_valid_mask,
            )
            if seed is None:
                provider = self._pending_seed_provider
                if provider is None:
                    raise UnknownBladeCoarseError(
                        "Operator bootstrap requires a hard_roi polygon for this formal frame"
                    )
                seed = provider(captured, image_path)
            if seed.mode != "hard_roi":
                raise UnknownBladeCoarseError("Operator bootstrap requires seed mode hard_roi")
            self._pending_seed = seed

        with performance_span("coarse.foreground"):
            if operator_staged or seed is not None:
                foreground: CoarseForegroundResult = bootstrap_blade_foreground(
                    stereo.rectified.left_ir,
                    stereo.depth_m,
                    prepared_occupancy.self_mask.integration_valid_mask,
                    self._foreground_config,
                    seed,
                )
            else:
                if self._generation is None:
                    raise UnknownBladeCoarseError(
                        "Automatic coarse foreground has no accepted blade generation"
                    )
                foreground = _projected_foreground_from_generation(
                    self._generation,
                    stereo=stereo,
                    integration_valid_mask=(
                        prepared_occupancy.self_mask.integration_valid_mask
                    ),
                    base_t_left_rectified=prepared_occupancy.base_t_camera,
                    foreground_config=self._foreground_config,
                    settings=self._settings,
                )
        if annotation_root is not None:
            _write_bootstrap_annotation_response(annotation_root, foreground)
        self._pending_foreground = foreground

    def prepare_engine_cycle(
        self,
        captured: CapturedStopScanView,
        stereo: StereoInferenceObservation,
        stereo_path: Path,
        occupancy_update: OccupancyFrameUpdate,
        occupancy_path: Path,
    ) -> Path:
        """Exact ``CoarseSciencePreparer`` hook used inside the cycle engine."""

        if self._pending_prepared is not None:
            raise UnknownBladeCoarseError("Prior coarse engine asset awaits acceptance")
        operator_staged = self._operator_capture_staged
        selection = self._pending_selection
        if selection is None and not operator_staged:
            raise UnknownBladeCoarseError(
                "Coarse capture was not explicitly staged by operator or selector"
            )
        if self._pending_foreground is None:
            raise UnknownBladeCoarseError(
                "Coarse foreground preflight was not completed before occupancy rebuild"
            )
        if selection is None:
            target_view_id = captured.bundle.view_id
            target_kind: CoarseTargetKind = "operator_seed"
            target_side = self._pending_operator_side
            side_proxy = (
                read_initialization(self._initialization).observation.proxy
                if self._initialization is not None
                else None
            )
        else:
            assert selection.target is not None
            if captured.bundle.view_id != selection.target.view_id:
                raise UnknownBladeCoarseError("Engine capture differs from staged coarse target")
            candidate = self._candidate(selection.target.view_id)
            target_view_id = selection.target.view_id
            target_kind = _candidate_kind(target_view_id)
            target_side = candidate.candidate.patch.side
            side_proxy = None
        prepared = prepare_unknown_blade_coarse_view(
            captured=captured,
            stereo=stereo,
            stereo_inference_path=stereo_path,
            occupancy_update=occupancy_update,
            occupancy_mapping_path=occupancy_path,
            hand_eye=self._hand_eye,
            settings=self._settings,
            foreground_config=self._foreground_config,
            seed=self._pending_seed,
            target_view_id=target_view_id,
            target_kind=target_kind,
            target_side=target_side,
            side_proxy=side_proxy,
            preflight_foreground=self._pending_foreground,
        )
        self._pending_prepared = prepared
        return prepared.coarse_view_path

    def accept_cycle(self, result: PerceptionCycleResult) -> Path:
        """Append the hook asset only after the engine/coordinator accepted its cycle."""

        prepared = self._pending_prepared
        if (
            prepared is None
            or result.coarse_scan_view_path is None
            or result.coarse_scan_view_path != prepared.coarse_view_path
        ):
            raise UnknownBladeCoarseError("Accepted cycle does not match the pending coarse asset")
        recorder = try_create_performance_timing(
            transaction_kind="coarse_generation_accept",
            identity={
                "target_view_id": prepared.target_view_id,
                "target_kind": prepared.target_kind,
                "target_side": prepared.target_side.value,
                "coarse_scan_view": str(prepared.coarse_view_path),
            },
        )
        if recorder is None:
            generation = self.accept_prepared_view(prepared)
        else:
            generation = self._accept_prepared_view_with_timing(recorder, prepared)
        self._pending_prepared = None
        self._pending_selection = None
        self._pending_operator_side = None
        self._pending_seed = None
        self._pending_seed_provider = None
        self._pending_foreground = None
        self._operator_capture_staged = False
        return generation

    def _accept_prepared_view_with_timing(
        self,
        recorder: PerformanceTimingRecorder,
        prepared: PreparedCoarseScienceView,
    ) -> Path:
        """Observe generation acceptance without expanding the public result contract."""

        status = "failed"
        error: str | None = None
        try:
            with (
                activate_performance_timing(recorder),
                performance_span("coarse.generation_accept"),
            ):
                generation = self.accept_prepared_view(prepared)
            status = "completed"
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            recorder.write_best_effort(
                prepared.coarse_view_path.parent / "coarse_generation_timing.json",
                status=status,
                error=error,
            )
        return generation

    def reject_cycle(self) -> None:
        """Forget staged state after the engine rejects/cancels the immutable cycle."""

        self._pending_prepared = None
        self._pending_selection = None
        self._pending_operator_side = None
        self._pending_seed = None
        self._pending_seed_provider = None
        self._pending_foreground = None
        self._pending_live_readback = None
        self._operator_capture_staged = False

    def take_live_readback(
        self,
        *,
        expected_coarse_view: str | Path,
    ) -> _CoarseScanViewReadback:
        """Transfer one accepted strict read to the command-incapable live observer."""

        readback = self._pending_live_readback
        self._pending_live_readback = None
        if readback is None:
            raise UnknownBladeCoarseError("Accepted coarse view has no live readback")
        if readback.root != Path(expected_coarse_view).resolve():
            raise UnknownBladeCoarseError(
                "Accepted coarse live readback differs from the coordinator result"
            )
        return readback

    def _candidate(self, view_id: str) -> EvaluatedCandidate:
        candidates: list[EvaluatedCandidate] = []
        if self._view_plan is not None:
            candidates.extend(read_view_plan(self._view_plan).result.filtered_plan.candidates)
        if self._discovery is not None:
            candidates.extend(self._discovery.filtered.candidates)
        matches = [item for item in candidates if item.candidate.view_id == view_id]
        if len(matches) != 1:
            raise UnknownBladeCoarseError(f"Expected one coarse target {view_id!r}")
        return matches[0]

    def accept_prepared_view(self, prepared: PreparedCoarseScienceView) -> Path:
        """Append a prepared view, creating all proxy assets on the first call."""

        with performance_span("coarse.scan_view_readback"):
            stored = read_coarse_scan_view(prepared.coarse_view_path)
            live_readback = _bind_coarse_scan_view_readback(stored)
        if stored.target_view_id != prepared.target_view_id:
            raise UnknownBladeCoarseError("Prepared coarse-view identity changed")
        if self._generation is None and self._initialization is None:
            with performance_span("coarse.first_view_initialization"):
                self._initialize_from_first_view(stored)
        assert self._initialization is not None
        assert self._view_plan is not None
        assert self._discovery_path is not None
        index = 0
        previous = None
        if self._generation is not None:
            with performance_span("coarse.generation_head_read"):
                previous = read_coarse_scan_generation(self._generation)
                index = previous.generation_index + 1
        output = self._output_root / "generations" / f"{index:06d}"
        generation = _append_coarse_scan_generation_from_verified(
            output,
            current=stored,
            source_initialization=self._initialization,
            source_view_plan=self._view_plan,
            source_discovery_plan=self._discovery_path,
            settings=self._settings,
            previous_generation=self._generation,
            verified_previous_generation=previous,
        )
        self._generation = generation
        self._pending_live_readback = live_readback
        return generation

    def _initialize_from_first_view(self, stored: StoredCoarseScanView) -> None:
        view = stored.reconstructed.view
        if stored.proxy_config != self._settings.proxy_model:
            raise UnknownBladeCoarseError(
                "First coarse view was not filtered with the active blade-envelope policy"
            )
        support = stored.proxy_support
        with performance_span("coarse.proxy_build"):
            proxy = build_bilateral_proxy(
                view.base_cloud.points_m[support.mask],
                view.base_t_projection_camera,
                self._settings.proxy_model,
            )
        observation = InitialObservation(
            view.source_view_id,
            view.planning_intrinsics,
            view.joint_positions_rad,
            view.base_t_left_ir,
            view.base_t_projection_camera,
            view.base_cloud,
            proxy,
            "foundation_stereo",
            view.source_sequence_index,
            view.source_frame_number,
            view.pose_authority,
            support.mask,
        )
        initialization = self._output_root / "initialization"
        view_plan = self._output_root / "proxy_view_plan"
        discovery_output = self._output_root / "fin_discovery_plan"
        source = stored.reconstructed.metadata["source"]
        created: list[Path] = []
        try:
            with performance_span("coarse.initialization_write"):
                write_initialization(
                    initialization,
                    observation,
                    stored.foreground.mask,
                    self._hand_eye,
                    self._settings.point_cloud,
                    self._settings.proxy_model,
                    self._settings.kinematics,
                    self._settings.hand_eye,
                    source_session=source["session"],
                    source_stereo_inference=source["stereo_inference"],
                )
            created.append(initialization)
            with performance_span("coarse.initial_plan_candidate_filter"):
                planning = plan_initial_observation(
                    observation,
                    self._settings.view_planning,
                    self._settings.view_filter,
                    self._reachability,
                    self._settings.point_cloud,
                    self._endpoint_validator,
                )
            with performance_span("coarse.view_plan_write"):
                write_view_plan(
                    view_plan,
                    planning,
                    self._settings.view_planning,
                    self._settings.view_filter,
                    source_initialization=initialization,
                    source_kinematics=self._source_kinematics,
                    joint_zero_offsets_rad=self._settings.kinematics.joint_zero_offsets_rad,
                )
            created.append(view_plan)
            with performance_span("coarse.fin_discovery_candidate_filter"):
                discovery = generate_fin_discovery_plan(
                    proxy,
                    planning.geometric_plan.footprint_m,
                    self._settings.view_planning,
                    self._settings.view_filter,
                    self._policy,
                    self._reachability,
                    self._settings.point_cloud,
                    tuple(float(value) for value in observation.seed_joint_positions_rad),
                    self._endpoint_validator,
                )
            with performance_span("coarse.discovery_plan_write"):
                discovery_path = _write_discovery_plan_asset(
                    discovery_output,
                    discovery,
                    source_initialization=initialization,
                    source_view_plan=view_plan,
                    source_kinematics=self._source_kinematics,
                )
            created.append(discovery_output)
        except Exception:
            for path in reversed(created):
                shutil.rmtree(path, ignore_errors=True)
            raise
        self._initialization = initialization.resolve()
        self._view_plan = view_plan.resolve()
        self._discovery_path = discovery_path
        self._discovery = discovery

    def select_next(self) -> NextViewSelection:
        if self._generation is None or self._discovery is None:
            raise UnknownBladeCoarseError("Coarse session has no accepted proxy generation")
        return select_coarse_next_view(
            self._generation,
            self._discovery,
            self._hand_eye,
            self._policy,
            require_additional_fin_evidence=self._requires_additional_fin_evidence,
        )

    def evaluate_transition(self) -> CoarsePhaseTransition:
        if self._generation is None or self._discovery is None:
            raise UnknownBladeCoarseError("Coarse session has no generation to evaluate")
        current = read_coarse_scan_generation(self._generation)
        transition = finalize_coarse_generation(
            current.root,
            self._discovery,
            self._policy,
            self._settings,
            output_coarse_model=self._output_root / "coarse_model_schema5",
            output_ready_generation=(
                self._output_root / "generations" / f"{current.generation_index + 1:06d}_schema5"
            ),
        )
        self._last_transition = transition
        self._requires_additional_fin_evidence = (
            transition.phase is CoarsePhase.COLLECTING_FIN_EVIDENCE
        )
        if transition.phase is CoarsePhase.READY_FOR_FINE:
            assert transition.ready_generation_path is not None
            self._generation = transition.ready_generation_path
        return transition
