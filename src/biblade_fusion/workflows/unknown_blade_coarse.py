"""Online coarse-science composition for one unknown bilateral finned blade.

This module is deliberately motion-free.  It prepares one stopped scientific view,
appends it to an immutable proxy-coverage generation, selects only endpoint-feasible
coarse targets, and promotes a generation to a schema-5 reference only after both
blade sides and both faces of the single fin on each side have evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import cos, radians, sin
from pathlib import Path
from uuid import uuid4

import numpy as np

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AppSettings, ViewFilterConfig, ViewPlanningConfig
from biblade_fusion.perception.bootstrap_foreground import (
    BootstrapForegroundConfig,
    BootstrapSeed,
    array_content_sha256,
    bootstrap_blade_foreground,
)
from biblade_fusion.perception.proxy import (
    BilateralBladeProxy,
    build_bilateral_proxy,
)
from biblade_fusion.planning import (
    BladeSide,
    CandidateStatus,
    CandidateView,
    CoverageLedger,
    EvaluatedCandidate,
    FilteredViewPlan,
    ReachabilityChecker,
    SurfacePatch,
    coverage_observation_id,
    create_coverage_ledger,
    filter_candidate_views,
    select_uncovered_candidates,
    update_coverage,
)
from biblade_fusion.storage.coarse_model import (
    read_coarse_model_summary,
    write_coarse_model,
)
from biblade_fusion.storage.coarse_scan import (
    CoarseTargetKind,
    StoredCoarseScanGeneration,
    StoredCoarseScanView,
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
    next_view_target_from_candidate,
)
from biblade_fusion.workflows.view_planning import plan_initial_observation


class UnknownBladeCoarseError(RuntimeError):
    """The coarse phase cannot safely prepare, recover, select, or promote."""


class CoarsePhase(StrEnum):
    COLLECTING = "collecting"
    COLLECTING_FIN_EVIDENCE = "collecting_fin_evidence"
    READY_FOR_FINE = "ready_for_fine"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class CoarseSciencePolicy:
    """Explicit completion and conservative fin-discovery policy."""

    discovery_tilt_deg: float = 15.0
    minimum_total_views: int = 6
    minimum_views_per_side: int = 3
    maximum_attempts_per_candidate: int = 2
    require_complete_proxy_coverage: bool = True
    maximum_discovery_translation_error_m: float = 0.020
    maximum_discovery_rotation_error_deg: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.discovery_tilt_deg < 45.0:
            raise ValueError("Coarse discovery tilt must lie in (0, 45) degrees")
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
class CoarseDiscoveryPlan:
    filtered: FilteredViewPlan
    policy_sha256: str

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


def generate_fin_discovery_plan(
    proxy: BilateralBladeProxy,
    geometric_footprint_m: tuple[float, float],
    planning_config: ViewPlanningConfig,
    filter_config: ViewFilterConfig,
    policy: CoarseSciencePolicy,
    reachability_checker: ReachabilityChecker,
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
                patch = SurfacePatch(
                    view_id,
                    side,
                    0,
                    0,
                    target,
                    normal,
                    extents,
                )
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
                        distance_policy="proxy_fin_discovery_oblique",
                    )
                )
    filtered = filter_candidate_views(
        tuple(candidates),
        proxy,
        filter_config,
        reachability_checker,
        deduplicate=False,
    )
    canonical = json.dumps(
        {
            "algorithm": "bilateral_two_axis_paired_oblique_fin_discovery_v1",
            "policy": asdict(policy),
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
    foreground = bootstrap_blade_foreground(
        stereo.rectified.left_ir,
        stereo.depth_m,
        integration_valid_mask,
        foreground_config,
        seed,
    )
    if foreground.valid_mask_content_sha256 != array_content_sha256(
        integration_valid_mask
    ) or integration_valid_mask_content_hash != occupancy_array_content_hash(
        integration_valid_mask
    ):
        raise UnknownBladeCoarseError("Coarse foreground is not integration-mask bound")
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

    initialization_root = Path(source_initialization).resolve()
    plan_root = Path(source_view_plan).resolve()
    discovery_root = Path(source_discovery_plan).resolve()
    initialization = read_initialization(initialization_root)
    plan = read_view_plan(plan_root)
    current = read_coarse_scan_view(new_view)
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
    if previous_generation is None:
        views = (current.root,)
        previous_coverage = None
        ledger: CoverageLedger = create_coverage_ledger(
            plan.result.geometric_plan,
            settings.coverage,
        )
    else:
        previous = read_coarse_scan_generation(previous_generation)
        expected_initialization = Path(
            str(previous.metadata["sources"]["initialization"]["root"])
        ).resolve()
        expected_plan = Path(str(previous.metadata["sources"]["view_plan"]["root"])).resolve()
        expected_discovery = Path(
            str(previous.metadata["sources"]["discovery_plan"]["root"])
        ).resolve()
        if (
            expected_initialization != initialization_root
            or expected_plan != plan_root
            or expected_discovery != discovery_root
        ):
            raise UnknownBladeCoarseError("Coarse generation changed proxy or view plan")
        if any(item.root == current.root for item in previous.views):
            raise UnknownBladeCoarseError("Coarse view was already accepted")
        views = (*tuple(item.root for item in previous.views), current.root)
        previous_coverage = previous.coverage_path
        ledger = read_coverage_ledger(previous.coverage_path).ledger
    source = current.reconstructed.metadata["source"]
    observation_id = coverage_observation_id(
        source["session"],
        current.reconstructed.view.source_view_id,
        current.reconstructed.view.source_sequence_index,
        current.reconstructed.view.source_frame_number,
    )
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
        write_coverage_ledger(
            coverage_path,
            ledger,
            source_plan=plan_root,
            source_initialization=initialization_root,
            previous_ledger=previous_coverage,
        )
        coverage_created = True
        return write_coarse_scan_generation(
            output,
            views=views,
            coverage=coverage_path,
            source_initialization=initialization_root,
            source_view_plan=plan_root,
            source_discovery_plan=discovery_root,
            previous_generation=previous_generation,
        )
    except Exception:
        if coverage_created:
            shutil.rmtree(coverage_path, ignore_errors=True)
        raise


def _candidate_kind(candidate_id: str) -> CoarseTargetKind:
    prefix = candidate_id.split("_fin_discovery_", 1)
    if len(prefix) != 2:
        return "proxy_normal"
    axis, sign_name = prefix[1].rsplit("_", 1)
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
    feasible = {item.candidate.view_id for item in discovery.endpoint_feasible}
    pairs = []
    for axis in ("major", "minor"):
        pair = (
            f"{side.value}_fin_discovery_{axis}_negative",
            f"{side.value}_fin_discovery_{axis}_positive",
        )
        if set(pair) <= feasible:
            pairs.append(pair)
    return tuple(pairs)


def _select_candidate(
    generation: StoredCoarseScanGeneration,
    discovery: CoarseDiscoveryPlan,
    policy: CoarseSciencePolicy,
    *,
    require_additional_fin_evidence: bool,
) -> EvaluatedCandidate:
    attempts = _candidate_attempts(generation)
    verified = _verified_discovery_ids(generation, discovery, policy)
    discovery_by_id = {item.candidate.view_id: item for item in discovery.endpoint_feasible}
    # At least one opposing pair on each side is a hard precondition.  An
    # unreachable member never counts as observed or complete.
    for side in (BladeSide.FRONT, BladeSide.BACK):
        pairs = _paired_discovery_ids(discovery, side)
        if not pairs:
            raise UnknownBladeCoarseError(
                f"No endpoint-feasible opposing fin-discovery pair exists on {side.value}"
            )
        if not any(set(pair) <= verified for pair in pairs):
            for pair in pairs:
                for view_id in pair:
                    if (
                        view_id not in verified
                        and attempts.get(view_id, 0) < policy.maximum_attempts_per_candidate
                    ):
                        return discovery_by_id[view_id]
            raise UnknownBladeCoarseError(
                f"Fin-discovery attempts exhausted without an opposing pair on {side.value}"
            )
    if require_additional_fin_evidence:
        for side in (BladeSide.FRONT, BladeSide.BACK):
            for pair in _paired_discovery_ids(discovery, side):
                for view_id in pair:
                    if (
                        view_id not in verified
                        and attempts.get(view_id, 0) < policy.maximum_attempts_per_candidate
                    ):
                        return discovery_by_id[view_id]

    plan_root = Path(str(generation.metadata["sources"]["view_plan"]["root"])).resolve()
    coverage = read_coverage_ledger(generation.coverage_path).ledger
    reduced = select_uncovered_candidates(read_view_plan(plan_root).result.filtered_plan, coverage)
    endpoint = {
        item.candidate.view_id: item
        for item in reduced.remaining
        if item.status is CandidateStatus.ENDPOINT_FEASIBLE and item.joint_positions_rad is not None
    }
    for view_id in reduced.sequence.ordered_view_ids:
        if attempts.get(view_id, 0) < policy.maximum_attempts_per_candidate:
            return endpoint[view_id]
    if reduced.blocked_patch_ids:
        raise UnknownBladeCoarseError(
            "Incomplete proxy patches have no endpoint-feasible target: "
            + ", ".join(reduced.blocked_patch_ids)
        )
    raise UnknownBladeCoarseError(
        "Coarse evidence is incomplete but all endpoint-feasible attempts are exhausted"
    )


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
    candidate = _select_candidate(
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
    return NextViewSelection(
        next_view_target_from_candidate(candidate, hand_eye),
        _sha256(generation.root / "generation.json"),
        _sha256(initialization_root / INITIALIZATION_METADATA_FILENAME),
        discovery.policy_sha256,
        required,
        min(incomplete, required),
        False,
        (
            f"coarse target kind={_candidate_kind(candidate.candidate.view_id)}",
            "endpoint IK is feasible; trajectory safety remains unproven here",
        ),
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
                    f"{side.value} fin lacks two-face coarse evidence"
                    for side in missing_fin_sides
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
            Path(str(record["path"])).resolve()
            for record in support["source_coarse_views"]
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
            "schema_version": 1,
            "artifact_kind": "biblade_fusion.coarse_fin_discovery_plan",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "motion_authorized": False,
            "policy_sha256": discovery.policy_sha256,
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
        payload.get("schema_version") != 1
        or payload.get("artifact_kind") != "biblade_fusion.coarse_fin_discovery_plan"
        or payload.get("motion_authorized") is not False
        or payload.get("policy_sha256") != discovery.policy_sha256
    ):
        raise UnknownBladeCoarseError("Persisted coarse discovery policy changed")
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
        recovered_generation: str | Path | None = None,
    ) -> None:
        source_kinematics_path = Path(source_kinematics).resolve()
        if not source_kinematics_path.is_file():
            raise ValueError("Coarse session requires a persisted kinematics source")
        self._settings = settings.model_copy(deep=True)
        self._hand_eye = hand_eye
        self._reachability = reachability_checker
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
        self._pending_prepared: PreparedCoarseScienceView | None = None
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
        discovery = generate_fin_discovery_plan(
            stored_initialization.observation.proxy,
            stored_plan.result.geometric_plan.footprint_m,
            self._settings.view_planning,
            self._settings.view_filter,
            self._policy,
            self._reachability,
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
        self._operator_capture_staged = False

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
        generation = self.accept_prepared_view(prepared)
        self._pending_prepared = None
        self._pending_selection = None
        self._pending_operator_side = None
        self._pending_seed = None
        self._operator_capture_staged = False
        return generation

    def reject_cycle(self) -> None:
        """Forget staged state after the engine rejects/cancels the immutable cycle."""

        self._pending_prepared = None
        self._pending_selection = None
        self._pending_operator_side = None
        self._pending_seed = None
        self._operator_capture_staged = False

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

        stored = read_coarse_scan_view(prepared.coarse_view_path)
        if stored.target_view_id != prepared.target_view_id:
            raise UnknownBladeCoarseError("Prepared coarse-view identity changed")
        if self._generation is None and self._initialization is None:
            self._initialize_from_first_view(stored)
        assert self._initialization is not None
        assert self._view_plan is not None
        assert self._discovery_path is not None
        index = 0
        if self._generation is not None:
            index = read_coarse_scan_generation(self._generation).generation_index + 1
        output = self._output_root / "generations" / f"{index:06d}"
        generation = append_coarse_scan_generation(
            output,
            new_view=stored.root,
            source_initialization=self._initialization,
            source_view_plan=self._view_plan,
            source_discovery_plan=self._discovery_path,
            settings=self._settings,
            previous_generation=self._generation,
        )
        self._generation = generation
        return generation

    def _initialize_from_first_view(self, stored: StoredCoarseScanView) -> None:
        view = stored.reconstructed.view
        if stored.proxy_config != self._settings.proxy_model:
            raise UnknownBladeCoarseError(
                "First coarse view was not filtered with the active blade-envelope policy"
            )
        support = stored.proxy_support
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
            planning = plan_initial_observation(
                observation,
                self._settings.view_planning,
                self._settings.view_filter,
                self._reachability,
            )
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
            discovery = generate_fin_discovery_plan(
                proxy,
                planning.geometric_plan.footprint_m,
                self._settings.view_planning,
                self._settings.view_filter,
                self._policy,
                self._reachability,
            )
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
