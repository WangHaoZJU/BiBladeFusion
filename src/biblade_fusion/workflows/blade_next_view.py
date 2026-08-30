"""Coverage-first next-view selection for the bilateral single-fin blade.

Scientific completion is derived only from an immutable fine-coverage generation.
The short-lived occupancy generation is intentionally not part of scoring; it is
consumed later by the segment safety preflight.  Rectified camera poses drive image
geometry while raw left-IR poses remain the robot IK target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from math import acos, degrees
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.calibration import (
    HandEyeCalibration,
    load_cs68_kinematics,
)
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AppSettings,
    FineFinalizationConfig,
    KinematicsConfig,
    MotionPreflightConfig,
    MultiViewFusionConfig,
    NextViewSelectionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
)
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    SurfaceRegion,
    generate_reacquisition_view,
)
from biblade_fusion.planning import (
    BladeClearanceEnvelope,
    CandidateStatus,
    EliteCs68IkChecker,
    EvaluatedCandidate,
    ReachabilityChecker,
    filter_candidate_views,
)
from biblade_fusion.planning.surface_coverage import SurfacePatchQuality
from biblade_fusion.planning.views import BladeSide, CandidateView
from biblade_fusion.robotics import Es68KinematicModel, load_es68_flange_t_tcp
from biblade_fusion.storage.coarse_model import (
    COARSE_MODEL_SCHEMA_VERSION,
    read_coarse_model_summary,
)
from biblade_fusion.storage.reconstructed_view import read_reconstructed_view
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.surface_coverage import (
    REACQUISITION_VIEW_ID_SCHEMA,
    StoredSurfaceCoverageGeneration,
    reacquisition_view_id,
    read_surface_coverage_generation,
)
from biblade_fusion.workflows.fine_completion import (
    FinalFineCompletionEvidence,
    finalize_fine_science,
    finalize_unaccepted_fine_science,
)
from biblade_fusion.workflows.stop_scan_coordinator import (
    BladePlanningAssetError,
    NextViewSelection,
    NextViewUnavailable,
    OccupancyGeneration,
    PerceptionCycleResult,
    next_view_target_from_candidate,
)

_SELECTOR_ALGORITHM = "bilateral_single_fin_coverage_priority_v2"


class FlangeForwardKinematics(Protocol):
    """Minimal calibrated ES68 FK boundary required for IK result verification."""

    def base_t_flange(self, joint_positions_rad: NDArray[np.float64]) -> PoseSE3: ...


ReachabilityFactory = Callable[[NDArray[np.float64]], ReachabilityChecker]
CoverageReader = Callable[[str], StoredSurfaceCoverageGeneration]
FineFinalizer = Callable[
    [StoredSurfaceCoverageGeneration], FinalFineCompletionEvidence
]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reacquisition_view_id(
    candidate: CandidateView,
    attempt: int,
    selection_policy_sha256: str,
) -> str:
    if attempt < 1:
        raise ValueError("Reacquisition attempt indices are one-based")
    return reacquisition_view_id(
        candidate.view_id,
        candidate.patch.patch_id,
        attempt,
        selection_policy_sha256,
    )


def _selection_policy_payload(
    *,
    hand_eye: HandEyeCalibration,
    selection_config: NextViewSelectionConfig,
    surface_quality_config: SurfaceQualityConfig,
    view_filter_config: ViewFilterConfig,
    kinematics_config: KinematicsConfig,
    motion_config: MotionPreflightConfig,
    expected_reference_root: Path,
    expected_reference_sha256: str,
    fk_implementation: str,
    fusion_config: MultiViewFusionConfig | None,
    tsdf_config: TSDFConfig | None,
    finalization_config: FineFinalizationConfig | None,
) -> dict[str, object]:
    return {
        "algorithm": _SELECTOR_ALGORITHM,
        "selection": selection_config.model_dump(mode="json"),
        "surface_quality": surface_quality_config.model_dump(mode="json"),
        "view_filter": view_filter_config.model_dump(mode="json"),
        "kinematics": kinematics_config.model_dump(mode="json"),
        "motion_endpoint_gate": {
            "maximum_translation_error_m": motion_config.maximum_endpoint_translation_error_m,
            "maximum_rotation_error_deg": motion_config.maximum_endpoint_rotation_error_deg,
        },
        "expected_reference": {
            "root": str(expected_reference_root.resolve()),
            "metadata_sha256": expected_reference_sha256,
        },
        "terminal_reconstruction": (
            {
                "fusion": fusion_config.model_dump(mode="json"),
                "tsdf": tsdf_config.model_dump(mode="json"),
                "finalization": finalization_config.model_dump(mode="json"),
            }
            if fusion_config is not None
            and tsdf_config is not None
            and finalization_config is not None
            else {"implementation": "injected_or_unconfigured"}
        ),
        "flange_T_left_ir": hand_eye.require_flange_primary().matrix.tolist(),
        "fk_implementation": fk_implementation,
    }


def production_selection_policy_payload(
    settings: AppSettings,
    hand_eye: HandEyeCalibration,
    *,
    reference_coarse_model: str | Path,
) -> dict[str, object]:
    """Rebuild the exact policy payload shared by production selection and capture."""

    reference_root = Path(reference_coarse_model).resolve()
    reference = read_coarse_model_summary(reference_root)
    if int(reference.metadata["schema_version"]) != COARSE_MODEL_SCHEMA_VERSION:
        raise BladePlanningAssetError("coverage selection requires a schema-5 coarse model")
    return _selection_policy_payload(
        hand_eye=hand_eye,
        selection_config=settings.next_view_selection,
        surface_quality_config=settings.surface_quality,
        view_filter_config=settings.view_filter,
        kinematics_config=settings.kinematics,
        motion_config=settings.motion_preflight,
        expected_reference_root=reference_root,
        expected_reference_sha256=_file_sha256(reference_root / "metadata.json"),
        fk_implementation=(
            f"{Es68KinematicModel.__module__}.{Es68KinematicModel.__qualname__}"
        ),
        fusion_config=settings.multi_view_fusion,
        tsdf_config=settings.tsdf,
        finalization_config=settings.fine_finalization,
    )


def _rotation_distance_deg(first: PoseSE3, second: PoseSE3) -> float:
    relative = first.rotation.T @ second.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return degrees(acos(cosine))


def _surface_envelope(surface: CurvedBladeSurface) -> BladeClearanceEnvelope:
    """Build a conservative OBB solely for generic geometric filtering."""

    points = np.vstack([patch.points_m for patch in surface.patches])
    local = (points - surface.center_m) @ surface.axes
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    extents = np.maximum(upper - lower, 1e-6)
    local_center = (lower + upper) / 2.0
    world_center = surface.center_m + surface.axes @ local_center
    return BladeClearanceEnvelope(
        PoseSE3.from_rotation_translation(
            "base",
            "selector_blade_proxy",
            surface.axes,
            world_center,
        ),
        extents,
    )


def _target_base_t_tcp(
    candidate: CandidateView,
    hand_eye: HandEyeCalibration,
) -> PoseSE3:
    flange_t_left_ir = hand_eye.require_flange_primary()
    canonical_camera = PoseSE3("base", "left_ir", candidate.base_t_left_ir.matrix)
    base_t_flange = canonical_camera.compose(flange_t_left_ir.inverse())
    return base_t_flange.compose(load_es68_flange_t_tcp())


class BladeCoverageNextViewSelector:
    """Select one endpoint-feasible view from independently measured fine coverage.

    ``None`` is never used for ambiguity: a target-less result is emitted only after
    every configured patch is independently proven complete.  Missing science
    assets fail the run; incomplete coverage with no feasible unused candidate is a
    typed planning block.
    """

    def __init__(
        self,
        *,
        hand_eye: HandEyeCalibration,
        selection_config: NextViewSelectionConfig,
        surface_quality_config: SurfaceQualityConfig,
        view_filter_config: ViewFilterConfig,
        kinematics_config: KinematicsConfig,
        motion_config: MotionPreflightConfig,
        expected_reference_root: str | Path,
        expected_reference_sha256: str,
        reachability_factory: ReachabilityFactory | None = None,
        fk_model: FlangeForwardKinematics | None = None,
        coverage_reader: Callable[
            [str], StoredSurfaceCoverageGeneration
        ] = read_surface_coverage_generation,
        fine_finalizer: FineFinalizer | None = None,
        fusion_config: MultiViewFusionConfig | None = None,
        tsdf_config: TSDFConfig | None = None,
        finalization_config: FineFinalizationConfig | None = None,
    ) -> None:
        self._hand_eye = hand_eye
        self._flange_t_left_ir = hand_eye.require_flange_primary()
        self._selection_config = NextViewSelectionConfig.model_validate(
            selection_config.model_dump(mode="json")
        )
        self._surface_quality_config = SurfaceQualityConfig.model_validate(
            surface_quality_config.model_dump(mode="json")
        )
        self._view_filter_config = ViewFilterConfig.model_validate(
            view_filter_config.model_dump(mode="json")
        )
        self._kinematics_config = KinematicsConfig.model_validate(
            kinematics_config.model_dump(mode="json")
        )
        self._motion_config = MotionPreflightConfig.model_validate(
            motion_config.model_dump(mode="json")
        )
        self._expected_reference_root = Path(expected_reference_root).resolve()
        self._expected_reference_sha256 = str(expected_reference_sha256)
        if len(self._expected_reference_sha256) != 64 or any(
            value not in "0123456789abcdef"
            for value in self._expected_reference_sha256
        ):
            raise BladePlanningAssetError(
                "Expected coarse-reference identity must be a SHA-256 digest"
            )
        self._coverage_reader = coverage_reader
        self._fine_finalizer = fine_finalizer
        self._last_cycle_key: tuple[str, int] | None = None
        self._last_surface_root: Path | None = None
        self._last_surface_generation_id: str | None = None

        if reachability_factory is None:
            model_path = self._kinematics_config.model_path
            if model_path is None:
                raise BladePlanningAssetError(
                    "Coverage next-view selection requires kinematics.model_path"
                )
            try:
                ik_model = load_cs68_kinematics(model_path)
            except Exception as exc:
                raise BladePlanningAssetError(
                    f"Cannot load the controller-specific ES68 kinematics: {exc}"
                ) from exc

            def build_checker(seed: NDArray[np.float64]) -> ReachabilityChecker:
                return EliteCs68IkChecker(
                    ik_model,
                    self._hand_eye,
                    seed,
                    self._kinematics_config,
                )

            reachability_factory = build_checker
        self._reachability_factory = reachability_factory

        try:
            self._fk_model = fk_model or Es68KinematicModel.from_resources(
                joint_zero_offsets_rad=(
                    self._kinematics_config.joint_zero_offsets_rad
                )
            )
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Cannot initialize the calibrated ES68 FK verifier: {exc}"
            ) from exc
        self._policy_payload = _selection_policy_payload(
            hand_eye=self._hand_eye,
            selection_config=self._selection_config,
            surface_quality_config=self._surface_quality_config,
            view_filter_config=self._view_filter_config,
            kinematics_config=self._kinematics_config,
            motion_config=self._motion_config,
            expected_reference_root=self._expected_reference_root,
            expected_reference_sha256=self._expected_reference_sha256,
            fk_implementation=(
                f"{type(self._fk_model).__module__}.{type(self._fk_model).__qualname__}"
            ),
            fusion_config=fusion_config,
            tsdf_config=tsdf_config,
            finalization_config=finalization_config,
        )
        self._policy_sha256 = _canonical_sha256(self._policy_payload)

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        hand_eye: HandEyeCalibration,
        *,
        reference_coarse_model: str | Path,
        science_authority: ScienceAcceptanceAuthority | None,
        experimental: bool = False,
    ) -> BladeCoverageNextViewSelector:
        """Build the production selector without connecting to the robot."""

        if science_authority is None and not experimental:
            raise BladePlanningAssetError(
                "Production fine selector requires a science acceptance authority"
            )
        if science_authority is not None and experimental:
            raise BladePlanningAssetError(
                "Experimental fine selector cannot claim a science acceptance authority"
            )

        reference_root = Path(reference_coarse_model).resolve()
        try:
            reference = read_coarse_model_summary(reference_root)
            if int(reference.metadata["schema_version"]) != COARSE_MODEL_SCHEMA_VERSION:
                raise ValueError("coverage selection requires a schema-5 coarse model")
            reference_sha256 = _file_sha256(reference_root / "metadata.json")
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Cannot pin the expected coarse-model reference: {exc}"
            ) from exc
        return cls(
            hand_eye=hand_eye,
            selection_config=settings.next_view_selection,
            surface_quality_config=settings.surface_quality,
            view_filter_config=settings.view_filter,
            kinematics_config=settings.kinematics,
            motion_config=settings.motion_preflight,
            expected_reference_root=reference_root,
            expected_reference_sha256=reference_sha256,
            fine_finalizer=(
                lambda state: finalize_unaccepted_fine_science(
                    state,
                    fusion_config=settings.multi_view_fusion,
                    tsdf_config=settings.tsdf,
                    surface_quality_config=settings.surface_quality,
                    finalization_config=settings.fine_finalization,
                )
                if experimental
                else lambda state: finalize_fine_science(
                    state,
                    fusion_config=settings.multi_view_fusion,
                    tsdf_config=settings.tsdf,
                    surface_quality_config=settings.surface_quality,
                    finalization_config=settings.fine_finalization,
                    science_authority=science_authority,  # type: ignore[arg-type]
                )
            ),
            fusion_config=settings.multi_view_fusion,
            tsdf_config=settings.tsdf,
            finalization_config=settings.fine_finalization,
        )

    @property
    def selection_policy_sha256(self) -> str:
        return self._policy_sha256

    @property
    def selection_policy_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(self._policy_payload))

    def _read_state(
        self,
        observation: PerceptionCycleResult,
    ) -> StoredSurfaceCoverageGeneration:
        path = observation.coverage_path
        if path is None:
            raise BladePlanningAssetError(
                "The current stopped perception cycle has no fine-coverage asset"
            )
        try:
            state = self._coverage_reader(str(path))
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Cannot verify the fine-coverage generation: {exc}"
            ) from exc
        if type(state) is not StoredSurfaceCoverageGeneration:
            raise BladePlanningAssetError(
                "Fine-coverage reader returned an untyped generation"
            )
        if state.root != path.resolve():
            raise BladePlanningAssetError(
                "Fine-coverage reader resolved a different generation path"
            )
        reference_sha256 = str(state.metadata["reference"]["metadata_sha256"])
        if (
            state.reference.root != self._expected_reference_root
            or reference_sha256 != self._expected_reference_sha256
        ):
            raise BladePlanningAssetError(
                "Fine coverage does not use the selector's pinned coarse reference"
            )
        if (
            state.quality_config.model_dump(mode="json")
            != self._surface_quality_config.model_dump(mode="json")
        ):
            raise BladePlanningAssetError(
                "Fine-coverage quality thresholds differ from the selector policy"
            )
        current = state.current_reconstructed_view_path
        observed = observation.reconstructed_view_path
        candidate_ids = self._candidate_capture_ids(state)
        if state.ledger.observation_ids:
            if current is None:
                raise BladePlanningAssetError(
                    "Non-empty fine coverage lacks its current reconstructed view"
                )
            if observed is not None:
                if current != observed:
                    raise BladePlanningAssetError(
                        "Fine coverage is not bound to this cycle's reconstructed view"
                    )
                self._validate_current_reconstruction(observation, state)
        elif current is not None:
            raise BladePlanningAssetError(
                "An empty fine ledger cannot cite a reconstructed observation"
            )
        elif observed is not None:
            raise BladePlanningAssetError(
                "This cycle's reconstructed view has no matching fine-coverage successor"
            )
        if observed is None and observation.bundle.view_id in candidate_ids:
            raise BladePlanningAssetError(
                "A captured planned fine candidate lacks its reconstructed view and "
                "fine-coverage successor"
            )
        return state

    def _candidate_capture_ids(
        self,
        state: StoredSurfaceCoverageGeneration,
    ) -> set[str]:
        base_ids = tuple(candidate.view_id for candidate in state.view_plan.candidates)
        generated_ids = tuple(
            _reacquisition_view_id(candidate, attempt, self._policy_sha256)
            for candidate in state.view_plan.candidates
            for attempt in range(
                1,
                self._selection_config.maximum_reacquisition_attempts_per_patch + 1,
            )
        )
        all_ids = (*base_ids, *generated_ids)
        if any(not value.strip() for value in base_ids) or len(set(all_ids)) != len(all_ids):
            raise BladePlanningAssetError(
                "Fine-view and reacquisition candidate IDs are not globally unique"
            )
        return set(all_ids)

    @staticmethod
    def _reacquisition_standoff_bounds(
        state: StoredSurfaceCoverageGeneration,
    ) -> tuple[float, float]:
        try:
            config = ViewPlanningConfig.model_validate(
                state.reference.metadata["view_plan"]["configuration"]
            )
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Cannot verify coarse-reference standoff bounds for reacquisition: {exc}"
            ) from exc
        lower = config.minimum_standoff_distance_m
        upper = config.maximum_standoff_distance_m
        if lower is None or upper is None:
            raise BladePlanningAssetError(
                "Bounded reacquisition requires coarse-reference minimum and maximum standoff"
            )
        return lower, upper

    def _require_reacquisition_policy_binding(
        self,
        state: StoredSurfaceCoverageGeneration,
    ) -> None:
        record = state.metadata.get("reacquisition_policy")
        expected = {
            "id_schema": REACQUISITION_VIEW_ID_SCHEMA,
            "selection_policy_sha256": self._policy_sha256,
            "selection_policy": self._policy_payload,
        }
        if record != expected:
            raise BladePlanningAssetError(
                "Fine coverage retry policy differs from the active selector policy"
            )

    def _validate_cycle_continuity(
        self,
        observation: PerceptionCycleResult,
        state: StoredSurfaceCoverageGeneration,
    ) -> None:
        cycle_key = (
            str(observation.bundle.view_id),
            int(observation.bundle.sequence_index),
        )
        if self._last_cycle_key is None:
            return
        assert self._last_surface_root is not None
        assert self._last_surface_generation_id is not None
        if cycle_key == self._last_cycle_key:
            if (
                state.root != self._last_surface_root
                or state.generation_id != self._last_surface_generation_id
            ):
                raise BladePlanningAssetError(
                    "Repeated selection for one perception cycle changed fine coverage"
                )
            return
        if observation.reconstructed_view_path is None:
            if (
                state.root != self._last_surface_root
                or state.generation_id != self._last_surface_generation_id
            ):
                raise BladePlanningAssetError(
                    "Transit capture did not carry the exact preceding fine generation"
                )
            return
        previous = state.metadata.get("previous_generation")
        if (
            state.previous_generation_path != self._last_surface_root
            or not isinstance(previous, dict)
            or str(previous.get("generation_id"))
            != self._last_surface_generation_id
        ):
            raise BladePlanningAssetError(
                "Fine-coverage successor is not linked to the preceding generation"
            )

    def _record_cycle_binding(
        self,
        observation: PerceptionCycleResult,
        state: StoredSurfaceCoverageGeneration,
    ) -> None:
        self._last_cycle_key = (
            str(observation.bundle.view_id),
            int(observation.bundle.sequence_index),
        )
        self._last_surface_root = state.root
        self._last_surface_generation_id = state.generation_id

    @staticmethod
    def _validate_current_reconstruction(
        observation: PerceptionCycleResult,
        state: StoredSurfaceCoverageGeneration,
    ) -> None:
        path = state.current_reconstructed_view_path
        assert path is not None
        try:
            stored = read_reconstructed_view(path)
            source = stored.metadata["source"]
            session_path = type(path)(str(source["session"])).resolve()
            stereo_value = source["stereo_inference"]
            stereo_path = (
                type(path)(str(stereo_value)).resolve()
                if stereo_value is not None
                else None
            )
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Cannot verify the current reconstructed view: {exc}"
            ) from exc
        view = stored.view
        bundle = observation.bundle
        if view.depth_source != "foundation_stereo":
            raise BladePlanningAssetError(
                "Fine coverage accepts only FoundationStereo reconstructed views"
            )
        if (
            session_path != observation.raw_session_path
            or stereo_path != observation.stereo_inference_path
        ):
            raise BladePlanningAssetError(
                "Reconstructed-view sources do not match this perception cycle"
            )
        if (
            view.source_view_id != bundle.view_id
            or view.source_sequence_index != bundle.sequence_index
            or view.source_frame_number != bundle.stereo.frame_number
            or state.ledger.observation_ids[-1] != bundle.view_id
            or not np.allclose(
                view.joint_positions_rad,
                bundle.selected_robot_state.joint_positions_rad,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise BladePlanningAssetError(
                "Reconstructed-view identity or robot joints do not match the capture"
            )

    def _required_patch_ids(
        self,
        state: StoredSurfaceCoverageGeneration,
    ) -> tuple[str, ...]:
        required_regions = tuple(
            SurfaceRegion(value) for value in self._selection_config.required_regions
        )
        required_set = set(required_regions)
        surface = state.surface
        actual_regions = set(state.required_regions)
        missing_regions = required_set - actual_regions
        if missing_regions:
            names = ", ".join(sorted(region.value for region in missing_regions))
            raise BladePlanningAssetError(
                f"Coarse reference lacks configured required regions: {names}"
            )
        if self._selection_config.require_each_region_on_both_blade_sides:
            missing = [
                f"{side.value}:{region.value}"
                for side in (BladeSide.FRONT, BladeSide.BACK)
                for region in required_regions
                if not any(
                    patch.side is side and patch.region is region
                    for patch in surface.patches
                )
            ]
            if missing:
                raise BladePlanningAssetError(
                    "Coarse reference violates bilateral region coverage: "
                    + ", ".join(missing)
                )
        if self._selection_config.require_two_observed_fin_faces_per_side:
            self._validate_bilateral_fin_reference(surface)
        patch_ids = tuple(
            patch.patch_id
            for patch in surface.patches
            if patch.region in required_set
        )
        if not patch_ids or any(
            patch_id not in state.required_patch_ids for patch_id in patch_ids
        ):
            raise BladePlanningAssetError(
                "Configured required patches do not match the verified reference"
            )
        return patch_ids

    @staticmethod
    def _validate_bilateral_fin_reference(surface: CurvedBladeSurface) -> None:
        for side in (BladeSide.FRONT, BladeSide.BACK):
            component = surface.fin_component(side)
            if component is None or not component.two_faces_observed:
                raise BladePlanningAssetError(
                    f"The {side.value} fin lacks two-face coarse evidence"
                )
            face_normals = np.asarray(
                [
                    patch.main_normal
                    for patch in surface.for_side(side)
                    if patch.region is SurfaceRegion.FIN_FACE
                ],
                dtype=np.float64,
            )
            if (
                len(face_normals) < 2
                or np.min(face_normals @ component.normal_axis) >= -0.5
                or np.max(face_normals @ component.normal_axis) <= 0.5
            ):
                raise BladePlanningAssetError(
                    f"The {side.value} fin plan does not expose both physical faces"
                )

    def _fk_verified_candidates(
        self,
        candidates: tuple[EvaluatedCandidate, ...],
    ) -> tuple[tuple[EvaluatedCandidate, ...], tuple[str, ...]]:
        accepted: list[EvaluatedCandidate] = []
        rejected: list[str] = []
        flange_t_tcp = load_es68_flange_t_tcp()
        for item in candidates:
            joints = item.joint_positions_rad
            if item.status is not CandidateStatus.ENDPOINT_FEASIBLE or joints is None:
                continue
            try:
                target = _target_base_t_tcp(item.candidate, self._hand_eye)
                predicted = self._fk_model.base_t_flange(joints).compose(
                    flange_t_tcp
                )
                translation_error = float(
                    np.linalg.norm(
                        predicted.translation_m - target.translation_m
                    )
                )
                rotation_error = _rotation_distance_deg(predicted, target)
            except Exception as exc:
                rejected.append(
                    f"{item.candidate.view_id}: FK verification failed ({exc})"
                )
                continue
            if (
                translation_error
                > self._motion_config.maximum_endpoint_translation_error_m
                or rotation_error
                > self._motion_config.maximum_endpoint_rotation_error_deg
            ):
                rejected.append(
                    f"{item.candidate.view_id}: FK residual "
                    f"{translation_error:.6f} m/{rotation_error:.3f} deg"
                )
                continue
            accepted.append(item)
        return tuple(accepted), tuple(rejected)

    def _rank_key(
        self,
        item: EvaluatedCandidate,
        quality: SurfacePatchQuality,
        current_joints: NDArray[np.float64],
    ) -> tuple[object, ...]:
        priorities = {
            SurfaceRegion(value): index
            for index, value in enumerate(self._selection_config.region_priority)
        }
        coverage_deficit = 1.0 - quality.coverage_fraction
        normal_deficit = 1.0 - quality.normal_consistency
        rmse_ratio = (
            quality.rmse_m / self._surface_quality_config.maximum_rmse_m
            if np.isfinite(quality.rmse_m)
            else float("inf")
        )
        joints = item.joint_positions_rad
        assert joints is not None
        delta = np.abs(joints - current_joints)
        if self._selection_config.use_joint_travel_only_as_tiebreak:
            joint_key = (float(np.max(delta)), float(np.sum(delta)))
        else:
            joint_key = (0.0, 0.0)
        return (
            priorities[quality.region],
            -coverage_deficit,
            -normal_deficit,
            -rmse_ratio,
            -item.candidate.visibility_fraction,
            -item.candidate.projection_fraction,
            -item.metrics.geometric_score,
            *joint_key,
            item.metrics.standoff_error_m,
            item.candidate.view_id,
        )

    def select_next(
        self,
        observation: PerceptionCycleResult,
        generation: OccupancyGeneration,
    ) -> NextViewSelection:
        """Return one target or a proof-backed completion decision.

        ``generation`` is accepted to satisfy the coordinator boundary, but is not
        read here.  Occupancy can veto the chosen segment later; it can never alter
        scientific coverage, ranking, or completion.
        """

        del generation
        state = self._read_state(observation)
        required_patch_ids = self._required_patch_ids(state)
        quality_by_id = {item.patch_id: item for item in state.quality.patches}
        if set(quality_by_id) != set(state.required_patch_ids):
            raise BladePlanningAssetError(
                "Fine-quality report does not match the fixed surface reference"
            )
        self._validate_cycle_continuity(observation, state)
        self._record_cycle_binding(observation, state)
        incomplete_patch_ids = tuple(
            patch_id
            for patch_id in required_patch_ids
            if not quality_by_id[patch_id].complete
        )
        reference_sha256 = str(
            state.metadata["reference"]["metadata_sha256"]
        )
        base_diagnostics = (
            f"algorithm={_SELECTOR_ALGORITHM}",
            f"coverage_generation={state.generation_id}",
            f"fine_observations={len(state.ledger.observation_ids)}",
        )
        if not incomplete_patch_ids:
            if self._fine_finalizer is None:
                raise BladePlanningAssetError(
                    "Fine coverage passed, but no terminal reconstruction finalizer "
                    "is configured"
                )
            try:
                final = self._fine_finalizer(state)
            except Exception as exc:
                raise BladePlanningAssetError(
                    f"Fine coverage passed but terminal reconstruction failed: {exc}"
                ) from exc
            if type(final) is not FinalFineCompletionEvidence:
                raise BladePlanningAssetError(
                    "Fine terminal finalizer returned untyped completion evidence"
                )
            return NextViewSelection(
                None,
                state.generation_id,
                reference_sha256,
                self._policy_sha256,
                len(required_patch_ids),
                0,
                True,
                (*base_diagnostics, "all required fine patches passed quality gates"),
                final.root,
                final.artifact_id,
                final.metadata_sha256,
            )

        candidates_by_patch = {
            candidate.patch.patch_id: candidate
            for candidate in state.view_plan.candidates
        }
        if len(candidates_by_patch) != len(state.view_plan.candidates) or set(
            candidates_by_patch
        ) != set(state.required_patch_ids):
            raise BladePlanningAssetError(
                "Fine-view candidates do not bijectively match reference patches"
            )
        projection_by_patch = {
            candidate.patch.patch_id: pose
            for candidate, pose in zip(
                state.view_plan.candidates,
                state.view_plan.candidate_base_t_left_rectified,
                strict=True,
            )
        }
        captured = set(state.ledger.observation_ids)
        selectable: list[tuple[str, CandidateView, PoseSE3, int]] = []
        exhausted_patch_ids: list[str] = []
        retry_bounds: tuple[float, float] | None = None
        for patch_id in incomplete_patch_ids:
            candidate = candidates_by_patch[patch_id]
            projection_pose = projection_by_patch[patch_id]
            if not (
                self._selection_config.exclude_already_captured_candidate_ids
                and candidate.view_id in captured
            ):
                selectable.append((patch_id, candidate, projection_pose, 0))
                continue
            self._require_reacquisition_policy_binding(state)
            if retry_bounds is None:
                retry_bounds = self._reacquisition_standoff_bounds(state)
            retry_count = 0
            for attempt, perturbation in enumerate(
                self._selection_config.reacquisition_perturbations,
                start=1,
            ):
                retry_view_id = _reacquisition_view_id(
                    candidate,
                    attempt,
                    self._policy_sha256,
                )
                if retry_view_id in captured:
                    continue
                proposed_distance = (
                    candidate.standoff_distance_m + perturbation.distance_offset_m
                )
                if not retry_bounds[0] <= proposed_distance <= retry_bounds[1]:
                    continue
                try:
                    retry_candidate, retry_projection = generate_reacquisition_view(
                        candidate,
                        projection_pose,
                        state.view_plan.left_rectified_t_left_ir,
                        perturbation,
                        view_id=retry_view_id,
                        minimum_standoff_distance_m=retry_bounds[0],
                        maximum_standoff_distance_m=retry_bounds[1],
                    )
                except (TypeError, ValueError) as exc:
                    raise BladePlanningAssetError(
                        f"Cannot generate bounded reacquisition view for {patch_id}: {exc}"
                    ) from exc
                selectable.append(
                    (patch_id, retry_candidate, retry_projection, attempt)
                )
                retry_count += 1
            if retry_count == 0:
                exhausted_patch_ids.append(patch_id)
        if not selectable:
            raise NextViewUnavailable(
                "Fine coverage remains incomplete, but every incomplete patch exhausted "
                "its bounded reacquisition attempt budget "
                f"({self._selection_config.maximum_reacquisition_attempts_per_patch} per patch)"
            )
        current_joints = np.asarray(
            observation.inference_robot_state_trace[-1].joint_positions_rad,
            dtype=np.float64,
        )
        try:
            checker = self._reachability_factory(current_joints.copy())
        except Exception as exc:
            raise NextViewUnavailable(
                f"Cannot initialize endpoint IK from the current stopped joints: {exc}"
            ) from exc
        candidates = tuple(item[1] for item in selectable)
        projection_poses = {
            item[1].view_id: item[2]
            for item in selectable
        }
        attempt_by_view_id = {item[1].view_id: item[3] for item in selectable}
        try:
            filtered = filter_candidate_views(
                candidates,
                _surface_envelope(state.surface),
                self._view_filter_config,
                checker,
                projection_poses=projection_poses,
                deduplicate=False,
            )
        except Exception as exc:
            raise BladePlanningAssetError(
                f"Fine candidate geometry is inconsistent: {exc}"
            ) from exc
        feasible, fk_rejections = self._fk_verified_candidates(
            filtered.endpoint_feasible
        )
        if not feasible:
            filter_reasons = tuple(
                f"{item.candidate.view_id}: {'; '.join(item.reasons)}"
                for item in filtered.candidates
                if item.status is not CandidateStatus.ENDPOINT_FEASIBLE
            )
            details = (*filter_reasons, *fk_rejections)
            suffix = f"; evidence: {' | '.join(details[:8])}" if details else ""
            raise NextViewUnavailable(
                "Fine coverage is incomplete, but no unused candidate passed "
                f"geometry, workspace, IK, and FK gates{suffix}"
            )
        selected = min(
            feasible,
            key=lambda item: self._rank_key(
                item,
                quality_by_id[item.candidate.patch.patch_id],
                current_joints,
            ),
        )
        selected_quality = quality_by_id[selected.candidate.patch.patch_id]
        return NextViewSelection(
            next_view_target_from_candidate(selected, self._hand_eye),
            state.generation_id,
            reference_sha256,
            self._policy_sha256,
            len(required_patch_ids),
            len(incomplete_patch_ids),
            False,
            (
                *base_diagnostics,
                f"selected_patch={selected_quality.patch_id}",
                f"selected_region={selected_quality.region.value}",
                f"selected_side={selected_quality.side.value}",
                f"coverage_fraction={selected_quality.coverage_fraction:.6f}",
                f"reacquisition_attempt={attempt_by_view_id[selected.candidate.view_id]}",
                f"exhausted_patch_count={len(exhausted_patch_ids)}",
                "occupancy is reserved exclusively for downstream segment safety",
            ),
        )
