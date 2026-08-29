"""Transactional fine-scan mask, reconstruction, and coverage preparation.

This module deliberately has no mutable "latest" state.  It materialises candidate
assets from an explicitly supplied, already accepted coverage generation.  The
FoundationStereo cycle engine decides whether that prepared successor is committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AppSettings
from biblade_fusion.mapping import OccupancyMapState
from biblade_fusion.perception.blade_foreground import (
    reference_guided_blade_mask,
)
from biblade_fusion.storage.blade_foreground import (
    read_blade_foreground_mask,
    write_blade_foreground_mask,
)
from biblade_fusion.storage.coarse_model import (
    COARSE_MODEL_SCHEMA_VERSION,
    read_coarse_model_summary,
)
from biblade_fusion.storage.reconstructed_view import (
    SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION,
    read_reconstructed_view,
    write_reconstructed_view,
)
from biblade_fusion.storage.surface_coverage import (
    REACQUISITION_VIEW_ID_SCHEMA,
    FineReacquisitionProvenance,
    StoredSurfaceCoverageGeneration,
    reacquisition_view_id,
    read_surface_coverage_generation,
    write_surface_coverage_generation,
)
from biblade_fusion.workflows.blade_next_view import production_selection_policy_payload
from biblade_fusion.workflows.occupancy_mapping import OccupancyFrameUpdate
from biblade_fusion.workflows.reconstruction import (
    reconstruct_foundation_stereo_view,
)
from biblade_fusion.workflows.stereo_inference import StereoInferenceObservation
from biblade_fusion.workflows.stop_scan_coordinator import (
    CapturedStopScanView,
    CapturePurpose,
)


class FineSciencePreparationError(ValueError):
    """One stopped cycle cannot produce a trustworthy fine-science transaction."""


@dataclass(frozen=True, slots=True)
class PreparedFineScienceAssets:
    """Science paths proposed by one cycle but not yet accepted by the engine."""

    blade_foreground_path: Path | None
    reconstructed_view_path: Path | None
    coverage_path: Path | None
    advances_coverage: bool

    def __post_init__(self) -> None:
        for name in (
            "blade_foreground_path",
            "reconstructed_view_path",
            "coverage_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).resolve())
        if (self.blade_foreground_path is None) != (self.reconstructed_view_path is None):
            raise ValueError("A fine reconstruction and its foreground mask are inseparable")
        if self.reconstructed_view_path is not None and self.coverage_path is None:
            raise ValueError("A fine reconstruction requires a coverage successor")
        if self.advances_coverage and self.coverage_path is None:
            raise ValueError("A coverage advance requires a prepared generation")


def validate_fine_science_startup(
    settings: AppSettings,
    hand_eye: HandEyeCalibration,
    *,
    reference_coarse_model: str | Path,
    accepted_coverage_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Pin one reference and optional explicit recovery generation before capture."""

    if not settings.blade_foreground.enabled:
        raise FineSciencePreparationError("Fine-science foreground extraction is disabled")
    reference_root = _verified_reference(Path(reference_coarse_model).resolve())
    policy_payload = production_selection_policy_payload(
        settings,
        hand_eye,
        reference_coarse_model=reference_root,
    )
    accepted_root = None
    if accepted_coverage_path is not None:
        accepted_root = _verified_accepted_coverage(
            Path(accepted_coverage_path).resolve(),
            reference_root=reference_root,
            settings=settings,
            selection_policy_payload=policy_payload,
        ).root
    return reference_root, accepted_root


def _verified_reference(path: Path) -> Path:
    reference = read_coarse_model_summary(path)
    if int(reference.metadata["schema_version"]) != COARSE_MODEL_SCHEMA_VERSION:
        raise FineSciencePreparationError(
            "Fine science requires an exact schema-5 coarse-model reference"
        )
    return reference.root.resolve()


def _verified_accepted_coverage(
    path: Path,
    *,
    reference_root: Path,
    settings: AppSettings,
    selection_policy_payload: dict[str, object],
) -> StoredSurfaceCoverageGeneration:
    try:
        state = read_surface_coverage_generation(
            path,
            require_foreground_bound_science=True,
        )
    except ValueError as exc:
        raise FineSciencePreparationError(
            "Accepted online fine-coverage lineage failed foreground-bound schema-3 verification"
        ) from exc
    if state.root != path.resolve() or state.reference.root != reference_root:
        raise FineSciencePreparationError(
            "Accepted fine coverage uses a different coarse-model reference"
        )
    if state.quality_config.model_dump(mode="json") != settings.surface_quality.model_dump(
        mode="json"
    ):
        raise FineSciencePreparationError("Accepted fine coverage uses a different quality policy")
    expected_policy = _selection_policy_record(selection_policy_payload)
    if state.metadata.get("reacquisition_policy") != expected_policy:
        raise FineSciencePreparationError(
            "Accepted fine coverage uses a different next-view/reacquisition policy"
        )
    return state


def _selection_policy_record(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "id_schema": REACQUISITION_VIEW_ID_SCHEMA,
        "selection_policy_sha256": digest,
        "selection_policy": canonical,
    }


def _candidate_target_authority(
    state: StoredSurfaceCoverageGeneration,
    view_id: str,
    *,
    settings: AppSettings,
    selection_policy_payload: dict[str, object],
) -> tuple[str, FineReacquisitionProvenance | None]:
    matches = tuple(
        candidate
        for candidate in state.view_plan.candidates
        if candidate.view_id == view_id
    )
    if len(matches) == 1:
        if view_id in state.ledger.observation_ids:
            raise FineSciencePreparationError(f"Fine candidate {view_id!r} was already committed")
        return matches[0].patch.patch_id, None
    if matches:
        raise FineSciencePreparationError(
            f"Capture {view_id!r} is not one unique fixed-reference candidate"
        )
    if view_id in state.ledger.observation_ids:
        raise FineSciencePreparationError(f"Fine candidate {view_id!r} was already committed")
    policy = _selection_policy_record(selection_policy_payload)
    if state.metadata.get("reacquisition_policy") != policy:
        raise FineSciencePreparationError("Fine retry policy differs from the accepted lineage")
    policy_sha256 = str(policy["selection_policy_sha256"])
    planning = settings.view_planning
    try:
        stored_planning = type(planning).model_validate(
            state.reference.metadata["view_plan"]["configuration"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FineSciencePreparationError(
            f"Cannot replay coarse-reference retry bounds: {exc}"
        ) from exc
    lower = stored_planning.minimum_standoff_distance_m
    upper = stored_planning.maximum_standoff_distance_m
    if lower is None or upper is None:
        raise FineSciencePreparationError("Fine retry requires bounded coarse standoff limits")
    quality_by_id = {item.patch_id: item for item in state.quality.patches}
    resolved: list[FineReacquisitionProvenance] = []
    for nominal in state.view_plan.candidates:
        for attempt, perturbation in enumerate(
            settings.next_view_selection.reacquisition_perturbations,
            start=1,
        ):
            expected_id = reacquisition_view_id(
                nominal.view_id,
                nominal.patch.patch_id,
                attempt,
                policy_sha256,
            )
            if expected_id != view_id:
                continue
            if nominal.view_id not in state.ledger.observation_ids:
                raise FineSciencePreparationError(
                    "Fine retry cannot precede its nominal patch capture"
                )
            quality = quality_by_id.get(nominal.patch.patch_id)
            if quality is None or quality.complete:
                raise FineSciencePreparationError(
                    "Fine retry target is not an incomplete fixed-reference patch"
                )
            proposed = nominal.standoff_distance_m + perturbation.distance_offset_m
            if not lower <= proposed <= upper:
                raise FineSciencePreparationError(
                    "Fine retry distance is outside the coarse planning interval"
                )
            resolved.append(
                FineReacquisitionProvenance(
                    view_id=expected_id,
                    nominal_candidate_id=nominal.view_id,
                    patch_id=nominal.patch.patch_id,
                    attempt=attempt,
                    distance_offset_m=perturbation.distance_offset_m,
                    tilt_deg=perturbation.tilt_deg,
                    azimuth_deg=perturbation.azimuth_deg,
                    selection_policy_sha256=policy_sha256,
                    reference_metadata_sha256=str(
                        state.metadata["reference"]["metadata_sha256"]
                    ),
                )
            )
    if len(resolved) != 1:
        raise FineSciencePreparationError(
            f"Capture {view_id!r} is not one replayable bounded fine retry"
        )
    return resolved[0].patch_id, resolved[0]


def _verify_prepared_candidate(
    assets: PreparedFineScienceAssets,
    *,
    captured: CapturedStopScanView,
    stereo_path: Path,
    previous: StoredSurfaceCoverageGeneration,
    expected_reacquisition: FineReacquisitionProvenance | None,
) -> None:
    assert assets.blade_foreground_path is not None
    assert assets.reconstructed_view_path is not None
    assert assets.coverage_path is not None
    stored_mask = read_blade_foreground_mask(assets.blade_foreground_path)
    stored_view = read_reconstructed_view(assets.reconstructed_view_path)
    successor = read_surface_coverage_generation(
        assets.coverage_path,
        require_foreground_bound_science=True,
    )
    view = stored_view.view
    identity = stored_mask.metadata["identity"]
    if (
        int(stored_view.metadata["schema_version"]) != SCIENCE_RECONSTRUCTED_VIEW_SCHEMA_VERSION
        or str(identity["view_id"]) != captured.bundle.view_id
        or int(identity["sequence_index"]) != captured.bundle.sequence_index
        or int(identity["frame_number"]) != captured.bundle.stereo.frame_number
        or view.source_view_id != captured.bundle.view_id
        or view.source_sequence_index != captured.bundle.sequence_index
        or view.source_frame_number != captured.bundle.stereo.frame_number
        or not np.array_equal(stored_mask.result.mask, stored_view.blade_mask)
    ):
        raise FineSciencePreparationError(
            "Prepared mask/reconstruction identity differs from the capture"
        )
    source = stored_view.metadata["source"]
    if (
        Path(str(source["session"])).resolve() != captured.raw_session_path
        or Path(str(source["stereo_inference"])).resolve() != stereo_path.resolve()
        or successor.previous_generation_path != previous.root
        or successor.current_reconstructed_view_path != assets.reconstructed_view_path
        or successor.ledger.observation_ids[-1] != captured.bundle.view_id
        or len(successor.ledger.observation_ids) != len(previous.ledger.observation_ids) + 1
        or successor.current_reacquisition != expected_reacquisition
    ):
        raise FineSciencePreparationError(
            "Prepared reconstruction/coverage lineage is inconsistent"
        )


def prepare_fine_science_assets(
    *,
    purpose: CapturePurpose,
    captured: CapturedStopScanView,
    stereo: StereoInferenceObservation,
    stereo_path: str | Path,
    occupancy_update: OccupancyFrameUpdate,
    occupancy_path: str | Path,
    settings: AppSettings,
    hand_eye: HandEyeCalibration,
    reference_coarse_model: str | Path,
    accepted_coverage_path: str | Path | None,
) -> PreparedFineScienceAssets:
    """Materialise the exact science branch for one coordinator-assigned purpose."""

    if type(purpose) is not CapturePurpose or captured.purpose is not purpose:
        raise FineSciencePreparationError("Fine-science capture purpose is not authoritative")
    if not settings.blade_foreground.enabled:
        raise FineSciencePreparationError("Fine-science foreground extraction is disabled")
    reference_root = _verified_reference(Path(reference_coarse_model).resolve())
    selection_policy_payload = production_selection_policy_payload(
        settings,
        hand_eye,
        reference_coarse_model=reference_root,
    )
    stereo_root = Path(stereo_path).resolve()
    occupancy_root = Path(occupancy_path).resolve()
    accepted = (
        _verified_accepted_coverage(
            Path(accepted_coverage_path).resolve(),
            reference_root=reference_root,
            settings=settings,
            selection_policy_payload=selection_policy_payload,
        )
        if accepted_coverage_path is not None
        else None
    )

    if purpose in {CapturePurpose.BOOTSTRAP, CapturePurpose.SAFETY_REFRESH}:
        # Safety-map recovery must never be blocked by the optional science branch.
        # On a resumed run the explicitly pinned generation is carried unchanged,
        # including across the coordinator's mandatory bootstrap observations.  On
        # a new run no scientific generation exists until one stopped observation
        # has produced a fresh MAP_READY safety map.
        if accepted is not None:
            return PreparedFineScienceAssets(None, None, accepted.root, False)
        if occupancy_update.snapshot.map_state is not OccupancyMapState.MAP_READY:
            return PreparedFineScienceAssets(None, None, None, False)
        coverage_path = captured.cycle_root / "surface_coverage"
        write_surface_coverage_generation(
            coverage_path,
            reference_coarse_model=reference_root,
            quality_config=settings.surface_quality,
            selection_policy_payload=selection_policy_payload,
        )
        initial = _verified_accepted_coverage(
            coverage_path,
            reference_root=reference_root,
            settings=settings,
            selection_policy_payload=selection_policy_payload,
        )
        if initial.ledger.observation_ids or initial.current_reconstructed_view_path is not None:
            raise FineSciencePreparationError("Initial fine coverage is not empty")
        return PreparedFineScienceAssets(None, None, initial.root, True)

    if purpose is CapturePurpose.TRANSIT:
        if accepted is None:
            raise FineSciencePreparationError(
                f"{purpose.value} capture has no accepted fine generation to carry"
            )
        return PreparedFineScienceAssets(None, None, accepted.root, False)

    if purpose is not CapturePurpose.CANDIDATE:
        raise FineSciencePreparationError(f"Unsupported capture purpose: {purpose.value}")
    if accepted is None:
        raise FineSciencePreparationError(
            "A fine candidate requires an accepted predecessor generation"
        )
    if occupancy_update.snapshot.map_state is not OccupancyMapState.MAP_READY:
        raise FineSciencePreparationError(
            "A fine candidate requires a fresh MAP_READY occupancy result"
        )
    target_patch_id, reacquisition = _candidate_target_authority(
        accepted,
        captured.bundle.view_id,
        settings=settings,
        selection_policy_payload=selection_policy_payload,
    )
    base_t_left_rectified = PoseSE3(
        "base",
        "left_rectified",
        occupancy_update.evidence.base_t_camera_matrix,
    )
    mask = reference_guided_blade_mask(
        stereo.depth_m,
        occupancy_update.integration_valid_mask,
        stereo.rectified.calibration.left,
        base_t_left_rectified,
        accepted.surface,
        target_patch_id,
        settings.blade_foreground,
    )
    mask_path = captured.cycle_root / "blade_foreground"
    write_blade_foreground_mask(
        mask_path,
        mask,
        view_id=captured.bundle.view_id,
        sequence_index=captured.bundle.sequence_index,
        frame_number=captured.bundle.stereo.frame_number,
        base_t_left_rectified=base_t_left_rectified,
        intrinsics=stereo.rectified.calibration.left,
        source_session=captured.raw_session_path,
        source_stereo_inference=stereo_root,
        source_occupancy_mapping=occupancy_root,
        reference_coarse_model=reference_root,
        source_integration_valid_mask_hash=(
            occupancy_update.evidence.integration_valid_mask_content_hash
        ),
        target_patch_id=target_patch_id,
    )
    reconstructed = reconstruct_foundation_stereo_view(
        captured.bundle,
        stereo,
        mask.mask,
        hand_eye,
        settings.point_cloud,
        kinematics_config=settings.kinematics,
        hand_eye_config=settings.hand_eye,
    )
    reconstructed_path = captured.cycle_root / "reconstructed_view"
    write_reconstructed_view(
        reconstructed_path,
        reconstructed,
        mask.mask,
        hand_eye,
        settings.point_cloud,
        settings.kinematics,
        settings.hand_eye,
        source_session=captured.raw_session_path,
        source_stereo_inference=stereo_root,
        source_blade_foreground_mask=mask_path,
    )
    coverage_path = captured.cycle_root / "surface_coverage"
    write_surface_coverage_generation(
        coverage_path,
        reference_coarse_model=reference_root,
        quality_config=settings.surface_quality,
        previous_generation=accepted.root,
        current_reconstructed_view=reconstructed_path,
        observation_id=captured.bundle.view_id,
        selection_policy_payload=selection_policy_payload,
        current_reacquisition=reacquisition,
    )
    assets = PreparedFineScienceAssets(
        mask_path,
        reconstructed_path,
        coverage_path,
        True,
    )
    _verify_prepared_candidate(
        assets,
        captured=captured,
        stereo_path=stereo_root,
        previous=accepted,
        expected_reacquisition=reacquisition,
    )
    return assets
