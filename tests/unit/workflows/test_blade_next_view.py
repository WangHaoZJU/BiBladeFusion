from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import biblade_fusion.workflows.blade_next_view as blade_next_view_module
from biblade_fusion.calibration import HandEyeCalibration
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AxisAlignedBoxConfig,
    KinematicsConfig,
    MotionPreflightConfig,
    NextViewSelectionConfig,
    SurfaceQualityConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.robot.base import RobotState
from biblade_fusion.perception.features import FinComponent
from biblade_fusion.perception.surface import (
    CurvedBladeSurface,
    CurvedSurfacePatch,
    CurvedViewPlan,
    SurfaceRegion,
    generate_reacquisition_view,
)
from biblade_fusion.planning import ReachabilityResult, ReachabilityState
from biblade_fusion.planning.surface_coverage import (
    SurfaceCoverageLedger,
    SurfacePatchEvidence,
    evaluate_surface_quality,
)
from biblade_fusion.planning.views import BladeSide, CandidateView, SurfacePatch
from biblade_fusion.robotics import load_es68_flange_t_tcp
from biblade_fusion.storage.coarse_model import StoredCoarseModelSummary
from biblade_fusion.storage.surface_coverage import (
    REACQUISITION_VIEW_ID_SCHEMA,
    StoredSurfaceCoverageGeneration,
)
from biblade_fusion.workflows.blade_next_view import (
    BladeCoverageNextViewSelector,
    _reacquisition_view_id,
)
from biblade_fusion.workflows.fine_completion import FinalFineCompletionEvidence
from biblade_fusion.workflows.stop_scan_coordinator import (
    BladePlanningAssetError,
    NextViewUnavailable,
)


def test_production_selector_factory_requires_science_authority(tmp_path: Path) -> None:
    with pytest.raises(BladePlanningAssetError, match="requires a science acceptance"):
        BladeCoverageNextViewSelector.from_settings(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            reference_coarse_model=tmp_path / "not-read",
            science_authority=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("experimental", "expected"),
    ((False, "accepted"), (True, "unaccepted")),
)
def test_selector_factory_binds_the_correct_terminal_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experimental: bool,
    expected: str,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "metadata.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        blade_next_view_module,
        "read_coarse_model_summary",
        lambda _path: SimpleNamespace(metadata={"schema_version": 5}),
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        blade_next_view_module,
        "finalize_fine_science",
        lambda state, **_kwargs: calls.append(("accepted", state)) or "accepted",
    )
    monkeypatch.setattr(
        blade_next_view_module,
        "finalize_unaccepted_fine_science",
        lambda state, **_kwargs: calls.append(("unaccepted", state)) or "unaccepted",
    )
    captured: dict[str, object] = {}

    def fake_init(_self, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(BladeCoverageNextViewSelector, "__init__", fake_init)

    BladeCoverageNextViewSelector.from_settings(
        load_settings("configs/default.yaml"),
        None,  # type: ignore[arg-type]
        reference_coarse_model=reference,
        science_authority=None if experimental else object(),  # type: ignore[arg-type]
        experimental=experimental,
    )
    finalizer = captured["fine_finalizer"]

    assert callable(finalizer)
    assert finalizer("state") == expected  # type: ignore[operator]
    assert calls == [(expected, "state")]


def _camera_rotation(outward_normal: np.ndarray) -> np.ndarray:
    camera_z = -np.asarray(outward_normal, dtype=np.float64)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ camera_z)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    camera_x = seed - camera_z * float(seed @ camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, camera_z)
    return np.column_stack((camera_x, camera_y, camera_z))


def _patch(
    patch_id: str,
    side: BladeSide,
    region: SurfaceRegion,
    *,
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    point_count: int = 36,
    row: int = 0,
) -> CurvedSurfacePatch:
    unit = np.asarray(normal, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ unit)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = seed - unit * float(seed @ unit)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    grid_size = int(np.ceil(np.sqrt(point_count)))
    values = np.linspace(-0.01, 0.01, grid_size)
    first_grid, second_grid = np.meshgrid(values, values, indexing="xy")
    coordinates = np.column_stack((first_grid.ravel(), second_grid.ravel()))[
        :point_count
    ]
    center_array = np.asarray(center, dtype=np.float64)
    points = (
        center_array
        + coordinates[:, :1] * first
        + coordinates[:, 1:] * second
    )
    return CurvedSurfacePatch(
        patch_id,
        side,
        region,
        row,
        0,
        0,
        points,
        np.tile(unit, (point_count, 1)),
        coordinates,
        center_array,
        np.column_stack((first, second, unit)),
        np.array([0.02, 0.02, 0.001]),
        unit,
        0.0,
        0.0,
    )


def _candidate(
    patch: CurvedSurfacePatch,
    *,
    calibration: PoseSE3 | None = None,
    raw_rotation: np.ndarray | None = None,
    raw_translation: np.ndarray | None = None,
) -> tuple[CandidateView, PoseSE3]:
    rectified_rotation = _camera_rotation(patch.main_normal)
    translation = patch.obb_center_m + 0.25 * patch.main_normal
    rectified = PoseSE3.from_rotation_translation(
        "base", "left_rectified", rectified_rotation, translation
    )
    if raw_rotation is None and raw_translation is None:
        raw = rectified.compose(
            calibration or PoseSE3.identity("left_rectified", "left_ir")
        )
    else:
        raw = PoseSE3.from_rotation_translation(
            "base",
            "left_ir",
            rectified_rotation if raw_rotation is None else raw_rotation,
            translation if raw_translation is None else raw_translation,
        )
    target = SurfacePatch(
        patch.patch_id,
        patch.side,
        patch.row,
        patch.column,
        patch.obb_center_m,
        patch.main_normal,
        patch.planar_extents_m,
    )
    return (
        CandidateView(
            patch.patch_id,
            target,
            raw,
            0.25,
            (0.05, 0.05),
            1.0,
            1.0,
            "unit_test",
        ),
        rectified,
    )


def _fin_component(side: BladeSide, *, two_faces: bool = True) -> FinComponent:
    sign = 1.0 if side is BladeSide.FRONT else -1.0
    x = np.linspace(-0.01, 0.01, 6)
    negative = np.column_stack((x, np.full(6, -0.001), np.full(6, 0.01 * sign)))
    positive = np.column_stack((x, np.full(6, 0.001), np.full(6, 0.01 * sign)))
    points = np.vstack((negative, positive))
    return FinComponent(
        f"{side.value}_fin",
        side,
        points,
        np.vstack(
            (
                np.tile([0.0, -1.0, 0.0], (6, 1)),
                np.tile([0.0, 1.0, 0.0], (6, 1)),
            )
        ),
        points[:, [0, 2]],
        np.full(12, 0.01),
        np.zeros(12, dtype=bool),
        np.zeros(12, dtype=bool),
        points.mean(axis=0),
        np.eye(3),
        np.array([0.02, 0.004, 0.002]),
        np.array([0.0, 1.0, 0.0]),
        0.0005,
        0.002,
        two_faces,
    )


def _surface(
    patches: tuple[CurvedSurfacePatch, ...],
    *,
    fin_components: tuple[FinComponent, ...] = (),
) -> CurvedBladeSurface:
    return CurvedBladeSurface(
        "base",
        patches,
        np.eye(3),
        np.mean(np.vstack([patch.obb_center_m for patch in patches]), axis=0),
        (0.1, 0.1),
        (0, 0),
        (1, 1),
        (0.05, 0.05),
        "configured_override",
        fin_components,
    )


def _quality_config(*, minimum_observed_points: int = 3) -> SurfaceQualityConfig:
    return SurfaceQualityConfig(
        maximum_surface_distance_m=0.002,
        minimum_incidence_cosine=0.5,
        completed_fraction=0.8,
        maximum_rmse_m=0.002,
        minimum_normal_consistency=0.8,
        minimum_observed_points=minimum_observed_points,
    )


def _stored_generation(
    tmp_path: Path,
    patches: tuple[CurvedSurfacePatch, ...],
    *,
    complete_patch_ids: frozenset[str] = frozenset(),
    quality_config: SurfaceQualityConfig | None = None,
    fin_components: tuple[FinComponent, ...] = (),
    candidate_overrides: dict[str, tuple[CandidateView, PoseSE3]] | None = None,
    calibration: PoseSE3 | None = None,
) -> StoredSurfaceCoverageGeneration:
    quality_config = quality_config or _quality_config()
    surface = _surface(patches, fin_components=fin_components)
    calibration = calibration or PoseSE3.identity("left_rectified", "left_ir")
    candidate_pairs = tuple(
        (candidate_overrides or {}).get(
            patch.patch_id,
            _candidate(patch, calibration=calibration),
        )
        for patch in patches
    )
    plan = CurvedViewPlan(
        surface,
        tuple(pair[0] for pair in candidate_pairs),
        tuple(pair[1] for pair in candidate_pairs),
        calibration,
        (0.05, 0.05),
    )
    evidence = tuple(
        SurfacePatchEvidence(
            patch.patch_id,
            (
                np.zeros(len(patch.points_m))
                if patch.patch_id in complete_patch_ids
                else np.full(len(patch.points_m), np.inf)
            ),
            (
                np.ones(len(patch.points_m))
                if patch.patch_id in complete_patch_ids
                else np.full(len(patch.points_m), -1.0)
            ),
        )
        for patch in patches
    )
    ledger = SurfaceCoverageLedger(evidence, ())
    quality = evaluate_surface_quality(ledger, surface, quality_config)
    root = (tmp_path / "coverage_generation").resolve()
    reference_root = (tmp_path / "coarse_reference").resolve()
    required_regions = tuple(dict.fromkeys(patch.region for patch in patches))
    reference_sha256 = "b" * 64
    return StoredSurfaceCoverageGeneration(
        root,
        "a" * 64,
        "c" * 64,
        StoredCoarseModelSummary(
            reference_root,
            {
                "view_plan": {
                    "configuration": ViewPlanningConfig(
                        standoff_distance_m=0.25,
                        minimum_standoff_distance_m=0.15,
                        maximum_standoff_distance_m=0.35,
                    ).model_dump(mode="json")
                }
            },
        ),
        surface,
        plan,
        ledger,
        quality,
        quality_config,
        tuple(patch.patch_id for patch in patches),
        required_regions,
        None,
        None,
        {"reference": {"metadata_sha256": reference_sha256}},
    )


def _robot_state(joints: np.ndarray) -> RobotState:
    return RobotState(
        1,
        1.0,
        joints,
        PoseSE3.identity("base", "tcp"),
        "RUNNING",
        "NORMAL",
        1.0,
    )


def _observation(
    state: StoredSurfaceCoverageGeneration,
    joints: np.ndarray | None = None,
    *,
    view_id: str = "transit_current_target",
    sequence_index: int = 0,
    reconstructed_view_path: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        bundle=SimpleNamespace(view_id=view_id, sequence_index=sequence_index),
        coverage_path=state.root,
        reconstructed_view_path=reconstructed_view_path,
        inference_robot_state_trace=(
            _robot_state(np.zeros(6) if joints is None else joints),
        ),
    )


def _hand_eye(tmp_path: Path) -> HandEyeCalibration:
    flange_t_left_ir = PoseSE3.identity("flange", "left_ir")
    tcp_t_left_ir = load_es68_flange_t_tcp().inverse().compose(flange_t_left_ir)
    return HandEyeCalibration(
        tcp_t_left_ir,
        "unit-test",
        20,
        0.001,
        0.2,
        tmp_path / "hand_eye.yaml",
        flange_t_left_ir=flange_t_left_ir,
    )


def _selection_config(
    *regions: SurfaceRegion,
    bilateral: bool = False,
    fin_faces: bool = False,
) -> NextViewSelectionConfig:
    values = tuple(region.value for region in regions)
    return NextViewSelectionConfig(
        required_regions=values,
        region_priority=values,
        require_each_region_on_both_blade_sides=bilateral,
        require_two_observed_fin_faces_per_side=fin_faces,
    )


def _workspace_filter() -> ViewFilterConfig:
    return ViewFilterConfig(
        workspace=AxisAlignedBoxConfig(
            name="unit-cell",
            minimum_m=(-2.0, -2.0, -2.0),
            maximum_m=(2.0, 2.0, 2.0),
        ),
        camera_clearance_radius_m=0.01,
    )


@dataclass
class _PoseSolution:
    matrix: np.ndarray
    joints: np.ndarray


class _ReachableChecker:
    def __init__(
        self,
        solutions: tuple[_PoseSolution, ...],
        seen_poses: list[np.ndarray],
    ) -> None:
        self._solutions = solutions
        self._seen_poses = seen_poses

    def check(self, base_t_left_ir: PoseSE3) -> ReachabilityResult:
        self._seen_poses.append(base_t_left_ir.matrix.copy())
        for item in self._solutions:
            if np.allclose(base_t_left_ir.matrix, item.matrix, rtol=0.0, atol=1e-12):
                return ReachabilityResult(
                    ReachabilityState.REACHABLE,
                    "unit-test IK solution",
                    item.joints,
                )
        return ReachabilityResult(
            ReachabilityState.UNREACHABLE,
            "unit-test pose has no IK solution",
        )


class _ReachabilityFactory:
    def __init__(
        self,
        candidates: tuple[CandidateView, ...],
        joints_by_id: dict[str, np.ndarray] | None = None,
    ) -> None:
        joints_by_id = joints_by_id or {}
        self.seeds: list[np.ndarray] = []
        self.seen_poses: list[np.ndarray] = []
        solutions: list[_PoseSolution] = []
        for index, candidate in enumerate(candidates):
            joints = np.asarray(
                joints_by_id.get(
                    candidate.view_id,
                    np.full(6, 0.01 * (index + 1)),
                ),
                dtype=np.float64,
            )
            if any(
                np.allclose(candidate.base_t_left_ir.matrix, item.matrix)
                for item in solutions
            ):
                continue
            solutions.append(_PoseSolution(candidate.base_t_left_ir.matrix.copy(), joints))
        self.solutions = tuple(solutions)

    def __call__(self, seed: np.ndarray) -> _ReachableChecker:
        self.seeds.append(np.asarray(seed, dtype=np.float64).copy())
        return _ReachableChecker(self.solutions, self.seen_poses)


class _UnreachableFactory:
    def __init__(self) -> None:
        self.seeds: list[np.ndarray] = []

    def __call__(self, seed: np.ndarray) -> Any:
        self.seeds.append(np.asarray(seed, dtype=np.float64).copy())

        class Checker:
            @staticmethod
            def check(_pose: PoseSE3) -> ReachabilityResult:
                return ReachabilityResult(
                    ReachabilityState.UNREACHABLE,
                    "unit-test endpoint is unreachable",
                )

        return Checker()


class _MappedFk:
    def __init__(
        self,
        candidates: tuple[CandidateView, ...],
        factory: _ReachabilityFactory,
    ) -> None:
        flange_t_left_ir = PoseSE3.identity("flange", "left_ir")
        self._by_joint: dict[tuple[float, ...], PoseSE3] = {}
        for candidate in candidates:
            solution = next(
                item
                for item in factory.solutions
                if np.allclose(item.matrix, candidate.base_t_left_ir.matrix)
            )
            camera = PoseSE3("base", "left_ir", candidate.base_t_left_ir.matrix)
            self._by_joint[tuple(float(value) for value in solution.joints)] = (
                camera.compose(flange_t_left_ir.inverse())
            )

    def base_t_flange(self, joint_positions_rad: np.ndarray) -> PoseSE3:
        return self._by_joint[
            tuple(float(value) for value in np.asarray(joint_positions_rad))
        ]


class _WrongFk:
    @staticmethod
    def base_t_flange(_joint_positions_rad: np.ndarray) -> PoseSE3:
        return PoseSE3.from_rotation_translation(
            "base", "flange", np.eye(3), [1.0, 1.0, 1.0]
        )


def _selector(
    tmp_path: Path,
    state: StoredSurfaceCoverageGeneration,
    selection_config: NextViewSelectionConfig,
    *,
    factory: Any,
    fk_model: Any,
    view_filter: ViewFilterConfig | None = None,
    coverage_reader: Any | None = None,
    fine_finalizer: Any | None = None,
) -> BladeCoverageNextViewSelector:
    final_root = tmp_path / "final_reconstruction"
    final_root.mkdir(exist_ok=True)
    selector = BladeCoverageNextViewSelector(
        hand_eye=_hand_eye(tmp_path),
        selection_config=selection_config,
        surface_quality_config=state.quality_config,
        view_filter_config=view_filter or _workspace_filter(),
        kinematics_config=KinematicsConfig(),
        motion_config=MotionPreflightConfig(),
        expected_reference_root=state.reference.root,
        expected_reference_sha256=str(
            state.metadata["reference"]["metadata_sha256"]
        ),
        reachability_factory=factory,
        fk_model=fk_model,
        coverage_reader=coverage_reader or (lambda _path: state),
        fine_finalizer=fine_finalizer
        or (
            lambda _state: FinalFineCompletionEvidence(
                final_root,
                "d" * 64,
                "e" * 64,
            )
        ),
    )
    state.metadata["reacquisition_policy"] = {
        "id_schema": REACQUISITION_VIEW_ID_SCHEMA,
        "selection_policy_sha256": selector.selection_policy_sha256,
        "selection_policy": selector.selection_policy_payload,
    }
    return selector


def _policy_sha(
    tmp_path: Path,
    state: StoredSurfaceCoverageGeneration,
    selection_config: NextViewSelectionConfig,
    *,
    fk_model: Any,
) -> str:
    return _selector(
        tmp_path,
        state,
        selection_config,
        factory=_UnreachableFactory(),
        fk_model=fk_model,
    ).selection_policy_sha256


def _with_patch_observations(
    state: StoredSurfaceCoverageGeneration,
    *,
    patch_id: str,
    observation_ids: tuple[str, ...],
    root: Path,
) -> StoredSurfaceCoverageGeneration:
    evidence = tuple(
        replace(
            item,
            observation_ids=(observation_ids if item.patch_id == patch_id else ()),
        )
        for item in state.ledger.evidence
    )
    return replace(
        state,
        ledger=SurfaceCoverageLedger(evidence, observation_ids),
        current_reconstructed_view_path=(root / "prior_reconstruction").resolve(),
    )


def test_region_priority_is_deterministic_and_occupancy_independent(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(-0.1, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.1, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
        _patch(
            "front_fin_root",
            BladeSide.FRONT,
            SurfaceRegion.FIN_ROOT,
            center=(0.0, 0.02, 0.02),
            normal=(0.0, 1.0, 0.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(
        candidates,
        joints_by_id={"front_fin_root": np.full(6, 1.0)},
    )
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.FIN_ROOT, SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    first = selector.select_next(_observation(state), object())
    second = selector.select_next(
        _observation(state), SimpleNamespace(generation_id="different-map")
    )

    assert first == second
    assert first.target is not None
    assert first.target.view_id == "front_fin_root"
    assert first.diagnostics[-1] == (
        "occupancy is reserved exclusively for downstream segment safety"
    )


def test_online_nbv_prefers_higher_expected_blade_information_gain(
    tmp_path: Path,
) -> None:
    low = _patch(
        "a_low_visibility",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(-0.1, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    high = _patch(
        "z_high_visibility",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.1, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    low_candidate, low_projection = _candidate(low)
    high_candidate, high_projection = _candidate(high)
    state = _stored_generation(
        tmp_path,
        (low, high),
        candidate_overrides={
            low.patch_id: (
                replace(
                    low_candidate,
                    visibility_fraction=0.2,
                    projection_fraction=0.5,
                ),
                low_projection,
            ),
            high.patch_id: (high_candidate, high_projection),
        },
    )
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == "z_high_visibility"
    assert len(decision.ranked_candidates) == 2
    assert [item.target.view_id for item in decision.ranked_candidates] == [
        "z_high_visibility",
        "a_low_visibility",
    ]
    assert decision.ranked_candidates[0].diagnostics == decision.diagnostics
    assert any(
        item.startswith("expected_scientific_gain=") for item in decision.diagnostics
    )
    assert any(item.startswith("gain_coverage_novelty=") for item in decision.diagnostics)


def test_only_complete_coverage_returns_a_targetless_decision(tmp_path: Path) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(
        tmp_path,
        patches,
        complete_patch_ids=frozenset(patch.patch_id for patch in patches),
    )

    def fail_factory(_seed: np.ndarray) -> Any:  # pragma: no cover - assertion path
        raise AssertionError("complete coverage must not initialize IK")

    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=fail_factory,
        fk_model=_WrongFk(),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.coverage_complete is True
    assert decision.target is None
    assert decision.incomplete_patch_count == 0
    assert decision.final_reconstruction_path == (
        tmp_path / "final_reconstruction"
    ).resolve()
    assert decision.final_reconstruction_id == "d" * 64
    assert decision.final_reconstruction_metadata_sha256 == "e" * 64


def test_complete_coverage_blocks_when_terminal_reconstruction_fails(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(
        tmp_path,
        patches,
        complete_patch_ids=frozenset(patch.patch_id for patch in patches),
    )

    def fail_finalization(_state: StoredSurfaceCoverageGeneration):
        raise ValueError("mesh contains a boundary loop")

    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=lambda _seed: pytest.fail("terminal failure must not initialize IK"),
        fk_model=_WrongFk(),
        fine_finalizer=fail_finalization,
    )

    with pytest.raises(BladePlanningAssetError, match="terminal reconstruction failed"):
        selector.select_next(_observation(state), object())


def test_current_stationary_joint_trace_is_the_dynamic_ik_seed(tmp_path: Path) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )
    first_seed = np.linspace(0.0, 0.05, 6)
    second_seed = np.linspace(0.2, 0.25, 6)

    selector.select_next(_observation(state, first_seed), object())
    selector.select_next(_observation(state, second_seed), object())

    np.testing.assert_array_equal(factory.seeds[0], first_seed)
    np.testing.assert_array_equal(factory.seeds[1], second_seed)


def test_incomplete_but_unreachable_raises_typed_planning_block(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    factory = _UnreachableFactory()
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_WrongFk(),
    )

    with pytest.raises(NextViewUnavailable, match="no unused candidate passed"):
        selector.select_next(_observation(state), object())


def test_incomplete_captured_patch_gets_one_unique_bounded_reacquisition_view(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
    )
    nominal = initial.view_plan.candidates[0]
    config = _selection_config(SurfaceRegion.SURFACE)
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=(nominal.view_id, "unrelated_science_view"),
        root=tmp_path,
    )
    probe_factory = _ReachabilityFactory(())
    policy_sha256 = _policy_sha(
        tmp_path,
        state,
        config,
        fk_model=_MappedFk((), probe_factory),
    )
    retry_id = _reacquisition_view_id(nominal, 1, policy_sha256)
    retry, _ = generate_reacquisition_view(
        nominal,
        initial.view_plan.candidate_base_t_left_rectified[0],
        initial.view_plan.left_rectified_t_left_ir,
        config.reacquisition_perturbations[0],
        view_id=retry_id,
        minimum_standoff_distance_m=0.15,
        maximum_standoff_distance_m=0.35,
    )
    factory = _ReachabilityFactory((retry,))
    selector = _selector(
        tmp_path,
        state,
        config,
        factory=factory,
        fk_model=_MappedFk((retry,), factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == retry_id
    assert decision.target.view_id not in state.ledger.observation_ids
    assert retry.standoff_distance_m == pytest.approx(
        nominal.standoff_distance_m
        + config.reacquisition_perturbations[0].distance_offset_m
    )
    assert "reacquisition_attempt=1" in decision.diagnostics
    assert decision.diagnostics[-1] == (
        "occupancy is reserved exclusively for downstream segment safety"
    )
    assert len(factory.seen_poses) == config.maximum_reacquisition_attempts_per_patch
    assert any(
        np.allclose(pose, retry.base_t_left_ir.matrix, rtol=0.0, atol=1e-12)
        for pose in factory.seen_poses
    )


def test_reacquisition_evaluates_later_slot_when_first_slot_has_no_ik(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
    )
    nominal = initial.view_plan.candidates[0]
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=(nominal.view_id,),
        root=tmp_path,
    )
    config = _selection_config(SurfaceRegion.SURFACE)
    probe_factory = _ReachabilityFactory(())
    policy_sha256 = _policy_sha(
        tmp_path,
        state,
        config,
        fk_model=_MappedFk((), probe_factory),
    )
    retries = tuple(
        generate_reacquisition_view(
            nominal,
            initial.view_plan.candidate_base_t_left_rectified[0],
            initial.view_plan.left_rectified_t_left_ir,
            perturbation,
            view_id=_reacquisition_view_id(nominal, attempt, policy_sha256),
            minimum_standoff_distance_m=0.15,
            maximum_standoff_distance_m=0.35,
        )[0]
        for attempt, perturbation in enumerate(
            config.reacquisition_perturbations,
            start=1,
        )
    )
    factory = _ReachabilityFactory((retries[1],))
    selector = _selector(
        tmp_path,
        state,
        config,
        factory=factory,
        fk_model=_MappedFk((retries[1],), factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == retries[1].view_id
    assert "reacquisition_attempt=2" in decision.diagnostics
    assert any(
        np.allclose(pose, retries[0].base_t_left_ir.matrix, rtol=0.0, atol=1e-12)
        for pose in factory.seen_poses
    )
    assert retries[0].view_id not in state.ledger.observation_ids


def test_reacquisition_id_is_bound_to_selection_policy(tmp_path: Path) -> None:
    state = _stored_generation(
        tmp_path,
        (
            _patch(
                "front_surface",
                BladeSide.FRONT,
                SurfaceRegion.SURFACE,
                center=(0.0, 0.0, 0.01),
                normal=(0.0, 0.0, 1.0),
            ),
            _patch(
                "back_surface",
                BladeSide.BACK,
                SurfaceRegion.SURFACE,
                center=(0.0, 0.0, -0.01),
                normal=(0.0, 0.0, -1.0),
            ),
        ),
    )
    nominal = state.view_plan.candidates[0]

    first = _reacquisition_view_id(nominal, 1, "a" * 64)
    second = _reacquisition_view_id(nominal, 1, "b" * 64)

    assert first != second
    assert first == _reacquisition_view_id(nominal, 1, "a" * 64)


def test_reacquisition_attempt_budget_exhaustion_is_finite_and_explicit(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
    )
    nominal = initial.view_plan.candidates[0]
    config = _selection_config(SurfaceRegion.SURFACE)
    policy_sha256 = _policy_sha(
        tmp_path,
        initial,
        config,
        fk_model=_WrongFk(),
    )
    consumed = (
        nominal.view_id,
        *(
            _reacquisition_view_id(nominal, attempt, policy_sha256)
            for attempt in range(
                1,
                config.maximum_reacquisition_attempts_per_patch + 1,
            )
        ),
    )
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=consumed,
        root=tmp_path,
    )

    selector = _selector(
        tmp_path,
        state,
        config,
        factory=lambda _seed: pytest.fail("exhausted retries must not initialize IK"),
        fk_model=_WrongFk(),
    )

    with pytest.raises(NextViewUnavailable, match=r"attempt budget \(3 per patch\)"):
        selector.select_next(_observation(state), object())


def test_reacquisition_skips_a_distance_outside_the_coarse_planning_interval(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
    )
    nominal = initial.view_plan.candidates[0]
    config = _selection_config(SurfaceRegion.SURFACE)
    initial = replace(
        initial,
        reference=StoredCoarseModelSummary(
            initial.reference.root,
            {
                "view_plan": {
                    "configuration": ViewPlanningConfig(
                        standoff_distance_m=0.25,
                        minimum_standoff_distance_m=0.15,
                        maximum_standoff_distance_m=0.25,
                    ).model_dump(mode="json")
                }
            },
        ),
    )
    probe_factory = _ReachabilityFactory(())
    policy_sha256 = _policy_sha(
        tmp_path,
        initial,
        config,
        fk_model=_MappedFk((), probe_factory),
    )
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=(
            nominal.view_id,
            _reacquisition_view_id(nominal, 1, policy_sha256),
        ),
        root=tmp_path,
    )
    retry_id = _reacquisition_view_id(nominal, 3, policy_sha256)
    retry, _ = generate_reacquisition_view(
        nominal,
        initial.view_plan.candidate_base_t_left_rectified[0],
        initial.view_plan.left_rectified_t_left_ir,
        config.reacquisition_perturbations[2],
        view_id=retry_id,
        minimum_standoff_distance_m=0.15,
        maximum_standoff_distance_m=0.25,
    )
    factory = _ReachabilityFactory((retry,))
    selector = _selector(
        tmp_path,
        state,
        config,
        factory=factory,
        fk_model=_MappedFk((retry,), factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == retry_id
    assert "reacquisition_attempt=3" in decision.diagnostics
    assert retry.standoff_distance_m == pytest.approx(0.23)


def test_reacquisition_candidate_still_requires_endpoint_ik(tmp_path: Path) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
    )
    nominal = initial.view_plan.candidates[0]
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=(nominal.view_id,),
        root=tmp_path,
    )
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=_UnreachableFactory(),
        fk_model=_WrongFk(),
    )

    with pytest.raises(NextViewUnavailable, match="no unused candidate passed"):
        selector.select_next(_observation(state), object())


def test_reacquisition_capture_without_coverage_successor_fails_closed(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    initial = _stored_generation(tmp_path, (front, back))
    nominal = initial.view_plan.candidates[0]
    state = _with_patch_observations(
        initial,
        patch_id=front.patch_id,
        observation_ids=(nominal.view_id,),
        root=tmp_path,
    )
    policy_sha256 = _policy_sha(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        fk_model=_WrongFk(),
    )
    retry_id = _reacquisition_view_id(nominal, 1, policy_sha256)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=lambda _seed: pytest.fail("missing science must fail before IK"),
        fk_model=_WrongFk(),
    )

    with pytest.raises(BladePlanningAssetError, match="planned fine candidate"):
        selector.select_next(
            _observation(state, view_id=retry_id, sequence_index=2),
            object(),
        )


def test_ik_pseudo_solution_is_rejected_by_authoritative_fk(tmp_path: Path) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    factory = _ReachabilityFactory(state.view_plan.candidates)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_WrongFk(),
    )

    with pytest.raises(NextViewUnavailable, match="FK residual"):
        selector.select_next(_observation(state), object())


def test_null_workspace_never_produces_a_motion_target(tmp_path: Path) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
        view_filter=ViewFilterConfig(camera_clearance_radius_m=0.01),
    )

    with pytest.raises(NextViewUnavailable, match="workspace bounds are not configured"):
        selector.select_next(_observation(state), object())


def test_small_fully_observed_patches_can_complete_absolute_point_gate(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
            point_count=9,
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
            point_count=9,
        ),
    )
    state = _stored_generation(
        tmp_path,
        patches,
        complete_patch_ids=frozenset(patch.patch_id for patch in patches),
        quality_config=_quality_config(minimum_observed_points=30),
    )
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=lambda _seed: pytest.fail("completed small patches must not run IK"),
        fk_model=_WrongFk(),
    )

    assert all(item.complete for item in state.quality.patches)
    assert selector.select_next(_observation(state), object()).coverage_complete


def test_bilateral_required_region_contract_rejects_a_missing_side(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
        _patch(
            "front_fin_face_negative",
            BladeSide.FRONT,
            SurfaceRegion.FIN_FACE,
            center=(0.0, -0.002, 0.02),
            normal=(0.0, -1.0, 0.0),
        ),
        _patch(
            "front_fin_face_positive",
            BladeSide.FRONT,
            SurfaceRegion.FIN_FACE,
            center=(0.0, 0.002, 0.02),
            normal=(0.0, 1.0, 0.0),
        ),
    )
    fins = (_fin_component(BladeSide.FRONT), _fin_component(BladeSide.BACK))
    state = _stored_generation(tmp_path, patches, fin_components=fins)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(
            SurfaceRegion.FIN_FACE,
            SurfaceRegion.SURFACE,
            bilateral=True,
            fin_faces=True,
        ),
        factory=lambda _seed: pytest.fail("invalid reference must fail before IK"),
        fk_model=_WrongFk(),
    )

    with pytest.raises(BladePlanningAssetError, match="back:fin_face"):
        selector.select_next(_observation(state), object())


def test_bilateral_fin_contract_requires_opposed_face_normals(tmp_path: Path) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
        _patch(
            "front_fin_face",
            BladeSide.FRONT,
            SurfaceRegion.FIN_FACE,
            center=(0.0, 0.002, 0.02),
            normal=(0.0, 1.0, 0.0),
        ),
        _patch(
            "back_fin_face",
            BladeSide.BACK,
            SurfaceRegion.FIN_FACE,
            center=(0.0, 0.002, -0.02),
            normal=(0.0, 1.0, 0.0),
        ),
    )
    fins = (_fin_component(BladeSide.FRONT), _fin_component(BladeSide.BACK))
    state = _stored_generation(tmp_path, patches, fin_components=fins)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(
            SurfaceRegion.FIN_FACE,
            SurfaceRegion.SURFACE,
            bilateral=True,
            fin_faces=True,
        ),
        factory=lambda _seed: pytest.fail("invalid reference must fail before IK"),
        fk_model=_WrongFk(),
    )

    with pytest.raises(BladePlanningAssetError, match="both physical faces"):
        selector.select_next(_observation(state), object())


def test_rectified_pose_drives_geometry_while_raw_pose_drives_ik_and_tcp(
    tmp_path: Path,
) -> None:
    front = _patch(
        "front_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.01),
        normal=(0.0, 0.0, 1.0),
    )
    back = _patch(
        "back_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, -0.01),
        normal=(0.0, 0.0, -1.0),
    )
    ordinary, _ = _candidate(front)
    calibration = PoseSE3.from_rotation_translation(
        "left_rectified",
        "left_ir",
        np.diag([-1.0, 1.0, -1.0]),
        np.zeros(3),
    )
    raw_candidate, rectified = _candidate(front, calibration=calibration)
    raw_back, rectified_back = _candidate(back, calibration=calibration)
    state = _stored_generation(
        tmp_path,
        (front, back),
        complete_patch_ids=frozenset({back.patch_id}),
        candidate_overrides={
            front.patch_id: (raw_candidate, rectified),
            back.patch_id: (raw_back, rectified_back),
        },
        calibration=calibration,
    )
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == front.patch_id
    np.testing.assert_allclose(factory.seen_poses[0], raw_candidate.base_t_left_ir.matrix)
    expected_tcp = PoseSE3(
        "base", "left_ir", raw_candidate.base_t_left_ir.matrix
    ).compose(PoseSE3.identity("flange", "left_ir").inverse()).compose(
        load_es68_flange_t_tcp()
    )
    np.testing.assert_allclose(decision.target.base_t_tcp_matrix, expected_tcp.matrix)
    assert not np.allclose(
        ordinary.base_t_left_ir.rotation,
        raw_candidate.base_t_left_ir.rotation,
    )


def test_duplicate_camera_poses_remain_distinct_semantic_candidates(
    tmp_path: Path,
) -> None:
    first = _patch(
        "z_surface",
        BladeSide.FRONT,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    second = _patch(
        "a_surface",
        BladeSide.BACK,
        SurfaceRegion.SURFACE,
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    state = _stored_generation(tmp_path, (first, second))
    candidates = state.view_plan.candidates
    assert np.array_equal(
        candidates[0].base_t_left_ir.matrix,
        candidates[1].base_t_left_ir.matrix,
    )
    shared_joints = np.full(6, 0.1)
    factory = _ReachabilityFactory(
        candidates,
        joints_by_id={candidate.view_id: shared_joints for candidate in candidates},
    )
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    decision = selector.select_next(_observation(state), object())

    assert decision.target is not None
    assert decision.target.view_id == "a_surface"
    assert len(factory.seen_poses) == 2


def test_transit_capture_carries_forward_nonempty_verified_coverage(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    initial = _stored_generation(tmp_path, patches)
    carried = replace(
        initial,
        ledger=SurfaceCoverageLedger(
            initial.ledger.evidence,
            ("prior_candidate",),
        ),
        current_reconstructed_view_path=(tmp_path / "prior_reconstruction").resolve(),
    )
    candidates = carried.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        carried,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    selector.select_next(
        _observation(carried, view_id="bootstrap_ready", sequence_index=6),
        object(),
    )
    decision = selector.select_next(
        _observation(
            carried,
            view_id="transit_front_surface_cycle_0007",
            sequence_index=7,
        ),
        object(),
    )

    assert decision.target is not None
    assert decision.surface_generation_id == carried.generation_id


def test_fine_target_stays_staged_across_transit_capture(tmp_path: Path) -> None:
    patches = (
        _patch(
            "a_front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(-0.1, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "b_back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.1, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    candidates = state.view_plan.candidates
    factory = _ReachabilityFactory(
        candidates,
        joints_by_id={
            "a_front_surface": np.zeros(6),
            "b_back_surface": np.ones(6),
        },
    )
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
    )

    first = selector.select_next(
        _observation(
            state,
            np.zeros(6),
            view_id="fine_transition_bootstrap_000",
            sequence_index=0,
        ),
        object(),
    )
    transit = selector.select_next(
        _observation(
            state,
            np.ones(6),
            view_id="transit_a_front_surface_cycle_0001",
            sequence_index=1,
        ),
        object(),
    )

    assert first.target is not None
    assert first.target.view_id == "a_front_surface"
    assert transit == first
    assert len(factory.seeds) == 1


def test_candidate_capture_without_science_successor_fails_closed(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    state = _stored_generation(tmp_path, patches)
    selector = _selector(
        tmp_path,
        state,
        _selection_config(SurfaceRegion.SURFACE),
        factory=lambda _seed: pytest.fail("missing science assets must fail before IK"),
        fk_model=_WrongFk(),
    )

    with pytest.raises(BladePlanningAssetError, match="lacks its reconstructed view"):
        selector.select_next(
            _observation(state, view_id="front_surface"),
            object(),
        )


def test_transit_capture_rejects_a_cross_wired_valid_generation(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    first = _stored_generation(tmp_path, patches)
    other_root = (tmp_path / "cross_wired_generation").resolve()
    other_root.mkdir()
    other = replace(
        first,
        root=other_root,
        generation_id="d" * 64,
    )
    current = [first]
    candidates = first.view_plan.candidates
    factory = _ReachabilityFactory(candidates)
    selector = _selector(
        tmp_path,
        first,
        _selection_config(SurfaceRegion.SURFACE),
        factory=factory,
        fk_model=_MappedFk(candidates, factory),
        coverage_reader=lambda _path: current[0],
    )
    selector.select_next(
        _observation(first, view_id="bootstrap_ready", sequence_index=3),
        object(),
    )
    current[0] = other

    with pytest.raises(BladePlanningAssetError, match="exact preceding fine generation"):
        selector.select_next(
            _observation(
                other,
                view_id="transit_front_surface_cycle_0004",
                sequence_index=4,
            ),
            object(),
        )


def test_first_cycle_rejects_coverage_from_an_unpinned_coarse_reference(
    tmp_path: Path,
) -> None:
    patches = (
        _patch(
            "front_surface",
            BladeSide.FRONT,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
        ),
        _patch(
            "back_surface",
            BladeSide.BACK,
            SurfaceRegion.SURFACE,
            center=(0.0, 0.0, -0.01),
            normal=(0.0, 0.0, -1.0),
        ),
    )
    expected = _stored_generation(tmp_path, patches)
    wrong_root = (tmp_path / "another_blade_reference").resolve()
    wrong = replace(
        expected,
        reference=StoredCoarseModelSummary(wrong_root, {}),
        metadata={"reference": {"metadata_sha256": "e" * 64}},
    )
    selector = _selector(
        tmp_path,
        expected,
        _selection_config(SurfaceRegion.SURFACE),
        factory=lambda _seed: pytest.fail("wrong reference must fail before IK"),
        fk_model=_WrongFk(),
        coverage_reader=lambda _path: wrong,
    )

    with pytest.raises(BladePlanningAssetError, match="pinned coarse reference"):
        selector.select_next(_observation(wrong), object())
