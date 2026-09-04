from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.unknown_blade_coarse as coarse_module
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AdaptiveIkViewSearchConfig,
    AxisAlignedBoxConfig,
    CoarseReachabilityFallbackConfig,
    CoverageConfig,
    PairedFinDiscoveryFallbackConfig,
    PointCloudConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.bootstrap_foreground import BootstrapSeed
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy, select_proxy_support
from biblade_fusion.planning import (
    CandidateMetrics,
    CandidateStatus,
    CandidateView,
    EndpointConfigurationCheck,
    EvaluatedCandidate,
    FilteredViewPlan,
    ReachabilityResult,
    ReachabilityState,
    SurfacePatch,
)
from biblade_fusion.planning.coverage import CoverageLedger, PatchCoverage
from biblade_fusion.storage.coarse_scan import StoredCoarseScanView
from biblade_fusion.storage.reconstructed_view import StoredReconstructedBladeView
from biblade_fusion.workflows.reconstruction import ReconstructedBladeView
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarseDiscoveryPlan,
    CoarseSciencePolicy,
    CoarseScienceSession,
    PreparedCoarseScienceView,
    UnknownBladeCoarseError,
    _resolve_operator_bootstrap_side,
    generate_fin_discovery_plan,
    select_coarse_next_view,
)


class _Reachable:
    def check(self, _pose: PoseSE3) -> ReachabilityResult:
        return ReachabilityResult(ReachabilityState.REACHABLE, "ok", np.zeros(6))


class _Unreachable:
    def check(self, _pose: PoseSE3) -> ReachabilityResult:
        return ReachabilityResult(ReachabilityState.UNREACHABLE, "no endpoint IK")


def _proxy() -> BilateralBladeProxy:
    return BilateralBladeProxy(
        PoseSE3.from_rotation_translation(
            "base",
            "proxy",
            np.eye(3),
            (0.0, 0.0, 0.60),
        ),
        np.asarray((0.40, 0.20, 0.010)),
        np.asarray((0.0, 0.0, 0.60)),
        np.asarray((0.04, 0.01, 0.0001)),
        200,
        190,
        150,
        1.0,
    )


def _attempt_09_proxy() -> BilateralBladeProxy:
    """Proxy persisted by the real 2026-09-01 attempt-09 first view."""

    return BilateralBladeProxy(
        PoseSE3(
            "base",
            "proxy",
            np.asarray(
                [
                    [0.0069772871, 0.8372570901, -0.5467649244, 0.5541443888],
                    [-0.0548605531, 0.5462752589, 0.8358071914, 0.0833355409],
                    [0.9984696476, 0.0241641594, 0.0497439064, 0.1910914094],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ),
        np.asarray((0.32, 0.15, 0.0786093025)),
        np.asarray((0.5523603162, 0.0840882321, 0.2062544096)),
        np.asarray((0.0067804791, 0.0010448889, 0.0000533712)),
        50483,
        50483,
        10252,
        0.7855682271,
    )


def test_fin_discovery_generates_two_opposing_axes_on_both_sides() -> None:
    policy = CoarseSciencePolicy(discovery_tilt_deg=15.0)
    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        ViewPlanningConfig(standoff_distance_m=0.30),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -0.5),
                maximum_m=(1.0, 1.0, 1.5),
            ),
            minimum_look_at_cosine=0.999,
            minimum_incidence_cosine=0.95,
            maximum_standoff_error_m=1e-6,
        ),
        policy,
        _Reachable(),
    )

    assert len(result.filtered.candidates) == 8
    assert len(result.endpoint_feasible) == 8
    identifiers = {item.candidate.view_id for item in result.endpoint_feasible}
    for side in ("front", "back"):
        for axis in ("major", "minor"):
            assert f"{side}_fin_discovery_{axis}_negative" in identifiers
            assert f"{side}_fin_discovery_{axis}_positive" in identifiers
    for item in result.endpoint_feasible:
        assert item.candidate.distance_policy == "proxy_fin_discovery_oblique"
        assert np.isclose(item.metrics.incidence_cosine, np.cos(np.deg2rad(15.0)))
        assert item.status.value == "endpoint_feasible"
    assert result.motion_authorized is False
    assert len(result.policy_sha256) == 64


def test_nonadaptive_fin_discovery_rejects_colliding_ik_endpoint() -> None:
    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        ViewPlanningConfig(standoff_distance_m=0.30),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -0.5),
                maximum_m=(1.0, 1.0, 1.5),
            ),
            minimum_incidence_cosine=0.95,
        ),
        CoarseSciencePolicy(),
        _Reachable(),
        current_joint_positions_rad=(0.0,) * 6,
        endpoint_validator=lambda _joints: EndpointConfigurationCheck(
            False,
            ("self_collision:forearm:camera",),
        ),
    )

    assert not result.endpoint_feasible
    assert all(item.status is CandidateStatus.REJECTED for item in result.filtered.candidates)
    assert all(
        any("self_collision:forearm:camera" in reason for reason in item.reasons)
        for item in result.filtered.candidates
    )


def test_adaptive_fin_discovery_keeps_opposing_semantics_without_locking_tilt() -> None:
    planning = ViewPlanningConfig(
        standoff_distance_m=0.30,
        adaptive_ik_view_search=AdaptiveIkViewSearchConfig(
            enabled=True,
            maximum_distance_expansions=0,
            roll_samples_deg=(0.0,),
            maximum_ik_feasible_candidates=1,
        ),
    )
    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        planning,
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="required_outer_workspace",
                minimum_m=(-1.0, -1.0, -1.0),
                maximum_m=(1.0, 1.0, 1.0),
            ),
            camera_clearance_radius_m=0.01,
            minimum_incidence_cosine=0.4,
        ),
        CoarseSciencePolicy(
            discovery_tilt_deg=15.0,
            discovery_tilt_samples_deg=(10.0, 30.0, 45.0, 60.0),
        ),
        _Reachable(),
        PointCloudConfig(minimum_depth_m=0.15, maximum_depth_m=1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    assert len(result.adaptive_searches) == 8
    assert len(result.endpoint_feasible) == 8
    assert all(
        item.candidate.view_id.endswith("_adaptive_0000")
        for item in result.endpoint_feasible
    )
    assert all(
        trace.result.recommended is not None
        and trace.result.recommended.parameters.tilt_deg == 45.0
        for trace in result.adaptive_searches
    )
    for side in (coarse_module.BladeSide.FRONT, coarse_module.BladeSide.BACK):
        assert len(coarse_module._paired_discovery_ids(result, side)) == 2  # noqa: SLF001
    assert result.motion_authorized is False


def test_adaptive_fin_discovery_preserves_bounded_same_semantics_fallbacks() -> None:
    planning = ViewPlanningConfig(
        standoff_distance_m=0.30,
        adaptive_ik_view_search=AdaptiveIkViewSearchConfig(
            enabled=True,
            maximum_distance_expansions=1,
            roll_samples_deg=(0.0, 45.0),
            maximum_ik_feasible_candidates=2,
        ),
    )

    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        planning,
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -1.0),
                maximum_m=(1.0, 1.0, 1.0),
            ),
            camera_clearance_radius_m=0.01,
            minimum_incidence_cosine=0.4,
        ),
        CoarseSciencePolicy(),
        _Reachable(),
        PointCloudConfig(minimum_depth_m=0.15, maximum_depth_m=1.0),
        (0.0,) * 6,
    )

    assert len(result.endpoint_feasible) == 16
    assert all(
        len(
            [
                item
                for item in result.endpoint_feasible
                if item.candidate.view_id.startswith(trace.result.nominal_view_id)
            ]
        )
        == 2
        for trace in result.adaptive_searches
    )


def test_adaptive_fin_discovery_handles_a_budget_exhausted_empty_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coarse_module,
        "search_adaptive_candidate_family",
        lambda candidate, *_args, **_kwargs: coarse_module.AdaptiveViewSearchResult(
            candidate.view_id,
            (),
            (),
            True,
        ),
    )

    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        ViewPlanningConfig(
            standoff_distance_m=0.30,
            adaptive_ik_view_search=AdaptiveIkViewSearchConfig(enabled=True),
        ),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -1.0),
                maximum_m=(1.0, 1.0, 1.0),
            )
        ),
        CoarseSciencePolicy(),
        _Reachable(),
        PointCloudConfig(minimum_depth_m=0.15, maximum_depth_m=1.0),
        (0.0,) * 6,
    )

    assert len(result.adaptive_searches) == 8
    assert result.filtered.candidates == ()


def test_fin_discovery_azimuth_search_adds_asymmetric_common_bias() -> None:
    baseline = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        ViewPlanningConfig(standoff_distance_m=0.30),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -1.0),
                maximum_m=(1.0, 1.0, 1.0),
            ),
            camera_clearance_radius_m=0.01,
            minimum_incidence_cosine=0.4,
        ),
        CoarseSciencePolicy(),
        _Reachable(),
    ).filtered.candidates[0].candidate

    nominal = coarse_module._fin_discovery_azimuth_deg(baseline)  # noqa: SLF001
    samples = coarse_module._fin_discovery_azimuth_samples_deg(  # noqa: SLF001
        baseline,
        (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    )
    offsets = tuple(
        round((value - nominal + 180.0) % 360.0 - 180.0, 1)
        for value in samples
    )

    assert offsets == (0.0, -67.5, 67.5, -45.0, 45.0, -22.5, 22.5)


def test_single_initial_view_gain_selects_the_unseen_back_side(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def evaluated(view_id: str, side: coarse_module.BladeSide) -> EvaluatedCandidate:
        normal = np.array([0.0, 0.0, 1.0 if side is coarse_module.BladeSide.FRONT else -1.0])
        patch = SurfacePatch(view_id, side, 0, 0, np.zeros(3), normal, (0.1, 0.1))
        candidate = CandidateView(
            view_id,
            patch,
            PoseSE3.from_rotation_translation(
                "base",
                "left_ir",
                np.diag([1.0, -1.0, -1.0]),
                0.3 * normal,
            ),
            0.3,
            (0.1, 0.1),
            1.0,
            1.0,
            "unit_test",
        )
        return EvaluatedCandidate(
            candidate,
            CandidateStatus.ENDPOINT_FEASIBLE,
            CandidateMetrics(1.0, 1.0, 1.0, 0.3, 0.0, 0.2, 1.0),
            (),
            np.zeros(6),
        )

    discovery_items = tuple(
        evaluated(f"{side}_fin_discovery_major_{sign}", coarse_module.BladeSide(side))
        for side in ("front", "back")
        for sign in ("negative", "positive")
    )
    proxy_items = (
        evaluated("front_patch", coarse_module.BladeSide.FRONT),
        evaluated("back_patch", coarse_module.BladeSide.BACK),
    )
    coverage_config = CoverageConfig(bins_per_axis=2, minimum_points_per_bin=1)
    coverage = CoverageLedger(
        (
            PatchCoverage(
                "front_patch",
                coarse_module.BladeSide.FRONT,
                0,
                0,
                np.ones((2, 2), dtype=np.int64),
            ),
            PatchCoverage(
                "back_patch",
                coarse_module.BladeSide.BACK,
                0,
                0,
                np.zeros((2, 2), dtype=np.int64),
            ),
        ),
        (),
        coverage_config,
        1,
        1,
    )
    generation = SimpleNamespace(
        views=(
            SimpleNamespace(
                target_view_id="operator_initial",
                target_side=coarse_module.BladeSide.FRONT,
            ),
        ),
        coverage_path=tmp_path / "coverage",
        metadata={"sources": {"view_plan": {"root": str(tmp_path / "plan")}}},
    )
    monkeypatch.setattr(
        coarse_module,
        "read_coverage_ledger",
        lambda _path: SimpleNamespace(ledger=coverage),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_view_plan",
        lambda _path: SimpleNamespace(
            result=SimpleNamespace(filtered_plan=FilteredViewPlan(proxy_items, ()))
        ),
    )

    # A first stopped posture need not already solve opposing fin pairs on both
    # sides.  A useful uncovered normal view is enough to start online motion.
    provisional, provisional_gain = coarse_module._select_candidate(  # noqa: SLF001
        generation,
        CoarseDiscoveryPlan(FilteredViewPlan((), ()), "a" * 64),
        CoarseSciencePolicy(),
        require_additional_fin_evidence=False,
    )
    assert provisional.candidate.view_id == "back_patch"
    assert provisional_gain.expected_gain > 0.0

    selected, gain = coarse_module._select_candidate(  # noqa: SLF001
        generation,
        CoarseDiscoveryPlan(FilteredViewPlan(discovery_items, ()), "a" * 64),
        CoarseSciencePolicy(),
        require_additional_fin_evidence=False,
    )

    assert selected.candidate.patch.side is coarse_module.BladeSide.BACK
    assert gain.proxy_coverage_deficit == 1.0
    assert gain.side_observation_deficit == 1.0
    assert gain.expected_gain > 0.0


def _attempt_09_planning(
    *paired_fallbacks: PairedFinDiscoveryFallbackConfig,
    generic_fallback: bool = False,
) -> ViewPlanningConfig:
    return ViewPlanningConfig(
        standoff_distance_m=0.37,
        minimum_standoff_distance_m=0.32,
        maximum_standoff_distance_m=0.40,
        coarse_reachability_fallbacks=(
            (
                CoarseReachabilityFallbackConfig(
                    distance_offset_m=-0.05,
                    tilt_deg=63.4,
                    azimuth_deg=50.7,
                ),
            )
            if generic_fallback
            else ()
        ),
        paired_fin_discovery_fallbacks=paired_fallbacks,
    )


def _attempt_09_front_pair() -> PairedFinDiscoveryFallbackConfig:
    return PairedFinDiscoveryFallbackConfig(
        side="front",
        axis="major",
        distance_offset_m=-0.05,
        total_tilt_deg=63.4,
        opposing_tilt_deg=34.5,
        common_bias_sign=-1,
    )


def _attempt_09_back_pair() -> PairedFinDiscoveryFallbackConfig:
    return PairedFinDiscoveryFallbackConfig(
        side="back",
        axis="major",
        distance_offset_m=-0.05,
        total_tilt_deg=63.4,
        opposing_tilt_deg=15.0,
        common_bias_sign=-1,
    )


def _attempt_09_filter() -> ViewFilterConfig:
    return ViewFilterConfig(
        workspace=AxisAlignedBoxConfig(
            name="es68_d435i_camera_candidate",
            minimum_m=(-0.22, -0.38, -0.05),
            maximum_m=(0.65, 0.40, 0.87),
        ),
        camera_clearance_radius_m=0.05,
        minimum_look_at_cosine=0.999,
        minimum_incidence_cosine=0.40,
        maximum_standoff_error_m=0.005,
    )


def test_attempt_09_generic_coarse_fallback_is_not_reinterpreted() -> None:
    result = generate_fin_discovery_plan(
        _attempt_09_proxy(),
        (0.5835816837, 0.3280652719),
        _attempt_09_planning(generic_fallback=True),
        _attempt_09_filter(),
        CoarseSciencePolicy(),
        _Reachable(),
    )

    baseline = tuple(
        item for item in result.filtered.candidates if "_fallback_" not in item.candidate.view_id
    )
    assert len(baseline) == 8
    assert all(item.status.value == "rejected" for item in baseline)
    assert all(
        "camera leaves workspace es68_d435i_camera_candidate" in item.reasons
        for item in baseline
    )
    assert len(result.filtered.candidates) == 8
    assert not coarse_module._paired_discovery_ids(result, coarse_module.BladeSide.FRONT)
    assert not coarse_module._paired_discovery_ids(result, coarse_module.BladeSide.BACK)
    with pytest.raises(UnknownBladeCoarseError, match="exists on front"):
        coarse_module._require_bilateral_discovery_pairs(result)  # noqa: SLF001


def test_attempt_09_second_explicit_pair_candidate_fits_workspace() -> None:
    result = generate_fin_discovery_plan(
        _attempt_09_proxy(),
        (0.5835816837, 0.3280652719),
        _attempt_09_planning(_attempt_09_front_pair(), _attempt_09_back_pair()),
        _attempt_09_filter(),
        CoarseSciencePolicy(),
        _Reachable(),
    )

    filtering = _attempt_09_filter()
    assert filtering.workspace is not None
    lower = (
        np.asarray(filtering.workspace.minimum_m)
        + filtering.camera_clearance_radius_m
    )
    upper = (
        np.asarray(filtering.workspace.maximum_m)
        - filtering.camera_clearance_radius_m
    )
    for side in (coarse_module.BladeSide.FRONT, coarse_module.BladeSide.BACK):
        pairs = coarse_module._paired_discovery_ids(result, side)  # noqa: SLF001
        assert len(pairs) == 1
        for view_id in pairs[0]:
            item = next(
                candidate
                for candidate in result.endpoint_feasible
                if candidate.candidate.view_id == view_id
            )
            position = item.candidate.base_t_left_ir.translation_m
            assert np.all(position >= lower)
            assert np.all(position <= upper)
            assert item.candidate.distance_policy == (
                "explicit_paired_fin_discovery_fallback_v1"
            )
            assert item.joint_positions_rad is not None


def test_fin_discovery_without_configured_fallback_remains_fail_closed() -> None:
    result = generate_fin_discovery_plan(
        _attempt_09_proxy(),
        (0.5835816837, 0.3280652719),
        _attempt_09_planning(),
        _attempt_09_filter(),
        CoarseSciencePolicy(),
        _Reachable(),
    )

    assert len(result.filtered.candidates) == 8
    assert not result.endpoint_feasible
    with pytest.raises(UnknownBladeCoarseError, match="explicit paired fallback evaluation"):
        coarse_module._require_bilateral_discovery_pairs(result)  # noqa: SLF001


def test_bounded_fin_fallback_never_promotes_an_ik_failure() -> None:
    result = generate_fin_discovery_plan(
        _attempt_09_proxy(),
        (0.5835816837, 0.3280652719),
        _attempt_09_planning(_attempt_09_front_pair(), _attempt_09_back_pair()),
        _attempt_09_filter(),
        CoarseSciencePolicy(),
        _Unreachable(),
    )

    assert not result.endpoint_feasible
    assert all(item.status.value == "rejected" for item in result.filtered.candidates)
    assert any("_fallback_" in item.candidate.view_id for item in result.filtered.candidates)
    assert all(item.joint_positions_rad is None for item in result.filtered.candidates)


def test_fin_discovery_never_promotes_geometry_only_to_endpoint_feasible() -> None:
    result = generate_fin_discovery_plan(
        _proxy(),
        (0.30, 0.20),
        ViewPlanningConfig(standoff_distance_m=0.30),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -0.5),
                maximum_m=(1.0, 1.0, 1.5),
            ),
            minimum_incidence_cosine=0.95,
        ),
        CoarseSciencePolicy(),
        reachability_checker=None,  # type: ignore[arg-type]
    )

    assert not result.endpoint_feasible
    assert all(item.status.value == "geometry_only" for item in result.filtered.candidates)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"discovery_tilt_deg": 0.0},
        {"discovery_tilt_samples_deg": ()},
        {"discovery_tilt_samples_deg": (15.0, 15.0)},
        {"discovery_gain_surface_weight": 0.50},
        {"minimum_total_views": 4, "minimum_views_per_side": 3},
        {"maximum_attempts_per_candidate": 0},
    ),
)
def test_coarse_policy_rejects_unsafe_completion_contracts(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CoarseSciencePolicy(**kwargs)  # type: ignore[arg-type]


def test_coarse_science_session_creates_proxy_plan_and_discovery_from_first_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("model: test\n", encoding="utf-8")
    proxy = replace(
        _proxy(),
        raw_point_count=6,
        finite_point_count=6,
        voxel_point_count=6,
    )
    intrinsics = CameraIntrinsics(4, 4, 3.0, 3.0, 1.5, 1.5, "none", ())
    pixel_uv = np.asarray([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])
    cloud = PointCloud(
        "base",
        np.asarray(
            [
                (-0.10, -0.05, 0.60),
                (0.00, -0.05, 0.60),
                (0.10, -0.05, 0.60),
                (-0.10, 0.05, 0.60),
                (0.00, 0.05, 0.60),
                (0.10, 0.05, 0.60),
            ]
        ),
        pixel_uv,
        (4, 4),
    )
    view = ReconstructedBladeView(
        "operator_0",
        0,
        10,
        intrinsics,
        np.zeros(6),
        PoseSE3.identity("base", "left_ir"),
        PoseSE3.identity("base", "left_rectified"),
        cloud,
        "foundation_stereo",
    )
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[pixel_uv[:, 1], pixel_uv[:, 0]] = True
    reconstructed = StoredReconstructedBladeView(
        view,
        mask,
        {
            "source": {
                "session": str(tmp_path / "session"),
                "stereo_inference": str(tmp_path / "stereo"),
            }
        },
    )
    settings = load_settings("configs/default.yaml")
    support = select_proxy_support(
        cloud.points_m,
        settings.proxy_model,
        frame=cloud.frame,
    )
    stored_view = StoredCoarseScanView(
        (tmp_path / "coarse_view").resolve(),
        reconstructed,
        SimpleNamespace(mask=mask),
        "operator_0",
        "operator_seed",
        coarse_module.BladeSide.FRONT,
        support,
        settings.proxy_model,
        {},
        "0" * 64,
        0,
    )
    calls: list[str] = []

    monkeypatch.setattr(coarse_module, "read_coarse_scan_view", lambda _path: stored_view)
    monkeypatch.setattr(coarse_module, "build_bilateral_proxy", lambda *_args: proxy)

    def write_initialization(output: Path, *_args: object, **_kwargs: object) -> Path:
        calls.append("initialization")
        output.mkdir(parents=True)
        (output / "metadata.json").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(coarse_module, "write_initialization", write_initialization)
    planning = SimpleNamespace(
        geometric_plan=SimpleNamespace(footprint_m=(0.3, 0.2)),
        filtered_plan=FilteredViewPlan((), ()),
    )
    monkeypatch.setattr(coarse_module, "plan_initial_observation", lambda *_args: planning)

    def write_view_plan(output: Path, *_args: object, **_kwargs: object) -> Path:
        calls.append("view_plan")
        output.mkdir(parents=True)
        (output / "view_plan.json").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(coarse_module, "write_view_plan", write_view_plan)
    discovery = generate_fin_discovery_plan(
        proxy,
        planning.geometric_plan.footprint_m,
        ViewPlanningConfig(standoff_distance_m=0.30),
        ViewFilterConfig(
            workspace=AxisAlignedBoxConfig(
                name="cell",
                minimum_m=(-1.0, -1.0, -0.5),
                maximum_m=(1.0, 1.0, 1.5),
            ),
            minimum_incidence_cosine=0.95,
        ),
        CoarseSciencePolicy(),
        _Reachable(),
    )
    monkeypatch.setattr(
        coarse_module,
        "generate_fin_discovery_plan",
        lambda *_args: discovery,
    )

    def write_discovery(output: Path, *_args: object, **_kwargs: object) -> Path:
        calls.append("discovery")
        output.mkdir(parents=True)
        (output / "discovery.json").write_text("{}", encoding="utf-8")
        return output.resolve()

    monkeypatch.setattr(coarse_module, "_write_discovery_plan_asset", write_discovery)
    generation_path = (tmp_path / "science" / "generations" / "000000").resolve()
    monkeypatch.setattr(
        coarse_module,
        "_append_coarse_scan_generation_from_verified",
        lambda output, **_kwargs: output.resolve(),
    )
    monkeypatch.setattr(
        coarse_module,
        "_bind_coarse_scan_view_readback",
        lambda view: SimpleNamespace(root=view.root),
    )

    session = CoarseScienceSession(
        settings=settings,
        hand_eye=SimpleNamespace(),  # type: ignore[arg-type]
        reachability_checker=_Reachable(),
        source_kinematics=kinematics,
        output_root=tmp_path / "science",
    )
    accepted = session.accept_prepared_view(
        PreparedCoarseScienceView(
            stored_view.root,
            tmp_path / "reconstructed",
            "operator_0",
            "operator_seed",
            coarse_module.BladeSide.FRONT,
        )
    )

    assert accepted == generation_path
    assert session.current_generation_path == generation_path
    assert session.discovery_plan is discovery
    assert calls == ["initialization", "view_plan", "discovery"]
    assert session.motion_authorized is False


@pytest.mark.parametrize(
    ("view_number", "expected_source_replays"),
    ((1, 1), (2, 3), (3, 6)),
)
def test_generation_accept_reuses_one_current_and_one_predecessor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    view_number: int,
    expected_source_replays: int,
) -> None:
    current_root = (tmp_path / f"view-{view_number}").resolve()
    current = SimpleNamespace(
        root=current_root,
        target_view_id=f"view-{view_number}",
    )
    previous_root = (
        (tmp_path / f"generation-{view_number - 1}").resolve()
        if view_number > 1
        else None
    )
    previous = (
        SimpleNamespace(
            root=previous_root,
            generation_index=view_number - 2,
        )
        if previous_root is not None
        else None
    )
    source_replays = 0

    def read_current(path: str | Path) -> SimpleNamespace:
        nonlocal source_replays
        assert Path(path).resolve() == current_root
        source_replays += view_number
        return current

    def read_previous(path: str | Path) -> SimpleNamespace:
        nonlocal source_replays
        assert previous_root is not None
        assert Path(path).resolve() == previous_root
        source_replays += view_number * (view_number - 1) // 2
        return previous

    captured: dict[str, object] = {}

    def append_generation(output: Path, **kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(output).resolve()

    monkeypatch.setattr(coarse_module, "read_coarse_scan_view", read_current)
    monkeypatch.setattr(coarse_module, "read_coarse_scan_generation", read_previous)
    monkeypatch.setattr(
        coarse_module,
        "_append_coarse_scan_generation_from_verified",
        append_generation,
    )
    readback = SimpleNamespace(root=current_root)
    monkeypatch.setattr(
        coarse_module,
        "_bind_coarse_scan_view_readback",
        lambda _view: readback,
    )
    session = object.__new__(CoarseScienceSession)
    session._generation = previous_root
    session._initialization = (tmp_path / "initialization").resolve()
    session._view_plan = (tmp_path / "view-plan").resolve()
    session._discovery_path = (tmp_path / "discovery").resolve()
    session._output_root = (tmp_path / "science").resolve()
    session._settings = load_settings("configs/default.yaml")

    output = session.accept_prepared_view(
        PreparedCoarseScienceView(
            current_root,
            tmp_path / "reconstructed",
            current.target_view_id,
            "operator_seed",
            coarse_module.BladeSide.FRONT,
        )
    )

    assert source_replays == expected_source_replays
    assert captured["current"] is current
    assert captured["verified_previous_generation"] is previous
    assert session.take_live_readback(expected_coarse_view=current_root) is readback
    assert output.name == f"{view_number - 1:06d}"


def test_discovery_is_re_evaluated_from_latest_stopped_posture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("model: test\n", encoding="utf-8")
    session = CoarseScienceSession(
        settings=load_settings("configs/default.yaml"),
        hand_eye=SimpleNamespace(),  # type: ignore[arg-type]
        reachability_checker=_Reachable(),
        source_kinematics=kinematics,
        output_root=tmp_path / "science",
    )
    initialization = (tmp_path / "initialization").resolve()
    view_plan = (tmp_path / "view-plan").resolve()
    generation = (tmp_path / "generation").resolve()
    prior = CoarseDiscoveryPlan(
        FilteredViewPlan((), ()),
        "a" * 64,
        current_joint_positions_rad=(0.0,) * 6,
    )
    refreshed = CoarseDiscoveryPlan(
        FilteredViewPlan((), ()),
        "b" * 64,
        current_joint_positions_rad=(0.2,) * 6,
    )
    session._generation = generation  # noqa: SLF001
    session._initialization = initialization  # noqa: SLF001
    session._view_plan = view_plan  # noqa: SLF001
    session._discovery_path = (tmp_path / "discovery").resolve()  # noqa: SLF001
    session._discovery = prior  # noqa: SLF001
    proxy = _proxy()
    monkeypatch.setattr(
        coarse_module,
        "read_initialization",
        lambda _path: SimpleNamespace(observation=SimpleNamespace(proxy=proxy)),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_view_plan",
        lambda _path: SimpleNamespace(
            result=SimpleNamespace(geometric_plan=SimpleNamespace(footprint_m=(0.3, 0.2)))
        ),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_coarse_scan_generation",
        lambda _path: SimpleNamespace(generation_index=3),
    )
    calls: list[tuple[float, ...]] = []

    def generate(*args: object, **kwargs: object) -> CoarseDiscoveryPlan:
        del kwargs
        calls.append(tuple(args[7]))  # type: ignore[arg-type]
        return refreshed

    monkeypatch.setattr(coarse_module, "generate_fin_discovery_plan", generate)

    def write(output: Path, *_args: object, **_kwargs: object) -> Path:
        output.mkdir(parents=True)
        (output / "discovery.json").write_text("{}\n", encoding="utf-8")
        return output.resolve()

    monkeypatch.setattr(coarse_module, "_write_discovery_plan_asset", write)

    session.refresh_discovery(
        current_joint_positions_rad=(0.2,) * 6,
        reachability_checker=_Reachable(),
    )

    assert calls == [(0.2,) * 6]
    assert session.discovery_plan is refreshed
    assert session._discovery_path is not None  # noqa: SLF001
    assert session._discovery_path.parent.name == "fin_discovery_revisions"  # noqa: SLF001


def test_operator_bootstrap_side_is_automatic_after_proxy_exists() -> None:
    proxy = _proxy()
    front_camera = PoseSE3.from_rotation_translation(
        "base", "left_rectified", np.eye(3), (0.0, 0.0, 0.9)
    )
    back_camera = PoseSE3.from_rotation_translation(
        "base", "left_rectified", np.eye(3), (0.0, 0.0, 0.3)
    )
    mid_camera = PoseSE3.from_rotation_translation(
        "base", "left_rectified", np.eye(3), (0.0, 0.0, 0.6)
    )

    assert (
        _resolve_operator_bootstrap_side(front_camera, proxy, None)
        is coarse_module.BladeSide.FRONT
    )
    assert (
        _resolve_operator_bootstrap_side(back_camera, proxy, None)
        is coarse_module.BladeSide.BACK
    )
    assert (
        _resolve_operator_bootstrap_side(front_camera, proxy, coarse_module.BladeSide.BACK)
        is coarse_module.BladeSide.BACK
    )
    with pytest.raises(UnknownBladeCoarseError, match="mid-plane"):
        _resolve_operator_bootstrap_side(mid_camera, proxy, None)


def test_engine_hook_requires_staging_and_appends_only_after_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("model: test\n", encoding="utf-8")
    session = CoarseScienceSession(
        settings=load_settings("configs/default.yaml"),
        hand_eye=SimpleNamespace(),  # type: ignore[arg-type]
        reachability_checker=_Reachable(),
        source_kinematics=kinematics,
        output_root=tmp_path / "science",
    )
    captured = SimpleNamespace(
        bundle=SimpleNamespace(view_id="operator_0"),
    )
    prepared = PreparedCoarseScienceView(
        (tmp_path / "cycle" / "coarse_scan_view").resolve(),
        (tmp_path / "cycle" / "coarse_reconstructed_view").resolve(),
        "operator_0",
        "operator_seed",
        coarse_module.BladeSide.FRONT,
    )
    monkeypatch.setattr(
        coarse_module,
        "prepare_unknown_blade_coarse_view",
        lambda **_kwargs: prepared,
    )

    with pytest.raises(UnknownBladeCoarseError, match="not explicitly staged"):
        session.prepare_engine_cycle(
            captured,
            SimpleNamespace(),
            tmp_path / "stereo",
            SimpleNamespace(),
            tmp_path / "occupancy",
        )

    session.stage_operator_capture()
    session._pending_foreground = SimpleNamespace()  # noqa: SLF001
    path = session.prepare_engine_cycle(
        captured,
        SimpleNamespace(),
        tmp_path / "stereo",
        SimpleNamespace(),
        tmp_path / "occupancy",
    )
    assert path == prepared.coarse_view_path
    assert session.current_generation_path is None
    accepted_generation = (tmp_path / "science" / "generations" / "000000").resolve()
    monkeypatch.setattr(
        session,
        "accept_prepared_view",
        lambda item: accepted_generation if item is prepared else None,
    )
    accepted = session.accept_cycle(
        SimpleNamespace(coarse_scan_view_path=prepared.coarse_view_path)  # type: ignore[arg-type]
    )
    assert accepted == accepted_generation
    timing = json.loads(
        (prepared.coarse_view_path.parent / "coarse_generation_timing.json").read_text(
            encoding="utf-8"
        )
    )
    assert timing["status"] == "completed"
    assert timing["identity"]["target_view_id"] == "operator_0"
    assert "coarse.generation_accept" in timing["spans"]
    session.stage_operator_capture(operator_side=coarse_module.BladeSide.BACK)
    session.reject_cycle()


def test_every_operator_bootstrap_requires_and_binds_its_own_hard_roi(
    tmp_path: Path,
) -> None:
    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("model: test\n", encoding="utf-8")
    session = CoarseScienceSession(
        settings=load_settings("configs/default.yaml"),
        hand_eye=SimpleNamespace(),  # type: ignore[arg-type]
        reachability_checker=_Reachable(),
        source_kinematics=kinematics,
        output_root=tmp_path / "science",
    )
    left = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    depth = np.full((64, 64), 0.8, dtype=np.float32)
    valid = np.ones((64, 64), dtype=np.bool_)
    provider_calls: list[Path] = []

    def capture_inputs(
        index: int,
    ) -> tuple[SimpleNamespace, SimpleNamespace, Path, SimpleNamespace]:
        cycle_root = tmp_path / f"cycle-{index}"
        cycle_root.mkdir()
        stereo_path = cycle_root / "stereo_inference"
        stereo_path.mkdir()
        (stereo_path / "metadata.json").write_text("{}\n", encoding="utf-8")
        bundle = SimpleNamespace(
            view_id=f"operator_{index}",
            sequence_index=index,
            stereo=SimpleNamespace(frame_number=100 + index),
        )
        observation = SimpleNamespace(
            rectified=SimpleNamespace(
                left_ir=left,
                source_frame_number=100 + index,
            ),
            depth_m=depth,
        )
        prepared = SimpleNamespace(
            bundle=bundle,
            stereo=observation,
            self_mask=SimpleNamespace(integration_valid_mask=valid),
        )
        captured = SimpleNamespace(bundle=bundle, cycle_root=cycle_root)
        return captured, observation, stereo_path, prepared

    first = capture_inputs(0)
    session.stage_operator_capture()
    with pytest.raises(UnknownBladeCoarseError, match="requires a hard_roi polygon"):
        session.preflight_engine_cycle(*first)
    session.reject_cycle()

    def provider(_captured, image_path: Path) -> BootstrapSeed:
        provider_calls.append(image_path)
        return BootstrapSeed.polygon(
            ((10, 10), (53, 10), (53, 53), (10, 53)),
            mode="hard_roi",
        )

    for index in (1, 2):
        item = capture_inputs(index)
        session.stage_operator_capture(seed_provider=provider)
        session.preflight_engine_cycle(*item)
        annotation_root = item[0].cycle_root / "bootstrap_annotation"
        assert (annotation_root / "left_rectified.png").is_file()
        assert (annotation_root / "request.json").is_file()
        response = json.loads(
            (annotation_root / "response.json").read_text(encoding="utf-8")
        )
        assert response["seed"]["mode"] == "hard_roi"
        assert response["mask_pixel_count"] > 500
        session.reject_cycle()

    assert provider_calls == [
        tmp_path / "cycle-1" / "bootstrap_annotation" / "left_rectified.png",
        tmp_path / "cycle-2" / "bootstrap_annotation" / "left_rectified.png",
    ]


def test_selected_coarse_view_uses_accepted_generation_projection_not_unseeded_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinematics = tmp_path / "kinematics.yaml"
    kinematics.write_text("model: test\n", encoding="utf-8")
    session = CoarseScienceSession(
        settings=load_settings("configs/default.yaml"),
        hand_eye=SimpleNamespace(),  # type: ignore[arg-type]
        reachability_checker=_Reachable(),
        source_kinematics=kinematics,
        output_root=tmp_path / "science",
    )
    generation = (tmp_path / "accepted-generation").resolve()
    session._generation = generation  # noqa: SLF001
    selection = SimpleNamespace(
        coverage_complete=False,
        target=SimpleNamespace(view_id="front:r0:c0"),
    )
    session.stage_selected_capture(selection)  # type: ignore[arg-type]

    left = np.zeros((8, 10), dtype=np.uint16)
    depth = np.full((8, 10), 0.8, dtype=np.float32)
    integration_valid = np.ones((8, 10), dtype=np.bool_)
    bundle = SimpleNamespace(
        view_id="front:r0:c0",
        sequence_index=4,
        stereo=SimpleNamespace(frame_number=17),
    )
    captured = SimpleNamespace(bundle=bundle, cycle_root=tmp_path / "cycle")
    stereo = SimpleNamespace(
        rectified=SimpleNamespace(left_ir=left, source_frame_number=17),
        depth_m=depth,
    )
    base_t_camera = PoseSE3.identity("base", "left_rectified")
    prepared = SimpleNamespace(
        bundle=bundle,
        stereo=stereo,
        self_mask=SimpleNamespace(integration_valid_mask=integration_valid),
        base_t_camera=base_t_camera,
    )
    projected = SimpleNamespace(algorithm="accepted_projection")
    calls: list[dict[str, object]] = []

    def project_from_generation(path: Path, **kwargs: object) -> object:
        calls.append({"path": path, **kwargs})
        return projected

    monkeypatch.setattr(
        coarse_module,
        "_projected_foreground_from_generation",
        project_from_generation,
    )
    monkeypatch.setattr(
        coarse_module,
        "bootstrap_blade_foreground",
        lambda *_args, **_kwargs: pytest.fail(
            "selected views must not run unseeded largest-component bootstrap"
        ),
    )

    session.preflight_engine_cycle(
        captured,
        stereo,
        tmp_path / "stereo",
        prepared,
    )

    assert session._pending_foreground is projected  # noqa: SLF001
    assert calls[0]["path"] == generation
    assert calls[0]["integration_valid_mask"] is integration_valid
    assert calls[0]["base_t_left_rectified"] == base_t_camera


def test_direct_automatic_coarse_adapter_cannot_fall_back_to_unseeded_component(
    tmp_path: Path,
) -> None:
    bundle = SimpleNamespace(
        view_id="front:r0:c0",
        sequence_index=4,
        stereo=SimpleNamespace(frame_number=17),
    )
    captured = SimpleNamespace(bundle=bundle)
    stereo = SimpleNamespace(
        source_view_id="front:r0:c0",
        source_sequence_index=4,
        rectified=SimpleNamespace(source_frame_number=17),
    )

    with pytest.raises(UnknownBladeCoarseError, match="projected foreground preflight"):
        coarse_module._prepare_unknown_blade_coarse_view(  # noqa: SLF001
            captured=captured,
            stereo=stereo,
            stereo_inference_path=tmp_path / "stereo",
            integration_valid_mask=np.ones((2, 2), dtype=np.bool_),
            integration_valid_mask_content_hash="a" * 64,
            integration_identity=("front:r0:c0", 4, 17),
            occupancy_mapping_path=tmp_path / "occupancy",
            hand_eye=SimpleNamespace(),
            settings=load_settings("configs/default.yaml"),
            foreground_config=coarse_module.BootstrapForegroundConfig(),
            seed=None,
            target_view_id="front:r0:c0",
            target_kind="proxy_normal",
            target_side=coarse_module.BladeSide.FRONT,
            side_proxy=None,
        )


def test_generation_append_removes_uncommitted_coverage_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings("configs/default.yaml")
    current = SimpleNamespace(
        root=(tmp_path / "coarse-view").resolve(),
        target_side=coarse_module.BladeSide.FRONT,
        proxy_config=settings.proxy_model,
        support_cloud=object(),
        reconstructed=SimpleNamespace(
            metadata={"source": {"session": str((tmp_path / "session").resolve())}},
            view=SimpleNamespace(
                source_view_id="coarse-0",
                source_sequence_index=0,
                source_frame_number=0,
                base_t_projection_camera=object(),
            ),
        ),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_initialization",
        lambda _path: SimpleNamespace(
            observation=SimpleNamespace(proxy=object()),
            metadata={
                "processing": {
                    "proxy_model": settings.proxy_model.model_dump(mode="json")
                }
            },
        ),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_view_plan",
        lambda _path: SimpleNamespace(result=SimpleNamespace(geometric_plan=object())),
    )
    monkeypatch.setattr(coarse_module, "read_coarse_scan_view", lambda _path: current)
    monkeypatch.setattr(
        coarse_module,
        "_camera_side",
        lambda *_args: coarse_module.BladeSide.FRONT,
    )
    ledger = SimpleNamespace()
    monkeypatch.setattr(coarse_module, "create_coverage_ledger", lambda *_args: ledger)
    monkeypatch.setattr(coarse_module, "update_coverage", lambda *_args: ledger)

    def write_coverage(path: Path, *_args: object, **_kwargs: object) -> Path:
        Path(path).mkdir(parents=True)
        return Path(path)

    monkeypatch.setattr(coarse_module, "write_coverage_ledger", write_coverage)
    generation_calls = 0

    def write_generation(path: Path, **_kwargs: object) -> Path:
        nonlocal generation_calls
        generation_calls += 1
        if generation_calls == 1:
            raise OSError("simulated generation commit failure")
        return Path(path).resolve()

    monkeypatch.setattr(
        coarse_module,
        "_write_coarse_scan_generation_from_verified",
        write_generation,
    )
    output = tmp_path / "generations" / "000000"
    kwargs = {
        "new_view": current.root,
        "source_initialization": tmp_path / "initialization",
        "source_view_plan": tmp_path / "view-plan",
        "source_discovery_plan": tmp_path / "discovery",
        "settings": settings,
    }

    with pytest.raises(OSError, match="generation commit failure"):
        coarse_module.append_coarse_scan_generation(output, **kwargs)
    assert not output.with_name("000000_coverage").exists()

    assert coarse_module.append_coarse_scan_generation(output, **kwargs) == output.resolve()


def test_ready_coarse_selection_keeps_the_run_initialization_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization = tmp_path / "initialization"
    generation_root = tmp_path / "generation"
    coarse_model = tmp_path / "coarse_model"
    coverage_root = tmp_path / "coverage"
    for root, filename in (
        (initialization, "metadata.json"),
        (generation_root, "generation.json"),
        (coarse_model, "metadata.json"),
    ):
        root.mkdir()
        (root / filename).write_text(f'{{"asset": "{root.name}"}}\n', encoding="utf-8")
    monkeypatch.setattr(
        coarse_module,
        "read_coarse_scan_generation",
        lambda _path: SimpleNamespace(
            root=generation_root.resolve(),
            coarse_model_path=coarse_model.resolve(),
            coverage_path=coverage_root.resolve(),
            metadata={"sources": {"initialization": {"root": str(initialization)}}},
        ),
    )
    monkeypatch.setattr(
        coarse_module,
        "read_coverage_ledger",
        lambda _path: SimpleNamespace(ledger=SimpleNamespace(patches=(1, 2, 3))),
    )
    discovery = CoarseDiscoveryPlan(FilteredViewPlan((), ()), "a" * 64)

    result = select_coarse_next_view(
        generation_root,
        discovery,
        SimpleNamespace(),  # type: ignore[arg-type]
        CoarseSciencePolicy(),
    )

    assert result.coverage_complete is True
    assert result.target is None
    assert result.reference_model_sha256 == coarse_module._sha256(
        initialization / "metadata.json"
    )
    assert result.reference_model_sha256 != coarse_module._sha256(
        coarse_model / "metadata.json"
    )


def _promotion_fixture(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
    views = []
    for index, side in enumerate(
        (
            coarse_module.BladeSide.FRONT,
            coarse_module.BladeSide.FRONT,
            coarse_module.BladeSide.BACK,
            coarse_module.BladeSide.BACK,
        )
    ):
        reconstructed_root = (tmp_path / f"reconstructed-{index}").resolve()
        reconstructed_root.mkdir()
        (reconstructed_root / "metadata.json").write_text("{}\n", encoding="utf-8")
        views.append(
            SimpleNamespace(
                root=(tmp_path / f"coarse-view-{index}").resolve(),
                target_side=side,
                reconstructed=SimpleNamespace(
                    view=SimpleNamespace(planning_intrinsics=object())
                ),
                metadata={
                    "sources": {"reconstructed_view": {"root": str(reconstructed_root)}}
                },
            )
        )
    generation = SimpleNamespace(
        root=(tmp_path / "generation").resolve(),
        coarse_model_path=None,
        coverage_path=(tmp_path / "coverage").resolve(),
        views=tuple(views),
        metadata={
            "sources": {
                "initialization": {"root": str((tmp_path / "initialization").resolve())},
                "view_plan": {"root": str((tmp_path / "view-plan").resolve())},
                "discovery_plan": {"root": str((tmp_path / "discovery").resolve())},
            }
        },
    )
    discovery = SimpleNamespace(endpoint_feasible=())
    return generation, discovery


def _matching_coarse_metadata(
    settings: object,
    source_roots: tuple[Path, ...],
) -> dict[str, object]:
    return {
        "schema_version": 5,
        "source_views": [{"path": str(path)} for path in source_roots],
        "proxy_support": {
            "configuration": settings.proxy_model.model_dump(mode="json"),
            "source_coarse_views": [
                {"path": str(path.parent / path.name.replace("reconstructed", "coarse-view"))}
                for path in source_roots
            ],
        },
        "fusion": {"configuration": settings.multi_view_fusion.model_dump(mode="json")},
        "surface": {
            "configuration": settings.surface_partition.model_dump(mode="json"),
            "fin_components": [
                {"side": "front", "two_faces_observed": True},
                {"side": "back", "two_faces_observed": True},
            ],
        },
        "view_plan": {"configuration": settings.view_planning.model_dump(mode="json")},
        "tsdf": {"configuration": settings.tsdf.model_dump(mode="json")},
        "quality": {"configuration": settings.surface_quality.model_dump(mode="json")},
    }


def _patch_promotion_gates(
    monkeypatch: pytest.MonkeyPatch,
    generation: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        coarse_module, "read_coarse_scan_generation", lambda _path: generation
    )
    monkeypatch.setattr(
        coarse_module,
        "read_coverage_ledger",
        lambda _path: SimpleNamespace(
            ledger=SimpleNamespace(completed_patch_ids=(), patches=())
        ),
    )
    monkeypatch.setattr(
        coarse_module,
        "_verified_discovery_ids",
        lambda *_args: {"negative", "positive"},
    )
    monkeypatch.setattr(
        coarse_module,
        "_paired_discovery_ids",
        lambda *_args: (("negative", "positive"),),
    )


def test_finalize_reuses_exact_verified_model_after_ready_generation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings("configs/default.yaml")
    generation, discovery = _promotion_fixture(tmp_path)
    _patch_promotion_gates(monkeypatch, generation)
    result = SimpleNamespace(
        surface=SimpleNamespace(
            fin_component=lambda _side: SimpleNamespace(two_faces_observed=True)
        )
    )
    build_calls: list[object] = []
    monkeypatch.setattr(
        coarse_module,
        "build_coarse_blade_model",
        lambda *_args: build_calls.append(object()) or result,
    )
    coarse_output = (tmp_path / "coarse-model").resolve()
    source_roots = tuple(
        Path(item.metadata["sources"]["reconstructed_view"]["root"]).resolve()
        for item in generation.views
    )
    metadata = _matching_coarse_metadata(settings, source_roots)
    model_write_calls: list[Path] = []

    def write_model(output: Path, *_args: object, **_kwargs: object) -> Path:
        output = Path(output).resolve()
        output.mkdir()
        model_write_calls.append(output)
        return output

    monkeypatch.setattr(coarse_module, "write_coarse_model", write_model)
    monkeypatch.setattr(
        coarse_module,
        "read_coarse_model_summary",
        lambda path: SimpleNamespace(root=Path(path).resolve(), metadata=metadata),
    )
    generation_write_calls: list[Path] = []

    def write_generation(output: Path, **_kwargs: object) -> Path:
        generation_write_calls.append(Path(output).resolve())
        if len(generation_write_calls) == 1:
            raise OSError("simulated interruption after coarse-model commit")
        return Path(output).resolve()

    monkeypatch.setattr(coarse_module, "write_coarse_scan_generation", write_generation)
    policy = CoarseSciencePolicy(
        minimum_total_views=4,
        minimum_views_per_side=2,
        require_complete_proxy_coverage=False,
    )

    with pytest.raises(OSError, match="simulated interruption"):
        coarse_module.finalize_coarse_generation(
            generation.root,
            discovery,
            policy,
            settings,
            output_coarse_model=coarse_output,
            output_ready_generation=tmp_path / "ready-generation",
        )
    recovered = coarse_module.finalize_coarse_generation(
        generation.root,
        discovery,
        policy,
        settings,
        output_coarse_model=coarse_output,
        output_ready_generation=tmp_path / "ready-generation",
    )

    assert recovered.phase is coarse_module.CoarsePhase.READY_FOR_FINE
    assert recovered.reference_coarse_model_path == coarse_output
    assert len(build_calls) == 1
    assert model_write_calls == [coarse_output]
    assert len(generation_write_calls) == 2


@pytest.mark.parametrize("tamper", ("source", "settings", "fin_evidence"))
def test_finalize_refuses_incompatible_existing_model_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    settings = load_settings("configs/default.yaml")
    generation, discovery = _promotion_fixture(tmp_path)
    _patch_promotion_gates(monkeypatch, generation)
    coarse_output = (tmp_path / "coarse-model").resolve()
    coarse_output.mkdir()
    source_roots = tuple(
        Path(item.metadata["sources"]["reconstructed_view"]["root"]).resolve()
        for item in generation.views
    )
    metadata = _matching_coarse_metadata(settings, source_roots)
    if tamper == "source":
        metadata["source_views"][0]["path"] = str((tmp_path / "other-view").resolve())
    elif tamper == "settings":
        metadata["tsdf"]["configuration"] = {"tampered": True}
    else:
        metadata["surface"]["fin_components"][1]["two_faces_observed"] = False
    monkeypatch.setattr(
        coarse_module,
        "read_coarse_model_summary",
        lambda _path: SimpleNamespace(root=coarse_output, metadata=metadata),
    )
    monkeypatch.setattr(
        coarse_module,
        "build_coarse_blade_model",
        lambda *_args: pytest.fail("an existing model must never be silently rebuilt"),
    )
    monkeypatch.setattr(
        coarse_module,
        "write_coarse_model",
        lambda *_args, **_kwargs: pytest.fail("an existing model must never be overwritten"),
    )
    monkeypatch.setattr(
        coarse_module,
        "write_coarse_scan_generation",
        lambda *_args, **_kwargs: pytest.fail("an incompatible model must not be promoted"),
    )

    with pytest.raises(UnknownBladeCoarseError, match="refusing to overwrite or reuse"):
        coarse_module.finalize_coarse_generation(
            generation.root,
            discovery,
            CoarseSciencePolicy(
                minimum_total_views=4,
                minimum_views_per_side=2,
                require_complete_proxy_coverage=False,
            ),
            settings,
            output_coarse_model=coarse_output,
            output_ready_generation=tmp_path / "ready-generation",
        )


def test_finalize_refuses_existing_model_that_fails_full_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings("configs/default.yaml")
    generation, discovery = _promotion_fixture(tmp_path)
    _patch_promotion_gates(monkeypatch, generation)
    coarse_output = (tmp_path / "coarse-model").resolve()
    coarse_output.mkdir()
    monkeypatch.setattr(
        coarse_module,
        "read_coarse_model_summary",
        lambda _path: (_ for _ in ()).throw(ValueError("checksum mismatch")),
    )
    monkeypatch.setattr(
        coarse_module,
        "write_coarse_model",
        lambda *_args, **_kwargs: pytest.fail("a corrupt asset must never be overwritten"),
    )

    with pytest.raises(UnknownBladeCoarseError, match="refusing to overwrite or reuse"):
        coarse_module.finalize_coarse_generation(
            generation.root,
            discovery,
            CoarseSciencePolicy(
                minimum_total_views=4,
                minimum_views_per_side=2,
                require_complete_proxy_coverage=False,
            ),
            settings,
            output_coarse_model=coarse_output,
            output_ready_generation=tmp_path / "ready-generation",
        )
