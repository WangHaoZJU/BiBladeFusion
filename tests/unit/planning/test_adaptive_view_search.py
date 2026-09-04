import json

import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import AxisAlignedBoxConfig, ViewFilterConfig
from biblade_fusion.perception.proxy import BilateralBladeProxy
from biblade_fusion.planning import (
    AdaptiveViewSearchConfig,
    BladeSide,
    CandidateView,
    EndpointConfigurationCheck,
    ReachabilityResult,
    ReachabilityState,
    SurfacePatch,
    adaptive_view_search_payload,
    evaluate_multi_seed_ik,
    generate_adaptive_candidate_family,
    search_adaptive_candidate_family,
)


def make_proxy() -> BilateralBladeProxy:
    return BilateralBladeProxy(
        PoseSE3.identity("base", "blade_proxy"),
        np.array([0.20, 0.10, 0.02]),
        np.zeros(3),
        np.array([1.0, 0.5, 0.0]),
        100,
        100,
        100,
        1.0,
    )


def wide_workspace_filter(**updates: float) -> ViewFilterConfig:
    values: dict[str, object] = {
        "workspace": AxisAlignedBoxConfig(
            name="physical_outer_boundary",
            minimum_m=(-2.0, -2.0, -2.0),
            maximum_m=(2.0, 2.0, 2.0),
        ),
        "camera_clearance_radius_m": 0.01,
    }
    values.update(updates)
    return ViewFilterConfig(**values)


def make_nominal() -> CandidateView:
    patch = SurfacePatch(
        "front_r00_c00",
        BladeSide.FRONT,
        0,
        0,
        np.array([0.0, 0.0, 0.01]),
        np.array([0.0, 0.0, 1.0]),
        (0.10, 0.08),
    )
    pose = PoseSE3.from_rotation_translation(
        "base",
        "front_r00_c00_left_ir",
        np.diag([1.0, -1.0, -1.0]),
        np.array([0.0, 0.0, 0.31]),
    )
    return CandidateView("front_r00_c00", patch, pose, 0.30, (0.20, 0.12))


class FixedChecker:
    def __init__(self, joints: np.ndarray) -> None:
        self.joints = joints

    def check(self, _pose: PoseSE3) -> ReachabilityResult:
        return ReachabilityResult(ReachabilityState.REACHABLE, "reachable", self.joints)


class RollSensitiveChecker:
    def check(self, pose: PoseSE3) -> ReachabilityResult:
        if pose.rotation[0, 0] < 0.8:
            return ReachabilityResult(
                ReachabilityState.REACHABLE,
                "roll branch reachable",
                np.full(6, 0.2),
            )
        return ReachabilityResult(ReachabilityState.UNREACHABLE, "nominal wrist branch fails")


class LazyBranchChecker:
    def __init__(self, branches: tuple[np.ndarray, ...]) -> None:
        self.branches = branches
        self.yielded = 0

    def iter_checks(self, _pose: PoseSE3):
        for branch in self.branches:
            self.yielded += 1
            yield ReachabilityResult(ReachabilityState.REACHABLE, "branch", branch)


def test_family_starts_at_ideal_then_stratifies_each_search_dimension() -> None:
    config = AdaptiveViewSearchConfig(
        distance_step_m=0.05,
        maximum_distance_expansions=1,
        tilt_samples_deg=(0.0, 30.0),
        azimuth_samples_deg=(0.0, 90.0),
        roll_samples_deg=(0.0, 45.0),
    )

    family = generate_adaptive_candidate_family(make_nominal(), config)

    first_parameters, first = family[0]
    second_parameters, second = family[1]
    assert first_parameters == first_parameters.__class__(0.30, 0.0, 0.0, 0.0, 0)
    assert second_parameters.roll_deg == 0.0
    assert second_parameters.distance_m == pytest.approx(0.30)
    assert second_parameters.tilt_deg == 30.0
    prefix = [item for item, _candidate in family[:8]]
    assert {item.roll_deg for item in prefix} == {0.0, 45.0}
    assert {item.distance_m for item in prefix} == {0.25, 0.30, 0.35}
    assert {item.azimuth_deg for item in prefix} == {0.0, 90.0}
    np.testing.assert_allclose(first.base_t_left_ir.translation_m, [0.0, 0.0, 0.31])
    for candidate in (first, second):
        view = candidate.patch.target_m - candidate.base_t_left_ir.translation_m
        np.testing.assert_allclose(
            candidate.optical_axis,
            view / np.linalg.norm(view),
            atol=1e-12,
        )


def test_multi_seed_ik_keeps_all_solutions_and_selects_nearest_current() -> None:
    far = np.full(6, 1.0)
    near = np.full(6, 0.1)

    evaluation = evaluate_multi_seed_ik(
        make_nominal().base_t_left_ir,
        (FixedChecker(far), FixedChecker(near)),
        np.zeros(6),
    )

    assert len(evaluation.solutions_rad) == 2
    assert evaluation.chosen_solution_index == 1
    np.testing.assert_allclose(evaluation.result.joint_positions_rad, near)


def test_multi_seed_ik_selects_nearest_collision_clear_branch() -> None:
    colliding_near = np.full(6, 0.1)
    clear_far = np.full(6, 0.4)

    def endpoint_validator(joints: np.ndarray) -> EndpointConfigurationCheck:
        if np.allclose(joints, colliding_near):
            return EndpointConfigurationCheck(False, ("self_collision:wrist:camera",))
        return EndpointConfigurationCheck(True)

    evaluation = evaluate_multi_seed_ik(
        make_nominal().base_t_left_ir,
        (FixedChecker(colliding_near), FixedChecker(clear_far)),
        np.zeros(6),
        endpoint_validator,
    )

    assert evaluation.endpoint_collision_checked is True
    assert evaluation.chosen_solution_index == 1
    assert evaluation.solution_blocking_reasons == (
        ("self_collision:wrist:camera",),
        (),
    )
    np.testing.assert_allclose(evaluation.result.joint_positions_rad, clear_far)


def test_multi_seed_ik_rejects_pose_when_every_branch_collides() -> None:
    evaluation = evaluate_multi_seed_ik(
        make_nominal().base_t_left_ir,
        (FixedChecker(np.full(6, 0.1)), FixedChecker(np.full(6, 0.4))),
        np.zeros(6),
        lambda _joints: EndpointConfigurationCheck(
            False,
            ("self_collision:forearm:camera",),
        ),
    )

    assert evaluation.result.state is ReachabilityState.UNREACHABLE
    assert evaluation.chosen_solution_index is None
    assert "every endpoint is collision blocked" in evaluation.result.message


def test_lazy_ik_stops_at_first_clear_endpoint_but_skips_colliding_branch() -> None:
    branches = tuple(np.full(6, value) for value in (0.1, 0.2, 0.3))
    checker = LazyBranchChecker(branches)

    evaluation = evaluate_multi_seed_ik(
        make_nominal().base_t_left_ir,
        (checker,),
        np.zeros(6),
        lambda joints: (
            EndpointConfigurationCheck(False, ("wrist_collision",))
            if np.allclose(joints, branches[0])
            else EndpointConfigurationCheck(True)
        ),
    )

    assert checker.yielded == 2
    assert len(evaluation.solutions_rad) == 2
    assert evaluation.chosen_solution_index == 1
    np.testing.assert_allclose(evaluation.result.joint_positions_rad, branches[1])


def test_search_never_escapes_configured_outer_workspace() -> None:
    narrow = AxisAlignedBoxConfig(
        name="empirical_camera_centres",
        minimum_m=(-0.05, -0.05, -0.05),
        maximum_m=(0.05, 0.05, 0.05),
    )
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=0,
        tilt_samples_deg=(0.0,),
        azimuth_samples_deg=(0.0,),
        roll_samples_deg=(0.0, 45.0),
        maximum_ik_feasible_candidates=1,
    )

    result = search_adaptive_candidate_family(
        make_nominal(),
        make_proxy(),
        ViewFilterConfig(workspace=narrow, camera_clearance_radius_m=0.01),
        (RollSensitiveChecker(),),
        np.zeros(6),
        config,
    )

    assert len(result.attempts) == 2
    assert len(result.ranked_feasible) == 0
    assert result.recommended is None
    assert all(
        "leaves workspace" in " ".join(attempt.evaluated.reasons)
        for attempt in result.attempts
    )
    assert result.motion_authorized is False


def test_fin_search_samples_every_tilt_at_nominal_distance_and_prefers_45_deg() -> None:
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=1,
        tilt_samples_deg=(15.0, 30.0, 45.0, 60.0),
        azimuth_samples_deg=(0.0,),
        roll_samples_deg=(0.0,),
        maximum_ik_feasible_candidates=4,
        sampling_order="distance_major",
        ranking_mode="fin_discovery",
        require_attempted_per_tilt=True,
    )

    result = search_adaptive_candidate_family(
        make_nominal(),
        make_proxy(),
        wide_workspace_filter(minimum_incidence_cosine=0.4),
        (FixedChecker(np.zeros(6)),),
        np.zeros(6),
        config,
    )

    assert [item.parameters.distance_m for item in result.attempts] == [0.30] * 4
    assert [item.parameters.tilt_deg for item in result.attempts] == [
        15.0,
        30.0,
        45.0,
        60.0,
    ]
    assert result.recommended is not None
    assert result.recommended.parameters.tilt_deg == 45.0


def test_fin_search_prefix_covers_tilt_roll_and_both_distance_directions() -> None:
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=1,
        tilt_samples_deg=(15.0, 45.0),
        azimuth_samples_deg=(90.0,),
        roll_samples_deg=(0.0, 45.0, -45.0, 90.0),
        maximum_ik_feasible_candidates=8,
        maximum_ik_attempts_per_family=8,
        sampling_order="distance_major",
        ranking_mode="fin_discovery",
        require_attempted_per_tilt=True,
    )

    result = search_adaptive_candidate_family(
        make_nominal(),
        make_proxy(),
        wide_workspace_filter(minimum_incidence_cosine=0.4),
        (FixedChecker(np.zeros(6)),),
        np.zeros(6),
        config,
    )

    parameters = [item.parameters for item in result.attempts]
    assert {item.tilt_deg for item in parameters} == {15.0, 45.0}
    assert {item.roll_deg for item in parameters} == {0.0, 45.0, -45.0, 90.0}
    assert {round(item.distance_m, 2) for item in parameters} == {0.26, 0.30, 0.34}


def test_realistic_32_attempt_prefix_never_starves_adaptive_dimensions() -> None:
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=3,
        tilt_samples_deg=(45.0, 30.0, 60.0, 20.0, 15.0, 10.0),
        azimuth_samples_deg=(0.0, 22.5, -22.5, 45.0, -45.0, 67.5, -67.5),
        roll_samples_deg=(0.0, 45.0, -45.0, 90.0),
        maximum_ik_attempts_per_family=32,
        ranking_mode="fin_discovery",
        sampling_order="distance_major",
    )

    prefix = [
        parameters
        for parameters, _candidate in generate_adaptive_candidate_family(
            make_nominal(), config
        )[:32]
    ]

    assert {item.tilt_deg for item in prefix} == set(config.tilt_samples_deg)
    assert {item.roll_deg for item in prefix} == set(config.roll_samples_deg)
    assert {item.azimuth_deg for item in prefix} == set(config.azimuth_samples_deg)
    distances = {round(item.distance_m, 2) for item in prefix}
    assert 0.26 in distances
    assert 0.34 in distances


def test_report_is_json_serializable_and_explicitly_non_executable() -> None:
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=0,
        tilt_samples_deg=(0.0,),
        azimuth_samples_deg=(0.0,),
        roll_samples_deg=(0.0,),
        maximum_ik_feasible_candidates=1,
    )
    result = search_adaptive_candidate_family(
        make_nominal(),
        make_proxy(),
        wide_workspace_filter(),
        (FixedChecker(np.zeros(6)),),
        np.zeros(6),
        config,
    )

    payload = adaptive_view_search_payload(result, config, np.zeros(6))
    json.dumps(payload)

    assert payload["motion_authorized"] is False
    assert payload["endpoint_collision_checked"] is False
    assert payload["trajectory_checked"] is False
    assert payload["summary"]["ik_feasible_count"] == 1


def test_report_records_endpoint_collision_filtering() -> None:
    config = AdaptiveViewSearchConfig(
        maximum_distance_expansions=0,
        tilt_samples_deg=(0.0,),
        azimuth_samples_deg=(0.0,),
        roll_samples_deg=(0.0,),
        maximum_ik_feasible_candidates=1,
    )
    result = search_adaptive_candidate_family(
        make_nominal(),
        make_proxy(),
        wide_workspace_filter(),
        (FixedChecker(np.zeros(6)),),
        np.zeros(6),
        config,
        endpoint_validator=lambda _joints: EndpointConfigurationCheck(True),
    )

    payload = adaptive_view_search_payload(result, config, np.zeros(6))

    assert payload["schema_version"] == 2
    assert payload["endpoint_collision_checked"] is True
    assert payload["attempts"][0]["endpoint_collision_checked"] is True
