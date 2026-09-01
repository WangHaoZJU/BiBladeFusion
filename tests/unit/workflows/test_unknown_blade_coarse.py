from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biblade_fusion.workflows.unknown_blade_coarse as coarse_module
from biblade_fusion.core.pose import PoseSE3
from biblade_fusion.core.settings import (
    AxisAlignedBoxConfig,
    ViewFilterConfig,
    ViewPlanningConfig,
    load_settings,
)
from biblade_fusion.devices.depth_camera import CameraIntrinsics
from biblade_fusion.perception.pointcloud import PointCloud
from biblade_fusion.perception.proxy import BilateralBladeProxy, select_proxy_support
from biblade_fusion.planning import (
    FilteredViewPlan,
    ReachabilityResult,
    ReachabilityState,
)
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
    discovery = CoarseDiscoveryPlan(FilteredViewPlan((), ()), "a" * 64)
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
        "append_coarse_scan_generation",
        lambda output, **_kwargs: output.resolve(),
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
    session.stage_operator_capture(operator_side=coarse_module.BladeSide.BACK)
    session.reject_cycle()


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

    monkeypatch.setattr(coarse_module, "write_coarse_scan_generation", write_generation)
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
