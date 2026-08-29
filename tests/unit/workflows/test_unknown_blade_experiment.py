from __future__ import annotations

from pathlib import Path

import pytest

from biblade_fusion.workflows.stop_scan_coordinator import (
    NextViewSelection,
    NextViewTarget,
)
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarsePhase,
    CoarsePhaseTransition,
)
from biblade_fusion.workflows.unknown_blade_experiment import (
    FinePhaseStep,
    UnknownBladeExperimentComponents,
    UnknownBladeExperimentController,
    UnknownBladeExperimentError,
    UnknownBladeExperimentPhase,
)


def _selection(index: int) -> NextViewSelection:
    target = NextViewTarget(
        f"coarse_{index}",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (
            (1.0, 0.0, 0.0, 0.4),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.5),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return NextViewSelection(
        target,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        10,
        1,
        False,
        ("simulation",),
    )


class _FineDriver:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> FinePhaseStep:
        self.steps += 1
        return FinePhaseStep(self.steps == 1, "fine bilateral coverage complete")


def test_controller_runs_operator_bootstrap_coarse_fin_handoff_and_fine(
    tmp_path: Path,
) -> None:
    append_count = 0
    selected_count = 0
    finalized_count = 0
    events = []
    reference = (tmp_path / "coarse_model").resolve()
    ready_generation = (tmp_path / "generation_ready").resolve()

    def append(view: Path, previous: Path | None) -> Path:
        nonlocal append_count
        assert view.is_absolute()
        if append_count > 0:
            assert previous is not None
        path = (tmp_path / f"generation_{append_count}").resolve()
        append_count += 1
        return path

    def select(_generation: Path, requires_fin: bool) -> NextViewSelection:
        nonlocal selected_count
        if selected_count == 1:
            assert requires_fin is True
        result = _selection(selected_count)
        selected_count += 1
        return result

    def finalize(generation: Path) -> CoarsePhaseTransition:
        nonlocal finalized_count
        finalized_count += 1
        if finalized_count == 1:
            return CoarsePhaseTransition(
                CoarsePhase.COLLECTING_FIN_EVIDENCE,
                ("front fin needs the orthogonal oblique pair",),
                generation,
            )
        return CoarsePhaseTransition(
            CoarsePhase.READY_FOR_FINE,
            ("schema-5 committed",),
            generation,
            ready_generation,
            reference,
        )

    controller = UnknownBladeExperimentController(
        UnknownBladeExperimentComponents(
            append,
            select,
            finalize,
            lambda path: _FineDriver() if path == reference else None,  # type: ignore[arg-type,return-value]
            events.append,
        )
    )
    assert controller.snapshot.phase is UnknownBladeExperimentPhase.WAITING_OPERATOR_BOOTSTRAP

    for index in range(3):
        snapshot = controller.capture_operator_bootstrap(
            lambda index=index: tmp_path / f"operator_view_{index}",
            occupancy_map_ready=index == 2,
        )
    assert snapshot.phase is UnknownBladeExperimentPhase.COARSE_SCAN
    assert snapshot.accepted_operator_bootstrap_views == 3

    first = controller.select_coarse_view()
    snapshot = controller.capture_selected_coarse_view(
        first,
        lambda selection: tmp_path / selection.target.view_id,  # type: ignore[union-attr]
    )
    assert snapshot.phase is UnknownBladeExperimentPhase.COARSE_SCAN
    assert snapshot.coarse_requires_additional_fin_evidence is True

    second = controller.select_coarse_view()
    snapshot = controller.capture_selected_coarse_view(
        second,
        lambda selection: tmp_path / selection.target.view_id,  # type: ignore[union-attr]
    )
    assert snapshot.phase is UnknownBladeExperimentPhase.FINE_SCAN
    assert snapshot.current_coarse_generation_path == ready_generation
    assert snapshot.reference_coarse_model_path == reference

    snapshot = controller.step_fine()
    assert snapshot.phase is UnknownBladeExperimentPhase.COMPLETE
    assert snapshot.attention_reason == "fine bilateral coverage complete"
    assert all(event.motion_authorized is False for event in events)


def test_controller_rejects_premature_map_ready(tmp_path: Path) -> None:
    controller = UnknownBladeExperimentController(
        UnknownBladeExperimentComponents(
            lambda _view, _previous: tmp_path / "generation",
            lambda _generation, _requires_fin: _selection(0),
            lambda generation: CoarsePhaseTransition(
                CoarsePhase.COLLECTING,
                ("collecting",),
                generation,
            ),
            lambda _reference: _FineDriver(),
        )
    )

    with pytest.raises(UnknownBladeExperimentError, match="minimum independent"):
        controller.capture_operator_bootstrap(
            lambda: tmp_path / "view",
            occupancy_map_ready=True,
        )
    assert controller.snapshot.phase is UnknownBladeExperimentPhase.BLOCKED


def test_schema5_handoff_is_atomic_when_fine_driver_validation_fails(
    tmp_path: Path,
) -> None:
    append_index = 0
    reference = (tmp_path / "reference").resolve()
    ready = (tmp_path / "ready").resolve()

    def append(_view: Path, _previous: Path | None) -> Path:
        nonlocal append_index
        path = (tmp_path / f"generation_{append_index}").resolve()
        append_index += 1
        return path

    controller = UnknownBladeExperimentController(
        UnknownBladeExperimentComponents(
            append,
            lambda _generation, _requires_fin: _selection(0),
            lambda generation: CoarsePhaseTransition(
                CoarsePhase.READY_FOR_FINE,
                ("schema-5 committed",),
                generation,
                ready,
                reference,
            ),
            lambda _reference: (_ for _ in ()).throw(RuntimeError("fine validation failed")),
        )
    )
    for index in range(3):
        controller.capture_operator_bootstrap(
            lambda index=index: tmp_path / f"view_{index}",
            occupancy_map_ready=index == 2,
        )
    predecessor = controller.snapshot.current_coarse_generation_path
    selection = controller.select_coarse_view()

    with pytest.raises(UnknownBladeExperimentError, match="fine handoff failed closed"):
        controller.capture_selected_coarse_view(
            selection,
            lambda _selection: tmp_path / "candidate",
        )

    snapshot = controller.snapshot
    assert snapshot.phase is UnknownBladeExperimentPhase.BLOCKED
    assert snapshot.reference_coarse_model_path is None
    assert snapshot.current_coarse_generation_path != ready
    assert snapshot.current_coarse_generation_path != predecessor
