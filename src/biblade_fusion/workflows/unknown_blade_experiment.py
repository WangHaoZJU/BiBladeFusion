"""Injectable phase machine for an unknown-blade bilateral scan experiment.

The controller does not own a robot or camera.  A production composition root
injects stopped-capture, coarse generation, schema-5 promotion and fine-run
components.  That makes operator-guided map bootstrap explicit and prevents an
implicit transition from an unknown map into autonomous motion.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from biblade_fusion.workflows.stop_scan_coordinator import NextViewSelection
from biblade_fusion.workflows.unknown_blade_coarse import (
    CoarsePhase,
    CoarsePhaseTransition,
)


class UnknownBladeExperimentError(RuntimeError):
    """An experiment phase transition was invalid or failed closed."""


class UnknownBladeExperimentPhase(StrEnum):
    WAITING_OPERATOR_BOOTSTRAP = "waiting_operator_bootstrap"
    COARSE_SCAN = "coarse_scan"
    BUILDING_SCHEMA5 = "building_schema5"
    FINE_SCAN = "fine_scan"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class FinePhaseStep:
    complete: bool
    attention_reason: str

    def __post_init__(self) -> None:
        reason = str(self.attention_reason).strip()
        if not reason:
            raise ValueError("Fine phase step requires an attention reason")
        object.__setattr__(self, "attention_reason", reason)


class FinePhaseDriver(Protocol):
    """Already configured fine engine/selector runner, created from one schema-5 root."""

    def step(self) -> FinePhaseStep: ...


@dataclass(frozen=True, slots=True)
class UnknownBladeExperimentSnapshot:
    phase: UnknownBladeExperimentPhase
    attention_reason: str
    accepted_operator_bootstrap_views: int
    current_coarse_generation_path: Path | None
    reference_coarse_model_path: Path | None
    coarse_requires_additional_fin_evidence: bool

    @property
    def motion_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class UnknownBladeExperimentComponents:
    """Production dependencies; every callback must independently verify its assets."""

    append_coarse_view: Callable[[Path, Path | None], Path]
    select_coarse_view: Callable[[Path, bool], NextViewSelection]
    finalize_coarse: Callable[[Path], CoarsePhaseTransition]
    create_fine_driver: Callable[[Path], FinePhaseDriver]
    event_sink: Callable[[UnknownBladeExperimentSnapshot], None] = lambda _snapshot: None


class UnknownBladeExperimentController:
    """State machine spanning operator bootstrap, coarse science and fine science.

    The first captures must each be explicitly triggered by the operator.  Merely
    reaching the configured count is insufficient: the caller must also report that
    the independently verified occupancy generation is ``MAP_READY``.  Coarse and
    fine drivers remain responsible for their existing guarded-motion approval.
    """

    def __init__(
        self,
        components: UnknownBladeExperimentComponents,
        *,
        minimum_operator_bootstrap_views: int = 3,
        recovered_generation_path: str | Path | None = None,
        recovered_reference_coarse_model_path: str | Path | None = None,
    ) -> None:
        if minimum_operator_bootstrap_views < 3:
            raise ValueError("Unknown-space bootstrap requires at least three operator views")
        self._components = components
        self._minimum_bootstrap = minimum_operator_bootstrap_views
        self._lock = threading.RLock()
        self._bootstrap_count = 0
        self._generation = (
            Path(recovered_generation_path).resolve()
            if recovered_generation_path is not None
            else None
        )
        self._reference = (
            Path(recovered_reference_coarse_model_path).resolve()
            if recovered_reference_coarse_model_path is not None
            else None
        )
        self._requires_fin = False
        self._fine_driver: FinePhaseDriver | None = None
        if self._reference is not None:
            if self._generation is None:
                raise ValueError("Recovered fine reference requires its coarse generation")
            self._fine_driver = components.create_fine_driver(self._reference)
            self._phase = UnknownBladeExperimentPhase.FINE_SCAN
            self._attention = "recovered schema-5 reference; fine phase is ready"
        elif self._generation is not None:
            self._phase = UnknownBladeExperimentPhase.COARSE_SCAN
            self._attention = "recovered coarse generation; select the next coarse view"
        else:
            self._phase = UnknownBladeExperimentPhase.WAITING_OPERATOR_BOOTSTRAP
            self._attention = "operator must place the robot and trigger bootstrap capture 1"
        self._emit()

    @property
    def snapshot(self) -> UnknownBladeExperimentSnapshot:
        with self._lock:
            return UnknownBladeExperimentSnapshot(
                self._phase,
                self._attention,
                self._bootstrap_count,
                self._generation,
                self._reference,
                self._requires_fin,
            )

    def _emit(self) -> None:
        self._components.event_sink(self.snapshot)

    def _require_phase(self, expected: UnknownBladeExperimentPhase) -> None:
        if self._phase is not expected:
            raise UnknownBladeExperimentError(
                f"Expected phase {expected.value}, found {self._phase.value}"
            )

    def capture_operator_bootstrap(
        self,
        capture_stopped_view: Callable[[], str | Path],
        *,
        occupancy_map_ready: bool,
    ) -> UnknownBladeExperimentSnapshot:
        """Trigger exactly one operator-requested stopped capture and accept its view."""

        with self._lock:
            self._require_phase(UnknownBladeExperimentPhase.WAITING_OPERATOR_BOOTSTRAP)
            try:
                view_path = Path(capture_stopped_view()).resolve()
                generation = self._components.append_coarse_view(view_path, self._generation)
            except Exception as exc:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = f"operator bootstrap failed closed: {exc}"
                self._emit()
                raise UnknownBladeExperimentError(self._attention) from exc
            self._generation = Path(generation).resolve()
            self._bootstrap_count += 1
            if occupancy_map_ready and self._bootstrap_count < self._minimum_bootstrap:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = (
                    "occupancy reported MAP_READY before the minimum independent "
                    "operator bootstrap count"
                )
                self._emit()
                raise UnknownBladeExperimentError(self._attention)
            if occupancy_map_ready:
                self._phase = UnknownBladeExperimentPhase.COARSE_SCAN
                self._attention = "occupancy MAP_READY; coarse endpoint selection is enabled"
            else:
                next_index = self._bootstrap_count + 1
                self._attention = (
                    "operator must reposition the stopped robot and trigger bootstrap "
                    f"capture {next_index}"
                )
            self._emit()
            return self.snapshot

    def select_coarse_view(self) -> NextViewSelection:
        """Select one endpoint-feasible target without granting trajectory permission."""

        with self._lock:
            self._require_phase(UnknownBladeExperimentPhase.COARSE_SCAN)
            if self._generation is None:
                raise UnknownBladeExperimentError("Coarse phase has no accepted generation")
            try:
                selection = self._components.select_coarse_view(
                    self._generation,
                    self._requires_fin,
                )
            except Exception as exc:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = f"coarse next-view selection failed closed: {exc}"
                self._emit()
                raise UnknownBladeExperimentError(self._attention) from exc
            if selection.coverage_complete:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = (
                    "coarse selector claimed completion before controller schema-5 promotion"
                )
                self._emit()
                raise UnknownBladeExperimentError(self._attention)
            self._attention = f"coarse target {selection.target.view_id} awaits guarded capture"
            self._emit()
            return selection

    def capture_selected_coarse_view(
        self,
        selection: NextViewSelection,
        capture_guarded_view: Callable[[NextViewSelection], str | Path],
    ) -> UnknownBladeExperimentSnapshot:
        """Run an injected guarded capture, append evidence, and evaluate promotion."""

        with self._lock:
            self._require_phase(UnknownBladeExperimentPhase.COARSE_SCAN)
            if selection.coverage_complete or selection.target is None:
                raise UnknownBladeExperimentError("Coarse capture requires an incomplete target")
            if self._generation is None:
                raise UnknownBladeExperimentError("Coarse phase has no predecessor generation")
            try:
                view_path = Path(capture_guarded_view(selection)).resolve()
                generation = Path(
                    self._components.append_coarse_view(view_path, self._generation)
                ).resolve()
                self._generation = generation
                self._phase = UnknownBladeExperimentPhase.BUILDING_SCHEMA5
                self._attention = "coarse evidence accepted; evaluating schema-5 gates"
                self._emit()
                transition = self._components.finalize_coarse(generation)
            except Exception as exc:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = f"coarse capture/update failed closed: {exc}"
                self._emit()
                raise UnknownBladeExperimentError(self._attention) from exc
            if transition.phase is CoarsePhase.READY_FOR_FINE:
                assert transition.reference_coarse_model_path is not None
                assert transition.ready_generation_path is not None
                # Constructing and validating the fine driver is the atomic handoff
                # point.  Controller state changes only after this succeeds.
                try:
                    driver = self._components.create_fine_driver(
                        transition.reference_coarse_model_path
                    )
                except Exception as exc:
                    self._phase = UnknownBladeExperimentPhase.BLOCKED
                    self._attention = f"schema-5 fine handoff failed closed: {exc}"
                    self._emit()
                    raise UnknownBladeExperimentError(self._attention) from exc
                self._fine_driver = driver
                self._generation = transition.ready_generation_path
                self._reference = transition.reference_coarse_model_path
                self._requires_fin = False
                self._phase = UnknownBladeExperimentPhase.FINE_SCAN
                self._attention = "schema-5 reference verified; fine phase is ready"
            elif transition.phase is CoarsePhase.COLLECTING_FIN_EVIDENCE:
                self._requires_fin = True
                self._phase = UnknownBladeExperimentPhase.COARSE_SCAN
                self._attention = "; ".join(transition.reasons)
            elif transition.phase is CoarsePhase.COLLECTING:
                self._requires_fin = False
                self._phase = UnknownBladeExperimentPhase.COARSE_SCAN
                self._attention = "; ".join(transition.reasons)
            else:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = "; ".join(transition.reasons)
            self._emit()
            return self.snapshot

    def step_fine(self) -> UnknownBladeExperimentSnapshot:
        """Advance one supervised fine-science step through the injected driver."""

        with self._lock:
            self._require_phase(UnknownBladeExperimentPhase.FINE_SCAN)
            if self._fine_driver is None:
                raise UnknownBladeExperimentError("Fine phase driver is unavailable")
            try:
                result = self._fine_driver.step()
            except Exception as exc:
                self._phase = UnknownBladeExperimentPhase.BLOCKED
                self._attention = f"fine phase failed closed: {exc}"
                self._emit()
                raise UnknownBladeExperimentError(self._attention) from exc
            self._attention = result.attention_reason
            if result.complete:
                self._phase = UnknownBladeExperimentPhase.COMPLETE
            self._emit()
            return self.snapshot
