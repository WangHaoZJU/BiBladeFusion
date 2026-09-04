"""Cooperative wall-clock budget for one online planning transaction.

The deadline is deliberately context-local rather than a worker-thread timeout.  Robot
planning calls native Pinocchio/HPP-FCL/OMPL code whose state must not be abandoned by
asynchronously killing a thread.  Python loops poll this shared absolute deadline, and
bounded native calls receive the remaining time when their API supports it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic


class PlanningDeadlineExceeded(TimeoutError):
    """The active online planning transaction exhausted its responsiveness budget."""


@dataclass(frozen=True, slots=True)
class PlanningDeadline:
    """One absolute monotonic deadline shared by selector and path preflight."""

    started_monotonic_s: float
    expires_monotonic_s: float
    maximum_duration_s: float
    monotonic_clock: Callable[[], float]


_ACTIVE_PLANNING_DEADLINE: ContextVar[PlanningDeadline | None] = ContextVar(
    "biblade_fusion_active_planning_deadline",
    default=None,
)


@contextmanager
def activate_planning_deadline(
    *,
    started_monotonic_s: float,
    maximum_duration_s: float,
    monotonic_clock: Callable[[], float] = monotonic,
) -> Iterator[PlanningDeadline]:
    """Install one absolute deadline without consuming an additional clock sample."""

    started = float(started_monotonic_s)
    duration = float(maximum_duration_s)
    if not math.isfinite(started):
        raise ValueError("planning deadline start must be finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("planning deadline duration must be finite and positive")
    deadline = PlanningDeadline(started, started + duration, duration, monotonic_clock)
    token = _ACTIVE_PLANNING_DEADLINE.set(deadline)
    try:
        yield deadline
    finally:
        _ACTIVE_PLANNING_DEADLINE.reset(token)


def remaining_planning_time_s(stage: str) -> float | None:
    """Return remaining time, or raise when the active transaction has expired."""

    deadline = _ACTIVE_PLANNING_DEADLINE.get()
    if deadline is None:
        return None
    label = str(stage).strip() or "unspecified planning stage"
    now = float(deadline.monotonic_clock())
    if not math.isfinite(now):
        raise PlanningDeadlineExceeded(
            f"planning/preflight clock became non-finite ({label})"
        )
    if now < deadline.started_monotonic_s:
        raise PlanningDeadlineExceeded(
            f"planning/preflight clock moved backwards ({label})"
        )
    remaining = deadline.expires_monotonic_s - now
    if remaining <= 0.0:
        elapsed = now - deadline.started_monotonic_s
        raise PlanningDeadlineExceeded(
            "planning/preflight exceeded its cooperative responsiveness budget "
            f"({label}): actual={elapsed:.9g}s, "
            f"limit={deadline.maximum_duration_s:.9g}s"
        )
    return remaining


def require_planning_time(stage: str) -> None:
    """Cooperative cancellation point; a no-op outside online planning."""

    remaining_planning_time_s(stage)
