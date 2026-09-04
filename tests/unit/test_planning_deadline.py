from __future__ import annotations

import pytest

from biblade_fusion.core.planning_deadline import (
    PlanningDeadlineExceeded,
    activate_planning_deadline,
    remaining_planning_time_s,
    require_planning_time,
)


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_deadline_is_absolute_cooperative_and_context_local() -> None:
    clock = _Clock()

    with activate_planning_deadline(
        started_monotonic_s=0.0,
        maximum_duration_s=1.0,
        monotonic_clock=clock,
    ):
        clock.now = 0.25
        assert remaining_planning_time_s("unit test") == pytest.approx(0.75)
        clock.now = 1.0
        with pytest.raises(PlanningDeadlineExceeded, match="cooperative"):
            require_planning_time("unit test expiry")

    assert remaining_planning_time_s("outside planning") is None
