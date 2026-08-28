"""Environment and hardware diagnostics with a cycle-safe lazy doctor import."""

from __future__ import annotations

from typing import TYPE_CHECKING

from biblade_fusion.diagnostics.types import CheckLevel, CheckResult

if TYPE_CHECKING:
    from biblade_fusion.core.settings import AppSettings


def run_doctor(settings: AppSettings) -> list[CheckResult]:
    from biblade_fusion.diagnostics.doctor import run_doctor as implementation

    return implementation(settings)

__all__ = ["CheckLevel", "CheckResult", "run_doctor"]
