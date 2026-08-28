"""Dependency-light diagnostic result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CheckLevel(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    level: CheckLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)
