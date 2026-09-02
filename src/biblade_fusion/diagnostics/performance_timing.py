"""Low-overhead, non-authoritative performance timing diagnostics.

The recorder deliberately stores one aggregate per fixed span name instead of a
per-call event trace.  Its memory use is therefore bounded by ``maximum_spans``
and does not grow with ray count, robot-state sample count, or transaction time.
Timing output is diagnostic only: callers use :func:`write_best_effort` so ordinary
diagnostic filesystem failures cannot replace the observed return value or exception.
Serialization and publication are synchronous and therefore add a small amount of
latency; that overhead must be included when interpreting duration gates.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar
from uuid import uuid4

_P = ParamSpec("_P")
_R = TypeVar("_R")

_ACTIVE_RECORDER: ContextVar[PerformanceTimingRecorder | None] = ContextVar(
    "biblade_fusion_performance_timing_recorder",
    default=None,
)


def _proc_io_counters() -> dict[str, int] | None:
    """Return Linux process I/O counters when available, without failing a run."""

    try:
        counters: dict[str, int] = {}
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            name, raw_value = line.split(":", 1)
            counters[name.strip()] = int(raw_value.strip())
        return counters
    except (OSError, ValueError):
        return None


def process_resource_snapshot() -> dict[str, object]:
    """Capture small process-wide resource counters for offline diagnostics."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_cpu_ns": time.process_time_ns(),
        # Linux reports KiB; macOS reports bytes.  The project runs on Linux, but
        # retain the native unit explicitly so this diagnostic is never ambiguous.
        "maximum_resident_set_size_native": int(usage.ru_maxrss),
        "maximum_resident_set_size_unit": "KiB_on_linux_bytes_on_macos",
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "proc_io": _proc_io_counters(),
    }


@dataclass(slots=True)
class _Aggregate:
    count: int = 0
    failure_count: int = 0
    inclusive_wall_ns: int = 0
    exclusive_wall_ns: int = 0
    inclusive_cpu_ns: int = 0
    exclusive_cpu_ns: int = 0
    maximum_wall_ns: int = 0
    maximum_cpu_ns: int = 0

    def add(
        self,
        *,
        wall_ns: int,
        cpu_ns: int,
        child_wall_ns: int,
        child_cpu_ns: int,
        failed: bool,
    ) -> None:
        self.count += 1
        self.failure_count += int(failed)
        self.inclusive_wall_ns += wall_ns
        self.exclusive_wall_ns += max(0, wall_ns - child_wall_ns)
        self.inclusive_cpu_ns += cpu_ns
        self.exclusive_cpu_ns += max(0, cpu_ns - child_cpu_ns)
        self.maximum_wall_ns = max(self.maximum_wall_ns, wall_ns)
        self.maximum_cpu_ns = max(self.maximum_cpu_ns, cpu_ns)

    def payload(self) -> dict[str, int]:
        return {
            "count": self.count,
            "failure_count": self.failure_count,
            "inclusive_wall_ns": self.inclusive_wall_ns,
            "exclusive_wall_ns": self.exclusive_wall_ns,
            "inclusive_cpu_ns": self.inclusive_cpu_ns,
            "exclusive_cpu_ns": self.exclusive_cpu_ns,
            "maximum_wall_ns": self.maximum_wall_ns,
            "maximum_cpu_ns": self.maximum_cpu_ns,
        }


@dataclass(slots=True)
class _ActiveSpan:
    name: str
    started_wall_ns: int
    started_cpu_ns: int
    child_wall_ns: int = 0
    child_cpu_ns: int = 0


class PerformanceTimingRecorder:
    """Aggregate nested wall/CPU spans with fixed, bounded memory."""

    def __init__(
        self,
        *,
        transaction_kind: str,
        identity: Mapping[str, object] | None = None,
        maximum_spans: int = 64,
        monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
        process_time_ns_clock: Callable[[], int] = time.process_time_ns,
    ) -> None:
        kind = str(transaction_kind).strip()
        if not kind:
            raise ValueError("Timing transaction_kind must be non-empty")
        if isinstance(maximum_spans, bool) or maximum_spans < 1:
            raise ValueError("maximum_spans must be a positive integer")
        self._transaction_kind = kind
        self._identity = dict(identity or {})
        # Prove the diagnostic identity is small and JSON-serializable up front.
        json.dumps(self._identity, allow_nan=False)
        self._maximum_spans = int(maximum_spans)
        self._monotonic_ns = monotonic_ns_clock
        self._process_time_ns = process_time_ns_clock
        self._aggregates: dict[str, _Aggregate] = {}
        self._stack: list[_ActiveSpan] = []
        self._owner_thread_id: int | None = None
        self._started_resources = process_resource_snapshot()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Measure one nested span and update a fixed aggregate on exit."""

        span_name = str(name).strip()
        if not span_name:
            raise ValueError("Timing span name must be non-empty")
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif thread_id != self._owner_thread_id:
            raise RuntimeError("One timing recorder cannot span multiple threads")
        if span_name not in self._aggregates:
            if len(self._aggregates) >= self._maximum_spans:
                raise RuntimeError("Timing recorder exceeded its fixed span-name bound")
            self._aggregates[span_name] = _Aggregate()

        active = _ActiveSpan(
            span_name,
            self._monotonic_ns(),
            self._process_time_ns(),
        )
        self._stack.append(active)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            completed_cpu_ns = self._process_time_ns()
            completed_wall_ns = self._monotonic_ns()
            popped = self._stack.pop()
            if popped is not active:
                raise RuntimeError("Timing spans exited out of nesting order")
            wall_ns = max(0, completed_wall_ns - active.started_wall_ns)
            cpu_ns = max(0, completed_cpu_ns - active.started_cpu_ns)
            self._aggregates[span_name].add(
                wall_ns=wall_ns,
                cpu_ns=cpu_ns,
                child_wall_ns=active.child_wall_ns,
                child_cpu_ns=active.child_cpu_ns,
                failed=failed,
            )
            if self._stack:
                self._stack[-1].child_wall_ns += wall_ns
                self._stack[-1].child_cpu_ns += cpu_ns

    def payload(
        self,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, object]:
        """Return deterministic aggregate data; never include a per-call trace."""

        if self._stack:
            raise RuntimeError("Cannot serialize timing diagnostics with active spans")
        normalized_status = str(status).strip()
        if not normalized_status:
            raise ValueError("Timing status must be non-empty")
        completed_resources = process_resource_snapshot()
        return {
            "schema_version": 1,
            "artifact_kind": "biblade_fusion.performance_timing_diagnostic",
            "authority": "diagnostic_only_not_safety_or_science_authority",
            "transaction_kind": self._transaction_kind,
            "identity": dict(self._identity),
            "status": normalized_status,
            "error": None if error is None else str(error)[:1000],
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "bounded_storage": {
                "maximum_span_names": self._maximum_spans,
                "recorded_span_names": len(self._aggregates),
                "per_call_trace_retained": False,
            },
            "resource_start": self._started_resources,
            "resource_end": completed_resources,
            "spans": {
                name: aggregate.payload() for name, aggregate in sorted(self._aggregates.items())
            },
        }

    def write_best_effort(
        self,
        path: str | Path,
        *,
        status: str,
        error: str | None = None,
    ) -> bool:
        """Atomically publish a no-clobber diagnostic and absorb ordinary I/O errors."""

        target = Path(path)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
        try:
            payload = self.payload(status=status, error=error)
            encoded = (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
            # Publish without replacing an earlier transaction diagnostic.  The
            # hard-link operation is atomic on the same filesystem and fails with
            # FileExistsError when a retry resolves to the same diagnostic name.
            os.link(temporary, target)
            temporary.unlink()
            return True
        except Exception:
            with suppress(Exception):
                temporary.unlink(missing_ok=True)
            return False


@contextmanager
def activate_performance_timing(
    recorder: PerformanceTimingRecorder,
) -> Iterator[PerformanceTimingRecorder]:
    """Make one recorder available to deeply nested workflow helpers."""

    token: Token[PerformanceTimingRecorder | None] = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def performance_span(name: str) -> Iterator[None]:
    """Measure best-effort while preserving the operation's return/exception semantics."""

    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return
    context = recorder.span(name)
    try:
        context.__enter__()
    except Exception:
        yield
        return
    try:
        yield
    except BaseException as operation_error:
        # The original operation exception is authoritative.  A broken clock or
        # diagnostic aggregate must never replace or suppress it.
        with suppress(Exception):
            context.__exit__(
                type(operation_error),
                operation_error,
                operation_error.__traceback__,
            )
        raise
    else:
        with suppress(Exception):
            context.__exit__(None, None, None)


def try_create_performance_timing(
    *,
    transaction_kind: str,
    identity: Mapping[str, object] | None = None,
    maximum_spans: int = 64,
) -> PerformanceTimingRecorder | None:
    """Create instrumentation if possible, otherwise leave the operation untimed."""

    try:
        return PerformanceTimingRecorder(
            transaction_kind=transaction_kind,
            identity=identity,
            maximum_spans=maximum_spans,
        )
    except Exception:
        return None


def performance_timed(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate one coarse function boundary with non-interfering timing."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with performance_span(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def active_performance_timing() -> PerformanceTimingRecorder | None:
    """Return the context-local recorder for explicit diagnostic composition."""

    return _ACTIVE_RECORDER.get()


__all__ = [
    "PerformanceTimingRecorder",
    "activate_performance_timing",
    "active_performance_timing",
    "performance_span",
    "performance_timed",
    "process_resource_snapshot",
    "try_create_performance_timing",
]
