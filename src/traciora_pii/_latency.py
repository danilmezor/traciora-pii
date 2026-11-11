"""Per-detector latency aggregator (benchmark-only).

A ``LatencyRecorder`` is opt-in: it lives behind a ``ContextVar`` and is
``None`` by default, so production code paths pay only one attribute load
per timed block. Benchmarks attach a recorder around an experiment run
to surface per-detector seconds without touching call signatures.

Usage::

    from traciora_pii._latency import LatencyRecorder, install_recorder, reset_recorder, time_detector

    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    try:
        with time_detector("regex"):
            ...
    finally:
        reset_recorder(token)
    print(recorder.snapshot())

The contract is intentionally narrow: callers wrap detector entry points
with ``time_detector("name")``. When no recorder is installed,
``time_detector`` is a near-no-op generator.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar, Token


class LatencyRecorder:
    """In-memory per-detector timing aggregator.

    Not thread-safe by design — benchmark drivers run single-threaded
    and a lock per record() would dwarf the work being measured.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, detector_name: str, elapsed_s: float) -> None:
        self._samples[detector_name].append(elapsed_s)

    def snapshot(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name in sorted(self._samples):
            samples = self._samples[name]
            if not samples:
                continue
            ordered = sorted(samples)
            idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
            out[name] = {
                "calls": len(samples),
                "total_s": sum(samples),
                "mean_ms": 1000.0 * statistics.fmean(samples),
                "p95_ms": 1000.0 * ordered[idx],
            }
        return out

    def reset(self) -> None:
        self._samples.clear()


_active: ContextVar[LatencyRecorder | None] = ContextVar(
    "traciora_pii_latency_recorder", default=None,
)


@contextmanager
def time_detector(name: str):
    """Time the wrapped block iff a recorder is installed; otherwise no-op."""
    rec = _active.get()
    if rec is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec.record(name, time.perf_counter() - t0)


def install_recorder(recorder: LatencyRecorder) -> Token:
    """Attach *recorder* to the current context. Returns a reset token."""
    return _active.set(recorder)


def reset_recorder(token: Token) -> None:
    _active.reset(token)


def get_active_recorder() -> LatencyRecorder | None:
    return _active.get()
