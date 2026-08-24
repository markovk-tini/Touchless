"""Latency dashboard — per-stage p50/p95/p99 the subscriber can see.

Phase-10 subscription polish. Subscribers paying for Iris want to
SEE that their assistant feels fast. Today latency lives in scattered
log events with no consolidated view.

This module collects per-stage timings into a bounded ring buffer
+ exposes a one-line summary the chat panel can pin to its header
("ASR 280ms · plan 410ms · reply 1.2s").

Stages tracked:

  * `asr`         — wake → final transcript ready
  * `plan`        — final transcript → plan decided (planner LLM or
                    local_intent or fast_path)
  * `tool`        — plan → tools dispatched + results in
  * `reply`       — results → final text composed (prose_renderer)
  * `tts`         — text → first audible word (when streaming)
  * `total`       — wake → first audible word

Each stage stores recent samples (1000) and computes p50 / p95 /
p99 on demand.

API:

  * `record(stage, ms)` — orchestrator / realtime client calls.
  * `summary()` — returns dict of {stage: {p50, p95, p99, samples}}.
  * `one_line()` — terse human string for chat-panel pinning.
  * `reset()` — admin.

Storage is in-process only — restarts wipe. The metric pill is
for instant feedback, not historical analytics.

Author: Konstantin Markov
"""
from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


_RING_SIZE = 1000

_STAGES = (
    "asr", "plan", "tool", "reply", "tts", "total")


@dataclass
class _StageBuffer:
    samples: List[float] = field(default_factory=list)
    total_count: int = 0     # lifetime (not just buffer)
    total_sum_ms: float = 0.0

    def add(self, ms: float, ring_size: int = _RING_SIZE) -> None:
        if ms < 0:
            return
        self.samples.append(float(ms))
        self.total_count += 1
        self.total_sum_ms += float(ms)
        if len(self.samples) > ring_size:
            # Drop oldest. Simple O(n); buffer is small.
            self.samples = self.samples[-ring_size:]

    def percentile(self, p: float) -> Optional[float]:
        if not self.samples:
            return None
        sorted_samples = sorted(self.samples)
        if p <= 0:
            return sorted_samples[0]
        if p >= 100:
            return sorted_samples[-1]
        idx = int(round((p / 100.0) * (len(sorted_samples) - 1)))
        return sorted_samples[idx]

    def stats(self) -> Dict[str, Optional[float]]:
        return {
            "samples": len(self.samples),
            "lifetime": self.total_count,
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "avg": (self.total_sum_ms / self.total_count
                    if self.total_count > 0 else None),
        }


class LatencyDashboard:
    """In-process latency tracker. Thread-safe."""

    def __init__(self, *, ring_size: int = _RING_SIZE) -> None:
        self._ring_size = ring_size
        self._lock = threading.RLock()
        self._stages: Dict[str, _StageBuffer] = {
            s: _StageBuffer() for s in _STAGES}

    def record(self, stage: str, ms: float) -> None:
        if stage not in self._stages:
            return
        with self._lock:
            self._stages[stage].add(ms, ring_size=self._ring_size)

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        with self._lock:
            return {s: self._stages[s].stats()
                    for s in _STAGES}

    def one_line(self) -> str:
        """Human-readable summary — 'ASR 280ms · plan 410ms · …'.
        Only includes stages with data."""
        with self._lock:
            parts: List[str] = []
            for stage in _STAGES:
                buf = self._stages[stage]
                if not buf.samples:
                    continue
                p50 = buf.percentile(50)
                parts.append(f"{stage} {self._fmt_ms(p50)}")
        return " · ".join(parts) if parts else "no samples yet"

    @staticmethod
    def _fmt_ms(ms: Optional[float]) -> str:
        if ms is None:
            return "--"
        if ms < 1000:
            return f"{int(round(ms))}ms"
        return f"{ms / 1000.0:.1f}s"

    def reset(self) -> None:
        with self._lock:
            for s in _STAGES:
                self._stages[s] = _StageBuffer()


# ---- singleton ------------------------------------------------------

_lock = threading.Lock()
_singleton: Optional[LatencyDashboard] = None


def global_dashboard() -> LatencyDashboard:
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = LatencyDashboard()
        return _singleton


def reset_global() -> None:
    global _singleton
    with _lock:
        _singleton = None


def record(stage: str, ms: float) -> None:
    global_dashboard().record(stage, ms)


def one_line() -> str:
    return global_dashboard().one_line()


def summary() -> Dict[str, Dict[str, Optional[float]]]:
    return global_dashboard().summary()


# ---- timer helper ---------------------------------------------------

@dataclass
class StageTimer:
    """Context manager for recording elapsed wall-clock into a
    stage bucket:

        with StageTimer("plan"):
            plan = planner.plan(text)
    """
    stage: str
    start_ms: float = 0.0

    def __enter__(self) -> "StageTimer":
        import time
        self.start_ms = time.time() * 1000.0
        return self

    def __exit__(self, *_):
        import time
        delta = (time.time() * 1000.0) - self.start_ms
        if delta >= 0:
            record(self.stage, delta)
        return False
