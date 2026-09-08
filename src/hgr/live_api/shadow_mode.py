"""Shadow mode.

Phase-3. When a new planner version / new tool routing decision /
new heuristic ships, the question "will this regress today's
behavior?" matters more than any test suite can prove. Shadow mode
lets us run the NEW planner in parallel with the PRODUCTION
planner, log both decisions, and silently report any divergences
without ever actually changing the user-visible behavior.

How it works:
  * Caller wraps the production decision with `record_baseline`.
  * Caller also runs the shadow code path and calls `record_shadow`
    with the same `turn_id`.
  * If shadow != baseline, the divergence is persisted to a
    per-day log + bus event.
  * Optional `shadow_fraction` (0.0 - 1.0) gates how often shadow
    runs at all — useful for cost-sensitive shadow paths (LLM calls).

Reporting:
  * `divergence_rate(window_sec)` — last N seconds, % of paired
    decisions that disagreed.
  * `recent_divergences(limit)` — for the developer UI to inspect.
  * `summary()` — overall counters.

The shadow log is short-lived (bounded ring buffer). For
historical analysis use the audit log + chain-of-thought layer.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Divergence:
    turn_id: str
    user_text: str
    feature: str
    baseline: Any
    shadow: Any
    ts: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class ShadowSummary:
    feature: str
    pairs_compared: int = 0
    divergences: int = 0
    shadow_skipped: int = 0
    shadow_errors: int = 0

    @property
    def divergence_rate(self) -> float:
        return (self.divergences / self.pairs_compared
                if self.pairs_compared else 0.0)


class ShadowMode:
    """One ShadowMode instance per shadowed feature (e.g., 'planner',
    'classifier', 'model_router'). State is per-feature so two
    independent shadow experiments don't pollute each other."""

    DEFAULT_RING_SIZE = 200
    # SEC-007 audit: pending baselines older than this are dropped
    # so a missing shadow-record call can't leak the dict forever.
    PENDING_BASELINE_TTL_SEC = 120.0

    def __init__(self, feature: str, *,
                 shadow_fraction: float = 1.0,
                 ring_size: int = DEFAULT_RING_SIZE) -> None:
        self.feature = feature
        self._fraction = max(0.0, min(1.0, shadow_fraction))
        self._lock = threading.RLock()
        self._summary = ShadowSummary(feature=feature)
        self._divergences: List[Divergence] = []
        self._max_ring = ring_size
        # We track baselines awaiting their paired shadow record so
        # divergence detection works even when shadow code runs
        # slightly after baseline.
        self._pending_baseline: Dict[str, Dict[str, Any]] = {}
        # shadow-1 audit: windowed divergence_rate divided recent
        # divergences by ALL-TIME pairs_compared, masking real
        # regression spikes. Track per-pair timestamps in a parallel
        # deque so the windowed denominator is correct.
        from collections import deque
        self._pair_log: "deque[tuple]" = deque(maxlen=2_000)
        # Bresenham accumulator for fractional sampling (missed-by-
        # panel: round(1/fraction) miscalibrates at fractions like
        # 0.66 / 0.34 / 0.6 by 16-25 percentage points).
        self._sample_accum = 0.0

    # ---- config -------------------------------------------------------

    def set_fraction(self, fraction: float) -> None:
        with self._lock:
            self._fraction = max(0.0, min(1.0, fraction))

    def fraction(self) -> float:
        with self._lock:
            return self._fraction

    def should_run_shadow(self) -> bool:
        """Deterministic sampling: Bresenham-style accumulator so the
        long-run sample rate exactly tracks `fraction` even for non-
        clean ratios like 0.66 (previous `round(1/fraction)` step
        miscalibrated by up to 25%). Avoids `time.time()` /
        `random.random()` so test runs are reproducible."""
        with self._lock:
            if self._fraction <= 0.0:
                return False
            if self._fraction >= 1.0:
                return True
            self._sample_accum += self._fraction
            if self._sample_accum >= 1.0:
                self._sample_accum -= 1.0
                return True
            return False

    # ---- recording ----------------------------------------------------

    def record_baseline(self, *, turn_id: str, user_text: str,
                        decision: Any,
                        notes: str = "") -> None:
        # SEC-007 audit: honor incognito. Never persist raw user_text
        # or decision payloads (which can carry tool args = PII) into
        # the in-memory bookkeeping when the user is private.
        try:
            from .incognito import is_incognito
            if is_incognito():
                return
        except Exception:
            pass
        with self._lock:
            self._evict_stale_pending(time.time())
            self._pending_baseline[turn_id] = {
                "user_text": user_text,
                "decision": decision,
                "ts": time.time(),
                "notes": notes,
            }

    def _evict_stale_pending(self, now: float) -> None:
        """Drop pending baselines older than PENDING_BASELINE_TTL_SEC.
        Called on every insert + every shadow record. SEC-007."""
        ttl = self.PENDING_BASELINE_TTL_SEC
        cutoff = now - ttl
        # Iterate over a copy to allow safe deletion.
        stale = [k for k, v in self._pending_baseline.items()
                 if v.get("ts", 0) < cutoff]
        for k in stale:
            self._pending_baseline.pop(k, None)
            self._summary.shadow_skipped += 1

    def record_shadow(self, *, turn_id: str,
                      decision: Any,
                      error: Optional[str] = None) -> Optional[Divergence]:
        with self._lock:
            now = time.time()
            self._evict_stale_pending(now)
            baseline = self._pending_baseline.pop(turn_id, None)
            if baseline is None:
                # Caller recorded shadow without a baseline — defensive
                # bookkeeping but no comparison possible.
                self._summary.shadow_skipped += 1
                return None
            self._summary.pairs_compared += 1
            is_divergent = error is not None
            if error is not None:
                self._summary.shadow_errors += 1
                # Errors count as divergence (production disagreed).
            else:
                if _decisions_equal(baseline["decision"], decision):
                    self._pair_log.append((now, False))
                    return None
            self._summary.divergences += 1
            self._pair_log.append((now, True))
            div = Divergence(
                turn_id=turn_id,
                user_text=baseline["user_text"],
                feature=self.feature,
                baseline=baseline["decision"],
                shadow=decision if error is None else f"ERROR: {error}",
                notes=baseline.get("notes", ""),
            )
            self._divergences.append(div)
            if len(self._divergences) > self._max_ring:
                self._divergences.pop(0)
            return div

    def mark_shadow_skipped(self, turn_id: str = "") -> None:
        """Caller declined to run shadow this turn (sampling / cost)."""
        with self._lock:
            self._summary.shadow_skipped += 1
            if turn_id:
                self._pending_baseline.pop(turn_id, None)

    # ---- reporting ----------------------------------------------------

    def summary(self) -> ShadowSummary:
        with self._lock:
            # Return a copy so the caller can't mutate our state.
            return ShadowSummary(
                feature=self._summary.feature,
                pairs_compared=self._summary.pairs_compared,
                divergences=self._summary.divergences,
                shadow_skipped=self._summary.shadow_skipped,
                shadow_errors=self._summary.shadow_errors,
            )

    def divergence_rate(self, window_sec: float = 0.0) -> float:
        with self._lock:
            if window_sec <= 0.0:
                return self._summary.divergence_rate
            # shadow-1 audit: count BOTH numerator and denominator
            # within the window. Previous code divided recent
            # divergences by all-time pairs_compared, masking spikes.
            cutoff = time.time() - window_sec
            pairs_in_window = 0
            div_in_window = 0
            for ts, was_divergent in self._pair_log:
                if ts < cutoff:
                    continue
                pairs_in_window += 1
                if was_divergent:
                    div_in_window += 1
            if pairs_in_window == 0:
                return 0.0
            return div_in_window / pairs_in_window

    def recent_divergences(self, *, limit: int = 20) -> List[Divergence]:
        with self._lock:
            return list(self._divergences[-limit:])

    def reset(self) -> None:
        with self._lock:
            self._summary = ShadowSummary(feature=self.feature)
            self._divergences.clear()
            self._pending_baseline.clear()
            self._pair_log.clear()
            self._sample_accum = 0.0


# ---- helpers ----------------------------------------------------------

def _decisions_equal(a: Any, b: Any) -> bool:
    """Equality for shadow comparisons. Handles plain types,
    dataclasses, lists, dicts, and cross-type comparisons
    (dataclass vs dict, namedtuple vs tuple) — normalizes each side
    to a canonical dict / scalar before comparing (shadow-4 audit)."""
    if a == b:
        return True
    norm_a = _to_canonical(a)
    norm_b = _to_canonical(b)
    return norm_a == norm_b


def _to_canonical(v: Any) -> Any:
    """Normalize a decision to a comparable form. Dataclasses + objects
    with __dict__ become dicts; lists / tuples become lists of their
    canonicalized members; everything else passes through."""
    try:
        import dataclasses
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: _to_canonical(val)
                    for k, val in dataclasses.asdict(v).items()}
    except Exception:
        pass
    if hasattr(v, "_asdict"):  # namedtuple
        try:
            return {k: _to_canonical(val) for k, val in v._asdict().items()}
        except Exception:
            pass
    if hasattr(v, "__dict__"):
        try:
            return {k: _to_canonical(val) for k, val in vars(v).items()}
        except Exception:
            return v
    if isinstance(v, dict):
        return {k: _to_canonical(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_canonical(x) for x in v]
    return v
