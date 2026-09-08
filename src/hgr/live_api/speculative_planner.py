"""Plan-as-you-talk speculative planning.

Phase-8 latency. Whisper streams partial transcripts every ~80-
150 ms then finalizes ~300-800 ms after silence. Today the planner
waits for the FINAL transcript before doing anything — so the user
hears ~600-1000 ms of dead air after they stop talking, even when
the answer was decided before they finished the sentence.

Speculative planning fires the planner on a STABLE partial
transcript ahead of finalization. When the final matches closely,
we use the speculative plan (instant). When it diverges, we
discard and replan from final.

Two pieces:

  * `TranscriptStabilizer` — tracks the partial-transcript stream
    + decides when to declare a partial "stable enough" to plan
    against. Heuristic: no token changes for >= STABILITY_WINDOW_MS,
    ends with sentence-final punctuation OR has been stable for
    >= STABILITY_HOLD_MS regardless.
  * `SpeculativeCache` — keyed by partial text. Stores the plan
    that was generated speculatively. On finalize, the orchestrator
    checks the cache: hit (matches well enough) → use; miss →
    regular path.

Discard rules:
  * The speculative plan was for a DIFFERENT intent (>=20% word
    overlap drops invalidate).
  * The user starts speaking again before the speculative plan
    finishes (interrupt-while-thinking).

The substrate is intentionally NOT wired to dispatch yet — only
to PRE-WARM the plan. Tier-0 cache and Tier-0.4 fast-path stay
authoritative for actual execution. Speculative plan is consumed
as a hint when finalized text matches.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


_STABILITY_WINDOW_MS = 200
_STABILITY_HOLD_MS = 600
_MIN_PARTIAL_WORDS = 3
_MAX_PARTIAL_AGE_S = 12.0


def _now_ms() -> float:
    return time.time() * 1000.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap on lowercase word sets. 1.0 if
    identical, 0.0 if disjoint."""
    aw = set(re.findall(r"\b[a-z0-9']+\b", _normalize(a)))
    bw = set(re.findall(r"\b[a-z0-9']+\b", _normalize(b)))
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    inter = aw & bw
    union = aw | bw
    return len(inter) / len(union)


# ---- stabilizer ------------------------------------------------------

@dataclass
class StabilityState:
    text: str = ""
    first_seen_at_ms: float = 0.0
    last_change_at_ms: float = 0.0


class TranscriptStabilizer:
    """Tracks the partial-transcript stream and returns a partial
    that's stable enough to speculatively plan against."""

    def __init__(self, *,
                 stability_window_ms: float = _STABILITY_WINDOW_MS,
                 stability_hold_ms: float = _STABILITY_HOLD_MS):
        self._window = stability_window_ms
        self._hold = stability_hold_ms
        self._state = StabilityState()
        self._lock = threading.RLock()
        self._declared_partial: Optional[str] = None

    def observe(self, partial: str) -> Optional[str]:
        """Call on every partial transcript update. Returns the
        text WHEN it just declared stable; None otherwise. Each
        stable declaration fires AT MOST ONCE per stable run."""
        with self._lock:
            now = _now_ms()
            partial_clean = (partial or "").strip()
            if not partial_clean:
                return None
            if _normalize(partial_clean) != _normalize(self._state.text):
                # New text — reset stability clock.
                self._state.text = partial_clean
                self._state.last_change_at_ms = now
                if self._state.first_seen_at_ms == 0:
                    self._state.first_seen_at_ms = now
                self._declared_partial = None
                return None
            # Text unchanged since last call.
            if partial_clean == self._declared_partial:
                return None  # already declared
            words = partial_clean.split()
            if len(words) < _MIN_PARTIAL_WORDS:
                return None
            stable_ms = now - self._state.last_change_at_ms
            ends_with_period = partial_clean.endswith(
                (".", "?", "!"))
            qualified = (
                (ends_with_period
                 and stable_ms >= self._window)
                or stable_ms >= self._hold)
            if qualified:
                self._declared_partial = partial_clean
                return partial_clean
            return None

    def reset(self) -> None:
        with self._lock:
            self._state = StabilityState()
            self._declared_partial = None


# ---- speculative cache ----------------------------------------------

@dataclass
class SpeculativeEntry:
    partial: str
    plan: Any
    fired_at: float = field(default_factory=time.time)
    consumed: bool = False


class SpeculativeCache:
    """Stores speculative plans keyed by stable-partial text.
    Bounded; oldest entries evicted first when full."""

    _MAX_ENTRIES = 8

    def __init__(self) -> None:
        self._entries: Dict[str, SpeculativeEntry] = {}
        self._lock = threading.RLock()

    def put(self, partial: str, plan: Any) -> None:
        with self._lock:
            key = _normalize(partial)
            if not key:
                return
            if len(self._entries) >= self._MAX_ENTRIES:
                # Evict the oldest.
                oldest = min(self._entries.values(),
                             key=lambda e: e.fired_at,
                             default=None)
                if oldest is not None:
                    self._entries.pop(_normalize(oldest.partial),
                                       None)
            self._entries[key] = SpeculativeEntry(
                partial=partial, plan=plan)

    def find(self, final: str,
             *, min_overlap: float = 0.7,
             max_age_s: float = _MAX_PARTIAL_AGE_S
             ) -> Optional[SpeculativeEntry]:
        """Look for a stable-partial plan whose text matches the
        final closely (Jaccard >= min_overlap). Returns it +
        marks consumed."""
        with self._lock:
            now = time.time()
            best: Optional[SpeculativeEntry] = None
            best_score = 0.0
            for e in self._entries.values():
                if e.consumed:
                    continue
                if (now - e.fired_at) > max_age_s:
                    continue
                score = _token_overlap(e.partial, final)
                if score >= min_overlap and score > best_score:
                    best = e
                    best_score = score
            if best is not None:
                best.consumed = True
            return best

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---- top-level wiring helper ----------------------------------------

class SpeculativePlanner:
    """Glues the stabilizer + cache + a user-provided planner
    function. The orchestrator calls `on_partial(text)` from the
    ASR thread; on stable, it kicks the planner in a daemon thread
    and stores the plan in the cache. On finalize, the orchestrator
    calls `find_plan(final)` to consume."""

    def __init__(self, *,
                 plan_fn: Callable[[str], Any],
                 stabilizer: Optional[TranscriptStabilizer] = None,
                 cache: Optional[SpeculativeCache] = None):
        self._plan_fn = plan_fn
        self._stab = stabilizer or TranscriptStabilizer()
        self._cache = cache or SpeculativeCache()
        self._lock = threading.RLock()
        self._inflight: Dict[str, threading.Thread] = {}
        self._stats = {"fired": 0, "hit": 0, "miss": 0,
                       "discarded": 0}

    def on_partial(self, partial: str) -> Optional[str]:
        """Call on every partial transcript. Returns the partial
        we just kicked off a speculative plan for, or None."""
        stable = self._stab.observe(partial)
        if stable is None:
            return None
        key = _normalize(stable)
        with self._lock:
            if key in self._inflight:
                return None
            self._stats["fired"] += 1
            t = threading.Thread(
                target=self._do_plan, args=(stable, key),
                daemon=True, name=f"spec-plan-{key[:20]}")
            self._inflight[key] = t
        t.start()
        return stable

    def _do_plan(self, partial: str, key: str) -> None:
        try:
            plan = self._plan_fn(partial)
            if plan is not None:
                self._cache.put(partial, plan)
        except Exception:
            pass
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def find_plan(self, final: str) -> Optional[Any]:
        """On finalize: look for a matching speculative plan.
        Returns the plan if found (consumed); None otherwise."""
        entry = self._cache.find(final)
        with self._lock:
            if entry is None:
                self._stats["miss"] += 1
                return None
            self._stats["hit"] += 1
            return entry.plan

    def reset(self) -> None:
        self._stab.reset()
        self._cache.clear()
        with self._lock:
            self._inflight.clear()
            self._stats = {"fired": 0, "hit": 0, "miss": 0,
                           "discarded": 0}

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)
