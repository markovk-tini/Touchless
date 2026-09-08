"""Sentinel — in-process background daemon for Iris.

Phase-3. So far Iris is reactive: a user speaks/types, Iris
replies. The Sentinel layer adds a small in-process scheduler
that runs background "watchers" between user turns. Watchers
emit signals onto the InvocationBus where the UI / planner /
notification gate can react.

Examples of watchers (built in later P3 batches):
  * focus-tracking → publish FOCUS_CHANGED events
  * calendar polling → upcoming meeting in 5 min → fire a briefing
  * file watcher → repo just got new commit → cache repo context
  * stuck-pattern detector → user has retried the same thing 3x

The Sentinel itself is NOT a watcher. It's:
  1. A scheduler that runs registered watchers at configurable
     intervals on a daemon thread pool.
  2. A pause/resume mechanism so the user can quiet it.
  3. A "budget" so watchers can't monopolize CPU.
  4. A bus subscription so it knows when the user is mid-interaction
     and can throttle its own activity.

Critical design point: the Sentinel runs IN-PROCESS (same Python
process as the UI). Multi-process daemon-spawn is intentionally
out of scope for v1 — it complicates installer + crash recovery
without a clear win. When a future need (background AI inference
that takes seconds and shouldn't block UI) demands it, a separate
worker process can subscribe to the same bus over a socket
(Phase-3-late or Phase-4).

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class SentinelState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class WatcherSpec:
    """Registration record for one watcher."""
    name: str
    fn: Callable[[], None]
    interval_sec: float
    last_run_at: float = 0.0
    last_duration_ms: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    disabled: bool = False
    # Per-watcher cost budget — if any single run takes longer
    # than this, the Sentinel will skip the next N runs of this
    # watcher to keep the system responsive.
    max_run_ms: int = 250
    skip_next: int = 0
    runs: int = 0


@dataclass
class SentinelStats:
    state: str
    watchers: int
    total_runs: int
    total_failures: int
    average_run_ms: float = 0.0
    last_tick_at: float = 0.0


class Sentinel:
    """In-process background scheduler. Single tick thread runs the
    watcher loop; watcher work itself runs synchronously on that thread
    so each one's `interval_sec` is a soft floor, not a guarantee.

    Watchers that need long-running work should hand it off to their
    own thread pool — Sentinel is for cheap, periodic polls."""

    # How often the tick thread wakes up to check what's ready.
    TICK_INTERVAL_SEC = 0.5

    # Auto-disable a watcher after N consecutive failures so a buggy
    # one can't spam the bus or crash the loop. User can re-enable
    # via Sentinel.reset_failures(name).
    AUTO_DISABLE_AFTER = 5

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._watchers: Dict[str, WatcherSpec] = {}
        self._state: SentinelState = SentinelState.STOPPED
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_runs = 0
        self._total_failures = 0
        self._total_runtime_ms = 0
        self._last_tick_at = 0.0
        # Reference to the user-activity gate — when the user is
        # actively typing/speaking, Sentinel pauses non-essential
        # watchers. Wired via mark_user_busy().
        self._user_busy_until: float = 0.0

    # ---- registration -------------------------------------------------

    def register(self, name: str, fn: Callable[[], None],
                 interval_sec: float = 30.0,
                 max_run_ms: int = 250) -> None:
        """Register a watcher. Calling register() with an existing
        name UPDATES the spec (so a hot-reload scenario works)."""
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        with self._lock:
            existing = self._watchers.get(name)
            # First-tick stampede fix: when register() is called for
            # ~10 watchers in quick succession (Phase-3 wiring), the
            # default last_run_at=0.0 made every interval gate pass
            # on the very first tick, so all watchers ran back-to-
            # back on the sentinel daemon thread within ~500ms of
            # start(). Seed last_run_at = now for FRESH registrations
            # so the first run is delayed by a full interval_sec and
            # the load is spread across ticks. Re-registration (hot
            # reload) preserves the existing last_run_at as before so
            # an already-running watcher's cadence isn't reset.
            self._watchers[name] = WatcherSpec(
                name=name, fn=fn, interval_sec=float(interval_sec),
                max_run_ms=int(max_run_ms),
                last_run_at=(existing.last_run_at if existing
                             else time.time()),
            )

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._watchers.pop(name, None) is not None

    def reset_failures(self, name: str) -> bool:
        with self._lock:
            w = self._watchers.get(name)
            if w is None:
                return False
            w.failure_count = 0
            w.consecutive_failures = 0
            w.disabled = False
            return True

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._state == SentinelState.RUNNING:
                return
            # sentinel-3 audit: a recent stop() may have released the
            # lock before join() completed. If the old tick thread is
            # still alive, join it (with timeout) before spawning a
            # new one so two daemons don't briefly coexist.
            old_thread = self._thread
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=2.0)
        with self._lock:
            if self._state == SentinelState.RUNNING:
                return
            self._stop_evt.clear()
            self._state = SentinelState.RUNNING
            self._thread = threading.Thread(
                target=self._tick_loop, name="iris-sentinel",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            self._state = SentinelState.STOPPED
            self._stop_evt.set()
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def pause(self) -> None:
        with self._lock:
            if self._state == SentinelState.RUNNING:
                self._state = SentinelState.PAUSED

    def resume(self) -> None:
        with self._lock:
            if self._state == SentinelState.PAUSED:
                self._state = SentinelState.RUNNING

    def mark_user_busy(self, hold_sec: float = 4.0) -> None:
        """The UI/voice/planner calls this when the user is actively
        interacting. Sentinel throttles non-essential watchers for
        the next hold_sec."""
        with self._lock:
            self._user_busy_until = max(self._user_busy_until,
                                        time.time() + max(0.0, hold_sec))

    def user_is_busy(self) -> bool:
        with self._lock:
            return time.time() < self._user_busy_until

    # ---- introspection -----------------------------------------------

    def state(self) -> SentinelStats:
        with self._lock:
            avg = (self._total_runtime_ms / self._total_runs
                   if self._total_runs else 0.0)
            return SentinelStats(
                state=self._state.value,
                watchers=len(self._watchers),
                total_runs=self._total_runs,
                total_failures=self._total_failures,
                average_run_ms=avg,
                last_tick_at=self._last_tick_at,
            )

    def watcher_specs(self) -> List[WatcherSpec]:
        with self._lock:
            return list(self._watchers.values())

    # ---- tick loop ----------------------------------------------------

    def _tick_loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._lock:
                running = self._state == SentinelState.RUNNING
                snapshot = list(self._watchers.values()) if running else []
            self._last_tick_at = time.time()
            if running:
                for w in snapshot:
                    if self._stop_evt.is_set():
                        break
                    self._maybe_run(w)
            # Sleep until the next tick or stop.
            self._stop_evt.wait(self.TICK_INTERVAL_SEC)

    def _maybe_run(self, w: WatcherSpec) -> None:
        if w.disabled:
            return
        now = time.time()
        # sentinel-2 audit: skip_next was decremented on EVERY tick,
        # not on every scheduled run — so a watcher with
        # interval_sec=30 burned through its 3-tick skip in 1.5 sec
        # (way below the intended 90 sec breather). Only consume a
        # skip when we'd otherwise have fired (interval gate passed).
        interval_gate_ready = (now - w.last_run_at) >= w.interval_sec
        if not interval_gate_ready:
            return
        if w.skip_next > 0:
            with self._lock:
                w.skip_next -= 1
                # Push last_run_at forward so the next interval gate
                # check waits a full interval — this is what makes the
                # skip_next * interval_sec breather actually happen.
                w.last_run_at = now
            return
        if self.user_is_busy() and w.interval_sec >= 5.0:
            # Throttle slow watchers during user activity. Fast ones
            # (interval < 5s) keep running — they're presumed to be
            # critical (focus tracking, etc.).
            return
        # Run with timing + failure handling.
        t0 = time.time()
        ok = True
        try:
            w.fn()
        except Exception as exc:
            ok = False
            self._record_failure(w, exc)
        dur_ms = int((time.time() - t0) * 1000)
        with self._lock:
            w.last_run_at = time.time()
            w.last_duration_ms = dur_ms
            w.runs += 1
            self._total_runs += 1
            self._total_runtime_ms += dur_ms
            if ok:
                w.consecutive_failures = 0
            if dur_ms > w.max_run_ms:
                # Cost-budget exceeded: skip the next 3 runs of this
                # watcher to give the rest of the system breathing room.
                w.skip_next = 3

    def _record_failure(self, w: WatcherSpec, exc: Exception) -> None:
        with self._lock:
            w.failure_count += 1
            w.consecutive_failures += 1
            self._total_failures += 1
            if w.consecutive_failures >= self.AUTO_DISABLE_AFTER:
                w.disabled = True
        # Per missed-by-panel finding: shipped Touchless is a windowed
        # PyInstaller build with no stderr console. Route through the
        # project's structured logger when available, fall back to
        # stderr only when running un-bundled (dev/test).
        try:
            from .live_api_logger import get_fallback_logger
            logger = get_fallback_logger()
            logger.exception(
                "sentinel_watcher_failed", exc,
                watcher=w.name,
                consecutive_failures=w.consecutive_failures,
                disabled=w.disabled,
            )
        except Exception:
            import sys
            print(f"[sentinel] watcher {w.name!r} raised "
                  f"{type(exc).__name__}: {exc}\n"
                  f"{traceback.format_exc(limit=4)}",
                  file=sys.stderr)


# ---- module singleton --------------------------------------------------

_sentinel: Optional[Sentinel] = None
_lock = threading.Lock()


def global_sentinel() -> Sentinel:
    global _sentinel
    if _sentinel is None:
        with _lock:
            if _sentinel is None:
                _sentinel = Sentinel()
    return _sentinel


def _reset_for_tests() -> None:
    # gate-3 audit pattern: take the same lock the constructor uses
    # so a concurrent global_sentinel() call doesn't race the reset.
    global _sentinel
    with _lock:
        target = _sentinel
        _sentinel = None
    if target is not None:
        try:
            target.stop(timeout=0.5)
        except Exception:
            pass
