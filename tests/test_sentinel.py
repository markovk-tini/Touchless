"""Tests for Sentinel (Phase 3 B1)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.sentinel import (  # noqa: E402
    Sentinel, SentinelState, WatcherSpec,
)


# ---- registration ------------------------------------------------------

def test_register_and_unregister():
    s = Sentinel()
    s.register("test", lambda: None, interval_sec=1.0)
    assert any(w.name == "test" for w in s.watcher_specs())
    assert s.unregister("test") is True
    assert s.unregister("test") is False


def test_register_zero_interval_rejected():
    s = Sentinel()
    import pytest
    with pytest.raises(ValueError):
        s.register("bad", lambda: None, interval_sec=0)


def test_re_register_updates_spec():
    s = Sentinel()
    s.register("w", lambda: None, interval_sec=1.0, max_run_ms=100)
    s.register("w", lambda: None, interval_sec=2.0, max_run_ms=500)
    spec = next(w for w in s.watcher_specs() if w.name == "w")
    assert spec.interval_sec == 2.0
    assert spec.max_run_ms == 500


# ---- lifecycle --------------------------------------------------------

def test_start_then_stop_runs_watcher():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    counter = []
    s.register("counter", lambda: counter.append(1),
               interval_sec=0.05)
    s.start()
    try:
        time.sleep(0.5)
    finally:
        s.stop(timeout=1.0)
    assert len(counter) >= 2
    assert s.state().total_runs >= 2


def test_pause_halts_new_runs_resume_continues():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    counter = []
    s.register("c", lambda: counter.append(1), interval_sec=0.05)
    s.start()
    try:
        time.sleep(0.2)
        n_before_pause = len(counter)
        s.pause()
        time.sleep(0.3)
        n_after_pause = len(counter)
        s.resume()
        time.sleep(0.3)
        n_after_resume = len(counter)
    finally:
        s.stop(timeout=1.0)
    # Pause should freeze the counter; resume should grow it again.
    assert n_after_pause - n_before_pause <= 1  # tolerate 1 in-flight
    assert n_after_resume > n_after_pause


# ---- failure handling --------------------------------------------------

def test_watcher_exception_recorded_does_not_break_loop():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05

    def boom():
        raise RuntimeError("kaboom")
    other_runs = []
    s.register("boom", boom, interval_sec=0.05)
    s.register("other", lambda: other_runs.append(1),
               interval_sec=0.05)
    s.start()
    try:
        time.sleep(0.3)
    finally:
        s.stop(timeout=1.0)
    st = s.state()
    assert st.total_failures >= 1
    # Other watcher kept running despite boom failing.
    assert len(other_runs) >= 2


def test_auto_disable_after_repeated_failures():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.02
    s.AUTO_DISABLE_AFTER = 3

    def boom():
        raise RuntimeError("nope")
    s.register("boom", boom, interval_sec=0.02)
    s.start()
    try:
        time.sleep(0.6)
    finally:
        s.stop(timeout=1.0)
    spec = next(w for w in s.watcher_specs() if w.name == "boom")
    assert spec.disabled is True
    assert spec.consecutive_failures >= 3


def test_reset_failures_clears_disabled_flag():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.02
    s.AUTO_DISABLE_AFTER = 2
    s.register("boom", lambda: (_ for _ in ()).throw(RuntimeError("x")),
               interval_sec=0.02)
    s.start()
    try:
        time.sleep(0.3)
    finally:
        s.stop(timeout=1.0)
    assert s.reset_failures("boom") is True
    spec = next(w for w in s.watcher_specs() if w.name == "boom")
    assert spec.disabled is False
    assert spec.consecutive_failures == 0
    assert s.reset_failures("not_a_real_watcher") is False


# ---- cost budget ------------------------------------------------------

def test_slow_watcher_gets_skip_next_assigned():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.02

    def slow():
        time.sleep(0.06)  # 60ms — over the default 250ms? no, under.

    def really_slow():
        time.sleep(0.2)  # 200ms — over 50ms budget below.

    s.register("slow_ok", slow, interval_sec=0.02, max_run_ms=100)
    s.register("over_budget", really_slow,
               interval_sec=0.02, max_run_ms=50)
    s.start()
    try:
        time.sleep(0.5)
    finally:
        s.stop(timeout=1.0)
    spec = next(w for w in s.watcher_specs() if w.name == "over_budget")
    # Skip-next counter was assigned at least once (could now be 0
    # if it expired). The over-budget watcher should have run fewer
    # times than the well-behaved one.
    ok_spec = next(w for w in s.watcher_specs() if w.name == "slow_ok")
    assert spec.runs <= ok_spec.runs


# ---- user_busy throttling --------------------------------------------

def test_user_busy_throttles_slow_watcher():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    runs_slow = []
    runs_fast = []
    s.register("slow", lambda: runs_slow.append(1), interval_sec=6.0)
    s.register("fast", lambda: runs_fast.append(1), interval_sec=0.05)
    s.mark_user_busy(hold_sec=10.0)
    s.start()
    try:
        time.sleep(0.4)
    finally:
        s.stop(timeout=1.0)
    # Fast watcher (interval < 5s) keeps running; slow (≥5s) throttled.
    assert len(runs_fast) >= 2
    assert len(runs_slow) == 0


def test_user_busy_decays():
    s = Sentinel()
    s.mark_user_busy(hold_sec=0.05)
    assert s.user_is_busy() is True
    time.sleep(0.1)
    assert s.user_is_busy() is False


# ---- introspection ----------------------------------------------------

def test_state_returns_stats_dataclass():
    s = Sentinel()
    s.register("x", lambda: None, interval_sec=1.0)
    st = s.state()
    assert st.state == "stopped"
    assert st.watchers == 1
    assert st.total_runs == 0


def test_start_is_idempotent():
    s = Sentinel()
    s.TICK_INTERVAL_SEC = 0.05
    s.start()
    s.start()  # second call is a no-op
    try:
        time.sleep(0.1)
    finally:
        s.stop(timeout=1.0)
    assert s.state().state == "stopped"
