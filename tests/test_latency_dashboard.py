"""Tests for latency_dashboard (Phase 10 B1)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.latency_dashboard import (  # noqa: E402
    LatencyDashboard, StageTimer, _StageBuffer, one_line,
    record, reset_global, summary,
)


def setup_function():
    reset_global()


# ---- buffer math ---------------------------------------------------

def test_buffer_records_sample():
    b = _StageBuffer()
    b.add(100.0)
    assert len(b.samples) == 1
    assert b.total_count == 1


def test_buffer_drops_oldest_at_limit():
    b = _StageBuffer()
    for i in range(1200):
        b.add(float(i), ring_size=1000)
    assert len(b.samples) == 1000
    assert b.total_count == 1200


def test_buffer_skips_negative_samples():
    b = _StageBuffer()
    b.add(-10)
    assert b.samples == []
    assert b.total_count == 0


def test_buffer_percentile_basic():
    b = _StageBuffer()
    for v in range(1, 101):
        b.add(float(v))
    # Indexing chooses round((p/100) * (n-1)) → small off-by-one
    # for round-half-to-even; close enough.
    assert 49 <= b.percentile(50) <= 52
    assert 94 <= b.percentile(95) <= 96
    assert 98 <= b.percentile(99) <= 100


def test_buffer_percentile_empty_returns_none():
    b = _StageBuffer()
    assert b.percentile(50) is None


def test_buffer_avg():
    b = _StageBuffer()
    b.add(100); b.add(200); b.add(300)
    stats = b.stats()
    assert stats["avg"] == 200


# ---- dashboard ---------------------------------------------------

def test_dashboard_record_unknown_stage_skips():
    d = LatencyDashboard()
    d.record("nonexistent_stage", 100)
    s = d.summary()
    assert s["asr"]["samples"] == 0


def test_dashboard_records_to_stage():
    d = LatencyDashboard()
    d.record("plan", 350)
    s = d.summary()
    assert s["plan"]["samples"] == 1
    assert s["plan"]["p50"] == 350


def test_dashboard_isolates_stages():
    d = LatencyDashboard()
    d.record("plan", 100)
    d.record("tool", 500)
    s = d.summary()
    assert s["plan"]["p50"] == 100
    assert s["tool"]["p50"] == 500


def test_dashboard_reset():
    d = LatencyDashboard()
    d.record("plan", 100)
    d.reset()
    s = d.summary()
    assert s["plan"]["samples"] == 0


def test_one_line_no_samples():
    d = LatencyDashboard()
    line = d.one_line()
    assert "no samples" in line


def test_one_line_with_data():
    d = LatencyDashboard()
    d.record("plan", 400)
    d.record("tool", 800)
    line = d.one_line()
    assert "plan" in line
    assert "tool" in line


def test_one_line_skips_empty_stages():
    d = LatencyDashboard()
    d.record("plan", 100)
    line = d.one_line()
    # Should mention 'plan' but NOT 'tool'.
    assert "plan" in line
    assert "tool" not in line


def test_one_line_formats_milliseconds():
    d = LatencyDashboard()
    d.record("plan", 423)
    line = d.one_line()
    assert "423ms" in line


def test_one_line_formats_seconds_for_large():
    d = LatencyDashboard()
    d.record("plan", 1500)
    line = d.one_line()
    assert "1.5s" in line


# ---- StageTimer context ------------------------------------------

def test_stage_timer_records_elapsed():
    with StageTimer("plan"):
        time.sleep(0.05)
    s = summary()
    assert s["plan"]["samples"] == 1
    assert s["plan"]["p50"] >= 45


def test_stage_timer_unknown_stage_safe():
    # Unknown stages get filtered by record(); StageTimer itself
    # doesn't validate — but the no-op should not raise.
    with StageTimer("bogus"):
        pass


# ---- global facade ------------------------------------------------

def test_global_record_and_summary():
    record("asr", 280)
    record("plan", 410)
    s = summary()
    assert s["asr"]["p50"] == 280
    assert s["plan"]["p50"] == 410


def test_global_one_line_aggregates():
    record("asr", 200)
    record("plan", 500)
    line = one_line()
    assert "asr" in line and "plan" in line


def test_lifetime_count_persists_across_buffer_drop():
    d = LatencyDashboard(ring_size=10)
    for i in range(100):
        d.record("plan", float(i))
    s = d.summary()
    assert s["plan"]["lifetime"] == 100
    # Buffer dropped older samples, so percentile reflects only
    # last 10.
    assert s["plan"]["p50"] >= 90
