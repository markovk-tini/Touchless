"""Tests for cost_surfaces formatters (Phase 3 B5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.cost_surfaces import (  # noqa: E402
    CostBadge, cost_badge_state, format_breakdown,
    format_history, format_today_summary, over_cap_message,
)


class _FakeMeter:
    def __init__(self, *, spent: float, cap: float,
                 by_model=None, history=None):
        self._spent = spent
        self.daily_cap_usd = cap
        self._by_model = by_model or {}
        self._history = history or []

    def today_total(self):
        return self._spent

    def by_model_today(self):
        # Pass through the shape the test set — dict OR list of dicts
        # (the real CostMeter returns a list of dicts; the formatter
        # accepts both).
        if isinstance(self._by_model, list):
            return list(self._by_model)
        return dict(self._by_model)

    def history_last_n_days(self, n):
        return list(self._history)[:n]


# ---- badge state -------------------------------------------------------

def test_badge_green_when_under_50pct():
    m = _FakeMeter(spent=1.0, cap=5.0)
    st = cost_badge_state(m)
    assert st.badge == CostBadge.GREEN
    assert st.percent_of_cap == 20.0


def test_badge_yellow_between_50_and_80pct():
    m = _FakeMeter(spent=3.0, cap=5.0)
    assert cost_badge_state(m).badge == CostBadge.YELLOW


def test_badge_orange_at_85pct():
    m = _FakeMeter(spent=4.25, cap=5.0)
    assert cost_badge_state(m).badge == CostBadge.ORANGE


def test_badge_red_at_or_over_cap():
    m = _FakeMeter(spent=5.0, cap=5.0)
    assert cost_badge_state(m).badge == CostBadge.RED
    m2 = _FakeMeter(spent=6.0, cap=5.0)
    assert cost_badge_state(m2).badge == CostBadge.RED


def test_badge_handles_zero_cap_gracefully():
    m = _FakeMeter(spent=1.0, cap=0.0)
    st = cost_badge_state(m)
    # Division-by-zero safety check.
    assert st.percent_of_cap == 0.0


# ---- format_today_summary ---------------------------------------------

def test_format_today_summary_text():
    m = _FakeMeter(spent=0.42, cap=5.0)
    out = format_today_summary(m)
    assert "$0.42" in out
    assert "$5.00" in out
    assert "8%" in out


# ---- format_breakdown -------------------------------------------------

def test_format_breakdown_sorts_by_spend_desc():
    m = _FakeMeter(spent=0.5, cap=5,
                   by_model={"haiku": 0.05,
                             "sonnet": 0.30,
                             "gpt5mini": 0.15})
    out = format_breakdown(m)
    # sonnet (highest) should appear before haiku.
    sonnet_pos = out.find("sonnet")
    haiku_pos = out.find("haiku")
    assert sonnet_pos < haiku_pos


def test_format_breakdown_caps_to_top_n():
    m = _FakeMeter(spent=0.5, cap=5,
                   by_model={f"m{i}": 0.01 for i in range(10)})
    out = format_breakdown(m, top_n=3)
    # Three model lines + the header.
    assert out.count("\n") == 3  # 1 header + 3 items


def test_format_breakdown_empty():
    m = _FakeMeter(spent=0, cap=5)
    out = format_breakdown(m)
    assert "No paid model calls today" in out


# ---- format_history ---------------------------------------------------

def test_format_history_renders_lines():
    m = _FakeMeter(spent=1, cap=5,
                   history=[{"date": "2026-06-01", "usd": 0.5},
                            {"date": "2026-06-02", "usd": 1.2}])
    out = format_history(m, days=7)
    assert "2026-06-01" in out
    assert "$0.50" in out
    assert "$1.20" in out


def test_format_history_empty():
    m = _FakeMeter(spent=0, cap=5, history=[])
    assert format_history(m, days=7) == ""


# ---- over_cap_message -------------------------------------------------

def test_over_cap_message_includes_spent_and_cap():
    m = _FakeMeter(spent=5.50, cap=5.0)
    msg = over_cap_message(m)
    assert "$5.50" in msg
    assert "$5.00" in msg
    # Missed-by-panel audit: wording changed from "tomorrow" to
    # "local midnight" so users near UTC/DST boundaries aren't
    # confused.
    assert "midnight" in msg.lower()
    assert "local model" in msg.lower()


def test_over_cap_message_when_cap_is_zero():
    m = _FakeMeter(spent=1.0, cap=0.0)
    msg = over_cap_message(m)
    # Should NOT crash with division-by-zero or print "$0.00 of $0.00".
    assert "cap" in msg.lower() and "0" in msg


def test_format_today_summary_no_cap_set():
    m = _FakeMeter(spent=2.34, cap=0.0)
    out = format_today_summary(m)
    assert "$2.34" in out
    assert "no cap" in out.lower()


def test_format_breakdown_accepts_list_shape():
    # Missed-by-panel: real CostMeter.by_model_today() returns a list
    # of dicts with 'model'+'cost_usd' keys. The formatter must accept
    # both shapes (dict OR list) without raising.
    m = _FakeMeter(spent=0.5, cap=5)
    m._by_model = [
        {"model": "haiku", "cost_usd": 0.05},
        {"model": "sonnet", "cost_usd": 0.30},
    ]
    out = format_breakdown(m)
    assert "sonnet" in out
    assert "haiku" in out


def test_format_history_accepts_cost_usd_key():
    # Missed-by-panel: real meter uses 'cost_usd', not 'usd'.
    m = _FakeMeter(spent=1, cap=5,
                   history=[{"date": "2026-06-01", "cost_usd": 0.75}])
    out = format_history(m, days=7)
    assert "2026-06-01" in out
    assert "$0.75" in out


def test_format_history_skips_malformed_rows():
    m = _FakeMeter(spent=1, cap=5,
                   history=[{"date": "2026-06-01", "cost_usd": 1.0},
                            {"missing": True},
                            {"date": "2026-06-02"}])  # no cost
    out = format_history(m, days=7)
    assert "2026-06-01" in out
    # Malformed rows should not appear, but shouldn't crash either.
    assert "missing" not in out
