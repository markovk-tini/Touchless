"""Tests for EarconDispatcher (Phase 3 wiring)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.earcon_wiring import (  # noqa: E402
    EarconDispatcher, _ALWAYS_DONE_TOOLS, _SKIP_DONE_TOOLS,
)
from hgr.live_api.earcons import EarconKind  # noqa: E402
from hgr.live_api.tool_invocation import (  # noqa: E402
    InvocationBus, InvocationSource, ToolInvocation,
)


class _FakePlayer:
    """In-memory earcon player used by the tests — never touches audio."""

    def __init__(self):
        self.played = []

    def play(self, kind):
        self.played.append(kind)
        return True


def _inv(tool: str, status: str, *, incognito: bool = False,
         invocation_id: str = "inv-1") -> ToolInvocation:
    inv = ToolInvocation(
        invocation_id=invocation_id, turn_id="turn", tool=tool,
        args={}, source=InvocationSource.PLANNER, status=status)
    if incognito:
        inv.extra["incognito"] = True
    return inv


def _make_dispatcher(player=None):
    return EarconDispatcher(player=player or _FakePlayer())


# ---- resolution table --------------------------------------------------

def test_ok_for_always_done_tool_plays_done():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("gmail_send", "ok"))
    assert p.played == [EarconKind.DONE]


def test_ok_for_skip_tool_plays_nothing():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("weather_get", "ok"))
    assert p.played == []


def test_ok_for_unknown_tool_plays_done_by_default():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("some_new_connector_tool", "ok"))
    assert p.played == [EarconKind.DONE]


def test_error_always_plays_error():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("weather_get", "error",
                          invocation_id="e1"))
    assert p.played == [EarconKind.ERROR]


def test_cancelled_plays_declined():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("gmail_send", "cancelled"))
    assert p.played == [EarconKind.DECLINED]


def test_needs_confirmation_plays_needs_confirm():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("file_delete", "needs_confirmation"))
    assert p.played == [EarconKind.NEEDS_CONFIRM]


def test_starting_plays_nothing():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("gmail_send", "starting"))
    assert p.played == []


# ---- incognito + dedup -------------------------------------------------

def test_incognito_suppresses_everything():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    d._on_invocation(_inv("gmail_send", "ok", incognito=True))
    d._on_invocation(_inv("file_delete", "error",
                          incognito=True, invocation_id="x2"))
    assert p.played == []


def test_same_invocation_status_dedupes():
    """The bus publishes the same ToolInvocation twice (start + final).
    The dispatcher must only earcon ONCE per (invocation_id, status)."""
    p = _FakePlayer()
    d = _make_dispatcher(p)
    inv = _inv("gmail_send", "ok", invocation_id="dup-1")
    d._on_invocation(inv)
    d._on_invocation(inv)  # exact same status — no second earcon
    assert p.played == [EarconKind.DONE]


def test_starting_then_ok_each_resolved_independently():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    start = _inv("gmail_send", "starting", invocation_id="t1")
    final = _inv("gmail_send", "ok", invocation_id="t1")
    d._on_invocation(start)  # starting → no earcon
    d._on_invocation(final)  # transition to ok → DONE
    assert p.played == [EarconKind.DONE]


# ---- attach + detach idempotency --------------------------------------

def test_attach_to_bus_idempotent_no_double_play():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    bus = InvocationBus()
    d.attach_to_bus(bus)
    d.attach_to_bus(bus)  # second call detaches first
    bus.publish(_inv("gmail_send", "ok"))
    assert p.played == [EarconKind.DONE]


def test_detach_stops_delivery():
    p = _FakePlayer()
    d = _make_dispatcher(p)
    bus = InvocationBus()
    d.attach_to_bus(bus)
    d.detach_from_bus()
    bus.publish(_inv("gmail_send", "ok"))
    assert p.played == []


# ---- table sanity ------------------------------------------------------

def test_always_done_tools_dont_overlap_skip_set():
    overlap = _ALWAYS_DONE_TOOLS & _SKIP_DONE_TOOLS
    assert overlap == set()


def test_always_done_includes_destructive_sends():
    # Sanity: send/post/upload tools must always chime done.
    must_chime = {"gmail_send", "ms_mail_send", "slack_post",
                  "drive_upload", "file_delete"}
    assert must_chime <= _ALWAYS_DONE_TOOLS


# ---- speak-while-iris-talking suppression ----------------------------

def test_done_suppressed_while_iris_is_speaking(monkeypatch):
    """When the realtime model is mid-response, suppress DONE chimes
    so they don't overlap her voice. ERROR / NEEDS_CONFIRM still fire
    via the force-through-quiet set."""
    p = _FakePlayer()
    d = _make_dispatcher(p)
    monkeypatch.setattr(d, "_iris_currently_speaking", lambda: True)
    d._on_invocation(_inv("gmail_send", "ok"))
    assert p.played == []  # DONE suppressed
    # ERROR still fires.
    d._on_invocation(_inv("file_delete", "error",
                          invocation_id="e1"))
    assert EarconKind.ERROR in p.played


def test_done_fires_when_iris_not_speaking(monkeypatch):
    p = _FakePlayer()
    d = _make_dispatcher(p)
    monkeypatch.setattr(d, "_iris_currently_speaking", lambda: False)
    d._on_invocation(_inv("gmail_send", "ok"))
    assert p.played == [EarconKind.DONE]
