"""Tests for StuckPatternDetector (Phase 3 B5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.stuck_pattern_detector import (  # noqa: E402
    StuckKind, StuckPatternDetector, StuckSignal,
    _stable_args_hash, _suggest_for_error, _suggest_for_action,
)
from hgr.live_api.tool_invocation import (  # noqa: E402
    InvocationBus, InvocationSource, ToolInvocation,
)


def _inv(tool: str, status: str = "error", error: str = "HTTP 401",
         args=None, incognito: bool = False) -> ToolInvocation:
    inv = ToolInvocation(
        invocation_id=f"inv-{tool}-{status}",
        turn_id="turn",
        tool=tool, args=args or {},
        source=InvocationSource.PLANNER,
        status=status, error=error if status == "error" else None,
    )
    if incognito:
        inv.extra["incognito"] = True
    return inv


# ---- repeated error ---------------------------------------------------

def test_repeated_error_fires_signal():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    bus = InvocationBus()
    d.attach_to_bus(bus)
    try:
        for _ in range(3):
            bus.publish(_inv("gmail_send", "error", "HTTP 401"))
    finally:
        d.detach_from_bus()
    assert any(s.kind == StuckKind.REPEATED_ERROR for s in fired)
    s = next(s for s in fired if s.kind == StuckKind.REPEATED_ERROR)
    assert s.error_class == "auth_revoked"
    assert s.tool == "gmail_send"
    assert "iris_setup_tool" in s.suggested_action


def test_isolated_errors_do_not_fire():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    bus = InvocationBus()
    d.attach_to_bus(bus)
    try:
        bus.publish(_inv("a", "error", "HTTP 401"))
        bus.publish(_inv("b", "error", "HTTP 503"))
    finally:
        d.detach_from_bus()
    assert not any(s.kind == StuckKind.REPEATED_ERROR for s in fired)


def test_incognito_invocations_ignored():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    bus = InvocationBus()
    d.attach_to_bus(bus)
    try:
        for _ in range(5):
            bus.publish(_inv("file_delete", "error", "HTTP 500",
                             incognito=True))
    finally:
        d.detach_from_bus()
    assert fired == []


# ---- repeated action (success path) ---------------------------------

def test_repeated_action_fires_signal():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    bus = InvocationBus()
    d.attach_to_bus(bus)
    try:
        for _ in range(5):
            bus.publish(_inv("volume_set", "ok", "",
                             args={"percent": 30}))
    finally:
        d.detach_from_bus()
    repeated = [s for s in fired if s.kind == StuckKind.REPEATED_ACTION]
    assert repeated, "expected REPEATED_ACTION signal"
    assert repeated[0].tool == "volume_set"


# ---- circular dialogue ------------------------------------------------

def test_circular_dialogue_fires_on_three_repeats():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    for _ in range(3):
        d.record_user_text("send Dani the report")
    assert any(s.kind == StuckKind.CIRCULAR_DIALOGUE for s in fired)


def test_two_repeats_does_not_fire_circular_dialogue():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    d.record_user_text("send the report")
    d.record_user_text("send the report")
    assert not any(s.kind == StuckKind.CIRCULAR_DIALOGUE for s in fired)


def test_different_user_text_does_not_fire():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    d.record_user_text("send dani")
    d.record_user_text("send alice")
    d.record_user_text("send bob")
    assert not any(s.kind == StuckKind.CIRCULAR_DIALOGUE for s in fired)


def test_record_user_text_ignores_empty():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    for _ in range(5):
        d.record_user_text("")
        d.record_user_text("   ")
    assert fired == []


def test_record_user_text_honors_incognito():
    # SEC-005 audit: private utterances must not enter the rolling
    # history (which is later attached to emitted signals).
    import hgr.live_api.incognito as incognito
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    incognito.set_incognito(True)
    try:
        for _ in range(5):
            d.record_user_text("very private query")
    finally:
        incognito.set_incognito(False)
    # No circular_dialogue should have fired, because nothing was
    # actually recorded.
    assert fired == []


def test_attach_to_bus_is_idempotent():
    # stuck-2 audit: re-attach must not leak the previous subscriber
    # OR deliver each invocation twice.
    d = StuckPatternDetector()
    bus = InvocationBus()
    d.attach_to_bus(bus)
    d.attach_to_bus(bus)  # second call — must replace, not add
    fired = []
    d.subscribe(lambda s: fired.append(s))
    try:
        for _ in range(3):
            bus.publish(_inv("gmail_send", "error", "HTTP 401"))
    finally:
        d.detach_from_bus()
    # Only one REPEATED_ERROR signal should fire (3 errors, not 6).
    repeated = [s for s in fired if s.kind.value == "repeated_error"]
    assert len(repeated) == 1


# ---- cooldown ---------------------------------------------------------

def test_cooldown_prevents_spam():
    d = StuckPatternDetector()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    bus = InvocationBus()
    d.attach_to_bus(bus)
    try:
        # Fire repeatedly — should only get 1 emission due to cooldown.
        for _ in range(10):
            bus.publish(_inv("gmail_send", "error", "HTTP 401"))
    finally:
        d.detach_from_bus()
    n_errs = sum(1 for s in fired if s.kind == StuckKind.REPEATED_ERROR)
    assert n_errs == 1


# ---- helpers ----------------------------------------------------------

def test_stable_args_hash_is_deterministic():
    h1 = _stable_args_hash({"a": 1, "b": 2})
    h2 = _stable_args_hash({"b": 2, "a": 1})  # different order
    assert h1 == h2


def test_stable_args_hash_changes_on_value_change():
    h1 = _stable_args_hash({"a": 1})
    h2 = _stable_args_hash({"a": 2})
    assert h1 != h2


def test_suggest_for_error_covers_known_classes():
    assert "contacts_search" in _suggest_for_error("gmail_send",
                                                    "recipient_invalid")
    assert "iris_setup_tool" in _suggest_for_error("gmail_send",
                                                    "auth_revoked")
    assert "backoff" in _suggest_for_error("x", "rate_limited")


def test_suggest_for_action_specializes_for_known_tools():
    assert "audio output" in _suggest_for_action("volume_set")
    assert "pin" in _suggest_for_action("open_url")
    assert "template" in _suggest_for_action("gmail_send")


# ---- reset ------------------------------------------------------------

def test_reset_clears_internal_state():
    d = StuckPatternDetector()
    for _ in range(3):
        d.record_user_text("hi")
    d.reset()
    fired = []
    d.subscribe(lambda s: fired.append(s))
    # After reset, two more "hi"s shouldn't trigger CIRCULAR_DIALOGUE
    # (we'd need ≥3 from the cleared state).
    d.record_user_text("hi")
    d.record_user_text("hi")
    assert not any(s.kind == StuckKind.CIRCULAR_DIALOGUE for s in fired)
