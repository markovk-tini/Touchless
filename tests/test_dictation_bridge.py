"""Tests for DictationIrisBridge (Phase 2 B5)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.dictation_bridge import (  # noqa: E402
    BridgeEvent, BridgeEventKind, DictationIrisBridge,
)


def _fresh() -> DictationIrisBridge:
    b = DictationIrisBridge()
    b.reset()
    return b


# ---- pub/sub roundtrip -------------------------------------------------

def test_publish_delivers_to_subscriber():
    b = _fresh()
    received = []

    def cb(evt):
        received.append(evt)

    b.subscribe(cb)
    evt = BridgeEvent(kind=BridgeEventKind.DICTATION_STARTED)
    b.publish(evt)
    assert len(received) == 1
    assert received[0].kind == BridgeEventKind.DICTATION_STARTED


def test_unsubscribe_stops_delivery():
    b = _fresh()
    received = []
    unsub = b.subscribe(lambda e: received.append(e))
    b.publish(BridgeEvent(kind=BridgeEventKind.DICTATION_STARTED))
    unsub()
    b.publish(BridgeEvent(kind=BridgeEventKind.DICTATION_STOPPED))
    assert len(received) == 1


def test_subscriber_exception_isolated():
    b = _fresh()
    good = []
    b.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("kaboom")))
    b.subscribe(lambda e: good.append(e))
    b.publish(BridgeEvent(kind=BridgeEventKind.DICTATION_STARTED))
    assert len(good) == 1


# ---- dictation text tracking ------------------------------------------

def test_last_dictation_text_returned_after_publish():
    b = _fresh()
    b.publish_dictation_text("hello world", window_title="Slack")
    assert b.last_dictation_text() == "hello world"
    assert b.last_dictation_window() == "Slack"


def test_last_dictation_text_expires_after_max_age():
    b = _fresh()
    b.publish_dictation_text("hi", window_title="Notes")
    # Force-old it.
    b._last_dictation_ts = time.time() - 120
    assert b.last_dictation_text(max_age_sec=60.0) is None


def test_last_dictation_text_none_when_empty():
    b = _fresh()
    assert b.last_dictation_text() is None


# ---- iris command bookkeeping -----------------------------------------

def test_publish_iris_command_recorded():
    b = _fresh()
    b.publish_iris_command(
        command="send email", tool="gmail_send",
        result_summary="Sent to dani@x")
    evts = b.recent_events(kind=BridgeEventKind.IRIS_COMMAND_ISSUED)
    assert len(evts) == 1
    assert evts[0].payload["tool"] == "gmail_send"


def test_recent_events_filtered_by_kind():
    b = _fresh()
    b.publish_dictation_text("a")
    b.publish_iris_command("cmd-1")
    b.publish_dictation_text("b")
    dict_only = b.recent_events(
        kind=BridgeEventKind.DICTATION_TEXT_INSERTED)
    iris_only = b.recent_events(
        kind=BridgeEventKind.IRIS_COMMAND_ISSUED)
    assert len(dict_only) == 2
    assert len(iris_only) == 1


def test_recent_events_respects_limit():
    b = _fresh()
    for i in range(20):
        b.publish_iris_command(f"cmd-{i}")
    evts = b.recent_events(
        kind=BridgeEventKind.IRIS_COMMAND_ISSUED, limit=5)
    assert len(evts) == 5
    # Should be the LAST 5 (newest tail).
    assert evts[-1].payload["command"] == "cmd-19"


def test_history_buffer_capped():
    b = _fresh()
    b._max_recent = 10
    for i in range(20):
        b.publish_iris_command(f"cmd-{i}")
    # Total recent events should be capped at 10.
    assert len(b.recent_events(limit=100)) <= 10


def test_reset_clears_state():
    b = _fresh()
    b.publish_dictation_text("hi")
    b.publish_iris_command("c")
    b.reset()
    assert b.last_dictation_text() is None
    assert b.recent_events() == []
