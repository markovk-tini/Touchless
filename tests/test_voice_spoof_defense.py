"""Tests for VoiceSpoofDefense (Phase 2 B4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.voice_spoof_defense import (  # noqa: E402
    ConfirmChannel, VoiceSpoofDefense,
)


def _fresh() -> VoiceSpoofDefense:
    d = VoiceSpoofDefense()
    d.reset()
    return d


# ---- read tier is always allowed without confirm -----------------------

def test_read_op_allowed_without_second_channel():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="weather_get", args_hash="h",
        source="voice", destructiveness="read",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is True


def test_write_op_allowed_in_normal_mode_without_second_channel():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="todo_add", args_hash="h",
        source="voice", destructiveness="write",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is True


# ---- destructive REQUIRES second channel -------------------------------

def test_destructive_voice_op_requires_second_channel():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is False
    assert r.require_second_channel is True


def test_destructive_allowed_with_gesture_confirm():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r.allowed is True


def test_voice_confirm_does_NOT_satisfy_second_channel():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.VOICE)
    assert r.allowed is False
    assert "voice cannot confirm itself" in r.reason


def test_irreversible_also_requires_second_channel():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="wire_transfer", args_hash="h",
        source="voice", destructiveness="irreversible",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is False


def test_irreversible_with_keyboard_confirm_allowed():
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="wire_transfer", args_hash="h",
        source="voice", destructiveness="irreversible",
        confirm_channel=ConfirmChannel.KEYBOARD)
    assert r.allowed is True


def test_typed_confirm_rejected_for_destructive():
    # SEC-007 audit: TYPED is NOT a trusted second channel for
    # destructive ops (WM_CHAR injection in scope).
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.TYPED)
    assert r.allowed is False
    assert "trusted" in r.reason or "second channel" in r.reason


def test_unknown_destructiveness_string_fails_closed():
    # SEC-007 (e): unknown destructiveness values default to
    # "destructive" — never silently allow.
    d = _fresh()
    r = d.check_destructive_voice_op(
        tool="x", args_hash="h",
        source="voice", destructiveness="destructiv",  # typo
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is False
    assert r.require_second_channel is True


def test_args_overrides_caller_supplied_hash():
    # When `args` is passed, the hash is computed internally from
    # the canonical JSON form — caller's `args_hash` is ignored.
    d = _fresh()
    # First call with full args + gesture confirm.
    r1 = d.check_destructive_voice_op(
        tool="file_delete", args={"path": "/x/y.txt"},
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r1.allowed is True
    # Second call with the same args within window → repeat refused.
    r2 = d.check_destructive_voice_op(
        tool="file_delete", args={"path": "/x/y.txt"},
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r2.allowed is False
    assert "repeat" in r2.reason.lower()


def test_compute_args_hash_canonical():
    from hgr.live_api.voice_spoof_defense import compute_args_hash
    a = compute_args_hash({"a": 1, "b": 2})
    b = compute_args_hash({"b": 2, "a": 1})
    assert a == b  # key-order independent
    c = compute_args_hash({"a": 1, "b": 3})
    assert a != c


# ---- repeat-attack detection -------------------------------------------

def test_repeat_same_op_under_window_refuses():
    d = _fresh()
    # Both calls confirm via gesture so the second-channel check
    # doesn't refuse first.
    d.check_destructive_voice_op(
        tool="file_delete", args_hash="abc",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="abc",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r.allowed is False
    assert "repeat" in r.reason.lower()


def test_repeat_attack_triggers_heightened_mode():
    d = _fresh()
    d.check_destructive_voice_op(
        tool="file_delete", args_hash="x",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    d.check_destructive_voice_op(
        tool="file_delete", args_hash="x",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert d.is_in_heightened_mode() is True
    # Heightened mode now requires second channel for WRITE too.
    r = d.check_destructive_voice_op(
        tool="todo_add", args_hash="y",
        source="voice", destructiveness="write",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is False


def test_different_args_hash_is_not_a_repeat():
    d = _fresh()
    d.check_destructive_voice_op(
        tool="file_delete", args_hash="path-a",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="path-b",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r.allowed is True


# ---- TTS-loop ----------------------------------------------------------

def test_tts_loop_blocks_all_destructive_ops():
    d = _fresh()
    d.report_tts_loop_detected(duration_sec=1.0)
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r.allowed is False
    assert "TTS loop" in r.reason
    # SEC-007 (c): the user should know they can override via gesture.
    assert r.require_second_channel is True


def test_tts_loop_also_raises_suspicion():
    # SEC-007 (c) audit: sustained TTS-loop attack must escalate to
    # heightened mode, not just sit at the timeout.
    d = _fresh()
    d.report_tts_loop_detected(duration_sec=1.0)
    d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert d.is_in_heightened_mode() is True


def test_tts_loop_clears_after_duration():
    # Module enforces a 0.5s floor on TTS-loop holds (see source);
    # wait past that to verify auto-clear.
    d = _fresh()
    d.report_tts_loop_detected(duration_sec=0.05)
    time.sleep(0.6)
    r = d.check_destructive_voice_op(
        tool="file_delete", args_hash="h",
        source="voice", destructiveness="destructive",
        confirm_channel=ConfirmChannel.GESTURE)
    assert r.allowed is True


# ---- reset --------------------------------------------------------------

def test_reset_clears_state():
    d = _fresh()
    d.report_tts_loop_detected(duration_sec=60)
    d.reset()
    assert d.is_in_heightened_mode() is False
    # And we should not be in TTS-loop mode either.
    r = d.check_destructive_voice_op(
        tool="x", args_hash="h", source="voice",
        destructiveness="read",
        confirm_channel=ConfirmChannel.NONE)
    assert r.allowed is True
