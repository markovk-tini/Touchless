"""Tests for TranscriptionRouter (Phase 2 B4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.transcription_router import (  # noqa: E402
    TranscriptionDecision, TranscriptionRouter, TranscriptTier,
    _starts_with_destructive, _why_ambiguous,
)


# ---- core routing ------------------------------------------------------

def test_short_clean_clip_routes_to_fast():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=1.5, fast_text="what time is it")
    assert d.tier == TranscriptTier.FAST


def test_long_clip_routes_to_accurate():
    r = TranscriptionRouter(long_clip_sec=4.0)
    d = r.decide(audio_seconds=8.0, fast_text="hi")
    assert d.tier == TranscriptTier.ACCURATE
    assert "long" in d.reason


def test_email_in_text_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0,
                 fast_text="send a note to alice@example.com")
    assert d.tier == TranscriptTier.ACCURATE
    assert "email" in d.reason


def test_url_in_text_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0,
                 fast_text="open touchless-control.com")
    assert d.tier == TranscriptTier.ACCURATE
    assert "URL" in d.reason or "domain" in d.reason


def test_path_in_text_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0,
                 fast_text=r"open C:\Users\K\Documents\budget.xlsx")
    assert d.tier == TranscriptTier.ACCURATE


def test_digit_run_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0,
                 fast_text="call 555 1234 8899")
    assert d.tier == TranscriptTier.ACCURATE


def test_destructive_verb_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0,
                 fast_text="delete the old backup")
    assert d.tier == TranscriptTier.ACCURATE
    assert "destructive" in d.reason


def test_low_confidence_escalates():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0, fast_text="hi",
                 fast_confidence=0.3)
    assert d.tier == TranscriptTier.ACCURATE
    assert "confidence" in d.reason


def test_high_confidence_stays_fast():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0, fast_text="hi",
                 fast_confidence=0.9)
    assert d.tier == TranscriptTier.FAST


def test_verbatim_param_forces_accurate():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=1.0, fast_text="hi",
                 user_requested_verbatim=True)
    assert d.tier == TranscriptTier.ACCURATE
    assert d.show_fast_preview is False


def test_verbatim_env_forces_accurate(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_VERBATIM_TRANSCRIPTION", "1")
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=1.0, fast_text="hi")
    assert d.tier == TranscriptTier.ACCURATE


def test_no_fast_text_falls_back_to_length_routing():
    r = TranscriptionRouter()
    d = r.decide(audio_seconds=2.0, fast_text=None)
    assert d.tier == TranscriptTier.FAST


# ---- helper functions --------------------------------------------------

def test_why_ambiguous_detects_email():
    assert _why_ambiguous("ping alice@x.com") == "contains email"


def test_why_ambiguous_clean_returns_empty():
    assert _why_ambiguous("what's the weather") == ""


def test_starts_with_destructive_picks_up_common_verbs():
    assert _starts_with_destructive("delete the file")
    assert _starts_with_destructive("Send the report")
    assert _starts_with_destructive("transfer 100 to bob")
    assert not _starts_with_destructive("what is the weather")


def test_starts_with_destructive_strips_politeness_prefix():
    # SEC-012 audit: 'please delete the file' must still escalate.
    assert _starts_with_destructive("please delete the file")
    assert _starts_with_destructive("can you send the report")
    assert _starts_with_destructive("could you wipe the cache")
    # And the SEC-012 expanded verb list.
    assert _starts_with_destructive("revoke the API token")
    assert _starts_with_destructive("format the drive")
    assert _starts_with_destructive("kill that process")
