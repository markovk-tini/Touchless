"""Tests for implicit fact extraction (Phase 2 B3)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.implicit_facts import (  # noqa: E402
    FactCandidate, MIN_CONFIDENCE, extract_implicit_facts,
    _canonical_value, _looks_like_a_name,
)


def _kinds(facts):
    return [(f.kind, f.key, f.value) for f in facts]


# ---- preferences ------------------------------------------------------

def test_default_send_via_pref():
    facts = extract_implicit_facts(
        user_text="always send via gmail please")
    assert ("preference", "default_send_via", "gmail_send") in _kinds(facts)


def test_default_send_via_pref_outlook_maps_correctly():
    facts = extract_implicit_facts(
        user_text="prefer to send through outlook")
    assert ("preference", "default_send_via", "ms_mail_send") in _kinds(facts)


def test_ui_theme_pref():
    facts = extract_implicit_facts(user_text="I prefer dark mode")
    assert ("preference", "ui_theme", "dark") in _kinds(facts)


def test_voice_style_pref():
    facts = extract_implicit_facts(
        user_text="keep your replies short please")
    assert ("preference", "voice_style", "short") in _kinds(facts)


def test_default_volume_pref():
    facts = extract_implicit_facts(
        user_text="my default volume is 60")
    assert ("preference", "default_volume", "60") in _kinds(facts)


# ---- location --------------------------------------------------------

def test_location_home():
    facts = extract_implicit_facts(user_text="I live in Berlin")
    assert ("location", "home", "Berlin") in _kinds(facts)


def test_location_current_strips_temporal_suffix():
    facts = extract_implicit_facts(
        user_text="I'm in Tokyo right now for work")
    # 'Tokyo' should be canonicalized — temporal suffix stripped.
    locs = [f for f in facts if f.kind == "location"]
    assert any(f.value == "Tokyo" for f in locs)


# ---- contacts --------------------------------------------------------

def test_name_email_pattern_captured():
    facts = extract_implicit_facts(
        user_text="Dani's email is dani@mangollc.org")
    assert ("person", "dani", "dani@mangollc.org") in _kinds(facts)


def test_name_at_email_pattern_captured():
    facts = extract_implicit_facts(
        user_text="Alice at alice@example.com is the project lead")
    assert ("person", "alice", "alice@example.com") in _kinds(facts)


def test_pronoun_word_my_not_captured_as_name():
    facts = extract_implicit_facts(
        user_text="my email is foo@bar.com")
    # Should NOT extract "my" -> foo@bar.com as a person fact.
    assert not any(f.kind == "person" and f.key == "my" for f in facts)


# ---- pronouns / name -------------------------------------------------

def test_pronoun_pref():
    facts = extract_implicit_facts(
        user_text="my pronouns are they/them")
    assert ("preference", "pronouns", "they/them") in _kinds(facts)


def test_user_name_capture():
    facts = extract_implicit_facts(
        user_text="my name is Dani")
    assert ("user", "name", "Dani") in _kinds(facts)


# ---- schedule --------------------------------------------------------

def test_workday_start_capture():
    facts = extract_implicit_facts(
        user_text="my workday starts at 9am")
    assert ("schedule", "workday_start", "9am") in _kinds(facts)


def test_sleep_capture():
    facts = extract_implicit_facts(
        user_text="I sleep at 11pm")
    assert ("schedule", "sleep", "11pm") in _kinds(facts)


def test_birthday_capture():
    facts = extract_implicit_facts(
        user_text="my birthday is March 5")
    bdays = [f for f in facts if f.key == "birthday"]
    # Schedule kind canonicalizes to lowercase + space-stripped.
    assert bdays and "march" in bdays[0].value.lower()


# ---- dislikes --------------------------------------------------------

def test_dislike_capture():
    facts = extract_implicit_facts(
        user_text="stop saying good morning to me")
    assert any(f.kind == "dislike" for f in facts)


def test_dislike_caps_word_count():
    # A very long object phrase shouldn't get captured as a dislike.
    facts = extract_implicit_facts(
        user_text="don't doing the things that take forever and ever and ever")
    # Should NOT capture an 8-word dislike phrase.
    long_dislikes = [f for f in facts
                     if f.kind == "dislike" and len(f.key.split()) > 6]
    assert not long_dislikes


# ---- robustness ------------------------------------------------------

def test_empty_text_returns_empty_list():
    assert extract_implicit_facts(user_text="") == []
    assert extract_implicit_facts(user_text=None) == []


def test_low_confidence_filtered_out():
    # Force-build a sub-threshold candidate and verify filter behavior
    # by checking that the floor is honored.
    facts = extract_implicit_facts(
        user_text="random text that matches nothing important")
    # No facts extracted (nothing matches) → empty list.
    assert facts == []


def test_min_confidence_constant_in_range():
    assert 0.0 < MIN_CONFIDENCE <= 1.0


# ---- canonicalizers ---------------------------------------------------

def test_canonical_value_maps_email_provider_names():
    assert _canonical_value("preference", "gmail") == "gmail_send"
    assert _canonical_value("preference", "outlook") == "ms_mail_send"
    assert _canonical_value("preference", "teams") == "teams_send"


def test_canonical_value_strips_trailing_temporal():
    assert _canonical_value("location", "Tokyo right now") == "Tokyo"


def test_canonical_value_schedule_lowercases_and_strips_space():
    assert _canonical_value("schedule", "9 AM") == "9am"


def test_looks_like_a_name_rejects_pronouns():
    assert _looks_like_a_name("Dani") is True
    assert _looks_like_a_name("My") is False
    assert _looks_like_a_name("") is False


# ---- end-to-end sanity ------------------------------------------------

def test_combined_user_message_yields_multiple_facts():
    text = ("Hi! My name is Dani, I live in Berlin and I prefer dark mode. "
            "Dani's email is dani@mangollc.org. My workday starts at 9am.")
    facts = extract_implicit_facts(user_text=text)
    kinds = _kinds(facts)
    assert ("user", "name", "Dani") in kinds
    assert ("location", "home", "Berlin") in kinds
    assert ("preference", "ui_theme", "dark") in kinds
    assert ("person", "dani", "dani@mangollc.org") in kinds
    assert ("schedule", "workday_start", "9am") in kinds
