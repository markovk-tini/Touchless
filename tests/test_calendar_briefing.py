"""Tests for calendar_briefing (Phase 3 B3)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.calendar_briefing import (  # noqa: E402
    BriefingScheduler, CalendarEvent, build_briefing,
    _name_from_attendee, _summarize_agenda,
    _summarize_attendees, DEFAULT_LEAD_MINUTES,
)


def _evt(start_in_min: int, **kw) -> CalendarEvent:
    base_ts = 1_700_000_000
    return CalendarEvent(
        id=kw.pop("id", "evt-1"),
        title=kw.pop("title", "Sync"),
        start_ts=base_ts + start_in_min * 60,
        end_ts=kw.pop("end_ts", base_ts + (start_in_min + 30) * 60),
        location=kw.pop("location", ""),
        description=kw.pop("description", ""),
        attendees=kw.pop("attendees", []),
        organizer=kw.pop("organizer", ""),
        is_remote=kw.pop("is_remote", False),
        meeting_url=kw.pop("meeting_url", ""),
    )


def _now_for(evt: CalendarEvent, *, minus_min: int) -> float:
    return evt.start_ts - minus_min * 60


# ---- headline phrasing -------------------------------------------------

def test_briefing_headline_for_starting_now():
    evt = _evt(start_in_min=0, title="Standup")
    b = build_briefing(evt, now=evt.start_ts)
    assert "starting now" in b.headline


def test_briefing_headline_for_one_minute_singular():
    evt = _evt(start_in_min=1, title="1:1")
    b = build_briefing(evt, now=_now_for(evt, minus_min=1))
    assert "1 minute" in b.headline


def test_briefing_headline_for_multiple_minutes():
    evt = _evt(start_in_min=5, title="Demo")
    b = build_briefing(evt, now=_now_for(evt, minus_min=5))
    assert "5 minutes" in b.headline


# ---- attendees summary -----------------------------------------------

def test_attendee_summary_handles_one_two_many():
    evt1 = _evt(0, attendees=["Dani"])
    b1 = build_briefing(evt1, now=evt1.start_ts)
    assert "Dani" in b1.attendees_summary

    evt2 = _evt(0, attendees=["Dani", "Alice"])
    b2 = build_briefing(evt2, now=evt2.start_ts)
    assert "Dani and Alice" == b2.attendees_summary

    evt3 = _evt(0, attendees=["A", "B", "C", "D", "E", "F", "G"])
    b3 = build_briefing(evt3, now=evt3.start_ts)
    assert "other" in b3.attendees_summary


def test_attendee_email_parsed_to_name():
    assert _name_from_attendee("Dani Markov <dani@x.com>") == "Dani Markov"
    assert _name_from_attendee("dani@example.com") == "Dani"
    assert _name_from_attendee("Dani") == "Dani"
    assert _name_from_attendee("") == ""


def test_attendee_summary_dedupes_case_insensitive():
    evt = _evt(0, attendees=["Dani", "DANI", "dani"])
    b = build_briefing(evt, now=evt.start_ts)
    # Three identical names → "Dani" only once.
    assert b.attendees_summary == "Dani"


def test_attendees_falls_back_to_organizer():
    evt = _evt(0, attendees=[], organizer="Boss <boss@x>")
    b = build_briefing(evt, now=evt.start_ts)
    assert "organized by" in b.attendees_summary


# ---- agenda summary -------------------------------------------------

def test_agenda_strips_join_lines():
    desc = ("Join Zoom Meeting\nhttps://zoom.us/j/12345\n"
            "Meeting ID: 999\nDial-in: 555\n\n"
            "Agenda: Discuss Q3 metrics and release timeline.")
    summary = _summarize_agenda(desc)
    assert "Zoom" not in summary
    assert "Q3 metrics" in summary


def test_agenda_truncates_long():
    desc = "Detailed agenda " * 100
    summary = _summarize_agenda(desc)
    assert len(summary) <= 180
    assert summary.endswith("…")


def test_agenda_empty_input():
    assert _summarize_agenda("") == ""
    assert _summarize_agenda(None) == ""


# ---- location vs URL --------------------------------------------------

def test_remote_meeting_url_kept_off_spoken_body_but_on_briefing_field():
    # SEC-004 + CLAUDE.md memory feedback_iris_conversational_rules:
    # never read URLs aloud. The meeting_url stays on the Briefing
    # dataclass for the UI to render as a clickable link but is
    # NOT inserted into `body` (which is the TTS surface).
    evt = _evt(0, is_remote=True, meeting_url="https://zoom.us/j/1",
               location="Headquarters")
    b = build_briefing(evt, now=evt.start_ts)
    assert "https://zoom.us/j/1" not in b.body
    assert b.meeting_url == "https://zoom.us/j/1"


def test_attendee_url_injection_rejected():
    # SEC-010: a calendar invite with attendee='Click https://evil/login'
    # must NOT echo the URL into the spoken briefing.
    evt = _evt(0, attendees=["Click https://evil.example/login to join"])
    b = build_briefing(evt, now=evt.start_ts)
    assert "https://evil.example" not in b.body
    # The attacker's name shouldn't appear at all (rejected, not
    # title-cased).
    assert "evil.example" not in b.attendees_summary


def test_agenda_strips_embedded_api_key():
    # SEC-004: secrets embedded in meeting descriptions must be redacted.
    desc = ("Notes from Bob: API key for demo: AIzaSyABCDEFGHIJKLMNOPQRSTUV "
            "please don't share. Discussion of Q3 metrics.")
    evt = _evt(0, description=desc)
    b = build_briefing(evt, now=evt.start_ts)
    assert "AIzaSy" not in b.agenda_summary
    # Real content should still be present.
    assert "Q3" in b.agenda_summary or "metrics" in b.agenda_summary


def test_past_event_uses_started_ago_wording():
    # calbrief-1: events that started ≥2 minutes ago must NOT be
    # announced as "starting now".
    evt = _evt(-5, title="Late Standup")
    b = build_briefing(evt, now=evt.start_ts + 5 * 60)
    assert "started" in b.headline.lower()
    assert "ago" in b.headline.lower()


def test_title_quotes_stripped_for_TTS():
    # Missed-by-panel: ASCII " is read literally as "quote" by SAPI.
    evt = _evt(5, title='"Project Phoenix" sync')
    b = build_briefing(evt, now=_now_for(evt, minus_min=5))
    assert '"' not in b.headline


def test_in_person_meeting_uses_location():
    evt = _evt(0, is_remote=False, location="Room 5")
    b = build_briefing(evt, now=evt.start_ts)
    assert "Where: Room 5" in b.body


# ---- recent email threads inclusion ----------------------------------

def test_recent_email_threads_appear_in_body():
    evt = _evt(5, title="Pricing call")
    b = build_briefing(evt, now=_now_for(evt, minus_min=5),
                       recent_email_threads=[
                           "Q3 pricing proposal",
                           "RFP follow-up",
                       ])
    assert "Q3 pricing proposal" in b.body
    assert "RFP follow-up" in b.body


def test_recent_email_threads_capped_at_three():
    evt = _evt(5)
    threads = [f"thread-{i}" for i in range(10)]
    b = build_briefing(evt, now=_now_for(evt, minus_min=5),
                       recent_email_threads=threads)
    body = b.body
    assert "thread-0" in body
    assert "thread-2" in body
    assert "thread-3" not in body


# ---- BriefingScheduler ------------------------------------------------

def test_scheduler_is_eligible_first_time():
    s = BriefingScheduler()
    assert s.is_eligible("e1", now=1000) is True


def test_scheduler_not_eligible_within_cooldown():
    s = BriefingScheduler()
    s.mark_briefed("e1", now=1000)
    assert s.is_eligible("e1", now=1100) is False  # 100s < 30min cooldown


def test_scheduler_eligible_after_cooldown():
    s = BriefingScheduler()
    s.mark_briefed("e1", now=1000)
    later = 1000 + 30 * 60 + 1
    assert s.is_eligible("e1", now=later) is True


def test_upcoming_within_returns_only_in_window():
    base = 2_000_000_000
    e1 = _evt(2, id="soon")
    e1.start_ts = base + 2 * 60
    e2 = _evt(30, id="later")
    e2.start_ts = base + 30 * 60
    e3 = _evt(-5, id="past")  # already started
    e3.start_ts = base - 5 * 60
    s = BriefingScheduler()
    upcoming = s.upcoming_within(
        [e1, e2, e3], lead_minutes=5, now=base)
    ids = [e.id for e in upcoming]
    assert ids == ["soon"]
