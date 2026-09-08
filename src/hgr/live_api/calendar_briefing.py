"""Calendar briefings.

Phase-3 Sentinel-driven. Runs periodically; reads the user's
upcoming meetings from a connector (Google Calendar, MS365); when
an event is starting soon, builds a short briefing block (title,
who's on it, agenda from the description, last 3 emails from
attendees) and dispatches it through the InterruptionGate.

This module owns the BUILDER logic — turning a calendar event into
a structured briefing — and exposes a `build_briefing(event, ...)`
function. The actual "should we interrupt" check is delegated to
InterruptionGate; the actual "fetch upcoming events" is delegated
to whatever calendar connector is registered.

Design: pure function for build_briefing so it's testable in
isolation. A separate `BriefingScheduler` glues it to the Sentinel
+ connector + gate.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional


# How long before a meeting we surface the briefing.
DEFAULT_LEAD_MINUTES = 5
# Don't re-brief the same event within this window.
COOLDOWN_PER_EVENT_SEC = 30 * 60


@dataclass
class CalendarEvent:
    """Minimal calendar event shape; connectors normalize to this."""
    id: str
    title: str
    start_ts: float       # epoch seconds
    end_ts: float = 0.0
    location: str = ""
    description: str = ""
    attendees: List[str] = field(default_factory=list)
    organizer: str = ""
    is_remote: bool = False
    meeting_url: str = ""


@dataclass
class Briefing:
    event_id: str
    title: str
    starts_in_minutes: int
    headline: str
    attendees_summary: str = ""
    location: str = ""
    meeting_url: str = ""
    agenda_summary: str = ""
    recent_email_threads: List[str] = field(default_factory=list)
    body: str = ""

    def as_text(self) -> str:
        return self.body or self.headline


def build_briefing(event: CalendarEvent, *,
                   now: Optional[float] = None,
                   recent_email_threads:
                   Optional[List[str]] = None) -> Briefing:
    """Pure function — given an event + optional context, render
    the briefing's user-facing text. No connector / clock side
    effects so tests can inject `now`.

    SEC-004 / SEC-010 audit: attendee strings, agenda text, and
    title are SCRUBBED of URLs, API-key patterns, and embedded
    secrets before being included in any spoken-aloud field. The
    meeting URL is preserved separately on the Briefing dataclass
    (`meeting_url`) so a UI can render it as a clickable link, but
    is NEVER inserted into `body` (which is read aloud) — per
    CLAUDE.md memory feedback_iris_conversational_rules.
    """
    when = now if now is not None else time.time()
    starts_in = int((event.start_ts - when) // 60)
    # Use the typographic-quote-free form so TTS doesn't say "quote"
    # (missed-by-panel finding).
    safe_title = _scrub_for_voice(event.title or "(untitled event)")
    if starts_in < -1:
        # calbrief-1 audit: don't lie about events that started
        # minutes ago. Surface the lateness so the user can react.
        late = -starts_in
        headline = f"{safe_title} started {late} minute" \
                   f"{'s' if late != 1 else ''} ago."
    elif starts_in <= 0:
        headline = f"{safe_title} is starting now."
    elif starts_in == 1:
        headline = f"{safe_title} starts in 1 minute."
    else:
        headline = f"{safe_title} starts in {starts_in} minutes."

    attendees_summary = _summarize_attendees(
        event.attendees, organizer=event.organizer)

    agenda = _summarize_agenda(event.description)

    threads = list(recent_email_threads or [])[:3]
    threads_block = ""
    if threads:
        # Scrub each thread subject too — it could contain a URL.
        safe_threads = [_scrub_for_voice(t) for t in threads if t]
        if safe_threads:
            threads_block = ("\nRecent email context:\n  - "
                             + "\n  - ".join(safe_threads))

    location_block = ""
    if event.location:
        location_block = f"\nWhere: {_scrub_for_voice(event.location)}"

    body_parts = [headline]
    if attendees_summary:
        body_parts.append(f"With: {attendees_summary}")
    if location_block:
        body_parts.append(location_block.lstrip("\n"))
    if agenda:
        body_parts.append(f"Agenda: {agenda}")
    if threads_block:
        body_parts.append(threads_block.lstrip("\n"))
    # Note: meeting_url is INTENTIONALLY not added to body — UI may
    # render the dataclass field separately as a clickable link.
    body = "\n".join(body_parts)

    return Briefing(
        event_id=event.id,
        title=safe_title,
        starts_in_minutes=max(0, starts_in),
        headline=headline,
        attendees_summary=attendees_summary,
        location=event.location,
        meeting_url=event.meeting_url,
        agenda_summary=agenda,
        recent_email_threads=threads,
        body=body,
    )


def _summarize_attendees(attendees: List[str],
                         organizer: str = "") -> str:
    """Best-effort attendee summary. 'Dani, Alice and 3 others'
    style — no email addresses spoken aloud."""
    cleaned: List[str] = []
    seen = set()
    for a in (attendees or []):
        name = _name_from_attendee(a)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append(name)
    if not cleaned:
        if organizer:
            o = _name_from_attendee(organizer)
            return f"organized by {o}" if o else ""
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    if len(cleaned) <= 5:
        return f"{', '.join(cleaned[:-1])} and {cleaned[-1]}"
    extras = len(cleaned) - 4
    return (f"{', '.join(cleaned[:4])} and {extras} other"
            + ("s" if extras != 1 else ""))


_EMAIL_RE = re.compile(r"<?([A-Za-z0-9._%+\-]+)@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}>?")


def _name_from_attendee(a: str) -> str:
    """Get a friendly name from an attendee string. Handles formats:
       'Dani Markov <dani@x>', 'dani@x', 'Dani'.

    SEC-010 audit: when no explicit email-bracket / email-domain
    structure is present, REJECT free-form attendee strings that
    look like attacker-injected text (URLs, too-long phrases, prompt
    payloads) and fall back to a generic label."""
    s = (a or "").strip()
    if not s:
        return ""
    if "<" in s and ">" in s:
        head = s.split("<", 1)[0].strip().strip('"')
        if head and _looks_like_safe_name(head):
            return head
    m = _EMAIL_RE.search(s)
    if m:
        local = m.group(1)
        return local.split(".")[0].title() if local else ""
    # Plain free-form attendee. Only accept if it really looks like a
    # name; otherwise drop to a generic label so attacker-controlled
    # invites can't inject URLs / prompts into TTS.
    if _looks_like_safe_name(s):
        return s.title() if s.islower() else s
    return ""


def _looks_like_safe_name(s: str) -> bool:
    """Reject anything that smells like an injection vector: URLs,
    control chars, overly long phrases, anything with weird symbols.
    The name token must be ≤40 chars, contain only letter/space/.-'
    characters."""
    if not s or len(s) > 40:
        return False
    if "://" in s or s.startswith("http"):
        return False
    if any(ord(c) < 32 for c in s):
        return False
    # Allow letters, spaces, common name punctuation.
    import re as _re
    return bool(_re.fullmatch(r"[A-Za-z][A-Za-z .,'\-]*", s))


# Patterns we strip from any free-form text before it reaches TTS.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}|pk-[A-Za-z0-9]{16,}|"
    r"AIza[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{16,}|"
    r"xox[bp]-[A-Za-z0-9_\-]{16,}|"
    r"Bearer\s+[A-Za-z0-9._\-]{20,})"
)
# Generic high-entropy ≥32-char alnum runs (look like tokens/secrets).
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]{32,}\b")
_PASSCODE_RE = re.compile(
    r"\bpass(?:code|word)\s*[:=]?\s*\S+", re.IGNORECASE)


def _scrub_for_voice(text: str) -> str:
    """Strip URLs, API keys, passcodes, and high-entropy tokens from
    text destined for TTS / LLM-context fields. SEC-004/SEC-010
    audit. Replaces matches with '[redacted]'."""
    if not text:
        return ""
    s = text
    s = _URL_RE.sub("[link]", s)
    s = _API_KEY_RE.sub("[redacted]", s)
    s = _LONG_TOKEN_RE.sub("[redacted]", s)
    s = _PASSCODE_RE.sub("[redacted]", s)
    # Drop ASCII double-quotes which TTS engines read literally.
    s = s.replace('"', "")
    return s


def _summarize_agenda(description: str) -> str:
    """Pull a 1-2 sentence agenda from a meeting description.
    Strips out Zoom join blocks, calendar links, telephony PINs."""
    desc = (description or "").strip()
    if not desc:
        return ""
    # SEC-004 audit: scrub URLs / API keys / passcodes even when they
    # appear mid-line. The line-skip patterns below only catch
    # well-formed boilerplate (start-of-line); the regex pass above
    # catches embedded secrets.
    desc = _scrub_for_voice(desc)
    # Drop noisy lines.
    lines = []
    skip_patterns = (
        r"^join\s+(zoom|teams|webex|meet)",
        r"^meeting\s+id",
        r"^passcode\b",
        r"^one\s+tap\s+mobile",
        r"^dial\s*-?\s*in",
        r"^https?\s*[:/]",
        r"^_{3,}",                          # horizontal rules
        r"^microsoft\s+teams\s+need\s+help",
    )
    skip_re = re.compile("|".join(skip_patterns), re.IGNORECASE)
    for ln in desc.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if skip_re.search(ln):
            continue
        lines.append(ln)
        if len(lines) >= 4:
            break
    summary = " ".join(lines)
    # Trim to ~180 chars.
    if len(summary) > 180:
        summary = summary[:177].rsplit(" ", 1)[0] + "…"
    return summary


class BriefingScheduler:
    """Manages briefing cooldowns + dispatch. Glue layer; tests cover
    the per-event de-dup logic without needing a real calendar
    connector."""

    def __init__(self) -> None:
        self._last_briefed_at: Dict[str, float] = {}

    def is_eligible(self, event_id: str,
                    *, now: Optional[float] = None) -> bool:
        when = now if now is not None else time.time()
        last = self._last_briefed_at.get(event_id)
        if last is None:
            return True  # never briefed → always eligible
        return (when - last) >= COOLDOWN_PER_EVENT_SEC

    def mark_briefed(self, event_id: str,
                     *, now: Optional[float] = None) -> None:
        self._last_briefed_at[event_id] = (
            now if now is not None else time.time())

    # Grace window for events that JUST started (so a slightly-late
    # Sentinel tick still surfaces the "starting now" briefing).
    JUST_STARTED_GRACE_SEC = 60

    def upcoming_within(self, events: List[CalendarEvent],
                        *, lead_minutes: int = DEFAULT_LEAD_MINUTES,
                        now: Optional[float] = None
                        ) -> List[CalendarEvent]:
        """Return events whose start_ts is within
        [now - JUST_STARTED_GRACE_SEC, now + lead_minutes*60] —
        small back-window so a slightly-late tick still catches an
        event the moment it starts, instead of dropping it forever."""
        when = now if now is not None else time.time()
        horizon = when + lead_minutes * 60
        floor = when - self.JUST_STARTED_GRACE_SEC
        return [e for e in events if floor <= e.start_ts <= horizon]
