"""Layer 0 deterministic command router for the Local Agent.

Wraps Touchless's existing `VoiceCommandProcessor` so typed (and later
voice) Live API commands try a fast, 100%-reliable intent match first
and only fall back to the LLM when the router doesn't recognize them.

Routing flow:
  1. User says/types: "open chrome"
  2. CommandRouter.try_route("open chrome") → matches Chrome intent
     with high confidence → executes VoiceCommandProcessor.execute()
  3. Result returned with status="ok" + a short description
  4. LiveApiManager skips the LLM entirely for this turn

  vs.

  1. User: "create a folder called X with main.py inside"
  2. CommandRouter doesn't recognize this → returns RouterResult(matched=False)
  3. LiveApiManager forwards to the LLM agent loop as before

The router is intentionally CONSERVATIVE — it only claims a match when
VoiceCommandProcessor's confidence threshold (0.56) is comfortably
exceeded. False negatives (router didn't match → LLM handles) are
fine; false positives (router matched the wrong thing → executes the
wrong action) are not. So we set the bar a notch higher than the voice
pipeline's: 0.7 confidence minimum here.

Why parse + execute manually rather than just calling execute():
  * VoiceCommandProcessor.execute can prompt for follow-up selection
    (e.g. "I found 3 spotify songs, which one?") — that flow is
    incompatible with our chat panel and Live API state machine.
  * We want to log the routed action separately (router vs llm path)
    for analytics + debugging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .live_api_logger import LiveApiLogger


def _apply_phrase_rewrites(text: str) -> str:
    """Normalize ambiguous phrasings before sending to the parser.
    Returns the rewritten text, or the original unchanged when nothing
    matches."""
    lowered = text.strip().lower()
    for pattern, replacement in _PHRASE_REWRITES:
        if lowered == pattern:
            return replacement
    return text


_ROUTER_CONFIDENCE_FLOOR = 0.75


# Phrases the underlying VoiceCommandProcessor parses ambiguously. We
# normalize them BEFORE parse() so the model picks the right action.
#
# CRITICAL: VoiceCommandProcessor's Spotify parser only matches
# action=next/previous/pause/etc. when the word "spotify" appears in
# the phrase. Plain "next song" or "pause music" return NO MATCH.
# So our rewrites must produce a phrase that INCLUDES "spotify" —
# verified by `python -c "from hgr.voice.command_processor import
# VoiceCommandProcessor; ..."` against the actual parser.
#
# Without this, "play next song" → "next song" → no match → falls
# through to the LLM, which is the bug the user hit.
_PHRASE_REWRITES: tuple[tuple[str, str], ...] = (
    # next track — the parser key phrase is "next song spotify".
    ("play the next song", "next song spotify"),
    ("play next song", "next song spotify"),
    ("play next track", "next song spotify"),
    ("play next", "next song spotify"),
    ("skip to next song", "next song spotify"),
    ("skip this song", "next song spotify"),
    ("skip song", "next song spotify"),
    ("next song", "next song spotify"),
    ("next track", "next song spotify"),
    ("skip", "next song spotify"),
    # previous track — "previous song spotify".
    ("play the previous song", "previous song spotify"),
    ("play previous song", "previous song spotify"),
    ("play previous track", "previous song spotify"),
    ("play previous", "previous song spotify"),
    ("go back a song", "previous song spotify"),
    ("play the last song", "previous song spotify"),
    ("previous song", "previous song spotify"),
    ("previous track", "previous song spotify"),
    # pause / resume — "pause spotify" already works as-is so it's
    # left out; we only rewrite the VARIANTS that don't match.
    ("pause music", "pause spotify"),
    ("pause the music", "pause spotify"),
    ("pause song", "pause spotify"),
    ("pause", "pause spotify"),
    ("resume music", "resume spotify"),
    ("resume the music", "resume spotify"),
    ("unpause", "resume spotify"),
    ("unpause music", "resume spotify"),
    ("unpause spotify", "resume spotify"),
    # shuffle / repeat
    ("shuffle on", "shuffle spotify"),
    ("toggle shuffle", "shuffle spotify"),
    ("turn on shuffle", "shuffle spotify"),
    ("repeat on", "repeat spotify"),
    ("toggle repeat", "repeat spotify"),
    ("turn on repeat", "repeat spotify"),
)

# Phrases that strongly suggest the user is giving a multi-step or
# constructive command. The router should NEVER claim these — they
# always belong to the LLM agent. Even a high-confidence intent match
# on text containing one of these is almost certainly a false positive
# (e.g. "create a folder called X" parses as "open file_explorer").
_MULTI_STEP_MARKERS = (
    " then ", " and then ", " after that ", " after which ",
    " followed by ", "; ", "\n",
)

# Words that signal the user wants the agent to BUILD or WRITE
# something — not open an existing thing. The router can't do these.
_BUILD_WORDS = (
    "create ", "make ", "write ", "build ", "generate ",
    "script", "code", "function", "program",
    " a folder ", " a file ", " a window ", " a script ",
)

# Coding-agent commands ("have claude…", "tell codex…", "watch my claude tab
# and approve…") need the LLM (send_to_coding_agent / follow_up_coding_agent /
# auto_approve). The deterministic router can't drive an agent, and would
# mis-parse these (e.g. open a file named 'claude'). Any mention routes to LLM.
_CODING_AGENT_MARKERS = ("claude", "codex", "coding agent", "vs code", "vscode")

# Opening File Explorer to specific FOLDERS (esp. several) belongs to the LLM's
# open_path (which opens each as its own window) — the deterministic processor
# mishandles it (it opened touchless-control.com/downloads for "downloads").
_FILE_BROWSE_MARKERS = (
    "file explorer", " explorer to ", "explorer window",
)


# Phrases that signal CONVERSATIONAL text rather than a command. The
# voice-pipeline parser is greedy ("I don't see X" can parse as
# "open X"), so we explicitly skip routing when the user is asking a
# question or stating an observation. Anything matching these belongs
# to the LLM, never the deterministic router.
_CONVERSATIONAL_LEADS = (
    "what ", "what's ", "whats ",
    "where ", "where's ", "wheres ",
    "when ", "when's ",
    "why ", "why's ",
    "how ", "how's ",
    "who ", "who's ",
    "which ",
    "can you ", "could you ", "would you ", "will you ",
    "do you ", "did you ", "does it ",
    "is it ", "is there ", "are there ", "are you ",
    "should i ", "should we ", "should you ",
    "i don't ", "i dont ", "i can't ", "i cant ",
    "i didn't ", "i didnt ", "i won't ", "i wont ",
    "i need ", "i want ", "i would like ",
    "hmm", "umm", "uhh", "actually ",
)

# Phrases that signal a CONTEXTUAL or web-page action — operating on a
# page/tab/link that's already open, or referencing prior state. The
# deterministic router can only launch apps / fire fixed intents; it
# can't click page elements or reason about "the page you just opened".
# These belong to the LLM (which has the web_* tools + page context).
# Without this, "open the block 15 page from there" greedily parses as
# "open app: block 15" and launches the wrong thing.
_WEB_CONTEXT_MARKERS = (
    " page", " tab", " link", " links", " result", " results",
    " from there", " from that", " from the page", " from chrome",
    " you just", " just opened", " that page", " this page",
    " on the page", " on that", " second link", " first link",
    " third link", " go back",
)

# Leads that are page/screen actions the router can't perform — always
# hand to the LLM. (Substring "click" alone would catch app names like
# "clickup", so we anchor to the start / a word boundary instead.)
_WEB_ACTION_LEADS = ("click ", "go to ", "scroll ", "go back")

# WINDOW-MANAGEMENT commands (move/resize/arrange a window across monitors).
# The deterministic router only OPENS apps — it can't move/maximize windows,
# and the voice parser greedily turns "move my file explorer window..." into
# "open file" and searches the disk. These belong to the LLM
# (move_window_to_monitor / drag).
_WINDOW_MGMT_LEADS = ("move ", "drag ", "resize ", "maximize ", "minimize ", "snap ", "put ")
_WINDOW_MGMT_MARKERS = (
    " window", " monitor", " second screen", " other screen", " other monitor",
    " maximize", " minimize", " to the left screen", " to the right screen",
)

# Navigation verbs that, ANYWHERE in the command, mean "open a site and do
# something there" — the deterministic opener can't navigate + act, so these
# belong to the LLM web_* tools. ("open chrome, GO TO hacker news, ...")
_WEB_NAV_MARKERS = (" go to ", " navigate to ", " head to ", " pull up ")

# Email composition WITH a recipient/body belongs to the LLM + the Outlook/
# email connector (outlook_compose), which parses "email X saying Y"
# reliably and pre-fills the draft. The deterministic processor only
# understands "email TO x SUBJECT y BODY z" and otherwise opens a BLANK
# composer, so route content-bearing email requests to the LLM instead.
# ("open outlook" / "check my email" have no content cue and stay here.)
_EMAIL_COMPOSE_WORDS = (
    "email", "e-mail", "compose", "draft an email", "send an email", "mail to",
)
_EMAIL_COMPOSE_CUES = (
    "@", " saying ", " about ", " that says ", " telling ", " tell ",
    " subject ", " body ", " message ", " re ",
)

# READING/summarizing mail goes to the LLM (iris opens Outlook + read_screen) —
# the deterministic voice parser greedily turns any "...email..." into a blank
# COMPOSE, so "read my latest email" / "summarize my unread emails" must skip
# the router. (Detected by an email noun + a read/look verb.)
_EMAIL_NOUNS = ("email", "emails", "inbox", "mailbox")
_EMAIL_READ_VERBS = (
    "read", "summar", "unread", "latest", "recent", "check", "show",
    "what", "any new", "do i have", "go through", "look at",
)

# Asking for someone's email ADDRESS (a contact lookup) is NOT a compose
# intent. The router's deterministic parser doesn't know contacts; let it
# fall through to Tier 1 (memory recall) / Tier 2 (LLM plan).
_EMAIL_LOOKUP_MARKERS = (
    "find ", "what is ", "what's ", "whats ", "tell me ", "give me ",
    "look up ", "find out ", "do you know ", "can you find ",
)
_EMAIL_LOOKUP_OBJECT = ("email address", "email for", "'s email", "s email")

# Phrases that ask Iris to REPORT CONTENT back (read a page, summarize, list
# results). The deterministic router can only open/search — it can't read a
# page and tell you what's on it. Anything asking for a spoken/written answer
# must go to the LLM (which has web_get_text / web_get_links). Without this,
# "open chrome and tell me the top three stories" parses as a plain Google
# search and never summarizes anything.
_REPORT_BACK_MARKERS = (
    " tell me", " read me", " read it", " summarize", " summary",
    " what are", " what's on", " whats on", " what does it say",
    " top three", " top 3", " top five", " top 5", " top ten", " top 10",
    " headlines", " give me the", " list the", " show me the",
    " how many", " find out", " look up", " search for and",
)

# Punctuation / patterns that suggest natural language (questions,
# multi-clause sentences) rather than a single deterministic command.
_CONVERSATIONAL_MARKERS = ("?", "...")

# Anything longer than this is almost certainly multi-step or
# conversational and should go through the LLM.
_ROUTER_MAX_TEXT_LENGTH = 90


@dataclass
class RouterResult:
    matched: bool
    # Human-readable summary for the chat panel (e.g. "Opened Chrome").
    message: str = ""
    # success=True only when matched AND the controller call succeeded.
    success: bool = False
    # The intent details (action, app_name, query, confidence) for logging.
    intent_action: str = ""
    intent_app: str = ""
    intent_query: str = ""
    intent_confidence: float = 0.0


class CommandRouter:
    """Deterministic intent matching layer that runs before the LLM."""

    def __init__(self, *, logger: LiveApiLogger) -> None:
        self._logger = logger
        # Lazy-init — the underlying processor pulls in chrome/spotify/
        # desktop controllers (tens of MB), so we delay until first use
        # in case the user only ever types LLM-bound commands.
        self._processor = None
        self._init_failed = False

    def _ensure_processor(self):
        if self._processor is not None or self._init_failed:
            return self._processor
        try:
            from ..voice.command_processor import VoiceCommandProcessor
            self._processor = VoiceCommandProcessor()
            self._logger.event("router_processor_ready")
        except Exception as exc:
            self._init_failed = True
            self._logger.exception("router_processor_init_failed", exc)
        return self._processor

    def try_route(self, text: str) -> RouterResult:
        """Attempt to handle `text` deterministically. Returns a
        RouterResult — `matched=True` means the LLM should be skipped
        for this turn."""
        text = (text or "").strip()
        if not text:
            return RouterResult(matched=False)

        # Phrase normalization — translate Live API style ("play next
        # song") to Touchless voice style ("next song"). Done BEFORE
        # build-word filtering because rewrites strip out the "play"
        # verb that would otherwise route us into Spotify play search.
        text = _apply_phrase_rewrites(text)

        # Pre-filter: even if the parser would happily match, certain
        # patterns are guaranteed false positives for the deterministic
        # router. E.g. "create a folder called X" parses as
        # "open file_explorer" — wrong! Skip the router for these and
        # let the LLM handle them properly.
        lower = text.lower()

        # Length ceiling: deterministic commands are almost always
        # short ("open chrome", "next song"). Long text is multi-step
        # or conversational.
        if len(text) > _ROUTER_MAX_TEXT_LENGTH:
            self._logger.event("router_skip_too_long", text_len=len(text))
            return RouterResult(matched=False)

        # Conversational text — questions, statements, hedge words.
        # The voice parser is greedy and will happily turn "I don't
        # see X" into "open X" which is never what the user means.
        if any(marker in lower for marker in _CONVERSATIONAL_MARKERS):
            self._logger.event("router_skip_conversational_marker", text_len=len(text))
            return RouterResult(matched=False)
        if any(lower.startswith(lead) for lead in _CONVERSATIONAL_LEADS):
            self._logger.event("router_skip_conversational_lead", text_len=len(text))
            return RouterResult(matched=False)

        if any(marker in lower for marker in _MULTI_STEP_MARKERS):
            self._logger.event("router_skip_multi_step", text_len=len(text))
            return RouterResult(matched=False)
        if any(word in lower for word in _BUILD_WORDS):
            self._logger.event("router_skip_build_word", text_len=len(text))
            return RouterResult(matched=False)
        # Coding-agent commands always go to the LLM.
        if any(m in lower for m in _CODING_AGENT_MARKERS):
            self._logger.event("router_skip_coding_agent", text_len=len(text))
            return RouterResult(matched=False)
        # File Explorer / folder-opening → LLM open_path (opens each folder as
        # its own window; the local processor mishandles multi-folder opens).
        if any(m in lower for m in _FILE_BROWSE_MARKERS):
            self._logger.event("router_skip_file_browse", text_len=len(text))
            return RouterResult(matched=False)
        # Contextual / web-page actions belong to the LLM (it has the
        # web_* tools + the open page), not the deterministic app-opener.
        if lower.startswith(_WEB_ACTION_LEADS) or any(m in lower for m in _WEB_CONTEXT_MARKERS):
            self._logger.event("router_skip_web_context", text_len=len(text))
            return RouterResult(matched=False)
        # Navigate-and-act ("...go to hacker news...") or report-back
        # ("...tell me the top three...") commands need the LLM's web tools —
        # the router can only open/search, not read a page back to the user.
        if any(m in lower for m in _WEB_NAV_MARKERS) or any(m in lower for m in _REPORT_BACK_MARKERS):
            self._logger.event("router_skip_web_read", text_len=len(text))
            return RouterResult(matched=False)
        # Window-management (move/resize/arrange across monitors) — the router
        # can only open apps, so these go to the LLM (move_window_to_monitor).
        if lower.startswith(_WINDOW_MGMT_LEADS) or any(m in lower for m in _WINDOW_MGMT_MARKERS):
            self._logger.event("router_skip_window_mgmt", text_len=len(text))
            return RouterResult(matched=False)
        # Email-with-content → LLM + email connector (fills recipient/body).
        if any(w in lower for w in _EMAIL_COMPOSE_WORDS) and any(c in lower for c in _EMAIL_COMPOSE_CUES):
            self._logger.event("router_skip_email_compose", text_len=len(text))
            return RouterResult(matched=False)
        # Read/summarize mail → LLM (iris opens Outlook + read_screen). Without
        # this the parser mis-fires "read my latest email" as a blank compose.
        if any(n in lower for n in _EMAIL_NOUNS) and any(v in lower for v in _EMAIL_READ_VERBS):
            self._logger.event("router_skip_email_read", text_len=len(text))
            return RouterResult(matched=False)
        # Asking for someone's email ADDRESS (a contact lookup) is not a
        # compose — let it fall through so Tier 1's email-lookup pattern (or
        # memory recall) can answer. Without this skip, "find Dani's email
        # and give it to me" parses as a blank compose.
        if (any(m in lower for m in _EMAIL_LOOKUP_MARKERS)
                and any(o in lower for o in _EMAIL_LOOKUP_OBJECT)):
            self._logger.event("router_skip_email_lookup", text_len=len(text))
            return RouterResult(matched=False)

        processor = self._ensure_processor()
        if processor is None:
            return RouterResult(matched=False)

        try:
            intent = processor.parse(text)
        except Exception as exc:
            self._logger.exception("router_parse_failed", exc, text_len=len(text))
            return RouterResult(matched=False)

        if intent is None:
            self._logger.event("router_no_match", text_len=len(text))
            return RouterResult(matched=False)

        # We're stricter than the voice pipeline — false positives feel
        # very broken from a chat UI ("I typed X and it did Y").
        confidence = float(getattr(intent, "confidence", 0.0) or 0.0)
        if confidence < _ROUTER_CONFIDENCE_FLOOR:
            self._logger.event(
                "router_below_threshold",
                action=getattr(intent, "action", ""),
                confidence=confidence,
                floor=_ROUTER_CONFIDENCE_FLOOR,
            )
            return RouterResult(matched=False)

        self._logger.event(
            "router_matched",
            action=getattr(intent, "action", ""),
            app=getattr(intent, "app_name", ""),
            query=getattr(intent, "query", "") or "",
            confidence=confidence,
        )

        # Defer to the processor's execute() so all the existing app-
        # opening / search / playback logic runs unchanged. We pass the
        # original text so the processor parses it the same way it
        # would for voice.
        try:
            result = processor.execute(text)
        except Exception as exc:
            self._logger.exception(
                "router_execute_failed", exc, action=getattr(intent, "action", "")
            )
            return RouterResult(
                matched=True,
                success=False,
                message=f"Router matched but execution failed: {exc}",
                intent_action=getattr(intent, "action", ""),
                intent_app=getattr(intent, "app_name", ""),
                intent_query=getattr(intent, "query", "") or "",
                intent_confidence=confidence,
            )

        success = bool(getattr(result, "success", False))
        # The voice pipeline's info_text / control_text are written for
        # a status overlay where context is implicit — when shown in a
        # chat panel they read like internal log lines ("chrome window
        # not found", "-", etc.). Build a concise human message based
        # on the intent action and success flag, falling back to the
        # raw fields only when they look user-friendly.
        message = _humanize_result_message(intent, result, success)
        self._logger.event(
            "router_executed",
            success=success,
            target=getattr(result, "target", ""),
            info=message,
        )

        return RouterResult(
            matched=True,
            success=success,
            message=str(message),
            intent_action=getattr(intent, "action", ""),
            intent_app=getattr(intent, "app_name", ""),
            intent_query=getattr(intent, "query", "") or "",
            intent_confidence=confidence,
        )


def _humanize_result_message(intent, result, success: bool) -> str:
    """Build a one-line, chat-friendly message describing what the router did.

    The voice pipeline's `info_text` / `control_text` are tuned for an
    overlay where context is implicit ("chrome window not found", "-").
    In a chat panel they look like internal log lines. We synthesize a
    cleaner message from the intent + success flag, with the raw
    pipeline message as a fallback only when it's already readable.
    """
    action = (getattr(intent, "action", "") or "").lower()
    app = (getattr(intent, "app_name", "") or "").replace("_", " ")
    query = (getattr(intent, "query", "") or "").strip()
    info = (getattr(result, "info_text", "") or "").strip()

    # Treat short/dash messages as "no useful info" — fall back to a
    # synthesized one.
    raw_useful = info and info not in {"-", "executed"} and len(info) > 3

    if success:
        if action == "open" and app:
            return f"Opened {app.title()}"
        if action == "search" and app:
            q = f" for {query}" if query else ""
            return f"Searched in {app.title()}{q}"
        if action == "play":
            # Spotify play returns a useful "Song: ... | Artist: ..." string.
            if raw_useful and "|" in info:
                # Keep but trim to the first two segments for readability.
                parts = [p.strip() for p in info.split("|")[:2]]
                return " | ".join(parts)
            return f"Started playback{f' ({query})' if query else ''}"
        if action == "next":
            return "Skipped to next track"
        if action == "previous":
            return "Skipped to previous track"
        if action == "pause":
            return "Paused"
        if action == "resume":
            return "Resumed"
        if action == "shuffle":
            return "Toggled shuffle"
        if action == "repeat":
            return "Toggled repeat"
        if raw_useful:
            return info
        return "Done"
    else:
        if action == "open" and app:
            return f"Couldn't open {app.title()}{f' ({info})' if raw_useful else ''}"
        if raw_useful:
            return f"Couldn't run that command — {info}"
        return "Couldn't run that command"

# Author: Konstantin Markov
