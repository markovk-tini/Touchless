"""Top-level orchestrator for the Live API prototype.

`LiveApiManager` is what the main window owns. It exposes Qt signals
for UI updates (state changes, status text, errors) and coordinates:

  * AudioStream     -> microphone capture
  * ScreenContext   -> screenshot capture
  * RealtimeClient  -> websocket to OpenAI Realtime API
  * ToolExecutor    -> dispatches tool calls into existing Touchless

All long-running work happens off the UI thread:
  * audio capture runs on the sounddevice thread
  * the websocket runs on its own daemon reader thread
  * a periodic screen-capture timer runs on a worker thread
  * tool execution runs on the websocket reader thread by default

Signals are emitted via QObject.signal so consumers in the UI
thread receive them via the normal Qt queued-connection mechanism.
"""
from __future__ import annotations

import enum
import getpass
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import (
    QObject, Signal, QMetaObject, Qt, Q_ARG, Slot, QThread,
)

from . import cortex_emit
from .audio_stream import AudioStream
from .config import LiveApiConfig, load_config
from .live_api_logger import LiveApiLogger
from .realtime_client import RealtimeClient
from .schemas import all_tool_schemas
from .screen_context import ScreenContext
from .tool_executor import ToolExecutor
from .tool_invocation import (
    InvocationSource, ToolInvocation, publish as _publish_invocation,
)
from .tool_registry import ToolRegistry
from .user_prefs import get_default_email
from . import cost_policy


SYSTEM_INSTRUCTIONS = (
    "You are Iris, the user's personal assistant — think Jarvis from Iron Man. "
    "Warm, witty, a touch playful, completely conversational. You sound like a "
    "smart friend sitting next to the user, not a help desk and not a "
    "narrator reading a teleprompter. You can absolutely hold a back-and-forth "
    "chat — banter, ask a clarifying question, riff on what the user said, "
    "follow up naturally on something from earlier in the conversation. "
    "You also control the user's Windows PC via tools. Use a tool for every "
    "action; never say something is done unless the tool result confirms it, "
    "and never claim a window/app is in some state without actually doing it. "
    "Screenshots show the FULL multi-monitor desktop — read the image; your "
    "own 'Touchless' window is not the target, ignore it. If unsure of screen "
    "state, get_screen_context first.\n"
    "HARD RULE — NEVER FABRICATE TOOL DATA: every name, count, date, "
    "subject, path, or quoted detail you report from a tool MUST come "
    "literally from that tool's response. If empty/missing, say so "
    "plainly ('no unread emails', 'nothing matched'); never invent "
    "plausible examples. Tone shapes HOW you phrase real facts, never "
    "license to invent them.\n"
    "HARD RULE — NEVER FABRICATE CONTACT / RECORD FIELDS: when a tool "
    "returns a person or record with a field that is empty, null, or "
    "missing (contact with emails=[], phones=[]; event with no location; "
    "task with no due date), report that field as absent in plain words "
    "— 'no email on file for John', 'I don't have a phone number for "
    "her', 'the event has no location set'. NEVER generate a plausible-"
    "looking substitute from context: no jsmith@company.com, no "
    "john@gmail.com, no 555-XXXX, no guessed office address, no "
    "'(email withheld)' or any synthetic marker. Concrete example: if "
    "contacts_lookup returns {name:'John Smith', emails:[], phones:[]}, "
    "say 'John Smith — no email or phone on file', not 'John Smith — "
    "john.smith@company.com' and not 'John Smith — jsmith@gmail.com'. "
    "If the user asks 'what's John's email' and the tool result shows "
    "emails=[] for John, the correct answer is 'I don't have an email "
    "for John on file' — full stop. This applies to every field on "
    "every record, not just contacts.\n"
    "HARD RULE — NEVER FABRICATE PERSONAL / RECALL CONTEXT: for recall, "
    "recap, or 'remind me' questions (including artifact retrieval like "
    "'show me the email we drafted', 'pull up that poem'), every "
    "concrete detail — topic, room, person, date, draft body — MUST "
    "come from your injected memory / recall-context note / session "
    "summary / visible RECENT CONVERSATION. If a 'Recall context for "
    "this turn' note says NONE or weak, treat memory as empty and ask "
    "the user to remind you ('I don't have specifics on that handy — "
    "want to walk me through it?'). Plausibility is NOT evidence; "
    "pattern-matching typical scenarios is fabrication. Do NOT invent "
    "a procedural backstory ('we never finished that', 'you said you'd "
    "send specifics later') to cover a gap.\n"
    "HARD RULE — CLARIFY BEFORE ACTING ON AMBIGUOUS NAMES: when the user "
    "says 'delete X' / 'open X' / 'send X' / 'change X' and there are two "
    "or more session artifacts with the same or overlapping name across "
    "DIFFERENT kinds (e.g. a sheet named 'notes' AND a task named 'notes'), "
    "do NOT silently pick one. Ask which they meant, naming both kinds: "
    "'I see a sheet and a task both called notes — which one?' Only after "
    "the user confirms do you dispatch the tool. If the user's own next "
    "turn already disambiguates ('the task' / 'the sheet'), you may act "
    "without another prompt.\n"
    "MEDIA TOOL SELECTION RULE (read BEFORE picking a tool):\n"
    "  • 'play <song/artist/playlist/album/anything-named>' → "
    "    ALWAYS spotify_play(query='<the named thing>'). NEVER "
    "    media_play_pause for these — that's just a toggle key, "
    "    it won't search, it won't change track. The user said a "
    "    name; they want THAT thing playing, not a blind toggle.\n"
    "  • Bare 'play' / 'resume' (no object) → media_play_pause OR "
    "    spotify_play() with no args.\n"
    "  • Bare 'pause' / 'stop' → media_play_pause or spotify_pause.\n"
    "  • 'skip' / 'next' → media_next_track or spotify_next.\n"
    "  • 'previous' / 'back' → media_previous_track or "
    "    spotify_previous.\n"
    "  • 'what's playing' → spotify_now_playing or "
    "    media_now_playing.\n"
    "MEDIA PLAYBACK RULE — REQUIRED after any play/queue command "
    "(spotify_play, run_quick_command with a media phrase, etc.):\n"
    "  1. ALWAYS write a short natural-language confirmation in the "
    "user's reply channel. Never dump raw tool call args or JSON. "
    "Never go silent.\n"
    "  2. Do NOT name a specific song or artist UNLESS the tool "
    "response from THIS call contains fresh now_playing_title / "
    "now_playing_artist fields. The previous song from earlier in "
    "the conversation is STALE — quoting it after a new play is "
    "fabrication.\n"
    "  3. Safe replies: echo the user's request ('Playing your feel-"
    "good rock playlist now.' / 'Queuing Poker Face.' / 'Pulling that "
    "up.'), or confirm without naming ('On it.' / 'Done, sir.' / "
    "'Got it.'). Match the active voice preset.\n"
    "  4. Never claim a specific track is playing from memory.\n"
    "  5. spotify_play / play_search_request AUTO-PLAYS the result. "
    "NEVER tell the user to 'hit play whenever you're ready' or "
    "'press play' or 'start it when you want' — the music is "
    "ALREADY playing once the tool returns started=true. Saying "
    "otherwise contradicts what's actually happening on their "
    "speakers.\n"
    "  6. If the tool returns started=false or error, say so plainly "
    "('Couldn't queue that — Spotify wasn't reachable') instead of "
    "pretending it worked.\n"
    "CALENDAR: when the turn is a calendar create/list/modify, a "
    "'Calendar context for this turn' session note (with the active "
    "tool priority + override grammar) will be injected — follow it. "
    "Always mention WHICH calendar the event landed in using the "
    "tool result's `calendar` field.\n"
    "LANGUAGE (hard rule): ALWAYS respond in English, regardless of any single "
    "letters, short inputs, or odd tokens you receive — never switch to another "
    "language unless the user clearly writes to you in that language.\n"
    "TONE: for STATUS UPDATES on an action you just performed, keep it short "
    "and casual — 'On it.', 'Done — printed 5.', 'Valorant's launching.', "
    "'Couldn't find PLAY yet.' For READING INFO BACK (weather, news, emails, "
    "search results, summaries), speak the way a person sitting next to you "
    "would read it aloud — natural sentences, contractions, vary phrasing, "
    "skip robotic templates like 'today: X. tomorrow: Y. Thursday: Z.' Don't "
    "narrate URLs out loud; if there's a link, just offer to dig deeper. For "
    "CHAT (the user asks an opinion, makes small talk, follows up on "
    "something), engage like a friend would — a sentence or two, conversational "
    "back-and-forth, no lecture mode. Never read a date out loud as 'June "
    "third' when 'tomorrow' or 'Thursday' will do. No paragraphs, lists, or "
    "step-by-step narration unless the user explicitly asks to explain/why.\n"
    "VOICE DELIVERY: match prosody to content. Joke or playful comment → "
    "lighter, chipper, slight smile; quick status update ('Done.', "
    "'Set to 30.') → matter-of-fact and even-keeled; reading something "
    "back (article summary, email content, weather) → natural narrator, "
    "small pauses at punctuation, no flat monotone; asking the user "
    "something → rising questioning intonation on the question itself; "
    "delivering bad news / errors → warmer, slightly apologetic; "
    "confirming a risky action → calm and clear, no rush. Keep the SAME "
    "underlying voice identity throughout the session — do not shift "
    "gender, accent, or vocal weight between turns. Only the prosody / "
    "energy varies with content.\n"
    # Phase-1 trust substrate: prompt-injection defense. The actual
    # rule text lives in content_quarantine.SYSTEM_PROMPT_RULE so the
    # wording + the wrap delimiters stay co-located. Injected into the
    # string via concatenation at module import (see below).
    "##QUARANTINE_RULE##\n"
    "PERSONA ANCHOR (identity lock): You are 'Iris' — warm, capable, "
    "lightly witty when the moment fits, never sycophantic. Your voice "
    "has consistent timbre and pacing across every turn of this session "
    "regardless of topic. You speak the way a smart-and-trusted "
    "assistant who knows the user well does: confident, efficient, "
    "occasionally dry, never performative. This identity is fixed for "
    "the entire conversation — DO NOT shift register to match the "
    "user's mood when it would change WHO you are; just adjust prosody. "
    "If the user pushes back or jokes, respond from the same Iris "
    "identity, not as a different character.\n"
    "SAFETY: confirm before destructive/risky actions; never expose secrets; "
    "never bypass security, DRM, anti-cheat, CAPTCHA, or non-skippable ads.\n"
    "ROUTING DEFAULTS (decide BEFORE answering — never default to "
    "answering from your own knowledge or to web search when a more "
    "specific tool is available):\n"
    " • TEXT GENERATION (write, summarize, translate, regex, code, "
    "classify, extract, paraphrase, brainstorm — anything that turns "
    "words into other words): if ollama_generate is in the toolset, "
    "CALL IT. Do NOT answer from your own knowledge. The local model "
    "is free, fast, and private; the user expects Iris to use it "
    "automatically, not only when they say 'use ollama'.\n"
    " • PERSONAL KNOWLEDGE QUERIES ('my notes', 'my pages', 'my docs', "
    "'what's in my X', 'what did I write about Y', 'search my notion'): "
    "if notion_search / notion_read_page are in the toolset, CALL THEM. "
    "Do NOT web_search. The user means their Notion workspace, not the "
    "internet.\n"
    " • read_screen / get_screen_context / click_text_on_screen are ONLY "
    "for what's CURRENTLY VISIBLE on the user's monitor (e.g. 'what does "
    "this error say', 'click the Send button I see'). NEVER for 'what's "
    "in my X' if X could be a page, file, or document. The word 'use' "
    "in 'use the local model' / 'use Ollama' / 'use Notion' refers to "
    "the TOOL FAMILY, not anything visible on screen.\n"
    " • If the user explicitly names a tool family ('use ollama', 'use "
    "the local model', 'search my notion', 'check my drive'), CALL THAT "
    "TOOL — never substitute another tool because you think you can "
    "answer directly.\n"
    "ACTION HIERARCHY — for ANY action (launch, click, trigger) use the FIRST "
    "that fits (earlier tiers beat pixel-clicking):\n"
    " 1) BYPASS the UI: a direct launch/exe, a CLI/launch arg, a URL/deep link "
    "(open_url), run_quick_command, or a dedicated tool.\n"
    " 2) APP CONTROLS: native window → read_ui/click_ui/set_field (UIA — exact, "
    "by name, no coordinates); a real web page → the web_* tools (DOM-precise).\n"
    " 3) SCREEN-READ (fallback only): click_text_on_screen (OCRs a VISIBLE text "
    "label and clicks its exact pixels — works on Chromium/webview UIs UIA "
    "can't see; double=true for double-click, retries=N to re-click if it "
    "missed); zoom_screen+click_zoom for icons/regions; click_screen "
    "(normalized 0-1 fractions) as the last resort.\n"
    "DRAWING (Paint etc.): to draw a SHAPE use draw_shape(shape, x1,y1,x2,y2) "
    "with a normalized bounding box — it handles square/rectangle/circle/oval/"
    "triangle/diamond/line in ONE call. Do NOT screenshot, zoom, or hunt for the "
    "app's shape-tool buttons (that wastes tokens and rate-limits you). For "
    "'to the right of'/'below' a previous shape, offset the box. Use draw_path "
    "only for freeform/custom strokes."
)
# Replace the ##QUARANTINE_RULE## marker with the canonical text from
# content_quarantine.SYSTEM_PROMPT_RULE. Done post-construction so
# the wording + the wrap delimiters in content_quarantine.py stay
# the single source of truth — edit the rule there and it propagates
# here automatically on next module load.
try:
    from .content_quarantine import SYSTEM_PROMPT_RULE as _QUARANTINE_RULE
    SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS.replace(
        "##QUARANTINE_RULE##", _QUARANTINE_RULE + "\n")
except Exception:
    SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS.replace(
        "##QUARANTINE_RULE##", "")


# The capability-search meta-tool. Always exposed (it's cheap — one
# schema); connector tools are loaded on demand when the model calls it,
# so the initial tool list stays lean no matter how many connectors exist.
FIND_CAPABILITY_TOOL: Dict[str, Any] = {
    "type": "function",
    "name": "find_capability",
    "description": (
        "Discover and load a fast API connector for a task BEFORE resorting "
        "to clicking/typing on the screen, OR before telling the user you "
        "can't do something. Many apps (Spotify, email / Gmail / Outlook, "
        "Google Calendar, Google Docs / Sheets / Slides, Google Drive "
        "(upload/save files), driving/walking directions, system volume, "
        "Discord voice, YouTube, Microsoft Office, the "
        "browser) have one-call API tools that are far faster and more "
        "reliable than GUI control. ALWAYS call this first when the user asks "
        "to act on such a service (e.g. 'play music', 'send an email', 'set "
        "the volume', 'add a calendar event', 'upload this to Drive') — do "
        "not assume you lack a capability until find_capability returns "
        "nothing. If a connector matches, its tools load into this session "
        "and you should then call the tool name it reports. If nothing "
        "matches, control the app via the screen as usual."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string",
                     "description": "What you want to do, in a few words."},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


def build_system_instructions() -> str:
    """SYSTEM_INSTRUCTIONS plus this machine's real paths + multi-step
    guidance, so the model never guesses the username (it hallucinated
    'C:\\Users\\Thomas\\...' in testing) and finishes a whole multi-step
    request in one turn instead of stopping after the first tool."""
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    home = Path.home()
    docs = home / "Documents"
    desktop = home / "Desktop"
    email = get_default_email()
    # Only advertise gmail_send when Gmail is actually connected — otherwise the
    # model calls a tool that doesn't exist, errors, and retries in a loop.
    gmail_ready = False
    try:
        from .connectors.google_client import GoogleClient
        gmail_ready = GoogleClient.shared().ready()
    except Exception:
        gmail_ready = False
    email_identity = (
        f"USER EMAIL: {email} — the user's own address; use it as the default "
        f"sender and as the recipient when they say 'email me/myself'. "
        if email else ""
    )
    if gmail_ready:
        email_actions = (
            "EMAIL ACTIONS: to SEND an email use gmail_send (Gmail API, reliable, "
            "confirms first). For a draft the user sends themselves, use "
            "outlook_compose. Always include a concise subject. "
            "NEVER pick outlook_compose / email_send / gmail_send when the "
            "utterance contains an A1-style spreadsheet cell reference "
            "([A-Z]+[0-9]+, e.g. C1, B12, AA3) or a spreadsheet/doc/slide "
            "context word (sheet, spreadsheet, doc, document, slide, cell, "
            "range, row, column) — those go to sheets_update_range / "
            "gdocs_append / slides_*. Treat the word 'email' as CONTENT (not "
            "an action) whenever it appears after 'to say/write/enter/type/"
            "put/set to/change to', inside quotes, or as the final token of a "
            "sheets/docs edit request. "
        )
    else:
        email_actions = (
            "EMAIL ACTIONS: to SEND an email use email_send — it opens the "
            "pre-filled draft and presses the mail send shortcut for the user "
            "(it confirms first). For a draft only (no send), use "
            "outlook_compose. Always include a concise subject. Call the send "
            "ONCE — if it reports it couldn't send, tell the user the draft is "
            "open to send manually; never retry in a loop. "
            "NEVER pick outlook_compose / email_send when the utterance "
            "contains an A1-style spreadsheet cell reference ([A-Z]+[0-9]+, "
            "e.g. C1, B12, AA3) or a spreadsheet/doc/slide context word "
            "(sheet, spreadsheet, doc, document, slide, cell, range, row, "
            "column) — those go to sheets_update_range / gdocs_append / "
            "slides_*. Treat the word 'email' as CONTENT (not an action) "
            "whenever it appears after 'to say/write/enter/type/put/set to/"
            "change to', inside quotes, or as the final token of a sheets/"
            "docs edit request. "
        )
    email_clause = email_identity + email_actions
    # Phase-3 polish: resolve the user-tunable persona block and
    # tack it onto the system prompt's tail so the user's prefs
    # (TOUCHLESS_PERSONA, memory-stored 'persona' fact, etc.)
    # override the baked-in tone.
    try:
        from .persona import get_persona_block
        persona = get_persona_block()
    except Exception:
        persona = ""
    # PERSONA PROMOTION (Phase-7 polish): the persona block goes
    # FIRST — before the long tool-list / safety rules. The model
    # anchors on opening tokens; burying tone in the middle of a
    # 20k-char prompt is why earlier "Jarvis" still sounded like
    # generic-helpful-assistant. Wrap in hard-rule language.
    persona_header = (
        f"### ACTIVE VOICE (REQUIRED — overrides default tone) ###\n"
        f"{persona}\n"
        f"### END ACTIVE VOICE ###\n\n"
        if persona else "")
    # Inject CURRENT TIME so the model never hallucinates the clock.
    # Without this, "what time is it" gets a confident-but-wrong
    # answer + the model triggers screen_read to "verify" — wasteful.
    import datetime as _dt
    try:
        _now = _dt.datetime.now().astimezone()
        time_clause = (
            f"CURRENT TIME (use this when asked; never guess, never "
            f"call get_screen_context to check the clock): "
            f"{_now.strftime('%A, %B %d, %Y at %I:%M %p %Z').strip()} "
            f"(ISO: {_now.isoformat(timespec='seconds')}).\n\n"
        )
    except Exception:
        time_clause = ""
    return (
        persona_header
        + time_clause
        + SYSTEM_INSTRUCTIONS
        + "\n\nMACHINE PATHS (use these EXACT paths; never guess the username): "
        f"Username {user}; Home {home}; Documents {docs}; Desktop {desktop}. "
        "Pass one (or a subfolder you made) as `base_dir` for file tools.\n"
        "CONNECTORS FIRST: when a dedicated API tool exists for the task "
        "(email, calendar, spotify, volume, discord, youtube, drive, "
        "files, web), prefer it over GUI clicking. Each tool's schema "
        "description explains its own usage — read it before calling.\n"
        "EMAIL READS — email_summary FIRST: for any 'read/summarize/check "
        "my email/inbox/unread' request, call email_summary "
        "(unread_only=true, max=50). It cascades Gmail → Microsoft → "
        "fallback so it finds the user's real inbox. EMIT "
        "`result.summary` VERBATIM — do not rephrase, re-summarize, or "
        "invent senders/subjects (you have hallucinated demo emails "
        "before). Only if `result.summary` is empty, fall back to "
        "summarizing from `result.messages` quoting fields verbatim. "
        "Use gmail_list / ms_mail_list ONLY for account-specific "
        "searches with a query.\n"
        + email_clause +
        "TYPING INTO AN APP: ALWAYS pass window_title to type_text / "
        "press_hotkey — the Iris chat is always-on-top, so untargeted "
        "keystrokes type into the wrong window.\n"
        "UAC / Windows security prompts are on the secure desktop — you "
        "CANNOT click them. Warn once and keep waiting.\n"
        "MULTI-STEP: a multi-action request is usually auto-split into "
        "'[Step N of M] do ONLY this…' sub-tasks — finish each ONE "
        "fully, give a one-line confirmation, don't do other steps in "
        "it. If a step slips through WITHOUT '[Step N of M]' wrapping "
        "(e.g. 'write me a haiku and add a task to drink some'), do ALL "
        "the actions yourself in sequence — call each tool in turn, "
        "don't stop after the first verb."
    )


class LiveApiState(enum.Enum):
    OFF = "off"
    CONNECTING = "connecting"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    ERROR = "error"


# Cortex viz: (visual-state-name, intensity 0..1) for each LiveApiState.
# Used inside _set_state to push core state to the Cortex window via
# cortex_emit. Mapping is best-effort and intentionally lossy (the
# viz has fewer + coarser visual states than the operational machine).
_CORTEX_STATE_MAP = {
    LiveApiState.OFF: ("idle", 0.3),
    LiveApiState.CONNECTING: ("idle", 0.45),
    LiveApiState.LISTENING: ("listening", 0.6),
    LiveApiState.THINKING: ("thinking", 0.8),
    LiveApiState.EXECUTING: ("thinking", 0.9),
    LiveApiState.ERROR: ("error", 0.7),
}


# A confirmation callback the manager will invoke for risky tools.
# The UI is expected to hand one over via `set_confirm_callback`.
ConfirmCallback = Callable[[str, str], bool]


_SERVICE_DISABLED_URL_RE = re.compile(
    r"https?://console\.(?:developers|cloud)\.google\.com/apis/api/"
    r"([a-z0-9_.-]+)\.googleapis\.com/overview\?project=(\d+)"
)
_API_LABEL = {
    "tasks": "Google Tasks",
    "forms": "Google Forms",
    "youtube": "YouTube Data",
    "sheets": "Google Sheets",
    "docs": "Google Docs",
    "slides": "Google Slides",
    "drive": "Google Drive",
    "gmail": "Gmail",
    "calendar": "Google Calendar",
    "photoslibrary": "Google Photos",
    "people": "Google People (Contacts)",
}


def _verb_for_tool(name: str) -> str:
    """Best-effort verb phrase for a tool's failure line."""
    n = (name or "").lower()
    if n.endswith("_upload"):
        return "upload that"
    if n.endswith("_send"):
        return "send that"
    if n.endswith("_create"):
        return "create that"
    if n.endswith("_delete"):
        return "delete that"
    if n.endswith("_update"):
        return "update that"
    if n.endswith("_list") or n.endswith("_search") or n.endswith("_read"):
        return "look that up"
    if n.endswith("_get"):
        return "fetch that"
    if n.endswith("_open") or n.endswith("_play"):
        return "open that"
    if n.endswith("_move"):
        return "move that"
    return f"run {name}"


def _format_connector_error(name: str, code: str, err_msg: str) -> str:
    """Map a connector error result to a short, human-friendly spoken line.

    Used by the universal error short-circuit in _dispatch_function_call
    so error responses never get handed to the realtime LLM (which can
    loop / stall while it "interprets" the failure).
    """
    c = (code or "").lower().strip()
    m = (err_msg or "").strip()
    n = (name or "").strip()
    tool_label = n.replace("_", " ") if n else "that"

    if c in ("not_ready", "not_connected", "not_authorized"):
        if n.startswith("photos_"):
            return ("I do not have access to your Google Photos yet — "
                    "want me to walk you through connecting it?")
        if n.startswith("gmail_") or n.startswith("gdocs_") or n.startswith("drive_") \
                or n.startswith("sheets_") or n.startswith("slides_") \
                or n.startswith("calendar_") or n.startswith("contacts_") \
                or n.startswith("forms_"):
            return ("I do not have access to your Google account for that yet "
                    "— want me to walk you through connecting it?")
        if n.startswith("ms_") or n.startswith("outlook_") or n.startswith("teams_"):
            return ("I do not have access to your Microsoft 365 account for "
                    "that yet — want me to walk you through connecting it?")
        if n.startswith("spotify_"):
            return ("Spotify is not connected yet — want me to walk you "
                    "through connecting it?")
        return (f"{tool_label} is not connected yet — want me to walk you "
                "through setting it up?")
    if c == "timeout":
        return (f"{tool_label} took too long to respond. Try again in a "
                "moment.")
    if c == "user_declined" or c == "user_declined_speed_bump":
        return "Okay, cancelled."
    if c == "not_found":
        return f"Could not find that file. {m}" if m else "Could not find that file."
    if "SERVICE_DISABLED" in m or "has not been used in project" in m:
        url_match = _SERVICE_DISABLED_URL_RE.search(m)
        if url_match:
            svc_key = url_match.group(1).lower()
            api_label = _API_LABEL.get(svc_key, f"{svc_key.title()} API")
            enable_url = url_match.group(0)
            return (
                f"The {api_label} API isn't enabled for this project. "
                f"Enable it at:\n{enable_url}\n"
                "— then retry in 1-2 minutes."
            )
        return (
            "That Google API isn't enabled for this project. Open the "
            "Cloud Console for this OAuth client, enable the API, and "
            "retry in 1-2 minutes."
        )
    if "403" in m or "permission" in m.lower() or "forbidden" in m.lower():
        if n.startswith("photos_"):
            return ("Google rejected that — looks like the Photos scope is "
                    "not granted. Reconnect Google to fix it.")
        return ("Google rejected that — looks like a required permission "
                "is not granted. Reconnecting the account should fix it.")
    if "401" in m or "unauthorized" in m.lower():
        return ("Auth expired for that account — reconnect it and I will "
                "try again.")
    if "429" in m or "rate" in m.lower():
        return "Hit a rate limit on that. Try again in a minute."
    if m:
        return f"Could not {_verb_for_tool(n)}: {m}"
    return f"Could not {_verb_for_tool(n)}."


class LiveApiManager(QObject):
    state_changed = Signal(object, str)        # (LiveApiState, status text)
    error_occurred = Signal(str)
    transcript_received = Signal(str)          # user speech transcript text
    assistant_text = Signal(str)               # assistant text deltas/snippets
    assistant_message_break = Signal()         # start a fresh assistant bubble
    tool_event = Signal(str, dict)             # ("called"/"completed", info)
    # Action chips for the chat UI to render below the most recent
    # assistant bubble. Payload is a list of action ids like
    # ["connect_gmail", "connect_ms", "read_outlook_screen"]. The UI
    # decides label + handler per id.
    suggested_actions = Signal(list)

    def __init__(
        self,
        *,
        config: Optional[LiveApiConfig] = None,
        external_action_router: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        text_only: bool = False,
        voice_output: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        # Publish ourselves at module level so cross-cutting
        # substrate (e.g., earcon_wiring) can check "is Iris
        # currently speaking?" without importing this whole class.
        # Most-recent instance wins; in practice the UI constructs
        # exactly one manager per session.
        import sys as _sys
        _sys.modules[__name__]._last_constructed = self  # type: ignore[attr-defined]
        self._config = config or load_config()
        # text_only=True means we don't open the mic / VAD pipeline; the
        # only inputs come from explicit `send_user_text` calls. Used by
        # the typed-command UI. voice_output controls whether the model's
        # spoken reply is played back through speakers — independent of
        # text_only so the chat panel can have typed input + voice reply
        # (text_only=True, voice_output=True).
        self._text_only = bool(text_only)
        self._voice_output = bool(voice_output)
        # Lazily-created PCM player; only built when voice_output is True.
        self._audio_player: Optional[Any] = None
        # Phase-1 substrate: append-only audit log. Subscribes to the
        # global InvocationBus so every planner + realtime tool call is
        # recorded. Built lazily on first use to avoid filesystem touches
        # at import-time. A SINGLE process-wide instance is sufficient —
        # multiple LiveApiManagers would race on the same DB, so we
        # share one. See `_ensure_audit_log` below.
        self._audit_log: Optional[Any] = None
        self._logger: Optional[LiveApiLogger] = None
        self._client: Optional[RealtimeClient] = None
        self._audio: Optional[AudioStream] = None
        self._screen: Optional[ScreenContext] = None
        self._executor: Optional[ToolExecutor] = None
        self._registry: Optional[ToolRegistry] = None
        # Iris planner (Phase 1: deterministic classifier → connector, no model).
        # Lazily built when the registry exists; reset at session start.
        self._iris_planner: Optional[Any] = None
        # Connector tool schemas loaded on demand by find_capability this
        # session (deduped by tool name). Added on top of the built-in tools.
        self._loaded_connector_schemas: List[Dict[str, Any]] = []
        self._screen_thread: Optional[threading.Thread] = None
        self._screen_stop = threading.Event()
        self._screen_request = threading.Event()
        self._state = LiveApiState.OFF
        self._lock = threading.RLock()
        self._confirm_callback: Optional[ConfirmCallback] = None
        self._external_action_router = external_action_router
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Set to True by stop() to make late WS-thread callbacks no-op.
        self._teardown_in_progress = False
        # Set to False by stop() so late WS-thread callbacks that would
        # construct Qt objects (QTimer.singleShot, edge pulses) against a
        # dying event dispatcher become no-ops instead.
        self._gui_alive = True
        # Names of every watcher this manager registered with the
        # global Sentinel via _register_phase_3_watchers /
        # _wire_proactive_nudges / _ensure_registry_wiring. stop()
        # iterates this list and unregisters each so the next reopen
        # gets a clean slate and idle sentinel ticks can't fire
        # callbacks against a torn-down manager. The Sentinel daemon
        # itself is process-wide and intentionally left running.
        self._registered_watcher_names: list[str] = []
        # True while the server has a response in progress. Background notes
        # (auto-approve conclude, etc.) must NOT call response.create while
        # one is active — the server rejects it with
        # conversation_already_has_active_response. Notes that arrive mid-
        # response are queued here and flushed on response.done.
        self._response_active = False
        self._pending_notes: list[str] = []
        # A model turn was requested while a response was active; fired once
        # on response.done so parallel tool calls don't each create one.
        self._response_requested = False
        # ---- response-cycle watchdog ----
        # Armed on response.created; disarmed on response.done /
        # response.cancelled / error / ws_close. If it fires before
        # response.done lands, the manager force-clears the _response_active
        # latch (which would otherwise wedge every subsequent turn), cancels
        # the server response, surfaces a friendly assistant_text, and drops
        # back to LISTENING. Without this, a Realtime server stall between
        # response.created and response.done leaves Iris pinned to THINKING
        # forever and silently swallows future user turns.
        self._response_watchdog: Optional[threading.Timer] = None
        try:
            self._RESPONSE_TIMEOUT_SEC = float(
                os.environ.get("TOUCHLESS_REALTIME_RESPONSE_TIMEOUT", "45")
            )
        except Exception:
            self._RESPONSE_TIMEOUT_SEC = 45.0
        # ---- deferred-request watchdog ----
        # _request_model_response silently defers when _response_active is
        # already True (the normal "parallel tool call" case). If the active
        # response NEVER drains (server cancelled / failed without emitting
        # response.done, ws bounce inherited a stale latch, etc.) the
        # deferred request would wedge forever — every subsequent user turn
        # would just flip _response_requested=True and return. This timer is
        # armed on defer and, if the active response hasn't completed within
        # the window, force-clears the latch and drains the queued request
        # so the user's turn actually reaches the server.
        self._deferred_request_watchdog: Optional[threading.Timer] = None
        try:
            self._DEFERRED_REQUEST_TIMEOUT_SEC = float(
                os.environ.get("TOUCHLESS_REALTIME_DEFERRED_TIMEOUT", "12")
            )
        except Exception:
            self._DEFERRED_REQUEST_TIMEOUT_SEC = 12.0
        # ---- response.create -> response.created watchdog ----
        # Armed in _fire_response_create after a successful send; disarmed in
        # the response.created handler (the normal case) or by the benign-
        # error / teardown paths. The post-create watchdog above is the SOLE
        # recovery path once response.created arrives, but it cannot help if
        # response.created NEVER arrives (e.g. the server silently dropped
        # response.create because the conversation already had an active
        # response, or the ws send failed). Without this pre-create timer,
        # _response_active stays latched True from the optimistic set in
        # _request_model_response and Iris hangs on THINKING forever.
        self._response_create_watchdog: Optional[threading.Timer] = None
        try:
            self._RESPONSE_CREATE_TIMEOUT_SEC = float(
                os.environ.get("TOUCHLESS_REALTIME_CREATE_TIMEOUT", "10")
            )
        except Exception:
            self._RESPONSE_CREATE_TIMEOUT_SEC = 10.0
        # ---- multi-step auto-continue ----
        # The realtime model tends to STOP after a text-only turn mid-task
        # (e.g. after giving a summary). When a multi-action request is armed,
        # a text-only turn that isn't a completion gets a silent 'continue'
        # nudge so the whole list finishes autonomously. Capped to avoid loops.
        self._multistep_active = False
        self._nudge_count = 0
        self._nudge_max = 6
        self._turn_text = ""          # assistant text accumulated this response
        # ---- deterministic-summary override ----
        # When a tool in DETERMINISTIC_SUMMARY_TOOLS returns a dict with a
        # non-empty top-level 'summary' string, we OVERRIDE the LLM's
        # rephrased reply with that exact summary. The realtime model would
        # otherwise paraphrase weather/email summaries — losing precision
        # and burning tokens. The override is set in _dispatch_function_call
        # after the tool runs, then consumed in the response.text.done
        # handler (which skips speaking the LLM text and lets the override
        # path own the reply). _last_override_tool tracks the most recent
        # summary tool so chained tool calls (e.g. weather_get THEN
        # gmail_send) don't apply the first tool's override to the second
        # tool's reply.
        self._pending_override_text: Optional[str] = None
        self._last_override_tool: str = ""
        # Response-id this override was armed against. The defensive wipe
        # in `response.created` fires ONLY when the armed id differs from
        # the new response — otherwise an override armed mid-dispatch
        # (e.g. confirm-gated tool that runs across a response boundary)
        # would be nuked before its consumer ever ran.
        self._override_armed_response_id: str = ""
        self._active_response_id_seen: str = ""
        # Action chips to surface after the next reply (Connect Gmail
        # button, etc.). Cleared each time we emit the signal.
        self._pending_actions: List[str] = []
        self._last_user_text = ""     # for realtime fact-extraction observation
        # Track whether the most recent user turn came from the mic
        # (voice) vs the typed input field. Safety gate uses this to
        # decide whether to surface the "[voice command]" confirm
        # banner + run voice-spoof defense. Defaults to False so
        # typed-only sessions don't get the voice treatment.
        self._last_input_was_voice: bool = False
        # Rolling conversation buffer for the Jarvis prose renderer — the
        # last ~6 (role, text) turns so replies can naturally reference
        # earlier context ("heads up before your meeting", "since you
        # mentioned the BBQ"). Bounded; oldest entries fall off.
        self._convo_buffer: List[str] = []
        self._CONVO_BUFFER_MAX = 8
        self._memory_summary_sent = False  # send memory note once per session
        self._failed_retries = 0      # retries used for a failed response turn
        # One-shot latch: True between when we send client.cancel_response()
        # due to a user-declined safety-gate modal and the resulting server
        # response.done arriving. Consumed in the response.done handler so
        # the retry block (built for TPM rate-limits) does NOT fire a phantom
        # response.create that would wedge the next user turn behind a stuck
        # _response_active latch.
        self._intentional_cancel_pending: bool = False
        # Monotonically-increasing turn counter — bumped in the
        # response.created handler so every new server response is a new
        # "turn". Used by _fire_override_inline to detect a stale queued
        # GUI-thread invocation racing against a fresh user turn: the WS
        # reader queues _fire_override_inline_on_gui_thread and returns
        # immediately, so if the user starts a new turn before the queued
        # slot runs, the slot would flash the pill back to LISTENING and
        # drop a stale decline bubble on top of the fresh turn. The slot
        # compares the turn_id it was queued with against the current
        # value and no-ops on mismatch.
        self._turn_id: int = 0
        self._last_nudge_ts = 0.0     # min-interval guard against nudge bursts
        # Task QUEUE — a multi-action command is split into atomic sub-tasks and
        # fed to the model ONE AT A TIME (each like its own prompt), so every
        # step is completed fully and the responses are paced (avoids the
        # rate-limit bursts that left steps half-done).
        self._task_queue: list[str] = []
        self._task_total = 0
        self._in_queue = False
        self._planning = False        # awaiting the model's task-list plan
        self._orig_command = ""        # the raw multi-action request being planned
        # Layer 0 deterministic command router — initialized lazily on
        # first start() once the per-session logger exists.
        self._command_router = None
        # Project-RAG indexer — a single ProjectMemoryStore instance shared
        # across sessions, plus a daemon thread that re-indexes known
        # sibling project roots on an interval. Both built lazily in
        # start(); the stop event lets stop() break the sleep loop
        # cleanly without waiting for the next interval.
        self._project_store: Optional[Any] = None
        self._indexer_thread: Optional[threading.Thread] = None
        self._stop_indexer = threading.Event()
        # Self-learning daemon — mines project files, git config, tool
        # call history, and connector endpoints (Gmail / MS365 contacts)
        # for facts to write into the memory store. Same lifecycle as
        # the project indexer: built lazily in start(), stop() sets the
        # stop event so the worker exits before its next interval.
        self._self_learner: Optional[Any] = None
        self._self_learner_thread: Optional[threading.Thread] = None
        self._stop_self_learner = threading.Event()
        # Local-handler TTS — synthesize spoken reply for Layer 0 / Layer 1
        # responses that DON'T go through the realtime websocket (which
        # already streams its own PCM16 audio). Uses OpenAI's REST TTS
        # endpoint (/v1/audio/speech) with response_format='pcm' so the
        # bytes can be fed straight into AudioPlayer (24 kHz mono int16).
        # Per-text cache keeps repeat phrases ('Done.', 'On it.') free.
        self._tts_lock = threading.Lock()
        self._tts_cache: Dict[str, bytes] = {}
        self._tts_cache_max = 64  # ~few MB at most for short replies
        # Conversational-speech rewriter cache. Keyed by ORIGINAL text so a
        # cache hit avoids both the gpt-5-mini round-trip AND keeps the
        # corresponding TTS-bytes cache (keyed on humanized text) coherent
        # across calls. Capped separately from the PCM cache.
        self._tts_humanize_cache: Dict[str, str] = {}
        self._tts_humanize_cache_max = 64
        # Cortex viz: monotonic timestamp until which the core should
        # stay PURPLE (speaking). _emit_reply_output_pulse sets this to
        # now + speak_ms when it flips the core to 'speaking'; _set_state
        # consults it to suppress competing 'listening' pushes that
        # would otherwise immediately overwrite the purple glow when an
        # internal state machine transition (e.g. response.done arriving
        # from the WS reader, or the Layer 0 router moving back to
        # LISTENING right after dispatching a local reply) races the
        # speaking pulse. UI state still updates normally — only the
        # cortex_emit.core_state('listening') call is suppressed.
        self._cortex_speaking_until: float = 0.0

    # ---- public API ----

    @property
    def config(self) -> LiveApiConfig:
        return self._config

    @property
    def state(self) -> LiveApiState:
        return self._state

    def set_confirm_callback(self, cb: Optional[ConfirmCallback]) -> None:
        self._confirm_callback = cb
        try:
            from .safety_gate import install_confirm_callback
            install_confirm_callback(cb)
        except Exception:
            pass

    def is_running(self) -> bool:
        return self._state not in (LiveApiState.OFF, LiveApiState.ERROR)

    def _ensure_audit_log(self) -> None:
        """Lazily construct the process-wide AuditLog and let it
        subscribe to the global InvocationBus. Idempotent — subsequent
        calls are no-ops. Failures are non-fatal: audit logging is a
        substrate convenience, not a session-blocking dependency."""
        if self._audit_log is not None:
            return
        try:
            from .audit_log import AuditLog
            self._audit_log = AuditLog()
            if self._logger:
                self._logger.event(
                    "audit_log_attached",
                    db_path=str(self._audit_log.db_path))
        except Exception as exc:
            if self._logger:
                self._logger.exception("audit_log_attach_failed", exc)
            self._audit_log = None
        # Phase-1 speed-pass: kick off the hot-tool prewarmer in a
        # background thread. Idempotent (start_background_prewarm
        # short-circuits on second call). The user perceives the win
        # on their FIRST tool call after Iris starts — imports + UIA
        # walker etc. are warm rather than cold.
        try:
            from .hot_prewarm import start_background_prewarm
            start_background_prewarm()
        except Exception as exc:
            if self._logger:
                self._logger.exception("hot_prewarm_start_failed", exc)
        # Phase-2 substrate: the reliability ledger and the stuck-
        # pattern detector both auto-subscribe to the global
        # InvocationBus on construction. Lazy-construct them here so
        # every tool call this session gets observed. Failure is
        # non-fatal — substrate, not session-blocking.
        try:
            from .reliability_ledger import global_ledger
            global_ledger()
            if self._logger:
                self._logger.event("reliability_ledger_attached")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "reliability_ledger_attach_failed", exc)
        try:
            from .stuck_pattern_detector import global_stuck_detector
            stuck = global_stuck_detector()
            stuck.attach_to_bus()
            if self._logger:
                self._logger.event("stuck_pattern_detector_attached")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "stuck_pattern_detector_attach_failed", exc)
        # Phase-3 wiring: subscribe the earcon dispatcher to the bus
        # so the user hears soft non-verbal acks (done/error/needs-
        # confirm) for tool calls instead of waiting for a full
        # spoken reply. Honors QuietMode and incognito internally.
        try:
            from .earcon_wiring import global_earcon_dispatcher
            global_earcon_dispatcher().attach_to_bus()
            if self._logger:
                self._logger.event("earcon_dispatcher_attached")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "earcon_dispatcher_attach_failed", exc)
        # Phase-3 polish: on the FIRST auth_revoked / not_connected
        # error per session, surface a 1-click "Reconnect X" chip
        # via the existing suggested_actions signal. Without this,
        # users had to wait for the proactive_nudge cooldown (5
        # errors in 30 min) or read the raw error.
        try:
            from .reauth_nudge import ReauthNudger
            self._reauth_nudger = ReauthNudger(
                notifier=lambda actions:
                    self.suggested_actions.emit(list(actions)))
            self._reauth_nudger.attach_to_bus()
            if self._logger:
                self._logger.event("reauth_nudger_attached")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "reauth_nudger_attach_failed", exc)
        # Phase-4 self-learning: skill consolidator watches the bus
        # for recurring multi-step shapes. When the same plan shape
        # has run successfully 3+ times, surface a "Want me to save
        # this as a skill?" nudge so future runs are 0-token replays.
        try:
            from .skill_consolidator import SkillConsolidator
            self._skill_consolidator = SkillConsolidator(
                handler=self._on_skill_consolidation_nudge)
            self._skill_consolidator.attach_to_bus()
            if self._logger:
                self._logger.event("skill_consolidator_attached")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "skill_consolidator_attach_failed", exc)

    def _on_skill_consolidation_nudge(self, nudge) -> None:
        """Surface a 'save this as a skill' suggestion in chat. Body
        queued into _pending_notes so Iris speaks it in the next
        reply window."""
        try:
            shape = (nudge.shape or "").replace(">", " → ")
            body = (f"I've noticed you've done this {nudge.occurrence_count}"
                    f" times: {shape}. Want me to save it as a "
                    f"\"{nudge.suggested_skill_name}\" skill so it's "
                    "one-tap next time?")
            should_kick = False
            with self._lock:
                self._pending_notes.append(f"💡 {body}")
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "skill_consolidation_nudge",
                    shape=nudge.shape,
                    count=nudge.occurrence_count,
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "skill_consolidation_nudge_failed", exc)

    def _track_watcher(self, name: str) -> None:
        """Record a watcher name we just registered with the global
        Sentinel so stop() can unregister exactly the set this manager
        wired in (and nothing else)."""
        with self._lock:
            if name not in self._registered_watcher_names:
                self._registered_watcher_names.append(name)

    def _register_phase_3_watchers(self) -> None:
        """Wire the Phase-3+ background watchers (sentinel start,
        standing-orders evaluator, anticipation engine, vision
        observer, proactive nudges, etc.). Idempotent: re-runs are
        no-ops because Sentinel.register() replaces watchers with the
        same name. Called from start() after the logger is created
        so wiring events land in the session log.

        Ordering note: the global Sentinel daemon is started AT THE
        END of this method, AFTER every watcher has been registered.
        Starting it first lets the tick thread spin while we're still
        wiring — and with last_run_at=0.0 the very first tick would
        run every just-registered watcher back-to-back on the daemon
        thread, contending with start()'s own GUI-thread work."""
        # Phase-3 wiring: prime InterruptionGate signals with a
        # synchronous baseline-False seed; the actual Win32 probe
        # runs on a daemon thread inside prime_signals_safely() so
        # the GUI thread is never blocked by psutil + ConsentStore.
        # Without the seed, the gate would fail-CLOSED for every
        # NORMAL/LOW interruption until the first watcher tick,
        # blocking legitimate startup earcons / briefings.
        try:
            from .system_signals import (
                prime_signals_safely,
                register_with_sentinel as _register_signals,
            )
            prime_signals_safely()
            _register_signals()
            self._track_watcher("system_signals")
            if self._logger:
                self._logger.event("system_signals_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception("system_signals_wire_failed", exc)
        # Phase-3 wiring: IDE focus watcher → repo context. When the
        # user switches to VS Code / Cursor / JetBrains with a known
        # project open, the watcher caches RepoContext so the planner
        # auto-injects project name + branch + recent commits into
        # prompts without the user having to say "while we're in
        # my Touchless repo...".
        try:
            from .repo_focus_watcher import (
                register_with_sentinel as _register_repo_focus)
            _register_repo_focus()
            self._track_watcher("repo_focus")
            if self._logger:
                self._logger.event("repo_focus_watcher_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "repo_focus_watcher_wire_failed", exc)
        # Phase-4 wiring: ambient screen awareness. Sentinel-tickable
        # capture (~30s) of a compact ScreenSummary. Honors incognito
        # + screen-sharing signal. Planner's _recall_context reads
        # the cached summary whenever the user's request looks
        # vision-relevant.
        try:
            from .screen_awareness import (
                register_with_sentinel as _register_screen)
            _register_screen()
            self._track_watcher("screen_awareness")
            if self._logger:
                self._logger.event("screen_awareness_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "screen_awareness_wire_failed", exc)
        # Phase-5 wiring: standing orders evaluator. Persistent
        # background goals ("watch my inbox for the Q3 contract")
        # survive restarts via SQLite. Sentinel tick every ~60s
        # evaluates conditions and fires the notifier when a goal's
        # trigger is met. The notifier surfaces the result through
        # the same pending-notes path as proactive nudges.
        try:
            from .standing_orders import (global_evaluator,
                                           global_store)
            evaluator = global_evaluator(
                notifier=self._on_standing_order_fire)
            # Registry will be set after _ensure_registry_wiring;
            # ensure the evaluator picks it up there.
            self._standing_orders_evaluator = evaluator
            self._standing_orders_store = global_store()
            from .sentinel import global_sentinel
            # 15s tick (was 60s): short-fuse reminders ("in 1
            # minute to check the bacon") used to take up to 2 min
            # to fire — felt broken. 15s gives sub-minute reminders
            # accuracy within ~15s and still keeps the tick cost
            # negligible (the evaluator just iterates active orders
            # — typically <20).
            global_sentinel().register(
                "standing_orders", evaluator.tick,
                interval_sec=15.0, max_run_ms=2000)
            self._track_watcher("standing_orders")
            if self._logger:
                self._logger.event(
                    "standing_orders_wired",
                    active=len(global_store().all_active()))
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "standing_orders_wire_failed", exc)
        # Phase-5: autonomous reliability advisor. Every ~6h scans
        # the reliability ledger; surfaces a 'this connector is
        # chronically unhealthy — want me to switch to <alternate>?'
        # nudge when the chronic threshold is crossed. Once-per-tool-
        # per-week cooldown so we don't nag.
        try:
            from .reliability_advisor import (
                register_with_sentinel as _register_advisor)
            _register_advisor(handler=self._on_reliability_advice)
            self._track_watcher("reliability_advisor")
            if self._logger:
                self._logger.event("reliability_advisor_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "reliability_advisor_wire_failed", exc)
        # Phase-6 wiring: anticipatory action engine. Sentinel-tickable
        # predictor that fires "your 9am starts in 12 min — want me
        # to prep?" / "you usually open this app around now — save
        # as a routine?" / "this standing order fired 12h ago and
        # you didn't act on it — refresh?" Registry-dependent
        # predicates (meeting_imminent) gracefully no-op until
        # _ensure_registry_wiring hands the registry over.
        try:
            from .anticipation_engine import (
                register_with_sentinel as _register_antic)
            self._anticipation_engine = _register_antic(
                handler=self._on_anticipation)
            self._track_watcher("anticipation")
            if self._logger:
                self._logger.event("anticipation_engine_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "anticipation_engine_wire_failed", exc)
        # Phase-6 wiring: vision observer. Sentinel-tickable
        # delta-detector that watches screen transitions and
        # surfaces "noticed an error popup — read it for you?"
        # / "looks like a stack trace — want me to look it up?"
        # / "you just opened stripe.com which you mentioned —
        # want a summary?" Cooldown-gated per pattern kind.
        try:
            from .vision_observer import (
                register_with_sentinel as _register_vobs)
            self._vision_observer = _register_vobs(
                handler=self._on_vision_observation)
            self._track_watcher("vision_observer")
            try:
                from .session_buffer import global_session_buffer
                self._vision_observer._session = (
                    global_session_buffer())
            except Exception:
                pass
            if self._logger:
                self._logger.event("vision_observer_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "vision_observer_wire_failed", exc)
        # Start the Sentinel daemon LAST — after every watcher above
        # has been registered. Doing this here (rather than first)
        # guarantees the tick thread can't fire a watcher mid-wiring,
        # which would have it racing start()'s GUI-thread setup.
        try:
            from .sentinel import global_sentinel
            global_sentinel().start()
            if self._logger:
                self._logger.event("sentinel_started")
        except Exception as exc:
            if self._logger:
                self._logger.exception("sentinel_start_failed", exc)

    def _watcher_deliverable(self) -> bool:
        """Shared early-exit for proactive-watcher callbacks. Sentinel
        runs on a process-global daemon thread; after stop() the
        manager is torn down but the registered ticks keep firing.
        Returns False when the manager has no live client or is in a
        teardown / non-running state, so callbacks bail without
        appending to _pending_notes or kicking _drain_pending."""
        if self._teardown_in_progress:
            return False
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR,
                           LiveApiState.CONNECTING):
            return False
        client = self._client
        if client is None:
            return False
        return True

    def _on_anticipation(self, anticipation) -> None:
        """AnticipationEngine fired — queue the headline + proactive
        surface it if we're idle."""
        if not self._watcher_deliverable():
            return
        try:
            body = f"🔮 {anticipation.headline}"
            should_kick = False
            with self._lock:
                self._pending_notes.append(body)
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "anticipation_queued",
                    kind=getattr(anticipation.kind, "value",
                                 str(anticipation.kind)),
                    severity=anticipation.severity,
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "anticipation_notify_failed", exc)

    def _on_vision_observation(self, observation) -> None:
        """VisionObserver fired — queue the headline + proactive
        surface it if we're idle. Uses 👁 prefix so the UI / TTS
        knows it came from on-screen reasoning."""
        if not self._watcher_deliverable():
            return
        try:
            body = f"👁 {observation.headline}"
            should_kick = False
            with self._lock:
                self._pending_notes.append(body)
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "vision_observation_queued",
                    kind=getattr(observation.kind, "value",
                                 str(observation.kind)),
                    severity=observation.severity,
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "vision_observation_notify_failed", exc)

    def _on_reliability_advice(self, advice) -> None:
        """ReliabilityAdvisor fired — queue the headline into
        pending notes + proactive-surface it if we're idle."""
        if not self._watcher_deliverable():
            return
        try:
            body = f"📊 {advice.headline}"
            should_kick = False
            with self._lock:
                self._pending_notes.append(body)
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "reliability_advice_queued",
                    tool=advice.tool,
                    error_rate=advice.error_rate,
                    alternate=advice.suggested_alternate,
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "reliability_advice_notify_failed", exc)

    def _on_standing_order_fire(self, order, result) -> bool:
        """Standing order fired. Queue the user-facing notification
        into _pending_notes + proactive-surface it if we're idle.
        Returns True when the reminder was queued/delivered to a path
        that will surface it, False when delivery is impossible right
        now (no client, OFF / CONNECTING state). The evaluator uses
        the return value to decide whether to mark the order as fired
        or leave it for retry on the next tick — without this, a
        disconnected client silently bumped fire_count and the
        reminder was lost forever."""
        client = self._client
        deliverable = (client is not None
                       and getattr(client, "connected", False)
                       and self._state not in (
                           LiveApiState.OFF, LiveApiState.ERROR,
                           LiveApiState.CONNECTING))
        if not deliverable:
            if self._logger:
                self._logger.event(
                    "standing_order_fire_deferred",
                    order_id=order.id,
                    trigger=order.trigger_kind,
                    state=(self._state.value
                           if self._state else "?"))
            return False
        try:
            label = order.short_label()
            body = f"📌 \"{label}\" — {result.detail}"
            should_kick = False
            with self._lock:
                self._pending_notes.append(body)
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "standing_order_fired",
                    order_id=order.id,
                    trigger=order.trigger_kind,
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
            return True
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "standing_order_notify_failed", exc)
            return False

    def _wire_proactive_nudges(self) -> None:
        """Phase-3 wiring: proactive nudges (long-idle, cost-cap
        approaching, unhealthy tool). Handler queues the nudge body
        into _pending_notes so the assistant surfaces it in its next
        reply window. Respects incognito + Gate."""
        try:
            from .proactive_nudges import (
                register_with_sentinel as _register_nudges)
            _register_nudges(handler=self._on_proactive_nudge)
            self._track_watcher("proactive_nudges")
            if self._logger:
                self._logger.event("proactive_nudges_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "proactive_nudges_wire_failed", exc)

    def _on_proactive_nudge(self, nudge) -> None:
        """Nudge handler — queues the nudge body into pending notes
        so it surfaces in the next reply window. When no response is
        currently active (user is idle), proactively kicks off a
        response.create so the nudge actually gets spoken instead of
        sitting in the queue forever."""
        if not self._watcher_deliverable():
            return
        try:
            body = getattr(nudge, "body", "") or ""
            if not body:
                return
            tag = "💡" if nudge.severity == "low" else "⚠️"
            should_kick = False
            with self._lock:
                self._pending_notes.append(f"{tag} {body}")
                # If nothing is in flight, surface the nudge now.
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "proactive_nudge_queued",
                    kind=getattr(nudge.kind, "value", str(nudge.kind)),
                    severity=nudge.severity,
                    kicked=should_kick)
            if should_kick:
                # Drain via the existing helper which handles the
                # response_active flip + pending_notes consumption.
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception("nudge_queue_failed", exc)
        # Registry-dependent wiring (calendar briefing, file
        # watcher) runs LATER via _ensure_registry_wiring() once
        # self._registry is built.

    def _ensure_registry_wiring(self) -> None:
        """Called by start() after self._registry is constructed.
        Wires Phase-3 watchers that need a real ToolRegistry to
        dispatch through. Idempotent: re-runs are no-ops because
        the Sentinel's register() replaces watchers with the same
        name."""
        if getattr(self, "_registry", None) is None:
            return
        # Phase-5: hand the registry to the standing-orders
        # evaluator so its predicates (inbox_match, tool_returns_ok)
        # can dispatch real tools. Without this they all return
        # "no registry available" and never fire.
        try:
            evaluator = getattr(self, "_standing_orders_evaluator",
                                None)
            if evaluator is not None:
                evaluator.set_registry(self._registry)
        except Exception:
            pass
        # Phase-6: anticipation engine also needs the registry for
        # its meeting_imminent predicate (calls calendar_list_events).
        try:
            engine = getattr(self, "_anticipation_engine", None)
            if engine is not None:
                engine.set_registry(self._registry)
        except Exception:
            pass
        # Calendar briefing: poll Google/MS Calendar every ~60s,
        # surface briefings through the interruption gate.
        try:
            from .calendar_briefing_watcher import (
                register_with_sentinel as _register_briefing)
            _register_briefing(
                registry=self._registry,
                notifier=self._on_briefing_due,
            )
            self._track_watcher("calendar_briefing")
            if self._logger:
                self._logger.event("calendar_briefing_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "calendar_briefing_wire_failed", exc)
        # File watcher: polling daemon over rule roots, dispatches
        # via the tool registry. No watchdog dep.
        try:
            from .file_watcher_daemon import (
                register_with_sentinel as _register_files)
            from .file_watcher_rules import RulesEngine
            if not hasattr(self, "_file_rules_engine") \
                    or self._file_rules_engine is None:
                self._file_rules_engine = RulesEngine()
            _register_files(
                engine=self._file_rules_engine,
                dispatcher=lambda action, args:
                    self._registry.call(action, args),
            )
            self._track_watcher("file_watcher")
            if self._logger:
                self._logger.event("file_watcher_wired")
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "file_watcher_wire_failed", exc)

    def _on_briefing_due(self, briefing, decision) -> None:
        """Calendar briefing watcher fires this when an upcoming
        meeting is eligible to be announced. Pushes the briefing
        body into the assistant's pending-notes queue + proactively
        surfaces it if no response is currently active."""
        if not self._watcher_deliverable():
            return
        try:
            if not getattr(briefing, "body", ""):
                return
            text = f"📅 {briefing.body}"
            should_kick = False
            with self._lock:
                self._pending_notes.append(text)
                if not self._response_active \
                        and not self._response_requested:
                    self._response_requested = True
                    should_kick = True
            if self._logger:
                self._logger.event(
                    "briefing_queued",
                    event_id=briefing.event_id,
                    starts_in_min=briefing.starts_in_minutes,
                    channel=getattr(decision, "suggested_channel",
                                    "voice"),
                    kicked=should_kick)
            if should_kick:
                try:
                    self._drain_pending()
                except Exception:
                    with self._lock:
                        self._response_requested = False
                        self._response_active = False
        except Exception as exc:
            if self._logger:
                self._logger.exception("briefing_queue_failed", exc)

    def start(self) -> None:
        # Fast state-reset section under the lock. We intentionally
        # DROP the lock before the heavy substrate wiring below — the
        # original implementation held self._lock across the entire
        # start() body (audit_log + bus subscribers + connectors +
        # project indexer + self-learner + audio player + Win32 signal
        # probes). That serialized 3-10 s of work onto the GUI thread
        # AND, if any construction raised, left the lock permanently
        # held — producing the "Application is not responding" overlay
        # and the freeze-on-second-open the user was hitting.
        with self._lock:
            # Response-cycle state ALWAYS resets, even if start() re-entered
            # while we're already CONNECTING / LISTENING / THINKING. Without
            # this, a re-entered start() that hits the early-return below
            # would leave any stale latch in place — and any prior session
            # that exited without draining (ws bounce, transient error,
            # response.failed) would silently swallow every subsequent user
            # turn via _request_model_response's deferred-but-never-sent path.
            # Heavy substrate (connectors, indexer, audio) is still guarded
            # to avoid rebuilding it for an already-live session.
            self._response_active = False
            self._pending_notes = []
            self._response_requested = False
            if self._state not in (LiveApiState.OFF, LiveApiState.ERROR):
                return
            self._teardown_in_progress = False
            # Re-arm the GUI-alive guard. stop() sets this False; without
            # re-arming here every code path guarded by
            # `getattr(self, "_gui_alive", True)` (_fire_override_inline,
            # the cortex-edge pulses, the queued GUI marshals) silently
            # becomes a permanent no-op after the first stop()/start()
            # cycle in the same process.
            self._gui_alive = True
            self._multistep_active = False
            self._nudge_count = 0
            self._turn_text = ""
            self._failed_retries = 0
            self._last_nudge_ts = 0.0
            self._task_queue = []
            self._in_queue = False
            self._planning = False
        # Defensive: clear any leftover response watchdog from a prior
        # session so its callback can't race the new session.
        self._cancel_response_watchdog()
        self._cancel_deferred_request_watchdog()
        # --- Heavy wiring OUTSIDE the lock from here on ---
        # Phase-1 substrate: ensure the audit log is subscribed so
        # every tool invocation this session is recorded. Idempotent
        # — the helper short-circuits if already built.
        self._ensure_audit_log()
        if not self._config.enabled:
            self._emit_error("Live API is disabled (TOUCHLESS_LIVE_API_ENABLED=false)")
            return
        backend_kind = (self._config.backend or "cloud").strip().lower()
        # Cloud needs an OpenAI key. Local doesn't.
        if backend_kind == "cloud" and not self._config.api_key:
            self._emit_error("OPENAI_API_KEY is not set")
            return

        self._logger = LiveApiLogger(
            log_dir=self._config.log_dir,
            debug_text_logging=self._config.debug_text_logging,
        )
        # Bind the tool-call log session id to this LiveApiManager
        # session so cross-tool co-occurrence buckets line up with
        # what the user perceives as one conversation. Best-effort.
        try:
            from .cortex.tool_call_log import set_session_id as _set_tc_sid
            _set_tc_sid(Path(self._logger.jsonl_log_path).stem)
        except Exception:
            pass
        self._logger.event(
            "session_start",
            backend=backend_kind,
            model=self._config.model if backend_kind == "cloud" else self._config.local_llm_model_filename,
            send_screen_always=self._config.send_screen_always,
            send_screen_interval_sec=self._config.send_screen_interval_sec,
        )

        self._screen = ScreenContext(
            max_width=self._config.screen_max_width,
            jpeg_quality=self._config.screen_jpeg_quality,
            logger=self._logger,
            debug_save_dir=(self._config.log_dir / "screenshots") if self._config.debug_save_screenshots else None,
        )
        self._executor = ToolExecutor(
            config=self._config,
            logger=self._logger,
            screen_context=self._screen,
            confirm_callback=self._confirm_callback,
            external_action_router=self._external_action_router,
            notify_callback=self._push_assistant_note,
        )
        # API-first connectors (OpenClaw-style). The registry only exposes
        # the ones whose available() is True this session; everything else
        # falls through to the GUI computer-use executor as before.
        try:
            from .connectors import build_connector_registry
            connectors = build_connector_registry(self._executor)
        except Exception as exc:
            self._logger.exception("connector_registry_init_failed", exc)
            connectors = None
        self._registry = ToolRegistry(self._executor, connectors)
        # Phase-3+ background watchers (sentinel, standing
        # orders, anticipation, vision, proactive nudges). Idempotent
        # — Sentinel.register replaces watchers with the same name,
        # so re-running on every start() is safe and ensures
        # persistent reminders fire even after the user closed and
        # reopened the Iris chat panel.
        try:
            self._register_phase_3_watchers()
            self._wire_proactive_nudges()
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "phase3_watcher_wiring_failed", exc)
        # Phase-3 wiring that needs the tool registry: register
        # the calendar-briefing watcher + file-watcher daemon.
        try:
            self._ensure_registry_wiring()
        except Exception as exc:
            if self._logger:
                self._logger.exception(
                    "phase3_registry_wiring_failed", exc)
        # Fresh session: nothing loaded on demand yet.
        self._loaded_connector_schemas = []
        self._iris_planner = None
        self._memory_summary_sent = False  # re-send on fresh session
        # Layer 0 router. Lazy-import keeps the manager loadable on
        # systems where Touchless's voice modules can't initialize
        # (e.g. headless CI without sounddevice).
        try:
            from .command_router import CommandRouter
            self._command_router = CommandRouter(logger=self._logger)
        except Exception as exc:
            self._logger.exception("command_router_init_failed", exc)
            self._command_router = None

        # Project-RAG: lazily build the shared ProjectMemoryStore,
        # register the known sibling roots, and spawn ONE daemon
        # indexer thread that runs incremental re-indexes on a
        # 5-min interval. All best-effort — a RAG failure must
        # NEVER block realtime startup. The thread itself is a
        # no-op when TOUCHLESS_IRIS_MEMORY=0.
        if (self._project_store is None
                and self._indexer_thread is None
                and os.environ.get("TOUCHLESS_IRIS_MEMORY", "1") != "0"):
            try:
                self._init_project_indexer()
            except Exception as exc:
                if self._logger:
                    self._logger.exception("project_indexer_init_failed", exc)
                self._project_store = None

        # Self-learning daemon — mines project files, git config,
        # tool call patterns, and email/calendar contacts for
        # facts. Same lifecycle as the project indexer; gated by
        # TOUCHLESS_SELF_LEARN_ENABLED (default on). Memory wiring
        # also requires TOUCHLESS_IRIS_MEMORY to be enabled — the
        # daemon writes through a MemoryManager and there's nothing
        # to do without that store.
        if (self._self_learner_thread is None
                and os.environ.get("TOUCHLESS_IRIS_MEMORY", "1") != "0"
                and os.environ.get("TOUCHLESS_SELF_LEARN_ENABLED", "1") != "0"):
            try:
                self._init_self_learner()
            except Exception as exc:
                if self._logger:
                    self._logger.exception("self_learner_init_failed", exc)
                self._self_learner = None

        backend_kind = (self._config.backend or "cloud").strip().lower()
        if backend_kind == "subscription":
            # Stub — the hosted Touchless proxy doesn't exist yet.
            # Surface a clear error so the chat panel can show a
            # "Subscription not yet available — falling back to
            # local model. Restart Iris to use it." note instead of
            # silently hanging on session start.
            self._emit_error(
                "Touchless subscription backend isn't shipped yet. "
                "Set OPENAI_API_KEY to use the cloud backend, or "
                "wait for the bundled local model to ship in a "
                "future Touchless update.")
            self._set_state(LiveApiState.ERROR, "Subscription unavailable")
            return
        if backend_kind == "local":
            # Local backend exposes the same shape as RealtimeClient
            # (start/stop/join, send_audio_chunk, send_tool_result,
            # request_response, on_event/on_connected/on_closed/
            # on_error). LiveApiManager treats them interchangeably.
            from .local_backend import LocalBackend
            self._client = LocalBackend(
                config=self._config,
                logger=self._logger,
                tools=self._current_tool_schemas(),
                system_instructions=build_system_instructions(),
                on_event=self._handle_event,
                on_connected=self._on_ws_connected,
                on_closed=self._on_ws_closed,
                on_error=self._on_ws_error,
                require_audio=not self._text_only,
            )
        else:
            self._client = RealtimeClient(
                config=self._config,
                logger=self._logger,
                tools=self._current_tool_schemas(),
                system_instructions=build_system_instructions(),
                on_event=self._handle_event,
                on_connected=self._on_ws_connected,
                on_closed=self._on_ws_closed,
                on_error=self._on_ws_error,
                text_only=self._text_only,
                voice_output=self._voice_output,
            )

        # Reset per-session diagnostic flag so the first audio-drop
        # event re-logs each new session (helps diagnose 'audio went
        # silent again' after a stop()/start() cycle).
        self._audio_drop_logged = False
        # Spin up the audio output player on demand. Lives as a child
        # of the manager so it gets cleaned up on stop(). Failure
        # leaves the player as None and audio deltas become no-ops.
        if self._voice_output and self._audio_player is None:
            try:
                from .audio_player import AudioPlayer
                self._audio_player = AudioPlayer(
                    sample_rate=self._config.audio_sample_rate,
                    parent=self,
                )
                if not self._audio_player.is_enabled():
                    # Init failed (no device, format unsupported) — drop
                    # the reference so we don't try to feed it bytes.
                    reason = "unknown"
                    try:
                        reason = self._audio_player.init_error() or "unknown"
                    except Exception:
                        pass
                    # Surface to terminal too — JSONL logs are easy
                    # to miss; users hit 'why is voice silent' a lot.
                    print(
                        f"AUDIO: player init failed (no device) — {reason}",
                        file=sys.stderr, flush=True,
                    )
                    if self._logger:
                        self._logger.event(
                            "audio_player_init_no_device",
                            reason=reason)
                    self._audio_player = None
                else:
                    print(
                        f"AUDIO: player init ok (voice_output={self._voice_output}, "
                        f"sample_rate={self._config.audio_sample_rate})",
                        file=sys.stderr, flush=True,
                    )
                    if self._logger:
                        self._logger.event(
                            "audio_player_init_ok",
                            voice_output=self._voice_output,
                            sample_rate=self._config.audio_sample_rate)
            except Exception as exc:
                print(
                    f"AUDIO: player init crashed — {type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
                if self._logger:
                    self._logger.exception("audio_player_init_failed", exc)
                self._audio_player = None
        elif self._voice_output and self._audio_player is not None:
            if self._logger:
                self._logger.event("audio_player_already_alive")
        elif not self._voice_output and self._logger:
            self._logger.event("audio_player_skipped_voice_off")

        if self._text_only:
            # No mic in text-only mode — the user types commands in
            # the UI instead of speaking. AudioStream stays None and
            # send_audio_chunk is never called.
            self._audio = None
        else:
            self._audio = AudioStream(
                sample_rate=self._config.audio_sample_rate,
                chunk_ms=self._config.audio_chunk_ms,
                on_chunk=self._on_audio_chunk,
                logger=self._logger,
            )

        self._set_state(LiveApiState.CONNECTING, "Connecting to Realtime API...")
        if not self._client.start():
            self._emit_error("WebSocket failed to start")
            return

    # ---- response.create gating ------------------------------------------
    # The server allows only ONE response in flight at a time; a second
    # response.create while one is active is rejected with
    # conversation_already_has_active_response. Every path that wants a model
    # turn (user input, function-call results — including PARALLEL tool calls
    # in one response — and background notes) goes through this gate, which
    # fires immediately when idle or defers to response.done when busy.

    def _fire_response_create(self, prefix_notes: Optional[list] = None) -> bool:
        """Actually send the response.create (with optional leading note text).
        Returns False if the client isn't in a state to send, or if the
        underlying WS send returned False (e.g. ws disconnected mid-send).
        On a successful send, arms the pre-create watchdog so a missing
        response.created (silent server drop) can be recovered from."""
        client = self._client
        if client is None or not getattr(client, "connected", False):
            return False
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR, LiveApiState.CONNECTING):
            return False
        try:
            if prefix_notes:
                client.send_text_message("\n\n".join(prefix_notes))
            ok = bool(client.request_response())
            if not ok:
                if self._logger:
                    try:
                        self._logger.event("request_response_send_failed")
                    except Exception:
                        pass
                return False
            self._arm_response_create_watchdog()
            return True
        except Exception as exc:
            if self._logger:
                self._logger.exception("request_response_failed", exc)
            return False

    def _arm_response_watchdog(self, response_id: str) -> None:
        """Arm a one-shot timer that recovers a stuck THINKING state.

        Called from the response.created handler. If response.done never
        arrives (server stalled / read thread parked on recv() with no
        bytes / model emitted nothing for a slow tool plan), this fires
        and force-clears the latch so the next user turn isn't silently
        deferred forever."""
        self._cancel_response_watchdog()

        def _fire() -> None:
            # Bail if we already returned to LISTENING via the normal path.
            if not self._response_active:
                return
            if self._logger:
                try:
                    self._logger.event(
                        "response_watchdog_timeout",
                        response_id=response_id,
                        sec=self._RESPONSE_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
            client = self._client
            if client is not None:
                try:
                    client.cancel_response()
                except Exception:
                    pass
            with self._lock:
                self._response_active = False
                self._response_requested = False
            # Drop any armed deterministic-summary override that belonged
            # to the stalled turn so it can't fire against the next reply.
            self._pending_override_text = None
            self._override_armed_response_id = ""
            self._last_override_tool = ""
            msg = (
                "I lost the thread on that one — say it again and I'll "
                "give it another shot."
            )
            try:
                self.assistant_text.emit(msg)
            except Exception:
                pass
            try:
                self._set_state(LiveApiState.LISTENING, "Listening")
            except Exception:
                pass

        try:
            t = threading.Timer(self._RESPONSE_TIMEOUT_SEC, _fire)
            t.daemon = True
            self._response_watchdog = t
            t.start()
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception("response_watchdog_arm_failed", exc)
                except Exception:
                    pass

    def _cancel_response_watchdog(self) -> None:
        """Disarm the response-cycle watchdog. Safe to call repeatedly."""
        t = self._response_watchdog
        self._response_watchdog = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _arm_response_create_watchdog(self) -> None:
        """Arm a one-shot timer that recovers a stuck THINKING when
        response.create was sent but response.created never arrived.

        The post-create watchdog (_arm_response_watchdog) is armed inside
        the response.created handler — if that event never lands, the
        latch stays wedged True with NO recovery. This pre-create timer
        unwedges _response_active, defensively cancels any phantom server
        response, surfaces a friendly retry message, and drops back to
        LISTENING."""
        self._cancel_response_create_watchdog()

        def _fire() -> None:
            # Bail if response.created already landed (the handler nulls
            # this timer) or if we're already idle via some other path.
            if self._response_create_watchdog is None:
                return
            if not self._response_active:
                return
            if self._logger:
                try:
                    self._logger.event(
                        "response_create_watchdog_timeout",
                        sec=self._RESPONSE_CREATE_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
            client = self._client
            if client is not None:
                try:
                    client.cancel_response()
                except Exception:
                    pass
            with self._lock:
                self._response_active = False
                self._response_requested = False
            # Drop any armed deterministic-summary override that belonged
            # to the never-started turn so it can't fire against a later one.
            self._pending_override_text = None
            self._override_armed_response_id = ""
            self._last_override_tool = ""
            try:
                self.assistant_text.emit(
                    "I didn't get a response back from the server — "
                    "say it again and I'll give it another shot."
                )
            except Exception:
                pass
            try:
                self._set_state(LiveApiState.LISTENING, "Listening")
            except Exception:
                pass

        try:
            t = threading.Timer(self._RESPONSE_CREATE_TIMEOUT_SEC, _fire)
            t.daemon = True
            self._response_create_watchdog = t
            t.start()
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception(
                        "response_create_watchdog_arm_failed", exc)
                except Exception:
                    pass

    def _cancel_response_create_watchdog(self) -> None:
        """Disarm the response.create -> response.created watchdog.
        Safe to call repeatedly."""
        t = self._response_create_watchdog
        self._response_create_watchdog = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _arm_deferred_request_watchdog(self) -> None:
        """Arm a one-shot timer that recovers a wedged deferred request.

        When _request_model_response is called while _response_active is
        already True, the new request is stashed in _response_requested and
        nothing is sent — the design assumes the active response will land
        a response.done and _drain_pending will fire the deferred request.
        But if the active response NEVER drains (server cancelled / failed
        with an unhandled terminal event, ws bounce inherited a stale
        latch, parallel tool-call orchestration glitched), no recovery
        path exists and every subsequent user turn just stacks behind the
        stuck latch forever — the 45 s response watchdog can't help
        because it's only armed inside the response.created handler.

        This timer force-clears the latch and drains the queued request
        so the user's turn actually reaches the server."""
        self._cancel_deferred_request_watchdog()

        def _fire() -> None:
            stuck = False
            with self._lock:
                if self._response_active and self._response_requested:
                    # Force-clear the stale latch so _drain_pending below
                    # can fire (it only re-enters the active state from a
                    # clean baseline).
                    self._response_active = False
                    stuck = True
            if not stuck:
                return
            if self._logger:
                try:
                    self._logger.event(
                        "deferred_request_watchdog_force_clear",
                        sec=self._DEFERRED_REQUEST_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
            try:
                self._drain_pending()
            except Exception as exc:
                if self._logger:
                    try:
                        self._logger.exception(
                            "deferred_request_drain_failed", exc)
                    except Exception:
                        pass

        try:
            t = threading.Timer(self._DEFERRED_REQUEST_TIMEOUT_SEC, _fire)
            t.daemon = True
            self._deferred_request_watchdog = t
            t.start()
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception(
                        "deferred_request_watchdog_arm_failed", exc)
                except Exception:
                    pass

    def _cancel_deferred_request_watchdog(self) -> None:
        """Disarm the deferred-request watchdog. Safe to call repeatedly."""
        t = self._deferred_request_watchdog
        self._deferred_request_watchdog = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _request_model_response(self) -> None:
        """Ask for a model turn. If a response is already active (e.g. this is
        one of several parallel tool calls, or the original response that
        emitted the call hasn't ended yet), remember it and fire once on
        response.done instead of colliding."""
        with self._lock:
            if self._response_active:
                self._response_requested = True
                deferred = True
            else:
                self._response_active = True  # optimistic; cleared on response.done
                deferred = False
        if deferred:
            # Defense-in-depth: a deferred request has NO other escape
            # timer (the 45s response watchdog only arms inside
            # response.created, the pre-create watchdog only arms after a
            # successful send). Without this, a stale True latch silently
            # swallows every subsequent user turn forever.
            self._arm_deferred_request_watchdog()
            return
        if not self._fire_response_create():
            with self._lock:
                self._response_active = False

    def _drain_pending(self) -> None:
        """Called on response.done. Fire a single follow-up response.create if
        anything was deferred while the just-finished response was active —
        coalescing all queued notes and any deferred request into ONE turn."""
        # A drain means whatever the deferred-request watchdog was guarding
        # is now being handled — cancel it so it can't fire against the
        # follow-up turn.
        self._cancel_deferred_request_watchdog()
        with self._lock:
            notes = self._pending_notes
            requested = self._response_requested
            if not notes and not requested:
                return
            self._pending_notes = []
            self._response_requested = False
            self._response_active = True  # optimistic; cleared on next done
        if not self._fire_response_create(prefix_notes=notes):
            # Delivery failed (client disconnected, state OFF, etc.).
            # Re-queue the notes so a future drain can replay them
            # instead of silently dropping reminders / nudges.
            with self._lock:
                self._response_active = False
                if notes:
                    self._pending_notes = list(notes) + self._pending_notes
                    self._response_requested = True

    # Phrases that mean the model considers the multi-step task finished — so
    # we stop nudging instead of pestering it after it's actually done.
    _DONE_PHRASES = (
        "all set", "all done", "everything is set", "everything's set",
        "everything is open", "that's everything", "that is everything",
        "fully set up", "workspace is set", "completed everything",
        "nothing left", "all of that is done", "everything's ready",
        "everything is ready", "is there anything else", "anything else you",
        "let me know if you need", "you're all set",
    )

    def _maybe_nudge_continue(self) -> bool:
        """Called on a text-only response.done while a multi-step task is armed.
        If the model signaled completion, disarm. Otherwise inject a silent
        'continue' nudge (capped) so it finishes the remaining steps. Returns
        True if a nudge was sent (and a new response requested)."""
        txt = (self._turn_text or "").lower()
        if any(p in txt for p in self._DONE_PHRASES):
            self._multistep_active = False
            return False
        if self._nudge_count >= self._nudge_max:
            self._multistep_active = False
            if self._logger:
                self._logger.event("multistep_nudge_budget_exhausted")
            return False
        # Min interval — never burst nudges (belt-and-suspenders on top of the
        # completed-only gate).
        now = time.time()
        if now - self._last_nudge_ts < 1.5:
            return False
        self._last_nudge_ts = now
        client = self._client
        if client is None or not getattr(client, "connected", False):
            self._multistep_active = False
            return False
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR, LiveApiState.CONNECTING):
            self._multistep_active = False
            return False
        self._nudge_count += 1
        if self._logger:
            self._logger.event("multistep_nudge", n=self._nudge_count)
        try:
            client.send_text_message(
                "Continue my request. FIRST confirm the previous step actually "
                "succeeded (check its result / that the window or result is "
                "there); if it failed, redo it. THEN do the next remaining step. "
                "One step at a time, validating each, until all are done. If "
                "everything is already complete, reply 'all set' and nothing else."
            )
            self._response_active = True  # optimistic; cleared on next done
            client.request_response()
            return True
        except Exception as exc:
            if self._logger:
                self._logger.exception("multistep_nudge_failed", exc)
            self._multistep_active = False
            self._response_active = False
            return False

    # -- planner -> realtime session note ----------------------------------
    @staticmethod
    def _build_session_note(user_text: str, steps: List[Any],
                            results: List[Any], reply: str) -> str:
        """Compact natural-language note describing what the iris planner
        just handled outside the model. Injected into realtime as a system
        message so the model can resolve follow-ups ("send him a thank you
        too") that reference the prior turn.

        Designed to stay small (< ~400 chars) so it doesn't bloat context.
        Surfaces the user's request, which tools ran, and a few "interesting"
        outputs (emails, links, titles, recipients) for entity resolution.
        """
        REDACT = {"password", "token", "access_token", "refresh_token",
                  "client_secret", "api_key", "secret"}
        INTERESTING = ("email", "recipient", "link", "url", "title", "subject",
                       "id", "name", "path", "filename", "query", "phone")
        ut = (user_text or "").strip()
        if len(ut) > 200:
            ut = ut[:200] + "..."
        tool_parts: List[str] = []
        facts: List[str] = []
        for step, sr in zip(steps or [], results or []):
            tool = getattr(step, "tool", "?")
            status = getattr(sr, "status", "?")
            tool_parts.append(f"{tool}={status}")
            out = getattr(sr, "output", None)
            if not isinstance(out, dict):
                continue
            for k, v in out.items():
                kl = str(k).lower()
                if kl in REDACT or any(r in kl for r in REDACT):
                    continue
                if kl in INTERESTING and v is not None:
                    sv = str(v)
                    if len(sv) > 120:
                        sv = sv[:120] + "..."
                    facts.append(f"{k}={sv}")
        # Format as an explicit turn-pair so the realtime LLM treats
        # this as part of the conversation history (e.g. for "what
        # have I asked you?" recaps), not as background-only context.
        # Previously wrapped in parens, which the LLM skipped from
        # turn-counting answers.
        rep = (reply or "").strip()
        if len(rep) > 200:
            rep = rep[:200] + "..."
        lines = [
            f'PRIOR TURN — user asked: "{ut}"',
            f'Iris answered: {rep or "(no spoken reply)"}',
        ]
        if tool_parts:
            lines.append(
                f"(tools that ran: {', '.join(tool_parts)})")
        if facts:
            lines.append(
                f"(facts: {'; '.join(facts[:6])})")
        note = "\n".join(lines)
        return note[:800]

    def _maybe_send_memory_summary(self) -> None:
        """Inject a compact memory-context note into the realtime session
        ONCE per session, before the user's first turn. Lets realtime
        answer recall questions ('where's my office?', 'what should you
        call me?') without needing Tier 2 recall.

        Every skip path now logs so we can audit when injection didn't
        happen — the absence of a log was itself the bug to diagnose."""
        if self._memory_summary_sent:
            return  # silent on this path — fires every turn after the first
        client = self._client
        if client is None or not hasattr(client, "send_session_note"):
            if self._logger:
                self._logger.event("memory_summary_skip_no_client",
                                   has_client=bool(client),
                                   has_method=(client is not None and hasattr(
                                       client, "send_session_note")))
            return
        planner = self._iris_planner
        memory = getattr(planner, "_memory", None) if planner is not None else None
        if memory is None or not hasattr(memory, "summary_for_session"):
            if self._logger:
                self._logger.event("memory_summary_skip_no_memory",
                                   has_planner=bool(planner),
                                   has_memory=bool(memory))
            return
        try:
            note = memory.summary_for_session()
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_summary_failed", exc)
            return
        if not note:
            self._memory_summary_sent = True  # nothing to send, don't keep trying
            if self._logger:
                self._logger.event("memory_summary_skip_empty")
            return
        # Anti-fabrication anchor: the summary above renders ONLY semantic
        # facts (episodes are excluded by construction). Without this
        # explicit gap-disclosure, the model treats absence-of-evidence as
        # license to invent plausible details when asked to recap a prior
        # conversation. Pair this note with the HARD RULE — NEVER
        # FABRICATE PERSONAL / RECALL CONTEXT in SYSTEM_INSTRUCTIONS.
        note = (note + "\n(SCOPE: the list above is the COMPLETE set of "
                       "personal notes you have on this user from prior "
                       "conversations. If the user asks about a topic, "
                       "person, place, or event that is NOT in this list, "
                       "you do NOT have notes on it — do NOT invent "
                       "details; ask the user to remind you instead of "
                       "guessing.)")
        try:
            client.send_session_note(note)
            self._memory_summary_sent = True
            if self._logger:
                self._logger.event("memory_summary_injected",
                                   chars=len(note),
                                   note_preview=note[:200])
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_summary_send_failed", exc)

    def _observe_realtime_turn(self) -> None:
        """Hand the just-completed realtime turn to the memory layer for
        background fact extraction. Skips when the planner already handled
        the turn (no last user text), when realtime returned mostly tool
        calls (assistant text empty), or when the user spoke nothing of
        substance. The actual extraction runs on a worker thread inside
        MemoryManager.observe_conversation — never blocks here."""
        planner = self._iris_planner
        memory = getattr(planner, "_memory", None) if planner is not None else None
        if memory is None or not hasattr(memory, "observe_conversation"):
            if self._logger:
                self._logger.event("observe_realtime_skip_no_memory",
                                   has_planner=bool(planner),
                                   has_memory=bool(memory))
            return
        ut = (self._last_user_text or "").strip()
        if not ut:
            if self._logger:
                self._logger.event("observe_realtime_skip_no_user_text")
            return
        if self._logger:
            self._logger.event("observe_realtime_dispatch",
                               user_text_len=len(ut),
                               assistant_text_len=len((self._turn_text or "").strip()))
        try:
            memory.observe_conversation(ut, (self._turn_text or "").strip())
        finally:
            # Consume the user text — don't observe the same turn twice if
            # multiple response.done events fire (e.g. a retry path).
            self._last_user_text = ""

    def _push_assistant_note(self, text: str) -> None:
        """Inject a proactive note into the live session from a BACKGROUND
        thread (e.g. auto-approve concluding) so the assistant relays it to
        the user without being prompted. Best-effort; no-op if not connected."""
        client = self._client
        if client is None or not getattr(client, "connected", False):
            return
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR, LiveApiState.CONNECTING):
            return
        with self._lock:
            if self._response_active:
                # Mid-response — queue; _drain_pending sends it on response.done.
                self._pending_notes.append(str(text))
                queued = True
            else:
                self._response_active = True
                queued = False
        if queued:
            if self._logger:
                self._logger.event("assistant_note_queued")
            return
        if not self._fire_response_create(prefix_notes=[str(text)]):
            with self._lock:
                self._response_active = False

    def stop(self) -> None:
        # Snapshot the things we need to tear down OUTSIDE the lock so the
        # WebSocket reader thread (which acquires the lock indirectly via
        # signal callbacks) can't deadlock against us. Setting the public
        # _state to OFF first also makes any in-flight callback bail
        # cleanly via their None-checks.
        with self._lock:
            if self._state == LiveApiState.OFF:
                return
            self._teardown_in_progress = True
            self._gui_alive = False
            self._response_active = False
            self._pending_notes = []
            self._response_requested = False
            self._multistep_active = False
            # Cancel the response-cycle watchdog so a pending timer
            # cannot fire against a torn-down session.
            self._cancel_response_watchdog()
            self._cancel_response_create_watchdog()
            self._cancel_deferred_request_watchdog()
            audio = self._audio
            client = self._client
            screen_thread = self._screen_thread
            executor = self._executor
            logger = self._logger
            player = self._audio_player
            self._audio = None
            self._client = None
            self._screen_thread = None
            self._executor = None
            self._registry = None
            self._screen = None
            self._command_router = None
            self._audio_player = None
            # Drop strong refs to the proactive engines so any late
            # sentinel callback that does manage to slip through the
            # unregister race below sees None and bails — also lets
            # the next session start() build fresh engines.
            self._anticipation_engine = None
            self._vision_observer = None
            self._standing_orders_evaluator = None
            # Don't drop the logger yet — we still want the stop events
            # written. Cleared once everything joined.
            self._screen_stop.set()
            self._screen_request.set()
            # Tell the project indexer to break out of its sleep loop
            # before the next interval. Daemon thread — we don't join.
            self._stop_indexer.set()
            # Same for the self-learning daemon: signal it to exit
            # before its next interval so the next session start can
            # spawn a fresh worker.
            self._stop_self_learner.set()
            if logger is not None:
                logger.event("session_stop_requested")
            self._set_state(LiveApiState.OFF, "Off")

        # Heavy/joining work outside the lock.
        # Unregister every proactive watcher this manager wired into
        # the process-global Sentinel. The Sentinel is a singleton
        # with its own daemon thread; without this, its ticks keep
        # invoking bound methods of `self` (calendar_briefing,
        # standing_orders, anticipation, vision_observer,
        # proactive_nudges, reliability_advisor, …) forever —
        # pinning the manager, accumulating notes in _pending_notes,
        # and racing into a torn-down Qt event dispatcher on
        # subsequent reopens. The daemon itself is left running
        # because the same instance is reused by the next session.
        try:
            from .sentinel import global_sentinel
            _sent = global_sentinel()
            # Snapshot + clear so a concurrent start() that begins
            # re-registering watchers can't have its new entries
            # yanked out from under it.
            with self._lock:
                _names = list(self._registered_watcher_names)
                self._registered_watcher_names = []
            _unregistered: list[str] = []
            for _name in _names:
                try:
                    if _sent.unregister(_name):
                        _unregistered.append(_name)
                except Exception:
                    pass
            if logger is not None and _unregistered:
                logger.event("sentinel_watchers_unregistered",
                             watchers=_unregistered)
        except Exception:
            if logger is not None:
                logger.warning("sentinel_teardown_exception")
        # Detach bus subscribers we attached in _ensure_audit_log so
        # the process-global InvocationBus does not accumulate dead
        # subscribers (with their captured `self` references) across
        # Iris reopens. Without this, every subsequent tool publish
        # fans out to ALL stale managers; each tries to emit Qt
        # signals on already-destroyed QObjects — the classic
        # PySide6 "freeing memory of an object that still has
        # receivers connected" crash signature.
        try:
            _audit = getattr(self, "_audit_log", None)
            if _audit is not None:
                try:
                    _audit.close()  # close() also unsubscribes
                except Exception:
                    pass
            for _sub_attr in ("_reauth_nudger", "_skill_consolidator"):
                _sub = getattr(self, _sub_attr, None)
                if _sub is not None and hasattr(_sub, "detach_from_bus"):
                    try:
                        _sub.detach_from_bus()
                    except Exception:
                        pass
            # Drop the strong refs so the lambda/notifier captures
            # release `self` and this manager can be GC'd.
            self._audit_log = None
            self._reauth_nudger = None
            self._skill_consolidator = None
        except Exception:
            if logger is not None:
                logger.warning("bus_subscriber_teardown_exception")
        # Tear down the audio output player so the Qt event loop can
        # drain its tail before subsequent work.
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass
        # Stop any background watchers (auto-approve thread, controllers) so
        # they don't keep polling — and moving the cursor — after the session
        # ends. The watcher is a daemon thread that otherwise outlives stop().
        if executor is not None:
            try:
                executor.close()
            except Exception:
                if logger is not None:
                    logger.warning("executor_close_exception")
        if audio is not None:
            try:
                audio.stop()
            except Exception:
                if logger is not None:
                    logger.warning("audio_stop_exception")
        if client is not None:
            try:
                client.stop()
                # Joining the reader thread guarantees no more callbacks
                # fire after we close the logger.
                client.join(timeout=2.0)
            except Exception:
                if logger is not None:
                    logger.warning("client_stop_exception")
        if screen_thread is not None:
            try:
                screen_thread.join(timeout=2.0)
            except Exception:
                pass

        with self._lock:
            if logger is not None:
                logger.event("session_stopped")
                logger.close()
            self._logger = None
            self._screen_stop.clear()
            self._screen_request.clear()
            self._pending_tool_calls.clear()

    # ---- Project-RAG indexer -------------------------------------------
    def _init_project_indexer(self) -> None:
        """Build the shared ProjectMemoryStore, register every existing
        known sibling project root into it, and spawn a single daemon
        thread that runs incremental ``index_project`` re-passes on a
        5-minute interval.

        The thread is daemonic, so it never blocks process exit; stop()
        sets ``_stop_indexer`` to break the sleep loop cleanly.
        """
        from .memory.project_memory import ProjectMemoryStore
        from .known_projects import existing_known_roots

        roots = existing_known_roots()
        store = ProjectMemoryStore(logger=self._logger)
        for entry in roots:
            try:
                store.register_project(entry["id"],
                                       entry["root"],
                                       label=entry.get("label"))
            except Exception as exc:
                if self._logger:
                    self._logger.exception("project_register_failed", exc,
                                           project_id=entry.get("id"))
        self._project_store = store
        # Fresh session: clear any stale stop signal from a previous run.
        self._stop_indexer.clear()
        ids = [e["id"] for e in roots]
        if not ids:
            # Nothing to index — skip the worker entirely.
            if self._logger:
                self._logger.event("project_indexer_no_roots")
            return

        def _worker() -> None:
            # Wait a few seconds before the first pass so the realtime
            # client gets to finish handshake without competing for
            # the embedder's first HTTP call.
            if self._stop_indexer.wait(timeout=5.0):
                return
            while not self._stop_indexer.is_set():
                for pid in ids:
                    if self._stop_indexer.is_set():
                        return
                    try:
                        self._project_store.index_project(pid)
                    except Exception as exc:
                        if self._logger:
                            self._logger.exception(
                                "project_index_failed", exc, project_id=pid)
                # 5-minute incremental re-pass; mtime-skip makes
                # this nearly free when nothing changed.
                if self._stop_indexer.wait(timeout=300.0):
                    return

        thread = threading.Thread(target=_worker,
                                  name="iris-project-indexer",
                                  daemon=True)
        thread.start()
        self._indexer_thread = thread
        if self._logger:
            self._logger.event("project_indexer_started",
                                project_ids=ids)

    # ---- Self-learning daemon -----------------------------------------
    def _init_self_learner(self) -> None:
        """Build the SelfLearner + a daemon thread that runs cycles on
        a 10-min interval (with a 30 s warm-up before the first cycle
        so the realtime handshake doesn't compete for the embedder).

        Same lifecycle pattern as ``_init_project_indexer``:
        ``_stop_self_learner`` is cleared here, set by ``stop()`` so
        the worker exits cleanly before the next interval.
        """
        from .learning import SelfLearner
        from .memory import MemoryManager
        from .cortex.tool_call_log import default_log_path

        # Construct a dedicated MemoryManager for the daemon. We don't
        # piggyback on the planner's manager because the planner is
        # lazily built (only after the user's first turn), and the
        # daemon should start mining as soon as the session opens.
        # SQLite is thread-safe across connections so two managers
        # writing to the same DB is fine.
        try:
            memory_manager = MemoryManager(
                logger=self._logger,
                project_store=self._project_store,
            )
        except Exception as exc:
            if self._logger:
                self._logger.exception("self_learner_memory_init_failed", exc)
            return

        # Activate action-classifier routing — ToolExecutor reads this
        # to call MemoryManager.set_fact() when a tool dispatch is
        # classified as a fact/preference at confidence >= 0.8. Without
        # this, classification still runs + logs but the writes are
        # skipped. One-line activation per impl note.
        try:
            if self._executor is not None:
                self._executor._memory = memory_manager
        except Exception:
            pass

        try:
            learner = SelfLearner(
                memory_manager=memory_manager,
                project_store=self._project_store,
                tool_call_log_path=default_log_path(),
                connector_registry=self._registry,
                logger=self._logger,
            )
        except Exception as exc:
            if self._logger:
                self._logger.exception("self_learner_construct_failed", exc)
            return
        self._self_learner = learner
        self._stop_self_learner.clear()

        # 10 minutes between cycles. The first cycle waits 30 s so the
        # realtime client gets to settle without competing for the
        # embedder's first HTTP call (mirrors the project indexer's
        # 5 s warm-up but longer since this daemon's first pass on a
        # fresh install can touch every registered project).
        cycle_interval = 600.0
        warmup = 30.0

        def _worker() -> None:
            if self._stop_self_learner.wait(timeout=warmup):
                return
            while not self._stop_self_learner.is_set():
                try:
                    report = learner.run_once()
                    if self._logger:
                        self._logger.event(
                            "self_learner_cycle_done",
                            sources=list(report.keys()),
                            total=sum(report.values()),
                        )
                except Exception as exc:
                    if self._logger:
                        self._logger.exception(
                            "self_learner_cycle_failed", exc)
                if self._stop_self_learner.wait(timeout=cycle_interval):
                    return

        thread = threading.Thread(target=_worker,
                                  name="iris-self-learner",
                                  daemon=True)
        thread.start()
        self._self_learner_thread = thread
        if self._logger:
            self._logger.event("self_learner_started",
                               interval_sec=cycle_interval,
                               warmup_sec=warmup)

    def self_learn_now(self) -> Dict[str, int]:
        """Manual / debug trigger that runs a single self-learn cycle
        synchronously. Returns the per-source fact counts so a CLI
        caller can confirm something happened (or didn't).

        Safe to call even when the background daemon is disabled —
        builds a transient SelfLearner if one wasn't constructed for
        this session. Returns ``{}`` when no MemoryManager can be
        built (e.g. TOUCHLESS_IRIS_MEMORY=0).
        """
        learner = self._self_learner
        if learner is None:
            # Try a one-shot construction so debug callers can use this
            # even when the daemon was disabled. Mirrors the eager
            # _init path but doesn't spawn a thread.
            if os.environ.get("TOUCHLESS_IRIS_MEMORY", "1") == "0":
                return {}
            try:
                from .learning import SelfLearner
                from .memory import MemoryManager
                from .cortex.tool_call_log import default_log_path
                memory_manager = MemoryManager(
                    logger=self._logger,
                    project_store=self._project_store,
                )
                learner = SelfLearner(
                    memory_manager=memory_manager,
                    project_store=self._project_store,
                    tool_call_log_path=default_log_path(),
                    connector_registry=self._registry,
                    logger=self._logger,
                )
            except Exception as exc:
                if self._logger:
                    self._logger.exception(
                        "self_learner_now_construct_failed", exc)
                return {}
        try:
            return learner.run_once()
        except Exception as exc:
            if self._logger:
                self._logger.exception("self_learner_now_failed", exc)
            return {}

    def request_screen_now(self) -> None:
        """Wake the screen worker so it captures and sends immediately."""
        self._screen_request.set()

    def send_user_text(self, text: str) -> bool:
        """Inject a typed user message into the current session.

        GUI-thread safe and NON-BLOCKING. This method performs only the
        fast preconditions (state check + transcript echo), then
        dispatches the actual routing/planner/realtime work to a
        background worker thread. All UI updates from the worker are
        marshaled back through Qt signals (queued connections), so the
        GUI thread never blocks on HTTP, time.sleep, or LLM planning.

        Routing flow (runs in worker thread `_run_send_user_text`):
          1. Echo the user's text in the chat (transcript_received,
             emitted here on the GUI thread for instant feedback).
          2. Try Layer 0 router — if it matches a known intent (open
             chrome, search X, play next song, ...), execute it
             instantly and DO NOT call the LLM.
          3. Try Layer 1 iris planner — deterministic classifier →
             connector, 0 model tokens.
          4. Otherwise forward to the backend (LLM agent loop).

        Returns True if the manager accepted the input (and dispatched
        the worker); False otherwise so the UI can show a hint.
        """
        text = (text or "").strip()
        if not text:
            return False
        client = self._client
        if client is None or not getattr(client, "connected", False):
            return False
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR, LiveApiState.CONNECTING):
            return False
        # Echo the user's typed message via the same signal voice
        # transcripts use, so the chat UI doesn't need a special path.
        # Done here on the GUI thread for instant echo before the worker
        # spins up (negligible cost — just a signal emit).
        self.transcript_received.emit(text)

        # Dispatch the slow path (router + planner + realtime send) to a
        # background thread so the GUI stays responsive. Wrapped in try/
        # except so a thread-spawn failure surfaces as a system bubble
        # instead of silently losing the user's input.
        try:
            worker = threading.Thread(
                target=self._run_send_user_text,
                args=(text,),
                name="iris-send-user-text",
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            if self._logger:
                self._logger.exception("send_user_text_dispatch_failed", exc)
            # Surface the failure to the chat as an assistant text so the
            # user knows the input wasn't dropped silently.
            dispatch_err = "Internal error — couldn't dispatch your message. Try again."
            try:
                self.assistant_text.emit(dispatch_err)
            except Exception:
                pass
            # Cortex viz: fire core→output pulse for this final reply.
            self._emit_reply_output_pulse(dispatch_err)
            # Local error reply — speak it too so voice-only users notice.
            try:
                self._speak_text(dispatch_err)
            except Exception:
                pass
            return False
        return True

    # NOTE: _run_send_user_text runs on a BACKGROUND THREAD. It must not
    # touch any Qt widget directly — only emit signals (which Qt auto-
    # marshals to the GUI thread via queued connection) and call
    # thread-safe client methods (RealtimeClient.send_* are guarded by
    # _send_lock; cortex_emit is best-effort thread-safe; the planner
    # and router internals use only stdlib + their own state).
    def _run_send_user_text(self, text: str) -> None:
        """Worker-thread implementation of send_user_text — runs the
        slow path (router, iris planner, LLM agent dispatch) off the
        GUI thread. All UI updates go through Qt signals."""
        # Immediately flip state to THINKING so the cortex core turns
        # orange the moment Send is clicked — don't wait for the
        # planner to decide. Final state transition (LISTENING /
        # SPEAKING) happens at the end of each layer's reply.
        try:
            self._set_state(LiveApiState.THINKING, "Thinking…")
        except Exception:
            pass
        try:
            self._send_user_text_pipeline(text)
        except Exception as exc:
            # Last-resort guard — never let a worker exception crash the
            # background thread silently. Log it and notify the user.
            if self._logger:
                self._logger.exception("send_user_text_worker_failed", exc)
            err_msg = "Internal error processing your message — please try again."
            try:
                self.assistant_text.emit(err_msg)
            except Exception:
                pass
            # Cortex viz: fire core→output pulse for this final reply.
            self._emit_reply_output_pulse(err_msg)
            # Local error reply — speak it too so voice-only users notice.
            try:
                self._speak_text(err_msg)
            except Exception:
                pass
            try:
                self._set_state(LiveApiState.LISTENING, "Ready (type a command)")
            except Exception:
                pass

    def _send_user_text_pipeline(self, text: str) -> None:
        """The actual router → planner → LLM pipeline. Extracted so
        _run_send_user_text can wrap it in a single try/except. Runs on
        a worker thread."""
        client = self._client
        # Client may have been torn down between dispatch and worker
        # wake-up; bail quietly in that case.
        if client is None or not getattr(client, "connected", False):
            return

        # ---- Layer 0: deterministic router ----
        router = self._command_router
        if router is not None:
            try:
                routed = router.try_route(text)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("router_unhandled", exc)
                routed = None
            if routed is not None and routed.matched and routed.success:
                # Handled locally by Touchless (free) — show it via the same
                # tool_event plumbing the LLM tool calls use, then close the turn.
                action_label = routed.intent_action or "router"
                # Cortex viz: light up the FULL path for local-router hits.
                # core → cap-tools → tool-<action> → core, STAGGERED so
                # the user sees actual traveling neurons at each hop
                # instead of a single overlapping flash.
                from PySide6.QtCore import QTimer as _QT_router
                from PySide6.QtWidgets import QApplication as _QApp_router
                def _router_path_pulses(label=action_label):
                    try:
                        cortex_emit.edge_pulse("core", "cap-tools",
                                                color="orange", duration_ms=280)
                    except Exception:
                        pass
                    def _step2():
                        try:
                            cortex_emit.edge_pulse("cap-tools", f"tool-{label}",
                                                    color="orange", duration_ms=280)
                        except Exception:
                            pass
                    def _step3():
                        try:
                            cortex_emit.node_activity(f"tool-{label}",
                                                       intensity=1.0, duration_ms=400)
                        except Exception:
                            pass
                    def _step4():
                        try:
                            cortex_emit.edge_pulse(f"tool-{label}", "core",
                                                    color="orange", duration_ms=280)
                        except Exception:
                            pass
                    try:
                        _app = _QApp_router.instance()
                        if _app is not None and getattr(
                                self, "_gui_alive", True):
                            _QT_router.singleShot(300, _app, _step2)
                            _QT_router.singleShot(600, _app, _step3)
                            _QT_router.singleShot(1000, _app, _step4)
                        else:
                            _step2()
                            _step3()
                            _step4()
                    except Exception:
                        pass
                # _worker runs on a planner worker thread; the
                # QTimer.singleShot calls inside _router_path_pulses
                # need the GUI thread's event dispatcher. Marshal the
                # whole helper onto the manager's owning thread via a
                # queued invokeMethod hop so timers are constructed
                # under a live dispatcher; if torn down, no-op.
                try:
                    _own_thread = self.thread()
                except Exception:
                    _own_thread = None
                if (_own_thread is not None
                        and QThread.currentThread() is not _own_thread
                        and getattr(self, "_gui_alive", True)):
                    try:
                        QMetaObject.invokeMethod(
                            self, "_run_router_path_pulses_on_gui_thread",
                            Qt.QueuedConnection,
                            Q_ARG(str, action_label),
                        )
                    except Exception:
                        pass
                else:
                    _router_path_pulses()
                self.tool_event.emit("called", {"name": f"router/{action_label}", "info": routed.message, "source": "touchless"})
                self.tool_event.emit("completed", {"name": f"router/{action_label}", "status": "ok", "source": "touchless"})
                if self._logger:
                    self._logger.event("routing_decision", **cost_policy.decision_record(
                        raw=text, tool=action_label, source="touchless", status="ok"))
                spoken = routed.message or "Done."
                self.assistant_text.emit(spoken)
                # Local handler — realtime won't speak this. Synthesize TTS
                # so voice_output=True users actually hear the response.
                self._speak_text(spoken)
                self._set_state(LiveApiState.LISTENING, "Ready (type a command)")
                # Cortex viz: fire core→output pulse LAST so the
                # 'speaking' core state isn't immediately overwritten
                # by the LISTENING set_state above. Pass the reply text
                # so purple-speaking duration tracks how long it'll take
                # to read/speak (longer reply = longer purple window).
                self._emit_reply_output_pulse(spoken)
                return
            # Matched but FAILED locally (e.g. an ambiguous app like 'paint' →
            # Paint vs Paint 3D, or 'pause' with nothing playing): DON'T show the
            # local clarification — the user can't answer it because that prompt
            # isn't in the model's context, so their reply hits the LLM bare and
            # derails. Fall through to the LLM, whose richer tools resolve it
            # (open_app 'paint' → classic Paint) and continue the task.
            if routed is not None and routed.matched and not routed.success and self._logger:
                self._logger.event(
                    "router_failed_fallback_llm",
                    action=getattr(routed, "intent_action", ""),
                    info=getattr(routed, "message", ""),
                )

        # ---- Layer 1: iris planner (deterministic classifier → connector,
        # 0 model tokens). Phase 1 of the JARVIS-level decision engine — see
        # docs/IRIS_PLANNER_DESIGN.md. Default on; disable with
        # TOUCHLESS_IRIS_PLANNER=0 to force the old LLM path for debugging.
        if os.environ.get("TOUCHLESS_IRIS_PLANNER", "1") != "0" and self._registry is not None:
            # Watchdog-fired sentinel is hoisted out of the try so the
            # post-except check can read it even when planner init /
            # memory summary raised before the timer was armed.
            _planner_watchdog_fired = {"v": False}
            try:
                if self._iris_planner is None:
                    from .planner.orchestrator import IrisPlanner
                    # Optional memory layer — wired when not explicitly
                    # disabled (TOUCHLESS_IRIS_MEMORY=0) so the planner can
                    # recall prior turns + facts. Failures are non-fatal:
                    # the planner just runs without memory.
                    memory = None
                    if os.environ.get("TOUCHLESS_IRIS_MEMORY", "1") != "0":
                        try:
                            from .memory import MemoryManager
                            memory = MemoryManager(logger=self._logger,
                                                    project_store=self._project_store)
                        except Exception as exc:
                            if self._logger:
                                self._logger.exception("memory_init_failed", exc)
                            memory = None
                    self._iris_planner = IrisPlanner(self._registry, self._logger,
                                                     confirm=self._confirm_callback,
                                                     memory=memory)
                # Now that the planner + memory are wired, send the one-time
                # session-start memory summary so realtime has the user's
                # known facts in context for ANY downstream realtime turn
                # this session. Idempotent: _memory_summary_sent guards
                # against re-firing on subsequent user turns.
                self._maybe_send_memory_summary()
                # ---- planner-handle watchdog ----
                # try_handle() runs the classifier + LLM planner +
                # executor + orchestrator (which itself does SQLite +
                # cot.db writes and a session-note WS send on the
                # return path). None of those hops have their own
                # timeout, and if any of them blocks the UI pill is
                # pinned to THINKING forever (see the 12:09:44 →
                # 12:12:57 stall after a successful sheets_create).
                # The existing WS watchdogs (_response_watchdog /
                # _deferred_request_watchdog / _response_create_
                # watchdog) only guard the realtime response cycle
                # and never arm on the planner path. Mirror them
                # here so this path recovers too.
                try:
                    _planner_watchdog_sec = float(
                        os.environ.get("TOUCHLESS_PLANNER_HANDLE_TIMEOUT",
                                       "45"))
                except Exception:
                    _planner_watchdog_sec = 45.0

                def _planner_watchdog_fire() -> None:
                    _planner_watchdog_fired["v"] = True
                    if self._logger:
                        try:
                            self._logger.event(
                                "planner_handle_watchdog_timeout",
                                sec=_planner_watchdog_sec,
                                text=(text or "")[:60])
                        except Exception:
                            pass
                    try:
                        self.assistant_text.emit(
                            "That one took longer than expected — "
                            "give it another shot.")
                    except Exception:
                        pass
                    try:
                        self._set_state(LiveApiState.LISTENING,
                                         "Ready (type a command)")
                    except Exception:
                        pass

                _planner_watchdog: Optional[threading.Timer]
                try:
                    _planner_watchdog = threading.Timer(
                        _planner_watchdog_sec, _planner_watchdog_fire)
                    _planner_watchdog.daemon = True
                    _planner_watchdog.start()
                except Exception:
                    _planner_watchdog = None
                try:
                    handled = self._iris_planner.try_handle(text)
                finally:
                    if _planner_watchdog is not None:
                        try:
                            _planner_watchdog.cancel()
                        except Exception:
                            pass
            except Exception as exc:
                if self._logger:
                    self._logger.exception("iris_planner_unhandled", exc)
                handled = None
            # If the watchdog fired we already emitted a friendly reply
            # and dropped back to LISTENING. Any late return from
            # try_handle would double-emit and confuse the user, so
            # swallow it here.
            if _planner_watchdog_fired.get("v"):
                return
            if handled is not None:
                # Unified shape: Phase 1 (classifier) returns a 1-step list,
                # Phase 2 (LLM plan + Executor) returns N steps. Either way we
                # iterate steps/results so the UI gets a per-step badge trail.
                steps = handled.get("steps") or []
                results = handled.get("results") or []
                for step, sr in zip(steps, results):
                    out = sr.output if isinstance(sr.output, dict) else {}
                    source = ("connector"
                              if self._registry.handles_connector(step.tool)
                              else "touchless")
                    pid = f"planner/{step.tool}/{getattr(step, 'id', 0)}"
                    self.tool_event.emit("called", {"name": step.tool, "call_id": pid, "source": source})
                    self.tool_event.emit("completed", {
                        "name": step.tool, "call_id": pid,
                        "status": out.get("status", sr.status or "ok"),
                        "source": source,
                        "error": out.get("error") or "",
                    })
                    if self._logger:
                        self._logger.event("routing_decision", **cost_policy.decision_record(
                            raw=text, tool=step.tool, source=source,
                            status=str(out.get("status", sr.status or ""))))
                planner_msg = handled["message"]
                self.assistant_text.emit(planner_msg)
                # Local planner — realtime won't speak this. Synthesize TTS
                # so voice_output=True users actually hear the response.
                self._speak_text(planner_msg)
                # Cortex viz: pass reply text so purple duration tracks
                # how long the reply takes to read/speak.
                self._emit_reply_output_pulse(planner_msg)
                # Realtime-session sync: tell the model what just happened so
                # follow-ups like "send him a thank you too" can resolve the
                # prior turn. Best-effort — never fail the user-visible reply.
                try:
                    client = self._client
                    if client is not None and hasattr(client, "send_session_note"):
                        note = self._build_session_note(text, steps, results,
                                                        handled.get("message") or "")
                        if note:
                            client.send_session_note(note)
                except Exception as exc:  # pragma: no cover - defensive
                    if self._logger:
                        self._logger.exception("session_note_send_failed", exc)
                self._set_state(LiveApiState.LISTENING, "Ready (type a command)")
                return

        # ---- Layer 2: LLM agent (planner didn't classify either) ----
        # Reset task state for this new user turn.
        self._multistep_active = False
        self._in_queue = False
        self._planning = False
        self._task_queue = []
        self._nudge_count = 0
        # Multi-action request → first ask the model to PLAN it into an ordered
        # task list (accurate decomposition), then run each task one at a time
        # with validation. Single actions go straight through.
        if self._looks_multistep(text):
            self._planning = True
            self._orig_command = text
            plan_prompt = (
                "PLAN ONLY — do not call any tools yet. Break my request into an "
                "ordered JSON array of the FEWEST coarse steps (don't over-"
                "split). Keep an app-open together with the action done IN that "
                "app as ONE step, and keep a search together with summarizing it "
                "as ONE step: e.g. 'Open Chrome and search the latest AI news, "
                "summarize it' is ONE step ('search the latest AI news and "
                "summarize it'); 'open Spotify and play lofi' is ONE step; "
                "'open project X in VS Code and open a Claude tab' is ONE step. "
                "Each string is a short imperative. Reply with ONLY the JSON "
                "array, nothing else. Request: " + text
            )
            ok = bool(client.send_text_message(plan_prompt))
            if ok:
                self._request_model_response()
            return
        # Refresh the model's clock before each turn — but ONLY when
        # the turn is plausibly time-sensitive. Unconditional injection
        # was costing input tokens every turn (the note is ~250 chars,
        # billed per turn server-side) for prompts that don't need it.
        # Gating on a tiny regex keeps "what time is it"/"how long
        # until" accurate without per-turn bloat on e.g. "add a meeting
        # at 3pm" (which already names the time the model needs).
        try:
            if self._looks_time_sensitive(text):
                import datetime as _dt
                _now = _dt.datetime.now().astimezone()
                time_note = (
                    f"Time check (use for any 'what time is it' / 'when' / "
                    f"'how long until' question this turn — do NOT call a "
                    f"tool to check the clock): "
                    f"{_now.strftime('%A, %B %d, %Y at %I:%M %p %Z').strip()} "
                    f"(ISO: {_now.isoformat(timespec='seconds')})."
                )
                client.send_session_note(time_note)
        except Exception:
            pass
        # Per-turn calendar-rules hook: the verbose CALENDAR RULES block
        # used to live in SYSTEM_INSTRUCTIONS and was paid for on EVERY
        # turn even though most turns aren't calendar-related. Inject it
        # only when the turn actually mentions a calendar verb.
        try:
            if self._looks_like_calendar(text):
                self._send_calendar_context_note(client)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("calendar_context_inject_failed", exc)
        # Per-turn recall hook: when the user asks a recall-shaped
        # question ("remind me what…", "what did we discuss…",
        # "last time", "earlier"), the planner isn't invoked, so the
        # realtime model has no visibility into episodic memory. Pull
        # episodes here and inject them — OR explicitly inject an
        # "empty context" signal — so the model can ground its answer
        # or admit ignorance instead of fabricating. See HARD RULE —
        # NEVER FABRICATE PERSONAL / RECALL CONTEXT in
        # SYSTEM_INSTRUCTIONS.
        try:
            if self._looks_like_recall(text):
                self._send_recall_context_note(client, text)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("recall_context_inject_failed", exc)
        ok = bool(client.send_text_message(text))
        if ok:
            # Track for fact-extraction on response.done. Only capture turns
            # that ACTUALLY went to realtime — planner-handled requests
            # already record themselves via the orchestrator's _record_turn.
            self._last_user_text = text
            self._last_input_was_voice = False
            self._record_convo_turn("user", text)
            self._request_model_response()
        return

    # ---- recall-context injection ----------------------------------------
    # Recall-shaped phrases that should trigger an episodic memory lookup
    # before the realtime model gets the user's text. Pattern matched at
    # word boundaries against the lowercased turn.
    # Artifact noun list used by the retrieval-shaped patterns below.
    # Kept narrow so phrases like "show me the weather" or "show me what's
    # playing" do NOT trigger recall — only nouns naming a generated
    # textual artifact (email/poem/draft/code/...) should match.
    _ARTIFACT_NOUN_GROUP = (
        r"(?:email|emails|message|messages|reply|replies|response|responses|"
        r"draft|drafts|note|notes|memo|memos|letter|letters|"
        r"poem|poems|haiku|haikus|song|songs|lyric|lyrics|"
        r"essay|essays|article|articles|post|posts|tweet|tweets|"
        r"caption|captions|speech|speeches|toast|toasts|"
        r"story|stories|script|scripts|"
        r"summary|summaries|outline|outlines|agenda|agendas|"
        r"recipe|recipes|paragraph|paragraphs|sentence|sentences|"
        r"code|snippet|snippets|function|functions|"
        r"query|queries|regex|sql|text)"
    )
    _RECALL_TRIGGER_PATTERNS = (
        r"\bremind me\b",
        r"\bwhat did (?:we|i|you) (?:say|discuss|talk about|mention|do)\b",
        r"\bwhat (?:was|were) (?:that|those|the) (?:thing|things)\b",
        r"\bwhat was the (?:thing|leak|issue|problem|deal|story)\b",
        r"\blast time\b",
        r"\bearlier (?:we|i|you|today|this week)\b",
        r"\bwe (?:talked|discussed|spoke|chatted) about\b",
        r"\bdid (?:we|i) (?:talk|discuss|mention|cover)\b",
        r"\b(?:do you|you) remember\b",
        r"\brecall\b",
        r"\bwho (?:is|was) [A-Z]?\w+\b",
        # Artifact-retrieval verbs ("show me / pull up / read me / find that
        # / where's that / read back / what did you write"). Noun-gated to
        # the artifact list so non-artifact uses ("show me the weather",
        # "show me what's playing") do NOT match.
        (r"\b(?:show|read|pull up|pull|bring up|find|get|grab|open|display)"
         r"\s+(?:me|us)?\s*(?:that|the|my|our)?\s*"
         + _ARTIFACT_NOUN_GROUP + r"\b"),
        r"\bread (?:it|that|them|the " + _ARTIFACT_NOUN_GROUP + r") back\b",
        (r"\bwhere(?:'s| is| are)?\s+(?:that|the|my)\s+"
         + _ARTIFACT_NOUN_GROUP + r"\b"),
        (r"\bwhat did you (?:write|draft|compose|make|generate|produce|"
         r"come up with)\b"),
        (r"\bthe " + _ARTIFACT_NOUN_GROUP +
         r"\s+(?:you|we)\s+(?:wrote|drafted|composed|made|generated)\b"),
    )

    def _looks_like_recall(self, text: str) -> bool:
        """True when the user's turn looks like a recall/recap question
        that needs episodic memory to answer faithfully."""
        if not text:
            return False
        low = text.strip().lower()
        if len(low) < 5:
            return False
        for pat in self._RECALL_TRIGGER_PATTERNS:
            if re.search(pat, low):
                return True
        return False

    # Per-turn time_note gating — only inject the clock when the turn
    # plausibly needs it (avoid paying ~250 chars/turn server-side for
    # turns that already name an explicit time, like "add a meeting at
    # 3pm" or have no time component at all).
    _TIME_SENSITIVE_PATTERNS = (
        r"\bwhat (?:time|day|date)\b",
        r"\bwhat'?s the (?:time|date|day)\b",
        r"\b(?:when|how long)\b",
        r"\b(?:today|tonight|tomorrow|yesterday)\b",
        r"\b(?:this|next|last)\s+(?:morning|afternoon|evening|night|"
        r"week|weekend|month|year|monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)\b",
        r"\b(?:in|after)\s+(?:an?\s+|\d+\s+)?(?:hour|minute|day|week)s?\b",
        r"\b(?:o'?clock|noon|midnight)\b",
    )

    def _looks_time_sensitive(self, text: str) -> bool:
        """True when the turn references relative time and might need
        the current clock injected."""
        if not text:
            return False
        low = text.strip().lower()
        if len(low) < 3:
            return False
        for pat in self._TIME_SENSITIVE_PATTERNS:
            if re.search(pat, low):
                return True
        return False

    # Per-turn calendar-rules gating — the CALENDAR RULES block used to
    # live in SYSTEM_INSTRUCTIONS (~1.8K chars). Now injected only when
    # the turn actually mentions a calendar verb.
    _CALENDAR_TRIGGER_PATTERNS = (
        r"\bcalendar\b",
        r"\bmeeting\b",
        r"\bappointment\b",
        r"\bevent\b",
        r"\bschedule\b",
        r"\b(?:reschedule|rescheduled)\b",
        r"\b(?:remind me|reminder)\s+(?:to|at|on|about|that|of)\b",
        r"\b(?:add|book|create|set up|put|stick)\s+(?:a |an )?"
        r"(?:meeting|appointment|event|reminder|invite|invitation)\b",
        r"\b(?:cancel|delete|move|reschedule)\s+(?:my |the |a |an )?"
        r"(?:meeting|appointment|event)\b",
        r"\b(?:what'?s on|do i have|anything on|what do i have)\b.*"
        r"\b(?:calendar|schedule|today|tomorrow|this week)\b",
    )

    def _looks_like_calendar(self, text: str) -> bool:
        """True when the turn looks like a calendar create/list/modify
        request, gating the per-turn CALENDAR-context injection."""
        if not text:
            return False
        low = text.strip().lower()
        if len(low) < 5:
            return False
        for pat in self._CALENDAR_TRIGGER_PATTERNS:
            if re.search(pat, low):
                return True
        return False

    # Calendar tool-priority + override grammar — injected per-turn
    # when _looks_like_calendar matches. Replaces the bulky CALENDAR
    # RULES block that previously rode in SYSTEM_INSTRUCTIONS on every
    # turn regardless of intent.
    _CALENDAR_CONTEXT_NOTE = (
        "Calendar context for this turn:\n"
        "CREATE — pick the tool that lands the event in a calendar the "
        "user actually views:\n"
        "  1) EXPLICIT OVERRIDE WINS: 'in google'/'gcal' → "
        "calendar_create_event. 'in outlook' → outlook_com_create_event. "
        "'microsoft 365'/'work calendar' → ms_calendar_create.\n"
        "  2) Otherwise default by toolset, in priority order:\n"
        "     (a) calendar_create_event (Google) — PREFERRED when "
        "available (syncs to phone + browser + New Outlook view).\n"
        "     (b) ms_calendar_create — when (a) is missing.\n"
        "     (c) outlook_com_create_event — last resort. On New "
        "Outlook this may fall back to an .ics file; warn the user "
        "it may not sync to their phone.\n"
        "  3) If a tool returns requires_user_confirm=true (.ics "
        "fallback), tell the user 'Outlook just opened a confirmation "
        "— hit Save & Close to add it' instead of claiming success.\n"
        "  4) ALWAYS mention WHICH calendar in your reply (use the "
        "tool result's `calendar` field).\n"
        "LIST/CHECK — when no platform is named, query BOTH "
        "ms_calendar_list AND calendar_list_events (Google), merge by "
        "start time, mention sources only if events come from both. "
        "If the user names a specific calendar, query only that one."
    )

    def _send_calendar_context_note(self, client: Any) -> None:
        """Inject the calendar tool-priority/override grammar as a
        session note so the realtime model only pays for it on turns
        that actually involve a calendar verb."""
        if not hasattr(client, "send_session_note"):
            return
        try:
            client.send_session_note(self._CALENDAR_CONTEXT_NOTE)
            if self._logger:
                self._logger.event("calendar_context_injected",
                                   chars=len(self._CALENDAR_CONTEXT_NOTE))
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("calendar_context_send_failed", exc)

    def _send_recall_context_note(self, client: Any, text: str) -> None:
        """Pull episodes via MemoryManager.recall and inject them — or
        an explicit 'NONE' signal — as a session note so the realtime
        model can ground a recall answer (or admit it has nothing) per
        the HARD RULE on personal/recall fabrication."""
        planner = self._iris_planner
        memory = getattr(planner, "_memory", None) if planner is not None else None
        if memory is None or not hasattr(memory, "recall"):
            return
        if not hasattr(client, "send_session_note"):
            return
        try:
            result = memory.recall(text, k=3) or {}
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("recall_context_lookup_failed", exc)
            return
        episodes = result.get("episodes") or []
        context = (result.get("context") or "").strip()
        # Weak-similarity hedge: even with episodes, a top sim near the
        # floor (_RECALL_MIN_SIM ~ 0.1) usually means tangential matches,
        # not a real memory of the topic the user is asking about. Tell
        # the model so it hedges accordingly.
        top_sim = 0.0
        if episodes:
            try:
                top_sim = max(float(e.get("sim", 0.0) or 0.0) for e in episodes)
            except (TypeError, ValueError):
                top_sim = 0.0
        if not episodes or not context:
            note = ("Recall context for this turn: NONE. You have NO "
                    "episodic notes matching the user's recall question. "
                    "Per the HARD RULE on personal/recall fabrication, do "
                    "NOT invent details, scenarios, or specifics. Ask the "
                    "user to remind you (e.g. 'I don't have anything on "
                    "that in my notes — can you walk me through it "
                    "again?').")
        else:
            weak = top_sim < 0.25
            hedge = ""
            if weak:
                hedge = (f" Recall is WEAK (top similarity {top_sim:.2f} "
                         f"— likely tangential, not a real match). Treat "
                         f"this as if context were empty unless the "
                         f"snippets clearly cover the user's question. "
                         f"When in doubt, ask the user to remind you.")
            note = ("Recall context for this turn — use ONLY these "
                    "snippets when answering the user's recall question; "
                    "if they don't cover what was asked, say so plainly "
                    "and ask the user to remind you (do NOT invent "
                    "details to fill the gap):\n" + context + hedge)
        # Artifact lookup: for retrieval-shaped turns ("show me the email
        # we drafted", "pull up that poem"), the user wants the literal
        # draft body — not a 120-char truncated outcome from the episodic
        # context. Pull matching llm_draft semantic rows so the model can
        # read them back verbatim. Bounded to drafts[:2] / 4000 chars each
        # so the session note stays sane.
        drafts: List[Any] = []
        if hasattr(memory, "find_llm_drafts_matching"):
            try:
                drafts = memory.find_llm_drafts_matching(text, k=3) or []
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("recall_draft_lookup_failed", exc)
                drafts = []
        if drafts:
            draft_lines = [
                "",
                "",
                "Draft artifacts matching this request — read back "
                "LITERALLY, do NOT paraphrase or invent a story about "
                "whether it was finished:",
            ]
            for d in drafts[:2]:
                key = getattr(d, "key", "") or ""
                val = getattr(d, "value", "") or ""
                draft_lines.append(f"[{key}]")
                draft_lines.append(val)
            note = note + "\n".join(draft_lines)
        try:
            client.send_session_note(note)
            if self._logger:
                self._logger.event("recall_context_injected",
                                   episodes=len(episodes),
                                   drafts=len(drafts),
                                   top_sim=round(top_sim, 3),
                                   chars=len(note))
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("recall_context_send_failed", exc)

    def _record_convo_turn(self, role: str, text: str) -> None:
        """Append a turn to the rolling conversation buffer used by the
        Jarvis prose renderer for natural follow-up references. Trims
        to the most recent _CONVO_BUFFER_MAX entries. Safe to call from
        any thread — list append is atomic in CPython.

        Phase-3: ALSO records into the global SessionBuffer so the
        Tier-2 planner gets the same conversation context on its
        next turn. Without this, voice/realtime-handled turns were
        invisible to the planner's recall block."""
        if not text:
            return
        snip = text.strip().replace("\r", " ").replace("\n", " ")
        # 2000-char cap (raised from 280) so a chat-generated email /
        # poem / code block survives the in-session 'show me what you
        # just drafted' before durable memory has indexed it. The
        # rolling buffer is itself length-capped by _CONVO_BUFFER_MAX
        # entries downstream so this can't grow unbounded.
        if len(snip) > 2000:
            snip = snip[:1997] + "..."
        try:
            self._convo_buffer.append(f"{role}: {snip}")
            if len(self._convo_buffer) > self._CONVO_BUFFER_MAX:
                # Trim from the front, keep the tail.
                del self._convo_buffer[:-self._CONVO_BUFFER_MAX]
        except Exception:
            pass
        # Mirror into the SessionBuffer so Tier-2 planner recalls
        # see voice/realtime turns too. Incognito-honored inside.
        try:
            from .session_buffer import global_session_buffer
            sb = global_session_buffer()
            if role == "user":
                sb.add_user(text)
            else:
                sb.add_assistant(text)
        except Exception:
            pass

    def _convo_context(self) -> str:
        """Render the convo buffer as a small block the renderer can
        scan for natural follow-up references."""
        try:
            return "\n".join(self._convo_buffer[-self._CONVO_BUFFER_MAX:])
        except Exception:
            return ""

    def _parse_plan(self, text: str) -> list:
        """Extract the JSON task array the model returned during planning."""
        try:
            m = re.search(r"\[.*\]", text or "", re.DOTALL)
            if not m:
                return []
            arr = json.loads(m.group(0))
            return [str(s).strip() for s in arr if str(s).strip()]
        except Exception:
            return []

    def _start_task_queue(self, tasks: list) -> bool:
        """Begin running an ordered task list one at a time. Returns True if a
        queue was started (≥2 tasks)."""
        if len(tasks) < 2:
            return False
        self._task_total = len(tasks)
        self._task_queue = tasks[1:]
        self._in_queue = True
        if self._logger:
            self._logger.event("task_queue_started", total=len(tasks), tasks=tasks)
        self._run_task(tasks[0])
        return True

    def _run_task(self, task: str) -> None:
        """Run ONE queued sub-task. FIRST try the local Touchless router (free,
        no LLM tokens) — for simple commands like 'play X on spotify', 'open
        chrome', 'next song'. Only if the router doesn't handle it does the task
        go to the LLM."""
        client = self._client
        if client is None or not getattr(client, "connected", False):
            self._in_queue = False
            return
        router = self._command_router
        if router is not None:
            try:
                routed = router.try_route(task)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("task_route_failed", exc)
                routed = None
            if routed is not None and routed.matched and routed.success:
                # Handled locally by Touchless — show it and advance (no LLM).
                label = (routed.intent_action or "command").lower()
                self.tool_event.emit("called", {"name": f"touchless/{label}", "info": routed.message})
                self.tool_event.emit("completed", {"name": f"touchless/{label}", "status": "ok"})
                self.assistant_message_break.emit()  # fresh bubble for this reply
                task_msg = routed.message or "Done."
                self.assistant_text.emit(task_msg)
                # Cortex viz: pass reply text so purple duration tracks
                # how long the reply takes to read/speak.
                self._emit_reply_output_pulse(task_msg)
                # Local task handler — realtime won't speak this. Synthesize
                # so the user hears progress on a multi-step plan.
                self._speak_text(task_msg)
                if self._logger:
                    self._logger.event("task_local", task=task, success=True)
                self._feed_next_task()
                return
            if routed is not None and routed.matched and not routed.success:
                # Touchless recognized it but FAILED to do it (e.g. Spotify
                # couldn't start the playlist) — fall back to the LLM to try
                # another way instead of silently skipping.
                if self._logger:
                    self._logger.event("task_local_failed_fallback_llm", task=task)
        # Not a local command (or local attempt failed) — hand it to the LLM.
        self._send_task(task, client)

    _ACTION_VERB_STARTS = (
        "open", "play", "search", "look", "move", "close", "minimize", "maximize",
        "restore", "focus", "start", "launch", "run", "create", "make", "build",
        "find", "show", "put", "summarize", "summarise", "write", "pull", "set",
        "go", "navigate", "drag",
        # Extra verbs that previously made 2-verb prompts like 'write a haiku
        # and add a task' miss the multistep heuristic — realtime then got
        # the raw multi-action text with no decomposition guidance and only
        # executed ONE of the actions.
        "add", "remove", "delete", "remind", "schedule",
        "send", "email", "text", "message",
        "tell", "give", "compose", "draft", "translate", "rephrase",
        "generate", "produce", "save", "upload", "download",
        "ask", "fetch", "read", "list", "check", "toggle", "mute", "unmute",
        "type", "click", "press", "scroll", "copy", "paste", "say", "answer",
    )

    def _looks_multistep(self, text: str) -> bool:
        low = (text or "").lower()
        seq = low.count(" then ") + low.count(" and ") + low.count(", ")
        verbs = sum(1 for v in self._ACTION_VERB_STARTS if (v + " ") in low)
        # Lowered from `seq>=2 and verbs>=2` to also catch single-and 2-verb
        # prompts ('write me a haiku and add a task to ...'). The LLM
        # planner can still produce a 1-step plan when it decides the verbs
        # belong to one task, so a false multistep trigger is cheap; a
        # false single-step trigger DROPS the second action (bug C20/C19).
        return ((seq >= 2 and verbs >= 2)
                or verbs >= 3
                or (verbs >= 2 and (" and " in low or " then " in low
                                    or "; " in low))
                or len(text) > 160)

    def _split_into_tasks(self, text: str) -> list:
        """Split a multi-action command into atomic sub-tasks. A new task starts
        only at a part that BEGINS with an action verb; parts without one are
        appended to the current task (so 'open Documents, Downloads, Pictures'
        stays one task, while 'open X, play Y, open Z' becomes three)."""
        if not text or not self._looks_multistep(text):
            return []
        import re as _re
        parts = _re.split(r"\s*,\s+and\s+|\s*,\s*|\s+and\s+then\s+|\s+then\s+|\s+and\s+|\s*;\s*", text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        tasks: list[str] = []
        for p in parts:
            first = p.lower().split()[0] if p.split() else ""
            starts_action = first in self._ACTION_VERB_STARTS
            if not tasks or starts_action:
                tasks.append(p)
            else:
                tasks[-1] = tasks[-1] + ", " + p  # intra-clause comma — merge back
        return tasks if len(tasks) >= 2 else []

    def _send_task(self, task: str, client) -> bool:
        """Send ONE sub-task as its own focused prompt and request a response."""
        n_done = self._task_total - len(self._task_queue)
        msg = (
            f"[Step {n_done} of {self._task_total}] Do ONLY this, completely, "
            f"using the right tool(s) and confirming it actually worked: {task}. "
            f"Then give a ONE-line confirmation and stop — do not do other steps."
        )
        ok = bool(client.send_text_message(msg))
        if ok:
            self._request_model_response()
        return ok

    def _feed_next_task(self) -> None:
        """Advance the task queue: send the next sub-task, or finish."""
        client = self._client
        if client is None or not getattr(client, "connected", False):
            self._in_queue = False
            return
        if not self._task_queue:
            self._in_queue = False
            if self._logger:
                self._logger.event("task_queue_done", total=self._task_total)
            return
        next_task = self._task_queue.pop(0)
        time.sleep(0.2)  # tiny pace; keep the queue fast (rate-limit pauses
        #                  are handled separately by the retry-after logic)
        self._run_task(next_task)

    # ---- websocket callbacks ----

    def _on_ws_connected(self) -> None:
        if self._logger:
            self._logger.event("session_ws_connected", text_only=self._text_only)
        # Start mic + (cloud-only) screen worker once the backend is up.
        # Local backend has no vision in Phase 1, so the screen worker
        # would just burn CPU/memory capturing JPEGs nobody reads.
        if self._audio is not None:
            self._audio.start()
        if (self._config.backend or "cloud").strip().lower() == "cloud":
            self._start_screen_worker()
        if self._text_only:
            self._set_state(LiveApiState.LISTENING, "Ready (type a command)")
        else:
            self._set_state(LiveApiState.LISTENING, "Listening")

    def _on_ws_closed(self, reason: Optional[str]) -> None:
        if self._logger:
            self._logger.event("session_ws_closed", reason=reason)
        # Disarm the response-cycle watchdog so it can't fire against a
        # closed/torn-down session.
        self._cancel_response_watchdog()
        self._cancel_response_create_watchdog()
        self._cancel_deferred_request_watchdog()
        # Clear the response-cycle latch so a reconnect that doesn't run
        # through start()'s reset block (or a re-attempt that beats the
        # state guard) can't inherit a stuck True flag and silently swallow
        # every subsequent user turn. Same for queued notes — they belong
        # to a torn-down session.
        with self._lock:
            self._response_active = False
            self._response_requested = False
            self._pending_notes = []
        if self._state != LiveApiState.OFF:
            # Spontaneous close — surface as error so user can retry.
            self._set_state(LiveApiState.ERROR, f"Connection closed ({reason or 'unknown'})")

    def _on_ws_error(self, message: str) -> None:
        self._cancel_response_watchdog()
        self._cancel_response_create_watchdog()
        self._cancel_deferred_request_watchdog()
        with self._lock:
            self._response_active = False
            self._response_requested = False
            self._pending_notes = []
        self._emit_error(message)

    # ---- audio ----

    def _on_audio_chunk(self, pcm16: bytes) -> None:
        client = self._client
        if client is None or not client.connected:
            return
        client.send_audio_chunk(pcm16)

    # ---- screen worker ----

    def _start_screen_worker(self) -> None:
        if self._screen_thread is not None:
            return
        self._screen_stop.clear()
        self._screen_request.clear()
        self._screen_thread = threading.Thread(
            target=self._screen_loop, name="LiveApiScreen", daemon=True
        )
        self._screen_thread.start()

    def _screen_loop(self) -> None:
        # Initial frame as soon as we're connected.
        self._capture_and_send_screen("session_start")
        interval = max(1.0, float(self._config.send_screen_interval_sec))
        while not self._screen_stop.is_set():
            triggered = self._screen_request.wait(timeout=interval)
            if self._screen_stop.is_set():
                break
            reason = "explicit_request" if triggered else "interval"
            if triggered:
                self._screen_request.clear()
            if not self._config.send_screen_always and not triggered:
                continue
            self._capture_and_send_screen(reason)

    def _capture_and_send_screen(self, reason: str) -> None:
        # Snapshot to locals so stop() can null these without racing us.
        screen = self._screen
        client = self._client
        logger = self._logger
        if screen is None or client is None or not client.connected:
            return
        started = time.time()
        try:
            frame = screen.capture()
        except Exception as exc:
            if logger:
                logger.exception("screen_capture_unhandled", exc)
            return
        if frame is None:
            return
        try:
            # Neutral caption: don't lead with the active window title
            # because the model otherwise parrots it back instead of
            # actually reading the image. The window title is still a
            # weak hint, just demoted.
            layout = (frame.monitor_layout + " ") if frame.monitor_layout else ""
            windows = (frame.open_windows + " ") if frame.open_windows else ""
            ok = client.send_screen_image(
                frame.b64,
                caption=(
                    "User's full multi-monitor screen capture below. "
                    + layout
                    + windows
                    + f"(reason: {reason}; foreground hint: "
                    f"{frame.active_window_title or 'unknown'})"
                ),
            )
        except Exception as exc:
            if logger:
                logger.exception("screen_send_unhandled", exc)
            return
        if logger:
            logger.event(
                "screen_send",
                reason=reason,
                ok=ok,
                jpeg_kb=round(len(frame.jpeg_bytes) / 1024.0, 2),
                window=frame.active_window_title,
            )
            logger.latency("screen_send", started, ok=ok)

    # ---- realtime event router ----

    def _handle_event(self, event: Dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        # Transcripts of the user's speech (audio in -> text).
        if kind == "conversation.item.input_audio_transcription.completed":
            transcript = str(event.get("transcript") or "")
            # Flag the next tool dispatch as voice-sourced so safety
            # gate runs the spoof-defense + surfaces the "[voice
            # command]" warning banner.
            self._last_input_was_voice = True
            if self._logger:
                self._logger.text("transcript_user", transcript)
            # Phase-4 wiring: pass the realtime transcript through
            # the TranscriptionRouter. The realtime model doesn't
            # expose a fast/accurate tier split — it returns one
            # transcript per turn. But the router's escalation rules
            # (email / URL / file path / digit-run / destructive
            # verb) still spot AMBIGUOUS tokens the user should
            # verify. Log it so the UI can surface a confirm chip.
            try:
                from .transcription_router import (
                    TranscriptionRouter, TranscriptTier)
                router = TranscriptionRouter()
                decision = router.decide(
                    audio_seconds=0.0,
                    fast_text=transcript)
                if decision.tier == TranscriptTier.ACCURATE \
                        and self._logger:
                    self._logger.event(
                        "realtime_transcript_flagged",
                        reason=decision.reason,
                        transcript_len=len(transcript))
                # Emit a flagged signal for the UI when the
                # ambiguity reason was a typed-token risk
                # (email/URL/path/digits) — those are the ones
                # where the user really should verify.
                risky = decision.reason and any(
                    x in decision.reason for x in
                    ("email", "URL", "path", "digit"))
                if risky and self._logger:
                    self._logger.event(
                        "realtime_transcript_risky",
                        reason=decision.reason)
            except Exception:
                pass
            self.transcript_received.emit(transcript)
            return

        if kind in {
            # GA event names + the retired beta names (kept for safety).
            "response.output_text.delta", "response.output_audio_transcript.delta",
            "response.audio_transcript.delta", "response.text.delta",
        }:
            delta = str(event.get("delta") or "")
            if delta:
                if self._logger:
                    self._logger.text("assistant_delta", delta)
                self._turn_text += delta  # for multi-step completion detection
                self.assistant_text.emit(delta)
            return

        if kind in {
            "response.output_text.done", "response.output_audio_transcript.done",
            "response.audio_transcript.done", "response.text.done",
        }:
            if self._logger:
                self._logger.event("assistant_done")
            # Unified audio path: Realtime no longer streams PCM by
            # default (see realtime_client._send_session_update). When
            # the model finishes a text reply, route the accumulated
            # turn text through the same _speak_text() pipeline that
            # Layer-0/Layer-1 local handlers use, so every reply —
            # router, planner, LLM — speaks with the user's configured
            # voice (marin) through the same TTS engine.
            #
            # Skip when TOUCHLESS_REALTIME_AUDIO=1 is set: in that mode
            # Realtime is already streaming audio deltas and speaking
            # the text itself, so double-speaking must be avoided.
            # ---- deterministic-summary override consumption ----
            # If a summary tool armed an override during this turn, REPLACE
            # the LLM's accumulated reply with the override text in BOTH
            # the chat bubble (via assistant_text.emit) and the spoken
            # audio (via _speak_text). The LLM may have already streamed
            # text into the bubble — we emit the override as a follow-up
            # so the displayed reply and spoken reply match the
            # deterministic summary. After consuming, clear the override
            # so the next turn starts clean.
            override = self._pending_override_text
            self._pending_override_text = None
            self._last_override_tool = ""
            self._override_armed_response_id = ""
            if override:
                try:
                    import sys as _sys
                    print(f"[OVERRIDE] fired chars={len(override)}",
                          file=_sys.stderr, flush=True)
                except Exception:
                    pass
                if self._logger:
                    try:
                        self._logger.event(
                            "deterministic_summary_override_fired",
                            chars=len(override))
                    except Exception:
                        pass
                try:
                    # Start a fresh bubble so the override replaces the
                    # LLM's (possibly already-streamed) paraphrase rather
                    # than appending to it.
                    self.assistant_message_break.emit()
                    self.assistant_text.emit(override)
                except Exception:
                    pass
                try:
                    self._emit_reply_output_pulse(override)
                except Exception:
                    pass
                if os.environ.get("TOUCHLESS_REALTIME_AUDIO", "0") != "1":
                    try:
                        self._speak_text(override)
                    except Exception as exc:
                        if self._logger:
                            try:
                                self._logger.exception(
                                    "override_speak_failed", exc)
                            except Exception:
                                pass
                self._record_convo_turn("assistant", override)
                if self._pending_actions:
                    try:
                        self.suggested_actions.emit(list(self._pending_actions))
                    except Exception:
                        pass
                    self._pending_actions = []
                self._set_state(LiveApiState.LISTENING, "Listening")
                return

            if os.environ.get("TOUCHLESS_REALTIME_AUDIO", "0") != "1":
                spoken = (self._turn_text or "").strip()
                if spoken:
                    try:
                        self._speak_text(spoken)
                    except Exception as exc:
                        if self._logger:
                            try:
                                self._logger.exception(
                                    "realtime_text_done_tts_failed", exc)
                            except Exception:
                                pass
                    self._record_convo_turn("assistant", spoken)
            self._set_state(LiveApiState.LISTENING, "Listening")
            return

        # PCM16 audio chunks for the spoken reply. Comes as base64; decode
        # and push to the player. Only happens when voice_output is on AND
        # the session was started with audio in output_modalities.
        if kind in {"response.output_audio.delta", "response.audio.delta"}:
            b64 = str(event.get("delta") or "")
            if not b64:
                return
            # Self-heal: if voice_output is on but the player got dropped
            # (or never built — toggle race), build one on demand here so
            # the first audio delta of a session doesn't get lost. The
            # AudioPlayer is fully thread-safe (Queue-backed); we can
            # construct it from the websocket reader thread.
            player = self._audio_player
            if player is None and self._voice_output:
                try:
                    from .audio_player import AudioPlayer
                    new_player = AudioPlayer(
                        sample_rate=self._config.audio_sample_rate,
                        parent=self,
                    )
                    if new_player.is_enabled():
                        self._audio_player = new_player
                        player = new_player
                        if self._logger:
                            self._logger.event("audio_player_lazy_built")
                    else:
                        if self._logger:
                            self._logger.event("audio_player_lazy_build_no_device")
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("audio_player_lazy_build_failed", exc)
            if player is None:
                # voice_output is off OR build failed — silently drop.
                # First-drop diagnostic so we know WHY no audio.
                if self._logger and not getattr(self, "_audio_drop_logged", False):
                    try:
                        self._audio_drop_logged = True
                        self._logger.event(
                            "audio_delta_no_player",
                            voice_output=self._voice_output)
                    except Exception:
                        pass
                return
            try:
                import base64 as _b64
                player.write(_b64.b64decode(b64))
            except Exception as exc:
                if self._logger:
                    self._logger.exception("audio_delta_decode_failed", exc)
            return

        if kind in {"response.output_audio.done", "response.audio.done"}:
            # Just a marker — nothing to do; the player drains its own buffer.
            return

        # NOTE: TTS for LOCAL handlers (Layer 0 router, Layer 1 iris planner,
        # multistep task router) lives in _speak_text() below. The realtime
        # path above already streams PCM via response.output_audio.delta, so
        # we MUST NOT also synthesize the matching response.text.delta — that
        # would double-speak. Local handlers call _speak_text(...) directly
        # right after their assistant_text.emit(...).

        if kind == "response.created":
            # Disarm the pre-create watchdog FIRST so it can't race against
            # the post-create handler / a later response in the same tick.
            self._cancel_response_create_watchdog()
            self._response_active = True
            self._turn_text = ""  # reset per-response assistant text
            # Bump the turn counter so any _fire_override_inline queued
            # against the PRIOR turn (WS reader thread already returned;
            # queued GUI slot hasn't run yet) sees the mismatch and
            # no-ops instead of flashing state onto this new turn.
            self._turn_id += 1
            new_resp_id = str((event.get("response") or {}).get("id") or "")
            self._active_response_id_seen = new_resp_id
            # Defensive: clear any STALE summary override left over from a
            # prior response. An override armed inside the CURRENT dispatch
            # (e.g. a confirm-gated deterministic-summary tool whose modal
            # blocked the read thread across a response boundary) must
            # survive this event so its consumer in response.text.done /
            # response.done can fire it.
            armed_id = self._override_armed_response_id
            if self._pending_override_text is not None and armed_id and armed_id != new_resp_id:
                self._pending_override_text = None
                self._last_override_tool = ""
                self._override_armed_response_id = ""
            # Each new response is a NEW reply — start a fresh chat bubble so
            # replies don't concatenate into one growing box.
            self.assistant_message_break.emit()
            self._set_state(LiveApiState.THINKING, "Thinking")
            # Arm the response-cycle watchdog so a server-side stall
            # between created and done can't leave the latch wedged.
            self._arm_response_watchdog(new_resp_id)
            return

        if kind == "response.done":
            # The response landed before our watchdog fired — disarm it
            # so a delayed timer can't yank the next turn back to LISTENING.
            self._cancel_response_watchdog()
            resp = event.get("response") or {}
            output = resp.get("output") or []
            output_kinds = [str(item.get("type") or "") for item in output if isinstance(item, dict)]
            status = str(resp.get("status") or "")
            status_details = resp.get("status_details")
            # ---- deterministic-summary override (cancel fallback) ----
            # When we cancelled the LLM mid-reply to save tokens (an
            # override was armed), response.text.done may NOT fire — the
            # response ends as status="cancelled" instead. Consume the
            # override HERE so the summary still reaches the user. If a
            # normal "completed" response landed and text.done already
            # consumed the override, _pending_override_text is already
            # None and this block is a no-op.
            override = self._pending_override_text
            if override and status in ("cancelled", "incomplete"):
                self._pending_override_text = None
                self._last_override_tool = ""
                self._override_armed_response_id = ""
                try:
                    import sys as _sys
                    print(f"[OVERRIDE] fired (cancelled) chars={len(override)}",
                          file=_sys.stderr, flush=True)
                except Exception:
                    pass
                if self._logger:
                    try:
                        self._logger.event(
                            "deterministic_summary_override_fired_cancelled",
                            chars=len(override), status=status)
                    except Exception:
                        pass
                try:
                    self.assistant_message_break.emit()
                    self.assistant_text.emit(override)
                except Exception:
                    pass
                try:
                    self._emit_reply_output_pulse(override)
                except Exception:
                    pass
                if os.environ.get("TOUCHLESS_REALTIME_AUDIO", "0") != "1":
                    try:
                        self._speak_text(override)
                    except Exception as exc:
                        if self._logger:
                            try:
                                self._logger.exception(
                                    "override_speak_failed", exc)
                            except Exception:
                                pass
                # Treat cancelled-with-override as a successful turn so
                # we don't fall into the retry / TPM-error path below.
                self._response_active = False
                self._failed_retries = 0
                self._set_state(LiveApiState.LISTENING, "Listening")
                self._drain_pending()
                return
            if self._logger:
                self._logger.event(
                    "response_done_summary",
                    status=status,
                    status_details=status_details,  # WHY it failed, if it did
                    output_kinds=output_kinds,
                    output_count=len(output),
                    usage=resp.get("usage"),
                )
            self._response_active = False
            # Consume the intentional-cancel latch set on the safety-gate
            # decline path. We already spoke a friendly "cancelled" message
            # inline via _fire_override_inline; retrying here would spawn a
            # phantom response.create and wedge the next user turn behind a
            # stuck _response_active latch.
            if self._intentional_cancel_pending:
                self._intentional_cancel_pending = False
                if status in ("cancelled", "failed", "incomplete"):
                    if self._logger:
                        try:
                            self._logger.event(
                                "intentional_cancel_absorbed", status=status)
                        except Exception:
                            pass
                    self._failed_retries = 0
                    self._set_state(LiveApiState.LISTENING, "Listening")
                    self._drain_pending()
                    return
            # A FAILED/empty response (server rejected, rate-limited, or
            # transient) is NOT a text-only yield — never nudge on it. Retry
            # with EXPONENTIAL BACKOFF (the realtime API fails rapid
            # back-to-back responses; spacing them out recovers). After a few
            # tries, surface a clear error instead of silently stalling.
            if status not in ("completed", ""):
                # Extract the failure reason. A token-per-minute (TPM) rate
                # limit tells us EXACTLY how long to wait ("try again in Xs") —
                # honor that instead of a too-short fixed backoff, with extra
                # retries, so the queue just pauses and resumes.
                err = (status_details or {}).get("error") if isinstance(status_details, dict) else None
                emsg = str((err or {}).get("message", "")) if isinstance(err, dict) else ""
                ecode = str((err or {}).get("code", "")) if isinstance(err, dict) else ""
                is_rate = ecode == "rate_limit_exceeded" or "rate limit" in emsg.lower()
                if is_rate:
                    # Tell the planner's scheduler so the next user turn can
                    # route around realtime via the cheap-LLM Phase 2 path.
                    try:
                        from .planner.scheduler import scheduler as _sched
                        _sched().record_rate_limit("realtime")
                    except Exception:
                        pass
                retry_after = None
                m = re.search(r"try again in ([\d.]+)\s*s", emsg)
                if m:
                    retry_after = float(m.group(1))
                max_retries = 8 if is_rate else 3
                if self._failed_retries < max_retries:
                    self._failed_retries += 1
                    if retry_after is not None:
                        delay = min(retry_after + 1.0, 30.0)  # wait past the window
                    else:
                        delay = min(1.0 * (2 ** (self._failed_retries - 1)), 8.0)
                    if self._logger:
                        self._logger.event("response_failed_retry", status=status,
                                           code=ecode, n=self._failed_retries, delay=round(delay, 1))
                    if is_rate and self._failed_retries == 1:
                        self._set_state(LiveApiState.THINKING, "Rate-limited — pausing…")
                    time.sleep(delay)
                    self._request_model_response()
                    return
                self._failed_retries = 0
                self._multistep_active = False
                self._in_queue = False
                self._emit_error(
                    "Hit the model's per-minute token limit and couldn't recover. "
                    "Some steps didn't finish — ask me to continue. (This is your "
                    "OpenAI account's TPM cap; a higher tier removes the pauses.)"
                )
                self._drain_pending()
                return
            self._failed_retries = 0
            # Realtime fact extraction: this turn went to realtime (not the
            # planner) and completed. Hand it to the memory layer to extract
            # any durable user-revealed facts. Async — never blocks the UI.
            try:
                self._observe_realtime_turn()
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("observe_realtime_failed", exc)
            had_toolcall = "function_call" in output_kinds
            # PLANNING: the model just returned the task list. Parse it and
            # start running the tasks one at a time. If it ignored planning and
            # used a tool, or returned no usable plan, fall back to the
            # heuristic split / running the command whole.
            if self._planning:
                self._planning = False
                if had_toolcall:
                    self._drain_pending()
                    return
                steps = self._parse_plan(self._turn_text) or self._split_into_tasks(self._orig_command)
                if self._logger:
                    self._logger.event("plan_parsed", count=len(steps), steps=steps)
                if self._start_task_queue(steps):
                    return
                # Couldn't plan into ≥2 tasks — just run the original command.
                client = self._client
                if client is not None and getattr(client, "connected", False):
                    client.send_text_message(self._orig_command)
                    self._request_model_response()
                    return
                self._drain_pending()
                return
            # TASK QUEUE: a sub-task is finished when the model yields a
            # text-only confirmation (tool turns auto-continue within the step).
            # Advance to the next sub-task, paced.
            if self._in_queue:
                if not had_toolcall:
                    self._feed_next_task()
                    return
                # still working this step — let the function-call loop continue
                self._drain_pending()
                return
            # Multi-step auto-continue (fallback for unsplit multi-step): only on
            # a COMPLETED, text-only turn, and not on completion.
            if self._multistep_active and not had_toolcall and self._maybe_nudge_continue():
                return  # nudge sent + new response requested
            # Fire any work deferred while it was active (parallel tool-call
            # results, queued background notes) as ONE follow-up turn.
            self._drain_pending()
            return

        if kind == "response.function_call_arguments.done":
            self._dispatch_function_call(event)
            return

        if kind == "error":
            err = event.get("error") or {}
            # Log the FULL error payload so the next debug pass can see
            # exactly what the server rejected (model, code, param, etc).
            if self._logger:
                self._logger.event("server_error_payload", payload=err)
            message = str(err.get("message") or err)
            code = str(err.get("code") or err.get("type") or "")
            param = str(err.get("param") or "")
            # Benign races we should NOT spam the user with:
            #  - response_cancel_not_active: we tried to cancel but the
            #    response already completed naturally. Harmless; happens
            #    by design when the deterministic-summary override
            #    arrives just after the LLM's text.done.
            #  - response.cancel-related conversation_already_has_active_response:
            #    similar lifecycle race during fast back-to-back tool calls.
            _benign_codes = {
                "response_cancel_not_active",
                "conversation_already_has_active_response",
            }
            if code in _benign_codes:
                # `conversation_already_has_active_response` means our
                # response.create was rejected — response.created will
                # NEVER arrive, so the pre-create watchdog must be
                # disarmed and the optimistically-latched
                # _response_active flipped back, or Iris wedges on
                # THINKING. Re-arm _response_requested so the next
                # response.done from the existing in-flight turn drains
                # the deferred work.
                if code == "conversation_already_has_active_response":
                    self._cancel_response_create_watchdog()
                    with self._lock:
                        self._response_active = False
                        self._response_requested = True
                return
            details = message
            if code:
                details = f"[{code}] {details}"
            if param:
                details = f"{details} (param={param})"
            self._emit_error(f"API error: {details}")
            return

        # Catch-all: many session/lifecycle events arrive — just log them.
        if self._logger:
            self._logger.event("event_passthrough", event_type=kind)

    # Outward-facing / irreversible connector actions that must be confirmed
    # by the user before they run (sending email, etc.).
    _CONFIRM_BEFORE_TOOLS = {"gmail_send", "email_send", "ms_mail_send", "teams_send",
                             "teams_channel_post"}

    # Tools whose JSON result contains a deterministic, pre-formatted
    # `summary` string that we want the user to hear/see VERBATIM — the
    # realtime model is NOT allowed to rephrase, truncate, or reorder
    # weather forecasts and email digests. When the tool returns a dict
    # with a non-empty top-level `summary` field AND its name is in this
    # set, _dispatch_function_call sets self._pending_override_text and
    # cancels the LLM's in-flight reply. The response.text.done handler
    # then short-circuits the LLM text and emits/speaks the summary.
    _DETERMINISTIC_SUMMARY_TOOLS = {
        "weather_get", "gmail_list", "ms_mail_list", "email_summary",
        "gmail_send", "ms_mail_send", "outlook_send", "teams_send",
        "contacts_create", "tasks_add", "tasks_complete", "tasks_delete",
        "tasks_list",
        "forms_create", "photos_upload",
        "calendar_create_event", "ms_calendar_create",
        # Docs/Sheets/Slides append + update tools. Same latency pattern
        # as gmail_send: connector returns fast, but the realtime reply
        # cycle can wedge on conversation_already_has_active_response
        # after successful tool completion. Short-circuit with the
        # _format_message summary so the pill returns to LISTENING
        # within ~50ms of tool return instead of hanging on THINKING.
        "gdocs_create", "gdocs_append_text",
        "sheets_create", "sheets_append_rows", "sheets_update_range",
        "slides_create", "slides_add_slide",
        "slides_set_slide_text", "slides_replace_text",
        "drive_upload",
    }

    def _confirm_connector_action(self, name: str, args: Dict[str, Any]) -> bool:
        """Ask the user before an irreversible connector action. Returns True
        to proceed. If no confirm UI is wired, proceed (the model + system
        instructions are the gate in that case)."""
        cb = self._confirm_callback
        if cb is None:
            return True
        if name in ("gmail_send", "email_send", "ms_mail_send"):
            to = str((args or {}).get("to") or (args or {}).get("recipient") or "")
            subject = str((args or {}).get("subject") or "")
            title = f"Send this email now to {to}?"
            detail = f"Subject: {subject or '(none)'}"
        elif name == "teams_send":
            to = str((args or {}).get("to") or "")
            title = f"Send this Teams message to {to}?"
            detail = str((args or {}).get("text") or "")[:200]
        elif name == "teams_channel_post":
            team = str((args or {}).get("team") or "")
            title = f"Post this to the '{team}' Teams channel?"
            detail = str((args or {}).get("text") or "")[:200]
        else:
            title, detail = f"Run {name}?", ""
        try:
            return bool(cb(title, detail))
        except Exception:
            return False

    def _current_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tools sent to the model. Strategy:

        * Built-ins and hand-written API connectors → eager (direct
          exposure, single-turn invocation). Stable, small, high-value.
        * MCP connectors → LAZY. Each MCP server can expose dozens of
          tools (`mcp_<server>_<tool>`); exposing them all eagerly
          bloats context and degrades tool-call accuracy. Instead we
          always expose `find_capability` and load matching MCP tools
          into the live session when the model calls it.
        * Anything already loaded mid-session via find_capability is
          merged in so subsequent turns can call those names directly
          without re-routing.
        """
        if self._registry is None:
            return list(all_tool_schemas())
        eager = self._registry.eager_tools()
        names = {s.get("name") for s in eager}
        # Always include the meta-tool so the model can reach MCP.
        if FIND_CAPABILITY_TOOL.get("name") not in names:
            eager.append(FIND_CAPABILITY_TOOL)
            names.add(FIND_CAPABILITY_TOOL["name"])
        # Merge any connector tools already loaded this session via
        # find_capability (so a follow-up turn can call them directly).
        for schema in self._loaded_connector_schemas:
            n = schema.get("name")
            if n and n not in names:
                eager.append(schema)
                names.add(n)
        return eager

    def _handle_find_capability(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Router: match the task to an available connector, load its tools
        into the live session, and tell the model what to call next."""
        registry = self._registry
        task = str((args or {}).get("task") or "").strip()
        if registry is None or not task:
            return {"status": "ok", "found": False,
                    "hint": "No connector match; control the app via the screen."}
        try:
            # Load the top 2 matches so closely-related connectors both
            # surface (e.g. 'send an email' loads BOTH gmail_send and
            # outlook_compose, letting the model pick send vs draft).
            hits = registry.search_connectors(task, limit=2)
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if not hits:
            return {"status": "ok", "found": False,
                    "hint": "No API connector for that — use screen control (click/type)."}
        existing = {s.get("name") for s in self._loaded_connector_schemas}
        loaded_connectors: List[str] = []
        tool_names: List[str] = []
        for hit in hits:
            for schema in hit.get("schemas", []):
                name = schema.get("name")
                if not name:
                    continue
                tool_names.append(name)
                if name not in existing:
                    self._loaded_connector_schemas.append(dict(schema))
                    existing.add(name)
            loaded_connectors.append(hit.get("id"))
        client = self._client
        if client is not None:
            try:
                client.update_tools(self._current_tool_schemas())
            except Exception as exc:
                if self._logger:
                    self._logger.exception("update_tools_failed", exc)
        return {
            "status": "ok", "found": True, "connectors": loaded_connectors,
            "loaded_tools": tool_names,
            "hint": (f"Loaded tools from {', '.join(loaded_connectors)} — now call "
                     f"the one that fits"
                     + (f", e.g. {tool_names[0]}." if tool_names else ".")),
        }

    def _fire_override_inline(self, text: str, name: str, call_id: str,
                              actions: Optional[List[Any]] = None) -> None:
        """Marshal the inline-override delivery onto the GUI thread.

        The real body (now `_fire_override_inline_on_gui_thread`)
        constructs QTimers via `_emit_reply_output_pulse` and emits Qt
        signals. This method is invoked from the WS reader thread, which
        has no Qt event dispatcher — calling singleShot from there logs
        "Timers cannot be started from another thread" while the window
        is open and, after the assistant window's dispatcher is torn
        down, dereferences freed Qt internals and can crash the process.
        Marshal everything onto self.thread() (the GUI thread that owns
        LiveApiManager) via a queued invokeMethod hop. If the GUI side
        has already been torn down, drop the call silently.

        Snapshots `_turn_id` at queue time so the queued GUI slot can
        detect a fresh user turn racing in and no-op instead of flashing
        stale LISTENING state (and a stale bubble) on top of it.
        """
        if not getattr(self, "_gui_alive", True):
            return
        queued_turn_id = int(getattr(self, "_turn_id", 0))
        try:
            QMetaObject.invokeMethod(
                self, "_fire_override_inline_on_gui_thread",
                Qt.QueuedConnection,
                Q_ARG(str, text), Q_ARG(str, name),
                Q_ARG(str, call_id), Q_ARG(object, actions),
                Q_ARG(int, queued_turn_id),
            )
        except Exception:
            pass

    @Slot(str, str, str, object, int)
    def _fire_override_inline_on_gui_thread(
            self, text: str, name: str, call_id: str,
            actions: Optional[List[Any]] = None,
            queued_turn_id: int = 0) -> None:
        """Deliver a deterministic-summary override IMMEDIATELY instead of
        arming it for the realtime text.done / response.done consumer.

        Used when the confirm-gated dispatch path straddled a response
        boundary — the normal arm-and-wait flow loses the override to the
        `response.created` defensive wipe or to the orphaned-tool-result
        race. Mirrors the consumer at the top of response.text.done:
        emits a fresh assistant bubble, speaks via _speak_text, records
        the turn, emits action chips, and returns the state pill to
        LISTENING.

        Always runs on the GUI thread (via `_fire_override_inline`'s
        queued invokeMethod hop).

        `queued_turn_id` is the `_turn_id` snapshot taken when the WS
        reader thread queued this call. If the current `_turn_id` has
        advanced (a new user turn's response.created landed between the
        queue and now), the whole slot no-ops — otherwise the pill would
        flash to LISTENING and a stale decline bubble would drop on top
        of the fresh turn.
        """
        if not getattr(self, "_gui_alive", True):
            return
        current_turn_id = int(getattr(self, "_turn_id", 0))
        if queued_turn_id and queued_turn_id != current_turn_id:
            if self._logger:
                try:
                    self._logger.event(
                        "override_inline_stale_skipped",
                        tool=name, queued=queued_turn_id,
                        current=current_turn_id)
                except Exception:
                    pass
            return
        try:
            import sys as _sys
            print(f"[OVERRIDE] inline tool={name} chars={len(text)}",
                  file=_sys.stderr, flush=True)
        except Exception:
            pass
        if self._logger:
            try:
                self._logger.event(
                    "deterministic_summary_override_fired_inline",
                    tool=name, chars=len(text))
            except Exception:
                pass
        try:
            self.tool_event.emit("override", {
                "name": name, "call_id": call_id,
                "chars": len(text), "reason": "confirm_gated_inline",
            })
        except Exception:
            pass
        # Clear any stale armed override so a future response.created
        # wipe check stays correct.
        self._pending_override_text = None
        self._last_override_tool = ""
        self._override_armed_response_id = ""
        try:
            self.assistant_message_break.emit()
            self.assistant_text.emit(text)
        except Exception:
            pass
        try:
            self._emit_reply_output_pulse(text)
        except Exception:
            pass
        if os.environ.get("TOUCHLESS_REALTIME_AUDIO", "0") != "1":
            try:
                self._speak_text(text)
            except Exception as exc:
                if self._logger:
                    try:
                        self._logger.exception(
                            "override_inline_speak_failed", exc)
                    except Exception:
                        pass
        try:
            self._record_convo_turn("assistant", text)
        except Exception:
            pass
        if actions:
            try:
                self.suggested_actions.emit(list(actions))
            except Exception:
                pass
        elif self._pending_actions:
            try:
                self.suggested_actions.emit(list(self._pending_actions))
            except Exception:
                pass
            self._pending_actions = []
        self._set_state(LiveApiState.LISTENING, "Listening")

    def _dispatch_function_call(self, event: Dict[str, Any]) -> None:
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or "")
        raw_args = event.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception:
            args = {}
        if self._logger:
            self._logger.event("function_call_received", tool=name, call_id=call_id)
        # Which layer is executing this — for the UI badge (Touchless / iris).
        # find_capability and any connector-owned tool = the API "connector"
        # layer; everything else is iris's GUI/built-in computer-use.
        if name == "find_capability" or (
            self._registry is not None and self._registry.handles_connector(name)
        ):
            source = "connector"
        else:
            source = "iris"
        self.tool_event.emit("called", {"name": name, "call_id": call_id, "source": source})
        self._set_state(LiveApiState.EXECUTING, f"Executing tool: {name}")

        # Phase-1 substrate: start a ToolInvocation observation so the
        # audit log / activity pill / undo registry see every realtime
        # tool dispatch through the same contract as the planner path.
        # Best-effort; the dispatch must work even if the bus is down.
        rt_inv: Optional[ToolInvocation] = None
        try:
            rt_inv = ToolInvocation.starting(
                tool=name, args=args,
                source=InvocationSource.REALTIME,
                turn_id=(self._active_response_id
                         if hasattr(self, "_active_response_id")
                         else None),
            )
        except Exception:
            rt_inv = None

        executor = self._executor
        client = self._client
        if executor is None or client is None:
            return

        # Whether THIS dispatch went through a blocking confirm path
        # (narrow _CONFIRM_BEFORE_TOOLS modal OR the universal safety_gate
        # for DESTRUCTIVE/IRREVERSIBLE tools). The deterministic-summary
        # override below uses this to fire INLINE for gated tools — the
        # blocking modal would otherwise straddle a realtime response
        # boundary and lose the override to the response.created wipe.
        was_gated = False

        try:
            if name == "find_capability":
                # Meta-tool: discover + load connector tools into the session.
                output = self._handle_find_capability(args)
            elif name == "gmail_send" and (
                self._registry is None or not self._registry.handles_connector("gmail_send")
            ):
                # Gmail isn't connected (no OAuth) so this tool isn't real —
                # fail cleanly and tell the model to stop, not retry.
                output = {
                    "status": "error", "code": "not_connected",
                    "error": ("Gmail direct-send is not connected. Open a draft "
                              "with outlook_compose for the user to send manually "
                              "(or have them connect Gmail). Do NOT retry."),
                }
            elif name in self._CONFIRM_BEFORE_TOOLS and not self._confirm_connector_action(name, args):
                # Irreversible action (e.g. sending email) the user declined.
                was_gated = True
                output = {"status": "cancelled", "code": "user_declined",
                          "error": "User declined; not sent."}
            else:
                if name in self._CONFIRM_BEFORE_TOOLS:
                    was_gated = True
                # Phase-1 trust gate (universal): pre-invocation confirm
                # for DESTRUCTIVE / IRREVERSIBLE tools the model picks.
                # This catches tools NOT in the older narrow
                # _CONFIRM_BEFORE_TOOLS list (which only covered email
                # sends). Failure-open when no UI callback installed.
                from .safety_gate import gate as _safety_gate, needs_confirmation as _needs_confirm
                if _needs_confirm(name, args):
                    was_gated = True
                # Phase-1 source tagging: voice input gets the
                # spoof-defense + "[voice command]" banner; typed
                # input does NOT (it's the user's keyboard — no
                # need to warn them they typed it). _last_input_was_voice
                # is set in the audio-transcription event handler /
                # cleared by send_user_text.
                _source = ("voice" if self._last_input_was_voice
                           else "typed")
                _allowed, _decline = _safety_gate(name, args,
                                                  source=_source)
                if not _allowed:
                    output = {"status": "cancelled",
                              "code": "user_declined_speed_bump",
                              "error": _decline or "user declined"}
                else:
                    # Route through the registry so API connectors (Gmail, etc.)
                    # handle their own tools; everything else falls through to the
                    # GUI/built-in executor.
                    registry = self._registry
                    output = (registry.call(name, args) if registry is not None
                              else executor.execute(name, args))
        except Exception as exc:  # defensive — executor already catches
            output = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        # On-demand vision: when the model explicitly asks to look at the
        # screen, push a real screenshot (input_image) BEFORE the tool
        # result so it can actually see it this turn. Cloud-only; the
        # local backend has no vision in Phase 1.
        if name == "get_screen_context" and (self._config.backend or "cloud").strip().lower() == "cloud":
            self._capture_and_send_screen("get_screen_context")
        # zoom_screen returns a high-res crop in _zoom_image_b64 — inject it as
        # an input_image so the model can SEE the zoomed area, then strip it
        # from the tool result (it's large and the model already gets the
        # picture).
        if name == "zoom_screen" and isinstance(output, dict):
            crop_b64 = output.pop("_zoom_image_b64", None)
            if crop_b64 and (self._config.backend or "cloud").strip().lower() == "cloud":
                try:
                    client.send_screen_image(
                        crop_b64,
                        caption=("Zoomed-in high-res crop of the area you asked to "
                                 "inspect. To click inside it, call click_zoom(x, y) "
                                 "with coords 0-1 RELATIVE TO THIS CROP."),
                    )
                except Exception as exc:
                    if self._logger:
                        self._logger.exception("zoom_image_send_failed", exc)
        if self._logger:
            self._logger.event("function_call_completed", tool=name, status=output.get("status"))
            # Structured routing-decision log: which cost tier handled this.
            self._logger.event("routing_decision", **cost_policy.decision_record(
                tool=name, source=source, status=str(output.get("status", "")),
                call_id=call_id))
        self.tool_event.emit("completed", {"name": name, "call_id": call_id, "status": output.get("status"), "source": source, "error": (output.get("error") if isinstance(output, dict) else "") or ""})

        # Mirror the realtime-created artifact (sheet/doc/slide/etc.) into
        # the planner's _last_artifacts map so 'open it' / 'open the sheet'
        # resolves regardless of which tier originally created the thing.
        # Without this, only planner-handled creations populate the tracker
        # and the realtime path leaves a stale prior artifact winning.
        planner = self._iris_planner
        if planner is not None and isinstance(output, dict) and output.get("link"):
            recorder = getattr(planner, "record_artifact_from_result", None)
            if callable(recorder):
                try:
                    recorder(name, output)
                except Exception as exc:
                    if self._logger:
                        self._logger.exception(
                            "realtime_artifact_record_failed", exc)

        # Phase-1 substrate: complete + publish the realtime ToolInvocation
        # observation. Audit log + activity pill subscribe to this; no-op
        # if the observation failed to start.
        if rt_inv is not None:
            try:
                rt_inv.complete(
                    status=str((output or {}).get("status") or "ok"),
                    output=output,
                    error=(output or {}).get("error"),
                )
                _publish_invocation(rt_inv)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("realtime_invocation_publish_failed", exc)

        # ---- deterministic-summary override ----
        # Weather/email "summary" tools return a pre-formatted string that
        # the realtime model is NOT allowed to rephrase. If THIS tool is
        # one of those AND it returned a non-empty top-level `summary`,
        # stash it; response.text.done will replace the LLM's reply with
        # this exact text. Chained-tool safety: we always OVERWRITE the
        # pending override here (later summary tools win), and clear it if
        # a non-summary tool fired most recently — so 'send me an email
        # with today's weather' speaks the gmail_send result, not the
        # weather summary captured earlier.
        # ---- universal connector-error short-circuit ----
        # Connector error/cancelled results carry no `summary` field, so
        # the deterministic-summary gate below would let them fall through
        # to the realtime LLM — which then has to narrate the failure and
        # often loops or stalls. Intercept here: compose a tiny, friendly
        # line from `error`/`code`, arm the override, cancel any in-flight
        # response, and return. We still send the tool result back so the
        # model's conversation log stays consistent.
        try:
            if isinstance(output, dict) and str(
                    output.get("status") or "").lower() in ("error", "cancelled"):
                err_msg = str(output.get("error") or "").strip()
                code = str(output.get("code") or "").strip()
                friendly = _format_connector_error(name, code, err_msg)
                # User declined a safety-gate modal? The WS reader thread
                # was BLOCKED inside the modal across the prior response's
                # lifecycle, so arming _pending_override_text and calling
                # response.cancel + response.create leaves the session in
                # the exact stalled state documented at the was_gated
                # branch below (orphaned function_call_output +
                # response.created never arrives -> pre-create watchdog
                # fires on the NEXT user turn with "I didn't get a
                # response back from the server"). Fire the friendly
                # cancel inline on the GUI thread instead — clears state,
                # returns to LISTENING, does not touch the deferred-
                # request machinery.
                gated_decline = (
                    was_gated
                    or code in ("user_declined", "user_declined_speed_bump")
                )
                if gated_decline:
                    # latch first so a missing/raising cancel_fn can't leak a phantom retry
                    self._intentional_cancel_pending = True
                # Still cancel the in-flight response so the LLM doesn't
                # burn tokens generating a reply we've already spoken.
                # (For non-gated errors this is unchanged from before.)
                if self._response_active:
                    try:
                        cancel_fn = getattr(client, "cancel_response", None)
                        if callable(cancel_fn):
                            cancel_fn()
                    except Exception as exc:
                        if self._logger:
                            try:
                                self._logger.exception(
                                    "error_override_cancel_failed", exc)
                            except Exception:
                                pass
                # Still submit the tool_result so the model's
                # conversation log stays consistent.
                try:
                    client.send_tool_result(call_id, output)
                except Exception:
                    pass
                try:
                    self.tool_event.emit("override", {
                        "name": name, "call_id": call_id,
                        "chars": len(friendly),
                        "reason": (
                            "user_declined_inline" if gated_decline
                            else "connector_error"
                        ),
                    })
                except Exception:
                    pass
                if self._logger:
                    try:
                        self._logger.event(
                            ("user_declined_override_inline" if gated_decline
                             else "connector_error_override_armed"),
                            tool=name, code=code, chars=len(friendly))
                    except Exception:
                        pass
                if gated_decline:
                    # Fire on GUI thread: clears _pending_override_text /
                    # _override_armed_response_id / _last_override_tool,
                    # emits the friendly message, speaks it, records the
                    # turn, sets state to LISTENING. Skip
                    # _request_model_response — no follow-up LLM reply is
                    # needed; the next user turn drives its own
                    # response.create.
                    self._fire_override_inline(friendly, name, call_id)
                    # Also proactively clear any stale response-cycle
                    # latches left over from the cancel we just sent, so
                    # a next user turn doesn't defer behind a phantom
                    # _response_active.
                    with self._lock:
                        self._response_active = False
                        self._response_requested = False
                    self._cancel_response_create_watchdog()
                    self._cancel_deferred_request_watchdog()
                    return
                # Non-gated connector error: keep the historical
                # arm-and-wait behavior so those cases still flow through
                # response.done -> override consumer -> drain, unchanged
                # from before.
                self._pending_override_text = friendly
                self._last_override_tool = name
                self._override_armed_response_id = self._active_response_id_seen
                self._request_model_response()
                self._set_state(LiveApiState.THINKING, "Thinking")
                return
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception("error_override_detect_failed", exc)
                except Exception:
                    pass

        # If the deterministic-summary override fires inline below (gated
        # path), we MUST NOT also ask the LLM for a follow-up reply — that
        # would double-respond and trigger the orphaned-call_id race the
        # confirm-gated path was losing to.
        fired_inline = False
        try:
            if isinstance(output, dict) and name in self._DETERMINISTIC_SUMMARY_TOOLS:
                summary_val = output.get("summary")
                if isinstance(summary_val, str) and summary_val.strip():
                    # Jarvis prose pass: rewrite the deterministic
                    # summary as conversational Iris-voice prose with
                    # fact-preservation guard + conversation context.
                    # Falls back to summary_val.strip() on any error,
                    # so we never DOWNGRADE from the verbatim summary.
                    base = summary_val.strip()
                    try:
                        from .prose_renderer import render_jarvis
                        composed = render_jarvis(
                            question=self._last_user_text,
                            tool_name=name,
                            tool_result=output,
                            fallback=base,
                            context=self._convo_context(),
                        )
                    except Exception:
                        composed = base
                    composed_text = composed or base
                    # Stash any suggested action chips so the chat UI can
                    # render Connect buttons / read-screen prompts beneath
                    # the composed reply when it fires.
                    actions: List[Any] = []
                    try:
                        raw_actions = output.get("suggested_actions") or []
                        if isinstance(raw_actions, list) and raw_actions:
                            actions = list(raw_actions)
                    except Exception:
                        actions = []
                    if was_gated:
                        # Confirm gate (modal QMessageBox) blocked this
                        # WS read thread across the prior response's
                        # lifecycle. The armed-then-text.done consumer
                        # path is unsafe here: the next response.created
                        # would either wipe the override or the orphaned
                        # function_call_output would stall the new
                        # response. Fire the override INLINE — emit the
                        # bubble + speak it + return to LISTENING — and
                        # skip the follow-up response.create entirely.
                        self._fire_override_inline(
                            composed_text, name, call_id, actions)
                        fired_inline = True
                    else:
                        self._pending_override_text = composed_text
                        self._last_override_tool = name
                        self._override_armed_response_id = self._active_response_id_seen
                        if actions:
                            self._pending_actions = actions
                        try:
                            import sys as _sys
                            print(f"[OVERRIDE] tool={name} chars={len(self._pending_override_text)}",
                                  file=_sys.stderr, flush=True)
                        except Exception:
                            pass
                        if self._logger:
                            try:
                                self._logger.event(
                                    "deterministic_summary_override_armed",
                                    tool=name, chars=len(self._pending_override_text))
                            except Exception:
                                pass
                        # Cancel the LLM's in-flight reply so we don't pay
                        # tokens for text we'll discard. Only fire when a
                        # response is actually active — calling cancel
                        # otherwise produces a benign but user-visible
                        # `response_cancel_not_active` error event.
                        if self._response_active:
                            try:
                                cancel_fn = getattr(client, "cancel_response", None)
                                if callable(cancel_fn):
                                    cancel_fn()
                            except Exception as exc:
                                if self._logger:
                                    try:
                                        self._logger.exception(
                                            "override_cancel_failed", exc)
                                    except Exception:
                                        pass
                        # Surface override in the tool-event stream for UI/debug.
                        try:
                            self.tool_event.emit("override", {
                                "name": name, "call_id": call_id,
                                "chars": len(self._pending_override_text),
                            })
                        except Exception:
                            pass
            elif name not in self._DETERMINISTIC_SUMMARY_TOOLS:
                # A non-override tool ran AFTER (or instead of) a summary
                # tool — its reply should NOT be hijacked by a stale
                # override from earlier in the same response cycle.
                if self._pending_override_text is not None:
                    if self._logger:
                        try:
                            self._logger.event(
                                "deterministic_summary_override_cleared",
                                cleared_by=name, prior_tool=self._last_override_tool)
                        except Exception:
                            pass
                    self._pending_override_text = None
                    self._override_armed_response_id = ""
                self._last_override_tool = ""
        except Exception as exc:  # never let override logic break the loop
            if self._logger:
                try:
                    self._logger.exception("override_detect_failed", exc)
                except Exception:
                    pass

        # Always submit the tool output (conversation.item.create is allowed
        # even mid-response). The follow-up response.create is GATED: if the
        # model emitted several tool calls in one response, each result lands
        # here while that response is still active — _request_model_response
        # defers and fires a single turn on response.done instead of N
        # colliding response.create calls.
        client.send_tool_result(call_id, output)
        self._request_model_response()
        if not fired_inline:
            self._set_state(LiveApiState.THINKING, "Thinking")

        # If the tool just changed something on screen, schedule a fresh
        # screenshot so the model sees the result of its own action.
        if name in {
            "click_screen", "type_text", "press_hotkey", "open_app",
            "open_url", "create_folder", "create_file", "write_file",
            "append_file", "skip_youtube_ad",
        }:
            self.request_screen_now()

    # ---- state helpers ----

    def _set_state(self, state: LiveApiState, status_text: str) -> None:
        # During/after teardown, drop late callbacks from the WebSocket
        # reader thread so we don't re-enter ERROR after the user
        # already clicked Stop.
        if self._teardown_in_progress and state != LiveApiState.OFF:
            return
        self._state = state
        logger = self._logger
        if logger is not None:
            try:
                logger.event("state_changed", state=state.value, status=status_text)
            except Exception:
                pass
        try:
            self.state_changed.emit(state, status_text)
        except Exception:
            pass
        # Cortex viz: best-effort push of the core state. Silent no-op
        # when the Cortex window isn't open (cortex_emit handles it).
        try:
            cortex_state, intensity = _CORTEX_STATE_MAP.get(state, ("idle", 0.4))
            # Speaking-window guard: while _emit_reply_output_pulse has
            # flipped the core to PURPLE (speaking) for a reply, swallow
            # any 'listening' push that would prematurely flip it back
            # to blue. The scheduled _return_to_listening (inside
            # _emit_reply_output_pulse) is the single source of truth for
            # the listening→blue transition during that window. We still
            # let through non-'listening' states (thinking, executing,
            # error) so genuine activity DOES interrupt the purple glow.
            suppress_cortex_push = False
            if (
                cortex_state == "listening"
                and self._cortex_speaking_until > time.time()
            ):
                suppress_cortex_push = True
            if not suppress_cortex_push:
                cortex_emit.core_state(cortex_state, intensity)
            # Diagnostic so users can confirm state changes are flowing
            # to the cortex viz layer (helps debug "core didn't change
            # color" — if you see these lines but no color shift, the
            # bus writer isn't wired; if you don't see them at all, the
            # planner isn't transitioning states).
            try:
                suffix = " [suppressed: speaking window]" if suppress_cortex_push else ""
                print(
                    f"CORE state -> {cortex_state} (intensity={intensity:.2f}) "
                    f"[from {state.value if hasattr(state,'value') else state}]"
                    f"{suffix}",
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass
        except Exception:
            pass

    def _emit_error(self, message: str) -> None:
        if self._logger:
            self._logger.error("session_error", message=message)
        self._set_state(LiveApiState.ERROR, f"Error: {message}")
        self.error_occurred.emit(message)

    # ---- local-handler TTS -------------------------------------------------
    # Realtime LLM responses come with their own audio. Local handlers (Layer
    # 0 router, Layer 1 iris planner, multistep task router) only emit text,
    # so without this they're silent even when voice_output=True. _speak_text
    # bridges that gap: synthesize via OpenAI's REST TTS endpoint and feed
    # the PCM straight into the existing AudioPlayer queue.
    #
    # Design choices:
    #   * Cloud backend only (uses the configured OpenAI key + voice). Local
    #     backend silently no-ops — local users opted out of cloud.
    #   * Runs the HTTP call on a daemon worker thread so it never blocks
    #     the websocket reader or the iris-send-user-text worker.
    #   * Caches by exact text so repeated short replies ('Done.', 'On it.')
    #     don't re-roundtrip / re-bill.
    #   * Skips trivially short / empty text (single-char acks aren't worth
    #     the latency).
    #   * Best-effort: any failure is logged and swallowed; the text reply
    #     still reached the UI through assistant_text.emit.

    # ---- conversational-TTS rewriter --------------------------------------
    # The raw text we synthesize to speech is usually weather lines, status
    # lines, or planner replies that read awkwardly out loud ('82°F (feels
    # like 71°F)', 'humidity: 28%', 'Corvallis — current: sunny'). This
    # helper does a lightweight gpt-5-mini pass to rewrite ONLY the spoken
    # form into natural casual English while keeping proper nouns + numbers
    # intact. The UI still shows the original text via assistant_text —
    # only the audio path uses the humanized version.
    #
    # Design:
    #   * Skip rewrite for short text (<50 chars) — those are already
    #     conversational and rewriting wastes latency + tokens.
    #   * Cache by ORIGINAL text so 'The weather is sunny' rewrites once
    #     per session.
    #   * 0.8s urllib timeout — must be tighter than the TTS HTTP timeout
    #     so a slow rewrite doesn't pile up behind the TTS call.
    #   * Best-effort: any failure / timeout / empty result returns the
    #     original text unchanged.
    #   * Runs INSIDE _run_tts_synthesize (already on a daemon worker), so
    #     it never blocks tool dispatch or the websocket reader.
    _TTS_HUMANIZE_MIN_CHARS = 50
    _TTS_HUMANIZE_MODEL = "gpt-5-mini"
    # Bumped 0.8 → 6.0s. The old 0.8s cap timed out on every
    # multi-sentence reply (3-day forecast etc.) — fallback returned
    # the raw text, but on slow uplinks the SUBSEQUENT TTS call also
    # missed its window and audio dropped entirely. 6s comfortably
    # covers gpt-4o-mini rewrites of long replies; longer than that
    # the user prefers silent text to a delayed voice anyway.
    _TTS_HUMANIZE_TIMEOUT = 6.0

    def _humanize_for_speech(self, text: str) -> str:
        """Rewrite `text` into natural casual speech for TTS playback.

        DISABLED by default — local handlers (weather, planner replies,
        etc.) now produce already-conversational text, and the LLM
        rewrite was adding 2-6 seconds of latency for marginal voice
        quality gain. Re-enable by setting TOUCHLESS_TTS_HUMANIZE=1
        in the environment if you want the rewrite path back.

        Returns the original text unchanged on short input, missing key,
        cache miss + network failure, timeout, or empty model response.
        Cached by original-text to avoid double-billing repeat phrases.
        """
        try:
            # Default OFF — text from local handlers is already casual.
            # Set TOUCHLESS_TTS_HUMANIZE=1 to opt in to the LLM rewrite.
            if os.environ.get("TOUCHLESS_TTS_HUMANIZE", "0") != "1":
                return text
            if not text:
                return text
            stripped = text.strip()
            if len(stripped) < self._TTS_HUMANIZE_MIN_CHARS:
                return text  # already short/conversational
            # Cache hit avoids both LLM round-trip AND keeps TTS-bytes
            # cache coherent (TTS cache is keyed on humanized text).
            try:
                with self._tts_lock:
                    cached = self._tts_humanize_cache.get(text)
                if isinstance(cached, str) and cached:
                    return cached
            except Exception:
                pass

            cfg = self._config
            api_key = getattr(cfg, "api_key", None) if cfg is not None else None
            if not api_key:
                return text

            import urllib.request
            import urllib.error

            system = (
                "Rewrite the input as a casual, friendly spoken reply — like "
                "a person texting back, NOT a formal robot. Use everyday "
                "phrasing ('it's sunny', 'looks like', 'about 80'). "
                "PRESERVE EVERY DETAIL from the input — don't drop the "
                "'feels like' temp, don't drop the wind, don't drop any "
                "number or fact. Convert technical formats spoken-style "
                "('82°F' -> '82 degrees', '28%' -> 'low humidity' is OK "
                "ONLY if the input is structured-stats; '3:45pm' -> 'three "
                "forty five pm'). PRESERVE all proper nouns, person names, "
                "place names, product names, email addresses, URLs. Output "
                "ONLY the rewritten text — no preamble, no quotes, no "
                "explanation. Length: aim for natural; do NOT cap word count."
            )
            # Bound input so a runaway long reply can't blow tokens.
            user = f"Rewrite for natural casual speech:\n{stripped[:1500]}"
            body = json.dumps({
                "model": self._TTS_HUMANIZE_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Bumped 200 -> 600 because the 200 cap was truncating
                # multi-sentence replies like 3-day forecasts.
                "max_completion_tokens": 600,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=self._TTS_HUMANIZE_TIMEOUT
                ) as resp:
                    if getattr(resp, "status", 200) != 200:
                        return text
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if self._logger:
                    try:
                        self._logger.event("tts_humanize_http_error",
                                            code=getattr(exc, "code", 0))
                    except Exception:
                        pass
                return text
            except Exception as exc:
                if self._logger:
                    try:
                        self._logger.event(
                            "tts_humanize_timeout"
                            if "timed out" in str(exc).lower()
                            else "tts_humanize_failed",
                            error=type(exc).__name__,
                        )
                    except Exception:
                        pass
                return text

            rewritten = ((payload.get("choices") or [{}])[0]
                         .get("message", {}).get("content") or "").strip()
            # Defensive strip — some models wrap output in quotes despite
            # the prompt's no-quotes instruction.
            if rewritten.startswith(('"', "'")) and rewritten.endswith(('"', "'")):
                rewritten = rewritten[1:-1].strip()
            if not rewritten:
                return text

            try:
                with self._tts_lock:
                    if len(self._tts_humanize_cache) >= self._tts_humanize_cache_max:
                        try:
                            oldest = next(iter(self._tts_humanize_cache))
                            self._tts_humanize_cache.pop(oldest, None)
                        except StopIteration:
                            pass
                    self._tts_humanize_cache[text] = rewritten
            except Exception:
                pass

            if self._logger:
                try:
                    self._logger.event("tts_humanize_ok",
                                        original_len=len(text),
                                        rewritten_len=len(rewritten))
                except Exception:
                    pass
            return rewritten
        except Exception as exc:
            # Outer guard — under no circumstances let the rewriter break
            # the TTS pipeline. Fall back to the original text.
            if self._logger:
                try:
                    self._logger.event("tts_humanize_failed",
                                        error=type(exc).__name__)
                except Exception:
                    pass
            return text

    def _emit_reply_output_pulse(self, text: str = "") -> None:
        """Fire the FULL output path in the cortex viz after each final
        assistant reply.

        Always: core → cap-realtime → rt-output (text/multimodal output)
        When voice on: ALSO core → cap-voice → voice-tts (audio output)

        `text`, if provided, drives the purple speaking-state duration
        so the core stays purple roughly as long as the reply takes to
        be typed/spoken (estimated at ~150 words/minute speech rate,
        floor 1.5s, ceiling 30s). Without text, falls back to a fixed
        1.2s window.
        """
        if not getattr(self, "_gui_alive", True):
            return
        try:
            own_thread = self.thread()
        except Exception:
            own_thread = None
        if own_thread is not None and QThread.currentThread() is not own_thread:
            try:
                QMetaObject.invokeMethod(
                    self, "_emit_reply_output_pulse_on_gui_thread",
                    Qt.QueuedConnection,
                    Q_ARG(str, text),
                )
            except Exception:
                pass
            return
        self._emit_reply_output_pulse_on_gui_thread(text)

    @Slot(str)
    def _emit_reply_output_pulse_on_gui_thread(self, text: str = "") -> None:
        if not getattr(self, "_gui_alive", True):
            return
        # Schedule each leg with a delay so the user sees neurons
        # TRAVELING through the path. Also flip core to SPEAKING
        # (purple) at the start, then back to LISTENING (blue) after
        # the path completes — so the core's color matches what's
        # actually happening (delivering output).
        from PySide6.QtCore import QTimer as _QT
        from PySide6.QtWidgets import QApplication as _QApp
        def _later(ms, fn):
            try:
                _app = _QApp.instance()
                if _app is not None:
                    _QT.singleShot(ms, _app, fn)
                else:
                    fn()
            except Exception:
                pass
        def _safe_pulse(a, b, color="magenta", ms=300):
            def _do():
                try:
                    cortex_emit.edge_pulse(a, b, color=color, duration_ms=ms)
                except Exception:
                    pass
            return _do
        # Core → SPEAKING (purple glow) for the delivery window.
        try:
            cortex_emit.core_state("speaking", 1.15)
        except Exception:
            pass
        # Tightened stagger 320→150ms per leg so the full path lands
        # within ~300ms instead of 1+ second after the text. User was
        # seeing the text appear THEN the cortex catch up; now they
        # arrive together.
        _safe_pulse("core", "cap-realtime", ms=180)()    # immediate
        _later(150, _safe_pulse("cap-realtime", "rt-output", ms=180))
        # Voice-on: parallel TTS path through cap-voice → voice-tts.
        if self._voice_output:
            _later(40,  _safe_pulse("core", "cap-voice", ms=180))
            _later(200, _safe_pulse("cap-voice", "voice-tts", ms=180))
        # Compute the purple-speaking duration from text length so the
        # core stays purple roughly as long as the reply takes to read
        # / be spoken. Speech rate ~150 wpm = 2.5 words/sec. Add a
        # bit of buffer so we don't return to blue mid-syllable. Floor
        # 1500ms (visible flash), ceiling 30000ms (don't run forever).
        try:
            word_count = max(1, len((text or "").split())) if text else 0
        except Exception:
            word_count = 0
        if word_count > 0:
            speak_ms = int(word_count / 2.5 * 1000) + 600
            speak_ms = max(1500, min(30000, speak_ms))
        else:
            speak_ms = 1200  # legacy default when no text passed
        # Mark the speaking window so _set_state suppresses any racing
        # 'listening' pushes that would overwrite the purple core before
        # _return_to_listening fires. Take the MAX with the existing
        # value so back-to-back replies extend (not shorten) the window.
        try:
            new_until = time.time() + (speak_ms / 1000.0)
            if new_until > self._cortex_speaking_until:
                self._cortex_speaking_until = new_until
        except Exception:
            pass
        def _return_to_listening():
            try:
                cortex_emit.core_state("listening", 0.85)
            except Exception:
                pass
            # Clear the guard so subsequent legitimate LISTENING
            # transitions push through normally. Only clear when we're
            # actually past the window — if a longer reply extended it,
            # leave it alone so the later _return_to_listening wins.
            try:
                if time.time() >= self._cortex_speaking_until:
                    self._cortex_speaking_until = 0.0
            except Exception:
                pass
        _later(speak_ms, _return_to_listening)

    @Slot(str)
    def _run_router_path_pulses_on_gui_thread(self, label: str) -> None:
        """GUI-thread executor for the router cortex-pulse path.

        Mirrors the inline _router_path_pulses helper in
        `_run_send_user_text_inner`: same staggered sequence using
        QTimer.singleShot bound to the QApplication. Invoked via
        QMetaObject.invokeMethod from the planner worker so the
        timers are constructed under the GUI thread's live event
        dispatcher rather than the worker's missing one.
        """
        if not getattr(self, "_gui_alive", True):
            return
        from PySide6.QtCore import QTimer as _QT_router
        from PySide6.QtWidgets import QApplication as _QApp_router
        try:
            cortex_emit.edge_pulse("core", "cap-tools",
                                    color="orange", duration_ms=280)
        except Exception:
            pass
        def _step2():
            try:
                cortex_emit.edge_pulse("cap-tools", f"tool-{label}",
                                        color="orange", duration_ms=280)
            except Exception:
                pass
        def _step3():
            try:
                cortex_emit.node_activity(f"tool-{label}",
                                           intensity=1.0, duration_ms=400)
            except Exception:
                pass
        def _step4():
            try:
                cortex_emit.edge_pulse(f"tool-{label}", "core",
                                        color="orange", duration_ms=280)
            except Exception:
                pass
        try:
            _app = _QApp_router.instance()
            if _app is not None and getattr(self, "_gui_alive", True):
                _QT_router.singleShot(300, _app, _step2)
                _QT_router.singleShot(600, _app, _step3)
                _QT_router.singleShot(1000, _app, _step4)
            else:
                _step2(); _step3(); _step4()
        except Exception:
            pass

    def _speak_text(self, text: str) -> None:
        """Synthesize `text` to speech via OpenAI TTS and queue it for
        playback through self._audio_player. Cloud backend + voice_output
        only; everything else is a silent no-op. Returns immediately —
        the HTTP request runs on a background thread.

        Always active when voice_output=True and cloud backend is in use;
        the marin voice (or its fallback) ensures local-handler replies
        match the realtime LLM's voice. Opt-out by setting
        TOUCHLESS_LOCAL_TTS=0 in the environment if needed."""
        try:
            # Diagnostic: surface why TTS skips so users can debug
            # 'why is voice silent for some prompts but not others'.
            def _skip(reason):
                try:
                    print(f"TTS skip: {reason} (text={text[:40]!r})",
                          file=sys.stderr, flush=True)
                except Exception:
                    pass
            if os.environ.get("TOUCHLESS_LOCAL_TTS", "1") == "0":
                _skip("TOUCHLESS_LOCAL_TTS=0 env opt-out")
                return
            if not self._voice_output:
                _skip("voice_output=False")
                return
            if self._audio_player is None:
                _skip("audio_player=None (init failed)")
                return
            cfg = self._config
            if cfg is None or not getattr(cfg, "api_key", None):
                _skip("no api_key")
                return
            if (getattr(cfg, "backend", "cloud") or "cloud").strip().lower() != "cloud":
                _skip("backend != cloud")
                return
            clean = (text or "").strip()
            if len(clean) < 2:
                _skip("text too short")
                return
            print(f"TTS fire: {len(clean)} chars (voice={getattr(cfg,'voice','?')})",
                  file=sys.stderr, flush=True)
            # OpenAI TTS hard limit is 4096 chars per request.
            if len(clean) > 4096:
                clean = clean[:4096]
            worker = threading.Thread(
                target=self._run_tts_synthesize,
                args=(clean,),
                name="iris-tts-synth",
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception("tts_dispatch_failed", exc)
                except Exception:
                    pass

    def _run_tts_synthesize(self, text: str) -> None:
        """Worker thread body: hit OpenAI TTS, push PCM into the player.
        All exceptions are caught — TTS failure must NEVER take down the
        background worker or break the chat reply."""
        try:
            cfg = self._config
            api_key = getattr(cfg, "api_key", None) if cfg is not None else None
            if not api_key:
                return
            # Voice may have been mutated by the picker UI since startup;
            # read it fresh each call.
            voice = (getattr(cfg, "voice", None) or "marin").strip() or "marin"
            # The user picks ONE voice via the voice-picker UI and
            # that voice is used for every reply, every persona.
            # Personas customize TONE via TTS instructions, NOT voice
            # ID. (Previous design swapped voice per persona; user
            # explicitly asked for the one-voice model.) Tone variation
            # comes from:
            #   1. Persona base instructions (butler / playful / etc.)
            #   2. Per-reply emotion hint from emotion_tagger
            persona_instructions: Optional[str] = None
            persona_name = "default"
            try:
                from . import persona_voice as _pv
                _tts = _pv.tts_config()
                persona_instructions = (_tts.get("instructions")
                                         or None)
                try:
                    persona_name = _pv.active_preset().name
                except Exception:
                    persona_name = "default"
            except Exception:
                persona_instructions = None
            # JARVIS-grade tonal adaptation: classify the EMOTION of
            # this specific reply (LLM-first, heuristic fallback)
            # and merge a delivery hint into the persona's base
            # instructions. Same persona, different tone per moment
            # — apologetic when something fails, dry when it's a
            # quip, urgent for a warning, etc.
            try:
                from . import emotion_tagger as _et
                tag = _et.classify(text, persona=persona_name)
                if persona_instructions:
                    persona_instructions = _et.combine_instructions(
                        persona_instructions, tag.emotion)
                else:
                    persona_instructions = _et.delivery_hint(
                        tag.emotion)
                if self._logger:
                    try:
                        self._logger.event(
                            "tts_emotion_tagged",
                            emotion=tag.emotion.value,
                            confidence=round(tag.confidence, 2),
                            source=tag.source)
                    except Exception:
                        pass
            except Exception:
                pass
            # Rewrite into natural casual speech BEFORE synthesis. UI still
            # shows the original (assistant_text was emitted earlier); only
            # the audio path uses the humanized form. Falls back to the
            # original text on short input, timeout, or any error.
            speech_text = self._humanize_for_speech(text)
            # Cache key includes voice + (truncated) instructions so a
            # persona swap doesn't replay the prior preset's PCM.
            instr_tag = ""
            if persona_instructions:
                import hashlib as _hl
                instr_tag = _hl.sha1(
                    persona_instructions.encode("utf-8")
                ).hexdigest()[:8]
            cache_key = f"{voice}|{instr_tag}|{speech_text}"
            pcm: Optional[bytes] = None
            with self._tts_lock:
                pcm = self._tts_cache.get(cache_key)
            player = self._audio_player
            if pcm is not None:
                # Cache hit — write the full blob in one go (instant).
                if player is None or not getattr(player, "is_enabled", lambda: False)():
                    return
                player.write(pcm)
            else:
                # Cache miss — STREAM chunks directly into the player as
                # they arrive from OpenAI so playback can start before the
                # full response has finished downloading. Cuts perceived
                # latency from ~3-5s (full-response wait) to ~0.5-1s for
                # long replies. Also accumulates bytes for caching.
                buf = bytearray()
                def _on_chunk(chunk: bytes) -> None:
                    if not chunk:
                        return
                    buf.extend(chunk)
                    p = self._audio_player
                    if p is None or not getattr(p, "is_enabled", lambda: False)():
                        return
                    try:
                        p.write(chunk)
                    except Exception:
                        pass
                self._http_openai_tts(api_key, voice, speech_text,
                                      on_chunk=_on_chunk,
                                      instructions=persona_instructions)
                if not buf:
                    return
                pcm = bytes(buf)
                with self._tts_lock:
                    if len(self._tts_cache) >= self._tts_cache_max:
                        try:
                            oldest = next(iter(self._tts_cache))
                            self._tts_cache.pop(oldest, None)
                        except StopIteration:
                            pass
                    self._tts_cache[cache_key] = pcm
            if self._logger:
                try:
                    self._logger.event("tts_synthesized",
                                        chars=len(text),
                                        spoken_chars=len(speech_text),
                                        humanized=(speech_text != text),
                                        bytes=len(pcm),
                                        voice=voice,
                                        cached=False)
                except Exception:
                    pass
        except Exception as exc:
            if self._logger:
                try:
                    self._logger.exception("tts_synthesis_failed", exc)
                except Exception:
                    pass

    # Fallback voice mapping for legacy TTS models (tts-1 / tts-1-hd) that
    # don't support the 2025-era voices. Keys are 2025 voices that may 400;
    # values are the closest tonal match available on legacy models.
    _TTS_VOICE_FALLBACK = {
        "marin": "sage",    # warm, conversational, female-leaning
        "cedar": "onyx",    # deeper, masculine
        "ballad": "sage",
        "verse": "sage",
        "ash": "onyx",
    }

    # Preferred TTS model. gpt-4o-mini-tts is the only model that supports
    # marin/cedar and the `instructions` parameter; tts-1 is the legacy
    # fallback when the new model is unavailable in the deployment region.
    _TTS_MODEL_PRIMARY = "gpt-4o-mini-tts"
    _TTS_MODEL_FALLBACK = "tts-1"

    def _http_openai_tts(self, api_key: str, voice: str, text: str,
                          on_chunk: Optional[Callable[[bytes], None]] = None,
                          instructions: Optional[str] = None) -> bytes:
        """POST to /v1/audio/speech and return raw PCM16 24 kHz mono bytes.
        Returns b'' on any error so the caller can early-exit cleanly.

        When `on_chunk` is provided, the response is streamed in 8 KB
        chunks and passed to the callback as they arrive — letting the
        caller START PLAYBACK before the full response finishes
        downloading. Cuts perceived audio latency dramatically for
        long replies. With no callback, behaves as before (full read).

        Voice consistency rule: the user-configured voice (marin by
        default) MUST be used for every reply, otherwise mixing two
        voices in one session sounds unprofessional. So transient
        failures (timeout / 429 / 5xx) retry the SAME (model, voice)
        with brief backoff before any fallback. Only HARD failures
        (400/404 on the voice itself) move on to the fallback voice.

        Strategy: gpt-4o-mini-tts + requested voice → retry on
        transient. Only on HARD voice-rejection move to fallback
        voice. Only if the whole model is rejected drop to tts-1.

        response_format='pcm' returns exactly the format AudioPlayer
        consumes — 24 kHz mono signed-16 little-endian, no header — so we
        can feed the bytes straight into .write() without decoding."""
        # Build the attempt sequence: (model, voice) pairs in priority order.
        fallback_voice = self._TTS_VOICE_FALLBACK.get(voice, voice)
        attempts = [
            (self._TTS_MODEL_PRIMARY, voice),
        ]
        if fallback_voice != voice:
            attempts.append((self._TTS_MODEL_PRIMARY, fallback_voice))
        attempts.append((self._TTS_MODEL_FALLBACK, fallback_voice))

        # Per-attempt transient-retry budget. Tight enough that a real
        # outage doesn't block the user for long, generous enough that
        # a one-off rate-limit blip doesn't flip the voice to sage.
        _TRANSIENT_RETRIES = 2
        _BACKOFF_S = 0.4

        last_err: Optional[str] = None
        for model, attempt_voice in attempts:
            try:
                import urllib.request
                import urllib.error
                payload_dict = {
                    "model": model,
                    "input": text,
                    "voice": attempt_voice,
                    "response_format": "pcm",
                }
                # gpt-4o-mini-tts supports an `instructions` field that
                # nudges delivery style. Persona-aware: when the caller
                # passes per-preset instructions (Jarvis butler, playful,
                # concise, etc.), use those — they're the actual TONE
                # change that makes a persona swap audible. Falls back
                # to a generic "smart friend" instruction otherwise.
                if "gpt-4o-mini-tts" in model:
                    if instructions:
                        payload_dict["instructions"] = (
                            instructions
                            + " | Never read URLs aloud — say 'I've got "
                            "the link' or skip the URL. Read dates as "
                            "'tomorrow' / 'Thursday'. If two values are "
                            "obviously the same ('65 feels like 65'), "
                            "say it once."
                        )
                    else:
                        payload_dict["instructions"] = (
                            "Read like a smart friend sitting next to "
                            "the listener — natural conversational "
                            "pace, contractions, small pauses at "
                            "commas. Never read URLs aloud — say "
                            "'I've got the link' or skip the URL. "
                            "Read dates as 'tomorrow' / 'Thursday'. "
                            "If two values are obviously the same, "
                            "say it once."
                        )
                payload = json.dumps(payload_dict).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/audio/speech",
                    data=payload,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    if getattr(resp, "status", 200) != 200:
                        continue
                    if on_chunk is not None:
                        # Streaming mode — read 8 KB chunks and forward
                        # to the callback so playback starts well before
                        # the full response finishes downloading.
                        buf = bytearray()
                        while True:
                            chunk = resp.read(8192)
                            if not chunk:
                                break
                            buf.extend(chunk)
                            try:
                                on_chunk(chunk)
                            except Exception:
                                pass
                        data = bytes(buf)
                    else:
                        data = resp.read() or b""
                    if data and self._logger:
                        try:
                            self._logger.event("tts_http_ok",
                                                model=model,
                                                voice=attempt_voice,
                                                streamed=(on_chunk is not None),
                                                fallback=(attempt_voice != voice
                                                          or model != self._TTS_MODEL_PRIMARY))
                        except Exception:
                            pass
                    return data
            except urllib.error.HTTPError as exc:
                code = getattr(exc, "code", 0)
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last_err = f"{code}: {body}"
                if self._logger:
                    try:
                        self._logger.event("tts_http_error",
                                            code=code,
                                            body=body,
                                            model=model,
                                            voice=attempt_voice)
                    except Exception:
                        pass
                # 401/403 (auth) won't be fixed by a retry; bail.
                if code in (401, 403):
                    return b""
                # 429 / 5xx are transient — retry the SAME (model, voice)
                # before falling through to a different voice. This keeps
                # the user-configured voice (marin) intact across blips.
                if code == 429 or 500 <= code < 600:
                    import time as _time
                    for retry in range(_TRANSIENT_RETRIES):
                        _time.sleep(_BACKOFF_S * (retry + 1))
                        try:
                            with urllib.request.urlopen(req, timeout=15.0) as resp2:
                                if getattr(resp2, "status", 200) == 200:
                                    if on_chunk is not None:
                                        buf2 = bytearray()
                                        while True:
                                            chunk = resp2.read(8192)
                                            if not chunk:
                                                break
                                            buf2.extend(chunk)
                                            try:
                                                on_chunk(chunk)
                                            except Exception:
                                                pass
                                        return bytes(buf2)
                                    return resp2.read() or b""
                        except Exception:
                            continue
                    # Retries exhausted: still skip to next (model, voice)
                    # rather than return silence — partial fallback beats
                    # nothing, and the loud log above flags the issue.
                continue
            except Exception as exc:
                if self._logger:
                    try:
                        self._logger.exception("tts_http_failed", exc)
                    except Exception:
                        pass
                last_err = str(exc)
                # Network-level transient — retry the same combo before
                # moving on, same reasoning as above.
                import time as _time
                for retry in range(_TRANSIENT_RETRIES):
                    _time.sleep(_BACKOFF_S * (retry + 1))
                    try:
                        with urllib.request.urlopen(req, timeout=15.0) as resp2:
                            if getattr(resp2, "status", 200) == 200:
                                if on_chunk is not None:
                                    buf2 = bytearray()
                                    while True:
                                        chunk = resp2.read(8192)
                                        if not chunk:
                                            break
                                        buf2.extend(chunk)
                                        try:
                                            on_chunk(chunk)
                                        except Exception:
                                            pass
                                    return bytes(buf2)
                                return resp2.read() or b""
                    except Exception:
                        continue
                continue
        return b""

# Author: Konstantin Markov
