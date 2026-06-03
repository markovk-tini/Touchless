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

from PySide6.QtCore import QObject, Signal

from . import cortex_emit
from .audio_stream import AudioStream
from .config import LiveApiConfig, load_config
from .live_api_logger import LiveApiLogger
from .realtime_client import RealtimeClient
from .schemas import all_tool_schemas
from .screen_context import ScreenContext
from .tool_executor import ToolExecutor
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
    "HARD RULE — NEVER FABRICATE TOOL DATA: when you report what a tool "
    "returned (emails, files, search results, calendar events, weather "
    "values, ANY structured data), every name, subject, number, count, "
    "date, file path, and quoted detail you mention MUST come literally "
    "from that tool's response. If the response is empty or the field is "
    "missing, say so plainly ('no unread emails', 'nothing matched') — "
    "do NOT invent plausible-sounding examples to fill the gap. If a tool "
    "returned 7 messages, report 7; if it returned 0, report 0; never "
    "guess. Your conversational tone (warm, witty, Jarvis-style) applies "
    "to HOW you phrase real facts, never to inventing facts. Hallucinated "
    "data is the worst failure mode — when in doubt, quote less, not more.\n"
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
        )
    else:
        email_actions = (
            "EMAIL ACTIONS: to SEND an email use email_send — it opens the "
            "pre-filled draft and presses the mail send shortcut for the user "
            "(it confirms first). For a draft only (no send), use "
            "outlook_compose. Always include a concise subject. Call the send "
            "ONCE — if it reports it couldn't send, tell the user the draft is "
            "open to send manually; never retry in a loop. "
        )
    email_clause = email_identity + email_actions
    return (
        SYSTEM_INSTRUCTIONS
        + "\n\nMACHINE PATHS (use these EXACT paths; never guess the username): "
        f"Username {user}; Home {home}; Documents {docs}; Desktop {desktop}. "
        "Pass one (or a subfolder you made) as `base_dir` for file tools.\n"
        "APPS: open_app names — code, chrome, spotify, notepad, explorer; retry "
        "a shorter/lowercase name on error. Open an app ON a monitor in ONE "
        "call: open_app(name, monitor='secondary', placement='maximize') — it "
        "launches, waits for the window, and places+focuses it ('open Discord "
        "on my second monitor').\n"
        "CONNECTORS (FAST API PATH — prefer these): for an action ON a specific "
        "app/service, prefer the dedicated API tool when one exists — it's one "
        "call, no screenshots or clicking. These are available directly when "
        "set up: volume_* (system volume), discord_* (Discord voice), "
        "youtube_* (a YouTube tab), chrome_* (browser), and the email/Google "
        "tools below. Only control the app on-screen (click/type) when no such "
        "tool is available for the task.\n"
        "READING THE SCREEN: to READ or SUMMARIZE on-screen text — a "
        "document, a chat, a web page — call read_screen (accurate OCR text, "
        "cheap, no image). Use get_screen_context (a picture) only to SEE "
        "layout or find something to click. If content is cut off below the "
        "fold (a long inbox/doc), call read_screen with scroll_passes (e.g. "
        "6) so it scrolls and reads ALL of it — don't say items may be "
        "hidden, just scroll.\n"
        "EMAIL READS / SUMMARIES (HARD RULE — email_summary FIRST, "
        "screen-read last): for ANY 'read/summarize/check/show my "
        "email(s) / inbox / unread' request, ALWAYS call email_summary "
        "FIRST — it is the PRIMARY tool for this. It auto-cascades "
        "Gmail → Microsoft (Outlook) → helpful fallback, so it always "
        "finds the user's real inbox whether it lives in Gmail or "
        "Outlook. Call it with unread_only=true AND max=50 (the cap) "
        "so you get every unread message. DO NOT call gmail_list or "
        "ms_mail_list directly for general 'summarize my unread' "
        "requests — they only check ONE account and will return 0 if "
        "the user's real inbox lives in the other account. Use "
        "gmail_list / ms_mail_list ONLY for account-specific SEARCHES "
        "with a query (e.g. 'emails from boss@example.com', 'find my "
        "Gmail message about the invoice'). FINAL REPLY: email_summary "
        "returns a `summary` field containing a faithful, "
        "deterministically-rendered summary of the real messages (built "
        "directly from the message array — every sender, subject, and "
        "snippet is real). EMIT `result.summary` VERBATIM as your reply — "
        "do not rephrase, do not re-summarize, do not re-rank, do not "
        "invent or substitute different senders/subjects. Writing your "
        "own summary from `result.messages` is BANNED here — you have "
        "hallucinated demo emails (Carl, Sarah, Mark, IT Support, etc.) "
        "in the past when given that freedom. Only if `result.summary` "
        "is missing or empty, fall back to summarizing from "
        "`result.messages` and quote sender/subject/snippet fields "
        "VERBATIM. If `result.count` is 0 and `result.source` is "
        "'none', no email connector is wired — fall back to "
        "open_app('outlook') + read_screen. Do NOT default to opening "
        "Outlook + read_screen when a connector is available. Do NOT "
        "compose a new email when asked to READ one. To read ONE "
        "email's full body via the connector pass include_body=true on "
        "email_summary (or gmail_read with the id) instead of clicking "
        "into the message in a window.\n"
        "PERSONAL TEAMS / APPS WITH NO API: teams_send/teams_channel_post work "
        "only for work/school Microsoft accounts. For a PERSONAL Teams account "
        "(or any app the connectors don't cover), DRIVE THE APP BY SCREEN: "
        "open_app('Teams'), read_screen to see it, click the search / 'New "
        "chat' box (click_text_on_screen), type_text the person's name, open "
        "their chat, click the message box, type_text the message, then press "
        "Enter (or click Send). Confirm with the user before sending. Never say "
        "you can't message someone — fall back to this GUI path.\n"
        "GUI CLICKING — BE EFFICIENT: call read_screen ONCE; it returns "
        "clickable_elements with screen-pixel x,y. Click them with "
        "click_screen(coordinate_space='screen', x=<px>, y=<py>) — exact, no "
        "guessing, no OCR retries. To fill a box AND send in ONE step, use "
        "click_type(x, y, coordinate_space='screen', text='...', submit=true) "
        "instead of separate click+type+send turns (fewer round trips, avoids "
        "rate limits). Do NOT call get_screen_context (a costly image) or "
        "re-read between actions; reuse the one read_screen result and only "
        "re-read if the screen actually changed.\n"
        + email_clause +
        "MICROSOFT ACCOUNTS: several can be connected at once. ms_* tools act "
        "on the ACTIVE one. If the user wants a different one (e.g. their school "
        "account for Teams/work mail), call ms_use_account('<email or name>') "
        "first; ms_list_accounts shows what's connected.\n"
        "FILES: open files/folders by NAME with open_path (don't ask where) — "
        "status='ambiguous'→list+ask, not_found→retry deep=true; open several "
        "at once via its `queries` list. read_file(name) returns a file's text "
        "(incl. PDF/docx) so you can SUMMARIZE without opening it ('summarize / "
        "what's in / read me X'). When the user names a file by DESCRIPTION not "
        "exact name ('my most recent screenshot', 'the three latest touchless "
        "drawings', 'the newest file on my desktop'), call list_files FIRST "
        "(name_contains + optional file_type; OMIT folder to search everywhere "
        "if unsure where it is — spaces/underscores match either way) and use "
        "the top result paths — results are newest-first, so 'the N latest X' = "
        "the first N. NEVER guess/invent a path for move_file/read_file/"
        "open_path, and don't fall back to open_path just to FIND a file. "
        "Build files with create_folder, create_file/"
        "write_file/append_file (base_dir = a real path above). To open a "
        "folder/PROJECT in VS Code (the user says 'in VS Code' / 'the project'), "
        "use open_in_editor(name) — that opens the EDITOR; open_path opens it in "
        "File Explorer instead, so don't use open_path for 'open … in VS Code'. "
        "Use the EXACT project name (don't shorten it); open_in_editor finds the "
        "folder or asks if it can't. If open_in_editor errors, RETRY "
        "open_in_editor (with the exact name/path) — do NOT fall back to "
        "open_path (that just opens File Explorer, the wrong thing).\n"
        "WINDOWS: move_window_to_monitor(window_title, monitor, placement) — "
        "placement is maximize/center OR a snap left/right/top/bottom/quadrant. "
        "For 'move/drag a window to the right/left side of monitor X' use "
        "placement='right'/'left' on that monitor — do NOT pixel-drag the title "
        "bar. Pass a SHORT generic title matching "
        "the user's words ('file explorer', or a folder name), NOT a full "
        "guessed title; status='ambiguous'→list the names and ask which, then "
        "call again. control_window(window_title, action=close/minimize/"
        "maximize/restore/focus) acts on that window IN PLACE — its maximize "
        "does NOT move it to another monitor. So for 'open X (maximized) on "
        "monitor 2' use open_app('X', monitor='secondary', placement='maximize') "
        "(or, if already open, move_window_to_monitor) — NOT control_window. "
        "NEVER Alt+F4 to close — it hits the focused window (often the "
        "assistant).\n"
        "DESKTOP DIALOGS / native apps: read_ui lists a window's controls; "
        "click_ui(name) clicks one; set_field types into a field. React to a "
        "prompt that appears later with wait_and_click('Yes'); for a keystroke "
        "prompt ('1. Yes / 2. No') use wait_and_press(text='proceed', "
        "keys=['1']).\n"
        "TYPING INTO AN APP (e.g. 'open Notepad, type X, save as Y'): open_app("
        "'notepad') → type_text(text, window_title='Notepad') → press_hotkey("
        "['ctrl','s'], window_title='Notepad') → in the Save dialog set_field("
        "'File name', 'Y', window_title='Save As') then wait_and_click('Save', "
        "window_title='Save As'). ALWAYS pass window_title to type_text/"
        "press_hotkey — the Iris chat is always-on-top, so untargeted keystrokes "
        "type into the wrong window (nothing lands in the app).\n"
        "AUTO-APPROVE: to keep approving an agent's prompts ('keep clicking yes "
        "/ watch my Claude tab / approve while I'm away'), call "
        "auto_approve(window_title='Visual Studio Code') — a background watcher "
        "(focus-independent) that clicks Yes/Allow/Keep; stop_auto_approve to "
        "stop. If the user just wants watching (no new task), call it directly. "
        "Always pass window_title so it stays locked on the right window.\n"
        "CODING AGENTS (Claude/Codex in VS Code): send_to_coding_agent(prompt=<"
        "task>, agent='claude'|'codex'). ACTUALLY call the tool — never write "
        "its JSON as a chat message. It opens a FRESH tab itself and types "
        "there, and auto-starts the approve-watcher — so call it EXACTLY ONCE, "
        "prompt only (no assume_focused, no pre-opening, no screenshot to "
        "'verify'; never type 'claude'/'codex' yourself). project_folder is for "
        "creating a BRAND-NEW project ('make a folder called iris demo') — it "
        "creates the folder under Documents and opens it. Do NOT use "
        "project_folder for an EXISTING project (it would create a duplicate); "
        "instead open_in_editor(name) to open the existing one, THEN "
        "send_to_coding_agent(open_only=true) for a Claude tab (or with a "
        "prompt if there's a task). To run+verify, "
        "end the prompt with 'run it, fix errors, re-run until it works, report "
        "the output' (Claude sees its own terminal). Then just say it's working "
        "and relay the watcher's done-note. follow_up_coding_agent(message) "
        "sends a follow-up into the SAME tab (e.g. paste output back) — never "
        "open a new tab for a follow-up. Codex: if the tab flow misses, "
        "via='terminal'. background=true ONLY if asked to run it without "
        "watching (it hands focus back). If the user only asks to OPEN a Claude "
        "tab (no task for Claude), call send_to_coding_agent(open_only=true) — "
        "do NOT invent a prompt or tell Claude to do anything.\n"
        "CROSS-APP RUN (e.g. MATLAB): send_to_coding_agent(project_folder='X', "
        "prompt to create main.m doing Y), then run_matlab_script(folder/"
        "main.m) — MATLAB opens and runs it, plots appear. Don't ask Claude to "
        "test MATLAB. Same idea elsewhere: have Claude write the file, then run "
        "it in the real app.\n"
        "GAMES: prefer a direct launch — open_app('valorant'/'genshin') boots "
        "the game. If it stops at a launcher PLAY/Start button, "
        "click_text_on_screen('PLAY' or 'Start Game', timeout_sec=60, "
        "double=true, retries=3) (polls, double-clicks the unfocused window, "
        "re-clicks if the layout shifted; verified=false → tell the user to "
        "click it). Launcher UIs are Chromium — read_ui/click_ui don't work.\n"
        "UAC / Windows security prompts: you CANNOT click them (secure desktop). "
        "When launching something that elevates (game launcher, installer), "
        "warn in one line ('click Yes if a Windows prompt appears, I can't') "
        "and keep waiting for the app — don't fail or try to click it.\n"
        "WEB: use web_* (never pixel-click links). web_navigate(url or query) "
        "opens/searches AND opens the browser itself (don't also open_app "
        "chrome). To SEARCH AND SUMMARIZE, do exactly TWO calls: "
        "web_navigate(query) → web_get_text, then write the summary. Do NOT "
        "open individual result links, click around, or call web_get_links — "
        "the results page already has the headlines; opening a link is slower "
        "and only needed if the user explicitly asks to open a result. Other "
        "tools when needed: web_get_links lists {index,text,url}; web_scroll; "
        "web_fill(label,text); web_wait_for; web_click(text); web_eval(js). To "
        "open the Nth result (only if asked): web_get_links → web_navigate(its "
        "url). web_* drive a dedicated debug Chrome (not the user's everyday "
        "one). A page may take 1-3s; if blank, retry once.\n"
        "run_quick_command: MEDIA / built-in voice commands ONLY ('play X on "
        "spotify', 'next song', 'pause') — not for web/clicking/windows.\n"
        "MULTI-STEP: a request with several actions is USUALLY auto-split and "
        "fed to you ONE sub-task at a time ('[Step N of M] do ONLY this…'). "
        "Finish that ONE task fully (right tool, confirm it actually worked), "
        "give a one-line confirmation (which ends the step — the next is sent "
        "automatically), and don't do other steps in it. If a step fails or is "
        "ambiguous, retry or ask — never skip it.\n"
        "MULTI-ACTION FALLBACK: occasionally a multi-action request slips "
        "through WITHOUT '[Step N of M]' wrapping (e.g. 'write me a haiku "
        "about coffee and add a task to drink some'). When that happens, do "
        "ALL the actions yourself in sequence — call each tool in turn, "
        "don't stop after the first verb. Treat 'X and Y', 'X, then Y', "
        "and comma-separated verb lists as multiple actions you must each "
        "complete, not as a single action."
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


class LiveApiManager(QObject):
    state_changed = Signal(object, str)        # (LiveApiState, status text)
    error_occurred = Signal(str)
    transcript_received = Signal(str)          # user speech transcript text
    assistant_text = Signal(str)               # assistant text deltas/snippets
    assistant_message_break = Signal()         # start a fresh assistant bubble
    tool_event = Signal(str, dict)             # ("called"/"completed", info)

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
        self._last_user_text = ""     # for realtime fact-extraction observation
        self._memory_summary_sent = False  # send memory note once per session
        self._failed_retries = 0      # retries used for a failed response turn
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

    def is_running(self) -> bool:
        return self._state not in (LiveApiState.OFF, LiveApiState.ERROR)

    def start(self) -> None:
        with self._lock:
            if self._state not in (LiveApiState.OFF, LiveApiState.ERROR):
                return
            self._teardown_in_progress = False
            self._response_active = False
            self._pending_notes = []
            self._response_requested = False
            self._multistep_active = False
            self._nudge_count = 0
            self._turn_text = ""
            self._failed_retries = 0
            self._last_nudge_ts = 0.0
            self._task_queue = []
            self._in_queue = False
            self._planning = False
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
        Returns False if the client isn't in a state to send."""
        client = self._client
        if client is None or not getattr(client, "connected", False):
            return False
        if self._state in (LiveApiState.OFF, LiveApiState.ERROR, LiveApiState.CONNECTING):
            return False
        try:
            if prefix_notes:
                client.send_text_message("\n\n".join(prefix_notes))
            client.request_response()
            return True
        except Exception as exc:
            if self._logger:
                self._logger.exception("request_response_failed", exc)
            return False

    def _request_model_response(self) -> None:
        """Ask for a model turn. If a response is already active (e.g. this is
        one of several parallel tool calls, or the original response that
        emitted the call hasn't ended yet), remember it and fire once on
        response.done instead of colliding."""
        with self._lock:
            if self._response_active:
                self._response_requested = True
                return
            self._response_active = True  # optimistic; cleared on response.done
        if not self._fire_response_create():
            with self._lock:
                self._response_active = False

    def _drain_pending(self) -> None:
        """Called on response.done. Fire a single follow-up response.create if
        anything was deferred while the just-finished response was active —
        coalescing all queued notes and any deferred request into ONE turn."""
        with self._lock:
            notes = self._pending_notes
            requested = self._response_requested
            if not notes and not requested:
                return
            self._pending_notes = []
            self._response_requested = False
            self._response_active = True  # optimistic; cleared on next done
        if not self._fire_response_create(prefix_notes=notes):
            with self._lock:
                self._response_active = False

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
        note = f'(iris planner handled: "{ut}". tools: {", ".join(tool_parts) or "none"}.'
        if facts:
            # Cap fact list so the note stays small.
            note += f" facts: {'; '.join(facts[:6])}."
        rep = (reply or "").strip()
        if rep:
            if len(rep) > 200:
                rep = rep[:200] + "..."
            note += f" reply to user: {rep}"
        note += ")"
        return note[:600]

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
            self._response_active = False
            self._pending_notes = []
            self._response_requested = False
            self._multistep_active = False
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
                        _QT_router.singleShot(300, _step2)
                        _QT_router.singleShot(600, _step3)
                        _QT_router.singleShot(1000, _step4)
                    except Exception:
                        pass
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
                handled = self._iris_planner.try_handle(text)
            except Exception as exc:
                if self._logger:
                    self._logger.exception("iris_planner_unhandled", exc)
                handled = None
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
        ok = bool(client.send_text_message(text))
        if ok:
            # Track for fact-extraction on response.done. Only capture turns
            # that ACTUALLY went to realtime — planner-handled requests
            # already record themselves via the orchestrator's _record_turn.
            self._last_user_text = text
            self._request_model_response()
        return

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
        if self._state != LiveApiState.OFF:
            # Spontaneous close — surface as error so user can retry.
            self._set_state(LiveApiState.ERROR, f"Connection closed ({reason or 'unknown'})")

    def _on_ws_error(self, message: str) -> None:
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
            if self._logger:
                self._logger.text("transcript_user", transcript)
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
            self._response_active = True
            self._turn_text = ""  # reset per-response assistant text
            # Defensive: clear any stale summary override that didn't get
            # consumed on the prior response (shouldn't happen in normal
            # flow, but a malformed event stream could leave one set).
            self._pending_override_text = None
            self._last_override_tool = ""
            # Each new response is a NEW reply — start a fresh chat bubble so
            # replies don't concatenate into one growing box.
            self.assistant_message_break.emit()
            self._set_state(LiveApiState.THINKING, "Thinking")
            return

        if kind == "response.done":
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

        executor = self._executor
        client = self._client
        if executor is None or client is None:
            return

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
                output = {"status": "cancelled", "code": "user_declined",
                          "error": "User declined; not sent."}
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
        self.tool_event.emit("completed", {"name": name, "call_id": call_id, "status": output.get("status"), "source": source})

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
        try:
            if isinstance(output, dict) and name in self._DETERMINISTIC_SUMMARY_TOOLS:
                summary_val = output.get("summary")
                if isinstance(summary_val, str) and summary_val.strip():
                    self._pending_override_text = summary_val.strip()
                    self._last_override_tool = name
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
                    # tokens for text we'll discard. Safe even if the
                    # model has not yet started streaming text: the server
                    # treats response.cancel as a no-op when no response
                    # is active.
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
        # Schedule each leg with a delay so the user sees neurons
        # TRAVELING through the path. Also flip core to SPEAKING
        # (purple) at the start, then back to LISTENING (blue) after
        # the path completes — so the core's color matches what's
        # actually happening (delivering output).
        from PySide6.QtCore import QTimer as _QT
        def _later(ms, fn):
            try:
                _QT.singleShot(ms, fn)
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
            # Rewrite into natural casual speech BEFORE synthesis. UI still
            # shows the original (assistant_text was emitted earlier); only
            # the audio path uses the humanized form. Falls back to the
            # original text on short input, timeout, or any error.
            speech_text = self._humanize_for_speech(text)
            # Cache key uses the humanized text so repeat rewrites that
            # collapse to the same spoken form reuse the same PCM bytes.
            cache_key = f"{voice}|{speech_text}"
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
                                      on_chunk=_on_chunk)
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
                          on_chunk: Optional[Callable[[bytes], None]] = None) -> bytes:
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
                # nudges delivery style. Push toward casual conversation
                # — closer to how a person actually replies, less robotic.
                if "gpt-4o-mini-tts" in model:
                    payload_dict["instructions"] = (
                        "Voice: Jarvis-style personal assistant — warm, "
                        "intelligent, a touch witty. Speak the way a "
                        "smart friend sitting next to the listener would "
                        "read this aloud: natural sentences, contractions, "
                        "small pauses at commas, slight smile when it fits, "
                        "no flat monotone, no robotic list-reading. Don't "
                        "spell out URLs or dates digit by digit; phrase "
                        "them naturally."
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
