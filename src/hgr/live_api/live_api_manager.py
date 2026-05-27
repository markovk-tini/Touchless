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
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from .audio_stream import AudioStream
from .config import LiveApiConfig, load_config
from .live_api_logger import LiveApiLogger
from .realtime_client import RealtimeClient
from .schemas import all_tool_schemas
from .screen_context import ScreenContext
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .user_prefs import get_default_email


SYSTEM_INSTRUCTIONS = (
    "You are Touchless Live Agent, controlling the user's own Windows PC via "
    "tools. Use a tool for every action; never say something is done unless the "
    "tool result confirms it, and never claim a window/app is in some state "
    "without actually doing it. Screenshots show the FULL multi-monitor "
    "desktop — read the image; your own 'Touchless' window is not the target, "
    "ignore it. If unsure of screen state, get_screen_context first.\n"
    "LANGUAGE (hard rule): ALWAYS respond in English, regardless of any single "
    "letters, short inputs, or odd tokens you receive — never switch to another "
    "language unless the user clearly writes to you in that language.\n"
    "TONE (hard rule): by default reply in ONE short sentence, ≤15 words, like a "
    "quick assistant — 'On it.', 'Done — it printed 5.', 'Valorant's "
    "launching.', 'Couldn't find PLAY yet.' No paragraphs, lists, or "
    "step-by-step narration. Expand to more than a sentence ONLY when the user "
    "explicitly asks to explain/why, then go short again.\n"
    "SAFETY: confirm before destructive/risky actions; never expose secrets; "
    "never bypass security, DRM, anti-cheat, CAPTCHA, or non-skippable ads.\n"
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
        + email_clause +
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
        "MULTI-STEP: a request with several actions is auto-split and fed to "
        "you ONE sub-task at a time ('[Step N of M] do ONLY this…'). Finish "
        "that ONE task fully (right tool, confirm it actually worked), give a "
        "one-line confirmation (which ends the step — the next is sent "
        "automatically), and don't do other steps in it. If a step fails or is "
        "ambiguous, retry or ask — never skip it."
    )


class LiveApiState(enum.Enum):
    OFF = "off"
    CONNECTING = "connecting"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    ERROR = "error"


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
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or load_config()
        # text_only=True means we don't open the mic / VAD pipeline; the
        # only inputs come from explicit `send_user_text` calls. Used by
        # the typed-command UI in Phase 1A. When voice comes back in
        # Phase 1B, leave this False.
        self._text_only = bool(text_only)
        self._logger: Optional[LiveApiLogger] = None
        self._client: Optional[RealtimeClient] = None
        self._audio: Optional[AudioStream] = None
        self._screen: Optional[ScreenContext] = None
        self._executor: Optional[ToolExecutor] = None
        self._registry: Optional[ToolRegistry] = None
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
            # Layer 0 router. Lazy-import keeps the manager loadable on
            # systems where Touchless's voice modules can't initialize
            # (e.g. headless CI without sounddevice).
            try:
                from .command_router import CommandRouter
                self._command_router = CommandRouter(logger=self._logger)
            except Exception as exc:
                self._logger.exception("command_router_init_failed", exc)
                self._command_router = None

            backend_kind = (self._config.backend or "cloud").strip().lower()
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
                )

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
            self._audio = None
            self._client = None
            self._screen_thread = None
            self._executor = None
            self._registry = None
            self._screen = None
            self._command_router = None
            # Don't drop the logger yet — we still want the stop events
            # written. Cleared once everything joined.
            self._screen_stop.set()
            self._screen_request.set()
            if logger is not None:
                logger.event("session_stop_requested")
            self._set_state(LiveApiState.OFF, "Off")

        # Heavy/joining work outside the lock.
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

    def request_screen_now(self) -> None:
        """Wake the screen worker so it captures and sends immediately."""
        self._screen_request.set()

    def send_user_text(self, text: str) -> bool:
        """Inject a typed user message into the current session.

        Routing flow:
          1. Echo the user's text in the chat (transcript_received).
          2. Try Layer 0 router — if it matches a known intent (open
             chrome, search X, play next song, ...), execute it
             instantly and DO NOT call the LLM. The chat shows what
             happened via tool_event signals.
          3. Otherwise forward to the backend (LLM agent loop) as before.

        UI-thread safe: returns True if the manager accepted the input;
        False otherwise so the UI can show a hint.
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
        self.transcript_received.emit(text)

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
                self.tool_event.emit("called", {"name": f"router/{action_label}", "info": routed.message, "source": "touchless"})
                self.tool_event.emit("completed", {"name": f"router/{action_label}", "status": "ok", "source": "touchless"})
                self.assistant_text.emit(routed.message or "Done.")
                self._set_state(LiveApiState.LISTENING, "Ready (type a command)")
                return True
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

        # ---- Layer 1: LLM agent (router didn't match) ----
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
            return ok
        ok = bool(client.send_text_message(text))
        if ok:
            self._request_model_response()
        return ok

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
                self.assistant_text.emit(routed.message or "Done.")
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
    )

    def _looks_multistep(self, text: str) -> bool:
        low = (text or "").lower()
        seq = low.count(" then ") + low.count(" and ") + low.count(", ")
        verbs = sum(1 for v in self._ACTION_VERB_STARTS if (v + " ") in low)
        return (seq >= 2 and verbs >= 2) or verbs >= 3 or len(text) > 160

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
            self._set_state(LiveApiState.LISTENING, "Listening")
            return

        if kind == "response.created":
            self._response_active = True
            self._turn_text = ""  # reset per-response assistant text
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
    _CONFIRM_BEFORE_TOOLS = {"gmail_send", "email_send"}

    def _confirm_connector_action(self, name: str, args: Dict[str, Any]) -> bool:
        """Ask the user before an irreversible connector action. Returns True
        to proceed. If no confirm UI is wired, proceed (the model + system
        instructions are the gate in that case)."""
        cb = self._confirm_callback
        if cb is None:
            return True
        if name in ("gmail_send", "email_send"):
            to = str((args or {}).get("to") or (args or {}).get("recipient") or "")
            subject = str((args or {}).get("subject") or "")
            title = f"Send this email now to {to}?"
            detail = f"Subject: {subject or '(none)'}"
        else:
            title, detail = f"Run {name}?", ""
        try:
            return bool(cb(title, detail))
        except Exception:
            return False

    def _current_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tools sent to the model: built-ins + the *available* connectors'
        tools, exposed DIRECTLY so a connector action is a single model turn.

        We intentionally do NOT route through a find_capability meta-tool: at
        this scale that added two extra model turns per action (discover →
        load → call), each re-processing the full ~50-tool context, which
        burned the Realtime tokens-per-minute budget and caused rate-limit
        pauses. Direct exposure is far cheaper here. (If the catalog ever
        grows huge — e.g. many MCP servers — revisit a lazy router.)"""
        if self._registry is not None:
            return self._registry.openai_tools()
        return list(all_tool_schemas())

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
        self.tool_event.emit("completed", {"name": name, "call_id": call_id, "status": output.get("status"), "source": source})

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

    def _emit_error(self, message: str) -> None:
        if self._logger:
            self._logger.error("session_error", message=message)
        self._set_state(LiveApiState.ERROR, f"Error: {message}")
        self.error_occurred.emit(message)

# Author: Konstantin Markov
