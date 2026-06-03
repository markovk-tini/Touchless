"""Touchless Assistant window — the in-app UI for the Live API agent.

A self-contained, modeless window that drives `LiveApiManager` and renders
an advanced chat transcript:

  * streaming assistant text (deltas append into the live bubble)
  * tool-call pills with live status (called -> ok/failed)
  * a "handled locally - 0 tokens" badge when the Layer 0 router catches a
    command, so the token-savings path is visible
  * a colored state header (Off / Connecting / Listening / Thinking /
    Executing / Error)
  * Start/Stop session control + a typed command box
  * a proper modal confirmation dialog for risky tools (marshalled from
    the websocket thread onto the GUI thread)

Phase 1b is text-only (`text_only=True`); voice activation and the
realtime conversation mode land in later phases. The module is kept out
of the giant main_window so it can be developed and tested in isolation:

    set PYTHONPATH=src
    set OPENAI_API_KEY=sk-...
    python -m hgr.app.ui.live_assistant_window
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import math

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QCursor, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...live_api.live_api_manager import LiveApiManager, LiveApiState
from .custom_gestures_chrome import apply_touchless_titlebar


class _ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` on left mouse press. Used for the
    "Iris" header so the user can open the Cortex visualization.

    Also emits ``rightClicked`` so callers can offer a secondary action
    (e.g. "open the standalone cortex pop-out" vs. "toggle the embedded
    side panel") without needing an extra button in the header row."""

    clicked = Signal()
    rightClicked = Signal()

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self.rightClicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


# Fallback palette (matches AppConfig defaults). Overridden per-instance
# from the passed config when available.
_DEFAULT_PALETTE = {
    "primary": "#0B3D91",
    "accent": "#1DE9B6",
    "surface": "#0F172A",
    "text": "#E5F6FF",
}

# State -> (label, color) for the header pill.
_STATE_STYLE = {
    LiveApiState.OFF: ("Off", "#64748B"),
    LiveApiState.CONNECTING: ("Connecting…", "#F59E0B"),
    LiveApiState.LISTENING: ("Ready", "#1DE9B6"),
    LiveApiState.THINKING: ("Thinking…", "#58E3FF"),
    LiveApiState.EXECUTING: ("Executing…", "#A855F7"),
    LiveApiState.ERROR: ("Error", "#EF4444"),
}


# Persist whether the chat speaks responses. Survives restarts.
_VOICE_ENABLED_SETTINGS_KEY = "live_api/voice_enabled"

# Persist whether the embedded Cortex panel was expanded when the user
# last closed Iris — reopening should honor their preference instead of
# always defaulting to the chat-only narrow layout.
_CORTEX_EXPANDED_SETTINGS_KEY = "live_api/cortex_expanded"

# Window geometry constants used by the expand/collapse animation. Kept
# as module-level constants so the warming/expand/collapse paths stay
# in lock-step — changing one place updates all three.
_CHAT_ONLY_WINDOW_W = 520
_CHAT_ONLY_WINDOW_H = 720
_EXPANDED_WINDOW_W = 1320
_EXPANDED_WINDOW_H = 780
_EXPANDED_CHAT_W = 440
_EXPANDED_CORTEX_W = 820
_CHAT_MIN_W_EXPANDED = 380
_CHAT_MAX_W_EXPANDED = 520


def _load_cortex_expanded(*, default: bool = False) -> bool:
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("Touchless", "Touchless")
        v = s.value(_CORTEX_EXPANDED_SETTINGS_KEY, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(v, int):
            return bool(v)
    except Exception:
        pass
    return default


def _save_cortex_expanded(expanded: bool) -> None:
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("Touchless", "Touchless")
        s.setValue(_CORTEX_EXPANDED_SETTINGS_KEY, bool(expanded))
        s.sync()
    except Exception:
        pass


def _load_voice_enabled(*, default: bool = True) -> bool:
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("Touchless", "Touchless")
        v = s.value(_VOICE_ENABLED_SETTINGS_KEY, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(v, int):
            return bool(v)
    except Exception:
        pass
    return default


def _save_voice_enabled(enabled: bool) -> None:
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("Touchless", "Touchless")
        s.setValue(_VOICE_ENABLED_SETTINGS_KEY, bool(enabled))
        s.sync()
    except Exception:
        pass


class LiveAssistantWindow(QWidget):
    """Modeless chat window wrapping a LiveApiManager (text-only)."""

    # Marshals a risky-tool confirmation from the websocket thread onto
    # the GUI thread. `holder` is a dict with an Event + result slot.
    _confirm_request = Signal(str, str, object)

    # Marshals Gmail-connect results from the OAuth worker thread to the UI.
    _gmail_result = Signal(bool, str)
    # Marshals Microsoft-365-connect results from its worker thread.
    _ms_result = Signal(bool, str)
    # Fired by the cortex payload-preload worker thread once the real
    # world payload has been loaded. Qt auto-marshals the slot call back
    # to the GUI thread (the window's owning thread), so the slot can
    # safely touch the QWebEngineView. Used to trigger a one-shot
    # view.reload() when the embedded cortex was built with a stale
    # / empty payload before the worker finished. No payload arg —
    # the slot reads self._embedded_cortex_payload directly.
    payloadReady = Signal()

    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._palette = dict(_DEFAULT_PALETTE)
        if config is not None:
            for key, attr in (
                ("primary", "primary_color"),
                ("accent", "accent_color"),
                ("surface", "surface_color"),
                ("text", "text_color"),
            ):
                val = getattr(config, attr, None)
                if val:
                    self._palette[key] = str(val)

        self.setWindowTitle("Touchless · Iris")
        self.setMinimumSize(520, 640)
        # Explicit initial size — setMinimumSize alone leaves the window at
        # Qt's default ~200x100 px until show() asks for layout, which is
        # what was flashing as a 'tiny window' before the layout settled.
        self.resize(520, 720)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        # Set the surface color via PALETTE (not just stylesheet) so Qt's
        # very first paint — which happens BEFORE stylesheets are processed —
        # already shows the dark Touchless background. Without this, the
        # window briefly flashes white between Windows drawing the chrome
        # and Qt processing setStyleSheet on the first paint cycle. Also
        # opt out of any system background so Qt strictly uses our palette.
        _surface = QColor(self._palette["surface"])
        _text = QColor(self._palette["text"])
        _qpal = QPalette()
        _qpal.setColor(QPalette.Window, _surface)
        _qpal.setColor(QPalette.Base, _surface)
        _qpal.setColor(QPalette.WindowText, _text)
        self.setPalette(_qpal)
        self.setAutoFillBackground(True)
        # Belt-and-suspenders: even with palette + early title bar styling,
        # Qt's very first paint cycle on Windows still briefly shows the
        # default chrome+white client area before our styles land. Hide the
        # window completely during that frame by starting opacity=0;
        # show_window() flips it back to 1.0 on the next event-loop tick
        # AFTER show() has fully painted, so the user only ever sees the
        # already-styled window appear.
        self.setWindowOpacity(0.0)

        # Voice output: load persisted toggle (default ON). Pass to
        # the manager so the session is created with audio output
        # enabled. Toggling the mute button only takes effect on the
        # NEXT session start (modalities change mid-session requires
        # a session reset on OpenAI's side).
        self._voice_enabled = _load_voice_enabled(default=True)
        self._manager = LiveApiManager(
            text_only=True, voice_output=self._voice_enabled)
        # Apply the persisted voice choice (if any) before anything that
        # snapshots the config — so the first session starts with it.
        try:
            from .voice_picker import load_saved_voice
            saved = load_saved_voice(default=getattr(self._manager.config, "voice", "sage"))
            self._manager.config.voice = saved
        except Exception:
            pass
        # Lazily-created Voice picker dialog (opens on toolbar click).
        self._voice_dialog: Optional[QWidget] = None
        self._current_assistant_label: Optional[QLabel] = None
        # call_id -> the QLabel showing that tool's status pill.
        self._tool_pills: dict = {}
        # Lazily-created Cortex visualization window (opens on header click).
        self._cortex_window: Optional[QWidget] = None
        # Typing-dots indicator state: shown ~250ms after the user sends
        # a prompt, cleared on the first response signal (assistant delta
        # / tool event / error). Without it the window APPEARS to freeze
        # while the planner / executor / realtime thinks. The 250ms delay
        # means quick replies (Layer-0 router / cached classifier hits)
        # never flash the indicator at all.
        self._typing_row: Optional[QWidget] = None
        self._typing_timer: Optional[QTimer] = None
        self._typing_step: int = 0
        # Per-dot widgets + opacity effects, populated by _show_typing.
        self._typing_dot_widgets: list = []
        self._typing_dot_effects: list = []
        # True between _on_send and the first response signal: blocks the
        # delayed _show_typing if a reply arrived inside the 250ms window.
        self._typing_pending: bool = False
        # Cortex thinking-pulse loop state. Set True in _on_send,
        # cleared by _on_assistant_delta on first reply.
        self._cortex_thinking_active: bool = False
        self._cortex_thinking_start_time: float = 0.0
        # Rotating index into _CORTEX_THINKING_RING so the scanning
        # pulse cycles through capabilities — reads as 'iris is
        # considering paths', not 'iris is firing tools repeatedly'.
        self._cortex_thinking_idx: int = 0
        # Current LiveApiState (mirrored from _on_state_changed). Used by
        # _reshow_typing_if_busy to decide whether to bring the dots back
        # between intermediate outputs (THINKING/EXECUTING = still
        # working; LISTENING = task done, stay hidden).
        self._current_state: Optional[LiveApiState] = None
        # Dynamic neuron-timing flag: True between _on_send (immediately
        # after the anticipatory pulses fire) and the first response
        # signal (assistant_text / tool_event / assistant_break / error
        # / state→LISTENING). While True, _fire_cortex_thinking_pulse
        # reschedules itself every ~600ms to keep neurons firing along
        # core→cap-tools so the visualization shows "thinking" instead
        # of flashing once and going silent for the rest of the latency
        # window. Capped at 30s (safety) in case all response signals
        # fail to arrive.
        self._cortex_thinking_active: bool = False
        self._cortex_thinking_start_time: float = 0.0

        # Lazy embedded Cortex (right-side splitter panel inside this
        # window). Distinct from `_cortex_window` (the pop-out modeless
        # CortexWindow) — both can coexist. None until the user
        # explicitly reveals it via the Iris header or _show_embedded_cortex().
        self._embedded_cortex_view: Optional[QWidget] = None
        self._embedded_cortex_bridge = None
        self._embedded_cortex_channel = None
        self._embedded_cortex_panel: Optional[QWidget] = None
        self._embedded_cortex_splitter: Optional[QSplitter] = None
        self._embedded_cortex_loaded: bool = False
        # Cached world payload — kept so refresh_cortex() can reuse
        # the latest snapshot without re-running the discovery scan.
        # Filled by a background worker thread started in show_window()
        # so payload loading (memory queries, gmail/ms365 fetches, fs
        # scans, embedder calls) never blocks the GUI thread. _warm and
        # the embed builders use whatever is in the cache; if the
        # worker hasn't finished, they use an empty payload (the
        # simulator falls back to its hardcoded demo data).
        self._embedded_cortex_payload: Optional[dict] = None
        # Tracks the provenance of the currently-installed payload on the
        # embedded view: None (no view yet), 'preloading' (built before
        # the worker finished, view is showing demo/empty data), 'ready'
        # (real payload installed). Used by _on_payload_ready to decide
        # whether a reload is needed when the worker thread finishes.
        self._embedded_cortex_payload_source: Optional[str] = None
        # Guard so concurrent payload-ready emissions can't trigger
        # overlapping view.reload() calls during the brief navigation
        # window.
        self._cortex_reloading: bool = False
        self._payload_thread_started: bool = False
        # Loading overlay shown over the cortex view while the page
        # boots, hidden on loadFinished. Cleared on close.
        self._cortex_loading_overlay = None
        # CortexEventBus that fans cortex_emit.* calls into the embedded
        # bridge.event signal. PRE-CREATED here in __init__ (without a
        # writer, in suspended mode) so any cortex_emit.* calls that
        # fire BEFORE the user clicks Iris (which would otherwise
        # silently no-op) are buffered into the bus. When the embedded
        # view loads, _build_embedded_cortex attaches the bridge writer
        # and resume()s — buffered events replay through the JS scene.
        # When the user collapses the panel, _collapse_cortex suspends
        # again but keeps the bus alive so events keep accumulating.
        self._cortex_bus = None
        try:
            from ...live_api.cortex.event_bus import CortexEventBus
            from ...live_api import cortex_emit
            self._cortex_bus = CortexEventBus(writer=None, parent=self)
            self._cortex_bus.suspend()  # No view yet — buffer until attach.
            cortex_emit.set_bus(self._cortex_bus)
        except Exception:
            self._cortex_bus = None
        # Floating ✕ button overlaid on the cortex panel; created on
        # first build, repositioned by the panel's resizeEvent hook.
        self._cortex_close_btn: Optional[QPushButton] = None
        # Persisted user preference — re-expand the cortex on next open
        # if the user closed Iris while the panel was visible.
        self._cortex_expanded_pref: bool = _load_cortex_expanded(default=False)
        # Set True once the warming QTimer fires, so showEvent doesn't
        # schedule duplicate warm passes if Qt re-raises showEvent (e.g.
        # after a minimize/restore cycle). Retained for backwards-
        # compatibility — the timer-driven warm has been removed in
        # favor of payload-ready-driven warming.
        self._cortex_warm_scheduled: bool = False
        # True if the user clicked Iris before the background payload
        # worker finished. When _on_payload_ready fires it'll call
        # _warm_cortex(), which sees this flag and immediately invokes
        # _expand_cortex() to honor the deferred click.
        self._cortex_expand_pending: bool = False
        # Standalone "✦ I R I S ✦" loading widget parked on the cortex
        # panel when the user clicks Iris before the payload is ready.
        # Replaced by the real QWebEngineView once _warm_cortex runs.
        self._cortex_expand_placeholder: Optional[QWidget] = None

        self._build_ui()
        self._wire_manager()
        # Wire the payload-ready signal so the embedded cortex view can
        # reload itself once the background payload worker finishes. The
        # signal is emitted from the worker thread; Qt marshals it to
        # the GUI thread automatically (DirectConnection in same thread
        # would be wrong — default AutoConnection picks QueuedConnection
        # across threads, which is what we want).
        try:
            self.payloadReady.connect(self._on_payload_ready)
        except Exception:
            pass

    # ---- UI construction ----

    def _build_ui(self) -> None:
        pal = self._palette
        self.setStyleSheet(
            f"QWidget {{ background-color: {pal['surface']}; color: {pal['text']}; }}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Header: clickable "Iris" title (opens Cortex viz) + state pill.
        header = QHBoxLayout()
        title = _ClickableLabel("Iris  🧠")
        title.setToolTip(
            "Click: toggle the embedded Iris Cortex side panel\n"
            "Right-click: open the full-size Cortex pop-out window"
        )
        title.setStyleSheet(
            f"QLabel {{ font-size: 18px; font-weight: 800; color: {pal['text']}; }}"
            f"QLabel:hover {{ color: {pal['accent']}; }}"
        )
        title.clicked.connect(self._toggle_embedded_cortex)
        title.rightClicked.connect(self._open_cortex_window)
        header.addWidget(title)
        header.addStretch(1)
        self._state_pill = QLabel("Off")
        self._state_pill.setAlignment(Qt.AlignCenter)
        self._set_state_pill(LiveApiState.OFF, "Off")
        header.addWidget(self._state_pill)
        root.addLayout(header)

        # Backend status pill — tells the user EXACTLY which engine is
        # running Iris this session. The "auto" backend gets resolved at
        # load_config() time into a concrete one; we render that + the
        # short reason so a free-tier user understands they're on the
        # local model and a key-having user sees they're on cloud.
        cfg = self._manager.config
        backend = str(getattr(cfg, "backend", "cloud")).lower() or "cloud"
        reason = str(getattr(cfg, "backend_auto_reason", "") or "")
        _BACKEND_BADGE = {
            "cloud":        ("Cloud · OpenAI", "#58E3FF",
                             "Best quality. Requires your OPENAI_API_KEY."),
            "local":        ("Local · Free (offline)", "#1DE9B6",
                             "Runs entirely on this PC. No key needed."),
            "subscription": ("Touchless Premium", "#F59E0B",
                             "Hosted backend — coming soon."),
        }
        label, color, tooltip = _BACKEND_BADGE.get(
            backend, (backend, "#94A3B8", ""))
        model = getattr(cfg, "model", "?")
        sub_text = f"Backend: {label} · model: {model}"
        if reason and "auto" not in reason.lower():
            # Keep the line short — full reason goes into the tooltip.
            pass
        sub = QLabel(sub_text)
        sub.setStyleSheet(
            f"font-size: 11px; color: {color}; font-weight: 600;")
        sub.setToolTip((tooltip + ("\n" + reason if reason else ""))
                       .strip() or label)
        root.addWidget(sub)

        # Body splitter: chat (left) | cortex viz (right, lazy).
        # The right pane is created hidden/zero-width — the first call
        # to _show_embedded_cortex() builds the QWebEngineView and
        # expands the splitter. Failure to build the view degrades to
        # chat-only without breaking anything.
        self._embedded_cortex_splitter = QSplitter(Qt.Horizontal)
        self._embedded_cortex_splitter.setHandleWidth(6)
        self._embedded_cortex_splitter.setChildrenCollapsible(True)
        self._embedded_cortex_splitter.setStyleSheet(
            "QSplitter::handle { background: transparent; }"
            "QSplitter::handle:hover { background: " + pal['accent'] + "33; }"
        )

        chat_container = QWidget()
        self._chat_container = chat_container
        chat_v = QVBoxLayout(chat_container)
        chat_v.setContentsMargins(0, 0, 0, 0)
        chat_v.setSpacing(10)

        # Transcript scroll area.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._transcript_host = QWidget()
        self._transcript = QVBoxLayout(self._transcript_host)
        self._transcript.setContentsMargins(2, 2, 2, 2)
        self._transcript.setSpacing(8)
        self._transcript.addStretch(1)
        self._scroll.setWidget(self._transcript_host)
        chat_v.addWidget(self._scroll, 1)

        # Track all bubble labels so we can re-apply their maximum width
        # whenever the chat container is resized (splitter drag, expand /
        # collapse, top-level resize). Without this the bubbles freeze at
        # whatever width was current at creation time and either cut off
        # or leave large gaps when the container changes width.
        self._bubble_labels: list[QLabel] = []
        chat_container.installEventFilter(self)

        self._add_system_bubble(
            "Press Start, then type a command. Quick commands (e.g. \"open chrome\") "
            "run locally with no tokens; complex requests use the model."
        )

        # Input row.
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command…")
        self._input.returnPressed.connect(self._on_send)
        self._input.setEnabled(False)
        self._input.setStyleSheet(
            f"QLineEdit {{ background:#0B1220; border:1px solid {pal['accent']}55; "
            f"border-radius:10px; padding:10px; font-size:13px; }}"
        )
        input_row.addWidget(self._input, 1)
        self._send_btn = QPushButton("Send")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._on_send)
        self._send_btn.setStyleSheet(self._button_style())
        input_row.addWidget(self._send_btn)
        chat_v.addLayout(input_row)

        self._embedded_cortex_splitter.addWidget(chat_container)

        # Cortex panel (placeholder until lazily filled).
        self._embedded_cortex_panel = QWidget()
        self._embedded_cortex_panel.setMinimumWidth(0)
        self._embedded_cortex_panel.setStyleSheet("background: #050a14;")
        panel_layout = QVBoxLayout(self._embedded_cortex_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        self._embedded_cortex_splitter.addWidget(self._embedded_cortex_panel)
        # Start collapsed — cortex sleeps until the user reveals it.
        self._embedded_cortex_splitter.setSizes([520, 0])
        self._embedded_cortex_splitter.setCollapsible(0, False)
        self._embedded_cortex_splitter.setCollapsible(1, True)
        self._embedded_cortex_splitter.setStretchFactor(0, 1)
        self._embedded_cortex_splitter.setStretchFactor(1, 1)
        root.addWidget(self._embedded_cortex_splitter, 1)

        # Controls row.
        controls = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_toggle_session)
        self._start_btn.setStyleSheet(self._button_style())
        controls.addWidget(self._start_btn)

        # Voice mute/unmute. Toggling persists and applies on the NEXT
        # session start (modalities can't be flipped mid-session — the
        # server rejects session.update that swaps output_modalities).
        self._voice_btn = QPushButton(
            self._voice_btn_label(self._voice_enabled))
        self._voice_btn.setToolTip(
            "Toggle spoken replies. Takes effect on next Start.")
        self._voice_btn.clicked.connect(self._on_toggle_voice)
        self._voice_btn.setStyleSheet(self._button_style(subtle=True))
        controls.addWidget(self._voice_btn)

        # One-click Gmail connect (only shown when the Google libs are present
        # and not yet connected). Runs the OAuth consent flow in the browser.
        self._gmail_btn = QPushButton("Connect Gmail")
        self._gmail_btn.setStyleSheet(self._button_style(subtle=True))
        self._gmail_btn.clicked.connect(self._on_connect_gmail)
        controls.addWidget(self._gmail_btn)
        self._gmail_result.connect(self._on_gmail_result)
        # NOTE: _refresh_gmail_button() and _refresh_ms_button() are deferred
        # to the END of _build_ui — they call setVisible() on these buttons,
        # and setVisible() on a widget whose owning layout hasn't yet been
        # attached to a parent (`root.addLayout(controls)` happens below)
        # makes Qt promote it to a top-level window. That's what caused the
        # "tiny white popup" that flashed every time Iris opened: an orphan
        # QPushButton at Qt's default 640x480, mistakenly shown as its own
        # window. Wire the layouts first, THEN refresh.

        # One-click Microsoft 365 connect (same pattern as Gmail).
        self._ms_btn = QPushButton("Connect Microsoft")
        self._ms_btn.setStyleSheet(self._button_style(subtle=True))
        self._ms_btn.clicked.connect(self._on_connect_ms)
        controls.addWidget(self._ms_btn)
        self._ms_result.connect(self._on_ms_result)

        # Voice picker — opens a modeless dialog where the user can
        # preview each Realtime voice and pick one as Iris's default.
        self._voice_btn = QPushButton("🔊 Voice")
        self._voice_btn.setStyleSheet(self._button_style(subtle=True))
        self._voice_btn.clicked.connect(self._open_voice_picker)
        controls.addWidget(self._voice_btn)

        # MCP server picker — opens the dialog where the user enables /
        # configures Model Context Protocol servers (Slack, GitHub,
        # Notion, Linear, filesystem, etc.). Changes apply on next Start.
        self._mcp_btn = QPushButton("🔌 MCP")
        self._mcp_btn.setStyleSheet(self._button_style(subtle=True))
        self._mcp_btn.setToolTip(
            "Enable / configure Model Context Protocol servers — "
            "GitHub, Slack, Notion, Linear, Filesystem, and more.")
        self._mcp_btn.clicked.connect(self._open_mcp_picker)
        controls.addWidget(self._mcp_btn)

        controls.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_transcript)
        clear_btn.setStyleSheet(self._button_style(subtle=True))
        controls.addWidget(clear_btn)
        root.addLayout(controls)

        # Now that the controls layout is attached to the root layout (which
        # is attached to `self`), the gmail/ms buttons have a real ancestor
        # widget. Safe to call setVisible() — they'll be shown/hidden inside
        # the controls row instead of as their own top-level windows.
        self._refresh_gmail_button()
        self._refresh_ms_button()

    def _button_style(self, *, subtle: bool = False) -> str:
        pal = self._palette
        bg = "transparent" if subtle else pal["primary"]
        return (
            f"QPushButton {{ background:{bg}; color:{pal['text']}; "
            f"border:1px solid {pal['accent']}55; border-radius:10px; "
            f"padding:9px 16px; font-weight:700; }}"
            f"QPushButton:hover {{ border:1px solid {pal['accent']}; }}"
            f"QPushButton:disabled {{ color:#64748B; border-color:#33415544; }}"
        )

    # ---- Cortex visualization ----

    def _open_cortex_window(self) -> None:
        """Open (or raise) the Iris Cortex 3D visualization window.

        Lazy import so we don't pay the QWebEngine startup cost unless
        the user actually clicks the Iris header. Failures degrade
        gracefully — clicking "Iris" must never break the assistant.
        """
        try:
            if self._cortex_window is not None:
                try:
                    self._cortex_window.show()
                    self._cortex_window.raise_()
                    self._cortex_window.activateWindow()
                    return
                except RuntimeError:
                    # Underlying C++ object deleted — recreate below.
                    self._cortex_window = None

            from ...live_api.cortex.window import CortexWindow

            self._cortex_window = CortexWindow(parent=None)
            self._cortex_window.show()
            self._cortex_window.raise_()
            self._cortex_window.activateWindow()
            # When the standalone CortexWindow closes it calls
            # cortex_emit.clear_bus(self.bus), which leaves cortex_emit
            # with NO bus — so subsequent neuron emits would silently
            # drop until the user re-opens cortex. Hook the destroyed
            # signal to re-register our pre-init buffering bus so
            # emits keep being captured for the embedded view.
            try:
                from ...live_api import cortex_emit
                def _restore_bus() -> None:
                    try:
                        if self._cortex_bus is not None:
                            cortex_emit.set_bus(self._cortex_bus)
                    except Exception:
                        pass
                self._cortex_window.destroyed.connect(_restore_bus)
            except Exception:
                pass
        except Exception as exc:
            self._add_system_bubble(f"⚠ Couldn't open Iris Cortex: {exc}")

    # ---- Embedded Cortex panel (right-hand side splitter pane) ----

    # Default width (px) of the embedded cortex panel when first shown.
    _EMBEDDED_CORTEX_WIDTH = 420

    def _toggle_embedded_cortex(self) -> None:
        """Expand the cortex side panel if collapsed, collapse it if open.

        Guarded end-to-end so a missing QWebEngine module (or any other
        failure in the cortex stack) just surfaces as a system bubble —
        the assistant window itself keeps working."""
        try:
            if self._is_cortex_expanded():
                self._collapse_cortex()
            else:
                self._expand_cortex()
        except Exception as exc:
            self._add_system_bubble(f"⚠ Couldn't toggle Iris Cortex panel: {exc}")

    def _is_cortex_expanded(self) -> bool:
        """True if the cortex panel currently occupies non-zero width."""
        splitter = self._embedded_cortex_splitter
        if splitter is None:
            return False
        sizes = splitter.sizes()
        return len(sizes) >= 2 and sizes[1] > 0

    def _show_embedded_cortex(self) -> None:
        """Legacy entry point — kept so any old callers (refresh_cortex,
        tests, etc.) still work. Routes through the new expand path."""
        self._expand_cortex()

    def _start_payload_preload(self) -> None:
        """Kick off the world-payload load on a background daemon thread.

        load_world_payload() does sync I/O — sqlite queries, embedder
        HTTP calls, gmail/ms365 fetches, filesystem scans — that adds
        up to seconds on first run. Doing it on the GUI thread (even
        deferred via QTimer) freezes the assistant window during the
        early click window the user noticed. Run it off-thread instead;
        store the result on self for the warm path / lazy build to use.
        """
        if self._payload_thread_started:
            return
        self._payload_thread_started = True
        import threading

        def _worker():
            try:
                from run_iris_simulator import load_world_payload
                payload = load_world_payload()
            except Exception:
                payload = {"projects": [], "tools": [], "memory": {},
                           "source": "missing"}
            self._embedded_cortex_payload = payload
            # Hand the rich payload to the tool executor so iris_query_node
            # can answer "what's inside <project>?" with real folder/file
            # structure (branches + leaves) instead of an empty list.
            # Best-effort — never break the cortex preload on import error.
            try:
                from ...live_api.tool_executor import set_rich_world_payload
                set_rich_world_payload(payload)
            except Exception:
                pass
            # Notify the GUI thread that real data has arrived. If the
            # embedded view was already built with a stale/empty payload,
            # the slot will trigger a one-shot reload so the cortex
            # actually shows real data. Emission from a background
            # thread is safe — Qt marshals the slot call onto the
            # owning thread via a QueuedConnection automatically.
            try:
                self.payloadReady.emit()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True,
                         name="cortex-payload-preload").start()

    def _eager_warm_view(self) -> None:
        """Eagerly build the embedded Cortex QWebEngineView + bridge
        BEFORE the world payload is ready.

        Called from show_window() one event-loop tick after the window
        first paints. Amortizes the ~200-500 ms QWebEngine renderer
        spawn + page boot + three.js scene build to app-show time so
        the user never waits for it on their first Iris click.

        Implementation: just routes through _warm_cortex() with whatever
        payload is currently cached (typically None / empty at this
        point, since the background preload worker is still running).
        _build_embedded_cortex() already handles a None payload by
        falling back to {"projects": [], "tools": [], ...}, the page
        falls back to its hardcoded demo data, and the panel stays
        collapsed so the demo data never paints visibly. When the
        background worker finishes, _on_payload_ready re-injects the
        real payload and reloads the page in place — no flash, the user
        only ever sees the real cortex.

        Idempotent — _warm_cortex() guards against double-build via
        _embedded_cortex_loaded. Failures degrade silently: the lazy
        build path inside _expand_cortex() still works as a fallback."""
        if self._embedded_cortex_loaded:
            return
        try:
            self._warm_cortex()
        except Exception:
            # Don't let eager-warm failures spam a system bubble — the
            # user hasn't even clicked Iris yet. The lazy build path in
            # _expand_cortex / _on_payload_ready will retry on demand.
            pass

    def _warm_cortex(self) -> None:
        """Build the embedded Cortex view EXACTLY ONCE, after the real
        world payload has loaded.

        Called from _on_payload_ready (NOT from a fixed startup timer).
        By the time we get here, self._embedded_cortex_payload is the
        real payload from the background loader — no empty fallback,
        no demo-data flash, no reload swap. The view is built with
        real data on its first and only paint.

        Behavior:
          * If a click on Iris arrived before the payload (i.e.
            _cortex_expand_pending is True OR a placeholder loading
            widget was attached to the panel), we honor that click as
            soon as the view is built by calling _expand_cortex().
          * Otherwise we mirror the old behavior: hide the panel so
            it sits ready for a later reveal.
          * Also honors the persisted "was expanded" preference here
            (used to live in showEvent); ensures the user lands back
            in an expanded cortex on a subsequent open of Iris.

        Failures degrade gracefully — _expand_cortex() still has a
        lazy build fallback in case this path didn't fire."""
        if self._embedded_cortex_loaded:
            return  # already built (defensive guard against double-emit)
        try:
            # Tear down any standalone loading widget that _expand_cortex
            # may have parked on the panel for the "fast click before
            # payload arrives" case. The real view is about to replace it.
            try:
                placeholder = getattr(self, "_cortex_expand_placeholder", None)
                if placeholder is not None:
                    try:
                        placeholder.hide()
                    except Exception:
                        pass
                    try:
                        placeholder.setParent(None)
                        placeholder.deleteLater()
                    except Exception:
                        pass
                    self._cortex_expand_placeholder = None
            except Exception:
                pass

            self._build_embedded_cortex()
            self._embedded_cortex_loaded = True

            # Decide whether we owe the user an expand. The ONLY way an
            # expand can be "pending" on startup is:
            #   1. They clicked Iris before the payload was ready and
            #      we set _cortex_expand_pending.
            # The persisted _cortex_expanded_pref is intentionally NOT
            # consulted here — the assistant window always opens with the
            # cortex collapsed, and the user must click 'Iris 🧠' to
            # expand it. The pref is still persisted so other code paths
            # can reference the user's last manual choice if needed.
            expand_now = bool(getattr(self, "_cortex_expand_pending", False))
            # Clear the deferred-click flag either way.
            self._cortex_expand_pending = False

            if expand_now:
                try:
                    self._expand_cortex()
                except Exception:
                    pass
            else:
                # Keep the panel hidden until the user reveals it. The
                # splitter is still collapsed to [chat_only, 0], so the
                # view is offscreen anyway, but hiding the panel widget
                # keeps any stray paint cycle from leaking through.
                if self._embedded_cortex_panel is not None:
                    try:
                        self._embedded_cortex_panel.hide()
                    except Exception:
                        pass
        except Exception:
            # Lazy loading will take over on first click. Silent fail so
            # we don't spam a system bubble at startup when the user
            # hasn't even asked for the cortex panel yet.
            pass

    def _expand_cortex(self) -> None:
        """Expand the window + splitter to show the Cortex beside chat.

        Grows the window from ~520x720 → ~1180x780, narrows the chat
        container to a fixed 280–360 px band so it doesn't stretch, and
        sizes the splitter so the cortex gets the wide right pane. If
        warming hasn't completed yet, falls back to a lazy build."""
        splitter = self._embedded_cortex_splitter
        panel = self._embedded_cortex_panel
        if splitter is None or panel is None:
            return

        # Payload-aware build gate. We only ever build the embedded
        # cortex view once the real world payload is in hand — otherwise
        # the view would render hardcoded demo data and the user would
        # see a visible swap when the worker finishes. Three cases:
        #
        #   A. View already built → fall through to the resize/show path.
        #   B. View NOT built AND real payload IS ready → build now
        #      (this is the lazy fallback in case _on_payload_ready
        #      didn't fire for some reason).
        #   C. View NOT built AND payload NOT ready → mark the expand
        #      as pending, park a loading overlay on the panel, grow
        #      the window so the user sees motion, and bail. When
        #      _on_payload_ready fires it'll call _warm_cortex(), which
        #      sees _cortex_expand_pending and re-invokes _expand_cortex.
        if not self._embedded_cortex_loaded:
            payload = self._embedded_cortex_payload
            payload_is_real = False
            try:
                if payload:
                    src = str(payload.get("source") or "")
                    payload_is_real = src not in ("", "preloading", "missing")
            except Exception:
                payload_is_real = False

            if payload_is_real:
                # Case B — real data is sitting in the cache, build it.
                try:
                    self._build_embedded_cortex()
                    self._embedded_cortex_loaded = True
                except Exception as exc:
                    self._add_system_bubble(
                        f"⚠ Couldn't load embedded Cortex: {exc}\n"
                        "Right-click 'Iris' to open the standalone window instead."
                    )
                    return
            else:
                # Case C — defer. Don't build with empty/preloading data.
                self._cortex_expand_pending = True
                try:
                    self._show_pending_expand_placeholder()
                except Exception:
                    pass
                # Still grow the window + reveal the panel so the user
                # gets immediate visual feedback that their click landed.
                # The real view will slot into the panel layout in place
                # of the placeholder when _warm_cortex runs.
                try:
                    target_w = _EXPANDED_WINDOW_W
                    target_h = _EXPANDED_WINDOW_H
                    self.resize(target_w, max(target_h, self.height()))
                    panel.show()
                    def _apply_pending_sizes(
                        cw: int = _EXPANDED_CHAT_W,
                        xw: int = _EXPANDED_CORTEX_W,
                    ) -> None:
                        try:
                            splitter.setSizes([cw, xw])
                        except Exception:
                            pass
                    QTimer.singleShot(0, _apply_pending_sizes)
                except Exception:
                    pass
                return

        # Pin the chat container to a narrow band so the splitter handle
        # has stable sizing constraints. Without this the chat would
        # stretch on next resize and the user would have to drag the
        # handle back to the configured ratio every time.
        try:
            chat_container = splitter.widget(0)
            if chat_container is not None:
                chat_container.setMinimumWidth(_CHAT_MIN_W_EXPANDED)
                chat_container.setMaximumWidth(_CHAT_MAX_W_EXPANDED)
        except Exception:
            pass

        # Clamp the target window width to the available screen — at
        # screen edges Qt would silently clamp anyway, but checking
        # ourselves lets us fall back to a narrower split that still
        # shows both halves rather than smushing the cortex.
        target_w = _EXPANDED_WINDOW_W
        target_h = _EXPANDED_WINDOW_H
        chat_w = _EXPANDED_CHAT_W
        cortex_w = _EXPANDED_CORTEX_W
        try:
            screen = self.screen()
            avail = screen.availableGeometry() if screen is not None else None
            if avail is not None and target_w > avail.width() - 40:
                target_w = max(900, avail.width() - 40)
                # Re-derive the split to fit the smaller window.
                cortex_w = max(440, target_w - chat_w - splitter.handleWidth() - 32)
        except Exception:
            pass

        # Resize the window, then set splitter sizes. Qt needs an event-
        # loop tick to settle the new geometry before setSizes() picks up
        # the new total width, so defer the splitter sizing one tick.
        self.resize(target_w, max(target_h, self.height()))
        panel.show()
        if self._embedded_cortex_view is not None:
            try:
                self._embedded_cortex_view.show()
            except Exception:
                pass
        # Re-expand path: the view is still loaded but the bus was
        # suspended by _collapse_cortex. Resume so the buffered events
        # (collected while the panel was hidden) replay into the JS
        # scene, and live emits start flowing again.
        try:
            if (self._cortex_bus is not None
                    and self._embedded_cortex_view is not None):
                self._cortex_bus.resume()
        except Exception:
            pass
        # Resume the Three.js render loop (paused on collapse to avoid
        # contending with Touchless's camera for GPU). Safe no-op if
        # the bridge function isn't loaded yet.
        try:
            self._set_cortex_render_paused(False)
        except Exception:
            pass

        def _apply_sizes(cw: int = chat_w, xw: int = cortex_w) -> None:
            try:
                splitter.setSizes([cw, xw])
            except Exception:
                pass
        QTimer.singleShot(0, _apply_sizes)
        # Reposition the floating close button after the panel grows.
        QTimer.singleShot(0, self._reposition_cortex_close_btn)

        _save_cortex_expanded(True)

        # Don't steal focus from the chat input — page load below would
        # otherwise pull focus into the web view.
        try:
            self._input.setFocus()
        except Exception:
            pass

    def _set_cortex_render_paused(self, paused: bool) -> None:
        """Pause / resume the Cortex Three.js render loop AND drop Qt
        updates on the embedded web view. This is the single biggest
        Touchless-FPS lever — without it the WebEngine helper process
        keeps rendering bloom+particles at 60 FPS even when the panel
        is hidden, contending with the camera pipeline for GPU.

        Two layers of pause:
          1. JS-side: call `window.cortexSetPaused(true)` so the
             requestAnimationFrame loop early-exits without touching
             the GPU. Safe no-op if the page hasn't loaded yet.
          2. Qt-side: `setUpdatesEnabled(False)` on the web view +
             `setVisible(False)` if collapsing, so Qt's compositor
             doesn't keep its surface live. On resume, re-enable both.

        Both layers must agree — JS alone leaves the Qt surface warm
        and Qt alone doesn't stop the JS render-loop work."""
        view = self._embedded_cortex_view
        # Layer 1: tell the JS scene to early-exit its frame() callback.
        if view is not None:
            try:
                page = view.page()
                if page is not None:
                    js = ("typeof window.cortexSetPaused === 'function' "
                          f"&& window.cortexSetPaused({'true' if paused else 'false'});")
                    page.runJavaScript(js)
            except Exception:
                pass
        # Layer 2: drop Qt paint updates on the embedded web view so the
        # compositor isn't asked to keep its texture warm. Reverse on
        # resume. Safe no-op when view isn't built yet.
        if view is not None:
            try:
                view.setUpdatesEnabled(not paused)
            except Exception:
                pass

    def _collapse_cortex(self) -> None:
        """Collapse the Cortex panel + shrink the window back to chat-only.

        Resizes to ~520x720, collapses the splitter to [520, 0], and
        hides the cortex panel. The view is kept alive (not destroyed)
        so the next expansion doesn't need to re-warm the QWebEngine
        view, bridge, or world payload.

        The cortex event bus is SUSPENDED (writer kept attached) so
        neurons keep firing into the bus's buffer while the panel is
        hidden — on re-expand we resume() and the buffered events
        replay through the still-loaded JS scene."""
        # Suspend the bus first so events stop being written to a
        # not-visible JS view (they keep accumulating in the buffer).
        try:
            if self._cortex_bus is not None:
                self._cortex_bus.suspend()
        except Exception:
            pass
        # CRITICAL for Touchless FPS: tell the Cortex JS to PAUSE its
        # Three.js render loop. Without this, the WebEngine helper
        # process keeps GPU-rendering at 60fps (bloom + particles +
        # postprocessing) while the panel is hidden, contending with
        # the main camera pipeline. Also flip Qt updates off on the
        # embedded view so the compositor doesn't keep its surface
        # warm. Best-effort — never block collapse if any of this
        # errors.
        try:
            self._set_cortex_render_paused(True)
        except Exception:
            pass
        splitter = self._embedded_cortex_splitter
        panel = self._embedded_cortex_panel
        if splitter is None or panel is None:
            return

        # Lift the chat-width pins so the narrow left pane can grow
        # back to fill the chat-only window.
        try:
            chat_container = splitter.widget(0)
            if chat_container is not None:
                chat_container.setMinimumWidth(0)
                chat_container.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
        except Exception:
            pass

        # Hide the panel first so the splitter doesn't fight the resize.
        try:
            panel.hide()
        except Exception:
            pass
        try:
            splitter.setSizes([_CHAT_ONLY_WINDOW_W, 0])
        except Exception:
            pass
        self.resize(_CHAT_ONLY_WINDOW_W, _CHAT_ONLY_WINDOW_H)

        _save_cortex_expanded(False)

        try:
            self._input.setFocus()
        except Exception:
            pass

    # ---- Cortex close button (overlaid in the panel's top-right) ----

    def _create_cortex_close_btn(self) -> QPushButton:
        """Create the small ✕ overlay button that collapses the cortex.

        Looks like part of the panel — semi-transparent bg, lighter on
        hover, sits in the top-right corner. Re-parents itself to the
        cortex panel and is repositioned by _reposition_cortex_close_btn
        on every panel resize."""
        pal = self._palette
        btn = QPushButton("✕", self._embedded_cortex_panel)
        btn.setToolTip("Close Cortex panel")
        btn.setFixedSize(28, 28)
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(15, 23, 42, 160);"
            f" color: {pal['text']};"
            f" border: 1px solid {pal['accent']}44;"
            "  border-radius: 14px;"
            "  font-size: 13px;"
            "  font-weight: 700;"
            "  padding: 0px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(29, 233, 182, 60);"
            f" color: {pal['text']};"
            f" border: 1px solid {pal['accent']};"
            "}"
            "QPushButton:pressed {"
            "  background: rgba(29, 233, 182, 120);"
            "}"
        )
        btn.clicked.connect(self._collapse_cortex)
        btn.raise_()
        btn.show()
        return btn

    def _reposition_cortex_close_btn(self) -> None:
        """Move the floating close button to the top-right of the panel.

        Called after every panel resize so the button stays glued to the
        corner instead of drifting when the splitter handle is dragged."""
        btn = getattr(self, "_cortex_close_btn", None)
        panel = self._embedded_cortex_panel
        if btn is None or panel is None:
            return
        try:
            x = max(0, panel.width() - btn.width() - 8)
            y = 8
            btn.move(x, y)
            btn.raise_()
        except Exception:
            pass

    def _build_loading_overlay_widget(self, parent):
        """Create an animated iris+dots loading widget for the cortex panel.

        Returns a QWebEngineView rendering ``loading_overlay.html`` from
        the cortex web folder. Using QWebEngine here lets the iris spin
        + dots pulse via smooth GPU-accelerated CSS @keyframes instead
        of a hand-rolled QPropertyAnimation stack. Background color is
        pre-painted on the page so we don't get a white flash before
        the CSS body rule applies. Returns None on import failure so
        callers can fall back to a static QLabel."""
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtCore import QUrl as _QUrl, Qt as _Qt
            from PySide6.QtGui import QColor as _QColor
            from pathlib import Path as _Path
            from ...live_api.cortex import window as _cortex_win_mod
            cortex_pkg_dir = _Path(_cortex_win_mod.__file__).resolve().parent
            html_path = cortex_pkg_dir / "web" / "loading_overlay.html"
            if not html_path.exists():
                return None
            view = QWebEngineView(parent)
            view.setAttribute(_Qt.WA_TransparentForMouseEvents, False)
            try:
                view.page().setBackgroundColor(_QColor(5, 10, 20))
            except Exception:
                pass
            try:
                view.setAttribute(_Qt.WA_OpaquePaintEvent, True)
            except Exception:
                pass
            view.load(_QUrl.fromLocalFile(str(html_path)))
            return view
        except Exception:
            return None

    def _show_pending_expand_placeholder(self) -> None:
        """Park a spinning-iris loading widget on the cortex panel so
        the user gets immediate visual feedback when they click Iris
        before the world payload is ready.

        Replaced by the real QWebEngineView once _warm_cortex runs
        (which also tears this placeholder down). Mirrors the inner
        loading overlay used inside _build_embedded_cortex so the
        visual style matches across the two states."""
        panel = self._embedded_cortex_panel
        if panel is None:
            return
        # Idempotent — don't stack multiple placeholders if the user
        # double-clicks Iris during the loading window.
        if getattr(self, "_cortex_expand_placeholder", None) is not None:
            try:
                self._cortex_expand_placeholder.show()
                self._cortex_expand_placeholder.raise_()
                return
            except Exception:
                self._cortex_expand_placeholder = None

        from PySide6.QtWidgets import QVBoxLayout as _VBox
        # Try the animated WebEngine loading view first; fall back to
        # the original static QLabel if anything in that path explodes
        # (missing HTML, WebEngine import error, etc).
        placeholder = self._build_loading_overlay_widget(panel)
        if placeholder is None:
            from PySide6.QtWidgets import QLabel as _QLabel
            from PySide6.QtCore import Qt as _Qt
            placeholder = _QLabel("✦  I R I S  ✦", panel)
            placeholder.setAlignment(_Qt.AlignCenter)
            placeholder.setStyleSheet(
                "QLabel {"
                " background: #050a14;"
                " color: rgba(180,140,255,0.85);"
                " font-family: -apple-system, Segoe UI, sans-serif;"
                " font-size: 24px; letter-spacing: 8px; font-weight: 300;"
                "}"
            )
        # Ensure the panel has a layout so the placeholder participates
        # in resize tracking without us hand-managing geometry.
        panel_layout = panel.layout()
        if panel_layout is None:
            panel_layout = _VBox(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)
        panel_layout.addWidget(placeholder)
        # Also set a baseline geometry so it covers the panel before the
        # layout pass settles (avoids a one-frame gap on first show).
        try:
            placeholder.setGeometry(0, 0, panel.width() or 800,
                                    panel.height() or 600)
        except Exception:
            pass
        placeholder.show()
        placeholder.raise_()
        self._cortex_expand_placeholder = placeholder

    def _build_embedded_cortex(self) -> None:
        """Construct the QWebEngineView + bridge + load the page.

        Called from _warm_cortex (and the lazy fallback in
        _expand_cortex) once the real world payload is available.
        Raises on any setup failure so the caller can surface a
        system bubble."""
        # Lazy imports keep the PySide6 WebEngine dependency optional —
        # users who never open the embedded cortex shouldn't pay its
        # startup cost (and shouldn't see import errors at boot).
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWebChannel import QWebChannel

        # Find run_iris_simulator.py at the repo root so we can reuse
        # its world-payload loader and script-injection helpers. The
        # script lives next to the package root, not inside src/.
        from pathlib import Path
        import sys as _sys

        here = Path(__file__).resolve()
        # .../src/hgr/app/ui/live_assistant_window.py → repo root is 4 levels up.
        repo_root = here.parents[4]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))

        from run_iris_simulator import (
            install_qwebchannel_bootstrap,
            install_world_injection,
            load_world_payload,
        )
        from ...live_api.cortex.bridge import CortexBridge, set_active_bridge
        from ...live_api.cortex.event_bus import CortexEventBus
        from ...live_api import cortex_emit

        # Build the web view — keep it pinned to a sane minimum width
        # so the splitter handle doesn't let the panel shrink to nothing.
        view = QWebEngineView(self._embedded_cortex_panel)
        view.setMinimumWidth(240)

        # Paint the view's underlying QWebEnginePage with the cortex
        # background color BEFORE the page loads. Without this, the
        # default white QWebEngineView surface flashes for a beat
        # while the HTML <body>'s dark-blue background paints.
        try:
            from PySide6.QtGui import QColor
            view.page().setBackgroundColor(QColor(5, 10, 20))
        except Exception:
            pass
        # Tell Qt not to clear the widget background behind the page —
        # the QWebEnginePage's setBackgroundColor owns the surface, so
        # the default Qt-side clear (which can flash white/black on
        # GPU surface reinit) is unnecessary and harmful here.
        try:
            from PySide6.QtCore import Qt as _Qt
            view.setAttribute(_Qt.WA_OpaquePaintEvent, True)
        except Exception:
            pass

        # QWebChannel must hold a Python reference to the bridge AND
        # the channel for the lifetime of the page, otherwise JS-side
        # events get silently dropped after the next GC pass. Store
        # both on `self` so they stay alive as long as the window does.
        bridge = CortexBridge(view)
        channel = QWebChannel(view.page())
        channel.registerObject("cortex", bridge)
        view.page().setWebChannel(channel)

        # Wire the JS-injection fallback so emit_project_added /
        # emit_project_removed can directly call into the page even when
        # the QWebChannel signal path drops events.
        #
        # CRITICAL: page.runJavaScript() is NOT thread-safe. Calling it
        # directly from a worker thread (where bridge.emit_* gets fired
        # — tool dispatch runs off the GUI thread) crashes the Qt
        # process at the C++ level. The earlier comment claiming "Qt
        # queues it onto the GUI thread automatically" was WRONG —
        # that's true for signals, not for direct method calls on
        # QWebEnginePage. We MUST marshal onto the GUI thread via
        # QTimer.singleShot(0, ...) which posts into the QApplication
        # event loop. This was the root cause of the iris_add_project
        # hard crash users saw.
        try:
            from PySide6.QtCore import QTimer as _QTimer
            _page_ref = view.page()
            def _run_js(src: str, _p=_page_ref) -> None:
                def _do() -> None:
                    try:
                        _p.runJavaScript(src)
                    except Exception:
                        pass
                try:
                    _QTimer.singleShot(0, _do)
                except Exception:
                    # Last-ditch direct call — only safe if we happen
                    # to be on the GUI thread already.
                    try:
                        _p.runJavaScript(src)
                    except Exception:
                        pass
            bridge.set_js_runner(_run_js)
        except Exception:
            pass

        # ALSO register a tool_executor-level JS runner so iris_add_project
        # and iris_remove_project handlers can directly invoke
        # handleProjectAdded() / handleProjectRemoved() in the page —
        # this is a belt-and-braces fallback that GUARANTEES the visual
        # transition fires even if the QWebChannel signal path drops the
        # event. Marshalled onto the GUI thread via QTimer.singleShot(0)
        # because tool handlers run on the Realtime worker thread, and
        # QWebEnginePage.runJavaScript is only safe from the GUI thread.
        try:
            from PySide6.QtCore import QTimer as _QTimer
            _page_ref_te = view.page()
            def _run_js_gui(src: str, _p=_page_ref_te) -> None:
                def _do() -> None:
                    try:
                        _p.runJavaScript(src)
                    except Exception:
                        pass
                try:
                    _QTimer.singleShot(0, _do)
                except Exception:
                    # Last-ditch direct call — only safe if we happen
                    # to be on the GUI thread already.
                    try:
                        _p.runJavaScript(src)
                    except Exception:
                        pass
            from ...live_api import tool_executor as _te_mod
            if hasattr(_te_mod, "set_cortex_js_runner"):
                _te_mod.set_cortex_js_runner(_run_js_gui)
        except Exception:
            pass

        # Register globally so non-GUI subsystems (tool_executor,
        # world_state, memory.manager) can push discrete signals
        # without taking a hard import dependency. Mirrors what
        # run_iris_simulator.wire_live_cortex_bridge does.
        try:
            set_active_bridge(bridge)
        except Exception:
            pass

        # ALSO wire the CortexEventBus → bridge.event signal so that
        # cortex_emit.edge_pulse / node_activity / tool_call calls
        # actually reach the JS handler in the embedded view. Without
        # this, edge_pulse fires from realtime_client / planner /
        # tool_executor but nothing renders here — only the standalone
        # CortexWindow saw the pulses. Mirrors window.py setup.
        #
        # The bus is PRE-CREATED in __init__ (suspended, no writer) so
        # events that fire before the user opens cortex were already
        # buffered. Here we attach the bridge writer and (later, after
        # the JS page is wired) resume() to replay the buffer. If the
        # pre-init failed (bus is None), fall back to the old build-
        # on-demand path so standalone behaviour still works.
        try:
            import sys as _sys
            if self._cortex_bus is not None:
                self._cortex_bus.set_writer(bridge.emit_event)
                # CRITICAL: do NOT call suspend() again here. The bus was
                # pre-suspended in __init__ and remains suspended until JS
                # is ready (jsReady handshake) or the loadFinished fallback
                # fires. Calling suspend() a second time was redundant and
                # obscured the real failure mode — events arrived at a
                # writer-less bus while we were chasing this bug.
                try:
                    _sys.stderr.write(
                        f"[cortex-wire] attached bridge.emit_event to "
                        f"pre-built bus; has_writer="
                        f"{self._cortex_bus.has_writer()}, is_suspended="
                        f"{self._cortex_bus.is_suspended()}\n"
                    )
                    _sys.stderr.flush()
                except Exception:
                    pass
                cortex_emit.set_bus(self._cortex_bus)
            else:
                self._cortex_bus = CortexEventBus(
                    writer=bridge.emit_event, parent=self,
                )
                self._cortex_bus.suspend()
                try:
                    _sys.stderr.write(
                        f"[cortex-wire] built fresh bus with writer; "
                        f"has_writer={self._cortex_bus.has_writer()}, "
                        f"is_suspended={self._cortex_bus.is_suspended()}\n"
                    )
                    _sys.stderr.flush()
                except Exception:
                    pass
                cortex_emit.set_bus(self._cortex_bus)
        except Exception:
            # Don't null self._cortex_bus on failure here — it may still
            # be the pre-init buffering instance that's catching events.
            pass

        # Wire the JS-ready handshake → bus.resume(). The bridge fires
        # jsReady() once cortex.event.connect() is wired on the JS side,
        # which is the ONLY moment buffered events can safely replay
        # without being lost between the bridge signal and the listener.
        # Belt-and-braces: the loadFinished handler below also calls
        # resume() with a short delay in case jsReady never fires (e.g.
        # asset error). resume() is idempotent so the double-call is safe.
        try:
            def _on_js_ready_embedded() -> None:
                try:
                    bus = self._cortex_bus
                    if bus is not None:
                        bus.resume()
                except Exception:
                    pass
            bridge.set_ready_callback(_on_js_ready_embedded)
        except Exception:
            pass

        # Use the payload preloaded on the background thread (started in
        # show_window). Under the new build-once flow this method is
        # only called AFTER the worker has populated the cache, so the
        # payload is real. The empty fallback below is purely defensive
        # in case a future caller bypasses the payload-ready gate —
        # we never want to freeze the GUI thread by running
        # load_world_payload() synchronously here.
        payload = self._embedded_cortex_payload or {
            "projects": [], "tools": [], "memory": {}, "source": "preloading"
        }
        self._embedded_cortex_payload = payload
        # Record where this payload came from so _on_payload_ready can
        # decide whether the view needs a reload when the preload
        # worker finishes. 'preloading' / 'missing' (or any falsy
        # source) → reload once real data arrives; anything else
        # (e.g. 'discovered', 'cache') → already real, skip the reload.
        try:
            src = str(payload.get("source") or "preloading")
        except Exception:
            src = "preloading"
        self._embedded_cortex_payload_source = src

        install_world_injection(view, payload)
        install_qwebchannel_bootstrap(view)

        # Tell the page it's being rendered inside the embedded side
        # panel (vs. the full-size standalone window) so it can hide its
        # built-in test prompts / dev overlays. Injected at
        # DocumentCreation so window.IRIS_EMBEDDED is true BEFORE the
        # page's own module script reads it.
        try:
            from PySide6.QtWebEngineCore import QWebEngineScript
            _embed_script = QWebEngineScript()
            _embed_script.setName("IrisEmbeddedFlag")
            _embed_script.setSourceCode("window.IRIS_EMBEDDED = true;")
            _embed_script.setInjectionPoint(QWebEngineScript.DocumentCreation)
            _embed_script.setWorldId(QWebEngineScript.MainWorld)
            _embed_script.setRunsOnSubFrames(False)
            view.page().scripts().insert(_embed_script)
        except Exception:
            pass

        # Locate iris_simulator.html relative to the cortex package.
        # Mirrors the lookup in run_iris_simulator.main but uses the
        # in-tree package path so it also resolves under PyInstaller.
        from ...live_api.cortex import window as _cortex_win_mod
        cortex_pkg_dir = Path(_cortex_win_mod.__file__).resolve().parent
        html_path = cortex_pkg_dir / "web" / "demos" / "iris_simulator.html"
        if not html_path.exists():
            raise FileNotFoundError(f"iris_simulator.html missing at {html_path}")

        view.load(QUrl.fromLocalFile(str(html_path)))

        # Build a loading overlay (spinning iris + animated dots on a
        # dark-blue background) that sits ON TOP of the QWebEngineView
        # while the page boots. Hidden once loadFinished fires + a
        # short grace period for the first three.js frame to paint.
        # Avoids the white flash + brief test-prompt flash users were
        # seeing. Uses a tiny WebEngine view rendering
        # loading_overlay.html for smooth CSS-driven animation, with a
        # static-label fallback if WebEngine setup fails.
        loading = self._build_loading_overlay_widget(
            self._embedded_cortex_panel)
        if loading is None:
            from PySide6.QtWidgets import QLabel as _QLabel
            from PySide6.QtCore import Qt as _Qt
            loading = _QLabel("✦  I R I S  ✦", self._embedded_cortex_panel)
            loading.setAlignment(_Qt.AlignCenter)
            loading.setStyleSheet(
                "QLabel {"
                " background: #050a14;"
                " color: rgba(180,140,255,0.85);"
                " font-family: -apple-system, Segoe UI, sans-serif;"
                " font-size: 24px; letter-spacing: 8px; font-weight: 300;"
                "}"
            )
            loading.setAttribute(_Qt.WA_TransparentForMouseEvents, False)
        loading.show()
        self._cortex_loading_overlay = loading

        # Restore chat focus once the page finishes loading + hide the
        # loading overlay after a brief grace period so the first
        # three.js frame has time to paint before we reveal the canvas.
        # ALSO resume the cortex event bus so any events buffered before
        # the user opened the panel (and during the page-boot window)
        # replay through the now-attached bridge writer.
        def _on_loaded(_ok: bool) -> None:
            try:
                self._input.setFocus()
            except Exception:
                pass
            from PySide6.QtCore import QTimer as _T
            def _hide():
                try:
                    # Keep the loading overlay visible if the page was
                    # booted with an empty (preloading) payload — under
                    # eager-warm we want the user to see the spinner if
                    # they expand Iris before the real payload arrives,
                    # not the hardcoded demo data. _on_payload_ready
                    # will trigger a reload and the post-reload
                    # loadFinished call here will hide the overlay then.
                    src = self._embedded_cortex_payload_source or ""
                    if src in ("preloading", "missing", ""):
                        return
                    if self._cortex_loading_overlay is not None:
                        self._cortex_loading_overlay.hide()
                except Exception:
                    pass
            _T.singleShot(450, _hide)
            # Fallback resume: if the bridge's jsReady() callback fires
            # first (the normal case), the bus is already drained and
            # this is a harmless no-op. We keep this fallback because a
            # broken asset / JS error could prevent jsReady() from ever
            # firing, in which case events would silently buffer forever.
            # Replays the buffered events in one batch via _flush() —
            # coalescing keeps it cheap.
            def _resume_bus():
                try:
                    bus = self._cortex_bus
                    if bus is not None and bus.is_suspended():
                        bus.resume()
                except Exception:
                    pass
            _T.singleShot(500, _resume_bus)
        try:
            view.page().loadFinished.connect(_on_loaded)
        except Exception:
            pass

        # Attach to the panel layout and stash references.
        panel_layout = self._embedded_cortex_panel.layout()
        if panel_layout is None:
            from PySide6.QtWidgets import QVBoxLayout as _VBox
            panel_layout = _VBox(self._embedded_cortex_panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)
        panel_layout.addWidget(view)

        # Size the loading overlay to cover the panel and keep it on top
        # via raise_(). Tracked by the panel's resizeEvent hook below.
        loading.setGeometry(0, 0, self._embedded_cortex_panel.width() or 800,
                            self._embedded_cortex_panel.height() or 600)
        loading.raise_()

        self._embedded_cortex_view = view
        self._embedded_cortex_bridge = bridge
        self._embedded_cortex_channel = channel

        # Floating ✕ button overlaid in the panel's top-right corner.
        # Built once on first cortex creation; repositioned on every
        # panel resize via the resizeEvent monkey-patch below.
        if getattr(self, "_cortex_close_btn", None) is None:
            try:
                self._cortex_close_btn = self._create_cortex_close_btn()
            except Exception:
                self._cortex_close_btn = None

        # Hook the panel's resizeEvent so the floating close button
        # tracks the top-right corner as the user drags the splitter
        # handle. Wrapping (vs. subclassing) keeps the rest of the
        # cortex setup untouched and avoids needing a custom QWidget
        # subclass purely for this one event override.
        try:
            _orig_resize = self._embedded_cortex_panel.resizeEvent

            def _panel_resize_event(event, _orig=_orig_resize):
                try:
                    _orig(event)
                except Exception:
                    pass
                self._reposition_cortex_close_btn()
                # Keep the loading overlay covering the full panel.
                try:
                    if getattr(self, "_cortex_loading_overlay", None) is not None:
                        self._cortex_loading_overlay.setGeometry(
                            0, 0,
                            self._embedded_cortex_panel.width(),
                            self._embedded_cortex_panel.height(),
                        )
                except Exception:
                    pass

            self._embedded_cortex_panel.resizeEvent = _panel_resize_event  # type: ignore[assignment]
        except Exception:
            pass

    def refresh_cortex(self) -> None:
        """Re-inject window.IRIS_WORLD with a fresh world payload.

        Useful when the world drifted between sessions but the live
        CortexEventBus signals didn't capture the delta (e.g. a project
        was added by a sibling tool while the panel was collapsed).
        Idempotent and silent on failure — does nothing if the embedded
        view hasn't been built yet."""
        if not self._embedded_cortex_loaded or self._embedded_cortex_view is None:
            return
        # Run the payload load on a background thread — load_world_payload
        # does sync I/O that would freeze the GUI thread for a second or
        # two. When the worker finishes, we marshall the re-injection
        # back to the GUI thread via QTimer.singleShot(0, ...).
        import threading
        from PySide6.QtCore import QTimer as _QTimer

        def _refresh_worker():
            try:
                from pathlib import Path
                import sys as _sys
                here = Path(__file__).resolve()
                repo_root = here.parents[4]
                if str(repo_root) not in _sys.path:
                    _sys.path.insert(0, str(repo_root))
                from run_iris_simulator import (
                    install_world_injection, load_world_payload,
                )
                payload = load_world_payload()
            except Exception:
                return
            self._embedded_cortex_payload = payload
            # Re-inject into the tool executor so iris_query_node sees
            # the fresh branches/leaves on its next call. Best-effort.
            try:
                from ...live_api.tool_executor import set_rich_world_payload
                set_rich_world_payload(payload)
            except Exception:
                pass

            def _apply():
                try:
                    install_world_injection(
                        self._embedded_cortex_view, payload)
                    import json as _json
                    payload_js = _json.dumps(
                        _json.dumps(payload, ensure_ascii=False))
                    self._embedded_cortex_view.page().runJavaScript(
                        f"window.IRIS_WORLD = JSON.parse({payload_js});"
                        "if (typeof rebuildWorld === 'function') rebuildWorld();"
                    )
                except Exception:
                    pass

            _QTimer.singleShot(0, _apply)

        threading.Thread(target=_refresh_worker, daemon=True,
                         name="cortex-payload-refresh").start()

    def _on_payload_ready(self) -> None:
        """Slot fired (on the GUI thread) when the background preload
        worker has populated self._embedded_cortex_payload.

        Two paths now exist:

          A. Eager-warm path (the common case under the new flow):
             show_window() called _eager_warm_view() which built the
             QWebEngineView with an empty payload during app-show.
             By the time payloadReady fires, _embedded_cortex_view is
             not None. We re-inject window.IRIS_WORLD with the real
             payload via runJavaScript, try the optional rebuildScene()
             hook, and (as a belt-and-braces fallback for pages without
             a rebuild hook) install a fresh world-injection script and
             reload the view so the page sees real data on its next
             DocumentCreation. The loading overlay is re-shown for the
             reload and hidden when loadFinished fires.

          B. Legacy lazy path (fallback): eager-warm failed (e.g.
             QWebEngine import error) so the view doesn't exist. Call
             _warm_cortex() now — it'll build the view with the real
             payload and honor any deferred Iris click.

        Failures are swallowed so a stuck refresh never breaks the rest
        of the assistant window."""
        try:
            # Path B — eager warm didn't run / didn't finish in time.
            if self._embedded_cortex_view is None and not self._embedded_cortex_loaded:
                self._warm_cortex()
                return

            # Path A — view already exists from eager warm. Re-inject
            # the real payload in place. No-op if no payload arrived
            # (defensive — payloadReady shouldn't fire empty).
            payload = self._embedded_cortex_payload
            if not payload:
                return
            try:
                src = str(payload.get("source") or "")
            except Exception:
                src = ""
            # Skip the reload if we somehow already installed real data
            # (e.g. duplicate signal emission, or a future code path
            # building with real data on first try).
            if (self._embedded_cortex_payload_source
                    and self._embedded_cortex_payload_source not in (
                        "preloading", "missing", "")):
                return
            if self._cortex_reloading:
                return
            self._cortex_reloading = True
            self._embedded_cortex_payload_source = src or "ready"

            view = self._embedded_cortex_view
            # 1) Push the real payload into the live JS context so any
            #    code that reads window.IRIS_WORLD on demand picks up
            #    the fresh data immediately.
            # 2) Call the optional rebuildScene() hook — if the page
            #    exposes it, the three.js scene can rebuild in place
            #    without a full reload. The current iris_simulator.html
            #    doesn't expose this hook yet, so the call is a no-op
            #    and the view.reload() below handles the visual swap.
            try:
                import json as _json
                payload_js = _json.dumps(
                    _json.dumps(payload, ensure_ascii=False))
                view.page().runJavaScript(
                    f"window.IRIS_WORLD = JSON.parse({payload_js});"
                    "if (typeof rebuildScene === 'function') {"
                    "  try { rebuildScene(); } catch (e) {}"
                    "}"
                )
            except Exception:
                pass

            # Install a fresh world-injection script so the next page
            # navigation (including the reload below) sees the real
            # payload at DocumentCreation. Without this, view.reload()
            # would re-run the OLD injection script and the page would
            # boot with stale/empty data again. Old scripts pile up but
            # only the last IRIS_WORLD assignment wins.
            try:
                from pathlib import Path
                import sys as _sys
                here = Path(__file__).resolve()
                repo_root = here.parents[4]
                if str(repo_root) not in _sys.path:
                    _sys.path.insert(0, str(repo_root))
                from run_iris_simulator import install_world_injection
                install_world_injection(view, payload)
            except Exception:
                pass

            # Re-inject the rich payload into the tool executor too,
            # mirroring _start_payload_preload (it already did this once,
            # but doing it again is cheap and protects against the
            # tool_executor being imported after the preload).
            try:
                from ...live_api.tool_executor import set_rich_world_payload
                set_rich_world_payload(payload)
            except Exception:
                pass

            # Show the loading overlay over the (possibly-visible)
            # cortex view while the reload runs so the user doesn't see
            # a one-frame flash of demo data being replaced by real
            # data. Hidden by the existing _on_loaded handler attached
            # in _build_embedded_cortex once loadFinished fires.
            try:
                if self._cortex_loading_overlay is not None:
                    self._cortex_loading_overlay.show()
                    self._cortex_loading_overlay.raise_()
            except Exception:
                pass

            # Reload the page so the freshly-installed injection script
            # fires at DocumentCreation with the real payload. This is
            # the visual swap from demo data → real data. The reload
            # reuses the cached HTML + JS so it's much faster than the
            # initial load (only DOM rebuild + three.js scene re-init,
            # no asset download). Wrapped in a singleShot(0) so the
            # script install + JS push above settle before reload.
            from PySide6.QtCore import QTimer as _QTimer
            def _do_reload() -> None:
                try:
                    view.reload()
                except Exception:
                    pass
                finally:
                    # Clear the in-flight guard — loadFinished fires
                    # after this, but we don't gate on it because a
                    # second payloadReady emission shouldn't re-reload.
                    self._cortex_reloading = False
            _QTimer.singleShot(0, _do_reload)
            return
        except Exception:
            # Failure to refresh must never break the assistant. Clear
            # the in-flight guard so a later refresh can still try.
            try:
                self._cortex_reloading = False
            except Exception:
                pass

    # ---- Voice picker ----

    def _open_mcp_picker(self) -> None:
        """Open (or raise) the modeless MCP-servers picker. Lets the user
        enable / disable / token-configure any server in the default
        catalog (GitHub, Slack, Notion, Linear, filesystem, ...). Saved
        changes take effect on the next Start. Failures degrade
        gracefully — clicking MCP must never break the assistant."""
        try:
            existing = getattr(self, "_mcp_dialog", None)
            if existing is not None:
                try:
                    existing.show()
                    existing.raise_()
                    existing.activateWindow()
                    return
                except RuntimeError:
                    self._mcp_dialog = None
            from .mcp_picker import McpPickerDialog
            dlg = McpPickerDialog(palette=self._palette, parent=self)
            dlg.servers_saved.connect(self._on_mcp_servers_saved)
            self._mcp_dialog = dlg
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as exc:
            self._add_system_bubble(f"Couldn't open MCP picker: {exc}")

    def _on_mcp_servers_saved(self) -> None:
        """Surface a one-line note in the transcript so the user knows
        the change is queued for the next session start."""
        self._add_system_bubble(
            "MCP servers saved — Stop and Start the session to load them.")

    def _open_voice_picker(self) -> None:
        """Open (or raise) the modeless Voice picker dialog. Saves the
        chosen voice to QSettings and applies it to the live config so
        the next session start uses it. Failures degrade gracefully —
        clicking Voice must never break the assistant."""
        try:
            if self._voice_dialog is not None:
                try:
                    self._voice_dialog.show()
                    self._voice_dialog.raise_()
                    self._voice_dialog.activateWindow()
                    return
                except RuntimeError:
                    self._voice_dialog = None

            from .voice_picker import VoicePickerDialog

            current = getattr(self._manager.config, "voice", "sage")
            api_key = getattr(self._manager.config, "api_key", None)
            self._voice_dialog = VoicePickerDialog(
                current_voice=current,
                api_key=api_key,
                palette=self._palette,
                parent=self,
            )
            self._voice_dialog.voice_chosen.connect(self._on_voice_chosen)
            self._voice_dialog.show()
            self._voice_dialog.raise_()
            self._voice_dialog.activateWindow()
        except Exception as exc:
            self._add_system_bubble(f"⚠ Couldn't open Voice picker: {exc}")

    def _on_voice_chosen(self, voice_id: str) -> None:
        """User saved a voice choice. Update the live config + tell
        the user; the change applies next time they start a session
        (Realtime session.voice is set during session.update)."""
        try:
            self._manager.config.voice = voice_id
        except Exception:
            pass
        running = False
        try:
            running = self._manager.is_running()
        except Exception:
            pass
        note = (
            f"Voice set to {voice_id}."
            + (" Stop + start the session to hear it." if running else " It'll be used next time you start.")
        )
        self._add_system_bubble(note)

    # ---- manager wiring ----

    def _wire_manager(self) -> None:
        self._manager.set_confirm_callback(self._confirm_tool)
        self._confirm_request.connect(self._on_confirm_request)
        self._manager.state_changed.connect(self._on_state_changed)
        self._manager.transcript_received.connect(self._on_transcript)
        self._manager.assistant_text.connect(self._on_assistant_delta)
        self._manager.assistant_message_break.connect(self._on_assistant_break)
        self._manager.tool_event.connect(self._on_tool_event)
        self._manager.error_occurred.connect(self._on_error)
        # Connect-Gmail / Connect-Outlook / Read-Outlook-screen action
        # chips that surface beneath the most recent assistant reply.
        try:
            self._manager.suggested_actions.connect(self._on_suggested_actions)
        except Exception:
            # Older managers without the signal — harmless.
            pass

    # ---- session control ----

    @staticmethod
    def _voice_btn_label(enabled: bool) -> str:
        return "🔊 Voice on" if enabled else "🔇 Voice off"

    def _on_toggle_voice(self) -> None:
        """Flip the spoken-reply toggle. Persists immediately; the change
        applies on the next session start (the server rejects modality
        swaps mid-session)."""
        self._voice_enabled = not self._voice_enabled
        _save_voice_enabled(self._voice_enabled)
        self._voice_btn.setText(self._voice_btn_label(self._voice_enabled))
        # ALWAYS sync the manager's flag to the UI's. Without this the
        # manager would keep its constructor-time value across stop/start
        # cycles and the next session would launch without audio output.
        # During an in-flight session this is informational only (the
        # server rejects modality swaps mid-session) — the UI message
        # below makes that clear to the user.
        try:
            self._manager._voice_output = self._voice_enabled
        except Exception:
            pass
        if self._manager.is_running():
            self._add_system_bubble(
                "Voice " + ("on" if self._voice_enabled else "off")
                + " — takes effect on next Start.")

    def _on_toggle_session(self) -> None:
        if self._manager.is_running():
            self._manager.stop()
            self._start_btn.setText("Start")
        else:
            # Pre-flight: if Iris auto-resolved to the local backend but
            # the GGUF model isn't on disk, offer a one-click download
            # before trying to start (otherwise local backend errors
            # immediately with a cryptic 'model not found'). Skip the
            # check entirely for cloud / subscription paths.
            if not self._preflight_local_model_check():
                return
            # Final sync — guarantees manager voice flag matches the UI
            # button state at session start. Without this, a toggle that
            # happened during a previous live session (and got dismissed
            # with "takes effect on next Start") could be lost if any
            # other path mutated the manager flag in between.
            try:
                self._manager._voice_output = self._voice_enabled
            except Exception:
                pass
            self._add_system_bubble("Starting session…")
            self._manager.start()
            self._start_btn.setText("Stop")

    def _preflight_local_model_check(self) -> bool:
        """If the local backend was auto-selected and no GGUF is present,
        offer a one-click download. Returns True to proceed with session
        start, False to abort (user cancelled or download failed)."""
        cfg = self._manager.config
        backend = str(getattr(cfg, "backend", "")).lower()
        if backend != "local":
            return True
        try:
            from ...live_api.local_backend import probe_local_backend
            probe = probe_local_backend(
                model_filename=cfg.local_llm_model_filename,
                require_audio=False)
        except Exception:
            return True  # never block start because the probe itself broke
        if probe.get("has_model"):
            return True
        if not probe.get("has_binary"):
            # Ship-side problem; nothing the user can do. Surface and
            # let session start so the existing error path takes over.
            self._add_system_bubble(
                "Local model runtime is missing from this build. "
                "Reinstall Touchless or use the cloud backend by setting "
                "OPENAI_API_KEY.")
            return False
        # has_binary + !has_model — user-recoverable. Offer the download.
        return self._offer_model_download(cfg, probe)

    def _offer_model_download(self, cfg, probe: dict) -> bool:
        """Show a confirm dialog with the GGUF download size. On Yes,
        run the download in a background thread with a system-bubble
        progress line, and start the session when done. Returns True
        when the model is now available, False otherwise."""
        from ...live_api.local_backend import (
            MODEL_DOWNLOAD_INFO, download_llm_model, preferred_model_download_dir,
        )
        filename = cfg.local_llm_model_filename
        info = MODEL_DOWNLOAD_INFO.get(filename) or {}
        label = info.get("label") or filename
        size_gb = (info.get("size_bytes") or 0) / 1_000_000_000
        dest_dir = preferred_model_download_dir()
        title = "Download local model?"
        detail = (
            f"Iris needs a local language model to run offline.\n\n"
            f"  Model:    {label}\n"
            f"  Size:     ~{size_gb:.1f} GB\n"
            f"  Saves to: {dest_dir}\n\n"
            "Downloaded once; reused across sessions and shared with "
            "the dictation grammar corrector. Cancel to use a different "
            "backend instead (set OPENAI_API_KEY).")
        try:
            from PySide6.QtWidgets import QMessageBox
            mbox = QMessageBox(self)
            mbox.setIcon(QMessageBox.Question)
            mbox.setWindowTitle(title)
            mbox.setText(detail)
            mbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            mbox.setDefaultButton(QMessageBox.Yes)
            QTimer.singleShot(0, lambda: self._tint_titlebar(mbox))
            if mbox.exec() != QMessageBox.Yes:
                self._add_system_bubble(
                    "Download skipped. Set OPENAI_API_KEY to use the "
                    "cloud backend, or click Start again to retry.")
                return False
        except Exception:
            return False
        progress_label = self._add_system_bubble(
            "Downloading local model… 0%")
        done_holder = {"ok": False, "msg": "", "path": None}
        cancel_flag = {"v": False}

        def _on_progress(downloaded: int, total: Optional[int]) -> None:
            if total:
                pct = int(downloaded * 100 / max(1, total))
                gb_d = downloaded / 1_000_000_000
                gb_t = total / 1_000_000_000
                txt = f"Downloading local model… {pct}% ({gb_d:.1f}/{gb_t:.1f} GB)"
            else:
                gb_d = downloaded / 1_000_000_000
                txt = f"Downloading local model… {gb_d:.2f} GB"
            # Marshal back to the GUI thread (this runs on the download
            # worker thread otherwise).
            QTimer.singleShot(0, lambda: progress_label.setText(txt))

        def _worker() -> None:
            ok, msg, path = download_llm_model(
                filename=filename,
                progress=_on_progress,
                cancelled=lambda: cancel_flag["v"])
            done_holder["ok"] = ok
            done_holder["msg"] = msg
            done_holder["path"] = path
            QTimer.singleShot(0, lambda: progress_label.setText(
                f"Download {'complete' if ok else 'failed'}: {msg}"))
        import threading
        t = threading.Thread(target=_worker, name="iris-model-download",
                             daemon=True)
        t.start()
        # Block the GUI loop without freezing it — pump events while we
        # wait, so progress updates render. This is a deliberate modal
        # wait; cancel is via the system close button → mbox above.
        from PySide6.QtCore import QCoreApplication
        while t.is_alive():
            QCoreApplication.processEvents()
            t.join(timeout=0.05)
        return bool(done_holder["ok"])

    def _tint_titlebar(self, widget) -> None:
        """Apply the Touchless deep-indigo titlebar tint to any window —
        used for ad-hoc dialogs (model download, etc.) that don't go
        through the usual showEvent path."""
        try:
            apply_touchless_titlebar(widget)
        except Exception:
            pass

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        # Fire cortex pulses IMMEDIATELY on Send (before send_user_text)
        # so the neuron view animates the instant the user clicks. The
        # realtime_client only emits voice-mic → core for audio
        # transcripts, not for typed input — so we synthesise the same
        # visual cue here: an anticipatory edge pulse along voice-mic →
        # core (the brightest, most striking effect), then a bigger
        # core flash. Each call is independently try/except'd so a
        # single emit failure can't break Send.
        try:
            from ...live_api import cortex_emit
        except Exception:
            cortex_emit = None  # type: ignore[assignment]
        # Send → core state will flip to THINKING (orange glow) via
        # the state-change signal. NO core pulse here — pulsing the
        # core on Send made it look like core was already outputting
        # before Iris had even read the prompt. The color change is
        # enough; real path pulses (planner → cap → tool → output)
        # take over from there.
        # transcript_received echoes the user's text, so we don't add the
        # user bubble here — the signal does it. Just close any open
        # assistant bubble so the next reply starts fresh.
        self._current_assistant_label = None
        if not self._manager.send_user_text(text):
            self._add_system_bubble("Not ready — start the session and wait for \"Ready\".")
            return
        # Start the cortex "thinking" pulse loop — fires a node_activity
        # on the core every ~600ms continuously while we wait for the
        # reply, so the cortex animation matches actual response time
        # (short for quick local replies, longer for slow cloud LLM).
        # Stopped on the first assistant_text signal (line 2189) or after
        # a 30s safety cap. Idempotent — calling twice during one turn
        # is harmless.
        try:
            self._start_cortex_thinking_loop()
        except Exception:
            pass
        # Show a typing indicator AFTER a short delay so the window
        # doesn't look frozen while Tier-2 plans / connectors execute /
        # realtime thinks. The delay (~250ms) means quick replies
        # (Layer-0 router / classifier hits) never flash the dots.
        self._typing_pending = True
        QTimer.singleShot(250, self._maybe_show_typing)

    def _start_cortex_thinking_loop(self) -> None:
        """Start the thinking-phase visualization.

        NO LOOPING. Fires a single orange core flash and arms a
        stuck-node detector: if no NEW cortex event has fired within
        STUCK_WINDOW_MS, the last-known active node gets a pulse so
        the user can see WHERE in the pipeline things are paused
        (e.g. waiting on an LLM call, slow tool, etc.). Cleared by
        the first reply byte in _on_assistant_delta.
        """
        import time as _t
        self._cortex_thinking_active = True
        self._cortex_thinking_start_time = _t.time()
        # Stuck detector state — last node we know is active. Updated
        # by the cortex_emit hooks (planner pre-dispatch, tool dispatch,
        # response done). Default to 'core' since that's where every
        # send begins.
        self._cortex_last_active_node = "core"
        self._cortex_last_event_ts = _t.time()
        try:
            from ...live_api import cortex_emit
            cortex_emit.node_activity("core", intensity=1.0, duration_ms=350)
        except Exception:
            pass
        # Arm the stuck-watcher (one-shot rescheduling, no scan loop).
        QTimer.singleShot(2000, self._check_cortex_stuck)

    def _check_cortex_stuck(self) -> None:
        """If processing has stalled at a node for >= STUCK_WINDOW_MS,
        pulse that node so the user sees where the bottleneck is.
        Re-arms itself while thinking is still active. NOT a path
        scanner — only pulses the actual last-active node."""
        if not getattr(self, "_cortex_thinking_active", False):
            return
        import time as _t
        now = _t.time()
        elapsed_total = (now - getattr(self, "_cortex_thinking_start_time", now)) * 1000
        if elapsed_total > 30000:
            self._cortex_thinking_active = False
            return
        since_last = (now - getattr(self, "_cortex_last_event_ts", now)) * 1000
        if since_last >= 1500:
            # No new path activity for 1.5s — pulse the LAST WORK node
            # so the user sees where we're stalled. SKIP 'core': core
            # is just the central connector, not an endpoint. If the
            # last node we know about IS core, suppress the pulse
            # (visually 'iris is between hops, not stuck').
            try:
                node = getattr(self, "_cortex_last_active_node", "core") or "core"
                if node and node != "core":
                    from ...live_api import cortex_emit
                    cortex_emit.node_activity(node, intensity=0.9, duration_ms=600)
            except Exception:
                pass
        QTimer.singleShot(1500, self._check_cortex_stuck)

    def _mark_cortex_active_node(self, node_id: str) -> None:
        """Called by cortex_emit hooks (or directly by the manager when
        a planner step / tool dispatch fires) to update the stuck
        detector's notion of where we currently are in the pipeline.

        Always updates the event timestamp (so stuck-watcher resets),
        but only updates the tracked node if it's NOT 'core' — core
        is a transient connector, never a meaningful 'stuck' target.
        """
        import time as _t
        self._cortex_last_event_ts = _t.time()
        if node_id and node_id != "core":
            self._cortex_last_active_node = node_id

    # Legacy compat — old code may still call _fire_thinking_pulse;
    # kept as a no-op shim. The thinking phase no longer loops;
    # _check_cortex_stuck handles 'stuck node' visualization instead.
    def _fire_thinking_pulse(self) -> None:
        return

    # ---- manager signal handlers (GUI thread) ----

    def _on_state_changed(self, state: LiveApiState, status: str) -> None:
        prev = self._current_state
        self._current_state = state
        self._set_state_pill(state, status)
        ready = state in (LiveApiState.LISTENING, LiveApiState.THINKING, LiveApiState.EXECUTING)
        self._input.setEnabled(ready)
        self._send_btn.setEnabled(ready)
        running = self._manager.is_running()
        self._start_btn.setText("Stop" if running else "Start")
        if ready and self._input.isEnabled():
            self._input.setFocus()
        # Task is fully done — clear any lingering dots. Catches the case
        # where a Tier-2 plan finished its last step but the post-output
        # _reshow_typing_if_busy timer fires AFTER the state transitioned.
        if state == LiveApiState.LISTENING and prev in (
                LiveApiState.THINKING, LiveApiState.EXECUTING):
            self._hide_typing()

    _BUSY_STATES = (LiveApiState.THINKING, LiveApiState.EXECUTING)

    def _reshow_typing_if_busy(self) -> None:
        """After an intermediate output (tool pill, text delta), bring the
        dots back so the user sees iris is still working. No-op if the
        task has reached a terminal state."""
        if self._current_state in self._BUSY_STATES:
            self._show_typing()

    def _on_transcript(self, text: str) -> None:
        self._current_assistant_label = None
        self._add_bubble(text, role="user")

    def _on_assistant_delta(self, delta: str) -> None:
        # Stop the cortex thinking-pulse loop — first sign of a reply.
        # The final core→output pulse is fired by LiveApiManager.
        self._cortex_thinking_active = False
        self._hide_typing()
        if self._current_assistant_label is None:
            self._current_assistant_label = self._add_bubble("", role="assistant")
        self._current_assistant_label.setText(self._current_assistant_label.text() + delta)

    def _on_assistant_break(self) -> None:
        # Close the current bubble so the next reply starts a fresh one
        # (separate messages instead of one growing text box). Then bring
        # the dots back if iris is still working — user expects to see
        # "still typing" between each visible output until the WHOLE task
        # is done (state -> LISTENING).
        self._current_assistant_label = None
        self._scroll_to_bottom()
        QTimer.singleShot(200, self._reshow_typing_if_busy)

    # Which layer ran a command — shown as a badge on each tool pill so the
    # user can see Touchless (free local), Connector (fast API), or iris
    # (GUI computer-use) handling the request. (badge, icon, color)
    _SOURCE_STYLE = {
        "touchless": ("TOUCHLESS", "⚡", "#1DE9B6"),
        "connector": ("CONNECTOR", "🔌", "#58E3FF"),
        "iris": ("IRIS", "👁", "#B388FF"),
    }

    def _on_tool_event(self, kind: str, info: dict) -> None:
        # Update cortex stuck-detector: a tool call/completion is a
        # known pipeline event — push the current node forward so the
        # stuck pulse lands on the right place if we hang waiting.
        name_for_mark = ""
        slug = "core"
        try:
            name_for_mark = str(info.get("name", "") or "")
            # Strip 'router/' prefix and use the tool slug as node id
            # (tool nodes in NODES dict use raw tool names).
            slug = name_for_mark.split("/", 1)[-1] if name_for_mark else "core"
            if slug:
                self._mark_cortex_active_node(slug if kind == "called" else "core")
        except Exception:
            pass
        # When a tool COMPLETES, fire the return-leg of the path:
        # tool-name → core, so the neuron doesn't visually die at the
        # tool node. The final core → output pulse fires later when
        # assistant_text emits (handled in LiveApiManager).
        if kind == "completed" and slug and slug != "core":
            try:
                from ...live_api import cortex_emit
                tool_node = f"tool-{slug}" if not slug.startswith("tool-") else slug
                cortex_emit.edge_pulse(tool_node, "core", color="orange",
                                        duration_ms=250)
            except Exception:
                pass
        # Each tool event = a visible output; drop the typing dots so the
        # pill takes their slot, then schedule a re-show so the dots come
        # back BETWEEN steps until the whole task settles to LISTENING.
        if kind == "called":
            self._hide_typing()
        if kind == "completed":
            QTimer.singleShot(200, self._reshow_typing_if_busy)
        name = str(info.get("name", "") or "")
        call_id = str(info.get("call_id", "") or name)
        source = str(info.get("source", "") or "")
        if not source:  # fallback for older event payloads
            source = "touchless" if name.startswith("router/") else "iris"
        badge, icon, color = self._SOURCE_STYLE.get(source, ("IRIS", "👁", "#B388FF"))
        display_name = name.split("/", 1)[-1] if name.startswith("router/") else name
        if kind == "called":
            label = self._add_tool_pill(f"{icon} {badge} · {display_name}…", color=color)
            self._tool_pills[call_id] = label
        elif kind == "completed":
            status = str(info.get("status", "") or "")
            ok = status == "ok"
            mark = "✓" if ok else "✕"
            if source == "touchless" and ok:
                text = f"{icon} {badge} · {display_name} — handled locally · 0 tokens {mark}"
            else:
                text = f"{icon} {badge} · {display_name} — {status or 'done'} {mark}"
            fill = color if ok else "#EF4444"
            label = self._tool_pills.get(call_id)
            if label is not None:
                label.setText(text)
                label.setStyleSheet(self._pill_style(fill))
            else:
                self._add_tool_pill(text, color=fill)

    def _on_error(self, message: str) -> None:
        self._hide_typing()
        self._add_bubble(f"⚠ {message}", role="error")

    # ---- Gmail connect (one-click OAuth) ----

    def _gmail_status(self) -> str:
        try:
            from ...live_api.connectors.google_client import status
            return status()
        except Exception:
            return "needs_libs"

    def _refresh_gmail_button(self) -> None:
        """Show/label the button per current state. Hidden when the Google
        libs aren't installed (nothing to connect) or no client is embedded."""
        st = self._gmail_status()
        if st in ("needs_libs", "needs_client"):
            self._gmail_btn.setVisible(False)
            return
        self._gmail_btn.setVisible(True)
        if st == "connected":
            self._gmail_btn.setText("Gmail ✓")
            self._gmail_btn.setEnabled(False)
        else:  # ready_to_connect
            self._gmail_btn.setText("Connect Gmail")
            self._gmail_btn.setEnabled(True)

    def _on_connect_gmail(self) -> None:
        self._gmail_btn.setEnabled(False)
        self._gmail_btn.setText("Connecting…")
        self._add_system_bubble("Opening your browser to connect Gmail — approve the consent screen.")

        def _worker() -> None:
            try:
                from ...live_api.connectors.google_client import connect
                ok, msg = connect()
            except Exception as exc:
                ok, msg = False, f"Gmail connect failed: {exc}"
            self._gmail_result.emit(ok, msg)

        threading.Thread(target=_worker, name="GmailConnect", daemon=True).start()

    def _on_gmail_result(self, ok: bool, message: str) -> None:
        self._add_system_bubble(("✓ " if ok else "⚠ ") + message)
        self._refresh_gmail_button()

    # ---- Microsoft 365 connect (one-click OAuth, mirrors Gmail) ----

    def _ms_status(self) -> str:
        try:
            from ...live_api.connectors.ms_graph_client import status
            return status()
        except Exception:
            return "needs_libs"

    def _refresh_ms_button(self) -> None:
        st = self._ms_status()
        if st in ("needs_libs", "needs_client"):
            self._ms_btn.setVisible(False)
            return
        self._ms_btn.setVisible(True)
        # Stay enabled when connected so the user can add ANOTHER account
        # (e.g. school + personal) and switch between them.
        self._ms_btn.setEnabled(True)
        self._ms_btn.setText("Microsoft ✓ (+ add)" if st == "connected" else "Connect Microsoft")

    def _on_connect_ms(self) -> None:
        self._ms_btn.setEnabled(False)
        self._ms_btn.setText("Connecting…")
        self._add_system_bubble("Opening your browser to connect Microsoft 365 — approve the sign-in.")

        def _worker() -> None:
            try:
                from ...live_api.connectors.ms_graph_client import MsGraphClient
                ok, msg = MsGraphClient.shared().connect()
            except Exception as exc:
                ok, msg = False, f"Microsoft connect failed: {exc}"
            self._ms_result.emit(ok, msg)

        threading.Thread(target=_worker, name="MsGraphConnect", daemon=True).start()

    def _on_ms_result(self, ok: bool, message: str) -> None:
        self._add_system_bubble(("✓ " if ok else "⚠ ") + message)
        self._refresh_ms_button()

    # ---- confirmation (cross-thread) ----

    def _confirm_tool(self, title: str, detail: str) -> bool:
        """Called on the websocket thread. Blocks until the GUI answers."""
        holder = {"event": threading.Event(), "result": False}
        self._confirm_request.emit(title, detail, holder)
        # Cap the wait so a closed window can't hang the WS thread forever.
        holder["event"].wait(timeout=120)
        return bool(holder["result"])

    def _on_confirm_request(self, title: str, detail: str, holder: dict) -> None:
        try:
            # Instantiate manually (not QMessageBox.question) so we can
            # paint the OS titlebar Touchless deep-indigo to match every
            # other Touchless window — the static .question() call doesn't
            # expose the box before it goes modal.
            mbox = QMessageBox(self)
            mbox.setIcon(QMessageBox.Question)
            mbox.setWindowTitle("Confirm action")
            mbox.setText(f"{title}\n\n{detail}")
            mbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            mbox.setDefaultButton(QMessageBox.No)
            # Apply the titlebar tint after the HWND exists. Qt creates it
            # lazily during show(), so this singleShot fires after the
            # event loop processes show — DwmSetWindowAttribute then has
            # a real window handle to colour.
            def _tint():
                try:
                    apply_touchless_titlebar(mbox)
                except Exception:
                    pass
            QTimer.singleShot(0, _tint)
            holder["result"] = mbox.exec() == QMessageBox.Yes
        finally:
            holder["event"].set()

    # ---- transcript helpers ----

    def _set_state_pill(self, state: LiveApiState, status: str) -> None:
        label, color = _STATE_STYLE.get(state, (status, "#64748B"))
        self._state_pill.setText(status or label)
        self._state_pill.setStyleSheet(
            f"background:{color}22; color:{color}; border:1px solid {color}88; "
            f"border-radius:11px; padding:3px 12px; font-weight:700; font-size:11px;"
        )

    def _pill_style(self, color: str) -> str:
        return (
            f"background:{color}18; color:{color}; border:1px solid {color}55; "
            f"border-radius:8px; padding:6px 10px; font-size:11px; "
            f"font-family:Consolas,monospace;"
        )

    def _add_tool_pill(self, text: str, *, color: str = "#58E3FF") -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(self._pill_style(color))
        self._insert_row(label, align=Qt.AlignLeft)
        return label

    def _add_system_bubble(self, text: str) -> QLabel:
        return self._add_bubble(text, role="system")

    # ---- inline action chips ----
    # Renders one row of small buttons beneath an assistant reply (e.g.
    # "Connect Gmail" / "Connect Outlook" / "Read Outlook screen" after
    # the email cascade returns an actionable error). Click handlers
    # route to existing UI flows so the chat doesn't need to duplicate
    # logic.

    _ACTION_LABELS: Dict[str, str] = {
        "connect_gmail": "Connect Gmail",
        "connect_ms": "Connect Outlook",
        "read_outlook_screen": "Read Outlook screen",
    }

    def _on_suggested_actions(self, actions: List[str]) -> None:
        """Slot for LiveApiManager.suggested_actions. Builds an inline
        row of clickable chips so the user can connect a missing
        account (or trigger a screen-read) in one click without
        leaving the chat."""
        if not actions:
            return
        row_widget = QWidget()
        rh = QHBoxLayout(row_widget)
        rh.setContentsMargins(8, 0, 8, 0)
        rh.setSpacing(6)
        any_added = False
        for action_id in actions:
            label = self._ACTION_LABELS.get(action_id)
            if not label:
                continue
            btn = QPushButton(label)
            btn.setStyleSheet(self._button_style(subtle=True))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda _checked=False, aid=action_id, b=btn:
                    self._run_chip_action(aid, b))
            rh.addWidget(btn)
            any_added = True
        if not any_added:
            return
        rh.addStretch(1)
        self._insert_row(row_widget, align=Qt.AlignLeft)

    def _run_chip_action(self, action_id: str, btn: QPushButton) -> None:
        """Dispatch an inline action chip click to the matching
        existing handler."""
        try:
            btn.setEnabled(False)
        except Exception:
            pass
        if action_id == "connect_gmail":
            self._on_connect_gmail()
        elif action_id == "connect_ms":
            self._on_connect_ms()
        elif action_id == "read_outlook_screen":
            self._add_system_bubble(
                "Opening Outlook and reading it now…")
            try:
                self._manager.send_user_text(
                    "open Outlook and read the inbox on screen, then "
                    "summarize the unread messages")
            except Exception as exc:
                self._add_bubble(f"⚠ Couldn't trigger screen read: {exc}",
                                 role="error")
                try:
                    btn.setEnabled(True)
                except Exception:
                    pass
        else:
            try:
                btn.setEnabled(True)
            except Exception:
                pass

    # ---- typing-dots indicator ----

    # Animation parameters. TICK_MS * PERIOD = ~1.1s per full cycle —
    # fast enough to read as active, slow enough for the bounce to land.
    _TYPING_TICK_MS = 55
    _TYPING_PERIOD = 20
    _TYPING_DOT_COUNT = 3
    _TYPING_BUBBLE_W = 78
    _TYPING_BUBBLE_H = 34

    def _maybe_show_typing(self) -> None:
        """Called by the 250ms QTimer after _on_send. Only actually shows
        the dots if a response hasn't already arrived (in which case
        _typing_pending was cleared by _hide_typing)."""
        if self._typing_pending:
            self._show_typing()

    def _show_typing(self) -> None:
        """Insert an assistant-style typing bubble with three bouncing /
        fading dots. Idempotent — if already showing, returns early."""
        if self._typing_row is not None:
            return
        self._typing_pending = False
        pal = self._palette
        # Bubble: same dark slate as assistant bubbles, FIXED size so it
        # doesn't reflow as the dots bounce and pulse inside it.
        bubble = QWidget()
        bubble.setFixedSize(self._TYPING_BUBBLE_W, self._TYPING_BUBBLE_H)
        bubble.setStyleSheet("background:#1E293B; border-radius:12px;")
        bh = QHBoxLayout(bubble)
        bh.setContentsMargins(14, 0, 14, 0)
        bh.setSpacing(7)
        bh.setAlignment(Qt.AlignCenter)
        dots: list = []
        effects: list = []
        for _ in range(self._TYPING_DOT_COUNT):
            dot = QLabel("●")
            # Blue (#58E3FF, same as the CONNECTOR pill accent) so
            # the dots read as Iris/AI-activity, not generic UI chrome.
            dot.setStyleSheet(
                "color:#58E3FF; font-size:14px; "
                "background:transparent;"
            )
            dot.setAlignment(Qt.AlignCenter)
            dot.setFixedSize(14, self._TYPING_BUBBLE_H)
            # Per-dot opacity effect so each dot fades independently.
            effect = QGraphicsOpacityEffect(dot)
            effect.setOpacity(0.15)
            dot.setGraphicsEffect(effect)
            bh.addWidget(dot)
            dots.append(dot)
            effects.append(effect)
        # Wrap in a row so the bubble hugs the LEFT of the transcript.
        row = QWidget()
        rh = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(bubble)
        rh.addStretch(1)
        self._transcript.insertWidget(self._transcript.count() - 1, row)
        self._scroll_to_bottom()
        self._typing_row = row
        self._typing_dot_widgets = dots
        self._typing_dot_effects = effects
        self._typing_step = 0
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(self._TYPING_TICK_MS)
        self._typing_timer.timeout.connect(self._tick_typing)
        self._typing_timer.start()
        self._tick_typing()  # paint frame 0 immediately

    def _tick_typing(self) -> None:
        if not self._typing_dot_effects:
            return
        period = self._TYPING_PERIOD
        # Each dot is offset by 1/3 of the period so they appear to
        # bounce left-to-right in a wave (iMessage-style).
        stagger = period // self._TYPING_DOT_COUNT
        t = self._typing_step
        for i, effect in enumerate(self._typing_dot_effects):
            phase = (2 * math.pi * ((t - i * stagger) % period)) / period
            # Smooth 0..1 sine wave, shifted so each dot starts dim.
            wave = 0.5 + 0.5 * math.sin(phase - math.pi / 2)
            # Fade: 0.12 (almost gone) -> 1.0 (solid) - more
            # dramatic so the wave reads clearly at small size.
            effect.setOpacity(0.12 + 0.88 * wave)
            # Bounce: negative top margin lifts the dot UP within the
            # fixed-height container. Max lift is 5px (slightly more
            # than before, to match the larger dots).
            dot = self._typing_dot_widgets[i]
            offset = int(round(-5 * wave))
            dot.setContentsMargins(0, offset, 0, -offset)
        self._typing_step = (self._typing_step + 1) % period

    def _hide_typing(self) -> None:
        # Cancel a pending delayed show - reply arrived inside the
        # 250ms window before _maybe_show_typing could fire.
        self._typing_pending = False
        if self._typing_timer is not None:
            self._typing_timer.stop()
            self._typing_timer.deleteLater()
            self._typing_timer = None
        if self._typing_row is not None:
            self._transcript.removeWidget(self._typing_row)
            self._typing_row.deleteLater()
            self._typing_row = None
        self._typing_dot_widgets = []
        self._typing_dot_effects = []
        self._typing_step = 0

    def _bubble_max_width(self) -> int:
        """Compute the max width for chat bubble labels.

        Derived from the current chat container width so bubbles wrap
        properly when the splitter is dragged or the window is resized.
        Falls back to a sensible default if the container isn't realized
        yet. The -32 accounts for layout padding + scrollbar gutter.
        """
        try:
            container = getattr(self, "_chat_container", None)
            if container is not None:
                w = container.width()
                if w > 0:
                    return max(220, w - 32)
        except Exception:
            pass
        return 480

    def _refresh_bubble_widths(self) -> None:
        try:
            target = self._bubble_max_width()
            for lbl in list(self._bubble_labels):
                try:
                    lbl.setMaximumWidth(target)
                except Exception:
                    pass
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            from PySide6.QtCore import QEvent
            if (
                obj is getattr(self, "_chat_container", None)
                and event.type() == QEvent.Resize
            ):
                self._refresh_bubble_widths()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _add_bubble(self, text: str, *, role: str) -> QLabel:
        pal = self._palette
        styles = {
            "user": (f"background:{pal['primary']}; color:{pal['text']};", Qt.AlignRight),
            "assistant": (f"background:#1E293B; color:{pal['text']};", Qt.AlignLeft),
            "system": ("background:transparent; color:#94A3B8; font-style:italic;", Qt.AlignHCenter),
            "error": ("background:#7F1D1D; color:#FECACA;", Qt.AlignLeft),
        }
        css, align = styles.get(role, styles["assistant"])
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        radius = "12px"
        pad = "9px 12px" if role != "system" else "2px 8px"
        label.setStyleSheet(f"{css} border-radius:{radius}; padding:{pad}; font-size:13px;")
        label.setMaximumWidth(self._bubble_max_width())
        try:
            self._bubble_labels.append(label)
        except Exception:
            pass
        self._insert_row(label, align=align)
        return label

    def _insert_row(self, widget: QWidget, *, align) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if align == Qt.AlignRight:
            row.addStretch(1)
            row.addWidget(widget)
        elif align == Qt.AlignHCenter:
            row.addStretch(1)
            row.addWidget(widget)
            row.addStretch(1)
        else:
            row.addWidget(widget)
            row.addStretch(1)
        # Insert before the trailing stretch (last item).
        self._transcript.insertLayout(self._transcript.count() - 1, row)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(0, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    def _clear_transcript(self) -> None:
        # Stop the typing timer first so it doesn't fire on a label that's
        # about to be deleted by the loop below.
        self._hide_typing()
        # Remove every row except the trailing stretch.
        while self._transcript.count() > 1:
            item = self._transcript.takeAt(0)
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    sub = layout.takeAt(0)
                    w = sub.widget()
                    if w is not None:
                        w.deleteLater()
                layout.deleteLater()
            elif item.widget() is not None:
                item.widget().deleteLater()
        self._tool_pills.clear()
        self._current_assistant_label = None

    # ---- lifecycle ----

    def show_window(self) -> None:
        # Apply the Touchless title bar BEFORE show() so Windows never
        # paints the default white caption — without this you see a
        # ~50ms white-window flash on first open while Windows draws the
        # default chrome before our showEvent recolors it.
        # winId() forces native HWND allocation without making the window
        # visible, giving DwmSetWindowAttribute a real handle to act on.
        try:
            self.winId()  # force HWND creation
            apply_touchless_titlebar(self)
        except Exception:
            pass
        # Window was created with opacity=0 in __init__ to swallow Qt's
        # default first-paint frame (white client + native chrome). show()
        # now triggers that frame invisibly, then we restore opacity to 1.0
        # on the next event-loop tick — by which point the title bar is
        # already indigo and the client is already the surface color, so
        # the user only ever sees the fully-styled window.
        self.show()
        self.raise_()
        self.activateWindow()
        QTimer.singleShot(0, lambda: self.setWindowOpacity(1.0))
        # Kick off world-payload preload on a background thread so the
        # cortex warm path has data ready by the time the user clicks
        # Iris. Idempotent (only spawns once per window lifetime).
        try:
            self._start_payload_preload()
        except Exception:
            pass
        # Eagerly construct the QWebEngineView + bridge + channel with
        # an EMPTY world payload so the renderer spawn, GPU surface
        # init, page boot, and three.js scene build all complete during
        # app-show time (panel is still collapsed so nothing is visible
        # to the user). When the user clicks Iris later, the view is
        # already warm and the expand happens instantly. When the
        # background payload worker finishes, _on_payload_ready will
        # re-inject window.IRIS_WORLD with the real data and trigger a
        # view.reload() so the page picks up the populated payload.
        #
        # Deferred one tick so the assistant window paints first — the
        # warm pass spawns the QWebEngine renderer process which can
        # briefly steal CPU during its handshake.
        try:
            QTimer.singleShot(0, self._eager_warm_view)
        except Exception:
            pass

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        # Safety net: if show_window() wasn't used (e.g. Qt re-shows the
        # window after a state change), the title bar still gets painted.
        try:
            apply_touchless_titlebar(self)
        except Exception:
            pass
        # Overlay: keep the Iris window ABOVE other windows so it stays visible
        # while it opens/arranges apps. Done via Win32 HWND_TOPMOST (coexists
        # with the custom titlebar; a Qt flag change here can hide the window).
        if self._overlay_on_top:
            self._set_topmost(True)
        # Re-arm Cortex render + typing-dots animation now that the
        # Iris window is back on screen. If the user minimized while
        # Iris was open, we paused both to free GPU/CPU for Touchless;
        # showEvent restores them.
        try:
            self._on_iris_visibility_change(visible=True)
        except Exception:
            pass

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        # User hid / closed-to-tray the Iris window. Park all the
        # animations so the WebEngine helper process stops contending
        # with the Touchless camera pipeline for GPU.
        try:
            self._on_iris_visibility_change(visible=False)
        except Exception:
            pass

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Minimize / restore lives here, not in show/hideEvent. We
        # want minimized-while-still-shown to also pause Cortex.
        try:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.WindowStateChange:
                minimized = bool(self.windowState() & Qt.WindowMinimized)
                self._on_iris_visibility_change(visible=not minimized)
        except Exception:
            pass
        super().changeEvent(event)

    def _on_iris_visibility_change(self, *, visible: bool) -> None:
        """Single funnel for show/hide/minimize/restore. Pauses the two
        animations that compete with Touchless's camera pipeline when
        the user can't see Iris anyway:
          * Cortex Three.js render loop (large GPU win)
          * Typing-dots QTimer (tiny CPU win, but pointless when hidden)
        """
        # Only pause Cortex if the panel is currently expanded — if
        # collapsed we already paused it on collapse.
        try:
            expanded = bool(self._cortex_panel_expanded())
        except Exception:
            expanded = False
        if expanded:
            try:
                self._set_cortex_render_paused(not visible)
            except Exception:
                pass
        # Typing dots: a 55ms QTimer wakeup that's wasted CPU when the
        # window isn't visible. Pause and restore.
        try:
            if not visible and self._typing_timer is not None:
                self._typing_timer.stop()
            elif visible and self._typing_row is not None \
                    and self._typing_timer is not None:
                if not self._typing_timer.isActive():
                    self._typing_timer.start()
        except Exception:
            pass

    def _cortex_panel_expanded(self) -> bool:
        """True when the embedded Cortex panel is currently showing."""
        panel = self._embedded_cortex_panel
        if panel is None:
            return False
        try:
            return panel.isVisible() and panel.width() > 4
        except Exception:
            return False

        # NOTE: we intentionally do NOT pre-warm the embedded Cortex view
        # on a fixed timer here anymore. Warming with an empty payload
        # caused the view to render hardcoded demo data and then visibly
        # swap to the real cortex once the background worker finished.
        # The view is now built EXACTLY ONCE, by _on_payload_ready, the
        # moment the real payload is in hand — no flash, no swap. If the
        # user clicks Iris before that, _expand_cortex shows a loading
        # overlay and queues the expand for after the build completes.
        #
        # The persisted "was expanded" preference is honored from
        # _warm_cortex() (which runs once payload is ready) rather than
        # here, so we don't auto-expand into an empty/demo view.
        pass

    _overlay_on_top = True  # Iris stays as an always-on-top overlay by default

    def _set_topmost(self, on: bool) -> None:
        """Make this window always-on-top (overlay) or normal, via Win32."""
        try:
            import ctypes
            hwnd = int(self.winId())
            HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
            SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST if on else HWND_NOTOPMOST,
                0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        try:
            self._manager.stop()
        except Exception:
            pass
        # Tear down the embedded cortex view + QWebChannel cleanly to
        # avoid dangling C++ references when the page tries to fire one
        # more event after Python-side teardown has started.
        try:
            if self._embedded_cortex_view is not None:
                page = self._embedded_cortex_view.page()
                if page is not None:
                    page.setWebChannel(None)
        except Exception:
            pass
        # If we registered as the active bridge globally, unregister so
        # the standalone CortexWindow (or a later embed) can take over
        # cleanly without colliding on cortex_emit.
        try:
            from ...live_api.cortex.bridge import (
                get_active_bridge,
                set_active_bridge,
            )
            if (
                self._embedded_cortex_bridge is not None
                and get_active_bridge() is self._embedded_cortex_bridge
            ):
                set_active_bridge(None)
        except Exception:
            pass
        # Drop the tool_executor JS-runner fallback — the view we
        # registered is about to be destroyed, so any further
        # runJavaScript on it would touch freed C++ state.
        try:
            from ...live_api import tool_executor as _te_mod
            if hasattr(_te_mod, "set_cortex_js_runner"):
                _te_mod.set_cortex_js_runner(None)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    """Standalone launcher for isolated testing."""
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = LiveAssistantWindow()
    win.show_window()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov
