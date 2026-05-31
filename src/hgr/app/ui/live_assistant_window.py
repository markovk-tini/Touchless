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
from typing import Optional

import math

from PySide6.QtCore import Qt, QTimer, Signal
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
    QVBoxLayout,
    QWidget,
)

from ...live_api.live_api_manager import LiveApiManager, LiveApiState
from .custom_gestures_chrome import apply_touchless_titlebar


class _ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` on left mouse press. Used for the
    "Iris" header so the user can open the Cortex visualization."""

    clicked = Signal()

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
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


class LiveAssistantWindow(QWidget):
    """Modeless chat window wrapping a LiveApiManager (text-only)."""

    # Marshals a risky-tool confirmation from the websocket thread onto
    # the GUI thread. `holder` is a dict with an Event + result slot.
    _confirm_request = Signal(str, str, object)

    # Marshals Gmail-connect results from the OAuth worker thread to the UI.
    _gmail_result = Signal(bool, str)
    # Marshals Microsoft-365-connect results from its worker thread.
    _ms_result = Signal(bool, str)

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

        self._manager = LiveApiManager(text_only=True)
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
        # Current LiveApiState (mirrored from _on_state_changed). Used by
        # _reshow_typing_if_busy to decide whether to bring the dots back
        # between intermediate outputs (THINKING/EXECUTING = still
        # working; LISTENING = task done, stay hidden).
        self._current_state: Optional[LiveApiState] = None

        self._build_ui()
        self._wire_manager()

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
        title.setToolTip("Open the Iris Cortex visualization")
        title.setStyleSheet(
            f"QLabel {{ font-size: 18px; font-weight: 800; color: {pal['text']}; }}"
            f"QLabel:hover {{ color: {pal['accent']}; }}"
        )
        title.clicked.connect(self._open_cortex_window)
        header.addWidget(title)
        header.addStretch(1)
        self._state_pill = QLabel("Off")
        self._state_pill.setAlignment(Qt.AlignCenter)
        self._set_state_pill(LiveApiState.OFF, "Off")
        header.addWidget(self._state_pill)
        root.addLayout(header)

        backend = "local" if str(getattr(self._manager.config, "backend", "cloud")).lower() == "local" else "cloud"
        sub = QLabel(f"Backend: {backend} · model: {getattr(self._manager.config, 'model', '?')}")
        sub.setStyleSheet("font-size: 11px; color: #94A3B8;")
        root.addWidget(sub)

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
        root.addWidget(self._scroll, 1)

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
        root.addLayout(input_row)

        # Controls row.
        controls = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_toggle_session)
        self._start_btn.setStyleSheet(self._button_style())
        controls.addWidget(self._start_btn)

        # One-click Gmail connect (only shown when the Google libs are present
        # and not yet connected). Runs the OAuth consent flow in the browser.
        self._gmail_btn = QPushButton("Connect Gmail")
        self._gmail_btn.setStyleSheet(self._button_style(subtle=True))
        self._gmail_btn.clicked.connect(self._on_connect_gmail)
        controls.addWidget(self._gmail_btn)
        self._gmail_result.connect(self._on_gmail_result)
        self._refresh_gmail_button()

        # One-click Microsoft 365 connect (same pattern as Gmail).
        self._ms_btn = QPushButton("Connect Microsoft")
        self._ms_btn.setStyleSheet(self._button_style(subtle=True))
        self._ms_btn.clicked.connect(self._on_connect_ms)
        controls.addWidget(self._ms_btn)
        self._ms_result.connect(self._on_ms_result)
        self._refresh_ms_button()

        controls.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_transcript)
        clear_btn.setStyleSheet(self._button_style(subtle=True))
        controls.addWidget(clear_btn)
        root.addLayout(controls)

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
        except Exception as exc:
            self._add_system_bubble(f"⚠ Couldn't open Iris Cortex: {exc}")

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

    # ---- session control ----

    def _on_toggle_session(self) -> None:
        if self._manager.is_running():
            self._manager.stop()
            self._start_btn.setText("Start")
        else:
            self._add_system_bubble("Starting session…")
            self._manager.start()
            self._start_btn.setText("Stop")

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        # transcript_received echoes the user's text, so we don't add the
        # user bubble here — the signal does it. Just close any open
        # assistant bubble so the next reply starts fresh.
        self._current_assistant_label = None
        if not self._manager.send_user_text(text):
            self._add_system_bubble("Not ready — start the session and wait for \"Ready\".")
            return
        # Show a typing indicator AFTER a short delay so the window
        # doesn't look frozen while Tier-2 plans / connectors execute /
        # realtime thinks. The delay (~250ms) means quick replies
        # (Layer-0 router / classifier hits) never flash the dots.
        self._typing_pending = True
        QTimer.singleShot(250, self._maybe_show_typing)

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
        label.setMaximumWidth(440)
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
