"""First-run macOS permission onboarding wizard.

macOS gates the features Touchless depends on behind separate TCC
permissions that CANNOT be granted programmatically:

  - Camera            hand tracking (AVFoundation)
  - Microphone        voice commands + dictation (AVFoundation)
  - Accessibility     synthetic cursor/keyboard via CGEvent
  - Screen Recording  instant clips, recordings, foreign-window titles
  - Automation        AppleScript control of Spotify / Chrome / etc.

Camera + Microphone prompt in-process and take effect immediately; the
other three must be toggled by the user in System Settings and (for
Accessibility / Screen Recording) require an app RESTART to activate.

This wizard shows one row per permission with a live status pill and a
one-click action that either fires the in-process prompt (camera/mic) or
opens the exact System Settings pane (deep link) and registers the app
there. A ~1.2 s poll keeps the pills current so the user watches grants
land without reopening the window.

Windows/Linux never see this window — it is only constructed on darwin
(the caller guards on ``sys.platform == "darwin"``). It is a plain modal
``QDialog`` (like the privacy prompt and the update dialog): at first run
the engine/live camera is not running yet, so there is no hand-driven
cursor to keep alive, and a modal is the simplest correct choice.

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...platform_compat import capabilities as caps
from ..ui.window_chrome import apply_touchless_chrome


# System Settings > Privacy & Security deep-link anchors. `open <url>` jumps
# straight to the relevant pane (far better UX than dumping the user at the
# top of System Settings). Same `subprocess.Popen(["open", ...])` idiom the
# rest of the mac code uses to launch apps.
_PRIVACY_PANE = "x-apple.systempreferences:com.apple.preference.security"
_ANCHORS = {
    "camera": "Privacy_Camera",
    "microphone": "Privacy_Microphone",
    "accessibility": "Privacy_Accessibility",
    "screen_recording": "Privacy_ScreenCapture",
    "automation": "Privacy_Automation",
}

# Ordered row definitions. `restart` marks permissions whose effect only
# activates after the app is relaunched (the TCC trust bit flips immediately,
# but the CGEvent/capture APIs bind it at process start). `info_only` marks
# Automation, which is granted per-target on first Apple Event and can't be
# proactively toggled or reliably probed for an arbitrary app.
_ROWS = (
    {"key": "camera", "name": "Camera",
     "why": "See your hands for gesture control.", "restart": False},
    {"key": "microphone", "name": "Microphone",
     "why": "Hear voice commands and dictation.", "restart": False},
    {"key": "accessibility", "name": "Accessibility",
     "why": "Move the cursor and send clicks & keystrokes.", "restart": True},
    {"key": "screen_recording", "name": "Screen Recording",
     "why": "Capture clips, recordings, and window titles.", "restart": True},
    {"key": "automation", "name": "Automation",
     "why": "Control apps like Spotify and Chrome.",
     "restart": False, "info_only": True},
)


# ---- status probes (all safe/no-op off macOS) --------------------------------

def _status(key: str) -> str:
    """Return 'granted' | 'denied' | 'pending' | 'perapp' for a permission key.
    'pending' = not decided yet (macOS will still prompt). 'perapp' is the
    Automation row (granted per-target on first use — nothing to check)."""
    try:
        if key == "camera":
            state = caps.camera_permission_state()
        elif key == "microphone":
            state = caps.microphone_permission_state()
        elif key == "accessibility":
            return "granted" if caps.is_accessibility_trusted(prompt=False) else "denied"
        elif key == "screen_recording":
            return "granted" if caps.is_screen_recording_trusted(prompt=False) else "denied"
        elif key == "automation":
            return "perapp"
        else:
            return "denied"
    except Exception:
        return "denied"
    # camera / microphone AVFoundation states
    if state == "authorized":
        return "granted"
    if state == "notDetermined":
        return "pending"
    if state == "restricted":
        # MDM / parental controls: the user CANNOT grant this — surface it as
        # a distinct, non-actionable state rather than a red "Needed" that
        # sends them to a Settings pane with a disabled toggle.
        return "restricted"
    return "denied"


def has_all_critical_permissions() -> bool:
    """True when every permission the app can actually verify is granted
    (Automation excluded — it can't be probed generically). The caller uses
    this to skip auto-showing the wizard when there's nothing to ask for."""
    if sys.platform != "darwin":
        return True
    return all(
        _status(k) == "granted"
        for k in ("camera", "microphone", "accessibility", "screen_recording")
    )


def _app_bundle_path() -> Optional[str]:
    """Path to the running Touchless.app bundle, or None when running from
    source (dev) where there's nothing to relaunch."""
    if not getattr(sys, "frozen", False):
        return None
    try:
        from pathlib import Path

        exe = Path(sys.executable).resolve()
        # <Touchless.app>/Contents/MacOS/Touchless -> the .app
        if exe.parent.name == "MacOS" and exe.parent.parent.name == "Contents":
            return str(exe.parent.parent.parent)
    except Exception:
        return None
    return None


class MacPermissionsWizard(QDialog):
    """Modal onboarding dialog. Constructed only on macOS."""

    # palette matches UpdateDialog / the app's dark-blue chrome
    _BG = "#0B3D91"
    _TEXT = "#E5F6FF"
    _ACCENT = "#1DE9B6"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        apply_touchless_chrome(self)
        self.setWindowTitle("Touchless — macOS Permissions")
        self.setMinimumWidth(520)
        self.setSizeGripEnabled(False)
        self.setStyleSheet(
            f"QDialog {{ background: {self._BG}; color: {self._TEXT}; }}"
            f"QLabel {{ color: {self._TEXT}; }}"
            "QPushButton {"
            f"  background: {self._ACCENT}; color: #003d2a; border: none;"
            "  border-radius: 8px; padding: 7px 14px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #29f0c1; }"
            "QPushButton:disabled {"
            "  background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.5);"
            "}"
            "QPushButton#linkBtn {"
            "  background: transparent; color: rgba(255,255,255,0.82);"
            "  font-weight: 600; padding: 6px 6px;"
            "}"
            "QPushButton#linkBtn:hover { color: white; }"
            "QFrame#permRow {"
            "  background: rgba(0,0,0,0.20);"
            "  border: 1px solid rgba(255,255,255,0.10);"
            "  border-radius: 10px;"
            "}"
        )
        # Per-row widget handles, keyed by permission key.
        self._pills: dict[str, QLabel] = {}
        self._buttons: dict[str, QPushButton] = {}
        self._build_ui()

        # Snapshot restart-gated permissions' grant state at open. A grant
        # already in place when this process launched is already effective —
        # only a grant that flips DURING this session needs a relaunch, so
        # only those show the "• reopen" hint.
        self._granted_at_open = {
            row["key"]: (_status(row["key"]) == "granted")
            for row in _ROWS if row.get("restart")
        }

        # Live refresh so grants land visibly without reopening the window.
        self._poll = QTimer(self)
        self._poll.setInterval(1200)
        self._poll.timeout.connect(self._refresh)
        self._poll.start()
        self._refresh()
        # Stop the poll and release the dialog once it closes (Done, Esc, or
        # the window close button all emit finished). Without this the hidden,
        # parent-owned dialog would keep probing TCC APIs every 1.2s for the
        # app's life, and reopening from Settings would stack timers.
        self.finished.connect(self._poll.stop)
        self.finished.connect(self.deleteLater)

    # ---- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(12)

        title = QLabel("Touchless needs a few macOS permissions")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(
            "Grant these so hand tracking, voice, and desktop control work. "
            "You can change them anytime in System Settings — or reopen this "
            "from Settings ▸ General ▸ Permissions."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 12px; color: rgba(229,246,255,0.82);")
        root.addWidget(subtitle)

        for row in _ROWS:
            root.addWidget(self._build_row(row))

        note = QLabel(
            "Accessibility and Screen Recording take effect after you "
            "reopen Touchless."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px; color: rgba(229,246,255,0.62);")
        root.addWidget(note)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        restart_btn = QPushButton("Quit & Reopen")
        restart_btn.setObjectName("linkBtn")
        restart_btn.setCursor(Qt.PointingHandCursor)
        restart_btn.setToolTip(
            "Relaunch Touchless so newly-granted Accessibility / Screen "
            "Recording permissions take effect."
        )
        restart_btn.clicked.connect(self._restart_app)
        footer.addWidget(restart_btn)
        footer.addStretch(1)
        done_btn = QPushButton("Done")
        done_btn.setCursor(Qt.PointingHandCursor)
        done_btn.clicked.connect(self.accept)
        footer.addWidget(done_btn)
        root.addLayout(footer)

    def _build_row(self, row: dict) -> QFrame:
        key = row["key"]
        frame = QFrame()
        frame.setObjectName("permRow")
        frame.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        name = QLabel(row["name"])
        name.setStyleSheet("font-size: 14px; font-weight: 700;")
        why = QLabel(row["why"])
        why.setWordWrap(True)
        why.setStyleSheet("font-size: 11px; color: rgba(229,246,255,0.72);")
        text_col.addWidget(name)
        text_col.addWidget(why)
        lay.addLayout(text_col, 1)

        pill = QLabel("…")
        pill.setAlignment(Qt.AlignCenter)
        pill.setMinimumWidth(96)
        self._pills[key] = pill
        lay.addWidget(pill, 0)

        if not row.get("info_only"):
            btn = QPushButton("Grant")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._on_action(k))
            self._buttons[key] = btn
            lay.addWidget(btn, 0)
        else:
            info_btn = QPushButton("Open")
            info_btn.setObjectName("linkBtn")
            info_btn.setCursor(Qt.PointingHandCursor)
            info_btn.setToolTip(
                "Automation is granted per app the first time Touchless "
                "controls it. This opens the list so you can review it."
            )
            info_btn.clicked.connect(lambda _=False, k=key: self._open_pane(k))
            self._buttons[key] = info_btn
            lay.addWidget(info_btn, 0)

        return frame

    # ---- actions -------------------------------------------------------------

    def _on_action(self, key: str) -> None:
        state = _status(key)
        if state == "restricted":
            # MDM / parental controls block this; nothing the user can do here.
            return
        if key == "camera":
            # notDetermined -> in-process prompt; already decided -> the OS
            # won't re-ask, so send the user to the exact Settings pane.
            if state == "pending":
                caps.request_camera_access()
            else:
                self._open_pane(key)
        elif key == "microphone":
            if state == "pending":
                caps.request_microphone_access()
            else:
                self._open_pane(key)
        elif key == "accessibility":
            # prompt=True registers the app under the Accessibility list AND
            # shows the OS prompt; also open the pane so the toggle is right
            # in front of the user.
            try:
                caps.is_accessibility_trusted(prompt=True)
            except Exception:
                pass
            self._open_pane(key)
        elif key == "screen_recording":
            try:
                caps.is_screen_recording_trusted(prompt=True)
            except Exception:
                pass
            self._open_pane(key)
        self._refresh()

    def _open_pane(self, key: str) -> None:
        anchor = _ANCHORS.get(key)
        url = f"{_PRIVACY_PANE}?{anchor}" if anchor else _PRIVACY_PANE
        try:
            subprocess.Popen(["open", url])
        except Exception:
            # Fall back to just opening System Settings if the deep link
            # is rejected on this macOS version.
            try:
                subprocess.Popen(["open", "-a", "System Settings"])
            except Exception:
                pass

    def _restart_app(self) -> None:
        """Relaunch the app so restart-gated grants activate. Detach a tiny
        shell that waits for our PID to exit then reopens the bundle; then
        quit. From source (no bundle) we can only quit — the dev relaunches."""
        bundle = _app_bundle_path()
        if bundle:
            try:
                pid = os.getpid()
                script = (
                    f'while kill -0 {pid} 2>/dev/null; do sleep 0.3; done; '
                    f'sleep 0.5; open "{bundle}"'
                )
                subprocess.Popen(
                    ["/bin/bash", "-c", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                pass
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.quit()
        except Exception:
            pass

    # ---- polling refresh -----------------------------------------------------

    def _refresh(self) -> None:
        for row in _ROWS:
            key = row["key"]
            state = _status(key)
            pill = self._pills.get(key)
            btn = self._buttons.get(key)
            # "• reopen" only for a restart-gated grant that flipped during
            # THIS session; one already in place at launch is already active.
            reopen_pending = (
                bool(row.get("restart"))
                and state == "granted"
                and not self._granted_at_open.get(key, False)
            )
            if pill is not None:
                text, color = self._pill_style(state, reopen_pending)
                pill.setText(text)
                pill.setStyleSheet(
                    f"background: {color}; color: #04121f; border-radius: 9px;"
                    "padding: 3px 8px; font-size: 11px; font-weight: 700;"
                )
            if btn is not None and not row.get("info_only"):
                # Only "pending" (never asked) and "denied" (can re-request /
                # deep-link) are actionable. "granted" and "restricted"
                # (MDM-blocked) disable the button so we never offer a Grant
                # that would dead-end at a disabled Settings toggle.
                actionable = state in ("pending", "denied")
                btn.setEnabled(actionable)
                if state == "granted":
                    btn.setText("Granted ✓")
                elif state == "restricted":
                    btn.setText("Blocked")
                else:
                    btn.setText("Grant")

    @staticmethod
    def _pill_style(state: str, reopen_pending: bool) -> tuple[str, str]:
        if state == "granted":
            label = "Granted • reopen" if reopen_pending else "Granted ✓"
            return label, "#1DE9B6"
        if state == "restricted":
            return "Restricted", "#9EC1FF"
        if state == "pending":
            return "Not set", "#FFD166"
        if state == "perapp":
            return "Per app", "#9EC1FF"
        return "Needed", "#FF8A8A"
