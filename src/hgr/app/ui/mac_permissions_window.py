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

This wizard shows one row per permission with a SINGLE state-aware control:
an accent "Enabled ✓" badge when granted, an "Enable" button when not (which
fires the in-process prompt for camera/mic or deep-links to the exact System
Settings pane for the rest), or a greyed "Blocked" when MDM/parental controls
forbid it. A ~1.2 s poll keeps the controls current so the user watches grants
land without reopening the window. Colors come from the live app theme so the
dialog matches the rest of the Touchless UI.

Windows/Linux never see this window — it is only constructed on darwin (the
caller guards on ``sys.platform == "darwin"``). It is a plain modal
``QDialog`` (like the privacy prompt and the update dialog): at first run the
engine/live camera is not running yet, so there is no hand-driven cursor to
keep alive, and a modal is the simplest correct choice.

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
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
     "why": "Control apps like Spotify and Chrome — approved per app on first use.",
     "restart": False, "info_only": True},
)


# In a source (dev) run we can't meaningfully change TCC, and the developer
# has usually already granted everything — which makes the wizard impossible
# to exercise. So in source builds we SIMULATE: permissions start ungranted
# and clicking a row toggles an in-memory store, giving a full visual test
# loop (Enable -> Enabled -> Relaunch hint, and back) without touching real
# TCC. Frozen (shipped) builds always read the real OS state.
_DEV_SIMULATE = not getattr(sys, "frozen", False)
_DEV_STATE: dict[str, str] = {}


def _dev_default(key: str) -> str:
    if key == "automation":
        return "perapp"
    if key in ("camera", "microphone"):
        return "pending"   # never asked -> renders "Enable"
    return "denied"


# ---- status probes (all safe/no-op off macOS) --------------------------------

def _status(key: str) -> str:
    """Return 'granted' | 'denied' | 'pending' | 'restricted' | 'perapp' for a
    permission key. 'pending' = not decided yet (macOS will still prompt).
    'restricted' = blocked by MDM/parental controls (user can't grant it).
    'perapp' is the Automation row (granted per-target on first use)."""
    if _DEV_SIMULATE:
        # Dev/source: report the simulated store so the wizard can be tested
        # with everything starting disabled. Real TCC is never consulted.
        return _DEV_STATE.get(key, _dev_default(key))
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


def permission_granted(key: str) -> bool:
    """Public: True if the given TCC permission ('camera' | 'microphone' |
    'accessibility' | 'screen_recording') is granted. Always True off macOS.
    Feature code uses this to gate an action and, when False, pop the wizard
    highlighting the offending permission (see MainWindow.ensure_mac_permission)."""
    if sys.platform != "darwin":
        return True
    return _status(key) == "granted"


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

    def __init__(self, parent: Optional[QWidget] = None,
                 highlight: Optional[str] = None) -> None:
        super().__init__(parent)
        apply_touchless_chrome(self)
        self.setWindowTitle("Touchless — macOS Permissions")
        self.setMinimumWidth(540)
        self.setSizeGripEnabled(False)
        # When opened because a feature was blocked, `highlight` names the
        # permission row to emphasise (accent glow) so the user sees exactly
        # which grant unblocks what they just tried to do.
        self._highlight = highlight if highlight in _ANCHORS else None

        # Pull the live theme so the dialog matches the rest of the app
        # instead of hardcoding a palette. Falls back to the brand defaults
        # when a parent/config isn't available (e.g. a standalone test).
        cfg = getattr(parent, "config", None)
        self._accent = str(getattr(cfg, "accent_color", None) or "#1DE9B6")
        self._text = str(getattr(cfg, "text_color", None) or "#E5F6FF")
        # Match the main app PAGE background (surface_color, the slate #0F172A),
        # NOT primary_color (the old royal-blue brand accent) — that mismatch
        # was why the wizard's blue looked off against the rest of the UI.
        self._surface = str(getattr(cfg, "surface_color", None) or "#0F172A")
        self.setStyleSheet(self._dialog_qss())

        # One control (button) per permission row, keyed by permission key,
        # plus a last-rendered-state cache so the 1.2s poll only re-styles a
        # control when its state actually changes (avoids flicker).
        self._controls: dict[str, QPushButton] = {}
        self._last_state: dict[str, str] = {}
        self._relaunch_btn: Optional[QPushButton] = None
        self._build_ui()

        # Snapshot restart-gated permissions' grant state at open. A grant
        # already in place when this process launched is already effective —
        # only a grant that flips DURING this session needs a relaunch, so
        # only those surface the "reopen" hint + Relaunch button.
        self._granted_at_open = {
            row["key"]: (_status(row["key"]) == "granted")
            for row in _ROWS if row.get("restart")
        }
        # Snapshot every row's state at open so Done can stay greyed until the
        # user actually changes a permission (any row differing from here).
        self._opening_states = {row["key"]: _status(row["key"]) for row in _ROWS}

        # Live refresh so grants land visibly without reopening the window.
        self._poll = QTimer(self)
        self._poll.setInterval(1200)
        self._poll.timeout.connect(self._refresh)
        self._poll.start()
        self._refresh(force=True)
        # Stop the poll and release the dialog once it closes (Done, Esc, or
        # the window close button all emit finished). Without this the hidden,
        # parent-owned dialog would keep probing TCC APIs every 1.2s for the
        # app's life, and reopening from Settings would stack timers.
        self.finished.connect(self._poll.stop)
        self.finished.connect(self.deleteLater)

    # ---- styling -------------------------------------------------------------

    @staticmethod
    def _rgba(hex_color: str, alpha: float) -> str:
        c = QColor(hex_color)
        return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"

    def _dialog_qss(self) -> str:
        """Base chrome: matches the settings panel — translucent innerCard
        rows on the primary background, accent-tinted borders, and a footer
        with an accent 'Done' primary action + a flat 'Relaunch' link."""
        return (
            f"QDialog {{ background: {self._surface}; color: {self._text}; }}"
            f"QLabel {{ color: {self._text}; background: transparent; }}"
            "QFrame#permRow {"
            "  background: rgba(255,255,255,0.04);"
            f"  border: 1px solid {self._rgba(self._accent, 0.22)};"
            "  border-radius: 16px;"
            "}"
            # Emphasised row when the wizard is opened for a specific blocked
            # feature — brighter accent fill + a 2px accent ring.
            "QFrame#permRowHi {"
            f"  background: {self._rgba(self._accent, 0.12)};"
            f"  border: 2px solid {self._rgba(self._accent, 0.90)};"
            "  border-radius: 16px;"
            "}"
            # Done = secondary action, greyed out until a permission actually
            # changes (disabled state below).
            "QPushButton#doneBtn {"
            f"  background: {self._rgba(self._accent, 0.16)};"
            f"  color: {self._text};"
            f"  border: 1px solid {self._rgba(self._accent, 0.55)};"
            "  border-radius: 12px; padding: 8px 22px; font-weight: 800;"
            "}"
            f"QPushButton#doneBtn:hover {{ background: {self._rgba(self._accent, 0.28)}; }}"
            "QPushButton#doneBtn:disabled {"
            "  background: rgba(127,127,127,0.10);"
            f"  color: {self._rgba(self._text, 0.40)};"
            "  border: 1px solid transparent;"
            "}"
            # Relaunch = a real PRIMARY button (solid accent), shown only once
            # a restart-gated grant flips this session.
            "QPushButton#relaunchBtn {"
            f"  background: {self._accent}; color: #04121f;"
            "  border: none; border-radius: 12px; padding: 8px 20px; font-weight: 800;"
            "}"
            f"QPushButton#relaunchBtn:hover {{ background: {self._rgba(self._accent, 0.82)}; }}"
        )

    def _control_qss(self, kind: str) -> str:
        """Per-control stylesheet for the single row control.
          - 'action'  : grey translucent, accent on hover (the settings-panel
                        button language) — used for Enable / Manage.
          - 'granted' : accent-tinted badge — used for Enabled ✓ (still
                        clickable to open Settings and review / turn off).
          - 'blocked' : faint grey, non-actionable — MDM/parental Restricted."""
        base = (
            "  border-radius: 12px; padding: 8px 16px;"
            "  font-weight: 800; min-width: 116px;"
        )
        if kind == "granted":
            return (
                "QPushButton {"
                f"  background: {self._rgba(self._accent, 0.16)};"
                f"  color: {self._accent};"
                f"  border: 1px solid {self._rgba(self._accent, 0.55)};"
                f"{base}"
                "}"
                f"QPushButton:hover {{ background: {self._rgba(self._accent, 0.24)}; }}"
            )
        if kind == "blocked":
            return (
                "QPushButton {"
                "  background: rgba(127,127,127,0.10);"
                "  color: rgba(229,246,255,0.45);"
                "  border: 1px solid transparent;"
                f"{base}"
                "}"
            )
        # 'action'
        return (
            "QPushButton {"
            "  background: rgba(255,255,255,0.08);"
            f"  color: {self._text};"
            "  border: 1px solid rgba(255,255,255,0.18);"
            f"{base}"
            "}"
            f"QPushButton:hover {{"
            f"  background: {self._rgba(self._accent, 0.20)};"
            f"  border: 1px solid {self._rgba(self._accent, 0.85)};"
            f"}}"
        )

    # ---- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(12)

        hi_name = None
        if self._highlight:
            hi_name = next((r["name"] for r in _ROWS if r["key"] == self._highlight), None)

        title = QLabel(
            f"Enable {hi_name} to continue" if hi_name
            else "Touchless needs a few macOS permissions"
        )
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(
            f"That action needs {hi_name} access (highlighted below). "
            "Grant it, then try again — you can review the rest here too."
            if hi_name else
            "Grant these so hand tracking, voice, and desktop control work. "
            "You can change them anytime in System Settings — or reopen this "
            "from Settings ▸ General ▸ Permissions."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: 12px; color: {self._rgba(self._text, 0.82)};")
        root.addWidget(subtitle)

        for row in _ROWS:
            root.addWidget(self._build_row(row))

        self._note = QLabel(
            "Accessibility and Screen Recording take effect after you reopen "
            "Touchless."
        )
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"font-size: 11px; color: {self._rgba(self._text, 0.60)};")
        root.addWidget(self._note)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)
        # Done: greyed out until a permission actually changes (see _refresh).
        self._done_btn = QPushButton("Done")
        self._done_btn.setObjectName("doneBtn")
        self._done_btn.setCursor(Qt.PointingHandCursor)
        self._done_btn.clicked.connect(self.accept)
        self._done_btn.setEnabled(False)
        footer.addWidget(self._done_btn)
        # Relaunch: a real primary button at bottom-right, shown only once a
        # restart-gated grant flips this session. Label avoids '&' — Qt would
        # eat it as a mnemonic (that turned "Quit & Reopen" into "Quit  Reopen").
        self._relaunch_btn = QPushButton("Relaunch Touchless")
        self._relaunch_btn.setObjectName("relaunchBtn")
        self._relaunch_btn.setCursor(Qt.PointingHandCursor)
        self._relaunch_btn.setToolTip(
            "Quit and reopen Touchless so newly-granted Accessibility / "
            "Screen Recording permissions take effect."
        )
        self._relaunch_btn.clicked.connect(self._restart_app)
        self._relaunch_btn.setVisible(False)
        footer.addWidget(self._relaunch_btn)
        root.addLayout(footer)

    def _build_row(self, row: dict) -> QFrame:
        key = row["key"]
        frame = QFrame()
        frame.setObjectName("permRowHi" if key == self._highlight else "permRow")
        frame.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(12)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        name = QLabel(row["name"])
        name.setStyleSheet("font-size: 14px; font-weight: 700;")
        why = QLabel(row["why"])
        why.setWordWrap(True)
        why.setStyleSheet(f"font-size: 11px; color: {self._rgba(self._text, 0.72)};")
        text_col.addWidget(name)
        text_col.addWidget(why)
        lay.addLayout(text_col, 1)

        # ONE control per row. Its label + style carry the whole state; there
        # is no separate status pill (that dual "Granted ✓ + Granted ✓" was
        # the confusing bit). _refresh() drives label/style/enabled per state.
        btn = QPushButton("…")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _=False, k=key: self._on_control(k))
        self._controls[key] = btn
        lay.addWidget(btn, 0)
        return frame

    # ---- actions -------------------------------------------------------------

    def _on_control(self, key: str) -> None:
        """Single click handler per row. Fires the REAL request API — which
        shows the little native permission prompt (while the permission is
        still undecided) AND registers Touchless under that permission in
        System Settings — instead of dumping the user in the full Settings
        page. Automation has no grant API (it's approved per-app on first
        control), so it opens the Automation list."""
        state = _status(key)
        if key == "automation":
            self._open_pane(key)
            return
        if state == "restricted":
            return
        if state == "granted" and not _DEV_SIMULATE:
            # Already on; there's nothing to prompt. Reviewing / turning it
            # off is a System Settings task, so open that specific pane.
            self._open_pane(key)
            return
        # Fire the real request. Dev-sim forces the 'pending' path so the
        # prompt/registration always fires, then flips the simulated badge so
        # the enabled/disabled loop stays exercisable in a source build.
        self._enable(key, "pending" if _DEV_SIMULATE else state)
        if _DEV_SIMULATE:
            _DEV_STATE[key] = "denied" if state == "granted" else "granted"
        self._refresh(force=True)

    def _enable(self, key: str, state: str) -> None:
        """Trigger the little native permission prompt for `key` and register
        Touchless under that permission.

        macOS only shows the camera/mic prompt while the status is 'pending'
        (undecided) — once decided it NEVER re-prompts, so a decided camera/mic
        falls back to its specific Settings pane (the only way left to change
        it). Accessibility and Screen Recording have no in-place grant at all:
        their request API shows a little 'Open System Settings' prompt and
        registers the app in the list — exactly the popup we want, so we do NOT
        also open the full-page pane for them."""
        if key == "camera":
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
            try:
                caps.is_accessibility_trusted(prompt=True)
            except Exception:
                pass
        elif key == "screen_recording":
            try:
                caps.is_screen_recording_trusted(prompt=True)
            except Exception:
                pass

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

    def _control_view(self, key: str, state: str, reopen_pending: bool) -> tuple:
        """(label, qss_kind, enabled) for a row's single control."""
        if key == "automation":
            return "Manage", "action", True
        if state == "granted":
            return ("Enabled • reopen" if reopen_pending else "Enabled ✓"), "granted", True
        if state == "restricted":
            return "Blocked", "blocked", False
        return "Enable", "action", True

    def _refresh(self, force: bool = False) -> None:
        any_reopen = False
        for row in _ROWS:
            key = row["key"]
            state = _status(key)
            reopen_pending = (
                bool(row.get("restart"))
                and state == "granted"
                and not self._granted_at_open.get(key, False)
            )
            any_reopen = any_reopen or reopen_pending
            btn = self._controls.get(key)
            if btn is None:
                continue
            # Only re-render when something actually changed (avoids the
            # per-tick stylesheet re-polish flicker). Encode reopen into the
            # cache key so the hint appears the moment a grant flips.
            cache_key = f"{state}:{int(reopen_pending)}"
            if not force and self._last_state.get(key) == cache_key:
                continue
            self._last_state[key] = cache_key
            label, kind, enabled = self._control_view(key, state, reopen_pending)
            btn.setText(label)
            btn.setEnabled(enabled)
            btn.setStyleSheet(self._control_qss(kind))

        if self._relaunch_btn is not None:
            self._relaunch_btn.setVisible(any_reopen)
        # Done stays greyed until at least one permission differs from its
        # state when the wizard opened.
        done = getattr(self, "_done_btn", None)
        if done is not None:
            changed = any(
                _status(row["key"]) != self._opening_states.get(row["key"])
                for row in _ROWS
            )
            done.setEnabled(changed)
