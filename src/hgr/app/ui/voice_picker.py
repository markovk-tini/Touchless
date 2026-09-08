"""Voice picker dialog for choosing Iris's spoken voice.

Lists the OpenAI Realtime voice options with a short description and
a Preview button that hits the OpenAI TTS endpoint (gpt-4o-mini-tts)
and plays the returned MP3 locally via QMediaPlayer. Click a row to
select that voice; Save updates the live LiveApiConfig + persists
to QSettings so the choice survives restarts.

Requires OPENAI_API_KEY in the env. If the key is missing or the
network call fails, the row shows a status message and stays usable
(other voices still selectable; just no preview audio).

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QSettings, QStandardPaths, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)


# Voice options. Tuple of (id, display name, blurb shown to user).
# Order is the user-facing order. Newer voices (marin / cedar / ballad)
# come first because they're MUCH more consistent within a session — the
# older ones (sage, coral, verse) can drift in tone/gender mid-session,
# which users notice as "the voice sounds different now".
VOICES: List[Tuple[str, str, str]] = [
    ("marin",   "Marin",   "Warm, expressive female · most consistent (recommended)"),
    ("cedar",   "Cedar",   "Warm, natural male · consistent"),
    ("ballad",  "Ballad",  "Narrator, expressive · great for stories/explanations"),
    ("coral",   "Coral",   "Warm, expressive · female-leaning (can drift)"),
    ("verse",   "Verse",   "Polished narrator · versatile (can drift)"),
    ("ash",     "Ash",     "Smooth · even-keeled male"),
    ("onyx",    "Onyx",    "Deep, authoritative · JARVIS energy"),
    ("alloy",   "Alloy",   "Neutral · conversational"),
    ("shimmer", "Shimmer", "Soft · gentle female"),
    ("sage",    "Sage",    "Calm, considered · prone to tone drift"),
]

# Rotating test lines so successive previews don't sound identical.
TEST_LINES = [
    "Hi, I'm Iris. How can I help?",
    "Done. The file is saved.",
    "Got it. Opening Chrome for you now.",
    "Memory updated. I'll remember that.",
    "Three tools available. Pick one.",
]

# OpenAI TTS endpoint. gpt-4o-mini-tts supports the full set of
# newer voices (sage, coral, ash, verse, etc.) — older models don't.
_TTS_URL = "https://api.openai.com/v1/audio/speech"
_TTS_MODEL = "gpt-4o-mini-tts"

_SETTINGS_ORG = "Touchless"
_SETTINGS_APP = "Touchless"
_SETTINGS_KEY = "live_api/voice"


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[voice-picker {time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def load_saved_voice(default: str = "marin") -> str:
    """Read the persisted voice id from QSettings, fall back to default."""
    try:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        value = s.value(_SETTINGS_KEY, default)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception as exc:
        _log(f"load_saved_voice failed: {exc}")
    return default


def save_voice(voice_id: str) -> None:
    """Persist the chosen voice id to QSettings."""
    try:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(_SETTINGS_KEY, voice_id)
        s.sync()
    except Exception as exc:
        _log(f"save_voice failed: {exc}")


class _PreviewFetcher(QObject):
    """Background TTS fetcher. Emits ``ready(voice_id, path)`` on
    success or ``failed(voice_id, message)`` on error."""

    ready = Signal(str, str)
    failed = Signal(str, str)

    def __init__(self, api_key: Optional[str], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._api_key = api_key
        self._tmp_paths: List[str] = []

    def request(self, voice_id: str, text: str) -> None:
        if not self._api_key:
            self.failed.emit(voice_id, "OPENAI_API_KEY not set")
            return
        thread = threading.Thread(
            target=self._do_fetch, args=(voice_id, text),
            name=f"VoicePreview:{voice_id}", daemon=True,
        )
        thread.start()

    def cleanup(self) -> None:
        for p in self._tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        self._tmp_paths.clear()

    def _do_fetch(self, voice_id: str, text: str) -> None:
        try:
            payload = json.dumps({
                "model": _TTS_MODEL,
                "voice": voice_id,
                "input": text,
            }).encode("utf-8")
            req = urllib.request.Request(
                _TTS_URL, method="POST", data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                audio = resp.read()
            tmp = tempfile.NamedTemporaryFile(
                prefix=f"iris_voice_{voice_id}_", suffix=".mp3", delete=False,
            )
            tmp.write(audio)
            tmp.close()
            self._tmp_paths.append(tmp.name)
            self.ready.emit(voice_id, tmp.name)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                body = ""
            self.failed.emit(voice_id, f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            self.failed.emit(voice_id, f"network: {e.reason}")
        except Exception as exc:
            self.failed.emit(voice_id, f"{type(exc).__name__}: {exc}")


class _VoiceRow(QFrame):
    """One row in the picker — name, blurb, preview button, status."""

    clicked = Signal(str)         # emits voice_id when the row body is clicked
    preview_clicked = Signal(str) # emits voice_id when ▶ Preview is clicked

    def __init__(self, voice_id: str, name: str, blurb: str,
                 palette: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._voice_id = voice_id
        self._palette = palette
        self._selected = False
        self._previewing = False
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(12)

        self._dot = QLabel("○")
        self._dot.setFixedWidth(14)
        self._dot.setAlignment(Qt.AlignCenter)
        self._dot.setStyleSheet(f"color: {palette['accent']}; font-size: 14px;")
        row.addWidget(self._dot)

        name_blurb = QVBoxLayout()
        name_blurb.setSpacing(2)
        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet(
            f"color: {palette['text']}; font-weight: 700; font-size: 14px;"
        )
        self._blurb_lbl = QLabel(blurb)
        self._blurb_lbl.setStyleSheet(
            "color: #94A3B8; font-size: 11px;"
        )
        name_blurb.addWidget(self._name_lbl)
        name_blurb.addWidget(self._blurb_lbl)
        row.addLayout(name_blurb, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #94A3B8; font-size: 10px;")
        self._status_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        row.addWidget(self._status_lbl)

        self._preview_btn = QPushButton("▶ Test")
        self._preview_btn.setFixedWidth(78)
        self._preview_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._preview_btn.setStyleSheet(self._preview_btn_style())
        self._preview_btn.clicked.connect(self._on_preview)
        row.addWidget(self._preview_btn)

        self._refresh_style()

    def voice_id(self) -> str:
        return self._voice_id

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Click anywhere on the row (except the preview button) selects.
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._voice_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._dot.setText("●" if selected else "○")
        self._refresh_style()

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status_lbl.setText(text or "")
        self._status_lbl.setStyleSheet(
            f"color: {'#EF4444' if error else '#1DE9B6'}; font-size: 10px;"
            if text else "color: #94A3B8; font-size: 10px;"
        )

    def set_previewing(self, on: bool) -> None:
        self._previewing = on
        self._preview_btn.setEnabled(not on)
        self._preview_btn.setText("…" if on else "▶ Test")

    def _on_preview(self) -> None:
        self.preview_clicked.emit(self._voice_id)

    def _refresh_style(self) -> None:
        pal = self._palette
        bg = f"rgba(29,233,182,0.10)" if self._selected else "transparent"
        border = pal['accent'] if self._selected else "transparent"
        self.setStyleSheet(
            f"_VoiceRow {{ background:{bg}; border: 1px solid {border}55; "
            f"border-radius: 8px; }}"
            f"_VoiceRow:hover {{ background: rgba(255,255,255,0.04); }}"
        )

    def _preview_btn_style(self) -> str:
        pal = self._palette
        return (
            f"QPushButton {{ background: rgba(255,255,255,0.04); "
            f"color: {pal['text']}; "
            f"border: 1px solid {pal['accent']}55; "
            f"border-radius: 6px; padding: 5px 10px; font-size: 11px; }}"
            f"QPushButton:hover {{ border-color: {pal['accent']}; }}"
            f"QPushButton:disabled {{ color: #64748B; border-color: #33415544; }}"
        )


class VoicePickerDialog(QDialog):
    """Modeless dialog letting the user pick + preview a voice for Iris."""

    voice_chosen = Signal(str)  # emitted when user clicks Save

    def __init__(self, *, current_voice: str, api_key: Optional[str],
                 palette: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Iris Voice")
        self.setModal(False)
        self.setMinimumWidth(460)
        self._palette = palette
        self._selected_voice = current_voice
        self._rows: List[_VoiceRow] = []
        # r51: install indigo chrome (previously no chrome — showed
        # OS-default white on Win10).
        from .window_chrome import install_indigo_chrome
        self._body = install_indigo_chrome(self, "Iris Voice")

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(0.85)
        # When playback finishes, clear the "previewing" state on the
        # row that was playing.
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._currently_playing_voice: Optional[str] = None

        self._fetcher = _PreviewFetcher(api_key, self)
        self._fetcher.ready.connect(self._on_preview_ready)
        self._fetcher.failed.connect(self._on_preview_failed)

        self._line_index = 0

        self._build_ui()
        self._select_internal(current_voice)

    # ---- UI ----

    def _build_ui(self) -> None:
        pal = self._palette
        self.setStyleSheet(
            f"QDialog {{ background-color: {pal['surface']}; color: {pal['text']}; }}"
        )
        root = QVBoxLayout(self._body)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(10)

        header = QLabel("Choose Iris's voice")
        header.setStyleSheet(
            f"color: {pal['text']}; font-weight: 800; font-size: 16px;"
        )
        sub = QLabel(
            "Click ▶ Test to hear each voice say a short line. Pick "
            "one and click Save — the change applies on the next "
            "session start."
        )
        sub.setStyleSheet("color: #94A3B8; font-size: 11px;")
        sub.setWordWrap(True)
        root.addWidget(header)
        root.addWidget(sub)

        # Voice rows
        for vid, name, blurb in VOICES:
            r = _VoiceRow(vid, name, blurb, self._palette, self)
            r.clicked.connect(self._select_internal)
            r.preview_clicked.connect(self._preview_voice)
            root.addWidget(r)
            self._rows.append(r)

        # Footer with status + Save/Cancel
        footer = QHBoxLayout()
        self._footer_status = QLabel("")
        self._footer_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        footer.addWidget(self._footer_status, 1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self._btn_style(subtle=True))
        cancel_btn.clicked.connect(self.reject)
        footer.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(self._btn_style(subtle=False))
        save_btn.clicked.connect(self._on_save)
        footer.addWidget(save_btn)

        root.addLayout(footer)

    def _btn_style(self, *, subtle: bool) -> str:
        pal = self._palette
        bg = "transparent" if subtle else pal["primary"]
        return (
            f"QPushButton {{ background:{bg}; color:{pal['text']}; "
            f"border:1px solid {pal['accent']}55; border-radius:8px; "
            f"padding:7px 16px; font-weight:700; font-size:12px; }}"
            f"QPushButton:hover {{ border:1px solid {pal['accent']}; }}"
        )

    # ---- selection ----

    def _select_internal(self, voice_id: str) -> None:
        self._selected_voice = voice_id
        for r in self._rows:
            r.set_selected(r.voice_id() == voice_id)

    def _on_save(self) -> None:
        save_voice(self._selected_voice)
        self.voice_chosen.emit(self._selected_voice)
        self.accept()

    # ---- preview playback ----

    def _preview_voice(self, voice_id: str) -> None:
        # Cancel any current playback so we don't overlap.
        try:
            self._player.stop()
        except Exception:
            pass
        # Mark current row as previewing.
        for r in self._rows:
            r.set_previewing(r.voice_id() == voice_id)
            r.set_status("")

        self._footer_status.setText("Fetching…")
        line = TEST_LINES[self._line_index % len(TEST_LINES)]
        self._line_index += 1
        self._fetcher.request(voice_id, line)

    def _on_preview_ready(self, voice_id: str, path: str) -> None:
        self._footer_status.setText("")
        self._currently_playing_voice = voice_id
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _on_preview_failed(self, voice_id: str, message: str) -> None:
        for r in self._rows:
            r.set_previewing(False)
        self._footer_status.setText(f"Preview failed: {message}")
        for r in self._rows:
            if r.voice_id() == voice_id:
                r.set_status("failed", error=True)
                # Auto-clear after 4s so the row isn't stuck looking broken.
                QTimer.singleShot(4000, lambda r=r: r.set_status(""))
                break

    def _on_playback_state(self, state) -> None:
        # When playback stops (finished or interrupted), clear the
        # "previewing" indicator on the row that was playing.
        if state == QMediaPlayer.StoppedState and self._currently_playing_voice:
            for r in self._rows:
                if r.voice_id() == self._currently_playing_voice:
                    r.set_previewing(False)
                    break
            self._currently_playing_voice = None

    # ---- lifecycle ----

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        try:
            self._player.stop()
        except Exception:
            pass
        try:
            self._fetcher.cleanup()
        except Exception:
            pass
        super().closeEvent(event)
