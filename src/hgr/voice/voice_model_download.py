"""Optional runtime download of the higher-accuracy whisper dictation
model (ggml-medium.en.bin).

Why this exists: the Microsoft Store build ships ONLY small.en to stay
under the Store's EXE/MSI package-size limit. The app is fully
functional on small.en, so medium.en is an OPTIONAL accuracy upgrade
the user pulls on demand via the "Voice Recognition Upgrade" button.
Because the app works without it and nothing blocks on the download,
this is runtime DLC — Store-policy compliant, unlike an install-time
downloader (which is what got the stub installer rejected).

The whisper resolver (live_api/local_backend.py / voice/whisper_stream.py)
already scans ~/Documents/TouchlessVoiceModels/ and prefers medium.en
when present, so a successful download here is auto-picked up on the
next dictation session / app restart with no further wiring.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..utils.runtime_paths import resource_path

MEDIUM_MODEL_FILENAME = "ggml-medium.en.bin"
MEDIUM_MODEL_URL = (
    "https://pub-3116ebd541fa4ca18a84371667d029fe.r2.dev/models/ggml-medium.en.bin"
)
# Exact byte size of the model on R2 — used to (a) report total progress
# before the server's Content-Length is read, and (b) sanity-check the
# finished file so a truncated/partial download isn't mistaken for a
# complete model.
MEDIUM_MODEL_SIZE_BYTES = 1533774781
# Minimum plausible size for "a real medium model is present" — guards
# medium_model_present() against counting a half-written temp file.
_MIN_VALID_MODEL_BYTES = 1_000_000_000


def voice_model_dir() -> Path:
    """User-writable dir the whisper resolver already scans first."""
    return Path.home() / "Documents" / "TouchlessVoiceModels"


def medium_model_present() -> bool:
    """True when the higher-accuracy model is already available — either
    bundled in this build (website builds ship it) or previously
    downloaded into a user model dir. Drives the upgrade button's
    visibility: present -> hide the button, absent -> show it."""
    candidates = [
        voice_model_dir() / MEDIUM_MODEL_FILENAME,
        Path.home() / "Documents" / "HGRVoiceModels" / MEDIUM_MODEL_FILENAME,
        resource_path("whisper.cpp", "models", MEDIUM_MODEL_FILENAME),
    ]
    for path in candidates:
        try:
            if path.exists() and path.stat().st_size >= _MIN_VALID_MODEL_BYTES:
                return True
        except Exception:
            continue
    return False


class VoiceModelDownloader(QObject):
    """Downloads ggml-medium.en.bin into the user model dir on a daemon
    thread, marshalling progress/finish to the GUI thread via signals.

    Downloads to a `.part` file and renames on success so an interrupted
    download never leaves a half-file the resolver would try to load."""

    progress = Signal(int, int)     # downloaded_bytes, total_bytes
    finished = Signal(bool, str)    # success, message

    _CHUNK = 1024 * 1024            # 1 MB read chunks
    _PROGRESS_EVERY = 4 * 1024 * 1024  # emit progress every ~4 MB

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._cancelled.clear()
        self._thread = threading.Thread(
            target=self._run, name="voice-model-download", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()

    def _run(self) -> None:
        target_dir = voice_model_dir()
        final_path = target_dir / MEDIUM_MODEL_FILENAME
        part_path = target_dir / (MEDIUM_MODEL_FILENAME + ".part")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.finished.emit(False, f"Couldn't create the model folder: {exc}")
            return
        try:
            req = urllib.request.Request(
                MEDIUM_MODEL_URL,
                headers={"User-Agent": "Touchless-VoiceModel/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                total = MEDIUM_MODEL_SIZE_BYTES
                try:
                    total = int(resp.headers.get("Content-Length") or total)
                except (TypeError, ValueError):
                    pass
                downloaded = 0
                last_emit = 0
                with open(part_path, "wb") as fh:
                    while True:
                        if self._cancelled.is_set():
                            self.finished.emit(False, "Download cancelled.")
                            return
                        chunk = resp.read(self._CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_emit >= self._PROGRESS_EVERY:
                            self.progress.emit(downloaded, total or downloaded)
                            last_emit = downloaded
                self.progress.emit(downloaded, total or downloaded)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._cleanup(part_path)
            self.finished.emit(False, f"Download failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover
            self._cleanup(part_path)
            self.finished.emit(False, f"Unexpected error: {type(exc).__name__}: {exc}")
            return
        # Sanity-check the downloaded size before promoting the .part to
        # the real filename — a truncated file must not be mistaken for a
        # complete model.
        try:
            actual = part_path.stat().st_size
        except OSError:
            self.finished.emit(False, "Download finished but the file is missing.")
            return
        if actual < _MIN_VALID_MODEL_BYTES:
            self._cleanup(part_path)
            self.finished.emit(
                False,
                "Download looks incomplete (too small) — please try again.",
            )
            return
        try:
            if final_path.exists():
                final_path.unlink()
            part_path.replace(final_path)
        except OSError as exc:
            self.finished.emit(False, f"Couldn't finalize the model file: {exc}")
            return
        self.finished.emit(True, "Voice recognition upgrade installed.")

    @staticmethod
    def _cleanup(part_path: Path) -> None:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

# Author: Konstantin Markov
