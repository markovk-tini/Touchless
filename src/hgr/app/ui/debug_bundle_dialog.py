"""In-app "Save debug bundle before closing?" dialog (v1.1.9.2 r17+).

Fires from MainWindow.closeEvent on every clean close of Touchless.
Gives the user a chance to save a single-zip triage bundle wherever
they want, without needing to launch the Touchless_Debug.bat wrapper
first. Also carries a "Don't ask again (re-enable from Settings)"
checkbox that flips config.skip_debug_bundle_prompt_on_close.

Behavior guarantees:
- The dialog runs its collector on a QThread, so the GUI thread never
  freezes during a 5-10 s bundle write.
- Skip closes with Rejected. Save closes with Accepted after the
  QThread finishes. Both codes are readable via dialog.exec()'s
  return, but callers should just read `dialog.skip_forever` for the
  "Don't ask again" state and rely on the closeEvent to proceed
  whether the user picked Save or Skip.
- Wrapped in try/except at every entry point so a broken import
  (missing PySide6 module, corrupt collector) never strands the user
  with a half-closed app.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QStandardPaths
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..debug_bundle import BundleContext, collect_debug_bundle


class _BundleWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)   # path
    failed = Signal(str)         # message

    def __init__(self, ctx: BundleContext, output_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._output_path = output_path

    def run(self) -> None:
        try:
            path = collect_debug_bundle(
                self._ctx, self._output_path,
                progress_cb=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
            )
            self.finished_ok.emit(str(path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class DebugBundleSavePromptDialog(QDialog):
    """The prompt users see on every clean close.

    Attributes read by the caller after exec():
      - skip_forever (bool): user checked "Don't ask again"
      - saved_path (Path | None): where the bundle landed (None if Skip)
    """

    def __init__(self, parent, ctx: BundleContext) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self.skip_forever: bool = False
        self.saved_path: Optional[Path] = None
        self._worker: Optional[_BundleWorker] = None
        self._built = False

        self.setWindowTitle("Save Touchless debug bundle?")
        self.setModal(True)
        # Compact enough to not feel like a real save dialog; large
        # enough that the filename + folder rows don't wrap.
        self.setMinimumWidth(560)
        try:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        except Exception:
            pass
        self._build_ui()

    # ------------------------------------------------------------------
    def _default_filename(self) -> str:
        stamp = time.strftime("%Y-%m-%d_%H-%M")
        return f"Touchless_Debug_{stamp}.zip"

    def _default_folder(self) -> str:
        try:
            desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
            if desktop:
                return desktop
        except Exception:
            pass
        return str(Path.home())

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        header = QLabel("Save debug info before closing?")
        header.setStyleSheet("font-size: 14px; font-weight: 600;")
        outer.addWidget(header)

        message = QLabel(
            "This creates a single .zip file with logs, system info, "
            "and a redacted copy of your settings so Touchless support "
            "can help you fix any issues. Your Spotify token and other "
            "credentials are not included."
        )
        message.setWordWrap(True)
        outer.addWidget(message)

        # Filename row
        fn_row = QHBoxLayout()
        fn_row.addWidget(QLabel("Filename:"))
        self._filename_edit = QLineEdit(self._default_filename())
        fn_row.addWidget(self._filename_edit, 1)
        outer.addLayout(fn_row)

        # Folder row (edit + Browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Save to:"))
        self._folder_edit = QLineEdit(self._default_folder())
        folder_row.addWidget(self._folder_edit, 1)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._on_browse)
        folder_row.addWidget(self._browse_btn)
        outer.addLayout(folder_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Ready")
        outer.addWidget(self._progress)

        # Status
        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        # Don't ask again — v1.1.9.2 (r18): kept as an object so the
        # skip_forever plumbing stays intact, but NOT shown: there is no
        # Settings toggle to re-enable the prompt yet, and CLAUDE.md
        # rule 6 forbids a one-way switch an end user cannot undo.
        self._skip_forever_cb = QCheckBox("Don't ask again (re-enable from Settings)")
        self._skip_forever_cb.setVisible(False)

        # Buttons
        btns = QDialogButtonBox()
        self._save_btn = btns.addButton("Save", QDialogButtonBox.AcceptRole)
        self._skip_btn = btns.addButton("Skip", QDialogButtonBox.RejectRole)
        self._save_btn.clicked.connect(self._on_save)
        self._skip_btn.clicked.connect(self._on_skip)
        outer.addWidget(btns)

        self._built = True

    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        try:
            folder = QFileDialog.getExistingDirectory(
                self, "Choose folder to save debug bundle",
                self._folder_edit.text() or self._default_folder(),
            )
            if folder:
                self._folder_edit.setText(folder)
        except Exception:
            pass

    def _resolve_output_path(self) -> Optional[Path]:
        folder = (self._folder_edit.text() or "").strip()
        filename = (self._filename_edit.text() or "").strip()
        if not folder or not filename:
            self._status.setText("Please provide both a folder and a filename.")
            return None
        if not filename.lower().endswith(".zip"):
            filename += ".zip"
        try:
            p = Path(folder) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._status.setText(f"Cannot create folder: {exc}")
            return None
        return p

    def _lock_inputs(self, locked: bool) -> None:
        self._save_btn.setEnabled(not locked)
        self._skip_btn.setEnabled(not locked)
        self._browse_btn.setEnabled(not locked)
        self._filename_edit.setReadOnly(locked)
        self._folder_edit.setReadOnly(locked)
        self._skip_forever_cb.setEnabled(not locked)

    def _on_save(self) -> None:
        self.skip_forever = self._skip_forever_cb.isChecked()
        output_path = self._resolve_output_path()
        if output_path is None:
            return
        self._lock_inputs(True)
        self._status.setText("Collecting…")
        self._worker = _BundleWorker(self._ctx, output_path, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_bundle_ok)
        self._worker.failed.connect(self._on_bundle_failed)
        self._worker.start()

    def _on_skip(self) -> None:
        self.skip_forever = self._skip_forever_cb.isChecked()
        self.reject()

    def _on_progress(self, pct: int, msg: str) -> None:
        try:
            self._progress.setValue(int(pct))
            self._progress.setFormat(f"{pct}%")
            if msg:
                self._status.setText(msg)
        except Exception:
            pass

    def _on_bundle_ok(self, path: str) -> None:
        self.saved_path = Path(path)
        self._status.setText(f"Saved: {path}")
        self._progress.setValue(100)
        self._progress.setFormat("Done")
        # Auto-close so shutdown proceeds — the caller waits on
        # exec() and the closeEvent is blocked until we return.
        self.accept()

    def _on_bundle_failed(self, err: str) -> None:
        # Show the error but let the user Skip so shutdown proceeds.
        self._status.setText("Bundle collector failed. Skipping.\n" + (err[:400] if err else ""))
        self._progress.setFormat("Failed")
        self._lock_inputs(False)
        # Show a bit of context, then let the user hit Skip.

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        # If the user closes the dialog via the window X during a
        # collection, wait briefly for the worker so the zip isn't
        # left half-written on disk.
        try:
            if self._worker is not None and self._worker.isRunning():
                if not self._worker.wait(2000):
                    # v1.1.9.2 (r18): still collecting after 2 s. Detach
                    # the QThread from this dialog so Qt never destroys
                    # a RUNNING thread with its parent (hard crash at
                    # exit); it deletes itself when it finishes.
                    try:
                        self._worker.setParent(None)
                        self._worker.finished.connect(self._worker.deleteLater)
                    except Exception:
                        pass
        except Exception:
            pass
        super().closeEvent(event)
