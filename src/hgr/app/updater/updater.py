"""Download the new release artifact and hand off to it.

Two update paths, depending on what asset GitHub serves:

A. App-only zip (Touchless_App_Update_<ver>.zip, ~50-150 MB):
   1. Download the zip to a stable temp location.
   2. Write a tiny `_apply_update.bat` helper next to the install
      that waits for Touchless.exe to exit, then unzips over the
      install directory and relaunches the app.
   3. Spawn the helper, exit our process. The helper does its
      work without admin prompts because the per-user install
      lives under %LOCALAPPDATA%\\Programs\\Touchless\\ — fully
      user-writable. UAC never appears.

B. Full installer .exe (~2.4 GB, first install or breaking change):
   1. Download to temp.
   2. Launch with Inno Setup silent flags (/SILENT
      /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS), exit ourselves.
   3. Installer replaces files and relaunches Touchless. UAC may
      fire if the user has an old install in Program Files; new
      installs to LocalAppData skip it.

The path is chosen by the ReleaseInfo.update_kind value.
"""
from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from .release_checker import ReleaseInfo


_PROGRESS_CHUNK_BYTES = 256 * 1024


def _update_work_dir() -> Path:
    """Where update downloads, the apply-helper bat, and logs live.

    Lives under %LOCALAPPDATA%\\Touchless\\Updates\\ instead of %TEMP%
    because Norton SONAR's dropper heuristic fires hard on the pattern
    "signed binary writes an executable into %TEMP% and then launches
    it" — that's exactly what got Touchless 1.1.2 quarantined for some
    users. %LOCALAPPDATA% reads as the app's own working directory, a
    legitimate spot for an app to stage its own update, so the same
    operations don't trip the same heuristic. Falls back to the OS
    temp dir if LOCALAPPDATA is somehow unset (shouldn't happen on
    Windows but the fallback keeps source-runs / unusual envs working).
    """
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = tempfile.gettempdir()
    d = Path(base) / "Touchless" / "Updates"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If even mkdir fails (perms, disk full), fall back to TEMP so
        # the updater can at least try — better a SONAR flag than no
        # update path at all.
        d = Path(tempfile.gettempdir()) / "Touchless_Update"
        d.mkdir(parents=True, exist_ok=True)
    return d


class _DownloadWorker(threading.Thread):
    """Plain threading.Thread + Qt-thread-safe signals via callbacks.

    We avoid moving this onto a QThread because the dialog already
    owns its parent thread; a callback-shim is simpler and less
    invasive. Callbacks are scheduled onto the GUI thread by the
    Updater itself via QTimer.singleShot.
    """

    def __init__(
        self,
        url: str,
        target_path: Path,
        on_progress: Callable[[int, int], None],
        on_done: Callable[[Path], None],
        on_failed: Callable[[str], None],
    ) -> None:
        super().__init__(daemon=True, name="UpdaterDownload")
        self._url = url
        self._target_path = target_path
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_failed = on_failed
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            self._target_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "Touchless-Updater/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                total_str = resp.headers.get("Content-Length") or "0"
                try:
                    total = int(total_str)
                except (TypeError, ValueError):
                    total = 0
                downloaded = 0
                last_progress_at = 0
                # Open in 'wb' (not append) — a partial leftover
                # download from a previous failed attempt would
                # otherwise corrupt the target.
                with open(self._target_path, "wb") as fh:
                    while True:
                        if self._cancelled.is_set():
                            self._on_failed("Cancelled")
                            return
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_progress_at >= _PROGRESS_CHUNK_BYTES:
                            self._on_progress(downloaded, total)
                            last_progress_at = downloaded
            # Final tick so the progress bar lands at 100% before
            # the dialog flips to "launching".
            self._on_progress(downloaded, total or downloaded)
            self._on_done(self._target_path)
        except urllib.error.URLError as exc:
            self._on_failed(f"Network error: {exc!s}")
        except OSError as exc:
            self._on_failed(f"Disk error: {exc!s}")
        except Exception as exc:  # pragma: no cover
            self._on_failed(f"Unexpected: {type(exc).__name__}: {exc!s}")


class Updater(QObject):
    """Drives the download → launch handoff. Owns the download
    thread, marshals callbacks to the GUI thread, and asks the app
    to exit cleanly after launching the installer."""

    progress = Signal(int, str)        # percent (or -1), status text
    failed = Signal(str)
    ready_to_launch = Signal(str)      # path to downloaded installer
    # Internal signal emitted by the worker thread when download
    # completes. Connected to _on_download_complete so the slot runs
    # on the GUI thread via Qt's automatic queued connection.
    _download_finished = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: _DownloadWorker | None = None
        self._target_path: Path | None = None
        self._info: Optional[ReleaseInfo] = None
        self._download_finished.connect(self._on_download_complete)

    def start_download(self, info: ReleaseInfo) -> None:
        if not info.download_url:
            self.failed.emit(
                "No update artifact attached to this release. Please update manually from the GitHub release page."
            )
            return
        self._info = info
        # Stable work folder under %LOCALAPPDATA% so partial downloads
        # survive retries within a session. Lives outside %TEMP%
        # specifically to avoid Norton SONAR's "drops exe in temp,
        # runs it" heuristic — see _update_work_dir for context.
        target_dir = _update_work_dir()
        if info.update_kind == "app-zip":
            target_path = target_dir / f"Touchless_App_Update_{info.version}.zip"
        else:
            target_path = target_dir / f"Touchless_Installer_{info.version}.exe"
        self._target_path = target_path

        def progress_cb(downloaded: int, total: int) -> None:
            if total > 0:
                pct = int((downloaded * 100) / total)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                msg = f"Downloading... {mb_done:.1f} MB of {mb_total:.1f} MB"
            else:
                pct = -1
                mb_done = downloaded / (1024 * 1024)
                msg = f"Downloading... {mb_done:.1f} MB"
            # Emit the signal directly. Qt automatically uses a queued
            # connection when the emitter and the receiver live on
            # different threads, so the slot fires on the GUI thread.
            # The previous QTimer.singleShot(0, ...) approach silently
            # no-op'd because singleShot needs a running Qt event loop
            # in the calling thread — which a plain threading.Thread
            # doesn't have. Result: download never reported progress
            # and stayed stuck on "Starting download...".
            try:
                self.progress.emit(pct, msg)
            except Exception:
                pass

        def done_cb(path: Path) -> None:
            # Emitting the internal signal from the worker thread
            # automatically queues delivery onto the GUI thread (Qt's
            # AutoConnection picks QueuedConnection across threads).
            try:
                self._download_finished.emit(str(path))
            except Exception:
                pass

        def failed_cb(reason: str) -> None:
            try:
                self.failed.emit(reason)
            except Exception:
                pass

        self._worker = _DownloadWorker(
            url=info.download_url,
            target_path=target_path,
            on_progress=progress_cb,
            on_done=done_cb,
            on_failed=failed_cb,
        )
        self._worker.start()

    def _on_download_complete(self, path: str) -> None:
        self.progress.emit(-1, "Launching installer...")
        # Emit immediately. The previous QTimer.singleShot(400, ...)
        # cosmetic delay was failing to fire on some Windows configs
        # — the lambda capturing `self` would get garbage-collected
        # before the timer's deadline, leaving the dialog frozen on
        # 'Launching installer...' indefinitely. Direct emit
        # guarantees the apply path runs synchronously on the GUI
        # thread that just received the download_finished signal.
        self.ready_to_launch.emit(path)

    def apply_update_and_exit(self, downloaded_path: str) -> bool:
        """Route to the appropriate handler based on the update kind
        the ReleaseChecker stamped on `self._info`. Returns False on
        any setup failure so the dialog can show an error."""
        if self._info is None:
            return False
        if self._info.update_kind == "app-zip":
            return self._apply_zip_and_exit(downloaded_path)
        return self.launch_installer_and_exit(downloaded_path)

    @staticmethod
    def is_install_dir_writable() -> bool:
        """Check whether the running app's install directory is
        writable by the current user without elevation. False for
        legacy Program Files installs where Windows requires admin.

        Used by ReleaseChecker BEFORE choosing the app-zip path —
        if the user is in a non-writable dir (didn't migrate from
        the old Program Files install), we force the full-installer
        path even when a zip asset is present, so the update has a
        chance of succeeding via UAC elevation rather than silently
        failing in PowerShell Expand-Archive."""
        if not getattr(sys, "frozen", False):
            return True   # source-run; the check doesn't apply
        try:
            install_dir = Path(sys.executable).resolve().parent
            probe = install_dir / ".touchless_write_probe"
            probe.write_bytes(b"x")
            probe.unlink(missing_ok=True)
            return True
        except (OSError, PermissionError):
            return False
        except Exception:
            return False

    # Backwards-compat shim — main_window connects to ready_to_launch
    # and may have been wired before the kind-based router existed.
    def launch_installer_and_exit(self, installer_path: str) -> bool:
        """Launch the downloaded .exe with Inno Setup silent flags
        and exit the current app. Used for full-installer updates."""
        if not os.path.exists(installer_path):
            return False
        # Inno Setup flags:
        #   /SILENT — minimal install UI (just a progress dialog)
        #   /CLOSEAPPLICATIONS — close any running Touchless first
        #   /RESTARTAPPLICATIONS — relaunch it after install
        #   /NORESTART — don't reboot Windows even if the installer
        #                thinks it needs to (it doesn't, we don't
        #                touch system DLLs)
        params = "/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /NORESTART"
        try:
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "open", installer_path, params, None, 1
            )
            if ret <= 32:
                return False
        except Exception:
            return False
        QTimer.singleShot(0, self._quit_app)
        return True

    def _apply_zip_and_exit(self, zip_path: str) -> bool:
        """Apply an app-only zip update by spawning a small batch
        helper that waits for our process to exit, unzips over the
        install directory, then relaunches the new Touchless.exe.

        No UAC needed: the per-user install dir (%LOCALAPPDATA%\\
        Programs\\Touchless\\) is fully user-writable. The helper
        runs at the same privilege level as the launching app.
        """
        # Python-side log alongside the bat's log so we know whether
        # this method was even reached and which step (if any) bailed.
        # Useful to disambiguate "bat didn't run" from "bat-writer
        # crashed silently". Build round 5 marker.
        py_log = _update_work_dir() / "_python_apply.log"
        def _plog(msg: str) -> None:
            try:
                py_log.parent.mkdir(parents=True, exist_ok=True)
                with open(py_log, "a", encoding="utf-8") as fh:
                    fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            except Exception:
                pass

        _plog(f"_apply_zip_and_exit ENTERED zip_path={zip_path}")
        if not os.path.exists(zip_path):
            _plog("FAIL: zip_path does not exist on disk")
            return False
        install_dir = self._resolve_install_dir()
        _plog(f"install_dir={install_dir}")
        if install_dir is None:
            _plog("FAIL: install_dir resolved to None (likely source-run, not frozen)")
            return False

        helper_path = self._write_apply_helper(zip_path, install_dir)
        _plog(f"helper_path={helper_path}")
        if helper_path is None:
            _plog("FAIL: _write_apply_helper returned None")
            return False

        try:
            # Use ShellExecuteW directly (not os.startfile) so we can
            # pass SW_HIDE — that hides the console window the .bat
            # would normally flash up. Same launch mechanism Windows
            # Explorer uses for a double-clicked file (which we
            # verified works), but invisible.
            #
            # Explicit invocation as `cmd.exe /c <bat>` so SW_HIDE
            # applies to the cmd console hosting the bat. (If we
            # passed the .bat directly to ShellExecuteW with SW_HIDE,
            # cmd.exe would still allocate its own visible console
            # because that's what the .bat shell association does.)
            _plog(f"calling ShellExecuteW with SW_HIDE for {helper_path}")
            params = f'/c "{helper_path}"'
            ret = ctypes.windll.shell32.ShellExecuteW(
                None,           # hwnd
                "open",         # verb
                "cmd.exe",      # file
                params,         # parameters
                None,           # working dir (None = inherit)
                0,              # SW_HIDE
            )
            _plog(f"ShellExecuteW returned {ret} (>32 means success)")
            if ret <= 32:
                _plog(f"FAIL: ShellExecuteW returned {ret}, treating as failure")
                return False
        except Exception as exc:
            _plog(f"FAIL: ShellExecuteW raised {type(exc).__name__}: {exc!s}")
            return False

        _plog("scheduling _quit_app")
        QTimer.singleShot(0, self._quit_app)
        return True

    def _resolve_install_dir(self) -> Optional[Path]:
        """Where is Touchless.exe installed? In a frozen PyInstaller
        bundle, sys.executable IS Touchless.exe — its parent is the
        install dir. In a dev source-run, we don't have an install
        to upgrade and bail."""
        if not getattr(sys, "frozen", False):
            return None
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            return None

    def _write_apply_helper(self, zip_path: str, install_dir: Path) -> Optional[Path]:
        """Write the apply-update batch helper next to the zip.

        Why a batch script and not Python: the user's machine has
        no Python interpreter outside the bundle, and we can't run
        the bundle while it's mid-replacement. cmd.exe + PowerShell
        Expand-Archive are universally available on Windows 10+.

        The script:
          1. Waits up to ~30s for Touchless.exe to be writable —
             checking the file LOCK directly (rename-in-place trick),
             not just the process list. After the .exe process exits,
             Windows can hold file handles open for several extra
             seconds, which used to silently break Expand-Archive.
          2. Extracts the zip to a STAGING dir (not directly over the
             install). If extraction fails partway, the user's
             install dir stays consistent.
          3. Robocopies staging → install_dir with retry semantics
             (60 retries, 1s apart) so any lingering file lock on a
             specific dependency DLL doesn't corrupt the install.
          4. Verifies Touchless.exe exists post-copy before relaunch.
          5. Logs every step to %LOCALAPPDATA%\\Touchless\\Updates\\
             _apply_update.log so failures are diagnosable instead
             of silent.
        """
        try:
            zip_dir = Path(zip_path).parent
            zip_dir.mkdir(parents=True, exist_ok=True)
            helper = zip_dir / "_apply_update.bat"
            content = (
                "@echo off\r\n"
                "rem [BUILD-MARKER: v1.1.3 — localappdata staging (Norton SONAR fix)]\r\n"
                "setlocal enabledelayedexpansion\r\n"
                f"set \"INSTALL_DIR={install_dir}\"\r\n"
                f"set \"UPDATE_ZIP={zip_path}\"\r\n"
                # Stage + log under %LOCALAPPDATA% (not %TEMP%) — Norton
                # SONAR weights %TEMP% heavily for dropper heuristics
                # and was quarantining 1.1.2 here. %LOCALAPPDATA% is
                # the app's own per-user dir and reads as legitimate.
                "set \"STAGING=%LOCALAPPDATA%\\Touchless\\Updates\\staging\"\r\n"
                "set \"LOG=%LOCALAPPDATA%\\Touchless\\Updates\\_apply_update.log\"\r\n"
                "echo [start] %DATE% %TIME% INSTALL_DIR=%INSTALL_DIR% > \"%LOG%\" 2>&1\r\n"
                "\r\n"
                "rem Initial settle window — gives Windows a chance to release\r\n"
                "rem file handles after the Touchless process exited.\r\n"
                "timeout /t 3 /nobreak >nul\r\n"
                "\r\n"
                "rem Loop: probe Touchless.exe writability by trying to rename\r\n"
                "rem it in place. Rename succeeds only when no process holds\r\n"
                "rem an exclusive handle. If it fails, sleep 1s and retry.\r\n"
                "set /a count=0\r\n"
                ":waitlock\r\n"
                "if not exist \"%INSTALL_DIR%\\Touchless.exe\" goto extract\r\n"
                "ren \"%INSTALL_DIR%\\Touchless.exe\" \"Touchless.exe\" >>\"%LOG%\" 2>&1\r\n"
                "if not errorlevel 1 goto extract\r\n"
                "if !count! geq 30 (\r\n"
                "  echo [warn] Touchless.exe still locked after 30s, attempting extract anyway >> \"%LOG%\"\r\n"
                "  goto extract\r\n"
                ")\r\n"
                "timeout /t 1 /nobreak >nul\r\n"
                "set /a count+=1\r\n"
                "goto waitlock\r\n"
                "\r\n"
                ":extract\r\n"
                "echo [info] extracting to staging dir %STAGING% >> \"%LOG%\"\r\n"
                "if exist \"%STAGING%\" rmdir /s /q \"%STAGING%\" >>\"%LOG%\" 2>&1\r\n"
                "mkdir \"%STAGING%\" >>\"%LOG%\" 2>&1\r\n"
                "powershell -NoProfile -ExecutionPolicy Bypass -Command "
                "\"Expand-Archive -LiteralPath '%UPDATE_ZIP%' -DestinationPath '%STAGING%' -Force\""
                " >>\"%LOG%\" 2>&1\r\n"
                "if errorlevel 1 (\r\n"
                "  echo [error] Expand-Archive failed errorlevel=!errorlevel! >> \"%LOG%\"\r\n"
                "  goto fail\r\n"
                ")\r\n"
                "if not exist \"%STAGING%\\Touchless.exe\" (\r\n"
                "  echo [error] staged Touchless.exe missing after extract >> \"%LOG%\"\r\n"
                "  goto fail\r\n"
                ")\r\n"
                "\r\n"
                "echo [info] robocopying staging into install dir >> \"%LOG%\"\r\n"
                "robocopy \"%STAGING%\" \"%INSTALL_DIR%\" /E /R:60 /W:1 /NFL /NDL /NJH /NJS"
                " >>\"%LOG%\" 2>&1\r\n"
                "rem robocopy uses bitmask exit codes; >=8 means failure.\r\n"
                "if !errorlevel! geq 8 (\r\n"
                "  echo [error] robocopy failed errorlevel=!errorlevel! >> \"%LOG%\"\r\n"
                "  goto fail\r\n"
                ")\r\n"
                "\r\n"
                "if not exist \"%INSTALL_DIR%\\Touchless.exe\" (\r\n"
                "  echo [error] post-copy Touchless.exe missing >> \"%LOG%\"\r\n"
                "  goto fail\r\n"
                ")\r\n"
                "\r\n"
                "echo [success] update applied, relaunching >> \"%LOG%\"\r\n"
                "start \"\" \"%INSTALL_DIR%\\Touchless.exe\"\r\n"
                "\r\n"
                "rmdir /s /q \"%STAGING%\" >nul 2>&1\r\n"
                "del \"%UPDATE_ZIP%\" >nul 2>&1\r\n"
                "endlocal\r\n"
                "exit /b 0\r\n"
                "\r\n"
                ":fail\r\n"
                "echo [fail] update aborted at %DATE% %TIME% >> \"%LOG%\"\r\n"
                "rem Don't pause — there's no console window. Just exit.\r\n"
                "rem Relaunch the OLD Touchless.exe so the user isn't left\r\n"
                "rem without an app. They'll see the update prompt again on\r\n"
                "rem next launch and can retry.\r\n"
                "if exist \"%INSTALL_DIR%\\Touchless.exe\" start \"\" \"%INSTALL_DIR%\\Touchless.exe\"\r\n"
                "endlocal\r\n"
                "exit /b 1\r\n"
            )
            helper.write_text(content, encoding="cp1252")
            return helper
        except Exception:
            return None

    def _quit_app(self) -> None:
        # Use the Qt application's quit() so any cleanup hooks
        # (config save, audio device release, server shutdown) run
        # before exit. Hard sys.exit as a last-resort fallback.
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.quit()
                # Give Qt a moment to flush; if it doesn't exit
                # within 1.5s, force it.
                threading.Timer(1.5, lambda: os._exit(0)).start()
                return
        except Exception:
            pass
        os._exit(0)

# Author: Konstantin Markov
