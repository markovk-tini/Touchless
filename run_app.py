from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path


# v1.1.7 round-31 debug log writer. In a PyInstaller --windowed
# bundle, `sys.stdout` / `sys.stderr` are `None` because runw.exe
# has no console. Previous builds installed a null-shim that
# silently dropped every write — meaning all the [perf-camera],
# [cap-res], [wasapi-bridge], [clip-cache] diagnostic lines the app
# emits (hundreds per session) got thrown away. That made it
# impossible to diagnose user problems in the shipped app.
#
# This replacement tees stderr/stdout into a persistent rotating
# log file at ~/Documents/Touchless/touchless_debug.log — dad can
# zip that file and send it and we see the full pipeline history.
# Source runs (real TextIO on stderr) still print to console AND
# tee to the file.
_LOG_ROTATE_BYTES = 5 * 1024 * 1024  # 5 MB

# v1.1.7 round-48 fix: on Windows machines with OneDrive Documents
# redirect, Controlled-Folder-Access, or non-standard user-profile
# ACLs, ``Path.home()/'Documents'/'Touchless'`` was either invisible
# to the user (redirected out from under File Explorer's Documents
# shortcut) or outright unwritable, and `_open_log_file()` failed
# silently. Resolve to `%LOCALAPPDATA%\Touchless\logs` first — per-
# user, always writable, never OneDrive-redirected, never CFA-
# protected — then fall back to Documents (legacy path, preserves
# any pre-existing rotation history so support tickets referencing
# `~/Documents/Touchless` still resolve), then to the OS temp dir.
# The winning path is stashed in `_LOG_PATH_RESOLVED` so the exit
# popup can display it verbatim.
def _candidate_log_dirs():
    candidates = []
    try:
        lad = os.environ.get("LOCALAPPDATA")
        if lad:
            candidates.append(Path(lad) / "Touchless" / "logs")
    except Exception:
        pass
    try:
        candidates.append(Path.home() / "AppData" / "Local" / "Touchless" / "logs")
    except Exception:
        pass
    try:
        candidates.append(Path.home() / "Documents" / "Touchless")
    except Exception:
        pass
    try:
        import tempfile as _tf
        candidates.append(Path(_tf.gettempdir()) / "Touchless")
    except Exception:
        pass
    seen = set(); out = []
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key); out.append(c)
    return out


def _open_log_file():
    """Try each candidate dir in order; the first that both mkdir's
    AND opens successfully wins. Sets `_LOG_PATH_RESOLVED` (module
    global) so the exit popup + any tray menu entry can surface it.
    Returns a text-mode append handle, or None if every candidate
    failed (permissions, disk full, AV lock, etc.)."""
    global _LOG_PATH_RESOLVED
    for log_dir in _candidate_log_dirs():
        log_path = log_dir / "touchless_debug.log"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        try:
            if log_path.exists() and log_path.stat().st_size > _LOG_ROTATE_BYTES:
                bak2 = log_path.with_suffix(".log.2")
                bak1 = log_path.with_suffix(".log.1")
                try:
                    if bak2.exists():
                        bak2.unlink()
                except Exception:
                    pass
                try:
                    if bak1.exists():
                        bak1.replace(bak2)
                except Exception:
                    pass
                try:
                    log_path.replace(bak1)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            fh = open(log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            continue
        _LOG_PATH_RESOLVED = log_path
        try:
            fh.write(f"[startup] log path resolved to {log_path}\n")
            fh.flush()
        except Exception:
            pass
        return fh
    _LOG_PATH_RESOLVED = None
    return None


_LOG_PATH_RESOLVED = None
_LOG_FILE = _open_log_file()

# Belt: flush on abnormal exit so the last diagnostic lines are
# captured even when the user force-closes via the tray or Task
# Manager (bypassing MainWindow.closeEvent).
try:
    import atexit as _atexit
    def _flush_log_on_exit():
        try:
            if _LOG_FILE is not None:
                _LOG_FILE.flush()
        except Exception:
            pass
    _atexit.register(_flush_log_on_exit)
except Exception:
    pass


class _TeeStream:
    """Duplex text stream: writes to `primary` (if not None) AND
    to `_LOG_FILE` (if not None). If both are None, silently drops.
    Every write is timestamped when the log file exists so a user
    submitting the file can see event ordering.
    """
    encoding = "utf-8"

    def __init__(self, primary, tag):
        self._primary = primary
        self._tag = tag  # "stdout" | "stderr"
        self._line_buf = ""

    def write(self, s):
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                s = ""
        # Primary passthrough (source runs).
        if self._primary is not None:
            try:
                self._primary.write(s)
            except Exception:
                pass
        # Log file with per-line timestamp.
        if _LOG_FILE is not None:
            try:
                self._line_buf += s
                while "\n" in self._line_buf:
                    line, self._line_buf = self._line_buf.split("\n", 1)
                    ts = time.strftime("%H:%M:%S")
                    _LOG_FILE.write(f"[{ts}] [{self._tag}] {line}\n")
            except Exception:
                pass
        return len(s)

    def flush(self):
        if self._primary is not None:
            try:
                self._primary.flush()
            except Exception:
                pass
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return bool(self._primary and self._primary.isatty())
        except Exception:
            return False

    def close(self):
        pass


# Install the tee wrappers. If _LOG_FILE is None (permissions or
# disk failure) and the primary is None too, writes silently drop
# — matching the previous null-shim behaviour so startup can't
# regress. If _LOG_FILE opened successfully, all diagnostics land
# in ~/Documents/Touchless/touchless_debug.log.
sys.stderr = _TeeStream(sys.stderr, "stderr")
sys.stdout = _TeeStream(sys.stdout, "stdout")

# Startup banner so anyone reading the log can see when a fresh
# session began + which build it is.
try:
    from hgr import BUILD_ROUND as _br
except Exception:
    _br = "?"
try:
    _banner = (
        f"\n===== Touchless startup: {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"build round {_br} python {sys.version.split()[0]} frozen={getattr(sys, 'frozen', False)} =====\n"
    )
    sys.stderr.write(_banner)
    sys.stderr.flush()
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# v1.1.7 round-48: publish the resolved log path onto the `hgr`
# package so main_window.closeEvent can display it without needing
# to re-import run_app (which is __main__ in the frozen build).
# Runs AFTER sys.path is set so `import hgr` works in source mode.
try:
    import hgr as _hgr_pkg
    _hgr_pkg._debug_log_path_resolved = _LOG_PATH_RESOLVED
    _hgr_pkg._debug_log_file = _LOG_FILE
except Exception:
    pass

from hgr.app.main import main


if __name__ == '__main__':
    raise SystemExit(main())

# Author: Konstantin Markov
