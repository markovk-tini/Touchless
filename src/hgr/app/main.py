from __future__ import annotations

import datetime as _dt
import sys
import time as _time
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ..config.app_config import APP_NAME, CONFIG_DIR, load_config, save_config
from ..utils.runtime_paths import resource_path
from .single_instance import acquire as acquire_single_instance
from .ui.main_window import MainWindow
from .ui.touchless_splash import TouchlessSplash


class _StderrTee:
    """Mirror every stderr write to a rolling debug log file in
    addition to the original console / Qt-capture stream. Lets the
    user paste a single file when reporting bugs instead of
    scrolling through transient stderr buffers. Failures opening
    or writing the file are swallowed — stderr capture must never
    break the app even if the disk is full or read-only."""

    def __init__(self, original, file_handle) -> None:
        self._original = original
        self._file = file_handle

    def write(self, data) -> int:  # noqa: D401
        try:
            self._original.write(data)
        except Exception:
            pass
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        try:
            return len(data)
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._original.isatty())
        except Exception:
            return False

    def fileno(self):
        return self._original.fileno()


def _install_debug_log_tee() -> Path | None:
    """Open ~/.touchless/touchless_debug.log (rolling, max 4 MB) and
    tee sys.stderr through it. Called once at startup before any
    Qt construction so [clip-audio], [voice], [clip-anchor], etc.
    all land in the file from the moment the app boots.

    Returns the log file path on success, None on failure. The
    file path is also written as the first line so the user knows
    where to find it. Existing logs are renamed to .prev for one
    generation of backup; the .prev file is overwritten on each
    launch so disk doesn't grow unbounded."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = CONFIG_DIR / "touchless_debug.log"
        # Roll over a single backup so we don't lose the prior
        # session if the user just relaunched to reproduce a bug.
        try:
            if log_path.exists() and log_path.stat().st_size > 0:
                backup = CONFIG_DIR / "touchless_debug.prev.log"
                try:
                    if backup.exists():
                        backup.unlink()
                except Exception:
                    pass
                log_path.replace(backup)
        except Exception:
            pass
        handle = log_path.open("w", encoding="utf-8", buffering=1)
        # Header so the file is self-describing when the user opens
        # it standalone. Timestamps are wall-clock local time.
        try:
            ts = _dt.datetime.now().isoformat(timespec="seconds")
        except Exception:
            ts = "unknown"
        try:
            handle.write(
                f"# Touchless debug log — session started {ts}\n"
                f"# Tees every sys.stderr write for the lifetime of\n"
                f"# this Touchless process. Safe to paste in bug\n"
                f"# reports. Previous session's log was rotated to\n"
                f"#   {(CONFIG_DIR / 'touchless_debug.prev.log')}\n"
                f"# ----------------------------------------------\n"
            )
            handle.flush()
        except Exception:
            pass
        sys.stderr = _StderrTee(sys.stderr, handle)
        # Also surface the path to the original stderr so a console-
        # attached user knows where the file lives without having
        # to grep the source.
        try:
            (sys.stderr._original if hasattr(sys.stderr, "_original") else sys.__stderr__).write(
                f"[touchless] debug log → {log_path}\n"
            )
        except Exception:
            pass
        return log_path
    except Exception:
        return None


def _resolve_app_icon():
    candidates = (
        resource_path('assets', 'icons', 'touchless_icon.ico'),
        resource_path('assets', 'icons', 'touchless_icon.png'),
        resource_path('assets', 'icons', 'hgr_icon.ico'),
        resource_path('assets', 'icons', 'hgr_icon.png'),
    )
    return next((path for path in candidates if path.exists()), None)


def main() -> int:
    # Install the stderr → debug-log tee BEFORE any other startup
    # code so every [clip-audio], [clip-anchor], [voice] line from
    # the moment the app boots lands in a single pasteable file.
    # Placement note: must run BEFORE acquire_single_instance so
    # even the "another instance was already running" branch leaves
    # a trace in the file.
    _install_debug_log_tee()

    # Bail before constructing the Qt app if another Touchless is
    # already running. When the bailing instance was launched via
    # a Jump-List task (Pause / Settings / Quit), `args` carries
    # the corresponding flag and acquire() PostMessages it to the
    # running instance before returning False.
    if not acquire_single_instance(sys.argv[1:]):
        return 0

    # Tell Windows this process is its own app, not a generic
    # Python interpreter, so the taskbar groups our windows under
    # the Touchless icon instead of the python.exe icon. MUST happen
    # before the first window is created or Windows caches the
    # wrong grouping. The same AUMID is used by the Jump List below.
    app_user_model_id = "Touchless.App.MarkovK"
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            app_user_model_id
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationName(APP_NAME)

    # --- macOS: keep cyclic GC on the main thread ---------------------------
    # PySide6 QWidgets wrap NSWindows, and AppKit aborts hard (EXC_BREAKPOINT,
    # "Must only be used from the main thread") if a window is torn down off the
    # main thread. Python's cyclic GC can run on ANY thread that crosses the
    # allocation threshold (e.g. the voice/worker threads); if a top-level
    # QWidget sits in a reference cycle, GC destroys it there -> NSWindow close
    # off-main-thread -> crash. Disable automatic GC and drive it from a
    # main-thread QTimer so widget teardown always happens on the GUI thread.
    # macOS-only; Windows keeps Python's default GC behavior.
    if sys.platform == "darwin":
        import gc as _gc
        from PySide6.QtCore import QTimer as _QTimer

        _gc.disable()
        app._gc_timer = _QTimer(app)  # held on app so it isn't itself collected
        app._gc_timer.setInterval(2000)
        app._gc_timer.timeout.connect(lambda: _gc.collect())
        app._gc_timer.start()

        # --- macOS: prevent App Nap -----------------------------------------
        # Touchless runs a real-time camera + gesture pipeline while the user is
        # focused on ANOTHER app (Chrome, Spotify, …), so Touchless is almost
        # always a BACKGROUND app. macOS App Nap then throttles a background
        # app's timers + run loop to conserve power, which capped the whole
        # pipeline at ~15 fps even though per-frame work is <20 ms (the extra
        # ~40 ms/frame was pure run-loop throttling, not compute — which is why
        # Lite/GPU/paint changes did nothing). Register a long-lived
        # user-initiated + latency-critical activity so macOS keeps us running
        # at full rate in the background. The returned token is held on `app`;
        # releasing it would end the activity.
        try:
            from Foundation import NSProcessInfo
            try:
                from Foundation import (
                    NSActivityUserInitiated,
                    NSActivityLatencyCritical,
                )
                _nap_opts = int(NSActivityUserInitiated) | int(NSActivityLatencyCritical)
            except Exception:
                # Raw flag values if pyobjc doesn't export the constants.
                _nap_opts = 0x00FFFFFF | 0xFF00000000
            app._nap_activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                _nap_opts, "Touchless real-time camera + gesture pipeline"
            )
            try:
                sys.stderr.write("[macos] App Nap disabled (beginActivity) for full background fps\n")
                sys.stderr.flush()
            except Exception:
                pass
        except Exception as _nap_exc:
            app._nap_activity = None
            try:
                sys.stderr.write(f"[macos] App Nap disable failed: {type(_nap_exc).__name__}: {_nap_exc}\n")
                sys.stderr.flush()
            except Exception:
                pass

        # --- macOS: raise main-thread QoS to USER_INTERACTIVE ---------------
        # A python process launched from a TERMINAL (not a .app bundle) is
        # given a low/utility QoS class by macOS, which throttles its run-loop
        # timers — the gesture QTimer that should fire at 66 Hz only fires
        # ~17 Hz even though the CPU is idle and each frame's work is <25 ms.
        # (This is separate from App Nap.) Promote the main thread so timers
        # fire at full rate. In a packaged .app the process QoS is already
        # user-interactive, but setting it is harmless there.
        try:
            import ctypes as _ctypes
            _libsystem = _ctypes.CDLL("/usr/lib/libSystem.dylib")
            # pthread_set_qos_class_self_np(qos_class_t, relative_priority)
            # QOS_CLASS_USER_INTERACTIVE = 0x21
            _rc = _libsystem.pthread_set_qos_class_self_np(0x21, 0)
            sys.stderr.write(f"[macos] main-thread QoS -> USER_INTERACTIVE (rc={_rc})\n")
            sys.stderr.flush()
        except Exception as _qos_exc:
            try:
                sys.stderr.write(f"[macos] QoS bump failed: {type(_qos_exc).__name__}: {_qos_exc}\n")
                sys.stderr.flush()
            except Exception:
                pass

    # Install the taskbar Jump List. Only attempts in frozen builds
    # where sys.executable is Touchless.exe (each task re-launches
    # the exe with a flag). Source runs use python.exe whose path
    # isn't a sensible IShellLink target, so we skip silently.
    if getattr(sys, "frozen", False):
        try:
            from .jumplist import install_jumplist
            install_jumplist(
                app_user_model_id=app_user_model_id,
                exe_path=Path(sys.executable),
            )
        except Exception:
            pass

    icon_path = _resolve_app_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    config = load_config()
    save_config(config)

    def _build_window() -> MainWindow:
        w = MainWindow(config)
        # NOTE: do NOT call w.setWindowIcon(QIcon(str(icon_path)))
        # here. MainWindow.__init__ already sets the window icon to
        # the tray-state-bordered variant (grey at startup) and wires
        # the tray's icon_changed signal to setWindowIcon so it
        # updates on engine state transitions. Overwriting with the
        # unmodified app icon here clobbered the grey OFF-state ring
        # until the first state change re-set it via signal.
        # app.setWindowIcon above still provides the base icon that
        # MainWindow reads via QApplication.windowIcon() for the
        # tray's renderer.
        return w

    TouchlessSplash.run_with(_build_window, config.accent_color, app)

    # HGR_PROFILE=1: profile the MAIN THREAD for the whole session and dump the
    # hottest functions on quit. This attributes the per-frame event-loop time
    # (timer _tick, queued signal receivers, paints) that the [lite_mode/timing]
    # stages don't cover — used to find the macOS fps cap. Zero cost when off.
    import os as _os
    if _os.environ.get("HGR_PROFILE") == "1":
        import cProfile
        import io as _io
        import pstats
        _pr = cProfile.Profile()
        _pr.enable()
        try:
            _rc = app.exec()
        finally:
            _pr.disable()
            try:
                _buf = _io.StringIO()
                pstats.Stats(_pr, stream=_buf).sort_stats("tottime").print_stats(35)
                sys.stderr.write("\n===== HGR_PROFILE: main-thread, top 35 by tottime =====\n")
                sys.stderr.write(_buf.getvalue())
                sys.stderr.flush()
            except Exception:
                pass
        return _rc
    return app.exec()

# Author: Konstantin Markov
