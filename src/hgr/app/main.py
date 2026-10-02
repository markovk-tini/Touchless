from __future__ import annotations

import sys
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ..config.app_config import APP_NAME, load_config, save_config
from ..utils.runtime_paths import resource_path
from .single_instance import acquire as acquire_single_instance
from .ui.main_window import MainWindow
from .ui.touchless_splash import TouchlessSplash


def _resolve_app_icon():
    candidates = (
        resource_path('assets', 'icons', 'touchless_icon.ico'),
        resource_path('assets', 'icons', 'touchless_icon.png'),
        resource_path('assets', 'icons', 'hgr_icon.ico'),
        resource_path('assets', 'icons', 'hgr_icon.png'),
    )
    return next((path for path in candidates if path.exists()), None)


def _install_smoke_hooks(app, window) -> None:
    """v1.1.9.2 (r18) build smoke gate. Inert unless the env vars are set,
    so a normal launch is untouched. Works frozen AND from source.

      HGR_AUTOSTART_ENGINE=1    1.5 s after the window is shown, press
                                START (MainWindow.start_engine — the START
                                button's slot — with the walkthrough
                                prompt skipped so nothing modal can block).
      HGR_SMOKE_EXIT_AFTER_S=N  N s after the window is shown, write a
                                JSON marker to HGR_SMOKE_MARKER_PATH and
                                quit through the real tray-Quit path
                                (_allow_real_close=True -> close()). The
                                debug-bundle prompt in closeEvent is
                                bypassed via an IN-MEMORY flag only; the
                                skip_debug_bundle_prompt_on_close pref is
                                never written to disk.

    Consumed by builder/windows/smoke_gate.ps1 (build step [2.5/6]).
    """
    import os

    autostart = os.environ.get("HGR_AUTOSTART_ENGINE", "").strip() == "1"
    exit_after_raw = os.environ.get("HGR_SMOKE_EXIT_AFTER_S", "").strip()
    if not autostart and not exit_after_raw:
        return

    import json
    import tempfile
    import threading
    import traceback
    from PySide6.QtCore import QTimer

    state = {"frames_seen": 0, "autostart_called": False, "autostart_error": ""}

    def _log(msg: str) -> None:
        try:
            sys.stderr.write(f"[smoke] {msg}\n")
            sys.stderr.flush()
        except Exception:
            pass

    # Byte offset into the tee'd stderr log (run_app.py publishes the
    # resolved path on the hgr package) so the marker's error tail only
    # covers THIS session, not yesterday's.
    log_path = None
    log_offset = 0
    try:
        import hgr as _hgr_pkg
        log_path = getattr(_hgr_pkg, "_debug_log_path_resolved", None)
        if log_path is not None:
            log_offset = int(Path(log_path).stat().st_size)
    except Exception:
        log_path = None

    def _worker():
        return getattr(window, "_worker", None)

    def _on_frame(*_args) -> None:
        state["frames_seen"] += 1

    def _autostart() -> None:
        state["autostart_called"] = True
        _log("HGR_AUTOSTART_ENGINE=1 -> start_engine(skip_tutorial_prompt=True)")
        try:
            window.start_engine(skip_tutorial_prompt=True)
            w = _worker()
            if w is not None:
                # raw_frame_ready is the uncapped per-frame signal
                # (debug_frame_ready is throttled to 30 Hz).
                w.raw_frame_ready.connect(_on_frame)
        except Exception:
            state["autostart_error"] = traceback.format_exc()
            _log("start_engine raised:\n" + state["autostart_error"])

    if autostart:
        QTimer.singleShot(1500, _autostart)

    if not exit_after_raw:
        return
    try:
        exit_after_s = max(1, int(float(exit_after_raw)))
    except ValueError:
        exit_after_s = 30
    marker_path = os.environ.get("HGR_SMOKE_MARKER_PATH", "").strip() or str(
        Path(tempfile.gettempdir()) / "touchless_smoke_marker.json"
    )

    def _error_tail() -> list:
        errors = []
        if state["autostart_error"]:
            errors.append("start_engine: " + state["autostart_error"].strip().splitlines()[-1])
        if log_path is not None:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(log_offset)
                    for line in fh:
                        low = line.lower()
                        if "traceback" in low or "fatal exception" in low:
                            errors.append(line.rstrip("\r\n"))
            except Exception:
                pass
        return errors[-20:]

    def _write_marker() -> None:
        w = _worker()
        cfg = getattr(window, "config", None)
        try:
            mode = (
                "gpu"
                if bool(getattr(cfg, "gpu_mode", False) or getattr(cfg, "prefer_gpu", False))
                else "lite"
                if bool(getattr(cfg, "lite_mode", False))
                else "default"
            )
        except Exception:
            mode = "unknown"
        payload = {
            # GestureWorker constructed + start() returned without raising.
            # On a no-camera machine this is still True (worker exists,
            # is_running stays False) — see camera_opened / engine_running.
            "engine_started": bool(
                state["autostart_called"] and not state["autostart_error"] and w is not None
            ),
            "engine_running": bool(w is not None and getattr(w, "is_running", False)),
            "camera_opened": bool(w is not None and getattr(w, "_cap", None) is not None),
            "first_frame_received": bool(
                w is not None and getattr(w, "_camera_first_frame_received", False)
            ),
            "frames_seen": int(state["frames_seen"]),
            "display_fps": float(getattr(w, "_fps", 0.0) or 0.0) if w is not None else 0.0,
            "mode": mode,
            "frozen": bool(getattr(sys, "frozen", False)),
            "autostart_requested": bool(autostart),
            "run_seconds": int(exit_after_s),
            "log_path": str(log_path) if log_path is not None else "",
            "errors": _error_tail(),
        }
        try:
            target = Path(marker_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(target)
            _log(f"marker written to {target}: {json.dumps(payload)}")
        except Exception:
            _log("marker write FAILED:\n" + traceback.format_exc())

    def _smoke_exit() -> None:
        _write_marker()
        # In-memory bypass of the r17 debug-bundle QDialog.exec() in
        # closeEvent. Deliberately NOT config.skip_debug_bundle_prompt_on_close.
        try:
            window._smoke_skip_debug_bundle_prompt = True
        except Exception:
            pass
        # Same sequence as MainWindow._on_tray_quit, plus an explicit
        # close() so closeEvent (stop_engine, geometry save) runs.
        tray = getattr(window, "_tray_icon", None)
        if tray is not None:
            try:
                tray.hide()
            except Exception:
                pass
        try:
            window._allow_real_close = True
        except Exception:
            pass
        try:
            window.close()
        except Exception:
            _log("close() raised:\n" + traceback.format_exc())
        QTimer.singleShot(0, app.quit)
        # Last resort if the Qt loop is wedged (modal dialog, stop_engine
        # hang): hard-exit with a distinct code the gate reports as
        # "did not exit cleanly". Daemon so a normal exit isn't held up.
        killer = threading.Timer(25.0, lambda: os._exit(4))
        killer.daemon = True
        killer.start()

    _log(f"HGR_SMOKE_EXIT_AFTER_S={exit_after_s} -> marker {marker_path}")
    QTimer.singleShot(exit_after_s * 1000, _smoke_exit)


def main() -> int:
    # v1.1.9.2 (r11) diagnostics: enable Python's faulthandler at
    # process start so any native SIGSEGV / SIGABRT dumps a Python
    # traceback to stderr AND to a rotating crash log before Windows
    # kills the process. Zero runtime cost, no behavior change, big
    # payoff the next time a dad-PC-class crash happens. Rotates: we
    # write to %LOCALAPPDATA%\Touchless\crash\faulthandler.log,
    # which the user can email us. Fully best-effort — a failure
    # here never blocks app start.
    try:
        import faulthandler
        import os as _r11_os
        try:
            _appdata = _r11_os.environ.get("LOCALAPPDATA") or str(Path.home())
            _crash_dir = Path(_appdata) / "Touchless" / "crash"
            _crash_dir.mkdir(parents=True, exist_ok=True)
            _fault_log = open(str(_crash_dir / "faulthandler.log"), "a", encoding="utf-8", buffering=1)
            faulthandler.enable(file=_fault_log, all_threads=True)
        except Exception:
            # Fall back to stderr — better than nothing.
            faulthandler.enable(all_threads=True)
    except Exception:
        # faulthandler is stdlib; if this fails, the interpreter is
        # broken in a way we cannot recover from anyway.
        pass

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

    # Keep Windows' record of where Touchless is installed in sync with where
    # it's actually running from. If the user moved the install folder (e.g. to
    # another drive), this rewrites the stale registry path so the installer /
    # Microsoft Store update path targets the real location instead of the old
    # one. Frozen + Windows only; no-op when already correct. Runs early so a
    # move is healed before any update check fires.
    if getattr(sys, "frozen", False):
        try:
            from .updater.install_location import heal_install_location
            heal_install_location()
        except Exception:
            pass
        # Auto-start half of the same re-home: if login-launch is enabled but
        # its Run-key command points at the pre-move location, repoint it.
        try:
            from ..utils import autostart
            autostart.heal()
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

    window = TouchlessSplash.run_with(_build_window, config.accent_color, app)
    # v1.1.9.2 (r18): build smoke gate env hooks (no-op when unset).
    try:
        _install_smoke_hooks(app, window)
    except Exception:
        pass
    return app.exec()

# Author: Konstantin Markov
