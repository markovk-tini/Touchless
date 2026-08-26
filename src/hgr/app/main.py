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


def main() -> int:
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

    TouchlessSplash.run_with(_build_window, config.accent_color, app)
    return app.exec()

# Author: Konstantin Markov
