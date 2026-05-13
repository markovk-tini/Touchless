from __future__ import annotations

import sys
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
    # already running — the existing instance gets focus, this
    # process exits silently. Common case: user double-clicks the
    # desktop shortcut while Touchless is minimized to tray.
    if not acquire_single_instance():
        return 0

    # Tell Windows this process is its own app, not a generic
    # Python interpreter, so the taskbar groups our windows under
    # the Touchless icon instead of the python.exe icon. Only
    # matters for source / dev runs -- the PyInstaller-built
    # Touchless.exe carries its icon directly in the binary and
    # Windows uses that. Safe no-op on non-Windows / older builds.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Touchless.App.MarkovK"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationName(APP_NAME)

    icon_path = _resolve_app_icon()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    config = load_config()
    save_config(config)

    def _build_window() -> MainWindow:
        w = MainWindow(config)
        if icon_path is not None:
            w.setWindowIcon(QIcon(str(icon_path)))
        return w

    TouchlessSplash.run_with(_build_window, config.accent_color, app)
    return app.exec()

# Author: Konstantin Markov
