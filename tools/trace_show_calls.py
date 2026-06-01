"""Boot Touchless with Qt show-paths fully instrumented. Logs a stack
trace for every top-level widget that becomes visible. Catches show(),
showNormal(), showMaximized(), setVisible(True), and Qt-internal
ShowEvent.

Usage:
    python tools/trace_show_calls.py
"""
from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PySide6.QtWidgets import QWidget

_orig_setVisible = QWidget.setVisible
_seen: set[int] = set()
_count = [0]


def _emit(self, label: str) -> None:
    try:
        if self.parent() is not None:
            return
        wid = int(self.winId()) if hasattr(self, "winId") else id(self)
    except Exception:
        wid = id(self)
    if wid in _seen:
        return
    _seen.add(wid)
    _count[0] += 1
    n = _count[0]
    cls = type(self).__name__
    try:
        title = self.windowTitle()
    except Exception:
        title = "?"
    try:
        size = f"{self.width()}x{self.height()}"
    except Exception:
        size = "?"
    try:
        pos = f"({self.x()},{self.y()})"
    except Exception:
        pos = "?"
    sys.stderr.write(
        f"\n>>> {label} #{n}  cls={cls}  size={size}  pos={pos}  title={title!r}\n"
    )
    for frame in traceback.extract_stack()[:-2]:
        if "site-packages" in frame.filename and "PySide" in frame.filename:
            continue
        if "trace_show_calls.py" in frame.filename:
            continue
        sys.stderr.write(
            f"    at {frame.filename}:{frame.lineno} in {frame.name}\n"
        )
    sys.stderr.flush()


def _traced_setVisible(self, visible):
    try:
        if visible:
            _emit(self, "VISIBLE")
    except Exception:
        pass
    return _orig_setVisible(self, visible)


QWidget.setVisible = _traced_setVisible

from hgr.app.main import main as _main  # noqa: E402
_main()
