"""Standalone preview of the proposed Iris Cortex visual upgrades.

Opens a QWebEngineView pointed at demos/index.html, which links to all
five visual upgrade demos:
  1. plasma_core.html       — noise-displaced ShaderMaterial core
  2. memory_ribbon.html     — pearl-filament TubeGeometry on recall
  3. route_tree.html        — Westworld speech-tree branching
  4. gaze_brackets.html     — Fresnel shell + screen-space SVG brackets
  5. stress.html            — RGBShiftPass driven by system pressure

Each demo simulates the relevant input (voice level, recall fire,
routing layer chosen, etc.) so you can SEE the upgrade in motion
before committing to integration.

  python run_plasma_demo.py

Author: Konstantin Markov
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication, QMainWindow
    from PySide6.QtWebEngineWidgets import QWebEngineView

    # Share the live simulator's world-data injection so that clicking
    # ★ Live simulator from the gallery boots with real project data
    # (instead of the demo fallback). The script is attached to the
    # view once and survives in-page navigations.
    from run_iris_simulator import install_world_injection, load_world_payload

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Iris Cortex — Visualization Demos")
    win.resize(1320, 880)

    view = QWebEngineView()

    try:
        payload = load_world_payload()
        install_world_injection(view, payload)
        n = payload.get("project_count", 0)
        print(f"Injected real world data: {n} project(s) from {payload.get('source')}")
    except Exception as exc:
        # Demos that don't read IRIS_WORLD won't care; the simulator
        # has its own hardcoded fallback. Don't let injection failure
        # break the gallery.
        print(f"WARN: world-data injection skipped: {exc}", file=sys.stderr)

    html_path = (
        SRC / "hgr" / "live_api" / "cortex" / "web" / "demos" / "index.html"
    )
    if not html_path.exists():
        print(f"ERROR: demos index missing at {html_path}", file=sys.stderr)
        return 2
    view.load(QUrl.fromLocalFile(str(html_path)))
    win.setCentralWidget(view)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov
