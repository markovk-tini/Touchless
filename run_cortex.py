"""Standalone launcher for the Iris Cortex visualization window.

Mirrors `run_app.py` exactly — injects src/ into sys.path before
importing — so you can preview the cortex without starting the
full Touchless app.

  python run_cortex.py

The window opens with an idle-scene seed (4 capability nodes +
a few placeholder context nodes + leaves). Real Iris events
only show up when the full app is running.

Author: Konstantin Markov
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgr.live_api.cortex.window import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov
