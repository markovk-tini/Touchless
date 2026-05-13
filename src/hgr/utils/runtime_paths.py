from __future__ import annotations

import sys
from pathlib import Path


def app_base_path() -> Path:
    """Return the runtime base directory for source and PyInstaller builds.

    Frozen (PyInstaller): sys._MEIPASS, which is the temp dir where the
    bundle's assets/ tree is extracted -- the path the spec configures
    as the resource root.

    Source: the REPO ROOT, i.e. the directory that contains assets/,
    builder/, signing/, etc. Computed as three parents up from this
    file (utils -> hgr -> src -> repo root). The earlier .parent
    expression returned src/hgr/utils/, which silently mis-resolved
    every resource_path('assets', ...) lookup -- source runs got an
    empty QIcon, which is why the taskbar and tray icons fell back to
    the parent python.exe icon."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[3]


def resource_path(*parts: str) -> Path:
    return app_base_path().joinpath(*parts)

# Author: Konstantin Markov
