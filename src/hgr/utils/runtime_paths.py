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


def build_channel() -> str:
    """Return the distribution channel this build was produced for.

    'store'   — Microsoft Store build. Updates are delivered by the
                Store, so the in-app GitHub auto-updater must NOT run
                (Store policy expects apps to update through the Store,
                and a self-updater would fight it).
    'website' — direct-download build from the Touchless website /
                GitHub release. The in-app GitHub auto-updater is the
                only update path.

    Read from a `build_channel.txt` marker the build pipeline writes
    into the bundle (see hgr_app.spec + build_windows.bat). Defaults
    to 'website' when the marker is absent — source runs and every
    pre-existing build keep their historical auto-update behavior."""
    try:
        marker = resource_path("build_channel.txt")
        if marker.exists():
            value = marker.read_text(encoding="utf-8").strip().lower()
            if value == "store":
                return "store"
    except Exception:
        pass
    return "website"

# Author: Konstantin Markov
