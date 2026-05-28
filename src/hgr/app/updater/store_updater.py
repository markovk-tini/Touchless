"""Microsoft Store update check for the UNPACKAGED (Win32 EXE) Store build.

The Store distributes Touchless as a "bring your own installer" Win32 app via
the Windows Package Manager (winget) channel. That means the installed app has
NO MSIX package identity, so the WinRT `StoreContext` API cannot be used. Instead
we query the Store's public `packageManifests` endpoint (the exact source winget
reads) for the latest published version + its installer URL, compare to the
running version, and — if newer — surface the SAME UpdateDialog the website
build uses.

Applying the update reuses the normal `Updater`: it downloads the Store's OWN
installer (the R2 URL the manifest points at — not GitHub) and runs it, so the
user updates from inside the app without ever visiting the Store. The version is
read from the Store, so the in-app prompt always matches what the Store offers.

Only the Store build runs this (build_channel() == 'store'); website/source
builds keep the GitHub `ReleaseChecker`. No third-party dependency — plain HTTP.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from .release_checker import ReleaseInfo, _is_newer, _strip_v_prefix
from ... import __version__ as RUNNING_VERSION


# The app's Microsoft Store product ID (Partner Center → Application identity →
# Store ID). Fixed for the lifetime of the listing.
STORE_PRODUCT_ID = "XPDLMKGM1SFQG1"
# Deep link to the Store product page — the graceful fallback if the manifest
# has no installer URL to self-apply.
STORE_DEEP_LINK = f"ms-windows-store://pdp/?productid={STORE_PRODUCT_ID}"
# winget's msstore source. Returns Data.Versions[].PackageVersion +
# .Installers[].InstallerUrl. Public, no auth.
_MANIFEST_URL = (
    "https://storeedgefd.dsx.mp.microsoft.com/v9.0/packageManifests/"
    f"{STORE_PRODUCT_ID}?Market=US"
)
_HTTP_TIMEOUT = 12.0


class StoreUpdateChecker(QThread):
    """Polls the Store manifest for a newer version. Emits update_available with
    a ReleaseInfo whose download_url is the Store's own installer (update_kind=
    'full-exe', so the existing Updater downloads + runs it), or no_update."""

    update_available = Signal(object)   # ReleaseInfo
    no_update = Signal()
    check_failed = Signal(str)          # parity with ReleaseChecker (manual UI)

    def __init__(self, parent=None, logger: Any = None) -> None:
        super().__init__(parent)
        self._logger = logger

    def run(self) -> None:
        try:
            version, installer_url = self._fetch_latest()
        except Exception as exc:
            self._log("store_manifest_fetch_failed", exc)
            self.check_failed.emit("Couldn't reach the Microsoft Store.")
            return
        if not version or not _is_newer(version, RUNNING_VERSION):
            self.no_update.emit()
            return
        info = ReleaseInfo(
            version=_strip_v_prefix(version),
            body=(
                f"Touchless {version} is available from the Microsoft Store.\n\n"
                "Click **Download Update** to install it now."
            ),
            download_url=installer_url or "",
            html_url=STORE_DEEP_LINK,
            # With an installer URL the normal Updater applies it in-app; with
            # none, fall back to 'store' so the button just opens the Store.
            update_kind="full-exe" if installer_url else "store",
        )
        self.update_available.emit(info)

    def _fetch_latest(self) -> Tuple[Optional[str], str]:
        """Return (latest_version, installer_url) from the Store manifest."""
        req = urllib.request.Request(
            _MANIFEST_URL,
            headers={"Accept": "application/json", "User-Agent": "Touchless-Updater"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        versions = (((data or {}).get("Data") or {}).get("Versions")) or []
        best_v: Optional[str] = None
        best_url = ""
        for v in versions:
            pv = str((v or {}).get("PackageVersion") or "").strip()
            if not pv:
                continue
            if best_v is None or _is_newer(pv, best_v):
                best_v = pv
                best_url = self._pick_installer_url(v.get("Installers") or [])
        return best_v, best_url

    @staticmethod
    def _pick_installer_url(installers: list) -> str:
        """Prefer the x64 installer URL; fall back to the first one present."""
        first = ""
        for inst in installers:
            url = str((inst or {}).get("InstallerUrl") or "").strip()
            if not url:
                continue
            if not first:
                first = url
            arch = str((inst or {}).get("Architecture") or "").lower()
            if arch in ("x64", "neutral", ""):
                return url
        return first

    def _log(self, kind: str, exc: Optional[BaseException] = None) -> None:
        if self._logger is None:
            return
        try:
            if exc is not None and hasattr(self._logger, "exception"):
                self._logger.exception(kind, exc)
            elif hasattr(self._logger, "event"):
                self._logger.event(kind)
        except Exception:
            pass
