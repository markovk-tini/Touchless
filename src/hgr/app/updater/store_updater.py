"""Microsoft Store update check for the UNPACKAGED (Win32 EXE) Store build.

The Store distributes Touchless as a "bring your own installer" Win32 app via
the Windows Package Manager (winget) channel. That means the installed app has
NO MSIX package identity, so the WinRT `StoreContext` API cannot be used.

v1.1.9.1: GitHub `/releases/latest` is the prompt trigger (same as website
builds). Release process is still Store-certified first, then GitHub, so
Store users see the popup when GitHub is published. The payload is the
GitHub app-zip (~140 MB in-place), not Partner Center's full installer.
The Store `packageManifests` listing is a fallback if GitHub has nothing
newer (or the latest fetch fails), and its installer URL is the fallback
when the zip is missing.

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

from .release_checker import (
    GITHUB_RELEASES_LATEST_URL,
    ReleaseInfo,
    _is_newer,
    _strip_v_prefix,
    is_safe_version,
    _parse_app_update_zip_sha256,
    _parse_full_installer_sha256,
)
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
# GitHub release endpoint — used to fetch the matching version's release notes
# AND the small app-zip URL when available, so Store users get the same fast
# in-app update path as website users instead of redownloading the 1.17 GB
# installer every release.
_GH_RELEASE_BY_TAG = (
    "https://api.github.com/repos/markovk-tini/HGR-App/releases/tags/v{tag}"
)
_HTTP_TIMEOUT = 12.0


class StoreUpdateChecker(QThread):
    """Polls GitHub for a newer version (Store builds). Emits update_available
    with the GitHub app-zip when present so Store users get the same in-place
    update website users do, not Partner Center's full installer."""

    update_available = Signal(object)   # ReleaseInfo
    no_update = Signal()
    check_failed = Signal(str)          # parity with ReleaseChecker (manual UI)

    def __init__(self, parent=None, logger: Any = None) -> None:
        super().__init__(parent)
        self._logger = logger

    def run(self) -> None:
        gh_tag = ""
        gh_html = ""
        try:
            gh_tag, gh_html = self._fetch_github_latest_meta()
        except Exception as exc:
            self._log("github_latest_fetch_failed", exc)

        gh_ver = _strip_v_prefix(gh_tag)
        if gh_ver and is_safe_version(gh_ver) and _is_newer(gh_ver, RUNNING_VERSION):
            installer_url = ""
            try:
                _store_ver, installer_url = self._fetch_latest()
            except Exception as exc:
                self._log("store_manifest_fetch_failed", exc)
            self._emit_update(gh_ver, gh_html or STORE_DEEP_LINK, installer_url)
            return

        # GitHub has nothing newer (or the latest fetch failed). Fall
        # back to the Store listing so a Store-only publish still surfaces.
        try:
            version, installer_url = self._fetch_latest()
        except Exception as exc:
            self._log("store_manifest_fetch_failed", exc)
            self.check_failed.emit("Couldn't reach GitHub or the Microsoft Store.")
            return
        if not version or not _is_newer(version, RUNNING_VERSION):
            self.no_update.emit()
            return
        self._emit_update(_strip_v_prefix(version), STORE_DEEP_LINK, installer_url)

    def _emit_update(self, version: str, html_url: str, installer_url: str) -> None:
        version_clean = _strip_v_prefix(version)
        if not is_safe_version(version_clean):
            self._log("store_unsafe_version", None)
            self.no_update.emit()
            return

        gh_body, gh_body_raw, gh_zip_url, gh_zip_size = self._fetch_github_release(
            version_clean
        )

        try:
            from .updater import Updater
            install_writable = Updater.is_install_dir_writable()
        except Exception:
            install_writable = True

        if gh_zip_url and install_writable:
            preferred_url = gh_zip_url
            preferred_size = gh_zip_size
            kind = "app-zip"
            fallback = installer_url or ""
        elif installer_url:
            preferred_url = installer_url
            preferred_size = 0
            kind = "full-exe"
            fallback = gh_zip_url or ""
        else:
            preferred_url = ""
            preferred_size = 0
            kind = "store"
            fallback = ""

        if gh_body:
            body = gh_body
        else:
            body = (
                f"Touchless {version_clean} is available.\n\n"
                "Click **Download Update** to install it now."
            )

        if kind == "app-zip":
            expected_sha256 = _parse_app_update_zip_sha256(gh_body_raw)
        elif kind == "full-exe":
            expected_sha256 = _parse_full_installer_sha256(gh_body_raw)
        else:
            expected_sha256 = ""
        self.update_available.emit(
            ReleaseInfo(
                version=version_clean,
                body=body,
                download_url=preferred_url,
                html_url=html_url or STORE_DEEP_LINK,
                size_bytes=preferred_size,
                update_kind=kind,
                fallback_url=fallback,
                expected_sha256=expected_sha256,
            )
        )

    def _fetch_github_latest_meta(self) -> Tuple[str, str]:
        """Return (tag_name, html_url) from GitHub /releases/latest."""
        req = urllib.request.Request(
            GITHUB_RELEASES_LATEST_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Touchless-Updater",
            },
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        tag = str((payload or {}).get("tag_name") or "").strip()
        html = str((payload or {}).get("html_url") or "").strip()
        return tag, html

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

    def _fetch_github_release(self, version: str) -> Tuple[str, str, str, int]:
        """Look up the GitHub release for `version` and return
        (release_body_clean, release_body_raw, app_zip_url, app_zip_size_bytes).

        Two body forms: the CLEAN body has the SHA/URL marker comments
        stripped (suitable for showing the user), the RAW body is the
        original text from GitHub (used to parse markers downstream).

        Best-effort. Returns ("", "", "", 0) on any failure — the caller
        falls back to the Store installer URL + a generic prompt.
        Never raises.

        Why we fetch this even on the Store channel: the Store's manifest
        carries no changelog and only the full installer URL. By pulling
        the matching version's GitHub release we get the same fast app-zip
        path website users enjoy AND the actual release notes — both of
        which the user explicitly asked for after the 1.1.3 in-app update
        felt like a slow generic reinstall.
        """
        try:
            tag = _strip_v_prefix(version)
            url = _GH_RELEASE_BY_TAG.format(tag=tag)
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "Touchless-Updater",
                },
            )
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            self._log("github_release_fetch_failed", exc)
            return ("", "", "", 0)

        body_raw = str((payload or {}).get("body") or "").strip()
        # Strip marker comments before showing to the user — they're
        # metadata for the Updater, not changelog content. The raw body
        # is kept separately so downstream can parse SHA-256 markers.
        try:
            from .release_checker import _strip_installer_markers
            body = _strip_installer_markers(body_raw)
        except Exception:
            body = body_raw
        zip_url = ""
        zip_size = 0
        # Strict prefix match — case-insensitive — so a future asset
        # like "Touchless_App_Update_Symbols_<ver>.zip" or any other
        # name that happens to contain "App_Update" doesn't get picked
        # up by accident. The earlier substring match was flagged in
        # the 1.1.4 audit as a real footgun (wrong-asset selection
        # leaves the user on the OLD Touchless.exe after the apply bat
        # fails on a missing staged Touchless.exe).
        prefix = "touchless_app_update_"
        for asset in (payload or {}).get("assets") or []:
            name = str((asset or {}).get("name") or "")
            lname = name.lower()
            if lname.startswith(prefix) and lname.endswith(".zip"):
                zip_url = str((asset or {}).get("browser_download_url") or "")
                try:
                    zip_size = int((asset or {}).get("size") or 0)
                except Exception:
                    zip_size = 0
                break
        return (body, body_raw, zip_url, zip_size)

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
