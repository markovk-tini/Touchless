"""GitHub Releases poller.

Runs the HTTP fetch on a Qt worker thread so the main thread never
blocks on network I/O. Emits `update_available(ReleaseInfo)` when
the latest release tag parses to a version newer than the running
app's `__version__`. Emits `no_update()` for same/older. Emits
`check_failed(reason)` if the fetch errored — the caller can
silently ignore that (offline laptops, GitHub rate limit, etc.)
without ever bothering the user.

Why GitHub Releases API specifically:
- It's free, requires no auth for public repos at modest poll
  rates (60 unauthenticated req/hr/IP, way more than we need),
  and returns a structured JSON we can parse without scraping.
- The release `body` field is markdown — it's exactly what the
  user types into "release notes" on the GitHub Releases UI, so
  the maintainer's existing release-writing workflow doubles as
  the changelog source.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from ... import __version__ as RUNNING_VERSION

GITHUB_RELEASES_LATEST_URL = (
    "https://api.github.com/repos/markovk-tini/HGR-App/releases/latest"
)
GITHUB_RELEASES_LIST_URL = (
    "https://api.github.com/repos/markovk-tini/HGR-App/releases?per_page=20"
)
# Full installer (~2.4 GB). Used for first-time installs and for any
# release where the developer wants every user to do a clean
# reinstall (e.g. PySide6 / OpenCV / whisper.cpp updates).
INSTALLER_ASSET_NAME = "Touchless_Installer.exe"
# App-only update package (~50-150 MB). Just Touchless.exe + small
# project assets. Updater extracts it over the existing install
# without running an installer, so no UAC is ever needed (provided
# the install lives under %LOCALAPPDATA%\Programs\Touchless\, which
# the per-user installer puts it in by default).
APP_UPDATE_ZIP_PREFIX = "Touchless_App_Update"   # matches Touchless_App_Update_<ver>.zip
# macOS update assets. A Windows .exe/.zip can't be applied on a Mac (and
# vice-versa), so on macOS the checker matches these platform-specific names
# instead — a zipped .app bundle for in-place updates, and a .pkg for full
# (re)installs. Their prefixes are distinct from the Windows ones so a single
# GitHub release can carry both platforms' assets without cross-matching.
MAC_INSTALLER_ASSET_NAME = "Touchless.pkg"
MAC_APP_UPDATE_ZIP_PREFIX = "Touchless_Mac_Update"   # matches Touchless_Mac_Update_<ver>.zip
HTTP_TIMEOUT_SECONDS = 8.0

# When the full installer is too big for GitHub's 2GB asset limit,
# the developer hosts it on Cloudflare (or any HTTPS host) and
# embeds a marker in the release body for the updater to find.
#
# Conventions:
#   <!-- full-installer-url: https://touchless.example.com/v1.0.6/Touchless_Installer.exe -->
#   <!-- full-installer-size: 2576980378 -->         (optional, bytes)
#
# Both markers are HTML comments so they render invisibly in the
# user-facing release notes on GitHub and in the in-app Updates
# panel. The size is optional — if omitted, the dialog shows
# "Full update available" without a MB figure.
# HTTPS only — a release-body typo / compromised release editor that
# slipped `http://` in here would otherwise route every auto-updater
# through cleartext, and combined with the lack of digest verification
# downstream that's an unattended-RCE surface. Reviewers flagged this
# in the 1.1.4 audit as high-severity even though the cohort impact
# starts small (only ops who edit release bodies can trip it).
_FULL_INSTALLER_URL_RE = re.compile(
    r"<!--\s*full-installer-url:\s*(https://\S+?)\s*-->",
    re.IGNORECASE,
)
_FULL_INSTALLER_SIZE_RE = re.compile(
    r"<!--\s*full-installer-size:\s*(\d+)\s*-->",
    re.IGNORECASE,
)
# SHA-256 of the artifact at the URL above. Optional but strongly
# encouraged on every release — the Updater verifies the downloaded
# bytes against this hash BEFORE running them, catching CDN corruption,
# MITM attacks, and (combined with HTTPS) any tampering that doesn't
# also compromise the release editor's account. 64 hex chars; case-
# insensitive. Empty string in the marker means "no verification" —
# allowed for backward compatibility with older releases but logged
# as a warning at apply time.
_FULL_INSTALLER_SHA256_RE = re.compile(
    r"<!--\s*full-installer-sha256:\s*([0-9A-Fa-f]{64})\s*-->",
    re.IGNORECASE,
)
# Same shape for the app-zip asset (the small in-place update path
# website + Store users both end up on). When present, the Updater
# refuses to run the apply bat if the downloaded zip's SHA-256
# doesn't match.
_APP_UPDATE_ZIP_SHA256_RE = re.compile(
    r"<!--\s*app-update-zip-sha256:\s*([0-9A-Fa-f]{64})\s*-->",
    re.IGNORECASE,
)
# macOS-specific SHA-256 markers (kept distinct from the Windows ones so a
# combined release can publish both platforms' hashes side by side).
_MAC_INSTALLER_SHA256_RE = re.compile(
    r"<!--\s*mac-installer-sha256:\s*([0-9A-Fa-f]{64})\s*-->",
    re.IGNORECASE,
)
_MAC_APP_UPDATE_ZIP_SHA256_RE = re.compile(
    r"<!--\s*mac-app-update-zip-sha256:\s*([0-9A-Fa-f]{64})\s*-->",
    re.IGNORECASE,
)
# Combined regex used to strip the markers from the body before
# showing it to the user. Captures any <!-- (full-installer-*|app-update-*|mac-*) ... -->
# line, including its trailing newline if present.
_FULL_INSTALLER_MARKER_RE = re.compile(
    r"<!--\s*(?:full-installer-(?:url|size|sha256)|app-update-zip-sha256"
    r"|mac-installer-sha256|mac-app-update-zip-sha256):[^>]*-->\s*\n?",
    re.IGNORECASE,
)


def _parse_external_full_installer(body: str) -> tuple[str, int]:
    """Pull `(url, size_bytes)` out of the release body. Returns
    `("", 0)` if no marker is present."""
    if not body:
        return ("", 0)
    url_match = _FULL_INSTALLER_URL_RE.search(body)
    if not url_match:
        return ("", 0)
    url = url_match.group(1).strip()
    size = 0
    size_match = _FULL_INSTALLER_SIZE_RE.search(body)
    if size_match:
        try:
            size = int(size_match.group(1))
        except ValueError:
            size = 0
    return (url, size)


def _parse_full_installer_sha256(body: str) -> str:
    """Pull the full-installer SHA-256 marker from the release body.
    Returns "" if no marker (legacy releases). Lowercased hex."""
    if not body:
        return ""
    m = _FULL_INSTALLER_SHA256_RE.search(body)
    return m.group(1).lower() if m else ""


def _parse_app_update_zip_sha256(body: str) -> str:
    """Pull the app-update-zip SHA-256 marker from the release body.
    Returns "" if no marker. Lowercased hex."""
    if not body:
        return ""
    m = _APP_UPDATE_ZIP_SHA256_RE.search(body)
    return m.group(1).lower() if m else ""


def _parse_mac_installer_sha256(body: str) -> str:
    """Pull the macOS .pkg SHA-256 marker from the release body.
    Returns "" if no marker. Lowercased hex."""
    if not body:
        return ""
    m = _MAC_INSTALLER_SHA256_RE.search(body)
    return m.group(1).lower() if m else ""


def _parse_mac_app_update_zip_sha256(body: str) -> str:
    """Pull the macOS app-zip SHA-256 marker from the release body.
    Returns "" if no marker. Lowercased hex."""
    if not body:
        return ""
    m = _MAC_APP_UPDATE_ZIP_SHA256_RE.search(body)
    return m.group(1).lower() if m else ""


def _strip_installer_markers(body: str) -> str:
    """Remove the marker comments from the body before display."""
    if not body:
        return body
    return _FULL_INSTALLER_MARKER_RE.sub("", body).rstrip()


@dataclass(frozen=True)
class ReleaseInfo:
    version: str            # e.g. "1.0.2" (tag with leading 'v' stripped)
    body: str               # Markdown release notes from GitHub
    download_url: str       # Preferred asset URL (zip if present, else exe)
    html_url: str           # GitHub release page (fallback)
    size_bytes: int = 0     # Preferred asset size, 0 if unknown
    # update_kind == "app-zip" means download_url points to the small
    # app-only zip and Updater should extract over the install dir;
    # "full-exe" means it points to the .exe installer and Updater
    # should launch it with silent flags. Defaults to "full-exe" so
    # legacy releases (only .exe asset) keep working.
    update_kind: str = "full-exe"
    # If both assets are present, this carries the OTHER one as a
    # fallback link in case the preferred asset fails to download or
    # extract. Empty if no fallback.
    fallback_url: str = ""
    # SHA-256 of the artifact at download_url, lowercase hex. The
    # Updater verifies this against the actual download bytes before
    # invoking the apply step — defends against CDN corruption + MITM
    # + a quietly-swapped artifact. Empty string means "no hash
    # published" (older releases); the Updater logs a warning and
    # proceeds. Required for the trust story on auto-update (where
    # there's no human-in-the-loop to sanity-check), so future
    # release scripts should always populate the marker.
    expected_sha256: str = ""


def _strip_v_prefix(version_str: str) -> str:
    """Strip leading 'v' so 'v1.2.3' and '1.2.3' compare equal."""
    return re.sub(r"^v", "", str(version_str or "").strip(), flags=re.IGNORECASE)


# Charset for a "safe" version string we're willing to interpolate into
# bat-helper command lines, registry values, and filesystem paths. Must
# start with a digit; the only allowed punctuation is `.`, `-`, `+` and
# alphanumerics. Hard-capped at 64 chars to bound any downstream buffer.
# Anything outside this charset is rejected at ReleaseInfo construction
# (and in StoreUpdateChecker for the Store path) so a tag like
# "v1.1.4\" /v Bad" or "v1.1.4 & calc.exe" can't reach a shell.
_SAFE_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+\-]{0,63}$")


def is_safe_version(version_str: str) -> bool:
    """True iff `version_str` matches our hardened version charset.
    Used as a guard at the trust boundary where remote tag data first
    becomes a ReleaseInfo. The auto-updater interpolates the result
    into shell + registry + filename contexts and is too critical a
    path to leave that interpolation unchecked."""
    if not version_str:
        return False
    return bool(_SAFE_VERSION_RE.match(version_str))


def _parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Legacy digit-tuple parser kept as a fallback for the rare
    case where a tag isn't valid PEP 440. Returns (0,) if unparseable,
    which makes any real version compare as 'newer'."""
    cleaned = _strip_v_prefix(version_str)
    cleaned = re.split(r"[-+]", cleaned, maxsplit=1)[0]
    parts = re.findall(r"\d+", cleaned)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts)


def _is_newer(remote: str, local: str) -> bool:
    """Compare two version strings using PEP 440 semantics so betas
    and post-releases work correctly:

      1.0.9 < 1.0.9.post1 < 1.0.10 < 1.1.0b1 < 1.1.0rc1 < 1.1.0

    The previous digit-tuple parser stripped letters and treated
    '1.0.9a' as equal to '1.0.9' — which silently broke updater
    delivery when 1.0.9a was published. PEP 440 (via the `packaging`
    library, which is already installed alongside pip) handles all
    standard release/pre-release/post-release identifiers correctly.

    Falls back to the legacy digit-tuple parser if either string
    isn't valid PEP 440 (e.g. a hand-tagged release with an exotic
    suffix). The fallback is conservative: it doesn't claim 'newer'
    for anything that's purely letter-stripped equal.
    """
    try:
        from packaging.version import Version, InvalidVersion
    except Exception:
        # `packaging` should always be present (it ships with pip),
        # but if for some reason it's missing in a hostile env, fall
        # straight back to the legacy comparator.
        return _parse_version_tuple(remote) > _parse_version_tuple(local)
    try:
        return Version(_strip_v_prefix(remote)) > Version(_strip_v_prefix(local))
    except InvalidVersion:
        return _parse_version_tuple(remote) > _parse_version_tuple(local)


class _CheckWorker(QObject):
    finished = Signal()
    update_available = Signal(object)   # ReleaseInfo
    no_update = Signal()
    check_failed = Signal(str)

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                GITHUB_RELEASES_LATEST_URL,
                headers={
                    # Bare github.com requests get user-agent-blocked.
                    "User-Agent": "Touchless-Updater/1.0",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            self.check_failed.emit(f"network: {exc!s}")
            self.finished.emit()
            return
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.check_failed.emit(f"parse: {exc!s}")
            self.finished.emit()
            return
        except Exception as exc:  # pragma: no cover — belt and suspenders
            self.check_failed.emit(f"unexpected: {type(exc).__name__}")
            self.finished.emit()
            return

        try:
            tag = str(data.get("tag_name") or "").strip()
            body_raw = str(data.get("body") or "").strip()
            html_url = str(data.get("html_url") or "").strip()
            assets = data.get("assets") or []
            installer_url = ""
            installer_size = 0
            zip_url = ""
            zip_size = 0
            # Match this platform's asset names. On macOS a Windows .exe/.zip
            # is useless (and vice-versa), so the target names differ per OS.
            is_mac = sys.platform == "darwin"
            installer_name = (MAC_INSTALLER_ASSET_NAME if is_mac else INSTALLER_ASSET_NAME).lower()
            zip_prefix = (MAC_APP_UPDATE_ZIP_PREFIX if is_mac else APP_UPDATE_ZIP_PREFIX).lower()
            for asset in assets:
                name = str(asset.get("name") or "").strip()
                lname = name.lower()
                url = str(asset.get("browser_download_url") or "").strip()
                try:
                    size = int(asset.get("size") or 0)
                except (TypeError, ValueError):
                    size = 0
                if lname == installer_name:
                    installer_url = url
                    installer_size = size
                elif lname.startswith(zip_prefix) and lname.endswith(".zip"):
                    zip_url = url
                    zip_size = size

            # If GitHub doesn't carry the full installer (2GB asset
            # limit blocks our 2.4GB .exe), look for a Cloudflare-
            # or other-host URL embedded in the release body via
            # the <!-- full-installer-url: ... --> marker. The
            # developer adds this when uploading the .exe to their
            # CDN; the auto-updater treats the external URL as
            # equivalent to a GitHub-hosted asset. macOS .pkg is small
            # enough to live on GitHub, so this CDN fallback is Windows-only.
            if not installer_url and not is_mac:
                ext_url, ext_size = _parse_external_full_installer(body_raw)
                if ext_url:
                    installer_url = ext_url
                    installer_size = ext_size

            # Hide the marker comments from the body shown to the
            # user — they're metadata for us, not changelog content.
            body = _strip_installer_markers(body_raw)
        except Exception as exc:
            self.check_failed.emit(f"shape: {type(exc).__name__}")
            self.finished.emit()
            return

        if not tag:
            self.check_failed.emit("missing tag_name in release payload")
            self.finished.emit()
            return

        if not _is_newer(tag, RUNNING_VERSION):
            self.no_update.emit()
            self.finished.emit()
            return

        # Prefer the small zip when it's present AND the running
        # app's install directory is user-writable. If the user is
        # on a legacy Program Files install (admin required to
        # write), the silent zip path can't succeed because the
        # helper batch runs unelevated — so we force the full
        # installer path, which can elevate via UAC. The user only
        # sees the smaller download once they migrate to the
        # per-user LocalAppData install location.
        try:
            from .updater import Updater
            install_writable = Updater.is_install_dir_writable()
        except Exception:
            install_writable = True

        if zip_url and install_writable:
            preferred_url = zip_url
            preferred_size = zip_size
            kind = "app-zip"
            fallback = installer_url
        elif installer_url:
            preferred_url = installer_url
            preferred_size = installer_size
            kind = "full-exe"
            fallback = zip_url   # (unused for full-exe path, but kept for symmetry)
        else:
            # Only zip exists but install isn't writable. Surface
            # the zip URL anyway with full-exe kind disabled — the
            # dialog will offer "Open release page" via html_url.
            preferred_url = ""
            preferred_size = 0
            kind = "full-exe"
            fallback = ""

        version_clean = re.sub(r"^v", "", tag, flags=re.IGNORECASE)
        if not is_safe_version(version_clean):
            # Refuse to ship a tag we can't safely interpolate downstream.
            # See is_safe_version() for the trust-boundary argument. The
            # user just sees "no update available", which is correct —
            # we have no safe way to apply this one.
            self.check_failed.emit(
                f"unsafe version string from release tag: {version_clean!r}"
            )
            self.finished.emit()
            return
        # Pull the SHA-256 marker that matches the asset we're going to
        # download. The Updater verifies the downloaded bytes against
        # this before invoking apply — silent corruption / MITM / a
        # quietly-swapped asset all get rejected at this gate. Both
        # markers are best-effort: missing markers mean "no verification"
        # which the Updater logs but proceeds with (backward compat for
        # legacy releases that pre-date the marker convention).
        if kind == "app-zip":
            expected_sha256 = (
                _parse_mac_app_update_zip_sha256(body_raw)
                if is_mac
                else _parse_app_update_zip_sha256(body_raw)
            )
        else:
            expected_sha256 = (
                _parse_mac_installer_sha256(body_raw)
                if is_mac
                else _parse_full_installer_sha256(body_raw)
            )
        info = ReleaseInfo(
            version=version_clean,
            body=body,
            download_url=preferred_url,
            html_url=html_url,
            size_bytes=preferred_size,
            update_kind=kind,
            fallback_url=fallback,
            expected_sha256=expected_sha256,
        )
        self.update_available.emit(info)
        self.finished.emit()


class ReleaseChecker(QObject):
    """Public facade. Owns its own QThread so callers don't have to.

    Usage:
        self._checker = ReleaseChecker(parent=self)
        self._checker.update_available.connect(self._on_update_available)
        self._checker.start()
    """

    update_available = Signal(object)   # ReleaseInfo
    no_update = Signal()
    check_failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _CheckWorker | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = _CheckWorker()
        self._worker.moveToThread(self._thread)
        self._worker.update_available.connect(self.update_available)
        self._worker.no_update.connect(self.no_update)
        self._worker.check_failed.connect(self.check_failed)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _cleanup(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()


@dataclass(frozen=True)
class ReleaseHistoryEntry:
    """A single release for the Updates settings panel's history list."""
    version: str       # tag with leading 'v' stripped
    body: str          # markdown release notes
    published_at: str  # ISO 8601 string from GitHub
    html_url: str
    is_current: bool   # True when this matches RUNNING_VERSION


class _HistoryWorker(QObject):
    finished = Signal()
    history_loaded = Signal(list)        # list[ReleaseHistoryEntry]
    history_failed = Signal(str)

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                GITHUB_RELEASES_LIST_URL,
                headers={
                    "User-Agent": "Touchless-Updater/1.0",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            self.history_failed.emit(f"{type(exc).__name__}: {exc!s}")
            self.finished.emit()
            return
        entries: list[ReleaseHistoryEntry] = []
        try:
            for item in data:
                tag = str(item.get("tag_name") or "").strip()
                if not tag:
                    continue
                version = re.sub(r"^v", "", tag, flags=re.IGNORECASE)
                entries.append(
                    ReleaseHistoryEntry(
                        version=version,
                        body=_strip_installer_markers(str(item.get("body") or "").strip()),
                        published_at=str(item.get("published_at") or "").strip(),
                        html_url=str(item.get("html_url") or "").strip(),
                        is_current=(version == RUNNING_VERSION),
                    )
                )
        except Exception as exc:
            self.history_failed.emit(f"shape: {type(exc).__name__}")
            self.finished.emit()
            return
        self.history_loaded.emit(entries)
        self.finished.emit()


class ReleaseHistoryFetcher(QObject):
    """Fetches the full list of releases for the Updates settings
    panel. Runs on a worker thread; emits `history_loaded(list)`
    on success or `history_failed(str)` on any error."""

    history_loaded = Signal(list)
    history_failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _HistoryWorker | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = _HistoryWorker()
        self._worker.moveToThread(self._thread)
        self._worker.history_loaded.connect(self.history_loaded)
        self._worker.history_failed.connect(self.history_failed)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _cleanup(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()

# Author: Konstantin Markov
