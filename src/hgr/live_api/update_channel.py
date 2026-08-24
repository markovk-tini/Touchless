"""Update channel hardening.

Phase-2. Touchless auto-updates from a Cloudflare R2 bucket via a
ShellExecuteW SW_HIDE spawn (see project_auto_update_mechanism
memory). The transport works; what's been missing is a hardening
layer that validates manifest integrity, signature provenance, and
rollback safety BEFORE the elevated installer launches.

This module provides:

  * `parse_manifest`  — strict JSON schema for the version manifest
                        (the file at /windows/latest.json).
  * `verify_manifest` — fields present, version formatted right,
                        SHA-256 checksum non-empty, signing chain
                        identifier matches expected ("Konstantin
                        Markov" individual cert).
  * `is_eligible`     — version comparison + skip-channel filter.
  * `download_plan`   — what to fetch, in what order, and where to
                        write it; lets the caller checkpoint between
                        steps so a mid-download crash doesn't trash
                        the install.
  * `verify_payload`  — once downloaded, recompute SHA-256 and
                        match it against the manifest. Mismatch =
                        refuse to launch.

The actual downloader / spawn lives elsewhere; this module owns the
SAFETY POLICY around it. Zero-dep so the test runner doesn't need
network access.

Author: Konstantin Markov
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Expected signer identity for the Touchless installer. The Azure
# Artifact Signing cert is registered to "Konstantin Markov" as an
# Individual code-signing cert (see project_signing_setup memory).
EXPECTED_SIGNER = "Konstantin Markov"


# Updates the auto-updater knows about. Add more channels as
# distribution evolves (beta, dev, etc.).
KNOWN_CHANNELS = ("stable", "beta", "dev")


SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?P<pre>[-+][A-Za-z0-9.\-]+)?$"
)


@dataclass
class UpdateManifest:
    """One version's release metadata."""
    version: str
    channel: str
    sha256_installer: str
    installer_url: str
    notes: str = ""
    signer: str = ""
    size_bytes: int = 0
    min_supported_version: str = "0.0.0"
    requires_rollback_safe: bool = False  # True for migrations w/o downgrade
    extras: Dict[str, Any] = field(default_factory=dict)


# ---- parsing -----------------------------------------------------------

def parse_manifest(blob: str | bytes) -> UpdateManifest:
    """Parse + validate a manifest JSON blob. Raises ValueError on
    schema problems; callers should catch and refuse the update."""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", errors="strict")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return _from_dict(data)


def _from_dict(d: Dict[str, Any]) -> UpdateManifest:
    required = ("version", "channel", "sha256_installer", "installer_url")
    for k in required:
        if k not in d or not d[k]:
            raise ValueError(f"manifest missing required field: {k}")
    ver = str(d["version"]).strip()
    if not SEMVER_RE.match(ver):
        raise ValueError(f"version must be semver-like: {ver!r}")
    channel = str(d["channel"]).strip().lower()
    if channel not in KNOWN_CHANNELS:
        raise ValueError(f"unknown channel: {channel!r}")
    sha = str(d["sha256_installer"]).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("sha256_installer must be a 64-char hex digest")
    url = str(d["installer_url"]).strip()
    if not url.lower().startswith("https://"):
        raise ValueError("installer_url must be https://")
    return UpdateManifest(
        version=ver,
        channel=channel,
        sha256_installer=sha,
        installer_url=url,
        notes=str(d.get("notes") or "")[:4000],
        signer=str(d.get("signer") or "").strip(),
        size_bytes=int(d.get("size_bytes") or 0),
        min_supported_version=str(d.get("min_supported_version")
                                  or "0.0.0").strip(),
        requires_rollback_safe=bool(d.get("requires_rollback_safe")),
        extras={k: v for k, v in d.items()
                if k not in ("version", "channel", "sha256_installer",
                             "installer_url", "notes", "signer",
                             "size_bytes", "min_supported_version",
                             "requires_rollback_safe")},
    )


# ---- verification ------------------------------------------------------

def verify_manifest(m: UpdateManifest,
                    *, expected_signer: str = EXPECTED_SIGNER
                    ) -> Tuple[bool, str]:
    """Run integrity checks on the parsed manifest. Returns
    (ok, reason). Should be called immediately after parse_manifest."""
    if not m.sha256_installer or len(m.sha256_installer) != 64:
        return False, "sha256_installer length invalid"
    if not m.installer_url.lower().startswith("https://"):
        return False, "installer_url must be https"
    # Signer must be PRESENT and MATCH — empty/omitted is treated as
    # an attack. Prior version short-circuited the equality check on
    # `m.signer`, which let a tampered manifest strip the field and
    # silently pass verification (SEC-001 audit finding).
    if expected_signer:
        if not m.signer:
            return False, "manifest signer missing — refusing update"
        if m.signer != expected_signer:
            return False, (f"unexpected signer {m.signer!r} "
                           f"(expected {expected_signer!r})")
    return True, "ok"


def is_eligible(*, current_version: str, manifest: UpdateManifest,
                user_channel: str = "stable",
                skip_versions: Optional[List[str]] = None
                ) -> Tuple[bool, str]:
    """Decide if THIS user should pick up THIS manifest.
    Returns (eligible, reason)."""
    if manifest.channel != user_channel.lower():
        return False, (f"channel mismatch ({manifest.channel} != "
                       f"{user_channel.lower()})")
    if compare_versions(manifest.version, current_version) <= 0:
        return False, "not newer than current"
    if (skip_versions and manifest.version in skip_versions):
        return False, "version is in skip list"
    if compare_versions(current_version,
                        manifest.min_supported_version) < 0:
        # The new release explicitly requires user to be at a higher
        # baseline. Auto-update can't bridge that — surface to user.
        return False, ("current version too old for direct upgrade; "
                       f"min supported = {manifest.min_supported_version}")
    return True, "ok"


def compare_versions(a: str, b: str) -> int:
    """Semver-ish compare. Returns -1 if a<b, 0 if equal, 1 if a>b.
    Pre-release tags are sorted lexicographically as a tiebreaker."""
    ma = SEMVER_RE.match(a or "0.0.0")
    mb = SEMVER_RE.match(b or "0.0.0")
    if not ma or not mb:
        # Fall back to string compare for non-semver inputs.
        return (a > b) - (a < b)
    pa = (int(ma["major"]), int(ma["minor"]), int(ma["patch"]))
    pb = (int(mb["major"]), int(mb["minor"]), int(mb["patch"]))
    if pa != pb:
        return -1 if pa < pb else 1
    pre_a, pre_b = ma["pre"] or "", mb["pre"] or ""
    # No pre-release > with pre-release ("1.0.0" > "1.0.0-rc1").
    if (pre_a == "") != (pre_b == ""):
        return 1 if pre_a == "" else -1
    return (pre_a > pre_b) - (pre_a < pre_b)


def verify_payload(*, file_path: str,
                   expected_sha256: str) -> Tuple[bool, str]:
    """Recompute the SHA-256 of `file_path` and compare to expected.
    Returns (ok, reason)."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
    except FileNotFoundError:
        return False, "payload file not found"
    except OSError as exc:
        return False, f"could not read payload: {exc}"
    actual = h.hexdigest().lower()
    expected = (expected_sha256 or "").strip().lower()
    if actual != expected:
        return False, f"sha256 mismatch (got {actual[:12]}…, "\
                      f"expected {expected[:12]}…)"
    return True, "ok"


_SAFE_INSTALLER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.exe$")


def _safe_installer_filename(installer_url: str) -> str:
    """Pull a SAFE filename from `installer_url`. Rejects anything
    containing `..`, `/`, `\\`, or characters outside `[A-Za-z0-9._-]`.
    Falls back to the canonical 'Touchless_Installer.exe' name when
    the URL ends in nothing usable. SEC-004 audit hardening — without
    this an attacker-controlled installer_url could write outside
    download_dir."""
    raw = installer_url.rsplit("/", 1)[-1].split("?", 1)[0]
    raw = raw.split("#", 1)[0]
    # URL-decode percent-escapes before checking (attackers love %2E%2E).
    try:
        from urllib.parse import unquote
        raw = unquote(raw)
    except Exception:
        pass
    raw = raw.strip()
    if not raw or ".." in raw or "/" in raw or "\\" in raw:
        return "Touchless_Installer.exe"
    if not _SAFE_INSTALLER_NAME_RE.match(raw):
        return "Touchless_Installer.exe"
    return raw


def download_plan(m: UpdateManifest, *,
                  download_dir: str) -> List[Dict[str, Any]]:
    """Return an ordered list of step dicts for the downloader.
    Each step is independently checkpointable. Today there's only one
    step (installer), but the shape leaves room for delta updates +
    code-sign cert refresh.

    `target_path` is anchored INSIDE `download_dir` — even with a
    maliciously-crafted installer_url, the target cannot escape the
    download directory."""
    from pathlib import Path
    fname = _safe_installer_filename(m.installer_url)
    base = Path(download_dir).resolve()
    target_path = (base / fname).resolve()
    # Final defense: refuse if resolve() resulted in escape.
    try:
        target_path.relative_to(base)
    except ValueError:
        target_path = base / "Touchless_Installer.exe"
    target = str(target_path)
    return [
        {
            "kind": "download_installer",
            "url": m.installer_url,
            "target_path": target,
            "expected_sha256": m.sha256_installer,
            "size_bytes": m.size_bytes,
        },
        {
            "kind": "verify",
            "target_path": target,
            "expected_sha256": m.sha256_installer,
        },
        {
            "kind": "launch_installer",
            "target_path": target,
            "requires_rollback_safe": m.requires_rollback_safe,
        },
    ]
