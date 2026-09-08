"""Tests for update_channel hardening (Phase 2 B6)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hgr.live_api.update_channel import (  # noqa: E402
    EXPECTED_SIGNER, KNOWN_CHANNELS, UpdateManifest,
    compare_versions, download_plan, is_eligible,
    parse_manifest, verify_manifest, verify_payload,
)


def _good_manifest_blob(*, version="1.2.3", channel="stable",
                        signer=EXPECTED_SIGNER) -> str:
    sha = "0" * 64
    return json.dumps({
        "version": version,
        "channel": channel,
        "sha256_installer": sha,
        "installer_url": "https://hgr-downloads.touchless/x.exe",
        "notes": "release notes",
        "signer": signer,
        "size_bytes": 1024,
    })


# ---- parse_manifest ----------------------------------------------------

def test_parse_valid_manifest():
    m = parse_manifest(_good_manifest_blob())
    assert m.version == "1.2.3"
    assert m.channel == "stable"
    assert len(m.sha256_installer) == 64
    assert m.signer == EXPECTED_SIGNER


def test_parse_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_manifest("not-json")


def test_parse_rejects_non_object_root():
    with pytest.raises(ValueError, match="object"):
        parse_manifest("[]")


def test_parse_rejects_missing_required():
    blob = json.dumps({"version": "1.0.0"})  # no channel, sha, url
    with pytest.raises(ValueError, match="missing required"):
        parse_manifest(blob)


def test_parse_rejects_non_semver():
    blob = json.dumps({
        "version": "v1",
        "channel": "stable",
        "sha256_installer": "0" * 64,
        "installer_url": "https://x/y.exe",
    })
    with pytest.raises(ValueError, match="semver"):
        parse_manifest(blob)


def test_parse_rejects_unknown_channel():
    blob = json.dumps({
        "version": "1.0.0",
        "channel": "weird",
        "sha256_installer": "0" * 64,
        "installer_url": "https://x/y.exe",
    })
    with pytest.raises(ValueError, match="channel"):
        parse_manifest(blob)


def test_parse_rejects_short_sha256():
    blob = json.dumps({
        "version": "1.0.0",
        "channel": "stable",
        "sha256_installer": "deadbeef",
        "installer_url": "https://x/y.exe",
    })
    with pytest.raises(ValueError, match="64-char hex"):
        parse_manifest(blob)


def test_parse_rejects_non_https_url():
    blob = json.dumps({
        "version": "1.0.0",
        "channel": "stable",
        "sha256_installer": "0" * 64,
        "installer_url": "http://insecure.example/x.exe",
    })
    with pytest.raises(ValueError, match="https"):
        parse_manifest(blob)


def test_parse_carries_extras():
    blob = json.dumps({
        "version": "1.0.0",
        "channel": "stable",
        "sha256_installer": "0" * 64,
        "installer_url": "https://x/y.exe",
        "experimental_flag": True,
    })
    m = parse_manifest(blob)
    assert m.extras.get("experimental_flag") is True


# ---- verify_manifest ---------------------------------------------------

def test_verify_passes_for_expected_signer():
    m = parse_manifest(_good_manifest_blob())
    ok, _ = verify_manifest(m)
    assert ok is True


def test_verify_rejects_wrong_signer():
    m = parse_manifest(_good_manifest_blob(signer="Attacker Inc."))
    ok, reason = verify_manifest(m)
    assert ok is False
    assert "signer" in reason.lower()


def test_verify_rejects_missing_signer():
    # SEC-001 audit: missing/empty signer is treated as attack.
    m = parse_manifest(_good_manifest_blob(signer=""))
    ok, reason = verify_manifest(m)
    assert ok is False
    assert "signer" in reason.lower()


def test_verify_allows_missing_signer_when_expected_disabled():
    # When the caller explicitly opts out (expected_signer="") the
    # check is bypassed — for testing / unsigned-channel scenarios.
    m = parse_manifest(_good_manifest_blob(signer=""))
    ok, _ = verify_manifest(m, expected_signer="")
    assert ok is True


# ---- is_eligible -------------------------------------------------------

def test_eligible_when_newer_same_channel():
    m = parse_manifest(_good_manifest_blob(version="1.2.3"))
    ok, _ = is_eligible(current_version="1.2.2", manifest=m)
    assert ok is True


def test_not_eligible_when_same_version():
    m = parse_manifest(_good_manifest_blob(version="1.2.3"))
    ok, reason = is_eligible(current_version="1.2.3", manifest=m)
    assert ok is False
    assert "not newer" in reason


def test_not_eligible_when_in_skip_list():
    m = parse_manifest(_good_manifest_blob(version="1.2.3"))
    ok, reason = is_eligible(current_version="1.2.2", manifest=m,
                             skip_versions=["1.2.3"])
    assert ok is False
    assert "skip" in reason


def test_not_eligible_when_channel_mismatch():
    m = parse_manifest(_good_manifest_blob(version="1.2.3",
                                            channel="beta"))
    ok, reason = is_eligible(current_version="1.0.0", manifest=m,
                             user_channel="stable")
    assert ok is False
    assert "channel" in reason


def test_not_eligible_when_below_min_supported(monkeypatch):
    blob = json.dumps({
        "version": "2.0.0", "channel": "stable",
        "sha256_installer": "0" * 64,
        "installer_url": "https://x/y.exe",
        "min_supported_version": "1.5.0",
    })
    m = parse_manifest(blob)
    ok, reason = is_eligible(current_version="1.0.0", manifest=m)
    assert ok is False
    assert "min supported" in reason


# ---- compare_versions --------------------------------------------------

def test_compare_versions_basic():
    assert compare_versions("1.0.0", "1.0.0") == 0
    assert compare_versions("1.0.0", "1.0.1") == -1
    assert compare_versions("2.0.0", "1.9.9") == 1
    assert compare_versions("1.10.0", "1.9.0") == 1


def test_compare_versions_pre_release_lower_than_release():
    assert compare_versions("1.0.0-rc1", "1.0.0") == -1
    assert compare_versions("1.0.0", "1.0.0-rc1") == 1


def test_compare_versions_handles_invalid_input():
    # Falls back to string comparison.
    assert compare_versions("abc", "xyz") == -1


# ---- verify_payload ----------------------------------------------------

def test_verify_payload_matching_sha():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        content = b"hello touchless installer payload"
        f.write(content)
        f.flush()
        sha = hashlib.sha256(content).hexdigest()
        ok, reason = verify_payload(file_path=f.name,
                                    expected_sha256=sha)
    assert ok is True


def test_verify_payload_mismatch():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"corrupted")
        f.flush()
        ok, reason = verify_payload(file_path=f.name,
                                    expected_sha256="0" * 64)
    assert ok is False
    assert "mismatch" in reason


def test_verify_payload_missing_file():
    ok, reason = verify_payload(file_path="/no/such/file.exe",
                                expected_sha256="0" * 64)
    assert ok is False
    assert "not found" in reason


# ---- download_plan -----------------------------------------------------

def test_download_plan_has_all_steps():
    import tempfile, os
    m = parse_manifest(_good_manifest_blob())
    with tempfile.TemporaryDirectory() as tmp:
        plan = download_plan(m, download_dir=tmp)
    kinds = [s["kind"] for s in plan]
    assert kinds == ["download_installer", "verify", "launch_installer"]
    assert plan[0]["url"] == m.installer_url
    assert plan[1]["expected_sha256"] == m.sha256_installer


def test_download_plan_filename_extracted_from_url():
    import tempfile
    m = parse_manifest(_good_manifest_blob())
    with tempfile.TemporaryDirectory() as tmp:
        plan = download_plan(m, download_dir=tmp)
    assert plan[0]["target_path"].endswith("x.exe")


def test_download_plan_rejects_path_traversal_filename():
    # SEC-004 audit: attacker-controlled installer_url cannot escape
    # download_dir even when the URL contains '../' / '..\\' / encoded
    # variants.
    import tempfile
    from hgr.live_api.update_channel import _safe_installer_filename
    # The sanitizer itself.
    assert _safe_installer_filename(
        "https://x.com/..%5C..%5CWindows%5Cservice.exe") \
        == "Touchless_Installer.exe"
    # Note: "https://x.com/../../evil.exe" — rsplit('/',1) returns
    # 'evil.exe' which IS a safe filename; the real defense for
    # this case is the relative_to() check end-to-end (below).
    assert _safe_installer_filename(
        "https://x.com/path\\to\\evil.exe") == "Touchless_Installer.exe"
    # End-to-end: download_plan target lives INSIDE download_dir.
    blob = '{"version":"1.2.3","channel":"stable",' \
           '"sha256_installer":"' + "0"*64 + '",' \
           '"installer_url":"https://x.com/..%5C..%5Cevil.exe",' \
           '"signer":"' + "Konstantin Markov" + '"}'
    m = parse_manifest(blob)
    with tempfile.TemporaryDirectory() as tmp:
        plan = download_plan(m, download_dir=tmp)
    # The target path must be INSIDE download_dir.
    from pathlib import Path
    base = Path(tmp).resolve()
    target = Path(plan[0]["target_path"]).resolve()
    target.relative_to(base)  # raises ValueError if escaped
    # And the filename must be the safe default.
    assert target.name == "Touchless_Installer.exe"


def test_safe_installer_filename_accepts_normal_names():
    from hgr.live_api.update_channel import _safe_installer_filename
    assert _safe_installer_filename(
        "https://r.com/Touchless_Installer_v1.0.7.exe") \
        == "Touchless_Installer_v1.0.7.exe"
    assert _safe_installer_filename(
        "https://r.com/abc.exe?token=xyz") == "abc.exe"


# ---- known channels constant -----------------------------------------

def test_known_channels_contains_stable():
    assert "stable" in KNOWN_CHANNELS
    assert "beta" in KNOWN_CHANNELS
