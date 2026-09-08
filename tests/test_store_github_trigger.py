"""Store builds must prompt from GitHub /releases/latest (v1.1.9.1).

Until 1.1.9, StoreUpdateChecker asked the Store listing first, so a
GitHub-only publish never showed a popup. Payload is still the GitHub
app-zip, not Partner Center's full installer.

These tests mock HTTP so they do not hit GitHub or the Store.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.request import Request


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args) -> None:
        return None


def _url_of(req) -> str:
    if isinstance(req, Request):
        return str(req.full_url)
    return str(req)


class StoreGitHubTriggerTests(unittest.TestCase):
    def _run_checker(self, urlopen_side_effect):
        from PySide6.QtWidgets import QApplication
        from hgr.app.updater.store_updater import StoreUpdateChecker

        QApplication.instance() or QApplication([])
        seen = []
        checker = StoreUpdateChecker()
        checker.update_available.connect(lambda info: seen.append(("update", info)))
        checker.no_update.connect(lambda: seen.append(("none", None)))
        checker.check_failed.connect(lambda reason: seen.append(("fail", reason)))
        with patch(
            "hgr.app.updater.store_updater.urllib.request.urlopen",
            side_effect=urlopen_side_effect,
        ), patch(
            "hgr.app.updater.updater.Updater.is_install_dir_writable",
            return_value=True,
        ):
            checker.run()
        return seen

    def test_github_newer_offers_zip_even_if_store_listing_is_old(self) -> None:
        """The 1.1.9.1 trigger: GitHub is 1.1.9.2, Store listing is still
        1.1.9. Store users must still get update_available with the zip."""

        def fake_urlopen(req, timeout=None):
            url = _url_of(req)
            if "api.github.com" in url and "/releases/latest" in url:
                return _FakeHTTPResponse({
                    "tag_name": "v1.1.9.2",
                    "html_url": "https://github.com/markovk-tini/Touchless/releases/tag/v1.1.9.2",
                    "body": "notes",
                    "assets": [],
                })
            if "packageManifests" in url:
                return _FakeHTTPResponse({
                    "Data": {
                        "Versions": [{
                            "PackageVersion": "1.1.9",
                            "Installers": [{
                                "Architecture": "x64",
                                "InstallerUrl": "https://example.com/full.exe",
                            }],
                        }],
                    },
                })
            if "api.github.com" in url and "/releases/tags/" in url:
                return _FakeHTTPResponse({
                    "body": "<!-- app-update-zip-sha256: " + ("a" * 64) + " -->",
                    "assets": [{
                        "name": "Touchless_App_Update_1.1.9.2.zip",
                        "browser_download_url": "https://github.com/example/Touchless_App_Update_1.1.9.2.zip",
                        "size": 140000000,
                    }],
                })
            raise AssertionError(f"unexpected url {url}")

        with patch("hgr.app.updater.store_updater.RUNNING_VERSION", "1.1.9"):
            seen = self._run_checker(fake_urlopen)
        self.assertEqual(len(seen), 1, seen)
        kind, info = seen[0]
        self.assertEqual(kind, "update")
        self.assertEqual(info.version, "1.1.9.2")
        self.assertEqual(info.update_kind, "app-zip")
        self.assertIn("Touchless_App_Update_1.1.9.2.zip", info.download_url)

    def test_github_same_version_is_no_update(self) -> None:
        def fake_urlopen(req, timeout=None):
            url = _url_of(req)
            if "api.github.com" in url and "/releases/latest" in url:
                return _FakeHTTPResponse({
                    "tag_name": "v1.1.9",
                    "html_url": "https://github.com/example/tag",
                    "body": "",
                    "assets": [],
                })
            if "packageManifests" in url:
                return _FakeHTTPResponse({"Data": {"Versions": []}})
            raise AssertionError(f"unexpected url {url}")

        with patch("hgr.app.updater.store_updater.RUNNING_VERSION", "1.1.9"):
            seen = self._run_checker(fake_urlopen)
        self.assertEqual(seen, [("none", None)])


if __name__ == "__main__":
    unittest.main()
