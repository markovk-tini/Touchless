"""r24: a field log must identify the build that produced it.

r21, r22 and r23 all logged "build round 57". A tester reported symptoms
from a stale install and two rounds of debugging went into code that was
never running on his machine.

The cause was not carelessness, it was coupling: BUILD_ROUND was mixed
into the camera-capability cache key, so bumping it silently discarded
everything the r21 probe had learned about the user's webcam -- which
costs an ffmpeg spawn and an antivirus prompt to rebuild. Both rounds
therefore chose not to bump it.

These tests keep the two concerns separated so that can never recur.
"""

import re
from pathlib import Path

import pytest

import hgr

ROOT = Path(__file__).resolve().parents[1]


class TestIdentityIsSeparateFromCacheInvalidation:
    def test_build_round_exists_and_is_an_int(self):
        assert isinstance(hgr.BUILD_ROUND, int)

    def test_there_is_a_dedicated_caps_probe_version(self):
        assert isinstance(hgr.CAMERA_CAPS_PROBE_VERSION, int)

    def test_the_caps_cache_key_does_not_depend_on_build_round(self):
        """The load-bearing assertion. If BUILD_ROUND leaks back into this
        stamp, bumping it wipes the user's learned camera capabilities and
        the next round will again be tempted to freeze it."""
        import inspect

        import hgr.app.integration.noop_engine as NE

        src = inspect.getsource(NE.GestureWorker._caps_stamp)
        # strip comments and the docstring: they legitimately explain the bug
        body = src.split('"""')[-1]
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in body.splitlines())
        assert "BUILD_ROUND" not in body
        assert "CAMERA_CAPS_PROBE_VERSION" in body

    def test_the_caps_stamp_is_stable_across_a_build_round_bump(self):
        import hgr.app.integration.noop_engine as NE

        w = NE.GestureWorker.__new__(NE.GestureWorker)
        before = w._caps_stamp()
        original = hgr.BUILD_ROUND
        try:
            hgr.BUILD_ROUND = original + 1
            assert w._caps_stamp() == before
        finally:
            hgr.BUILD_ROUND = original

    def test_bumping_the_probe_version_DOES_invalidate_the_cache(self):
        """The cache must still be invalidatable when the probe changes."""
        import hgr.app.integration.noop_engine as NE

        w = NE.GestureWorker.__new__(NE.GestureWorker)
        before = w._caps_stamp()
        original = hgr.CAMERA_CAPS_PROBE_VERSION
        try:
            hgr.CAMERA_CAPS_PROBE_VERSION = original + 1
            assert w._caps_stamp() != before
        finally:
            hgr.CAMERA_CAPS_PROBE_VERSION = original


class TestTheStartupBannerIdentifiesTheBinary:
    def _banner_block(self):
        src = (ROOT / "run_app.py").read_text(encoding="utf-8")
        i = src.index("Startup banner")
        return src[i:i + 2000]

    def test_it_logs_the_app_version(self):
        assert "v{_ver}" in self._banner_block()

    def test_it_logs_the_build_round(self):
        assert "build round {_br}" in self._banner_block()

    def test_it_logs_the_exe_size_and_mtime(self):
        """Round alone is not enough -- two builds can share a round. Size
        plus mtime pins the exact binary that wrote the log."""
        block = self._banner_block()
        assert "exe={_exe_id}" in block
        assert "st_size" in block
        assert "st_mtime" in block

    def test_it_prefers_sys_executable_when_frozen(self):
        """In the shipped app the interesting binary is Touchless.exe, not
        the script path."""
        block = self._banner_block()
        assert "sys.executable" in block
        assert 'getattr(sys, "frozen", False)' in block

    def test_the_whole_thing_is_failure_tolerant(self):
        """A stat() that throws must never stop the app from starting."""
        block = self._banner_block()
        assert block.count("except Exception:") >= 3
        assert '_exe_id = "unknown"' in block

    def test_run_app_still_parses(self):
        import ast

        ast.parse((ROOT / "run_app.py").read_text(encoding="utf-8"))
