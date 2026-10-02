"""r24 hotfix: BOTH branches of `_apply_perf_camera_path` must execute.

r24 routed all four camera re-opens in `_apply_perf_camera_path` through
`_open_index_taking_the_light_window(index, device_name)`. Three of those
sites live inside `if want_ffmpeg:`, where `device_name` is bound. The
fourth is in the `else:` body, where it is not -- so Python treated it as
an unbound local and the whole ffmpeg -> OpenCV restore raised
`UnboundLocalError`, after `self._cap` had already been set to None and the
old capture released. Net effect: turning Lite or GPU back OFF on a machine
where ffmpeg actually engaged left a dead camera.

It shipped because the guard test asserted on the function's SOURCE TEXT
(`"open_camera_by_index(" not in body`) instead of running it. A text
assertion cannot see a name-binding error on a branch it never takes.

So these tests CALL the function, once down each branch, with the module's
Windows-only and DirectShow dependencies stubbed. That is the only shape
that would have caught this.
"""

import sys
import types

import pytest

import hgr.app.integration.noop_engine as NE

W = NE.GestureWorker


class _FfmpegCap:
    """Type NAME is what the branch test keys on (`"FfmpegMjpegCapture" in
    type(self._cap).__name__`), so the name has to match exactly."""


_FfmpegCap.__name__ = "FfmpegMjpegCapture"


class _OpenCvCap:
    pass


def _worker(cap, *, lite=False, gpu=False):
    w = W.__new__(W)
    w._running = True
    w._cap = cap
    w._camera_info = types.SimpleNamespace(
        index=0, display_name="FULL HD 1080P Webcam (Camera 0)"
    )
    w.config = types.SimpleNamespace(
        lite_mode=lite, gpu_mode=gpu, camera_scan_limit=4,
        camera_force_short_shutter=True,
        camera_force_short_shutter_user_chose=True,
        preferred_camera_index=0,
    )
    w._ffmpeg_preflight_device = None
    w._release_ffmpeg_preflight = lambda idx: None
    w._ffmpeg_memo_says_skip = lambda *a, **k: False
    w._apply_default_capture_tuning = lambda res: None
    w.calls = []
    return w


@pytest.fixture
def win32(monkeypatch):
    """The function early-returns on non-Windows; the bug is Windows-only."""
    monkeypatch.setattr(NE.sys, "platform", "win32")
    monkeypatch.setattr(NE, "release_capture_serialised", lambda cap: None)
    monkeypatch.setattr(NE.time, "sleep", lambda s: None)


class TestTheOpenCvRestoreBranchActuallyRuns:
    """`want_ffmpeg=False` while currently on ffmpeg -- the else body."""

    def test_it_does_not_raise(self, win32):
        w = _worker(_FfmpegCap())
        w._open_index_taking_the_light_window = (
            lambda idx, name: (w.calls.append((idx, name)),
                               (object(), _OpenCvCap()))[1]
        )
        w._apply_perf_camera_path(want_ffmpeg=False)
        assert w.calls, "the else branch never reached the re-open"

    def test_it_passes_a_usable_device_name(self, win32):
        """An empty name silently disables the light lift; the Qt display
        name is in hand for free and `_procamp_for_device` strips the
        ' (Camera N)' suffix, so it resolves to the same device."""
        w = _worker(_FfmpegCap())
        w._open_index_taking_the_light_window = (
            lambda idx, name: (w.calls.append((idx, name)),
                               (object(), _OpenCvCap()))[1]
        )
        w._apply_perf_camera_path(want_ffmpeg=False)
        _idx, name = w.calls[0]
        assert "FULL HD 1080P Webcam" in name, name

    def test_the_camera_is_left_attached(self, win32):
        """`self._cap = None` happens BEFORE the branch. If the branch
        raises, the engine is left with no capture and no recovery."""
        w = _worker(_FfmpegCap())
        fresh = _OpenCvCap()
        w._open_index_taking_the_light_window = (
            lambda idx, name: (object(), fresh)
        )
        w._apply_perf_camera_path(want_ffmpeg=False)
        assert w._cap is fresh

    def test_a_failed_reopen_is_survived(self, win32):
        w = _worker(_FfmpegCap())
        w._open_index_taking_the_light_window = lambda idx, name: (None, None)
        w._apply_perf_camera_path(want_ffmpeg=False)   # must not raise


class TestTheFfmpegBranchStillResolvesTheRealName:
    """The DirectShow name must keep being resolved INSIDE this branch.
    Hoisting it would put an `ffmpeg -list_devices` subprocess (6 s
    timeout) on the GUI thread for the OpenCV path too -- checkpoint 2.11
    records r20 cutting ffmpeg spawns per launch from 10 to 1."""

    def test_the_authoritative_name_overrides_the_qt_name(self, win32,
                                                          monkeypatch):
        from hgr.app.camera import ffmpeg_capture as FC
        monkeypatch.setattr(FC, "resolve_dshow_device_for_index",
                            lambda idx, qt_name_hint="": "FULL HD 1080P Webcam")
        w = _worker(_OpenCvCap(), lite=True)
        w._open_index_taking_the_light_window = (
            lambda idx, name: (w.calls.append((idx, name)), (None, None))[1]
        )
        monkeypatch.setattr(
            NE, "open_ffmpeg_cap_with_fps_fallback",
            lambda *a, **k: None, raising=False)
        try:
            w._apply_perf_camera_path(want_ffmpeg=True)
        except Exception:
            pass
        if w.calls:
            assert w.calls[0][1] == "FULL HD 1080P Webcam", (
                "the ffmpeg branch must use the DirectShow name, not the "
                "Qt name with its ' (Camera N)' suffix"
            )

    def test_the_resolve_is_not_hoisted_out_of_the_branch(self):
        """Guards the fix itself against the 'obvious' repair."""
        import ast
        import inspect
        import textwrap

        src, start = inspect.getsourcelines(W._apply_perf_camera_path)
        fn = ast.parse(textwrap.dedent("".join(src))).body[0]
        split = next(
            n for n in ast.walk(fn)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "want_ffmpeg" and n.orelse
        )
        lo, hi = split.orelse[0].lineno, split.orelse[-1].end_lineno
        body = "\n".join(
            "".join(src).splitlines()[lo - 1:hi]
        )
        assert "resolve_dshow_device_for_index" not in body, (
            "the else branch must not enumerate DirectShow devices -- that "
            "spawns ffmpeg on the GUI thread during the mode-swap freeze"
        )


class TestEveryNameIsBoundOnEveryPath:
    """The general form of the bug: a name bound in one branch and used in
    another. Catches the next one mechanically."""

    def test_no_possibly_unbound_locals(self):
        import ast
        import inspect
        import textwrap

        src, start = inspect.getsourcelines(W._apply_perf_camera_path)
        fn = ast.parse(textwrap.dedent("".join(src))).body[0]
        split = next(
            n for n in ast.walk(fn)
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "want_ffmpeg" and n.orelse
        )

        def first(nodes, ctx):
            """{name: earliest line with that context} within `nodes`."""
            out = {}
            for node in nodes:
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ctx):
                        if n.id not in out or n.lineno < out[n.id]:
                            out[n.id] = n.lineno
            return out

        import builtins

        before = set(first(
            [n for n in fn.body if n is not split], ast.Store))
        if_writes = set(first(split.body, ast.Store))
        else_reads = first(split.orelse, ast.Load)
        else_writes = first(split.orelse, ast.Store)

        leaked = []
        for name, read_at in else_reads.items():
            if name not in if_writes or name in before:
                continue
            if hasattr(builtins, name):
                continue
            # Bound in the else branch itself BEFORE the read -> fine.
            # `recovered` is the legitimate case: both branches assign it.
            wrote_at = else_writes.get(name)
            if wrote_at is not None and wrote_at <= read_at:
                continue
            leaked.append(name)

        assert not leaked, (
            f"read in the else branch but only bound in the if branch: "
            f"{sorted(leaked)}"
        )
