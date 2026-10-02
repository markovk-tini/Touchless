"""r26b: `_unstick_inherited_short_shutter` must be able to FIRE.

It has existed since r20d for "a previous session left this camera on a
very short manual exposure, so the preview is black and there is no way
out". It had never once been able to run. Its only call site was:

    if (self._wants_ffmpeg_cap() or self._low_fps_active) and _on_ffmpeg_cap:
        _unstuck = self._unstick_inherited_short_shutter(open_result)

`_on_ffmpeg_cap` is True exactly when the capture IS an
`FfmpegMjpegCapture`, and the helper's second guard is
`if "FfmpegMjpegCapture" in type(cap).__name__: return False`. The sole
caller guaranteed the one type the callee refuses. Unreachable on every
path, in every mode -- while the reference rig sat at `-6/Manual` across
three restarts with a dark live view.

Every existing test for it called the helper DIRECTLY, so all of them
passed against code that could never run in the app. That is the gap
these tests close: they assert on reachability, not on behaviour.

Caught only because the frozen r26 smoke run still showed
`Exposure=-6/Manual` at both START and STOP with `session changes: none`.
"""

import ast
import inspect
import textwrap
import types

import pytest

import hgr.app.integration.noop_engine as NE

W = NE.GestureWorker


class _Ffmpeg:
    def get(self, p):
        return -6.0

    def set(self, p, v):
        return True


_Ffmpeg.__name__ = "FfmpegMjpegCapture"


class _OpenCv:
    def __init__(self, exposure=-6.0):
        self._exp = exposure
        self.sets = []

    def isOpened(self):
        return True

    def get(self, p):
        import cv2
        if p == getattr(cv2, "CAP_PROP_EXPOSURE", -991):
            return self._exp
        return -1.0          # DSHOW: AUTO_EXPOSURE is always -1.0

    def set(self, p, v):
        self.sets.append((p, v))
        return True

    def release(self):
        pass


def _worker(monkeypatch, *, com_manual=True, boost=False):
    monkeypatch.setattr(NE.sys, "platform", "win32")
    w = W.__new__(W)
    w.config = types.SimpleNamespace(camera_force_short_shutter=boost)
    w._camera_info = types.SimpleNamespace(
        display_name="USB Video Device (Camera 0)")
    w._short_shutter_active_for_display = False
    w._note_driver_write = lambda p="Exposure": None
    w._exposure_flag_per_com = lambda name: com_manual
    return w


def _code_of(fn) -> str:
    """Source of `fn` with its docstring and comments removed.

    A text guard over a function that DOCUMENTS the pattern it forbids
    will always match its own explanation. Strip the prose first, or the
    assertion is about the comment rather than the code.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        node.body.pop(0)
    return ast.unparse(node)


class TestItIsActuallyReachable:
    def test_the_old_call_site_could_never_fire(self, monkeypatch):
        """Documents the bug so nobody reinstates that shape."""
        w = _worker(monkeypatch)
        out = w._unstick_inherited_short_shutter(
            (types.SimpleNamespace(display_name="USB Video Device"), _Ffmpeg()))
        assert out is False, (
            "the only historical caller guaranteed an FfmpegMjpegCapture, "
            "which this function refuses by design"
        )

    def test_there_is_now_a_pre_open_caller(self):
        assert hasattr(W, "_maybe_unstick_before_open")

    def test_the_pre_open_caller_is_wired_into_open_camera(self):
        src = inspect.getsource(W._open_camera)
        body = "\n".join(
            ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert "_maybe_unstick_before_open()" in body

    def test_it_runs_before_the_light_lift(self):
        """Un-sticking brightens the frame, so the lift must measure the
        corrected picture -- otherwise it lifts gamma to compensate for a
        short shutter we were about to remove anyway."""
        src = inspect.getsource(W._open_camera)
        body = "\n".join(
            ln for ln in src.splitlines() if not ln.strip().startswith("#"))
        assert (body.index("_maybe_unstick_before_open()")
                < body.index("_maybe_lift_light_before_open()"))

    def test_the_pre_open_caller_passes_a_real_capture_not_an_ffmpeg_one(self):
        """The whole bug was the caller handing over the refused type."""
        body = _code_of(W._maybe_unstick_before_open)
        assert "cv2.VideoCapture" in body
        assert "FfmpegMjpegCapture" not in body


class TestItHonoursCheckpoint39:
    """A driver write belongs on a throwaway cap in the pre-open window,
    never on the live engine cap or a ThreadedCvCapture with a running
    reader."""

    def test_it_opens_its_own_throwaway_under_the_graph_lock(self):
        src = inspect.getsource(W._maybe_unstick_before_open)
        assert "DSHOW_GRAPH_LOCK" in src
        assert "cv2.CAP_DSHOW" in src

    def test_it_releases_that_throwaway(self):
        src = inspect.getsource(W._maybe_unstick_before_open)
        assert "release()" in src
        assert "finally:" in src

    def test_it_never_reaches_for_self_cap(self):
        body = _code_of(W._maybe_unstick_before_open)
        assert "self._cap" not in body


class TestTheHelperStillSelfGates:
    def test_it_fires_on_a_com_confirmed_manual_latch(self, monkeypatch):
        w = _worker(monkeypatch, com_manual=True)
        calls = []
        monkeypatch.setattr(NE, "_dshow_auto_exposure_on",
                            lambda cap, log_tag="": (calls.append(log_tag), True)[1])
        out = w._unstick_inherited_short_shutter(
            (types.SimpleNamespace(display_name="USB Video Device"),
             _OpenCv(exposure=-6.0)))
        assert out is True and calls

    def test_it_refuses_when_boost_is_on(self, monkeypatch):
        """Somebody is asking for a short shutter; do not fight them."""
        w = _worker(monkeypatch, com_manual=True, boost=True)
        out = w._unstick_inherited_short_shutter(
            (types.SimpleNamespace(display_name="USB Video Device"),
             _OpenCv(exposure=-6.0)))
        assert out is False

    def test_it_refuses_when_com_says_auto_and_the_camera_is_unknown(
            self, monkeypatch):
        """The r17 shape: a Kiyo in AUTO reads -4.0 and trips the bare
        `< -2.5` threshold. The flag is what saves it."""
        w = _worker(monkeypatch, com_manual=False)
        w._camera_info = types.SimpleNamespace(
            display_name="Razer Kiyo Pro (Camera 0)")
        out = w._unstick_inherited_short_shutter(
            (types.SimpleNamespace(display_name="Razer Kiyo Pro"),
             _OpenCv(exposure=-4.0)))
        assert out is False


class TestNoOtherDeadCallSites:
    """Generic: a guard that rejects a type its only caller guarantees is
    a whole feature that cannot run. Cheap to check mechanically."""

    def test_the_ffmpeg_gated_site_is_no_longer_the_only_one(self):
        src = inspect.getsource(NE)
        n = src.count("_unstick_inherited_short_shutter")
        assert n >= 3, (
            f"expected a definition plus at least two call sites, found "
            f"{n} references"
        )
