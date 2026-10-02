"""r25: the tuning throttle must be keyed on the CAPTURE, not the clock.

`_apply_default_capture_tuning` writes the fps request, the MJPG FOURCC
hint and the r49 short shutter. It is throttled to one pass per 10 s
because each `cap.set` on DirectShow costs 500-800 ms of stream
renegotiation, and re-issuing them against a capture that already has
them is pure frozen UI.

Keyed on wall-clock alone it inverted its own purpose. The ffmpeg branch
pre-flights a THROWAWAY capture (stamping the clock), ffmpeg then fails,
and the fallback opens a BRAND NEW OpenCV capture -- whose tuning is then
skipped because something else touched the clock 0.0 s ago. The field log
shows the pair on every Lite/GPU swap:

    [preflight-release] auto-exposure ON via backend=DSHOW (0.75) ok=True
    [default-tuning] SKIP throttled (last apply 0.0s ago ...)

so Lite and GPU ended on an untuned capture at AUTO exposure: no fps
request, no MJPG hint, no short shutter. Strictly worse than Default,
which is exactly the report that Lite and GPU "do nothing".

"Driver state persists" is true of ONE capture. A different object has
never been tuned.
"""

import types

import pytest

import hgr.app.integration.noop_engine as NE

W = NE.GestureWorker


class _Cap:
    """Distinct identity per instance; that is the whole point."""

    def __init__(self):
        self.sets = []

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0

    def set(self, prop, val):
        self.sets.append((prop, val))
        return True

    def getBackendName(self):
        return "DSHOW"


def _worker(monkeypatch):
    monkeypatch.setattr(NE.sys, "platform", "win32")
    w = W.__new__(W)
    w.config = types.SimpleNamespace(
        camera_force_short_shutter=False,
        camera_force_short_shutter_user_chose=True,
        lite_mode=False, gpu_mode=False, low_fps_mode=False,
    )
    w._camera_info = types.SimpleNamespace(
        index=0, display_name="FULL HD 1080P Webcam (Camera 0)")
    w._low_fps_active = False
    w._reached = []
    # Stop after the throttle: everything past it is the behaviour under
    # test's *effect*, not the decision we are pinning here.
    w._wants_ffmpeg_cap = lambda: w._reached.append("proceeded") or False
    return w


def _run(w, cap):
    try:
        w._apply_default_capture_tuning((None, cap))
    except Exception:
        pass


class TestANewCaptureIsAlwaysTuned:
    def test_a_different_capture_inside_the_window_is_not_throttled(
            self, monkeypatch):
        """The regression. Two captures, back to back, no clock advance."""
        w = _worker(monkeypatch)
        _run(w, _Cap())
        n_after_first = len(w._reached)
        _run(w, _Cap())          # brand new object, 0.0 s later
        assert len(w._reached) > n_after_first, (
            "a freshly opened capture inherited the previous capture's "
            "throttle and was left completely untuned"
        )

    def test_the_same_capture_inside_the_window_is_throttled(self,
                                                             monkeypatch):
        """The throttle's real job, preserved."""
        w = _worker(monkeypatch)
        cap = _Cap()
        _run(w, cap)
        n = len(w._reached)
        _run(w, cap)             # same object
        assert len(w._reached) == n, (
            "re-tuning the same capture costs 500-800ms per cap.set for "
            "nothing"
        )

    def test_the_same_capture_is_tuned_again_once_the_window_lapses(
            self, monkeypatch):
        w = _worker(monkeypatch)
        cap = _Cap()
        _run(w, cap)
        n = len(w._reached)
        w._default_tuning_last_at = NE.time.monotonic() - 11.0
        _run(w, cap)
        assert len(w._reached) > n

    def test_identity_not_equality(self, monkeypatch):
        """Two captures that compare equal are still two captures."""
        class _Eq(_Cap):
            def __eq__(self, other):
                return isinstance(other, _Eq)
            __hash__ = None

        w = _worker(monkeypatch)
        _run(w, _Eq())
        n = len(w._reached)
        _run(w, _Eq())
        assert len(w._reached) > n


class TestTheThrottleStateIsRecorded:
    def test_both_the_clock_and_the_capture_are_stamped(self, monkeypatch):
        w = _worker(monkeypatch)
        cap = _Cap()
        _run(w, cap)
        assert getattr(w, "_default_tuning_last_at", 0) > 0
        _ref = getattr(w, "_default_tuning_last_cap_ref", None)
        assert callable(_ref) and _ref() is cap

    def test_a_none_capture_does_not_poison_the_key(self, monkeypatch):
        """`open_result` can be (None, None) on a failed reopen. That must
        not register as 'the capture we tuned'."""
        w = _worker(monkeypatch)
        _run(w, None)
        assert getattr(w, "_default_tuning_last_cap_ref", "unset") is None
        # and a real capture straight after is still tuned
        n = len(w._reached)
        _run(w, _Cap())
        assert len(w._reached) > n
