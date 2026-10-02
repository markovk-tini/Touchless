"""r20d: a camera left latched dark must get un-stuck in EVERY mode.

The r53 kick that normally does this lives inside the capture-tuning
routine, which returns early in Lite, GPU and Low-FPS mode. So a webcam
latched at a very short manual exposure by a previous session had no exit
path in exactly the modes a performance-conscious user is most likely to
be in. With the Boost toggle off there is no gamma lift either, so the
user simply sees a black preview and no hand tracking.

The write is strictly directional: it only ever hands exposure back to
the driver. These tests pin that it fires when it should, and refuses in
every case where it might fight something else.
"""

import types

import pytest

from hgr.app.integration import noop_engine as NE


GENERIC = "FULL HD 1080P Webcam (Camera 0)"
PREMIUM = "Razer Kiyo Pro (Camera 0)"
DEVRIG = "USB Video Device (Camera 0)"      # what the dev rig actually reports
OLD = "Camera 0 (DirectShow)"


class FakeCap:
    def __init__(self, auto=0.25, exposure=-6.0):
        self._auto, self._exp = auto, exposure

    def get(self, prop):
        import cv2
        if prop == getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", -1):
            return self._auto
        if prop == getattr(cv2, "CAP_PROP_EXPOSURE", -2):
            return self._exp
        return 0.0

    def set(self, *a):
        return True


class Worker:
    """Only what the helper touches."""

    _unstick_inherited_short_shutter = NE.GestureWorker._unstick_inherited_short_shutter

    #: r26 added a COM read of the real Auto/Manual flag, because
    #: OpenCV's CAP_PROP_AUTO_EXPOSURE is unimplemented on DirectShow and
    #: returns -1.0 always. These cases pre-date it and exercise the
    #: CLASSIFIER route, so they report "unknown" and leave the classifier
    #: to decide exactly as before. The COM route has its own file,
    #: tests/test_inherited_shutter_unstick.py — including the guard that
    #: a camera in Auto can never read as Manual, which is what keeps the
    #: r17 Kiyo regression impossible.
    #:
    #: Stubbed rather than left to the real implementation so these stay
    #: hermetic: the real one would enumerate this machine's actual
    #: cameras over COM and the result would depend on the hardware
    #: plugged in.
    _exposure_flag_per_com = staticmethod(lambda *a, **k: None)

    def __init__(self, name=GENERIC, force=False, ss_active=False):
        self.config = types.SimpleNamespace(camera_force_short_shutter=force)
        self._camera_info = types.SimpleNamespace(display_name=name)
        self._short_shutter_active_for_display = ss_active
        self.writes = []

    def _note_driver_write(self, prop="Exposure"):
        self.writes.append(prop)


@pytest.fixture(autouse=True)
def _clean_env_and_capture_write(monkeypatch):
    for k in ("HGR_FORCE_SHORT_SHUTTER", "HGR_FFMPEG_SHORT_SHUTTER"):
        monkeypatch.delenv(k, raising=False)
    calls = []
    monkeypatch.setattr(
        NE, "_dshow_auto_exposure_on",
        lambda cap, log_tag="": (calls.append(log_tag), True)[1],
    )
    return calls


def _run(worker, cap):
    return worker._unstick_inherited_short_shutter(
        (types.SimpleNamespace(display_name=worker._camera_info.display_name), cap)
    )


# ------------------------------------------------------------ it fires

def test_it_fires_on_a_generic_uvc_latched_manual(_clean_env_and_capture_write):
    w = Worker()
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is True
    assert w.writes == ["Exposure"]
    assert _clean_env_and_capture_write == ["[unstick-inherited]"]


def test_it_fires_on_a_short_exposure_even_when_auto_is_unreadable():
    """DirectShow reports -1.0 for the auto flag, which is the field
    case: the camera reads -6 exposure and an unknown auto state."""
    w = Worker()
    assert _run(w, FakeCap(auto=-1.0, exposure=-6.0)) is True


# -------------------------------------------------------- it refuses

def test_it_refuses_on_the_dev_rigs_camera():
    """The Kiyo Pro reports as 'USB Video Device', which the classifier
    scores unknown. Unknown must never trigger a driver write."""
    w = Worker(name=DEVRIG)
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False
    assert w.writes == []


def test_it_refuses_on_a_premium_camera():
    w = Worker(name=PREMIUM)
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False


def test_it_refuses_on_the_old_hardcoded_name():
    w = Worker(name=OLD)
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False


def test_it_refuses_when_boost_is_on():
    """Boost wants the short shutter. Do not fight the user."""
    w = Worker(force=True)
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False
    assert w.writes == []


def test_it_refuses_when_short_shutter_is_active_for_display():
    """A preflight in an ffmpeg mode may legitimately hold it."""
    w = Worker(ss_active=True)
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False


@pytest.mark.parametrize("env", ["HGR_FORCE_SHORT_SHUTTER", "HGR_FFMPEG_SHORT_SHUTTER"])
def test_it_refuses_when_an_override_asks_for_short_shutter(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    w = Worker()
    assert _run(w, FakeCap(auto=0.25, exposure=-6.0)) is False


def test_it_refuses_when_the_camera_is_already_healthy():
    """Exposure -4 on auto is the known-good state. Leave it alone."""
    w = Worker()
    assert _run(w, FakeCap(auto=0.75, exposure=-4.0)) is False
    assert w.writes == []


def test_it_refuses_on_an_ffmpeg_capture():
    class FfmpegMjpegCapture(FakeCap):
        pass

    w = Worker()
    assert _run(w, FfmpegMjpegCapture()) is False


def test_it_refuses_when_there_is_no_capture():
    w = Worker()
    assert w._unstick_inherited_short_shutter((None, None)) is False


def test_a_nonsense_exposure_reading_does_not_trigger_it():
    w = Worker()
    assert _run(w, FakeCap(auto=None, exposure=-99.0)) is False

def test_media_foundation_manual_still_fires():
    """0.75 is DirectShow Auto and 1.0 is Media Foundation MANUAL, and
    both round to 1. Manual has to win that tie or a camera stuck under
    Media Foundation would never be un-stuck."""
    w = Worker()
    assert _run(w, FakeCap(auto=1.0, exposure=-6.0)) is True


def test_directshow_auto_is_left_alone_even_at_a_short_exposure():
    w = Worker()
    assert _run(w, FakeCap(auto=0.75, exposure=-7.0)) is False
    assert w.writes == []

# ------------------------------------------------------ source guards

def _engine_source():
    import io as _io
    import pathlib
    return _io.open(
        pathlib.Path(NE.__file__), encoding="utf-8"
    ).read()


def test_the_unstick_runs_before_the_mode_early_return():
    """It has to be inside the branch that skips the rest of the tuning,
    above that branch's return, or Lite and GPU mode never reach it."""
    src = _engine_source()
    # r22 added "and _on_ffmpeg_cap" to this gate, so match the stable
    # prefix rather than the whole line.
    i = src.index("self._wants_ffmpeg_cap() or self._low_fps_active")
    block = src[i:i + 4000]
    call = block.find("self._unstick_inherited_short_shutter(")
    ret = block.find(chr(10) + "            return")
    assert call != -1, "the un-stick is not in the skip branch"
    assert ret != -1
    assert call < ret, "the un-stick must run before that branch returns"


def test_the_r53_kick_gate_is_unchanged():
    """PERFORMANCE_CHECKPOINT 2.4. This round must not touch it."""
    src = _engine_source()
    assert "_should_kick = _armed or _known_generic_needs_kick" in src
    assert "or _user_chose_kick" not in src


def test_the_unstick_never_writes_a_manual_exposure():
    """The whole safety argument is that this write is directional."""
    import inspect

    body = inspect.getsource(NE.GestureWorker._unstick_inherited_short_shutter)
    assert "_dshow_auto_exposure_on" in body
    for banned in ("CAP_PROP_EXPOSURE,", "set(cv2.CAP_PROP_EXPOSURE"):
        assert banned not in body, banned
