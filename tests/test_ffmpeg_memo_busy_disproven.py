"""r22: a "busy" verdict that the OpenCV fallback disproves must count.

Field log, 2026-09-25, the user's father's PC. Every Lite/GPU mode entry
did this, forever:

    ffmpeg_capture] 30 fps open hung silently
    ffmpeg_capture] 30 fps retry also silent-hung - camera is being held
    perf-camera] ffmpeg cap failed, falling back to OpenCV
    ffmpeg-memo] failed as 'busy' - not a strike, memo unchanged

The OpenCV fallback then opened that same camera immediately, so nothing
was holding it. Because "busy" never struck, the memo never learned, and
every later mode switch paid the same ~10 s freeze and spawned fresh
ffmpeg processes for the antivirus to prompt about.
"""

import inspect
import io
import pathlib
import re

import pytest

from hgr.app.camera import ffmpeg_memo as M
from hgr.app.integration import noop_engine as NE

CAM = "FULL HD 1080P Webcam"
SRC = io.open(pathlib.Path(NE.__file__), encoding="utf-8").read()


# ------------------------------------------------------- the rule

@pytest.mark.parametrize("kind", [M.KIND_BUSY, M.KIND_SILENT])
def test_a_disproven_busy_verdict_counts_in_full_on_the_first_try(kind):
    """r24 strengthens r22: a disproven verdict goes straight to the
    threshold instead of earning one strike at a time.

    r22 made it count. But at one strike per occurrence the user still
    had to sit through a SECOND 20-26 s frozen mode switch, and a second
    antivirus prompt, to learn something the first one already proved:
    OpenCV opened this camera seconds later, so nothing was holding it.
    """
    memo, changed = M.record_failure({}, CAM, 640, 480, kind,
                                     device_confirmed_free=True)
    assert changed is True
    assert memo[M.memo_key(CAM, 640, 480)] == M.DEFAULT_STRIKE_THRESHOLD
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is True


@pytest.mark.parametrize("kind", [M.KIND_BUSY, M.KIND_SILENT])
def test_an_unchallenged_busy_verdict_still_does_not_count(kind):
    """The camera really could be held by another app. Unchanged rule."""
    memo, changed = M.record_failure({}, CAM, 640, 480, kind)
    assert changed is False
    assert memo == {}


def test_a_hard_failure_counts_either_way():
    for free in (True, False):
        _, changed = M.record_failure({}, CAM, 640, 480, M.KIND_HARD,
                                      device_confirmed_free=free)
        assert changed is True


def test_the_field_sequence_converges_after_exactly_one_switch():
    """r24: the doomed cascade is paid once, ever -- not once per
    session until two strikes accumulate."""
    memo = {}
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is False
    memo, _ = M.record_failure(memo, CAM, 640, 480, M.KIND_BUSY,
                               device_confirmed_free=True)
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is True


def test_an_ordinary_hard_failure_still_needs_two_strikes():
    """Only a DISPROVEN verdict is conclusive on sight. A plain format
    rejection could still be a one-off, so the r20 rule stands."""
    memo = {}
    memo, _ = M.record_failure(memo, CAM, 640, 480, M.KIND_HARD)
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is False
    memo, _ = M.record_failure(memo, CAM, 640, 480, M.KIND_HARD)
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is True


def test_a_genuinely_busy_premium_camera_is_never_demoted():
    """The property the two-strike rule exists to protect. When Synapse
    really is holding a Kiyo Pro the OpenCV fallback fails too, so
    device_confirmed_free is False and nothing is ever recorded."""
    memo = {}
    for _ in range(10):
        memo, changed = M.record_failure(memo, "Razer Kiyo Pro", 640, 480,
                                         M.KIND_BUSY, device_confirmed_free=False)
        assert changed is False
    assert memo == {}
    assert M.should_skip_ffmpeg(memo, "Razer Kiyo Pro", 640, 480) is False


def test_it_still_clamps_at_the_threshold():
    memo = {}
    for _ in range(6):
        memo, _ = M.record_failure(memo, CAM, 640, 480, M.KIND_BUSY,
                                   device_confirmed_free=True)
    assert memo[M.memo_key(CAM, 640, 480)] == M.DEFAULT_STRIKE_THRESHOLD


def test_the_kill_switch_and_clear_still_work(monkeypatch):
    memo = {M.memo_key(CAM, 640, 480): 9}
    monkeypatch.setenv(M._ENV_KILL_SWITCH, "0")
    assert M.should_skip_ffmpeg(memo, CAM, 640, 480) is False
    monkeypatch.delenv(M._ENV_KILL_SWITCH, raising=False)
    assert M.should_skip_ffmpeg(M.clear_device(memo, CAM), CAM, 640, 480) is False


# ---------------------------------------------------- the wiring

def _region(start, end):
    i = SRC.index(start)
    return SRC[i:SRC.index(end, i)]


@pytest.mark.parametrize("start,end", [
    ("def _apply_perf_camera_path", "\n    def set_low_fps_mode"),
    ("def _upgrade_to_ffmpeg_capture_if_lite", "\n    def _draw_low_fps_badge"),
])
def test_the_verdict_is_recorded_after_the_fallback_not_before(start, end):
    """Recording before the fallback cannot know if the camera was free."""
    region = _region(start, end)
    attempt = region.find("open_ffmpeg_cap_with_fps_fallback")
    # r24: `_apply_perf_camera_path` routes its re-opens through
    # `_open_index_taking_the_light_window`, which takes the pre-open
    # driver-light window and then calls `open_camera_by_index`. Either
    # spelling is the OpenCV fallback for the purposes of this ordering
    # rule.
    reopen = min(
        (i for i in (region.find("open_camera_by_index", attempt),
                     region.find("_open_index_taking_the_light_window",
                                 attempt))
         if i != -1),
        default=-1,
    )
    assert reopen != -1, "no OpenCV fallback after the ffmpeg attempt"
    # Only actual RECORDING counts; site B defines a closure up front and
    # calls it at each exit, so the definition may precede the reopen.
    # A path that defines the _record_memo closure records only where it
    # CALLS it; the self._ffmpeg_memo_record inside that closure body is a
    # definition, not an execution, and legitimately sits earlier.
    pattern = (r"_record_memo\((?:True|False)\)" if "def _record_memo" in region
               else r"self\._ffmpeg_memo_record\(")
    calls = [m.start() for m in re.finditer(pattern, region)]
    assert calls, "nothing records the memo in this path"
    early = [c for c in calls if attempt < c < reopen]
    assert not early, (
        "the memo is recorded before the fallback, which is what decides "
        "whether the camera was ever actually busy"
    )


def test_every_exit_of_the_retry_loop_records_exactly_once():
    region = _region("def _upgrade_to_ffmpeg_capture_if_lite",
                     "\n    def _draw_low_fps_badge")
    assert region.count("_record_memo(True)") == 2, "success exits"
    assert region.count("_record_memo(False)") == 1, "give-up exit"
    assert "_memo_written" in region, "no guard against double-recording"


def test_the_helper_passes_the_flag_through():
    body = inspect.getsource(NE.GestureWorker._ffmpeg_memo_record)
    assert "device_confirmed_free" in body
    sig = inspect.signature(NE.GestureWorker._ffmpeg_memo_record)
    assert sig.parameters["device_confirmed_free"].default is False


def test_the_preflight_settles_longer_than_the_race_it_lost():
    """0.40 s was not enough on the field rig: the probe enumerated the
    camera, the pre-flight ran, and ffmpeg hung 0.4 s later."""
    body = inspect.getsource(NE.GestureWorker._preflight_short_shutter_for_ffmpeg)
    sleeps = [float(m) for m in re.findall(r"time\.sleep\(([\d.]+)\)", body)]
    assert sleeps, "the pre-flight must settle before handing over the camera"
    assert max(sleeps) >= 0.60
