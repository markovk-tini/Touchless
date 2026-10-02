"""r20 source guards for the ffmpeg-memo wiring.

Two ordering rules carry real consequences and are easy to undo by
accident during a later edit, so they are pinned against the source:

1. The memo skip must run BEFORE `_preflight_short_shutter_for_ffmpeg`.
   The preflight writes EXPOSURE=-6 to the driver and is only undone on
   the ffmpeg-failure branch. Skipping after it would latch a camera
   dark for a capture that is never opened.
2. On the initial-open path the skip must also run BEFORE
   `release_capture_serialised`, or a perfectly good OpenCV capture is
   thrown away just to be reopened.
"""

import inspect
import io
import pathlib
import re

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "src" / "hgr" / "app" / "integration" / "noop_engine.py"
SRC = io.open(ENGINE, encoding="utf-8").read()


def _region(start_marker: str, end_marker: str) -> str:
    i = SRC.index(start_marker)
    j = SRC.index(end_marker, i)
    return SRC[i:j]


@pytest.fixture(scope="module")
def initial_open_region():
    return _region("def _upgrade_to_ffmpeg_capture_if_lite", "\n    def _draw_low_fps_badge")


@pytest.fixture(scope="module")
def perf_path_region():
    return _region("def _apply_perf_camera_path", "\n    def set_low_fps_mode")


def _assert_before(region, earlier, later, why):
    i = region.find(earlier)
    j = region.find(later)
    assert i != -1, f"missing {earlier!r}"
    assert j != -1, f"missing {later!r}"
    assert i < j, why


# ------------------------------------------------- initial open path

def test_initial_open_skips_before_the_preflight(initial_open_region):
    _assert_before(
        initial_open_region,
        "_ffmpeg_memo_says_skip",
        "_preflight_short_shutter_for_ffmpeg",
        "the memo skip must precede the EXPOSURE=-6 preflight write",
    )


def test_initial_open_skips_before_releasing_the_working_capture(initial_open_region):
    _assert_before(
        initial_open_region,
        "_ffmpeg_memo_says_skip",
        "release_capture_serialised",
        "the memo skip must precede throwing away the OpenCV capture",
    )


def test_initial_open_records_the_failure_kind(initial_open_region):
    assert "failure_out=_ffmpeg_failure" in initial_open_region
    assert "_ffmpeg_memo_record" in initial_open_region


# --------------------------------------------------- mode-swap path

def test_perf_path_skips_before_the_preflight(perf_path_region):
    _assert_before(
        perf_path_region,
        "_ffmpeg_memo_says_skip",
        "_preflight_short_shutter_for_ffmpeg",
        "the memo skip must precede the EXPOSURE=-6 preflight write",
    )


def test_perf_path_records_the_failure_kind(perf_path_region):
    assert "failure_out=_ffmpeg_failure" in perf_path_region
    assert "_ffmpeg_memo_record" in perf_path_region


# ------------------------------------------- the helpers are bound

def test_the_memo_helpers_are_the_bound_definitions():
    """main_window/noop_engine both carry a dead duplicate-method
    region; a helper defined there would never run. Assert Python
    actually bound the definitions we edited."""
    from hgr.app.integration.noop_engine import GestureWorker

    for name in ("_ffmpeg_memo_says_skip", "_ffmpeg_memo_record"):
        fn = getattr(GestureWorker, name)
        src = inspect.getsource(fn)
        assert "ffmpeg_memo" in src, f"{name} bound to an unexpected definition"


def test_only_one_definition_of_each_helper():
    for name in ("_ffmpeg_memo_says_skip", "_ffmpeg_memo_record"):
        n = len(re.findall(rf"^    def {name}\(", SRC, re.M))
        assert n == 1, f"{name} defined {n} times"

def test_perf_path_skips_before_releasing_the_working_capture(perf_path_region):
    """The decision has to come before the capture release, the 600 ms
    settle and the reopen. Checking after them still costs seconds of
    frozen UI on every mode swap for a camera whose answer is known."""
    _assert_before(
        perf_path_region,
        "_ffmpeg_memo_says_skip",
        "release_capture_serialised",
        "the memo skip must precede releasing the working capture",
    )


def test_both_skip_paths_release_any_preflight_we_own(initial_open_region, perf_path_region):
    """Skipping must not strand an exposure latch this session applied.
    The helper self-gates, so it is a no-op when we latched nothing."""
    for region, name in ((initial_open_region, "initial open"),
                         (perf_path_region, "mode swap")):
        i = region.find("_ffmpeg_memo_says_skip")
        assert i != -1, name
        assert "_release_ffmpeg_preflight" in region[i:i + 1600], name


def test_the_memo_decision_lives_in_exactly_one_place_per_path(perf_path_region):
    assert perf_path_region.count("_ffmpeg_memo_says_skip") == 1
