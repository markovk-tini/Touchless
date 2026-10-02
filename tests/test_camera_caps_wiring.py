"""r21: the capability lookup has to reach both ffmpeg open paths, and
must never turn "we don't know yet" into "unsupported".
"""

import inspect
import io
import pathlib
import re

import pytest

from hgr.app.integration import noop_engine as NE

SRC = io.open(pathlib.Path(NE.__file__), encoding="utf-8").read()


def _region(start, end):
    i = SRC.index(start)
    j = SRC.index(end, i)
    return SRC[i:j]


@pytest.fixture(scope="module")
def perf_path():
    return _region("def _apply_perf_camera_path", "\n    def set_low_fps_mode")


@pytest.fixture(scope="module")
def initial_open():
    return _region("def _upgrade_to_ffmpeg_capture_if_lite", "\n    def _draw_low_fps_badge")


@pytest.mark.parametrize("region_name", ["perf_path", "initial_open"])
def test_both_paths_consult_the_camera_before_attempting(region_name, request):
    region = request.getfixturevalue(region_name)
    learn = region.find("_camera_caps_learn")
    plan = region.find("_camera_caps_plan")
    attempt = region.find("open_ffmpeg_cap_with_fps_fallback")
    assert learn != -1 and plan != -1, "capability lookup missing"
    assert learn < attempt and plan < attempt, (
        "the camera must be asked before the fast path is attempted"
    )


@pytest.mark.parametrize("region_name", ["perf_path", "initial_open"])
def test_both_paths_open_the_advertised_frame_rate(region_name, request):
    region = request.getfixturevalue(region_name)
    assert "fps_candidates=_caps_fps" in region, (
        "the open must use the rate the camera advertises, not a fixed guess"
    )


@pytest.mark.parametrize("region_name", ["perf_path", "initial_open"])
def test_a_camera_with_no_compressed_pin_skips_without_attempting(region_name, request):
    region = request.getfixturevalue(region_name)
    i = region.find("_known and _plan is None")
    assert i != -1, "no early-out for a camera that advertises nothing usable"
    after = region[i:i + 900]
    attempt = region.find("open_ffmpeg_cap_with_fps_fallback")
    assert region.find("return", i) < attempt, "the skip must return before attempting"
    assert "_release_ffmpeg_preflight" in after, (
        "skipping must not strand an exposure latch"
    )


def test_unknown_capabilities_fall_back_to_the_old_behaviour():
    """None from the probe means 'not learned', never 'unsupported'."""
    body = inspect.getsource(NE.GestureWorker._camera_caps_plan)
    assert "return None, False" in body
    src = inspect.getsource(NE.GestureWorker._camera_caps_get)
    assert "return None" in src


def test_the_cache_is_stamped_so_a_new_build_relearns():
    body = inspect.getsource(NE.GestureWorker._camera_caps_get)
    assert "_caps_stamp" in body
    stamp = inspect.getsource(NE.GestureWorker._caps_stamp)
    assert "BUILD_ROUND" in stamp and "__version__" in stamp


def test_the_probe_is_never_run_while_a_capture_is_open(perf_path, initial_open):
    """It talks to DirectShow directly; a live capture would block it."""
    for region in (perf_path, initial_open):
        learn = region.find("_camera_caps_learn")
        release = region.find("release_capture_serialised")
        assert release != -1 and release < learn, (
            "the capture must be released before the camera is questioned"
        )


def test_each_helper_has_one_definition():
    for name in ("_camera_caps_learn", "_camera_caps_plan", "_camera_caps_get",
                 "_camera_caps_store", "_caps_stamp"):
        n = len(re.findall(rf"^    def {name}\(", SRC, re.M))
        assert n == 1, f"{name} defined {n} times"

def test_the_probe_settles_before_handing_the_camera_over():
    """The probe is itself a DirectShow open. Without a settle, the
    capture attempt right after it races the driver teardown and hangs
    silently with empty stderr -- observed on the dev rig."""
    body = inspect.getsource(NE.GestureWorker._camera_caps_learn)
    assert "time.sleep" in body
    i = body.find("_camera_caps_store")
    j = body.find("time.sleep")
    assert i < j, "the settle must come after the probe, not before"
