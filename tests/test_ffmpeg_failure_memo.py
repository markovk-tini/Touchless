"""r20: the ffmpeg-failure memo must never demote a good camera.

The dangerous direction is a false positive: a Kiyo Pro that is busy
for a moment must not be remembered as ffmpeg-incapable, because that
would permanently drop it to the slower OpenCV/YUY2 path. These tests
pin that asymmetry.
"""


import pytest

from hgr.app.camera import ffmpeg_memo as M


DEV = "Razer Kiyo Pro"
OTHER = "FULL HD 1080P Webcam"


@pytest.fixture(autouse=True)
def _clear_kill_switch(monkeypatch):
    monkeypatch.delenv("HGR_FFMPEG_MEMO", raising=False)


# --------------------------------------------------------------- keys

def test_key_is_device_and_resolution_lowercased():
    assert M.memo_key(DEV, 1280, 720) == "razer kiyo pro|1280x720"


def test_key_is_none_without_a_device_name():
    assert M.memo_key("", 1280, 720) is None
    assert M.memo_key("   ", 1280, 720) is None


def test_key_is_none_for_a_nonsense_size():
    assert M.memo_key(DEV, 0, 720) is None
    assert M.memo_key(DEV, "wide", 720) is None


# ------------------------------------------------------- strike rules

def test_one_hard_failure_does_not_skip():
    memo, changed = M.record_failure({}, DEV, 1280, 720, M.KIND_HARD)
    assert changed is True
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is False


def test_two_hard_failures_skip():
    memo, _ = M.record_failure({}, DEV, 1280, 720, M.KIND_HARD)
    memo, _ = M.record_failure(memo, DEV, 1280, 720, M.KIND_HARD)
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is True


@pytest.mark.parametrize("kind", [M.KIND_SILENT, M.KIND_BUSY, "anything-else"])
def test_soft_failures_never_count_however_many(kind):
    memo = {}
    for _ in range(25):
        memo, changed = M.record_failure(memo, DEV, 1280, 720, kind)
        assert changed is False
    assert memo == {}
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is False


def test_a_busy_kiyo_never_gets_demoted_even_after_a_hard_one_off():
    """One hard strike plus any number of handle races stays usable."""
    memo, _ = M.record_failure({}, DEV, 1280, 720, M.KIND_HARD)
    for _ in range(10):
        memo, _ = M.record_failure(memo, DEV, 1280, 720, M.KIND_SILENT)
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is False


def test_strikes_are_clamped_at_the_threshold():
    memo = {}
    for _ in range(9):
        memo, _ = M.record_failure(memo, DEV, 1280, 720, M.KIND_HARD)
    assert memo == {"razer kiyo pro|1280x720": M.DEFAULT_STRIKE_THRESHOLD}


def test_unnamed_device_is_never_recorded():
    memo, changed = M.record_failure({}, "", 1280, 720, M.KIND_HARD)
    assert changed is False and memo == {}


# --------------------------------------------------------- isolation

def test_resolutions_do_not_inherit_each_others_strikes():
    memo = {"razer kiyo pro|1280x720": 2}
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is True
    assert M.should_skip_ffmpeg(memo, DEV, 640, 480) is False


def test_devices_do_not_inherit_each_others_strikes():
    memo = {"razer kiyo pro|1280x720": 2}
    assert M.should_skip_ffmpeg(memo, OTHER, 1280, 720) is False


def test_empty_memo_never_skips():
    assert M.should_skip_ffmpeg({}, DEV, 1280, 720) is False
    assert M.should_skip_ffmpeg(None, DEV, 1280, 720) is False


def test_corrupt_values_are_treated_as_zero():
    assert M.should_skip_ffmpeg({"razer kiyo pro|1280x720": "lots"}, DEV, 1280, 720) is False
    assert M.should_skip_ffmpeg({"razer kiyo pro|1280x720": -5}, DEV, 1280, 720) is False


# ------------------------------------------------------- kill switch

def test_env_kill_switch_disables_the_skip(monkeypatch):
    memo = {"razer kiyo pro|1280x720": 99}
    monkeypatch.setenv("HGR_FFMPEG_MEMO", "0")
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is False
    monkeypatch.setenv("HGR_FFMPEG_MEMO", "1")
    assert M.should_skip_ffmpeg(memo, DEV, 1280, 720) is True


# ------------------------------------------------------ invalidation

def test_clear_device_drops_every_resolution_for_that_device():
    memo = {"razer kiyo pro|1280x720": 2, "razer kiyo pro|640x480": 1,
            "full hd 1080p webcam|1280x720": 2}
    memo, changed = M.clear_device(memo, DEV)
    assert changed is True
    assert memo == {"full hd 1080p webcam|1280x720": 2}


def test_prune_drops_unplugged_devices():
    memo = {"razer kiyo pro|1280x720": 2, "old cam|1280x720": 2}
    memo, changed = M.prune(memo, [DEV])
    assert changed is True
    assert memo == {"razer kiyo pro|1280x720": 2}


def test_prune_with_an_empty_device_list_is_a_no_op():
    """An empty enumeration means the probe failed, not that the
    webcam was unplugged -- dropping the memo there would restore the
    doomed cascade on exactly the machine that needs the memo."""
    memo = {"razer kiyo pro|1280x720": 2}
    out, changed = M.prune(memo, [])
    assert changed is False and out == memo
