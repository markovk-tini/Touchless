"""r20: the DirectShow enumeration cache must cut ffmpeg spawns without
ever handing a caller a stale device list.

Each enumeration spawns an ffmpeg.exe, and on machines running consumer
antivirus every spawn is a fresh scan the user sees as a popup. But the
list is also used to map a camera index positionally onto a device name,
so a stale list after a replug would open the wrong camera. Hence: cache
by default, explicit freshness where correctness depends on it, and
invalidation on any device change.
"""

import io
import pathlib

import pytest

from hgr.app.camera import ffmpeg_capture as F


@pytest.fixture
def fake_enum(monkeypatch):
    """Replace the subprocess with a counter."""
    calls = {"n": 0, "devices": ["Cam A"]}

    def _fake_run(*a, **kw):
        calls["n"] += 1

        class R:
            stderr = "".join(f'[dshow @ 0x0] "{d}" (video)\n' for d in calls["devices"])
            stdout = ""
        return R()

    monkeypatch.setattr(F.subprocess, "run", _fake_run)
    monkeypatch.setattr(F, "locate_ffmpeg", lambda: "ffmpeg.exe")
    monkeypatch.setattr(F.sys, "platform", "win32")
    F.invalidate_dshow_device_cache()
    yield calls
    F.invalidate_dshow_device_cache()


def test_repeated_calls_spawn_once(fake_enum):
    for _ in range(6):
        assert F.list_dshow_video_devices() == ["Cam A"]
    assert fake_enum["n"] == 1


def test_use_cache_false_always_spawns(fake_enum):
    F.list_dshow_video_devices()
    F.list_dshow_video_devices(use_cache=False)
    F.list_dshow_video_devices(use_cache=False)
    assert fake_enum["n"] == 3


def test_invalidation_forces_a_fresh_read(fake_enum):
    assert F.list_dshow_video_devices() == ["Cam A"]
    fake_enum["devices"] = ["Cam A", "Cam B"]
    assert F.list_dshow_video_devices() == ["Cam A"]      # still cached
    F.invalidate_dshow_device_cache()
    assert F.list_dshow_video_devices() == ["Cam A", "Cam B"]
    assert fake_enum["n"] == 2


def test_an_empty_result_is_never_cached(fake_enum):
    """An empty enumeration is usually a transient failure. Caching it
    would strand the app with no camera for the whole TTL."""
    fake_enum["devices"] = []
    assert F.list_dshow_video_devices() == []
    assert F.list_dshow_video_devices() == []
    assert fake_enum["n"] == 2


def test_a_real_spawn_is_logged_so_it_can_be_counted(fake_enum, monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(F.sys, "stderr", buf)
    F.list_dshow_video_devices()
    F.list_dshow_video_devices()
    assert buf.getvalue().count("spawned ffmpeg") == 1


def test_the_ttl_is_a_backstop_not_the_mechanism():
    assert F._DEVICE_LIST_TTL_S >= 30.0


# --------------------------------------------------- wiring source guards

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hgr"


def _read(rel):
    return io.open(SRC_ROOT / rel, encoding="utf-8").read()


def test_the_positional_fallback_never_uses_the_cache():
    """camera_utils maps a camera index positionally into this list."""
    src = _read("app/camera/camera_utils.py")
    assert "list_dshow_video_devices()" not in src, (
        "every camera_utils call must pass use_cache explicitly"
    )
    assert src.count("list_dshow_video_devices(use_cache=False)") == 2


def test_the_wait_for_camera_poll_loop_never_uses_the_cache():
    src = _read("app/ui/main_window.py")
    i = src.find("ffmpeg_devices = list_dshow_video_devices")
    assert i != -1
    assert "use_cache=False" in src[i:i + 80]


def test_device_change_and_camera_recovery_both_invalidate():
    assert "invalidate_dshow_device_cache" in _read("app/ui/main_window.py")
    assert "invalidate_dshow_device_cache" in _read("app/integration/noop_engine.py")


def test_the_verbose_device_dump_is_opt_in():
    """It used to spawn a second enumeration purely to print one line."""
    src = _read("app/integration/noop_engine.py")
    i = src.find("HGR_FFMPEG_DEVICE_DUMP")
    assert i != -1
    assert "list_dshow_video_devices()" in src[i:i + 500]
