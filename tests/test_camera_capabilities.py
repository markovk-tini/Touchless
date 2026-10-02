"""Ask the camera what it supports instead of discovering it by failing.

Entering Lite or GPU mode used to be trial and error: ask for MJPG at
60 fps, then 30, then retry once. On a webcam with no suitable pin that
cascade is doomed before it starts, and it costs seconds of frozen UI and
three antivirus-visible ffmpeg launches every time. DirectShow will just
tell us, in one call, so the decision becomes a lookup.
"""

import pytest

from hgr.app.camera import camera_capabilities as CC


# Real output from a Razer Kiyo Pro, trimmed. Note the shape that matters:
# the uncompressed pin tops out at 30 fps while MJPG reaches 60, which is
# precisely why the fast path exists.
KIYO = """
[dshow @ 0000] DirectShow video device options (from video devices)
[dshow @ 0000]  Pin "Capture"
[dshow @ 0000]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30 (tv, bt470bg)
[dshow @ 0000]   pixel_format=nv12  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000]   vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=60.0002
[dshow @ 0000]   vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=60.0002 (pc, bt470bg)
[dshow @ 0000]   vcodec=mjpeg  min s=1280x720 fps=5 max s=1280x720 fps=60.0002
[dshow @ 0000]   vcodec=h264  min s=1920x1080 fps=5 max s=1920x1080 fps=60.0002
"""

# A cheap webcam that offers MJPG, but only at 30 fps. Asking this one for
# 60 is what produces "Could not set video options".
CHEAP_30 = """
[dshow @ 0000]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000]   vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000]   vcodec=mjpeg  min s=1280x720 fps=5 max s=1280x720 fps=30
"""

# A webcam with no compressed pin at all. The fast path can never work.
NO_MJPG = """
[dshow @ 0000]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000]   pixel_format=yuyv422  min s=1280x720 fps=5 max s=1280x720 fps=10
"""


# ------------------------------------------------------------ parsing

def test_it_parses_a_real_camera():
    modes = CC.parse_list_options(KIYO)
    fmts = {m["format"] for m in modes}
    assert {"mjpeg", "yuyv422", "nv12", "h264"} <= fmts


def test_duplicate_colour_range_variants_collapse():
    """ffmpeg prints each pin twice, once with a colour-range suffix."""
    modes = CC.parse_list_options(KIYO)
    mjpeg_640 = [
        m for m in modes
        if m["format"] == "mjpeg" and (m["width"], m["height"]) == (640, 480)
    ]
    assert len(mjpeg_640) == 1


def test_it_reads_the_advertised_maximum_frame_rate():
    modes = CC.parse_list_options(KIYO)
    mjpeg = next(m for m in modes if m["format"] == "mjpeg" and m["width"] == 640)
    yuyv = next(m for m in modes if m["format"] == "yuyv422" and m["width"] == 640)
    assert mjpeg["max_fps"] > 59
    assert yuyv["max_fps"] == 30.0


def test_garbage_in_does_not_crash():
    for junk in ("", None, "no modes here", "vcodec=mjpeg with no sizes"):
        assert CC.parse_list_options(junk) == []


# -------------------------------------------------------- the lookup

def test_a_capable_camera_gets_the_exact_mode_it_advertises():
    pick = CC.best_compressed_mode(CC.parse_list_options(KIYO), 640, 480, 60)
    assert pick["format"] == "mjpeg"
    assert (pick["width"], pick["height"]) == (640, 480)
    assert pick["fps"] == 60.0
    assert pick["exact_size"] is True


def test_a_30fps_camera_is_asked_for_30_not_60():
    """This is the whole point. Asking a 30 fps pin for 60 is what the
    field log shows being rejected with 'Could not set video options'."""
    pick = CC.best_compressed_mode(CC.parse_list_options(CHEAP_30), 640, 480, 60)
    assert pick["format"] == "mjpeg"
    assert pick["fps"] == 30.0
    assert pick["advertised_max_fps"] == 30.0


def test_a_camera_with_no_compressed_pin_returns_nothing():
    modes = CC.parse_list_options(NO_MJPG)
    assert CC.supports_compressed(modes) is False
    assert CC.best_compressed_mode(modes, 640, 480, 60) is None


def test_unknown_capabilities_never_look_like_a_refusal():
    """None means 'we do not know', and callers must fall back rather
    than treat it as 'unsupported'."""
    assert CC.best_compressed_mode(None, 640, 480, 60) is None
    assert CC.supports_compressed(None) is False


def test_it_never_upgrades_you_to_a_bigger_stream():
    """If the exact size is unavailable, pick something no larger."""
    modes = CC.parse_list_options(CHEAP_30)
    pick = CC.best_compressed_mode(modes, 800, 600, 60)
    assert pick is not None
    assert pick["width"] <= 800 and pick["height"] <= 600
    assert pick["exact_size"] is False


def test_a_request_smaller_than_every_pin_finds_nothing():
    pick = CC.best_compressed_mode(CC.parse_list_options(CHEAP_30), 320, 240, 60)
    assert pick is None


def test_the_advertised_rate_is_rounded_down_not_up():
    """Drivers report 60.0002; asking for that verbatim invites a reject."""
    pick = CC.best_compressed_mode(CC.parse_list_options(KIYO), 640, 480, 60)
    assert pick["fps"] == 60.0
    assert isinstance(pick["fps"], float)


@pytest.mark.parametrize("want,expected", [(60, 30.0), (30, 30.0), (15, 15.0)])
def test_the_request_is_clamped_not_replaced(want, expected):
    pick = CC.best_compressed_mode(CC.parse_list_options(CHEAP_30), 640, 480, want)
    assert pick["fps"] == expected


# --------------------------------------------------------- the probe

def test_the_probe_reports_unknown_rather_than_raising(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("ffmpeg missing")
    monkeypatch.setattr(CC.subprocess, "run", _boom)
    assert CC.probe_camera_capabilities("Cam", "ffmpeg.exe") is None


def test_the_probe_ignores_the_exit_code(monkeypatch):
    """ffmpeg always exits non-zero for -list_options, so the return code
    carries no information and must not be treated as failure."""
    class R:
        returncode = 1
        stderr = KIYO
        stdout = ""
    monkeypatch.setattr(CC.subprocess, "run", lambda *a, **kw: R())
    modes = CC.probe_camera_capabilities("Cam", "ffmpeg.exe")
    assert modes and CC.supports_compressed(modes)


def test_the_probe_needs_a_device_and_a_binary():
    assert CC.probe_camera_capabilities("", "ffmpeg.exe") is None
    assert CC.probe_camera_capabilities("Cam", "") is None
