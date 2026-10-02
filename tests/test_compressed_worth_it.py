"""r25: don't pay for the MJPG cascade when it buys nothing.

Switching to the ffmpeg-MJPG capture is expensive: it releases a working
capture, sleeps 600 ms for the DirectShow graph to tear down, spawns
ffmpeg (which antivirus prompts on), and on failure retries and falls
back. The field log measures 15-20 s of frozen UI per mode swap, twice,
and a `paint_gap gap=20297ms`.

That price is worth paying when MJPG lifts a YUY2 bandwidth ceiling --
the case it was built for, e.g. `mjpeg 1280x720@30` against
`yuyv422 1280x720@8`.

It is not worth paying when both pins run at the same rate. The field
camera advertises BOTH `mjpeg 640x480@30` and `yuyv422 640x480@30`, and
640x480 is exactly what Lite and GPU request -- so the whole cascade buys
zero fps and then fails anyway. The r21 probe had already collected the
answer; nothing ever asked the question.
"""

import pytest

from hgr.app.camera.camera_capabilities import compressed_is_worth_it as worth

#: The field rig's camera, verbatim from its own `ffmpeg -list_options`.
FIELD = [
    {"format": "mjpeg", "width": 1920, "height": 1080, "max_fps": 30.0},
    {"format": "mjpeg", "width": 1280, "height": 720, "max_fps": 30.0},
    {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0},
    {"format": "yuyv422", "width": 1920, "height": 1080, "max_fps": 5.0},
    {"format": "yuyv422", "width": 1280, "height": 720, "max_fps": 8.0},
    {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
]

#: A premium camera: compressed really is faster at the same size.
KIYO = [
    {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 60.0002},
    {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
]


class TestTheFieldCamera:
    def test_no_gain_at_the_size_lite_and_gpu_request(self):
        assert worth(FIELD, 640, 480) is False

    def test_real_gain_at_720p(self):
        """30 vs 8 -- here the cascade earns its cost."""
        assert worth(FIELD, 1280, 720) is True

    def test_real_gain_at_1080p(self):
        assert worth(FIELD, 1920, 1080) is True

    def test_the_verdict_is_per_size(self):
        """The same camera must be able to say yes at 720p and no at
        480p; a whole-camera verdict would lose the fast path."""
        assert worth(FIELD, 1280, 720) is not worth(FIELD, 640, 480)


class TestTheReferenceRigKeepsItsFastPath:
    def test_a_faster_compressed_pin_still_wins(self):
        assert worth(KIYO, 640, 480) is True

    def test_a_hair_of_difference_is_not_a_reason_to_pay(self):
        """30.0 and 30.0002 are the same pin rate written two ways."""
        modes = [
            {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0002},
            {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
        ]
        assert worth(modes, 640, 480) is False


class TestMissingEvidenceNeverBlocksTheFastPath:
    @pytest.mark.parametrize("modes", [None, [], [{}]])
    def test_nothing_learned_means_no_opinion(self, modes):
        assert worth(modes, 640, 480) is None

    def test_only_compressed_known_is_no_opinion(self):
        assert worth([{"format": "mjpeg", "width": 640, "height": 480,
                       "max_fps": 30.0}], 640, 480) is None

    def test_only_uncompressed_known_is_no_opinion(self):
        assert worth([{"format": "yuyv422", "width": 640, "height": 480,
                       "max_fps": 30.0}], 640, 480) is None

    def test_a_size_we_never_probed_is_no_opinion(self):
        assert worth(FIELD, 800, 600) is None

    @pytest.mark.parametrize("bad", [
        {"format": "mjpeg", "width": "x", "height": 480, "max_fps": 30.0},
        {"format": "mjpeg", "width": 640, "height": 480, "max_fps": "fast"},
        {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 0},
        {"format": "mjpeg", "width": 640, "height": 480, "max_fps": -1},
    ])
    def test_malformed_modes_are_skipped_not_fatal(self, bad):
        assert worth([bad], 640, 480) is None
        # and a malformed entry alongside good ones does not flip a verdict
        assert worth(FIELD + [bad], 640, 480) is False


class TestItPicksTheBestPinPerSide:
    def test_the_fastest_compressed_pin_counts(self):
        modes = [
            {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 15.0},
            {"format": "mjpg", "width": 640, "height": 480, "max_fps": 60.0},
            {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
        ]
        assert worth(modes, 640, 480) is True

    def test_a_format_in_neither_list_is_ignored(self):
        """h264 is not a pin this app ever requests -- COMPRESSED_FORMATS
        is mjpeg only. Counting it as raw would let an h264 pin veto the
        MJPG path; counting it as compressed would make us request a
        stream we do not decode. It is evidence for neither side."""
        assert worth([{"format": "h264", "width": 640, "height": 480,
                       "max_fps": 60.0}], 640, 480) is None
        mixed = [
            {"format": "h264", "width": 640, "height": 480, "max_fps": 60.0},
            {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0},
            {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
        ]
        assert worth(mixed, 640, 480) is False

    def test_the_fastest_uncompressed_pin_counts(self):
        modes = [
            {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0},
            {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 5.0},
            {"format": "nv12", "width": 640, "height": 480, "max_fps": 30.0},
        ]
        assert worth(modes, 640, 480) is False
