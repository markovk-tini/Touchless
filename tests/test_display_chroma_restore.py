"""r23: the short-shutter display lift must not desaturate the preview.

`_compensate_short_shutter_for_display` lifted Y and merged the ORIGINAL
Cr/Cb back. YCrCb chroma excursion scales with signal level, so lifting
luma ~8x with chroma untouched drops perceived saturation by the same
factor -- measured 20% of true at the field camera's luma median of 10,
which is the user's "dim and practically black and white". The flat
`_SAT_GAIN = 1.15` meant to offset it was ~6.6x too small.

Two properties matter and both are pinned here:

1. The correction is PER PIXEL. A single gain derived from the frame
   median fixes the dark regions but over-saturates everything brighter
   (measured 148% on a lit face), which reads garish.
2. It happens on the DISPLAY fork only. The gamma stage's return value
   IS `detection_frame`, so chroma work there would change what
   MediaPipe sees. The detection frame must stay bit-identical to
   pre-r23 output.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import hgr.app.integration.noop_engine as NE


def _worker(active=True):
    w = NE.GestureWorker.__new__(NE.GestureWorker)
    w._short_shutter_active_for_display = active
    w._cap = type("C", (), {})()
    return w


def _scene(median, seed=7):
    """A full-gamut frame scaled to a given luma median."""
    rng = np.random.RandomState(seed)
    h, w = 120, 160
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = np.tile(np.arange(w) * 180 // w, (h, 1)).astype(np.uint8)
    hsv[..., 1] = rng.randint(60, 255, (h, w)).astype(np.uint8)
    hsv[..., 2] = rng.randint(40, 255, (h, w)).astype(np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    m = float(np.median(cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)))
    dim = np.clip(bright.astype(np.float32) * (median / m), 0, 255).astype(np.uint8)
    return bright, dim


def _settle(w, frame, n=16):
    """The median is sampled every 15 ticks, so run past one sample."""
    out = frame
    for _ in range(n):
        out = w._compensate_short_shutter_for_display(frame)
    return out


def _mean_sat(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean())


def _pre_r23_gamma(frame, gamma):
    """Exactly what the gamma stage produced before r23."""
    lut = np.clip(
        np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0, 0, 255
    ).astype(np.uint8)
    y, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    return cv2.cvtColor(cv2.merge([cv2.LUT(y, lut), cr, cb]), cv2.COLOR_YCrCb2BGR)


class TestDetectionPathIsUntouched:
    """The highest-risk property: tracking must not change at all."""

    def test_detection_frame_is_bit_identical_to_pre_r23(self):
        w = _worker()
        _, dim = _scene(10.0)
        detection_frame = _settle(w, dim)
        ref = _pre_r23_gamma(dim, w._ss_disp_target_gamma)
        assert np.array_equal(detection_frame, ref)

    def test_the_display_stage_does_not_mutate_its_input(self):
        w = _worker()
        _, dim = _scene(10.0)
        detection_frame = _settle(w, dim)
        before = detection_frame.copy()
        w._apply_display_local_contrast_if_needed(detection_frame)
        assert np.array_equal(detection_frame, before)

    def test_the_display_frame_really_does_differ(self):
        """Guard against the fix silently becoming a no-op."""
        w = _worker()
        _, dim = _scene(10.0)
        detection_frame = _settle(w, dim)
        display = w._apply_display_local_contrast_if_needed(detection_frame)
        assert not np.array_equal(display, detection_frame)


class TestSaturationIsRestored:
    def test_a_dim_frame_keeps_most_of_its_colour(self):
        w = _worker()
        bright, dim = _scene(10.0)
        display = w._apply_display_local_contrast_if_needed(_settle(w, dim))
        kept = _mean_sat(display) / _mean_sat(bright)
        # pre-r23 measured ~0.20 here
        assert kept > 0.80, f"only kept {kept:.0%} of saturation"

    def test_it_does_not_over_saturate(self):
        w = _worker()
        bright, dim = _scene(10.0)
        display = w._apply_display_local_contrast_if_needed(_settle(w, dim))
        assert _mean_sat(display) / _mean_sat(bright) < 1.25

    def test_a_bright_region_is_not_over_saturated(self):
        """The defect a single median-derived gain has: regions brighter
        than the median get the same chroma gain but less luma gain."""
        rng = np.random.RandomState(11)
        h, w_ = 120, 160
        hsv = np.zeros((h, w_, 3), np.uint8)
        hsv[..., 0] = np.tile(np.arange(w_) * 180 // w_, (h, 1)).astype(np.uint8)
        hsv[..., 1] = rng.randint(80, 220, (h, w_)).astype(np.uint8)
        v = np.full((h, w_), 40, np.uint8)
        v[30:90, 30:90] = 110          # a lit face against a dark room
        hsv[..., 2] = v
        true = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        m = float(np.median(cv2.cvtColor(true, cv2.COLOR_BGR2GRAY)))
        dim = np.clip(true.astype(np.float32) * (10.0 / m), 0, 255).astype(np.uint8)

        w = _worker()
        display = w._apply_display_local_contrast_if_needed(_settle(w, dim))

        def sat_at(f):
            return float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[40:80, 40:80, 1].mean())

        ratio = sat_at(display) / sat_at(true)
        assert 0.75 < ratio < 1.25, f"lit region at {ratio:.0%} of true saturation"

    def test_neutral_grey_stays_neutral(self):
        """A chroma gain must not invent colour on a grey wall."""
        w = _worker()
        grey = np.full((120, 160, 3), 12, np.uint8)
        display = w._apply_display_local_contrast_if_needed(_settle(w, grey))
        b, g, r = display[..., 0].astype(int), display[..., 1].astype(int), display[..., 2].astype(int)
        assert int(np.abs(b - g).max()) <= 6
        assert int(np.abs(g - r).max()) <= 6

    def test_no_chroma_channel_clipping(self):
        w = _worker()
        _, dim = _scene(10.0)
        display = w._apply_display_local_contrast_if_needed(_settle(w, dim))
        ycc = cv2.cvtColor(display, cv2.COLOR_BGR2YCrCb)
        pinned = ((ycc[..., 1] == 0) | (ycc[..., 1] == 255)
                  | (ycc[..., 2] == 0) | (ycc[..., 2] == 255)).mean()
        assert pinned < 0.01


class TestTheTableCannotGoStale:
    def test_it_is_keyed_on_gamma_alone(self):
        """No median in the key means nothing to go stale."""
        a, b = _worker(), _worker()
        _, dim = _scene(10.0)
        _settle(a, dim)
        _settle(b, dim)
        assert np.array_equal(a._ss_disp_chroma_tbl, b._ss_disp_chroma_tbl)

    def test_releasing_the_latch_clears_the_carried_state(self):
        w = _worker()
        _, dim = _scene(10.0)
        _settle(w, dim)
        assert w._ss_disp_chroma_tbl is not None
        w._short_shutter_active_for_display = False
        w._compensate_short_shutter_for_display(dim)
        assert w._ss_disp_chroma_tbl is None
        assert w._ss_disp_src_y is None
        assert not hasattr(w, "_ss_disp_target_gamma")

    def test_a_bright_frame_after_release_is_returned_untouched(self):
        """The stale-8x-gain-on-the-next-camera case."""
        w = _worker()
        _, dim = _scene(10.0)
        _settle(w, dim)
        w._short_shutter_active_for_display = False
        w._compensate_short_shutter_for_display(dim)
        w._short_shutter_active_for_display = True
        bright = np.clip(
            np.random.RandomState(1).randint(120, 200, (120, 160, 3)), 0, 255
        ).astype(np.uint8)
        assert w._compensate_short_shutter_for_display(bright) is bright


class TestFailSafe:
    def test_a_mismatched_stash_falls_back_instead_of_crashing(self):
        w = _worker()
        _, dim = _scene(10.0)
        gamma_out = _settle(w, dim)
        w._ss_disp_src_y = np.zeros((4, 4), np.uint8)
        out = w._apply_display_local_contrast_if_needed(gamma_out)
        assert out.shape == dim.shape and out.dtype == np.uint8

    def test_a_missing_table_falls_back_instead_of_crashing(self):
        w = _worker()
        _, dim = _scene(10.0)
        gamma_out = _settle(w, dim)
        w._ss_disp_chroma_tbl = None
        out = w._apply_display_local_contrast_if_needed(gamma_out)
        assert out.shape == dim.shape and out.dtype == np.uint8

    def test_the_1_15_fallback_is_still_there(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._apply_display_local_contrast_if_needed)
        assert "_SAT_GAIN = 1.15" in src


class TestReferenceRigIsInert:
    """Kiyo Pro: the latch is never raised, so both stages must be
    exact identity passthroughs -- not merely equal, the same object."""

    def test_both_stages_return_the_input_object(self):
        w = _worker(active=False)
        _, dim = _scene(10.0)
        a = w._compensate_short_shutter_for_display(dim)
        assert a is dim
        assert w._apply_display_local_contrast_if_needed(a) is a

    def test_an_already_bright_frame_is_returned_untouched(self):
        """Both stages, not just the gamma one.

        The latch can outlive the driver's short-shutter state -- it is
        raised on the r49 path which never sets `_ffmpeg_preflight_device`,
        so `_release_ffmpeg_preflight` cannot lower it. When that happens
        over a bright sensor, a CLAHE-only gate would keep running a
        low-light enhancement stack on a well-exposed frame, which is the
        "washed-out / dim live view" PERFORMANCE_CHECKPOINT section 3.5
        records as a reverted wrong path. Asserting the SECOND stage here
        is what catches that.
        """
        w = _worker(active=True)
        bright = np.clip(
            np.random.RandomState(4).randint(120, 200, (120, 160, 3)), 0, 255
        ).astype(np.uint8)
        gamma_out = _settle(w, bright)
        assert gamma_out is bright
        assert w._apply_display_local_contrast_if_needed(gamma_out) is gamma_out

    def test_a_stuck_latch_over_a_bright_sensor_is_inert(self):
        """The field-rig sequence: the hint is applied, the latch goes up,
        the driver later returns to auto, and nothing lowers the latch.
        The preview must still look native."""
        w = _worker(active=True)
        _, dim = _scene(10.0)
        _settle(w, dim)                      # latch up, lift active
        bright = np.clip(
            np.random.RandomState(9).randint(120, 200, (120, 160, 3)), 0, 255
        ).astype(np.uint8)
        # latch deliberately left True, sensor now bright
        for _ in range(16):
            out = w._compensate_short_shutter_for_display(bright)
        assert out is bright
        assert w._apply_display_local_contrast_if_needed(out) is out


class TestTheTableSurvivesBeingCleared:
    """The rebuild must fire whenever the table is MISSING, not only when
    gamma moves.

    Three paths drop the table without touching `_ss_disp_lut_gamma`: the
    latch-off exit, the bright-frame return, and the build-exception
    handler. Gating the rebuild on gamma alone meant the table stayed
    None afterwards and the display silently fell back to the flat 1.15
    -- the grey preview this whole change exists to remove. On the field
    camera it was certain rather than likely: every median under ~13
    clips gamma to the same 0.35, so the cache always hit.
    """

    def _sat_after(self, w, frame):
        return _mean_sat(w._apply_display_local_contrast_if_needed(_settle(w, frame)))

    def test_a_bright_frame_in_the_middle_does_not_kill_colour_forever(self):
        w = _worker()
        _, dim = _scene(10.0)
        first = self._sat_after(w, dim)
        bright = np.clip(
            np.random.RandomState(3).randint(120, 200, (120, 160, 3)), 0, 255
        ).astype(np.uint8)
        _settle(w, bright)
        assert w._ss_disp_chroma_tbl is None          # cleared, as designed
        again = self._sat_after(w, dim)
        assert w._ss_disp_chroma_tbl is not None, "table was never rebuilt"
        assert again == pytest.approx(first, rel=0.02)

    def test_toggling_the_latch_off_and_on_does_not_kill_colour(self):
        w = _worker()
        _, dim = _scene(10.0)
        first = self._sat_after(w, dim)
        w._short_shutter_active_for_display = False
        w._compensate_short_shutter_for_display(dim)
        w._short_shutter_active_for_display = True
        again = self._sat_after(w, dim)
        assert again == pytest.approx(first, rel=0.02)

    def test_the_rebuild_gate_does_not_depend_on_gamma_alone(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._compensate_short_shutter_for_display)
        assert '_ss_disp_chroma_tbl", None) is None' in src


class TestTheDrawingOverlayDoesNotGetChromaBlown:
    """`_blend_camera_drawing_overlay` mutates the frame in place BETWEEN
    the two display stages, so the stashed pre-lift luma stops describing
    those pixels. Left stale, a brush stroke is re-coloured using the
    camera pixel underneath it -- gains up to 8x. The shape guard cannot
    catch it, because the shape does not change.
    """

    def _worker_with_stroke(self):
        w = _worker()
        h, wd = 120, 160
        canvas = np.zeros((h, wd, 4), np.uint8)
        canvas[40:70, 40:110, :3] = (40, 40, 210)
        canvas[40:70, 40:110, 3] = 255
        w._drawing_render_target = "camera"
        w._camera_draw_canvas = canvas
        w._drawing_tool = "hidden"
        w._camera_draw_point = lambda shape: None
        return w

    def test_the_overlay_invalidates_the_stash(self):
        w = self._worker_with_stroke()
        _, dim = _scene(10.0)
        f = _settle(w, dim.copy())
        assert w._ss_disp_src_y is not None
        w._blend_camera_drawing_overlay(f)
        assert w._ss_disp_src_y is None

    def test_a_stroke_keeps_its_hue_and_does_not_clip(self):
        w = self._worker_with_stroke()
        _, dim = _scene(10.0)
        f = _settle(w, dim.copy())
        w._blend_camera_drawing_overlay(f)
        painted = f[55, 70].copy()
        shown = w._apply_display_local_contrast_if_needed(f)[55, 70]

        def to_hsv(px):
            return cv2.cvtColor(np.uint8([[px]]), cv2.COLOR_BGR2HSV)[0, 0]

        a, b = to_hsv(painted), to_hsv(shown)
        hue_drift = min(abs(int(a[0]) - int(b[0])), 180 - abs(int(a[0]) - int(b[0])))
        assert hue_drift <= 3, f"hue moved {hue_drift*2} degrees"
        assert int(b[1]) < 255, "saturation clipped"

    def test_an_untouched_frame_keeps_its_stash(self):
        """The overlay must only invalidate when it actually mutates."""
        w = _worker()
        w._drawing_render_target = "screen"      # not the camera target
        w._camera_draw_canvas = None
        _, dim = _scene(10.0)
        f = _settle(w, dim.copy())
        w._blend_camera_drawing_overlay(f)
        assert w._ss_disp_src_y is not None
