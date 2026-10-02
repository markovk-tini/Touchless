"""r23: the short-shutter exposure must suit the camera, not a constant.

The literal -6.0 we shipped since r49 was tuned on a 60 fps camera. On
the 30 fps field webcam it threw away a full stop of light for frames
that camera can never deliver, which is what produced the luma-10
preview. These tests pin the physics and both rigs.
"""

import math

import pytest

from hgr.app.camera.exposure_policy import (
    DEFAULT_EXPOSURE,
    MAX_EXPOSURE,
    MIN_EXPOSURE,
    advertised_fps_for,
    exposure_for_camera,
    exposure_for_fps,
)


class TestExposureForFps:
    def test_sixty_fps_camera_keeps_the_historical_minus_six(self):
        """The dev rig (Kiyo Pro, mjpeg 640x480@60) must not move."""
        assert exposure_for_fps(60.0) == -6.0

    def test_thirty_fps_camera_gets_a_full_extra_stop(self):
        """The field rig. -5.0 is twice the light at the same 30 fps."""
        assert exposure_for_fps(30.0) == -5.0

    def test_the_chosen_shutter_always_fits_inside_one_frame(self):
        """The invariant the whole module exists to hold.

        Only asserted where neither clamp binds. Past 64 fps the
        MIN_EXPOSURE floor deliberately wins -- we would rather overrun
        the frame period than darken the sensor below the -6 that r49
        already proved is the useful limit.
        """
        for fps in (16.0, 20.0, 24.0, 25.0, 30.0, 50.0, 60.0, 64.0):
            shutter = 2.0 ** exposure_for_fps(fps)
            assert shutter <= 1.0 / fps + 1e-9, (
                f"{fps} fps: shutter {shutter*1000:.2f} ms overruns "
                f"the {1000.0/fps:.2f} ms frame period"
            )

    def test_past_the_floor_we_choose_light_over_the_frame_period(self):
        """Above 64 fps the -6 floor binds, on purpose."""
        for fps in (90.0, 120.0, 240.0):
            assert exposure_for_fps(fps) == MIN_EXPOSURE

    def test_never_shorter_than_the_historical_floor(self):
        """A 120 fps camera must not be darkened past -6."""
        assert exposure_for_fps(120.0) == MIN_EXPOSURE
        assert exposure_for_fps(1000.0) == MIN_EXPOSURE

    def test_never_longer_than_the_bright_manual_ceiling(self):
        """A very slow camera stops at -4; we will not trade more fps."""
        assert exposure_for_fps(5.0) == MAX_EXPOSURE
        assert exposure_for_fps(1.0) == MAX_EXPOSURE

    @pytest.mark.parametrize("bad", [None, 0.0, -30.0, float("nan"), float("inf"), "x", object()])
    def test_unknown_or_nonsense_fps_keeps_pre_r23_behaviour(self, bad):
        assert exposure_for_fps(bad) == DEFAULT_EXPOSURE

    def test_result_is_always_a_whole_stop_in_range(self):
        for fps in range(1, 241):
            v = exposure_for_fps(float(fps))
            assert v == float(int(v)), f"{fps} fps gave a fractional stop {v}"
            assert MIN_EXPOSURE <= v <= MAX_EXPOSURE

    def test_monotonic_faster_camera_never_gets_a_longer_shutter(self):
        prev = None
        for fps in range(5, 241):
            v = exposure_for_fps(float(fps))
            if prev is not None:
                assert v <= prev, f"{fps} fps got a longer shutter than {fps-1}"
            prev = v


class TestAdvertisedFpsFor:
    FIELD = [
        {"format": "mjpeg", "width": 1920, "height": 1080, "max_fps": 30.0},
        {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0},
        {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
    ]
    DEV = [
        {"format": "mjpeg", "width": 640, "height": 480, "max_fps": 60.0},
        {"format": "yuyv422", "width": 640, "height": 480, "max_fps": 30.0},
    ]

    def test_picks_the_highest_rate_at_that_resolution(self):
        """Uncompressed pin tops out at 30, MJPG reaches 60: take 60."""
        assert advertised_fps_for(self.DEV, 640, 480) == 60.0

    def test_field_camera_reports_thirty(self):
        assert advertised_fps_for(self.FIELD, 640, 480) == 30.0

    def test_other_resolutions_are_ignored(self):
        assert advertised_fps_for(self.FIELD, 1280, 720) is None

    @pytest.mark.parametrize("modes", [None, [], [{"bogus": 1}], [{"width": "x"}]])
    def test_unknown_is_never_mistaken_for_unsupported(self, modes):
        assert advertised_fps_for(modes, 640, 480) is None

    def test_malformed_entries_are_skipped_not_fatal(self):
        modes = [{"width": 640, "height": 480}, *self.FIELD]
        assert advertised_fps_for(modes, 640, 480) == 30.0


class TestExposureForCamera:
    def test_field_rig_end_to_end(self):
        """mjpeg 640x480@30 -> -5.0, the whole point of r23."""
        assert exposure_for_camera(TestAdvertisedFpsFor.FIELD, 640, 480) == -5.0

    def test_dev_rig_end_to_end_is_unchanged(self):
        assert exposure_for_camera(TestAdvertisedFpsFor.DEV, 640, 480) == -6.0

    def test_no_capability_data_falls_back_to_the_old_constant(self):
        assert exposure_for_camera(None, 640, 480) == DEFAULT_EXPOSURE
        assert exposure_for_camera([], 640, 480) == DEFAULT_EXPOSURE

    def test_a_stop_of_light_is_exactly_a_doubling(self):
        """Why -5 fixes the dim preview: twice the photons, same fps."""
        field = exposure_for_camera(TestAdvertisedFpsFor.FIELD, 640, 480)
        assert 2.0 ** field == pytest.approx(2.0 * (2.0 ** DEFAULT_EXPOSURE))
        # and it still fits a 30 fps frame period
        assert 2.0 ** field <= 1.0 / 30.0

    def test_field_camera_at_an_unprobed_size_is_not_guessed(self):
        assert exposure_for_camera(TestAdvertisedFpsFor.FIELD, 1280, 720) == DEFAULT_EXPOSURE


class TestCapsLookupIsNameSourceTolerant:
    """The capability cache is keyed on the DIRECTSHOW device name, but
    the exposure write site only has the Qt display name -- and
    `resolve_dshow_device_for_index` itself strips a trailing
    "(Camera N)" from the Qt name before comparing, which proves the two
    can differ. An exact-key miss would silently return the old -6.0, so
    the fix would look shipped and change nothing. That is how r23 first
    went out (it read `CameraInfo.name`, which does not exist).
    """

    FIELD = [{"format": "mjpeg", "width": 640, "height": 480, "max_fps": 30.0},
             {"format": "mjpeg", "width": 1920, "height": 1080, "max_fps": 30.0}]
    DEV = [{"format": "mjpeg", "width": 640, "height": 480, "max_fps": 60.0}]

    def _worker(self, store):
        import hgr.app.integration.noop_engine as NE
        w = NE.GestureWorker.__new__(NE.GestureWorker)
        cfg = type("C", (), {})()
        cfg.camera_capabilities = {}
        w.config = cfg
        stamp = w._caps_stamp()
        cfg.camera_capabilities = {
            k: {"stamp": stamp, "modes": v} for k, v in store.items()
        }
        return w

    def _one(self):
        return self._worker({"full hd 1080p webcam": self.FIELD})

    @pytest.mark.parametrize("name", [
        "FULL HD 1080P Webcam",              # the DirectShow name itself
        "FULL HD 1080P Webcam (Camera 0)",   # the Qt name the app builds
        "  full hd 1080p WEBCAM  ",          # case and padding
        "FULL HD 1080P Webcam Device",       # substring either direction
    ])
    def test_every_realistic_name_form_finds_the_camera(self, name):
        assert self._one()._short_shutter_exposure_for(name) == -5.0

    def test_a_single_learned_camera_is_used_even_on_a_name_miss(self):
        """With one learned camera there is nothing else it could be."""
        assert self._one()._short_shutter_exposure_for("Integrated Camera") == -5.0

    def test_an_empty_name_stays_loud_instead_of_being_papered_over(self):
        """An empty name means the CALLER is broken. The single-camera
        fallback must NOT rescue it, or the r23 bug class hides again."""
        assert self._one()._short_shutter_exposure_for("") == DEFAULT_EXPOSURE

    def test_with_two_cameras_learned_it_refuses_to_guess(self):
        w = self._worker({"full hd 1080p webcam": self.FIELD, "razer kiyo pro": self.DEV})
        assert w._short_shutter_exposure_for("Some Other Cam") == DEFAULT_EXPOSURE
        assert w._short_shutter_exposure_for("Razer Kiyo Pro") == -6.0
        assert w._short_shutter_exposure_for("FULL HD 1080P Webcam") == -5.0

    def test_nothing_learned_falls_back_to_the_old_constant(self):
        assert self._worker({})._short_shutter_exposure_for("FULL HD 1080P Webcam") == DEFAULT_EXPOSURE

    def test_the_lookup_never_enumerates_devices_or_spawns_ffmpeg(self):
        """It must not be able to add an antivirus prompt."""
        import ast
        import inspect
        import re
        import textwrap
        import hgr.app.integration.noop_engine as NE

        src = textwrap.dedent(
            inspect.getsource(NE.GestureWorker._caps_for_exposure_lookup)
        )
        # Strip the docstring and comments: they legitimately NAME the
        # resolver they are explaining. Only executable code counts.
        tree = ast.parse(src)
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            fn.body = fn.body[1:]
        code = "\n".join(
            re.sub(r"#.*$", "", line) for line in ast.unparse(tree).splitlines()
        )
        for forbidden in ("resolve_dshow_device_for_index", "list_dshow_video_devices",
                          "subprocess", "probe_camera_capabilities", "_camera_caps_learn"):
            assert forbidden not in code, f"lookup reaches for {forbidden}"
