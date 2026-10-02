"""r24: pay back the light a short shutter costs, at the driver.

The user asked for Boost to "do the short shutter speed without dimming
live view". PERFORMANCE_CHECKPOINT 3.5 says the display is the wrong
place to fix brightness, so the knob has to be a driver property.

Three things make that dangerous, and these tests pin the handling of
each: the physically correct property (Gain) is often not exposed at all,
value ranges are driver-specific with no 0-255 convention, and the
DIRECTION of Gamma is a convention rather than a guarantee -- a wrong
guess makes the preview worse. Hence propose-measure-revert.
"""

import pytest

from hgr.app.camera import light_policy as LP


def _prop(index, value, mn, mx, default, flags=LP.FLAG_MANUAL):
    return {"index": index, "value": value, "min": mn, "max": mx,
            "default": default, "flags": flags, "caps": 2}


#: The field camera, as its START snapshot reports it: Brightness, Contrast,
#: Hue, Saturation, Sharpness, Gamma, WhiteBalance, BacklightCompensation.
#: NOTE there is no Gain and no ColorEnable -- the driver returned
#: E_PROP_ID_UNSUPPORTED for both, so dshow_controls dropped them.
FIELD = {
    "Brightness": _prop(0, 0, -64, 64, 0),
    "Contrast": _prop(1, 0, 0, 100, 0),
    "Hue": _prop(2, 0, -180, 180, 0),
    "Saturation": _prop(3, 58, 0, 100, 64),
    "Sharpness": _prop(4, 3, 0, 7, 3),
    "Gamma": _prop(5, 165, 100, 500, 100),
    "WhiteBalance": _prop(7, 4600, 2800, 6500, 4600, LP.FLAG_AUTO),
    "BacklightCompensation": _prop(8, 0, 0, 2, 0),
}

#: A camera that does expose real sensor gain.
WITH_GAIN = dict(FIELD, Gain=_prop(9, 0, 0, 100, 0))


class TestItPrefersTheRightKnob:
    def test_gain_wins_when_the_camera_has_it(self):
        """Gain amplifies signal before the curve -- physically correct."""
        name, idx, target, orig = LP.plan_lift(WITH_GAIN)
        assert name == "Gain"
        assert idx == 9 and orig == 0 and target > 0

    def test_it_falls_back_to_gamma_when_gain_is_absent(self):
        """The field rig. No Gain in the snapshot at all."""
        assert "Gain" not in FIELD
        name, idx, target, orig = LP.plan_lift(FIELD)
        assert name == "Gamma"
        assert idx == 5 and orig == 165 and target > 165

    def test_brightness_is_the_last_resort(self):
        """An offset raises the noise floor and flattens contrast."""
        only_brightness = {"Brightness": _prop(0, 0, -64, 64, 0)}
        name, _, _, _ = LP.plan_lift(only_brightness)
        assert name == "Brightness"

    def test_the_order_is_gain_then_gamma_then_brightness(self):
        assert LP.LIFT_ORDER == ("Gain", "Gamma", "Brightness")

    def test_a_tried_property_is_not_proposed_again(self):
        first = LP.plan_lift(WITH_GAIN)[0]
        second = LP.plan_lift(WITH_GAIN, already_tried={first})[0]
        assert second != first


class TestItNeverGuessesARange:
    def test_the_target_respects_the_drivers_own_max(self):
        for cam in (FIELD, WITH_GAIN):
            name, _, target, _ = LP.plan_lift(cam)
            assert cam[name]["min"] <= target <= cam[name]["max"]

    def test_it_stops_short_of_the_raw_maximum(self):
        """Full Brightness on a UVC is a washed-out grey card."""
        name, _, target, _ = LP.plan_lift(FIELD)
        e = FIELD[name]
        ceiling = e["default"] + LP.CEILING_FRACTION * (e["max"] - e["default"])
        assert target <= ceiling
        assert target < e["max"]

    def test_a_property_already_at_the_ceiling_is_skipped(self):
        hot = {"Gamma": _prop(5, 480, 100, 500, 100)}
        assert LP.plan_lift(hot) is None

    def test_an_inverted_or_degenerate_range_is_skipped(self):
        assert LP.plan_lift({"Gamma": _prop(5, 10, 100, 100, 100)}) is None
        assert LP.plan_lift({"Gamma": _prop(5, 10, 500, 100, 100)}) is None

    def test_an_auto_property_is_left_to_the_driver(self):
        """WhiteBalance is Auto on the field camera. Taking a property off
        Auto is a bigger promise than brightening a preview."""
        auto_only = {"Gamma": _prop(5, 165, 100, 500, 100, LP.FLAG_AUTO)}
        assert LP.plan_lift(auto_only) is None

    @pytest.mark.parametrize("bad", [None, {}, {"Gamma": None}, {"Gamma": {}},
                                     {"Gamma": {"value": "x"}}, "nonsense"])
    def test_malformed_input_is_never_fatal(self, bad):
        assert LP.plan_lift(bad) is None


class TestItOnlyActsOnADarkFrame:
    @pytest.mark.parametrize("luma", [0.0, 10.0, 44.0, 69.9])
    def test_a_dark_frame_qualifies(self, luma):
        assert LP.frame_is_dark_enough_to_lift(luma) is True

    @pytest.mark.parametrize("luma", [70.0, 100.0, 180.0, 255.0])
    def test_a_well_exposed_frame_is_left_alone(self, luma):
        assert LP.frame_is_dark_enough_to_lift(luma) is False

    @pytest.mark.parametrize("bad", [None, "x", float("nan"), -1.0])
    def test_an_unknown_luma_never_triggers_a_driver_write(self, bad):
        assert LP.frame_is_dark_enough_to_lift(bad) is False


class TestTheMeasureAndRevertContract:
    """The safety net. Gamma's direction is a driver convention, not a
    guarantee, so a write is proposed and then judged on the frame."""

    def test_a_real_improvement_is_kept(self):
        assert LP.verdict(10.0, 40.0) is True

    def test_a_write_that_darkened_the_frame_is_rejected(self):
        assert LP.verdict(40.0, 10.0) is False

    def test_a_negligible_change_is_rejected(self):
        """Not worth a permanent deviation from the user's camera."""
        assert LP.verdict(40.0, 42.0) is False
        assert LP.verdict(40.0, 40.0) is False

    def test_the_threshold_is_where_it_says_it_is(self):
        assert LP.verdict(40.0, 40.0 + LP.MIN_USEFUL_GAIN) is True
        assert LP.verdict(40.0, 40.0 + LP.MIN_USEFUL_GAIN - 0.1) is False

    @pytest.mark.parametrize("b,a", [(None, 40.0), (40.0, None), (None, None),
                                     ("x", 40.0), (40.0, "x")])
    def test_an_unmeasurable_result_is_reverted_not_defended(self, b, a):
        assert LP.verdict(b, a) is False


class TestItStaysPure:
    def test_no_camera_or_ui_imports(self):
        import inspect

        src = inspect.getsource(LP)
        for forbidden in ("import cv2", "PySide6", "comtypes", "subprocess",
                          "open(", "os.environ"):
            assert forbidden not in src
