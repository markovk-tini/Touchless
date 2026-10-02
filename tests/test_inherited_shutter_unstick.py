"""r25: un-stick a camera a previous session left on a manual short shutter.

Observed on the reference rig 2026-09-25. The START ledger read
`USB Video Device: ... Exposure=-6/Manual`, Boost was off so the engine
correctly wrote nothing (`session changes: none`), and the live view was
dark with no way back short of the Settings reset button.

`_unstick_inherited_short_shutter` exists for exactly this and could not
fire: its first condition demanded that `classify_camera_shutter_hint`
POSITIVELY return True, and a generic "USB Video Device" classifies as
None -- unknown, not positive. So the one camera most likely to be left
latched was the one the recovery refused to touch.

The condition was not arbitrary. OpenCV's `cap.get(CAP_PROP_AUTO_EXPOSURE)`
is unimplemented on DirectShow and returns -1.0 always, so the "is it
manual?" half of the staleness test can never pass there, leaving only
`exposure < -2.5`. A Razer Kiyo Pro reads **-4.0 in normal AUTO mode** and
trips that threshold -- which is precisely how r17 knocked it out of auto
into HDR and cost 60 -> 25 fps (checkpoint 3.9).

The fix is to stop asking OpenCV. `dshow_controls` reads IAMCameraControl
over COM and returns the real Auto/Manual flag -- it is the module that
prints `-6/Manual` vs `-4/Auto` in the ledger. A camera in Auto can then
never satisfy the test, so r17's false positive is impossible by
construction rather than by classifier allow-list.

These tests pin both halves: the stuck generic camera IS recovered, and
the Kiyo Pro in auto is NOT touched.
"""

import types

import pytest

import hgr.app.integration.noop_engine as NE

W = NE.GestureWorker

FLAG_AUTO = 1
FLAG_MANUAL = 2


def _snap(name, exposure_value, exposure_flags):
    return {
        name: {
            "camera_control": {
                "Exposure": {"index": 4, "value": exposure_value,
                             "flags": exposure_flags, "min": -11, "max": -2,
                             "default": -6},
            },
            "video_proc_amp": {
                "Brightness": {"index": 0, "value": 128, "flags": FLAG_MANUAL,
                               "min": 0, "max": 255, "default": 128},
            },
        }
    }


def _worker(snap, monkeypatch, *, boost=False):
    from hgr.app.camera import dshow_controls as DC
    monkeypatch.setattr(DC, "snapshot_all", lambda only_name=None: snap)
    monkeypatch.setattr(NE.sys, "platform", "win32")
    w = W.__new__(W)
    w.config = types.SimpleNamespace(camera_force_short_shutter=boost)
    w._short_shutter_active_for_display = False
    return w


class TestTheComFlagIsWhatWeAsk:
    """OpenCV cannot answer this on DirectShow; COM can."""

    def test_manual_is_reported_as_manual(self, monkeypatch):
        w = _worker(_snap("USB Video Device", -6, FLAG_MANUAL), monkeypatch)
        assert w._exposure_flag_per_com("USB Video Device (Camera 0)") is True

    def test_auto_is_reported_as_auto(self, monkeypatch):
        w = _worker(_snap("Razer Kiyo Pro", -4, FLAG_AUTO), monkeypatch)
        assert w._exposure_flag_per_com("Razer Kiyo Pro (Camera 0)") is False

    def test_the_qt_camera_suffix_still_matches(self, monkeypatch):
        """The engine carries the Qt name; COM enumerates without the
        ' (Camera N)' suffix."""
        w = _worker(_snap("USB Video Device", -6, FLAG_MANUAL), monkeypatch)
        assert w._exposure_flag_per_com("USB Video Device (Camera 12)") is True

    def test_an_unreadable_snapshot_is_unknown_not_manual(self, monkeypatch):
        """Unknown must never be treated as 'stuck' -- that would be a
        driver write on no evidence."""
        w = _worker({}, monkeypatch)
        assert w._exposure_flag_per_com("USB Video Device") is None

    def test_a_com_failure_is_unknown(self, monkeypatch):
        from hgr.app.camera import dshow_controls as DC
        monkeypatch.setattr(DC, "snapshot_all", lambda only_name=None: (
            (_ for _ in ()).throw(RuntimeError("COM unavailable"))))
        monkeypatch.setattr(NE.sys, "platform", "win32")
        w = W.__new__(W)
        assert w._exposure_flag_per_com("USB Video Device") is None

    def test_a_camera_without_exposure_is_unknown(self, monkeypatch):
        snap = {"Weird Cam": {"camera_control": {"Zoom": {"index": 3,
                "value": 100, "flags": FLAG_MANUAL}}, "video_proc_amp": {}}}
        w = _worker(snap, monkeypatch)
        assert w._exposure_flag_per_com("Weird Cam") is None


class TestTheSectionSelectorIsReal:
    """`_procamp_for_device` now serves both COM sections; the light lift
    still has to get IAMVideoProcAmp, not IAMCameraControl."""

    def test_default_section_is_still_procamp(self):
        snap = _snap("Cam", -6, FLAG_MANUAL)
        block, _how = W._procamp_for_device(snap, "Cam")
        assert "Brightness" in block and "Exposure" not in block

    def test_camera_control_is_selectable(self):
        snap = _snap("Cam", -6, FLAG_MANUAL)
        block, _how = W._procamp_for_device(snap, "Cam",
                                            section="camera_control")
        assert "Exposure" in block and "Brightness" not in block


class TestTheKiyoIsNeverTouched:
    """The r17 regression, pinned. Checkpoint 3.9: a bare exposure
    threshold fired on a Kiyo Pro reading -4.0 in normal auto and knocked
    it into HDR, 60 -> 25 fps."""

    def test_auto_at_minus_four_is_not_manual(self, monkeypatch):
        """-4.0 trips the old `< -2.5` threshold. The flag is what saves
        it, and the flag says Auto."""
        w = _worker(_snap("Razer Kiyo Pro", -4, FLAG_AUTO), monkeypatch)
        assert w._exposure_flag_per_com("Razer Kiyo Pro") is False

    @pytest.mark.parametrize("exp", [-2.6, -4.0, -5.0, -6.0, -7.0])
    def test_no_exposure_value_in_auto_ever_reads_manual(self, exp,
                                                         monkeypatch):
        """However short the driver drives it, Auto is Auto."""
        w = _worker(_snap("Razer Kiyo Pro", exp, FLAG_AUTO), monkeypatch)
        assert w._exposure_flag_per_com("Razer Kiyo Pro") is not True

    def test_the_gate_still_refuses_when_boost_is_on(self, monkeypatch):
        """Condition 3: if anything is ASKING for a short shutter we must
        not fight it, however stuck the camera looks."""
        import inspect
        import re

        src = inspect.getsource(W._unstick_inherited_short_shutter)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        i = body.index("camera_force_short_shutter")
        j = body.index("_exposure_flag_per_com")
        assert i < j, (
            "the 'is anyone asking for a short shutter?' guard must be "
            "evaluated before the staleness test, not after"
        )


class TestTheWideningIsScoped:
    def test_the_classifier_is_still_honoured(self):
        """A positive classifier verdict must still be sufficient on its
        own -- the COM flag widens the gate, it does not replace it."""
        import inspect
        import re

        src = inspect.getsource(W._unstick_inherited_short_shutter)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "_clf_says_generic or _com_manual is True" in body

    def test_unknown_com_plus_unknown_classifier_does_nothing(self,
                                                              monkeypatch):
        """The dangerous middle: we know nothing about the camera and
        nothing about the flag. Must not write."""
        w = _worker({}, monkeypatch)
        assert w._exposure_flag_per_com("Some Unknown Cam") is None
        from hgr.app.camera.camera_utils import classify_camera_shutter_hint
        assert classify_camera_shutter_hint("Some Unknown Cam") is not True
