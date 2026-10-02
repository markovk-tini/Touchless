"""Pure-logic tests for hgr.app.camera.dshow_controls (no COM, no camera)."""
from __future__ import annotations

from hgr.app.camera import dshow_controls as dc


def _prop(index, value, flags):
    return {"index": index, "value": value, "flags": flags, "min": -11, "max": -2, "default": -6, "caps": 3}


def _snap(exposure=(-4, dc.FLAG_AUTO), brightness=(233, dc.FLAG_MANUAL), wb=(3950, dc.FLAG_AUTO)):
    return {
        "USB Video Device": {
            "camera_control": {"Exposure": _prop(4, *exposure)},
            "video_proc_amp": {"Brightness": _prop(0, *brightness), "WhiteBalance": _prop(7, *wb)},
        }
    }


def test_same_state_is_no_change():
    assert dc.diff(_snap(), _snap()) == []


def test_auto_value_drift_is_not_a_change():
    # Exposure in Auto moved -4 -> -5 because the room got darker: driver-owned.
    assert dc.diff(_snap(), _snap(exposure=(-5, dc.FLAG_AUTO))) == []
    # White balance in Auto drifting is not a change either.
    assert dc.diff(_snap(), _snap(wb=(4200, dc.FLAG_AUTO))) == []


def test_flag_flip_is_a_change_even_with_same_value():
    # The r17 / "0.75 then 3.0" signature: Auto -> Manual at the same value.
    changes = dc.diff(_snap(), _snap(exposure=(-4, dc.FLAG_MANUAL)))
    assert len(changes) == 1
    c = changes[0]
    assert c["property"] == "Exposure" and c["before_flags"] == dc.FLAG_AUTO and c["after_flags"] == dc.FLAG_MANUAL
    assert "Exposure -4/Auto -> -4/Manual" in dc.format_changes(changes)


def test_manual_value_change_is_a_change():
    changes = dc.diff(_snap(), _snap(brightness=(128, dc.FLAG_MANUAL)))
    assert [c["property"] for c in changes] == ["Brightness"]
    assert changes[0]["before_value"] == 233 and changes[0]["after_value"] == 128


def test_short_shutter_write_is_a_change():
    # r49 ON path: Auto/-4 -> Manual/-6.
    changes = dc.diff(_snap(), _snap(exposure=(-6, dc.FLAG_MANUAL)))
    assert len(changes) == 1 and changes[0]["after_value"] == -6


def test_missing_camera_after_is_ignored():
    assert dc.diff(_snap(), {}) == []


def test_format_snapshot_mentions_flags():
    s = dc.format_snapshot(_snap())
    assert "Exposure=-4/Auto" in s and "Brightness=233/Manual" in s


def test_restore_is_noop_without_changes():
    assert dc.restore(_snap(), []) == []
