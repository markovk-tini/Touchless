"""r20: undoing a camera-control change may only ever hand control back
to the driver.

The field failure this prevents: a camera is found at Exposure=-6/Manual
(a latch left behind by an earlier crashed session), which makes the
preview almost black. The engine un-latches it to Auto during the
session. A faithful "put it back how we found it" restore on exit would
write -6/Manual again, so the next launch is dark too, forever.
"""

from hgr.app.camera import dshow_controls as DC


def _change(before_value, before_flags, after_value, after_flags, prop="Exposure"):
    return {
        "camera": "FULL HD 1080P Webcam", "section": "camera_control",
        "property": prop, "index": 4,
        "before_value": before_value, "before_flags": before_flags,
        "after_value": after_value, "after_flags": after_flags,
    }


def test_auto_now_and_manual_before_is_refused():
    """The exact dark-camera case."""
    ch = _change(-6, DC.FLAG_MANUAL, -4, DC.FLAG_AUTO)
    ok, refused = DC.split_restorable([ch])
    assert ok == [] and refused == [ch]


def test_manual_now_and_auto_before_is_restored():
    """We took control away, so we must give it back."""
    ch = _change(-4, DC.FLAG_AUTO, -6, DC.FLAG_MANUAL)
    ok, refused = DC.split_restorable([ch])
    assert ok == [ch] and refused == []


def test_value_change_within_manual_is_restored():
    ch = _change(-4, DC.FLAG_MANUAL, -6, DC.FLAG_MANUAL)
    ok, refused = DC.split_restorable([ch])
    assert ok == [ch] and refused == []


def test_value_change_within_auto_is_restored():
    ch = _change(3950, DC.FLAG_AUTO, 4600, DC.FLAG_AUTO, prop="WhiteBalance")
    ok, refused = DC.split_restorable([ch])
    assert ok == [ch] and refused == []


def test_a_mixed_batch_is_split_not_dropped():
    keep = _change(-4, DC.FLAG_AUTO, -6, DC.FLAG_MANUAL)
    drop = _change(-6, DC.FLAG_MANUAL, -4, DC.FLAG_AUTO, prop="Focus")
    ok, refused = DC.split_restorable([keep, drop])
    assert ok == [keep] and refused == [drop]


def test_empty_and_none_are_safe():
    assert DC.split_restorable([]) == ([], [])
    assert DC.split_restorable(None) == ([], [])


def test_corrupt_entries_are_refused_not_written():
    bad = {"camera": "x", "property": "Exposure", "after_flags": object()}
    ok, refused = DC.split_restorable([bad])
    assert ok == [] and refused == [bad]
