"""Pay back the light a short shutter costs, at the driver.

Background
----------
The r49 short shutter is what makes a cheap UVC webcam deliver its full
frame rate in a dim room -- measured 3x on the field camera. It does that
by collecting less light, so the picture goes dark. r23 halved the
problem by choosing the longest shutter that still fits a frame period
(``exposure_policy``) and stopped the display lift from desaturating, but
the frame is still darker than it was on auto exposure, and
PERFORMANCE_CHECKPOINT 3.5 is explicit that a display-side stack is the
wrong place to fix brightness: "fighting the camera driver rather than
getting the raw signal right".

Getting the raw signal right means more gain at the sensor or ISP. Three
things make that awkward, and this module exists to handle all three
without ever guessing:

1. **Sensor Gain is often not exposed.** The field camera advertises
   Brightness, Contrast, Hue, Saturation, Sharpness, Gamma, WhiteBalance
   and BacklightCompensation -- but NOT Gain. So the physically correct
   knob is simply unavailable there and the plan has to fall back.

2. **Value ranges are driver-specific.** There is no 0-255 convention.
   ``dshow_controls.snapshot_all`` already queries ``GetRange`` for every
   property, every session, so the real min/max/default/step are on hand
   and nothing here has to assume.

3. **Direction is driver-specific too.** On most UVC cameras a higher
   Gamma lifts midtones, but that is a convention, not a guarantee, and
   a wrong guess makes the preview WORSE. So this module never claims a
   write will help: it proposes one, and the caller measures the frame
   and calls :func:`verdict` to decide whether to keep it.

Order of preference
-------------------
``Gain`` first -- it is the physically correct knob, amplifying signal
before the curve. Then ``Gamma``, which lifts shadows while leaving
highlights roughly alone, which is exactly the shape a short-shutter
frame needs. ``Brightness`` last: it is an offset, so it raises the noise
floor and flattens contrast, and it is the most likely of the three to
look bad.

Pure module: no cv2, no COM, no Qt, no I/O. Unit-testable without a
camera.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: Tried in order. First one the camera actually exposes wins.
LIFT_ORDER = ("Gain", "Gamma", "Brightness")

#: How far toward the ceiling a single step moves.
LIFT_FRACTION = 0.5

#: Never push a property past ``default + CEILING_FRACTION * (max - default)``.
#: A driver's max is frequently unusable -- full Brightness on a UVC is a
#: washed-out grey card -- so the ceiling is deliberately short of it.
CEILING_FRACTION = 0.6

#: Below this median luma (0-255) the frame is dark enough to be worth a
#: driver write at all. Above it, leave the camera alone.
DARK_LUMA = 70.0

#: A lift has to move the median at least this much to be worth keeping.
#: Smaller than this and we are paying a driver write, and a permanent
#: deviation from the user's camera settings, for nothing.
MIN_USEFUL_GAIN = 6.0

#: DirectShow IAMVideoProcAmp flags.
FLAG_AUTO = 1
FLAG_MANUAL = 2


def _entry(procamp: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(procamp, dict):
        return None
    e = procamp.get(name)
    return e if isinstance(e, dict) else None


def frame_is_dark_enough_to_lift(median_luma: Optional[float]) -> bool:
    """True when the frame is dark enough that a driver write is warranted."""
    try:
        v = float(median_luma)
    except (TypeError, ValueError):
        return False
    return 0.0 <= v < DARK_LUMA


def plan_lift(
    procamp: Optional[Dict[str, Any]],
    already_tried: Optional[set] = None,
) -> Optional[Tuple[str, int, int, int]]:
    """Propose the next light-recovery write.

    Returns ``(property_name, dshow_index, target_value, original_value)``
    or None when there is nothing safe left to try.

    A property is skipped when the camera does not expose it, when the
    driver currently owns it (Auto), when it is already at or past the
    ceiling, or when the computed target would not actually move it.
    """
    tried = set(already_tried or ())
    for name in LIFT_ORDER:
        if name in tried:
            continue
        e = _entry(procamp, name)
        if not e:
            continue  # this camera does not expose it -- the field rig has no Gain
        try:
            cur = int(e["value"])
            lo = int(e["min"])
            hi = int(e["max"])
            dflt = int(e["default"])
            idx = int(e["index"])
            flags = int(e.get("flags", FLAG_MANUAL))
        except (KeyError, TypeError, ValueError):
            continue
        if hi <= lo:
            continue
        if flags == FLAG_AUTO:
            # The driver is moving this with the light. Writing a manual
            # value would take ownership of it, which is a bigger promise
            # than "we brightened the preview".
            continue
        ceiling = dflt + CEILING_FRACTION * (hi - dflt)
        ceiling = max(lo, min(hi, ceiling))
        if cur >= ceiling:
            continue
        target = int(round(cur + LIFT_FRACTION * (ceiling - cur)))
        target = max(lo, min(hi, target))
        if target == cur:
            continue
        return (name, idx, target, cur)
    return None


def verdict(before_luma: Optional[float], after_luma: Optional[float]) -> bool:
    """True when a lift moved the picture enough to justify keeping it.

    The caller measures the frame before and after. This is what makes
    the module safe on a driver whose Gamma direction is unknown: a write
    that darkened the frame, or barely moved it, is reverted rather than
    defended.
    """
    try:
        b = float(before_luma)
        a = float(after_luma)
    except (TypeError, ValueError):
        return False
    return (a - b) >= MIN_USEFUL_GAIN
