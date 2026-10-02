"""Pick a short-shutter exposure that suits the camera in front of us.

Background
----------
The r49 short-shutter trick writes a manual ``CAP_PROP_EXPOSURE`` so a
cheap UVC webcam stops stretching its shutter in a dim room. That works:
the 2026-07-27 A/B measured 2-2.5x the delivered fps on a generic Realtek
sensor. But the value it writes has always been the literal ``-6.0``, and
that constant was tuned against a 60 fps camera.

DirectShow expresses ``CAP_PROP_EXPOSURE`` as log2(seconds), so:

===========  ==============  =================
 exposure     shutter open    fps it can reach
===========  ==============  =================
 -4.0         62.50 ms        16
 -5.0         31.25 ms        32
 -6.0         15.62 ms        64
===========  ==============  =================

A camera cannot deliver N fps while its shutter is open longer than 1/N,
so the *shortest useful* exposure is the longest one that still fits a
frame period. Going shorter than that buys frames the camera will never
produce -- it is pure light thrown away.

That is exactly what happened on the field rig. Its webcam advertises
``mjpeg 640x480@30`` (one frame every 33.3 ms), and we wrote -6.0, a
15.6 ms shutter. Half the light budget was discarded for a 64 fps
ceiling on a camera that stops at 30, which is why the preview came back
at a luma median of 10/255. At -5.0 the same camera runs the same 30 fps
with twice the light.

The 60 fps dev-rig camera still resolves to -6.0, so this is a no-op
there by construction.

Pure module: no cv2, no Qt, no I/O. Unit-testable without a camera.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# The historical constant. Returned whenever we cannot do better, so a
# camera we know nothing about behaves exactly as it did before r23.
DEFAULT_EXPOSURE = -6.0

# Never hand the driver a shutter shorter than this: below -6 the
# exposure buys nothing on any webcam we ship against and only darkens.
MIN_EXPOSURE = -6.0

# Never hand it one longer than this either. -4.0 is the value the
# release paths already trust as "bright manual", and it caps the sensor
# at 16 fps, so it is the furthest we are willing to trade fps for light.
MAX_EXPOSURE = -4.0


def exposure_for_fps(target_fps: Optional[float]) -> float:
    """Longest whole-stop exposure that still fits one frame at ``target_fps``.

    Returns :data:`DEFAULT_EXPOSURE` when ``target_fps`` is missing or
    nonsensical, so an unknown camera keeps the pre-r23 behaviour.
    """
    try:
        fps = float(target_fps)
    except (TypeError, ValueError):
        return DEFAULT_EXPOSURE
    if not math.isfinite(fps) or fps <= 0.0:
        return DEFAULT_EXPOSURE
    # log2 of the frame period, floored to a whole stop. Flooring (not
    # rounding) is what keeps the shutter inside the frame period: at
    # 30 fps the exact answer is -4.91, and -4.0 would overrun it.
    stop = math.floor(math.log2(1.0 / fps))
    return float(min(MAX_EXPOSURE, max(MIN_EXPOSURE, stop)))


def advertised_fps_for(
    modes: Optional[List[Dict[str, Any]]],
    width: int,
    height: int,
) -> Optional[float]:
    """Best advertised fps at ``width`` x ``height``, or None if unknown.

    ``modes`` is what :func:`camera_capabilities.parse_list_options`
    returns. Any mode at that resolution counts, compressed or not: the
    question here is what the sensor can time, not which pin we open.
    """
    if not modes:
        return None
    best: Optional[float] = None
    for mode in modes:
        try:
            if int(mode["width"]) != int(width) or int(mode["height"]) != int(height):
                continue
            fps = float(mode["max_fps"])
        except (KeyError, TypeError, ValueError):
            continue
        if fps > 0.0 and (best is None or fps > best):
            best = fps
    return best


def exposure_for_camera(
    modes: Optional[List[Dict[str, Any]]],
    width: int,
    height: int,
) -> float:
    """Short-shutter exposure to write for this camera at this size."""
    return exposure_for_fps(advertised_fps_for(modes, width, height))
