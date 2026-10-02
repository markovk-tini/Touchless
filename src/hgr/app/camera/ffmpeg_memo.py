"""r20: per-camera memory of ffmpeg-MJPG capture failures.

Why this exists
---------------
On some webcams the ffmpeg MJPG capture path can never work: the
device simply has no MJPG pin at the size we ask for, so every launch
re-runs the same doomed cascade -- `-list_devices`, then a 60 fps
attempt, then a 30 fps attempt, then a silent-hang retry. On the field
machine that cost ~25 s of frozen UI per mode swap and spawned six
`ffmpeg.exe` processes in two minutes, each of which Norton flagged,
which made Lite mode unusable.

Design constraints (these are the interesting part)
---------------------------------------------------
A naive "it failed once, never try again" memo is actively harmful on
a *good* camera: a Razer Kiyo Pro that happens to be held by Razer
Synapse or the Windows Camera app for a moment fails the same call,
and a one-strike memo would permanently demote it to the slower
OpenCV/YUY2 path with no way back. So:

* **Only HARD failures count.** `FfmpegMjpegCapture` already separates
  a format rejection (non-zero returncode, or a fatal stderr pattern
  such as "could not set video options") from a silent hang, which is
  the signature of another process still holding the DirectShow
  handle. A silent hang or a busy device NEVER records a strike.
* **Two strikes, across sessions.** One hard failure can still be a
  one-off; two means the device really has no such mode.
* **Keyed per device AND per resolution.** A camera with no
  1280x720 MJPG pin may well have a 640x480 one.
* **Self-invalidating.** Keys carry the device name, so plugging in a
  different webcam is simply a different key. `prune` drops entries
  for devices that are no longer present.
* **Kill switch.** `HGR_FFMPEG_MEMO=0` disables the skip entirely.

Everything here is pure: dict in, dict out, no I/O, no Qt, no cv2.
That is what makes it testable without a camera.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

#: Failure classifications a caller may report.
KIND_HARD = "hard"      # ffmpeg rejected the format outright -- counts
KIND_SILENT = "silent"  # startup hung, no stderr -- handle race, never counts
KIND_BUSY = "busy"      # device held by another process -- never counts

#: Hard failures needed before the ffmpeg attempt is skipped.
DEFAULT_STRIKE_THRESHOLD = 2

_ENV_KILL_SWITCH = "HGR_FFMPEG_MEMO"


def memo_enabled() -> bool:
    """False when the user set HGR_FFMPEG_MEMO=0."""
    try:
        raw = os.environ.get(_ENV_KILL_SWITCH)
    except Exception:
        return True
    if raw is None:
        return True
    return str(raw).strip() != "0"


def memo_key(device_name: str, width: int, height: int) -> Optional[str]:
    """`"<lower-cased device name>|<w>x<h>"`, or None if unusable.

    Returning None (rather than raising or inventing a key) means an
    unnamed device can never be memoised, which is the safe direction:
    it gets a full attempt every time.
    """
    name = str(device_name or "").strip().lower()
    if not name:
        return None
    try:
        w = int(width)
        h = int(height)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{name}|{w}x{h}"


def strikes_for(memo: Optional[Dict[str, int]], device_name: str,
                width: int, height: int) -> int:
    key = memo_key(device_name, width, height)
    if not key or not memo:
        return 0
    try:
        return max(0, int(memo.get(key, 0)))
    except Exception:
        return 0


def should_skip_ffmpeg(memo: Optional[Dict[str, int]], device_name: str,
                       width: int, height: int,
                       *, threshold: int = DEFAULT_STRIKE_THRESHOLD) -> bool:
    """True when this device+size has hit the hard-failure threshold."""
    if not memo_enabled():
        return False
    return strikes_for(memo, device_name, width, height) >= int(threshold)


def record_failure(memo: Optional[Dict[str, int]], device_name: str,
                   width: int, height: int, kind: str,
                   *, threshold: int = DEFAULT_STRIKE_THRESHOLD,
                   device_confirmed_free: bool = False):
    """Return `(new_memo, changed)`.

    Normally only `KIND_HARD` increments: a silent hang or a "busy"
    verdict usually means another process holds the DirectShow graph,
    which is temporary and must not disable the fast path for good.

    `device_confirmed_free` overrides that. Pass it when the SAME
    camera was opened successfully moments after the ffmpeg attempt
    failed -- by the OpenCV fallback, for instance. That disproves
    "something else holds it", leaving only our own teardown race or a
    driver that cannot deliver the format it advertises. Both are
    permanent for this camera, so they count.

    Without this, a camera that fails this way is retried on every
    single mode switch, forever: seconds of frozen UI and a fresh
    antivirus prompt each time, which is exactly what the field log
    showed.

    The count is clamped at `threshold` so a long-running session
    cannot inflate it without bound.
    """
    base: Dict[str, int] = dict(memo or {})
    if str(kind) != KIND_HARD and not device_confirmed_free:
        return base, False
    key = memo_key(device_name, width, height)
    if not key:
        return base, False
    current = 0
    try:
        current = max(0, int(base.get(key, 0)))
    except Exception:
        current = 0
    if current >= int(threshold):
        return base, False
    if device_confirmed_free:
        # r24: a DISPROVEN verdict is conclusive on the first occurrence,
        # so it goes straight to the threshold rather than earning one
        # strike at a time.
        #
        # Two strikes exist to protect a good camera that was merely busy
        # for a moment. But `device_confirmed_free` means OpenCV opened
        # THIS camera seconds later -- the proof that nothing was holding
        # it. What remains is our own teardown race or a driver lying
        # about the formats it advertises, and the paragraph above already
        # calls both "permanent for this camera". Making the caller
        # observe it twice just buys a second 20-26 s frozen mode switch
        # and a second antivirus prompt for information we already have.
        #
        # The Kiyo Pro protection is untouched: when Synapse or the
        # Camera app really is holding the device, the OpenCV fallback
        # fails too, so `device_confirmed_free` is False and the ordinary
        # two-strike path still applies.
        base[key] = int(threshold)
        return base, True
    base[key] = current + 1
    return base, True


def clear_device(memo: Optional[Dict[str, int]], device_name: str):
    """Forget every resolution recorded for one device."""
    base: Dict[str, int] = dict(memo or {})
    name = str(device_name or "").strip().lower()
    if not name:
        return base, False
    drop = [k for k in base if k.rsplit("|", 1)[0] == name]
    for k in drop:
        base.pop(k, None)
    return base, bool(drop)


def prune(memo: Optional[Dict[str, int]], known_device_names: Iterable[str]):
    """Drop entries whose device is no longer enumerated.

    Called with an EMPTY device list this is a no-op, because an empty
    enumeration usually means the probe failed, not that every camera
    was unplugged.
    """
    base: Dict[str, int] = dict(memo or {})
    known = {str(n or "").strip().lower() for n in (known_device_names or ()) if str(n or "").strip()}
    if not known or not base:
        return base, False
    drop = [k for k in base if k.rsplit("|", 1)[0] not in known]
    for k in drop:
        base.pop(k, None)
    return base, bool(drop)


def describe(memo: Optional[Dict[str, int]]) -> str:
    if not memo:
        return "empty"
    return ", ".join(f"{k}={v}" for k, v in sorted(memo.items()))
