"""Ask a camera what it supports, once, instead of discovering it by
failing.

Why this exists
---------------
Entering Lite or GPU mode used to work by trial and error: request MJPG
at 60 fps, and if the driver refused, request 30, and if that hung, retry
once. On a webcam with no suitable MJPG pin that whole cascade is doomed
from the start, and it costs several seconds of frozen UI plus three
antivirus-visible ffmpeg launches, every single time, because nothing
remembered the outcome.

DirectShow will simply tell us. `ffmpeg -list_options true` enumerates
every pin the driver advertises, with its pixel format or codec, its
resolutions and its maximum frame rate, without opening a capture. One
call, about five seconds, and the answer is definitive.

That turns the decision into a lookup:

  * the camera advertises MJPG at the size we want -> open it, asking for
    the frame rate it actually claims rather than a hopeful 60
  * it advertises MJPG only at other sizes -> use one of those
  * it advertises no MJPG at all -> skip the fast path entirely, with no
    failed attempts and no freeze

Everything here is pure except `probe_camera_capabilities`, which is the
one function that runs ffmpeg.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional

# "vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=60.0002"
# "pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30"
_MODE_RE = re.compile(
    r"(?:vcodec|pixel_format)\s*=\s*(?P<fmt>\S+).*?"
    r"max\s+s\s*=\s*(?P<w>\d+)x(?P<h>\d+)\s+fps\s*=\s*(?P<fps>[\d.]+)",
    re.IGNORECASE,
)

#: Formats that give us a compressed stream. Anything else means the
#: driver would hand us raw frames, which is the slow path we are trying
#: to avoid.
COMPRESSED_FORMATS = ("mjpeg", "mjpg")

#: Raw pin formats, for `compressed_is_worth_it` only. Deliberately NOT
#: folded into COMPRESSED_FORMATS' complement: a format in neither list
#: (h264, say) is evidence for neither side and must be ignored rather
#: than silently counted as raw. Widening COMPRESSED_FORMATS itself would
#: change which pin `best_compressed_mode` asks ffmpeg for, which is a
#: different and much riskier decision.
UNCOMPRESSED_FORMATS = (
    "yuyv422", "yuv420p", "nv12", "uyvy422", "rgb24", "bgr24", "gray",
)


def parse_list_options(text: str) -> List[Dict[str, Any]]:
    """Turn `ffmpeg -list_options` output into a list of modes.

    Each mode is `{"format", "width", "height", "max_fps"}`. Duplicate
    entries (ffmpeg prints a colour-range variant of each pin) collapse
    to one, keeping the highest frame rate seen for that combination.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    for line in str(text or "").splitlines():
        m = _MODE_RE.search(line)
        if not m:
            continue
        try:
            w = int(m.group("w"))
            h = int(m.group("h"))
            fps = float(m.group("fps"))
        except Exception:
            continue
        if w <= 0 or h <= 0 or fps <= 0:
            continue
        fmt = str(m.group("fmt") or "").strip().lower()
        key = (fmt, w, h)
        prev = best.get(key)
        if prev is None or fps > prev["max_fps"]:
            best[key] = {"format": fmt, "width": w, "height": h, "max_fps": fps}
    return sorted(
        best.values(), key=lambda d: (d["format"], -d["width"], -d["max_fps"])
    )


def supports_compressed(modes: Optional[List[Dict[str, Any]]]) -> bool:
    """Does this camera offer a compressed stream at any size?"""
    for m in modes or ():
        if str(m.get("format", "")).lower() in COMPRESSED_FORMATS:
            return True
    return False


def compressed_is_worth_it(
    modes: Optional[List[Dict[str, Any]]],
    want_w: int,
    want_h: int,
) -> Optional[bool]:
    """Would the compressed pin actually beat the uncompressed one here?

    Returns True when compressed is strictly faster at this size, False
    when it is not, and None when we have not learned enough to say.

    Why this exists. Taking the ffmpeg-MJPG path is not free: it releases
    a working capture, sleeps 600 ms for the DirectShow graph to tear
    down, spawns `ffmpeg` (which antivirus prompts on), and on failure
    retries and falls back -- 15-20 s of frozen UI on the field rig, per
    mode swap. That is worth paying when MJPG lifts a YUY2 bandwidth
    ceiling, which is the case it was built for: at 1280x720 a cheap UVC
    typically offers `mjpeg 30` against `yuyv422 8`.

    It is NOT worth paying when both pins advertise the same rate. The
    field camera reports `mjpeg 640x480@30` AND `yuyv422 640x480@30`, so
    at the size Lite and GPU actually request the entire cascade buys
    exactly zero fps -- and then fails anyway, leaving the camera worse
    off than Default. Nothing in the app had ever asked the question,
    even though the capability probe had already collected the answer.

    Only compares at the requested size. A camera that is faster
    compressed at 720p but equal at 480p should keep the fast path at
    720p, and that is what per-size comparison gives.
    """
    best_c = 0.0
    best_u = 0.0
    seen_c = seen_u = False
    for m in modes or ():
        try:
            if int(m.get("width", 0)) != int(want_w):
                continue
            if int(m.get("height", 0)) != int(want_h):
                continue
            fps = float(m.get("max_fps", 0.0))
        except (TypeError, ValueError):
            continue
        if fps <= 0:
            continue
        if str(m.get("format", "")).lower() in COMPRESSED_FORMATS:
            seen_c = True
            best_c = max(best_c, fps)
        elif str(m.get("format", "")).lower() in UNCOMPRESSED_FORMATS:
            seen_u = True
            best_u = max(best_u, fps)
    if not seen_c or not seen_u:
        # One side unknown -> no opinion. Never block the fast path on
        # missing evidence; the historical behaviour is to try ffmpeg.
        return None
    # A hair of tolerance: 30.0 vs 30.0002 is the same pin rate written
    # two ways, not a reason to pay 15 s.
    return bool(best_c > best_u + 0.5)


def best_compressed_mode(
    modes: Optional[List[Dict[str, Any]]],
    want_w: int,
    want_h: int,
    want_fps: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """Pick the compressed mode to actually ask for, or None.

    Prefers the exact requested size. Falls back to the largest
    compressed mode that is no bigger than the request, so we never
    silently upgrade someone to a heavier stream than they asked for.
    The returned `fps` is clamped to what the driver claims, which is the
    whole point: asking a 30 fps pin for 60 is what produced the "Could
    not set video options" rejection in the field.
    """
    compressed = [
        m for m in (modes or ())
        if str(m.get("format", "")).lower() in COMPRESSED_FORMATS
    ]
    if not compressed:
        return None
    try:
        want_w = int(want_w)
        want_h = int(want_h)
    except Exception:
        return None

    exact = [m for m in compressed if m["width"] == want_w and m["height"] == want_h]
    pool = exact or [
        m for m in compressed
        if m["width"] <= want_w and m["height"] <= want_h
    ]
    if not pool:
        return None
    chosen = max(pool, key=lambda m: (m["width"] * m["height"], m["max_fps"]))
    fps = min(float(want_fps), float(chosen["max_fps"]))
    # Drivers advertise things like 60.0002; asking for that verbatim is
    # asking for trouble, so round down to a whole frame rate.
    fps = float(int(fps)) if fps >= 1 else fps
    return {
        "format": chosen["format"],
        "width": int(chosen["width"]),
        "height": int(chosen["height"]),
        "fps": fps,
        "advertised_max_fps": float(chosen["max_fps"]),
        "exact_size": bool(exact),
    }


def probe_camera_capabilities(
    device_name: str,
    ffmpeg_path: str,
    *,
    timeout: float = 12.0,
    sub_kwargs: Optional[dict] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Run the one ffmpeg call that enumerates a camera's pins.

    Returns the parsed modes, or None when the probe could not run. None
    means "unknown", never "unsupported" -- callers must fall back to
    their previous behaviour rather than treating it as a refusal.
    """
    if not device_name or not ffmpeg_path:
        return None
    cmd = [
        str(ffmpeg_path), "-hide_banner",
        "-f", "dshow", "-list_options", "true",
        "-i", f"video={device_name}",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(timeout),
            **(sub_kwargs or {}),
        )
    except Exception:
        return None
    # ffmpeg writes the enumeration to stderr and always exits non-zero
    # for -list_options, so the return code tells us nothing useful.
    text = (completed.stderr or "") + "\n" + (completed.stdout or "")
    modes = parse_list_options(text)
    return modes or None


def describe(modes: Optional[List[Dict[str, Any]]]) -> str:
    if not modes:
        return "unknown"
    return ", ".join(
        f"{m['format']} {m['width']}x{m['height']}@{m['max_fps']:.0f}"
        for m in modes
    )
