"""Probe what modes the DShow camera advertises. Run with Touchless closed.

Quick usage:
    python tools/probe_camera_modes.py

Prints, for every video device DirectShow enumerates, the exhaustive
list of (pixel_format / vcodec, resolution, fps) tuples the driver
claims to support. If the driver reports mjpeg 1280x720 @ 60 but the
app measures 25 fps at runtime, the 25 fps is 100% driver-side
(auto-exposure / HDR / lighting) — not a code bug.
"""
from __future__ import annotations

import re
import subprocess
import sys

FFMPEG = r"C:\Users\Konstantin Markov\ffmpeg-7.0.2-full_build\bin\ffmpeg.EXE"
CREATE_NO_WINDOW = 0x08000000


def run_ffmpeg(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return "[timeout] ffmpeg took >15s — camera may be held by another app (quit Touchless, Razer Synapse, Windows Camera)"
    except FileNotFoundError:
        return f"[error] ffmpeg not found at {FFMPEG!r}"
    return (proc.stderr or "") + (proc.stdout or "")


def list_video_devices(text: str) -> list[str]:
    # ffmpeg 7.x dropped the "DirectShow video devices" section header
    # and now tags each device line inline: [dshow @ 0x...] "Name" (video)
    # or (audio). Match on the (video) tag directly and skip
    # "Alternative name" rows (which repeat the same device under a
    # PnP path we don't want as the friendly name).
    devices: list[str] = []
    for line in text.splitlines():
        if "Alternative name" in line:
            continue
        if not line.rstrip().endswith("(video)"):
            continue
        match = re.search(r'"([^"]+)"', line)
        if match:
            devices.append(match.group(1))
    return devices


MODE_LINE = re.compile(
    r"(?:pixel_format|vcodec)=(\S+).*?"
    r"min s=(\d+x\d+) fps=([\d.]+).*?"
    r"max s=(\d+x\d+) fps=([\d.]+)"
)


def parse_modes(text: str) -> dict[tuple[str, str], set[float]]:
    modes: dict[tuple[str, str], set[float]] = {}
    for line in text.splitlines():
        match = MODE_LINE.search(line)
        if not match:
            continue
        fmt, _min_size, min_fps, max_size, max_fps = match.groups()
        modes.setdefault((fmt, max_size), set()).update({float(min_fps), float(max_fps)})
    return modes


def summarize(device_name: str, modes: dict[tuple[str, str], set[float]]) -> None:
    if not modes:
        print(f"  {device_name}: no modes parsed (see raw output above)")
        return
    print(f"\n  Advertised modes for {device_name!r}:")
    grouped: dict[str, list[tuple[str, set[float]]]] = {}
    for (fmt, res), fps_set in modes.items():
        grouped.setdefault(fmt, []).append((res, fps_set))
    for fmt in sorted(grouped):
        print(f"    [{fmt}]")
        entries = sorted(grouped[fmt], key=lambda x: -int(x[0].split("x")[0]))
        for res, fps_set in entries:
            fps_str = "/".join(f"{f:g}" for f in sorted(fps_set, reverse=True))
            print(f"      {res} @ {fps_str} fps")


def main() -> int:
    print("Probing DShow video devices via ffmpeg...")
    device_text = run_ffmpeg(["-f", "dshow", "-list_devices", "true", "-i", "dummy"])
    devices = list_video_devices(device_text)
    if not devices:
        print("No video devices enumerated. Raw output:")
        print(device_text)
        return 1
    print(f"\nFound {len(devices)} video device(s): {devices}")
    for name in devices:
        print("\n" + "=" * 72)
        print(f"Querying: {name}")
        print("=" * 72)
        mode_text = run_ffmpeg(
            ["-f", "dshow", "-list_options", "true", "-i", f"video={name}"]
        )
        print("Raw ffmpeg output (stderr):")
        print(mode_text)
        summarize(name, parse_modes(mode_text))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
