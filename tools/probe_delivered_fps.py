"""Measure the fps the camera DRIVER actually delivers, independent of Touchless.

Run with Touchless closed:

    python tools/probe_delivered_fps.py            # read-only (default)
    python tools/probe_delivered_fps.py --index 1  # another camera

Why: the Razer Kiyo Pro advertises 60 fps in every mode but silently
delivers ~20-25 fps while its driver is in auto-exposure / HDR mode.
When Touchless "feels slow", run this first. If the DELIVERED number
is ~20 here, no version of the app can go faster: fix the driver state
in Razer Synapse (HDR off, exposure Manual) and re-run until this
reports ~60. Recorded 2026-09-24 after an app-side AUTO_EXPOSURE write
(since removed) knocked the reference rig's Kiyo Pro into this state.

Optional, EXPLICIT opt-in (never done by the app itself):

    python tools/probe_delivered_fps.py --set-manual-exposure -5

writes CAP_PROP_AUTO_EXPOSURE=0.25 (manual) + CAP_PROP_EXPOSURE=<value>
through DirectShow, then re-measures. Synapse is the source of truth;
use this only when Synapse is unavailable. Some UVC drivers ignore the
AUTO_EXPOSURE write; the readback line tells you what latched.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2


def _measure(index: int, w: int, h: int, seconds: float = 4.0) -> str:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        cap.release()
        return f"{w}x{h}: could not open camera index {index}"
    ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    ex = cap.get(cv2.CAP_PROP_EXPOSURE)
    neg = cap.get(cv2.CAP_PROP_FPS)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(15):
        cap.read()
    n = 0
    t0 = time.perf_counter()
    frame = None
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if ok:
            n += 1
    dt = time.perf_counter() - t0
    mean = float(frame.mean()) if frame is not None else -1.0
    cap.release()
    time.sleep(0.5)
    return (
        f"{aw}x{ah}  negotiated={neg:.0f} fps  DELIVERED={n / dt:.1f} fps  "
        f"AUTO_EXPOSURE={ae} EXPOSURE={ex} frame_mean={mean:.0f}"
    )


def _set_manual(index: int, exposure: float) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"could not open camera index {index}")
        return
    before = (cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_EXPOSURE))
    ok1 = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    ok2 = cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    time.sleep(0.4)
    after = (cap.get(cv2.CAP_PROP_AUTO_EXPOSURE), cap.get(cv2.CAP_PROP_EXPOSURE))
    cap.release()
    time.sleep(0.5)
    print(f"manual-exposure write: auto_ok={ok1} exp_ok={ok2} before={before} after={after}")


def _set_auto(index: int) -> None:
    """DirectShow: CAP_PROP_AUTO_EXPOSURE rounds to 1 -> Auto (OpenCV writes
    the driver's DEFAULT exposure with the Auto flag). 0.75 is that value.
    Do NOT follow it with 3.0 - on DirectShow 3.0 means Manual."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"could not open camera index {index}")
        return
    before = cap.get(cv2.CAP_PROP_EXPOSURE)
    ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    time.sleep(0.4)
    after = cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.release()
    time.sleep(0.5)
    print(f"auto-exposure write: ok={ok} exposure before={before} after={after}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--seconds", type=float, default=4.0, help="measure window per mode")
    ap.add_argument(
        "--set-manual-exposure", type=float, default=None, metavar="EXP",
        help="EXPLICIT opt-in: write AUTO_EXPOSURE=0.25 (manual) + EXPOSURE=EXP, then re-measure",
    )
    ap.add_argument(
        "--set-auto", action="store_true",
        help="EXPLICIT opt-in: return the driver to AUTO exposure (DirectShow 0.75), then re-measure",
    )
    args = ap.parse_args()
    print("Close Touchless before running. Read-only unless --set-manual-exposure / --set-auto is given.")
    for w, h in ((1280, 720), (640, 480)):
        print(_measure(args.index, w, h, args.seconds))
    if args.set_auto:
        _set_auto(args.index)
        for w, h in ((1280, 720), (640, 480)):
            print("after auto: " + _measure(args.index, w, h, args.seconds))
    if args.set_manual_exposure is not None:
        _set_manual(args.index, args.set_manual_exposure)
        for w, h in ((1280, 720), (640, 480)):
            print("after manual: " + _measure(args.index, w, h, args.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Author: Konstantin Markov
