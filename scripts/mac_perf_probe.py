#!/usr/bin/env python3
"""macOS performance probe — isolates the gesture-pipeline bottleneck.

Run in a Mac Terminal (needs Camera permission for Terminal):
    cd ~/Touchless && ./.venv-mac/bin/python scripts/mac_perf_probe.py

Measures, in isolation:
  1. cv2/AVFoundation camera capture fps at the camera's DEFAULT resolution
  2. camera capture fps after requesting 1280x720 and 640x480
  3. MediaPipe Hands inference fps at 640x480 and 1280x720
  4. BGR->RGB convert + resize cost

Tells us whether 3.3 fps is capture-bound (AVFoundation) or inference-bound.
Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import time
import cv2
import numpy as np


def _measure_capture(cap, n=60):
    for _ in range(6):  # warmup
        cap.read()
    t0 = time.time()
    got = 0
    shape = None
    for _ in range(n):
        ok, frame = cap.read()
        if ok and frame is not None:
            got += 1
            shape = frame.shape
    dt = time.time() - t0
    return (got / dt if dt > 0 else 0.0), shape


def main() -> int:
    print("=" * 60)
    print(" Touchless macOS performance probe")
    print("=" * 60)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("ERROR: could not open camera 0 (grant Terminal Camera permission).")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"default capture resolution: {w}x{h}, prop_fps={cap.get(cv2.CAP_PROP_FPS)}")
    fps, shape = _measure_capture(cap)
    print(f"[1] capture @ default:        {fps:5.1f} fps  (frame {shape})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps, shape = _measure_capture(cap)
    print(f"[2] capture @ req 1280x720:   {fps:5.1f} fps  (frame {shape})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps, shape = _measure_capture(cap)
    print(f"[3] capture @ req 640x480:    {fps:5.1f} fps  (frame {shape})")
    cap.release()

    # MediaPipe inference (no camera) at two resolutions
    try:
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(model_complexity=1, max_num_hands=2)
        for res in ((640, 480), (1280, 720)):
            frame = (np.random.rand(res[1], res[0], 3) * 255).astype("uint8")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _ in range(3):
                hands.process(rgb)
            t0 = time.time()
            for _ in range(30):
                hands.process(rgb)
            dt = time.time() - t0
            print(f"[4] mediapipe hands @ {res[0]}x{res[1]}: {30 / dt:5.1f} fps")
        hands.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[4] mediapipe probe failed: {type(exc).__name__}: {exc}")

    # Convert + resize cost on a 1080p frame
    big = (np.random.rand(1080, 1920, 3) * 255).astype("uint8")
    t0 = time.time()
    for _ in range(60):
        small = cv2.resize(big, (640, 360))
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    print(f"[5] resize1080->640 + BGR2RGB x60: {(time.time()-t0)*1000/60:.1f} ms each")

    print("\nInterpretation: if [1]/[2] are ~3 fps but [4] is 20+ fps, the bottleneck")
    print("is cv2/AVFoundation capture (fix: cap resolution / faster capture path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
