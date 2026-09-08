"""Throwaway live diagnostic for the face-exclusion hand filter.

Run it, point your webcam at yourself, and watch:
  * CYAN box  = a face the BlazeFace detector found (confirms face detection
                is working on you).
  * GREEN hand box + dots = a hand the filter KEEPS (treated as a real hand).
  * RED hand box + dots   = a hand the filter would DROP as a face-latch
                            (its centre is inside a face box and it's mostly
                            contained by it).

To see the filter actually catch something, try to provoke a false hand:
hold a loose hand shape right over your face, or tilt your head — if the hand
model latches onto your face, that detection will show up RED (dropped).
A normal hand held out to control stays GREEN.

Usage:
    .venv\\Scripts\\python.exe face_filter_demo.py            # camera 0
    .venv\\Scripts\\python.exe face_filter_demo.py 1          # camera index 1

Press  q  or  Esc  to quit. This file is a dev tool — not shipped, safe to delete.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from src.hgr.gesture.tracking.face_filter import FaceExclusionFilter


def main() -> int:
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {cam_index}. Try a different index, e.g. `python face_filter_demo.py 1`.")
        return 1

    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Same thresholds the real pipeline uses.
    flt = FaceExclusionFilter()
    flt._ensure_detector()

    print("Running. CYAN=face, GREEN=hand kept, RED=hand dropped. Press q/Esc to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, matching the app's pipeline
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Faces (refresh every frame here — it's just a demo).
        flt._refresh_faces(rgb)
        faces = list(flt._cached_faces)
        for (fx0, fy0, fx1, fy1) in faces:
            cv2.rectangle(frame, (int(fx0 * w), int(fy0 * h)), (int(fx1 * w), int(fy1 * h)), (255, 255, 0), 2)
            cv2.putText(frame, "face", (int(fx0 * w), max(0, int(fy0 * h) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Hands (raw model output), colour by the filter's keep/drop decision.
        result = hands.process(rgb)
        kept = dropped = 0
        for hand_landmarks in (getattr(result, "multi_hand_landmarks", None) or []):
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            hb = (float(raw[:, 0].min()), float(raw[:, 1].min()),
                  float(raw[:, 0].max()), float(raw[:, 1].max()))
            cx = (hb[0] + hb[2]) * 0.5
            cy = (hb[1] + hb[3]) * 0.5
            # New logic: a real-looking hand is NEVER dropped, even over a face.
            is_hand = flt._looks_like_hand(raw)
            on_face = False
            for fb in faces:
                inside = fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]
                if inside and flt._containment_fraction(hb, fb) >= flt._containment:
                    on_face = True
                    break
            drop = (not is_hand) and on_face
            color = (0, 0, 255) if drop else (0, 255, 0)
            if drop:
                dropped += 1
            else:
                kept += 1
            cv2.rectangle(frame, (int(hb[0] * w), int(hb[1] * h)), (int(hb[2] * w), int(hb[3] * h)), color, 2)
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)
            verdict = "DROPPED (face-latch)" if drop else ("kept (hand)" if is_hand else "kept (on-face? no)")
            cv2.putText(frame, f"{verdict}  hand-shape={'Y' if is_hand else 'N'} on-face={'Y' if on_face else 'N'}",
                        (int(hb[0] * w), max(0, int(hb[1] * h) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"faces:{len(faces)}  hands kept:{kept}  dropped:{dropped}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Face filter demo  (q/Esc to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    flt.close()
    hands.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
