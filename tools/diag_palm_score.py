"""Diagnostic: report raw palm-detector scores for a still image.

Usage:
    python tools/diag_palm_score.py <image> [crop_x0 crop_y0 crop_x1 crop_y1]

Prints the top decoded palm candidates BEFORE any score gate so we can
see where a given pose lands relative to min_detection_confidence.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, "src")

from hgr.gesture.tracking.onnx_runtime import (  # noqa: E402
    _OnnxPalmDetector,
    build_onnx_directml_runtime,
)


def main() -> int:
    path = sys.argv[1]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"could not read {path}")
        return 1
    if len(sys.argv) >= 6:
        x0, y0, x1, y1 = (int(v) for v in sys.argv[2:6])
        img = img[y0:y1, x0:x1]
    print(f"input crop: {img.shape[1]}x{img.shape[0]}")

    shim = build_onnx_directml_runtime()
    if shim is None:
        print("ONNX/DirectML runtime unavailable")
        return 2

    for width in (640, 960):
        scale = width / float(img.shape[1])
        frame = cv2.resize(
            img,
            (width, max(1, int(round(img.shape[0] * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = _OnnxPalmDetector(
            shim._palm_session,
            shim._anchors,
            score_threshold=0.05,
            nms_threshold=0.3,
            top_k=16,
        )
        palms = det.detect(rgb)
        palms.sort(key=lambda p: -p["score"])
        top = [round(float(p["score"]), 3) for p in palms[:5]]
        print(f"process_width={width}: {len(palms)} candidate(s), top scores={top}")
        for p in palms[:3]:
            b = np.asarray(p["bbox"], dtype=float) / scale
            print(f"    score={p['score']:.3f} bbox={[round(v, 1) for v in b.tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Author: Konstantin Markov
