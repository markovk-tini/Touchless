"""Render a top-down path diagram from a saved dynamic gesture.

Two views:
  * takes  — each take's wrist path (thin colored trails) + fingertip
             trails from take 1 + start/mid/end skeletons
  * average — thick mean wrist path + thick mean palm-center path,
              with faint individual takes underneath

Used by the Custom Gestures "Paths" button.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe landmark indices.
_WRIST = 0
_MIDDLE_MCP = 9
_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + MCPs ≈ palm center
_FINGERTIPS = (4, 8, 12, 16, 20)

_HAND_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

_TAKE_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (80, 220, 80),
    (80, 180, 255),
    (255, 160, 60),
    (200, 80, 255),
    (60, 220, 220),
)

_TIP_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (180, 140, 255),  # thumb
    (80, 220, 80),    # index
    (80, 200, 255),   # middle
    (255, 160, 60),   # ring
    (200, 80, 255),   # pinky
)

_POSE_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (160, 160, 160),
    (90, 200, 255),
    (60, 60, 255),
)

_AVG_WRIST_BGR = (255, 255, 255)
_AVG_PALM_BGR = (40, 200, 255)


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _mean_path(paths: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """Frame-wise mean of equal-length (T, 2) paths. Returns None if empty."""
    if not paths:
        return None
    stacked = np.stack([np.asarray(p, dtype=np.float32) for p in paths], axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def _resample_xy(path: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a (T, 2) path to `length` frames."""
    src = np.asarray(path, dtype=np.float32)
    if src.shape[0] == length:
        return src.copy()
    if src.shape[0] < 2:
        return np.repeat(src[:1], length, axis=0)
    t_src = np.linspace(0.0, 1.0, src.shape[0], dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, length, dtype=np.float64)
    out = np.empty((length, 2), dtype=np.float32)
    out[:, 0] = np.interp(t_dst, t_src, src[:, 0])
    out[:, 1] = np.interp(t_dst, t_src, src[:, 1])
    return out


def _absolute_keypoint_paths(
    wrists: Sequence[np.ndarray],
    samples: Sequence[np.ndarray],
    key_indices: Sequence[int],
    landmark_id: int,
) -> List[np.ndarray]:
    """Build absolute (T, 2) trails for one landmark across takes."""
    if landmark_id not in key_indices:
        return []
    col = list(key_indices).index(landmark_id)
    out: List[np.ndarray] = []
    for wrist, sample in zip(wrists, samples):
        T = min(wrist.shape[0], sample.shape[0])
        if T < 2 or col >= sample.shape[1]:
            continue
        abs_xy = wrist[:T] + sample[:T, col, :2]
        out.append(abs_xy.astype(np.float32))
    return out


def _palm_center_paths(
    wrists: Sequence[np.ndarray],
    samples: Sequence[np.ndarray],
    key_indices: Sequence[int],
) -> List[np.ndarray]:
    """Approximate palm-center trail: mean of wrist + available MCPs.

    Falls back to middle-MCP if present, else wrist alone.
    """
    available = [i for i in _PALM_LANDMARKS if i in key_indices or i == _WRIST]
    if not available:
        return [w.copy() for w in wrists]

    paths: List[np.ndarray] = []
    for wrist, sample in zip(wrists, samples):
        T = min(wrist.shape[0], sample.shape[0])
        if T < 2:
            continue
        pts = []
        for lm in available:
            if lm == _WRIST:
                pts.append(wrist[:T])
                continue
            if lm not in key_indices:
                continue
            col = list(key_indices).index(lm)
            if col >= sample.shape[1]:
                continue
            pts.append(wrist[:T] + sample[:T, col, :2])
        if not pts:
            paths.append(wrist[:T].copy())
        else:
            paths.append(np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32))
    return paths


def render_dynamic_path_preview(
    gesture: Any,
    *,
    width: int = 720,
    height: int = 540,
    margin: int = 48,
    mode: str = "takes",
) -> np.ndarray:
    """Return a BGR image diagram of the gesture's recorded paths.

    `mode`:
      * "takes"   — individual wrist trails + fingertip trails + poses
      * "average" — thick mean wrist + thick mean palm center
    """
    import cv2

    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    name = str(getattr(gesture, "name", "gesture") or "gesture")
    kind = str(getattr(gesture, "kind", "") or "")
    if kind and kind != "dynamic":
        _put(canvas, f"{name} is a static pose — no motion path.", (24, 40), (180, 180, 180))
        return canvas

    wrists: List[np.ndarray] = []
    for w in (getattr(gesture, "wrist_trajectories", None) or []):
        arr = _as_array(w)
        if arr is not None and arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
            wrists.append(arr[:, :2].astype(np.float32))

    key_indices = [int(i) for i in (getattr(gesture, "key_point_indices", None) or [])]
    samples: List[np.ndarray] = []
    for s in (getattr(gesture, "sample_trajectories", None) or []):
        arr = _as_array(s)
        if arr is not None and arr.ndim == 3 and arr.shape[0] >= 2:
            samples.append(arr.astype(np.float32))

    if not wrists and not samples:
        _put(canvas, f"{name}: no stored trajectories.", (24, 40), (180, 180, 180))
        _put(
            canvas,
            "Re-record this dynamic gesture to capture path data.",
            (24, 68),
            (140, 140, 140),
        )
        return canvas

    # Align take counts.
    n = min(len(wrists), len(samples)) if (wrists and samples) else len(wrists) or len(samples)
    if wrists and samples:
        wrists = wrists[:n]
        samples = samples[:n]
    elif not wrists and samples:
        # Fall back: landmark 0 column as "wrist".
        wrists = [s[:, 0, :2].copy() for s in samples]

    tip_paths: List[List[np.ndarray]] = []
    for tip in _FINGERTIPS:
        tip_paths.append(_absolute_keypoint_paths(wrists, samples, key_indices, tip))
    palm_paths = _palm_center_paths(wrists, samples, key_indices)

    # Fit canvas to everything we might draw.
    all_parts: List[np.ndarray] = list(wrists) + list(palm_paths)
    for group in tip_paths:
        all_parts.extend(group)
    if not all_parts:
        _put(canvas, f"{name}: no drawable paths.", (24, 40), (180, 180, 180))
        return canvas
    all_xy = np.concatenate(all_parts, axis=0)
    min_xy = all_xy.min(axis=0)
    max_xy = all_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    usable_w = float(width - 2 * margin)
    usable_h = float(height - 2 * margin - 72)
    scale = min(usable_w / float(span[0]), usable_h / float(span[1]))
    origin_x = margin + (usable_w - span[0] * scale) * 0.5
    origin_y = 72 + margin + (usable_h - span[1] * scale) * 0.5

    def to_px(xy: np.ndarray) -> Tuple[int, int]:
        x = int(round(origin_x + (float(xy[0]) - float(min_xy[0])) * scale))
        y = int(round(origin_y + (float(xy[1]) - float(min_xy[1])) * scale))
        return x, y

    def draw_polyline(
        path: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        *,
        mark_ends: bool = False,
    ) -> None:
        pts = [to_px(p) for p in path]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
        if mark_ends and pts:
            cv2.circle(canvas, pts[0], max(5, thickness + 2), (220, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], max(4, thickness + 1), color, -1, cv2.LINE_AA)

    intent_mag = float(getattr(gesture, "intent_magnitude", 0.0) or 0.0)
    thr = getattr(gesture, "match_threshold", None)
    thr_txt = f"{float(thr):.2f}" if thr is not None else "default"
    features = getattr(gesture, "sample_features", None) or []
    mode_label = "AVERAGE" if mode == "average" else "TAKES"
    _put(canvas, f"Paths: {name}  [{mode_label}]", (16, 28), (235, 235, 235), scale=0.7)
    _put(
        canvas,
        f"takes={len(wrists)}  intent_mag={intent_mag:.2f}  "
        f"threshold={thr_txt}  spring_features={'yes' if features else 'NO'}",
        (16, 50),
        (160, 160, 160),
        scale=0.48,
    )
    if not features:
        _put(
            canvas,
            "WARNING: no SPRING features — looping gestures cannot fire. Re-record.",
            (16, 70),
            (60, 60, 255),
            scale=0.45,
        )

    if mode == "average":
        # Faint individuals underneath.
        for path in wrists:
            draw_polyline(path, (70, 70, 70), 1)
        target_len = int(np.median([p.shape[0] for p in wrists])) if wrists else 32
        target_len = max(8, target_len)
        aligned = [_resample_xy(p, target_len) for p in wrists]
        avg_wrist = _mean_path(aligned)
        if avg_wrist is not None:
            draw_polyline(avg_wrist, _AVG_WRIST_BGR, 5, mark_ends=True)
            _put(canvas, "thick white = mean wrist", (16, height - 36), _AVG_WRIST_BGR, scale=0.42)
        if palm_paths:
            aligned_palm = [_resample_xy(p, target_len) for p in palm_paths]
            avg_palm = _mean_path(aligned_palm)
            if avg_palm is not None:
                draw_polyline(avg_palm, _AVG_PALM_BGR, 4, mark_ends=True)
                _put(
                    canvas,
                    "thick amber = mean palm center",
                    (260, height - 36),
                    _AVG_PALM_BGR,
                    scale=0.42,
                )
    else:
        # Individual wrist takes.
        for i, path in enumerate(wrists):
            color = _TAKE_COLORS_BGR[i % len(_TAKE_COLORS_BGR)]
            draw_polyline(path, color, 2, mark_ends=True)
            pts0 = to_px(path[0])
            cv2.putText(
                canvas, f"T{i + 1}", (pts0[0] + 8, pts0[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        # Fingertip trails from take 1 (absolute = wrist + relative tip).
        if wrists and samples and key_indices:
            for tip_i, tip_lm in enumerate(_FINGERTIPS):
                group = tip_paths[tip_i]
                if not group:
                    continue
                draw_polyline(group[0], _TIP_COLORS_BGR[tip_i], 1)

            # Start / mid / end skeletons on take 1.
            sample0 = samples[0]
            wrist0 = wrists[0]
            T = min(sample0.shape[0], wrist0.shape[0])
            frames_to_draw = (
                (0, "start", _POSE_COLORS_BGR[0]),
                (max(0, T // 2), "mid", _POSE_COLORS_BGR[1]),
                (max(0, T - 1), "end", _POSE_COLORS_BGR[2]),
            )
            index_of = {idx: k for k, idx in enumerate(key_indices)}
            for frame_i, label, color in frames_to_draw:
                wrist_xy = wrist0[frame_i]
                pts21 = np.repeat(wrist_xy.reshape(1, 2), 21, axis=0)
                for lm_idx, col in index_of.items():
                    if lm_idx < 0 or lm_idx >= 21 or col >= sample0.shape[1]:
                        continue
                    pts21[lm_idx] = wrist_xy + sample0[frame_i, col, :2]
                px = [to_px(p) for p in pts21]
                for a, b in _HAND_EDGES:
                    cv2.line(canvas, px[a], px[b], color, 1, cv2.LINE_AA)
                for p in px:
                    cv2.circle(canvas, p, 2, color, -1, cv2.LINE_AA)
                wx, wy = px[0]
                cv2.putText(
                    canvas, label, (wx + 10, wy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )

        _put(
            canvas,
            "thin tip trails = take-1 fingertips  •  colored = wrist takes",
            (16, height - 36),
            (120, 120, 120),
            scale=0.42,
        )

    _put(
        canvas,
        "camera view (x right, y down)  •  ring = start  •  filled = end",
        (16, height - 16),
        (120, 120, 120),
        scale=0.42,
    )
    return canvas


def _put(
    canvas: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    scale: float = 0.55,
) -> None:
    import cv2
    cv2.putText(
        canvas, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA,
    )


# Author: Konstantin Markov
