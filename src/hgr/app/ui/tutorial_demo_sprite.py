"""Play the Control Guide gesture clips (MP4 / PNG) directly in a
small picture-in-picture inset over the live camera frame, with a
thin green border. This replaces the earlier procedural-render
attempts and the segmented-RGBA-cutout approach — both turned out
glitchy. Playing the original photographed clip in a bordered
inset is reliable, recognizable, and zero-magic.

Step → asset mapping:

    swipe_left   → GestureGuide/SwipeLeft.mp4   (looped video)
    swipe_right  → GestureGuide/SwipeRight.mp4  (looped video)
    right_two    → GestureGuide/Two.png         (static)
    right_fist   → GestureGuide/Fist.png        (static)
    wheel_pose   → GestureGuide/Wheel Pose.png  (static)
    left_three   → GestureGuide/Left Three.png  (static)
    left_one     → GestureGuide/Left One.png    (static)

The clip is a single still image or a list of BGR frames, both
played at the configured fps. The runtime composites by drawing a
filled rectangle of the clip and a `border_color` rectangle stroke
around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


_STEP_TO_ASSET = {
    "swipe_left":  "SwipeLeft.mp4",
    "swipe_right": "SwipeRight.mp4",
    "right_two":   "Two.png",
    "right_fist":  "Fist.png",
    "wheel_pose":  "Wheel Pose.png",
    "left_three":  "Left Three.png",
    "left_one":    "Left One.png",
    # Mouse-mode tutorial: shows the activation pose first, then
    # swaps to the live click demo once mouse mode turns on.
    "mouse_clicks": "Mouse Clicks.mp4",
    "mouse_demo":   "Mouse Demo.mp4",
    # Mouse-mode SCROLL phase (added after click practice succeeds):
    # the user has to scroll to the top + bottom bars in the
    # right-side practice arena. The inset clip in the live viewer's
    # top-right shows the two-finger scroll pose so the user can
    # mirror the mechanic without leaving the camera view.
    "tutorial_scrolling": "tutorial_scrolling.mp4",
    # Volume tutorial: animated VolControl clip for the adjust phase,
    # static Mute.png for the mute / unmute phase.
    "volume_pose":  "VolControl.mp4",
    "mute_pose":    "Mute.png",
}


@dataclass(frozen=True)
class ClipPlayback:
    """Per-tick playback state. `frame` is None when the clip
    isn't loaded; `visible` is always True for the picture-in-
    picture inset (no hide/reset cycle needed — the inset is
    always shown while the step is active)."""
    frame: Optional[np.ndarray]
    visible: bool


class TutorialClip:
    """A single Control Guide asset: either a static BGR image or
    a looped BGR frame sequence loaded from an MP4. No alpha
    channel needed — the clip is rendered as an opaque inset with
    a border, not composited onto the underlying camera pixels."""

    __slots__ = ("_frames", "_fps", "_is_video")

    def __init__(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
            cap = cv2.VideoCapture(str(path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if fps <= 1.0:
                fps = 30.0
            frames: List[np.ndarray] = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            self._frames = frames
            self._fps = fps
            self._is_video = True
        else:
            img = cv2.imread(str(path))
            self._frames = [img] if img is not None else []
            self._fps = 1.0
            self._is_video = False

    @property
    def is_loaded(self) -> bool:
        return len(self._frames) > 0

    @property
    def is_static(self) -> bool:
        return not self._is_video

    def at(self, t: float) -> ClipPlayback:
        if not self._frames:
            return ClipPlayback(frame=None, visible=False)
        if not self._is_video:
            return ClipPlayback(frame=self._frames[0], visible=True)
        idx = int(float(t) * self._fps) % len(self._frames)
        return ClipPlayback(frame=self._frames[idx], visible=True)


def resolve_sprite(name: str) -> Optional[TutorialClip]:
    """Locate the Control Guide asset for `name` (source-mode
    `GestureGuide/` first, then PyInstaller bundle equivalent) and
    return a TutorialClip wrapping it. None if nothing is found.

    Name kept as `resolve_sprite` so call sites don't need to
    change — the old RGBA sprite system has been replaced
    in-place."""
    asset_filename = _STEP_TO_ASSET.get(name)
    if asset_filename is None:
        return None

    candidates: List[Path] = []
    src_root = Path(__file__).resolve().parents[4]
    candidates.append(src_root / "GestureGuide" / asset_filename)
    try:
        from ...utils.runtime_paths import app_base_path
        base = Path(app_base_path())
        candidates.append(base / "GestureGuide" / asset_filename)
    except Exception:
        pass

    for path in candidates:
        if not path.is_file():
            continue
        try:
            clip = TutorialClip(path)
        except Exception:
            continue
        if clip.is_loaded:
            return clip
    return None


def composite_sprite(
    frame: np.ndarray,
    clip_frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    *,
    alpha: float = 1.0,
    mirror: bool = False,
    border_color: Tuple[int, int, int] = (181, 232, 28),
    border_thick: int = 3,
) -> None:
    """Draw the clip frame as a picture-in-picture inset at the
    target rect with a thin green border.

    rect is (cx, cy, w, h) where cx/cy is the CENTRE of the inset
    (so call sites that pass main_center keep working). The clip
    is letterbox-fitted into rect with its native aspect ratio
    preserved. A `border_thick`-pixel rectangle in `border_color`
    is drawn just outside the inset.

    `alpha` is honoured for fade-in/out at step boundaries (we
    blend the inset against the underlying frame). `mirror` flips
    the clip horizontally — used when the recorded handedness is
    the opposite of the user's hand for that step.

    The border colour default (181, 232, 28) BGR is the Touchless
    accent — same shade as the rest of the tutorial chrome."""
    if alpha <= 0.001 or clip_frame is None:
        return
    cx, cy, rw, rh = rect
    if rw <= 0 or rh <= 0:
        return

    ch, cw = clip_frame.shape[:2]
    if ch == 0 or cw == 0:
        return
    scale = min(rw / float(cw), rh / float(ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(clip_frame, (new_w, new_h), interpolation=interp)
    if mirror:
        resized = cv2.flip(resized, 1)

    fx0 = cx - new_w // 2
    fy0 = cy - new_h // 2
    fx1 = fx0 + new_w
    fy1 = fy0 + new_h

    fh, fw = frame.shape[:2]
    cx0 = max(0, fx0)
    cy0 = max(0, fy0)
    cx1 = min(fw, fx1)
    cy1 = min(fh, fy1)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    sx0 = cx0 - fx0
    sy0 = cy0 - fy0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    inset = resized[sy0:sy1, sx0:sx1]
    if alpha >= 0.999:
        frame[cy0:cy1, cx0:cx1] = inset
    else:
        a = float(alpha)
        blended = (
            frame[cy0:cy1, cx0:cx1].astype(np.float32) * (1.0 - a)
            + inset.astype(np.float32) * a
        )
        frame[cy0:cy1, cx0:cx1] = blended.astype(np.uint8)

    # Border: stroked rectangle just outside the inset, fully
    # opaque even when the inset is fading in/out so the user can
    # always see the bounds.
    if border_thick > 0:
        bx0 = max(0, cx0 - 1)
        by0 = max(0, cy0 - 1)
        bx1 = min(fw - 1, cx1)
        by1 = min(fh - 1, cy1)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1),
                      border_color, border_thick, cv2.LINE_AA)
