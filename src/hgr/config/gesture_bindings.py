"""Shared registry of bindable actions, gesture poses, and the user's
configured bindings between them.

Why this module exists: both the Settings → Gesture Binds UI
(main_window.py) and the live engine (noop_engine.py) need to read
the same data — what poses exist, what actions exist, what each action
defaults to, and what the user has overridden. Importing from
main_window.py would drag in PySide6/UI deps, so the registries and
all pure resolution helpers live here.

Naming conventions:
- pose_id: stable string identifier for a gesture pose. Static poses
  use plain ids ("right_two", "mute", "wheel_pose", ...). Custom
  gestures use the "custom:<gesture_name>" prefix.
- action_id: stable string identifier for a bindable action. Static
  actions use plain ids ("open_spotify", "system_mute_toggle", ...).
  Custom-gesture-default actions use "custom_action:<gesture_name>".
"""
from __future__ import annotations

from typing import Optional


# Each pose: (pose_id, display_label, image_filename, description).
# Order is the display order in the All Gesture Poses list.
_GESTURE_BIND_POSES: list[tuple[str, str, str, str]] = [
    (
        "left_one",
        "Left Hand One",
        "Left One.png",
        "Face your left palm toward the monitor, extend only the index finger, and keep the thumb, middle, ring, and pinky closed. Hold for ~1s.",
    ),
    (
        "left_two",
        "Left Hand Two",
        "Left Two.png",
        "Face your left palm toward the monitor, extend the index and middle fingers in a V, and keep the thumb, ring, and pinky closed. Hold for ~1s.",
    ),
    (
        "left_three",
        "Left Hand Three",
        "Left Three.png",
        "Face your left palm toward the monitor. Extend the index, middle, and ring fingers and fold the thumb and pinky. Hold for ~1s.",
    ),
    (
        "left_four",
        "Left Hand Four",
        "Left Hand Four.png",
        "Face your left palm toward the monitor. Extend the index, middle, ring, and pinky fingers and fold the thumb across the palm. Hold for ~1s.",
    ),
    (
        "left_fist",
        "Left Hand Fist",
        "Fist.png",
        "Face your left palm toward the monitor and close all five fingers into a tight, compact fist. Hold briefly to cancel voice listening or a save-location prompt (the file stays in the default folder).",
    ),
    (
        "right_two",
        "Right Hand Two",
        "Two.png",
        "Face your right palm toward the monitor, extend the index and middle fingers in a V, and keep the thumb, ring, and pinky closed. Hold for ~1s.",
    ),
    (
        "right_fist",
        "Right Hand Fist",
        "Fist.png",
        "Face your right palm toward the monitor and close all five fingers into a tight, compact fist. Hold for ~1s.",
    ),
    (
        "right_three",
        "Right Hand Three (together)",
        "Three.png",
        "Face your right palm toward the monitor. Extend the index, middle, and ring fingers and hold them TOGETHER (touching, not spread); fold the thumb and pinky. Hold for ~1s.",
    ),
    (
        "right_four",
        "Right Hand Four",
        "Four.png",
        "Face your right palm toward the monitor. Extend the index, middle, ring, and pinky fingers; fold the thumb across the palm. Hold for ~1s.",
    ),
    (
        "mute",
        "Mute",
        "Mute.png",
        "Face your right palm toward the monitor. Extend the thumb and pinky outward (a 'call me' shape) while folding the index, middle, and ring fingers. Hold for ~1s.",
    ),
    (
        "wheel_pose",
        "Gesture Wheel",
        "Wheel Pose.png",
        "Face your right palm toward the monitor and make the wheel pose (thumb, index, and pinky extended; middle and ring folded). Hold for ~1s.",
    ),
    (
        "screen_wheel",
        "Screen Wheel",
        "ScreenWheel.png",
        "Face your right palm toward the monitor. Extend the index finger and pinky while folding the thumb, middle, and ring (a 'rock on' shape). Hold for ~1s.",
    ),
    (
        "close_window",
        "Close Window",
        "close.png",
        "Face your right palm toward the monitor. Extend the thumb out sideways (horizontal, not pointing up) and fold the index, middle, ring, and pinky into the palm. Hold for ~1s to close the focused window.",
    ),
    (
        "right_pinch",
        "Right Hand Pinch",
        "",
        "Face your right palm toward the monitor. Curl the middle, ring, and pinky into the palm. Curve the thumb and index toward each other in a C-shape (they don't have to touch). Used to grab and move drawings; combine with left-hand pinch to stretch.",
    ),
    (
        "left_pinch",
        "Left Hand Pinch",
        "",
        "Face your left palm toward the monitor. Curl the middle, ring, and pinky into the palm. Curve the thumb and index toward each other in a C-shape (they don't have to touch). Used to grab and move drawings; combine with right-hand pinch to stretch.",
    ),
]


# Each action: (action_id, display_label, default_pose_id).
_GESTURE_BIND_ACTIONS: list[tuple[str, str, str]] = [
    ("voice_command_listen", "Start voice command listening", "left_one"),
    ("voice_cancel", "Cancel voice or save prompt", "left_fist"),
    # r53: dictation_toggle temporarily removed; left_two now hosts
    # instant_clip (moved from right_one).
    ("mouse_mode_toggle", "Toggle mouse mode on/off", "left_three"),
    ("drawing_mode_toggle", "Toggle drawing mode on/off", "left_four"),
    ("open_spotify", "Open or focus Spotify", "right_two"),
    ("play_pause", "Play or pause media", "right_fist"),
    ("chrome_mode_toggle", "Toggle Chrome mode on/off", "right_three_together"),
    ("youtube_mode_toggle", "Toggle YouTube mode on/off", "right_four_together"),
    ("open_chrome", "Open or focus Chrome", "right_three"),
    ("open_touchless", "Open or focus Touchless", "right_four"),
    ("system_mute_toggle", "Mute or unmute system audio", "mute"),
    ("open_gesture_wheel", "Open Spotify/Chrome wheel", "wheel_pose"),
    ("open_screen_wheel", "Open screen capture wheel", "screen_wheel"),
    ("close_active_window", "Close the focused window", "close_window"),
    # Instant clip: same as "clip that" voice command, no prompt and
    # no save dialog — clips the configured duration (default 60 s)
    # ending at the moment the gesture fires and auto-saves to
    # clips_save_dir with a timestamped filename. r53: default pose
    # moved from right_one to left_two after dictation was retired.
    ("instant_clip", "Save instant clip (no prompt)", "left_two"),
]


# Static pose_id <-> (handedness, recognizer label) bidirectional map.
# Used by the engine to translate detected labels into pose_ids and
# back, so a user-bound swap can be applied as a label rewrite at the
# entry of static-pose handlers.
#
# Notes on coverage:
#   - "wheel_pose" maps to recognizer label "wheel_pose".
#   - "screen_wheel" intentionally has no entry: the engine detects it
#     by inspecting hand_reading.fingers (index+pinky extended, others
#     folded) rather than via the static recognizer's label space, so
#     a label-rewrite remap can't reach that code path. Rebinding the
#     screen_wheel pose is therefore a no-op in this iteration; future
#     work should plumb the binding through _utility_wheel_pose_active.
STATIC_POSE_LABEL_MAP: dict[str, tuple[str, str]] = {
    "left_one":   ("Left",  "one"),
    "left_two":   ("Left",  "two"),
    "left_three": ("Left",  "three"),
    "left_four":  ("Left",  "four"),
    "left_fist":  ("Left",  "fist"),
    "right_one":  ("Right", "one"),
    "right_two":  ("Right", "two"),
    "right_fist": ("Right", "fist"),
    "right_three": ("Right", "three"),
    "right_three_together": ("Right", "three_together"),
    "right_four":  ("Right", "four"),
    "right_four_together": ("Right", "four_together"),
    "mute":       ("Right", "mute"),
    "wheel_pose": ("Right", "wheel_pose"),
    "right_pinch": ("Right", "pinch"),
    "left_pinch":  ("Left",  "pinch"),
}

STATIC_LABEL_TO_POSE: dict[tuple[str, str], str] = {
    v: k for k, v in STATIC_POSE_LABEL_MAP.items()
}


def gesture_bind_actions() -> list[tuple[str, str, str]]:
    """Public accessor for the action registry. Returns a fresh copy
    so callers can't mutate the module-level list."""
    return list(_GESTURE_BIND_ACTIONS)


def gesture_bind_poses() -> list[tuple[str, str, str, str]]:
    """Public accessor for the pose registry. Returns a fresh copy."""
    return list(_GESTURE_BIND_POSES)


def default_pose_for_action(action_id: str) -> Optional[str]:
    """Return the default pose_id for an action_id (the pose that fires
    the action when the user has not remapped). Returns the gesture's
    own pose for custom actions."""
    if not action_id:
        return None
    if action_id.startswith("custom_action:"):
        return f"custom:{action_id.split(':', 1)[1]}"
    for aid, _label, default_pose in _GESTURE_BIND_ACTIONS:
        if aid == action_id:
            return default_pose
    return None


def resolve_gesture_binding(config, action_id: str) -> str:
    """Return the pose_id currently bound to action_id (default if unset).

    Custom gestures default to themselves: action_id `custom_action:foo`
    defaults to pose_id `custom:foo` unless the user has remapped it."""
    user_map = getattr(config, "gesture_bindings", None) or {}
    if action_id in user_map and user_map[action_id]:
        return str(user_map[action_id])
    return default_pose_for_action(action_id) or ""


def action_bound_to_pose(config, pose_id: str) -> Optional[str]:
    """Inverse lookup: return the action_id whose effective binding
    (user-set if present, otherwise default) currently points to
    pose_id. Returns None if no action is bound to this pose.

    For custom poses, the implicit default binding is action_id
    `custom_action:<name>` — kept consistent with default_pose_for_action.
    """
    if not pose_id:
        return None
    user_map = getattr(config, "gesture_bindings", None) or {}
    # Explicit user bindings take precedence.
    for action_id, bound_pose in user_map.items():
        if bound_pose == pose_id:
            return action_id
    # Default bindings: only count if the user hasn't remapped this
    # action away from its default.
    for action_id, _label, default_pose in _GESTURE_BIND_ACTIONS:
        if default_pose == pose_id and action_id not in user_map:
            return action_id
    # Custom default: a "custom:<name>" pose maps to its own
    # "custom_action:<name>" UNLESS the user has explicitly remapped
    # custom_action:<name> to something else.
    if pose_id.startswith("custom:"):
        name = pose_id.split(":", 1)[1]
        owner_action = f"custom_action:{name}"
        if owner_action not in user_map:
            return owner_action
    return None


def pose_id_for_static_label(handedness: Optional[str], label: str) -> Optional[str]:
    """Map (handedness, recognizer label) -> pose_id, or None if there's
    no static pose registered for that combination."""
    if not handedness or not label:
        return None
    return STATIC_LABEL_TO_POSE.get((handedness, label))


def static_label_for_pose_id(pose_id: str) -> Optional[tuple[str, str]]:
    """Inverse: pose_id -> (handedness, recognizer label) if it's a
    static pose, otherwise None (custom poses have no static label)."""
    return STATIC_POSE_LABEL_MAP.get(pose_id)

# Author: Konstantin Markov
