"""Runtime integration glue for dynamic custom gestures.

Sits between the live engine (noop_engine.GestureWorker) and the
DynamicGestureClassifier. Responsibilities:

  * Load the registry, filter for kind="dynamic" entries, build a
    DynamicGestureClassifier with one template per gesture.
  * Track registry-file mtime so reload() can pick up new gestures
    without restarting the app — mirrors the static CustomGestureRunner.
  * Per-frame: normalize landmarks (wrist origin + palm scale), feed
    into the classifier, dispatch the matched gesture's action via
    the shared `fire_once` cooldown gate.

Kept separate from the static `CustomGestureRunner` so the two paths
can be developed and tested independently. They share the registry
file and the action-fire layer, nothing else.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

from .action import fire_once
from .dynamic_classifier import (
    _DEFAULT_MATCH_THRESHOLD,
    DynamicGestureClassifier,
    DynamicGestureTemplate,
)
from .dynamic_recording import normalize_frame
from .registry import GestureRegistry, registry_path


class DynamicGestureRuntime:
    """Live matcher for dynamic gestures.

    The engine instantiates ONE of these for the session and calls
    `process_frame()` once per camera frame. `reload()` should be
    called on construction and is also exposed for the engine's
    "maybe_reload_if_changed" mtime-watch path.
    """

    def __init__(self) -> None:
        self._registry: Optional[GestureRegistry] = None
        self._classifier: Optional[DynamicGestureClassifier] = None
        # Mapping from gesture name → registry entry, so on match we
        # can pull the Action + handedness without re-reading the
        # registry on every fire.
        self._gestures_by_name: dict = {}
        # Mtime tracking for the auto-reload watchdog.
        self._registry_path = registry_path()
        self._registry_mtime: float = 0.0
        self._last_mtime_check_at: float = 0.0
        # Hand-loss debounce: when the engine reports the hand left,
        # we want the classifier to forget the in-progress segment so
        # it doesn't get matched against half-tracked motion. Reset
        # on next `process_frame`.
        self._needs_reset = False
        # Most-recent fire — exposed via `current_match` so the live
        # banner can show the dynamic gesture's name briefly after
        # detection. Dynamic gestures fire instantaneously (no
        # hold-to-activate state like static gestures), so we keep
        # the label visible for a short window after each fire.
        self._last_fire_name: Optional[str] = None
        self._last_fire_handedness: Optional[str] = None
        self._last_fire_at: float = 0.0
        self._fire_label_visible_seconds: float = 1.2

    # ---- lifecycle ----

    def reload(self) -> None:
        """Re-read the registry and rebuild the classifier."""
        try:
            self._registry = GestureRegistry()
            self._registry.load()
        except Exception:
            self._registry = None
            self._classifier = None
            self._gestures_by_name = {}
            return
        templates = []
        by_name = {}
        try:
            entries = list(self._registry._gestures.values())  # noqa: SLF001
        except Exception:
            entries = []
        for gesture in entries:
            if str(getattr(gesture, "kind", "static") or "static") != "dynamic":
                continue
            try:
                template = self._template_from_registry_entry(gesture)
            except Exception:
                continue
            if template is None:
                continue
            templates.append(template)
            by_name[gesture.name] = gesture
        self._gestures_by_name = by_name
        threshold = self._live_match_threshold()
        if templates:
            self._classifier = DynamicGestureClassifier(
                templates, match_threshold=threshold,
            )
        else:
            self._classifier = None
        self._registry_mtime = self._read_registry_mtime()

    def maybe_reload_if_changed(self, now: float) -> None:
        """Throttled mtime watch — call from the hot path."""
        if now - self._last_mtime_check_at < 3.0:
            return
        self._last_mtime_check_at = now
        current = self._read_registry_mtime()
        if current > 0.0 and current != self._registry_mtime:
            self.reload()

    # ---- per-frame ----

    def process_frame(
        self,
        landmarks_21x3: np.ndarray,
        *,
        palm_scale: float,
        handedness: Optional[str],
        timestamp: float,
        dispatch: bool = True,
    ) -> Optional[str]:
        """Push one frame into the classifier. Returns the gesture
        name on match, else None.

        `landmarks_21x3` is the raw (21, 3) array from MediaPipe;
        we normalize internally (wrist origin + palm scale) so the
        engine doesn't have to know the dynamic-gesture conventions.

        `dispatch=True` (default) calls `fire_once` so the action
        runs and cooldowns apply — this is what the live engine
        wants. `dispatch=False` returns the matched gesture name
        WITHOUT firing the action; the sandbox uses that to surface
        detection feedback while still gating real action delivery
        behind its "Fire actions" checkbox.
        """
        if self._classifier is None:
            return None
        if landmarks_21x3 is None:
            return None
        try:
            if self._needs_reset:
                self._classifier.reset()
                self._needs_reset = False
            raw = landmarks_21x3.astype(np.float32)
            scale = max(float(palm_scale), 1e-6)
            normalized = normalize_frame(raw, scale)
            # Pass the absolute wrist position (in palm units) so the
            # classifier can gate out segments where the wrist barely
            # moved — necessary because `normalize_frame` subtracts
            # the per-frame wrist, erasing whole-hand swipe motion
            # from the (21, 3) trajectory and making "swipe up" look
            # identical to "hand entered view and held still".
            wrist_palm_scaled = raw[0] / scale
            match = self._classifier.update(
                normalized, float(timestamp),
                wrist_palm_scaled=wrist_palm_scaled,
            )
        except Exception:
            return None
        if match is None:
            return None
        # Look up the gesture for action dispatch + handedness check.
        gesture = self._gestures_by_name.get(match.gesture_name)
        if gesture is None:
            return None
        # Handedness gate: skip if the gesture is hand-specific and
        # this frame's hand doesn't match. Either-hand gestures
        # (handedness=None) fire regardless.
        if gesture.handedness and handedness and gesture.handedness != handedness:
            return None
        if not dispatch:
            self._last_fire_name = gesture.name
            self._last_fire_handedness = gesture.handedness
            self._last_fire_at = float(timestamp)
            return gesture.name
        try:
            fired = fire_once(gesture.name, gesture.action)
        except Exception:
            fired = False
        if fired:
            self._last_fire_name = gesture.name
            self._last_fire_handedness = gesture.handedness
            self._last_fire_at = float(timestamp)
        return gesture.name if fired else None

    def current_match(self, now: float) -> Optional[tuple]:
        """Mirror of CustomGestureRunner.current_match for the
        dynamic path. Returns (name, handedness) for a brief window
        after each fire so the live banner can show the gesture's
        name; None outside that window. Static-runner equivalent
        returns the held gesture during the hold; dynamic gestures
        have no hold so we use a short post-fire visibility window
        instead.
        """
        if self._last_fire_name is None:
            return None
        if now - self._last_fire_at > self._fire_label_visible_seconds:
            return None
        return (self._last_fire_name, self._last_fire_handedness)

    def hand_lost(self) -> None:
        """Tell the runtime the engine no longer has a tracked hand.
        Next frame will reset the classifier so partial-segment state
        from before the loss doesn't bleed into the new tracking."""
        self._needs_reset = True

    def has_dynamic_gestures(self) -> bool:
        return self._classifier is not None and bool(self._gestures_by_name)

    def has_loop_or_complex_templates(self) -> bool:
        if self._classifier is None:
            return False
        try:
            return bool(self._classifier.has_loop_or_complex_templates())
        except Exception:
            return False

    def spring_debug_rows(self):
        """Sandbox diagnostic: latest SPRING cost vs threshold per gesture."""
        if self._classifier is None:
            return []
        try:
            return self._classifier.spring_debug_rows()
        except Exception:
            return []

    def should_preempt_builtin_horizontal_swipe(
        self,
        landmarks_21x3: Optional[np.ndarray],
        *,
        palm_scale: float,
        now: float,
    ) -> bool:
        """See DynamicGestureClassifier.preempts_builtin_horizontal_swipe.

        Also true during the post-fire banner window so a builtin
        swipe that latches a frame later than the custom fire is
        still swallowed.
        """
        if self.has_loop_or_complex_templates():
            return True
        if self.current_match(now) is not None:
            return True
        try:
            if self._classifier is not None and self._classifier.live_path_looks_like_loop():
                return True
        except Exception:
            pass
        if self._classifier is None or landmarks_21x3 is None:
            return False
        try:
            raw = landmarks_21x3.astype(np.float32)
            scale = max(float(palm_scale), 1e-6)
            normalized = normalize_frame(raw, scale)
            return bool(self._classifier.preempts_builtin_horizontal_swipe(normalized))
        except Exception:
            return False

    # ---- internal ----

    def _read_registry_mtime(self) -> float:
        try:
            return float(self._registry_path.stat().st_mtime)
        except Exception:
            return 0.0

    @staticmethod
    def _live_match_threshold() -> float:
        """Allow override via env var, same pattern as the static
        runner uses for HGR_CUSTOM_GESTURES_LIVE_THRESHOLD. Default
        matches the classifier's tuned _DEFAULT_MATCH_THRESHOLD (0.18)
        — the previous 0.30 default left the live engine ~67% LOOSER
        than the classifier was tuned for, which is why every
        moderate hand motion fired a match."""
        raw = os.environ.get("HGR_DYNAMIC_GESTURES_THRESHOLD", "").strip()
        try:
            return float(raw) if raw else float(_DEFAULT_MATCH_THRESHOLD)
        except (TypeError, ValueError):
            return float(_DEFAULT_MATCH_THRESHOLD)

    @staticmethod
    def _template_from_registry_entry(gesture) -> Optional[DynamicGestureTemplate]:
        try:
            indices = list(int(i) for i in gesture.key_point_indices)
            if not indices:
                return None
            trajectories = [
                np.asarray(t, dtype=np.float32)
                for t in gesture.sample_trajectories
            ]
            if not trajectories:
                return None
            # Drop any malformed trajectory whose feature count
            # doesn't match (e.g. registry was hand-edited).
            valid = [
                t for t in trajectories
                if t.ndim == 3 and t.shape[1] == len(indices)
            ]
            if not valid:
                return None
            # Wrist channel — rehydrate only when the registry stored
            # a 1-to-1 parallel set of well-shaped (T, 3) wrist
            # trajectories. Legacy records (pre-wrist-channel) have an
            # empty list here; we silently fall back to finger-only
            # matching by passing an empty wrist list and strength 0.
            raw_wrist = getattr(gesture, "wrist_trajectories", None) or []
            wrist: List[np.ndarray] = []
            if len(raw_wrist) == len(valid):
                for w in raw_wrist:
                    try:
                        arr = np.asarray(w, dtype=np.float32)
                    except Exception:
                        wrist = []
                        break
                    if arr.ndim != 2 or arr.shape[1] != 3:
                        wrist = []
                        break
                    wrist.append(arr)
            try:
                strength = float(getattr(gesture, "wrist_motion_strength", 0.0) or 0.0)
            except (TypeError, ValueError):
                strength = 0.0
            if not wrist:
                strength = 0.0
            # v1.1.8.1: prefer the per-template match_threshold saved
            # in the registry. Legacy records (no field) get an on-the-
            # fly auto-threshold from pairwise DTW between takes, then
            # cached on the DynamicGestureTemplate for this session.
            # v1.1.8.2: the on-the-fly DTW pairwise threshold was
            # removed. Templates now persist match_threshold at build
            # time (SPRING-native pairwise self-scoring). Legacy
            # templates without a saved threshold fall through with
            # `None`, which lets the classifier apply its own default
            # (they still won't fire because they also lack
            # sample_features — see the WARN log in __init__).
            saved_threshold = getattr(gesture, "match_threshold", None)
            template_threshold: Optional[float] = (
                float(saved_threshold) if saved_threshold is not None else None
            )
            # v1.1.8.2 SPRING features. Legacy templates (schema < 3)
            # have empty sample_features and fall back to segment-DTW.
            raw_features = getattr(gesture, "sample_features", None) or []
            sample_features_list: List[np.ndarray] = []
            for feat in raw_features:
                try:
                    arr = np.asarray(feat, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] > 0:
                        sample_features_list.append(arr)
                except Exception:
                    continue
            # v1.1.8.2 (post-audit r2) intent-signature fields.
            intent_direction = getattr(gesture, "intent_direction", None)
            intent_magnitude = float(getattr(gesture, "intent_magnitude", 0.0) or 0.0)
            intent_window_seconds = float(
                getattr(gesture, "intent_window_seconds", 0.0) or 0.0
            )
            raw_tip_ext = getattr(gesture, "intent_fingertip_extension", None)
            intent_fingertip_extension = (
                list(raw_tip_ext) if raw_tip_ext and len(raw_tip_ext) == 5 else None
            )
            return DynamicGestureTemplate(
                name=str(gesture.name),
                key_point_indices=indices,
                sample_trajectories=valid,
                wrist_trajectories=wrist,
                wrist_motion_strength=strength,
                match_threshold=template_threshold,
                sample_features=sample_features_list,
                intent_direction=(
                    list(intent_direction) if intent_direction else None
                ),
                intent_magnitude=intent_magnitude,
                intent_window_seconds=intent_window_seconds,
                intent_fingertip_extension=intent_fingertip_extension,
            )
        except Exception:
            return None
