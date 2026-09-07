"""Live runner for custom gestures inside the running gesture engine.

Owns a GestureClassifier + per-frame hold/cooldown state machine. The
running GestureWorker calls process_frame(frame_bgr, now) every frame
that has a camera image; the runner runs its OWN MediaPipe pass to
extract hand landmarks, classifies them, and handles
hold-to-activate, grace-window flicker tolerance, and cooldown via
action.fire_once.

Why a private MediaPipe pass: the live engine may be using an ONNX
DirectML hand-landmark model (GPU mode), whose landmark coordinates
drift slightly from raw MediaPipe — enough to push classifier scores
~0.10–0.15 below their MediaPipe-trained baseline. Running our own
MediaPipe instance for custom gestures keeps the recognition path
identical to the recorder/sandbox so the live experience matches
exactly what the user sees while testing.

CPU cost: ~5–10 ms/frame at 30 fps. The MediaPipe Hands model is
loaded LAZILY on the first frame, and only when the registry has
at least one custom gesture (static, dynamic, or pose-sequence),
so there's zero overhead for users who haven't recorded anything.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from .action import fire_once
from .classifier import GestureClassifier
from .registry import CustomGesture, GestureRegistry


_DEFAULT_HOLD_SECONDS = 1.0
_DEFAULT_GRACE_SECONDS = 0.2


class CustomGestureRunner:
    """Per-worker custom-gesture controller. Cheap to construct, cheap
    to call from the hot loop.

    Threading note: the classifier matrix is built once on reload() and
    then read-only on each .process() call, so no locking is needed
    even though .reload() may be invoked from the Qt main thread while
    .process() runs on the worker thread — `np.ndarray @ vec` is a pure
    read.
    """

    def __init__(
        self,
        *,
        default_hold_seconds: float = _DEFAULT_HOLD_SECONDS,
        grace_seconds: float = _DEFAULT_GRACE_SECONDS,
        binding_resolver=None,
        image_overlay_handler=None,
    ) -> None:
        self._registry = GestureRegistry()
        self._classifier: Optional[GestureClassifier] = None
        self._hold_name: Optional[str] = None
        self._hold_started_at = 0.0
        self._last_match_at = 0.0
        self._fired_for_hold = False
        self._default_hold = float(default_hold_seconds)
        self._grace = float(grace_seconds)
        # Optional callback consulted on the firing edge. Signature:
        #   binding_resolver(gesture_name: str) -> bool
        # Returns True if the engine fired a user-remapped action for this
        # gesture (so the runner skips fire_once on its stored action and
        # marks the hold as fired). Returns False to let the runner fire
        # the gesture's stored action as usual. Decoupled from the engine
        # via callback so the runner stays pure (no engine import).
        self._binding_resolver = binding_resolver
        # Qt-coupled bridge for the "show_overlay_drawing" action kind.
        # Signature: image_overlay_handler(filename: str) -> None
        # Called on the firing edge when a gesture's stored action is
        # show_overlay_drawing. The engine's handler emits a Qt signal
        # that the main window listens to so the actual QWidget
        # mutation happens on the GUI thread — keeps this runner pure
        # (no Qt import) and the engine pure (no widget code).
        self._image_overlay_handler = image_overlay_handler
        # Idle-throttle: when no candidate is being held, only run the
        # classifier at ~20 Hz (50 ms intervals) instead of every frame.
        # Custom gestures need ~1 s of hold to fire, so a 50 ms sampling
        # interval is plenty for hold detection while halving the
        # classifier cost when the user isn't actually holding a gesture.
        # When a hold IS in progress, sample every frame for responsive
        # release detection.
        self._idle_sample_interval_s = 0.05
        self._last_classify_at: float = 0.0
        # Registry file mtime tracking — used by maybe_reload_if_changed()
        # so the live runner picks up gestures saved by the recorder /
        # sandbox / wizard without depending on every save path
        # explicitly calling worker.reload_custom_gestures().
        self._registry_mtime: float = 0.0
        self._last_mtime_check_at: float = 0.0
        # Throttled per-frame diagnostic — one print every 2 s so the
        # log isn't flooded but the user can SEE why their custom
        # gesture isn't firing (no match? wrong hand? score below
        # threshold?). Set HGR_CUSTOM_GESTURES_DEBUG=0 to silence.
        self._last_debug_log_at: float = 0.0
        import os
        self._debug_enabled: bool = os.environ.get(
            "HGR_CUSTOM_GESTURES_DEBUG", "1"
        ).strip() not in ("0", "false", "False", "")
        # Async MediaPipe warm-up state. mp.solutions.hands.Hands()
        # construction loads the landmark + palm-detect model files
        # and takes ~1-3 s on first call — historically this ran on
        # the GUI thread the first time process_frame() saw a hand,
        # which froze the camera display for the duration. We now
        # pre-warm in a background thread the moment the runner has
        # at least one gesture registered. process_frame() handles
        # _mp_hands is None gracefully (returns None until ready).
        self._mp_hands = None
        self._mp_init_in_flight = False
        self.reload()
        self._registry_mtime = self._read_registry_mtime()

    @staticmethod
    def _read_registry_mtime() -> float:
        try:
            from .registry import registry_path
            path = registry_path()
            return float(path.stat().st_mtime) if path.exists() else 0.0
        except Exception:
            return 0.0

    def maybe_reload_if_changed(self, now: float) -> None:
        """If the registry file's mtime has advanced since our last
        read, reload from disk. Throttled to one stat() per ~3 s so
        the per-frame cost is negligible. Lets the live engine pick
        up newly-saved gestures from the recorder / sandbox / wizard
        even when the save path didn't explicitly trigger a reload."""
        if now - self._last_mtime_check_at < 3.0:
            return
        self._last_mtime_check_at = now
        current = self._read_registry_mtime()
        if current > 0.0 and current != self._registry_mtime:
            self.reload()
            self._registry_mtime = current

    @property
    def has_gestures(self) -> bool:
        if self._classifier is None:
            return False
        return bool(self._classifier._gestures)  # internal, but cheap to peek

    def _registry_wants_private_mp(self) -> bool:
        """True when any custom gesture (static / dynamic / sequence)
        needs recorder-matching MediaPipe landmarks on GPU / Lite."""
        if self.has_gestures:
            return True
        try:
            if self._registry is None:
                return False
            return bool(self._registry.list())
        except Exception:
            return False

    @property
    def current_match(self) -> Optional[Tuple[str, Optional[str]]]:
        """If a gesture is currently being classified / held this
        frame, return (name, handedness). Used by the live overlay so
        the custom gesture's name appears over the hand banner — same
        affordance built-in gestures get. Returns None when no match
        is active. `handedness` may be None for legacy/either-hand
        gestures.
        """
        if self._hold_name is None:
            return None
        gh: Optional[str] = None
        if self._registry is not None:
            try:
                g = self._registry.get(self._hold_name)
                if g is not None:
                    gh = g.handedness
            except Exception:
                gh = None
        return (self._hold_name, gh)

    def reload(self) -> None:
        """Re-read the registry from disk and rebuild the classifier.
        Call this after the user adds / edits / deletes a gesture so
        the live pipeline picks up changes without restarting the app.

        Uses the classifier's default threshold — process_frame() runs
        its own MediaPipe pass to keep landmark distribution identical
        to the recorder, so live scores match what the sandbox shows.
        Override via HGR_CUSTOM_GESTURES_LIVE_THRESHOLD env var if
        needed.
        """
        import os
        env_threshold = os.environ.get("HGR_CUSTOM_GESTURES_LIVE_THRESHOLD", "").strip()
        # Default 0.78 (vs sandbox's 0.88) — real-time live frames have
        # more landmark jitter than recorded samples, and complex poses
        # like 'index slightly curled + middle extended + pinky half'
        # often score in the 0.78–0.85 range even with MediaPipe in
        # both pipelines. The classifier's confidence-margin check
        # (0.05) still gates false positives between similar gestures.
        try:
            threshold = float(env_threshold) if env_threshold else 0.78
        except (TypeError, ValueError):
            threshold = 0.78
        try:
            self._registry = GestureRegistry()
            self._registry.load()
            classifier = GestureClassifier(self._registry, threshold=threshold)
            classifier.reload()
            self._classifier = classifier
        except Exception:
            self._classifier = None
        # Reset hold state so a stale half-completed hold from before
        # the reload doesn't fire post-reload.
        self._hold_name = None
        self._fired_for_hold = False
        # Kick off the async MediaPipe warm-up now that we know
        # whether the registry has any gestures to classify.
        self._warm_mediapipe_async()

    def _warm_mediapipe_async(self) -> None:
        """Construct the private MediaPipe Hands instance on a
        background thread so the first hand-in-frame after a gesture
        is added doesn't pay the 1-3 s model-load cost on the main
        thread. Idempotent — repeated calls are no-ops while a load
        is in flight or already complete.

        IMPORTANT: the `import mediapipe` itself runs on the CALLING
        thread (main, in practice), NOT the background thread. A
        previous version did the import on the bg thread, which
        raced against the engine's own `import mediapipe` in
        start_engine -> HandDetector -> _load_mediapipe_cpu_runtime.
        Python's import lock detects the cross-thread cycle and
        aborts one of the imports mid-way -- leaving a
        partially-built mediapipe module in sys.modules and the
        engine then crashes with `NameError: name 'core' is not
        defined` when it touches mediapipe.tasks.python.audio.
        Doing the import here (main thread) loads + caches the
        module into sys.modules before any worker thread can race
        against it; subsequent `import mediapipe` calls (from
        engine workers, prewarm threads, etc.) hit the cached
        module and don't take the import lock at all.

        The slow part is the actual `mp.solutions.hands.Hands(...)`
        construction (~1-3 s model load), which still runs on the
        background thread so the main thread doesn't block."""
        if self._mp_hands is not None or self._mp_init_in_flight:
            return
        if not self._registry_wants_private_mp():
            return
        # Synchronous import on the calling thread. Cheap if the
        # module is already cached, ~0.5 s on cold cache. Avoids
        # the deadlock that bit users in v1.1.0b6.
        try:
            import mediapipe as mp  # noqa: F401  -- import side-effect is the point
        except Exception as exc:
            print(f"[custom-gestures] MediaPipe import failed: {exc}")
            self._mp_hands = None
            return
        self._mp_init_in_flight = True

        def _runner() -> None:
            try:
                import mediapipe as mp  # already cached; just for the binding
                instance = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                # Publish the ready instance with a single attribute
                # write — process_frame() reads _mp_hands without a
                # lock, so the GIL guarantees it sees either the old
                # None or the fully-constructed object, never a
                # partially-initialised one.
                self._mp_hands = instance
            except Exception as exc:
                print(f"[custom-gestures] MediaPipe Hands ctor failed: {exc}")
                self._mp_hands = None
            finally:
                self._mp_init_in_flight = False

        threading.Thread(
            target=_runner,
            name="custom-gestures-mp-init",
            daemon=True,
        ).start()

    def _ensure_mediapipe(self) -> None:
        """Lazy fallback for paths that didn't go through reload() —
        kicks the same async warm-up. Returns immediately; the
        caller (process_frame) checks _mp_hands and bails when it
        is still None."""
        self._warm_mediapipe_async()

    def _should_skip_classify(self, now: float) -> bool:
        """Return True if this frame's classifier work can be skipped
        for throttling. Active only when no hold is in progress —
        once a candidate gesture is being held, every frame is
        classified so release detection stays snappy."""
        if self._hold_name is not None:
            return False
        if (now - self._last_classify_at) < self._idle_sample_interval_s:
            return True
        return False

    def process_engine_hands(
        self,
        hands: list,
        now: float,
    ) -> Optional[str]:
        """Classify using already-extracted hand landmarks from the live
        engine's hand-tracking pipeline. Skips the runner's private
        MediaPipe pass entirely — saves ~5–10 ms/frame.

        `hands` is a list of (landmarks_21x3, handedness_or_None) tuples,
        in arbitrary order. Same hand-picking logic as process_frame:
        if a gesture is currently being held, prefer the hand whose
        label matches the held gesture's stored handedness.

        Caller is responsible for ensuring the landmarks are
        MediaPipe-compatible — i.e., produced by a MediaPipe runtime
        with the same model_complexity the recorder used (default 1).
        ONNX/DirectML landmarks drift enough from MediaPipe to push
        classifier scores 0.10–0.15 below their trained baseline; in
        that case use process_frame() instead so the private MediaPipe
        pass keeps the landmark distribution consistent."""
        if self._classifier is None or not self.has_gestures:
            return None
        if not hands:
            self.hand_lost(now)
            return None
        if self._should_skip_classify(now):
            return None
        self._last_classify_at = now

        pick: Optional[Tuple[np.ndarray, Optional[str]]] = None
        if self._hold_name is not None and self._registry is not None:
            held_g = self._registry.get(self._hold_name)
            desired = held_g.handedness if held_g is not None else None
            if desired in ("Left", "Right"):
                for lm, lab in hands:
                    if lab == desired:
                        pick = (lm, lab)
                        break
        if pick is None:
            pick = hands[0]

        return self.process(pick[0], now, handedness=pick[1])

    def extract_hands(
        self, frame_bgr: np.ndarray
    ) -> list:
        """MediaPipe Hands pass matching the recorder (complexity=1).

        Used on GPU / Lite so static, dynamic, and pose-sequence
        custom gestures see the same landmark distribution they were
        trained on. Returns a list of (landmarks_21x3, handedness).
        """
        self._ensure_mediapipe()
        if self._mp_hands is None or frame_bgr is None:
            return []
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(rgb)
        except Exception:
            return []
        if not result.multi_hand_landmarks:
            return []
        hands: list[Tuple[np.ndarray, Optional[str]]] = []
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label: Optional[str] = None
            try:
                if result.multi_handedness and i < len(result.multi_handedness):
                    raw = str(result.multi_handedness[i].classification[0].label)
                    label = raw if raw in ("Left", "Right") else None
            except Exception:
                pass
            lm = np.array(
                [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                dtype=np.float32,
            )
            hands.append((lm, label))
        return hands

    def process_frame(self, frame_bgr: np.ndarray, now: float) -> Optional[str]:
        """Run a private MediaPipe pass on the camera frame to extract
        hand landmarks, then classify + advance hold state. Returns the
        name of the gesture whose action just fired this frame, or None.

        Bypasses the live engine's hand-landmark runtime (which may be
        ONNX in GPU mode), so live recognition uses the same MediaPipe
        landmark distribution the user trained against.
        """
        if self._classifier is None or not self.has_gestures:
            return None
        if frame_bgr is None:
            self.hand_lost(now)
            return None
        if self._should_skip_classify(now):
            # Skip both MediaPipe AND classify — they're both wasted
            # work on this frame. The throttle bound preserves snappy
            # hold detection while reclaiming the ~5–10 ms MediaPipe
            # pass on the half of the frames it would have run.
            return None
        self._last_classify_at = now
        hands = self.extract_hands(frame_bgr)
        if not hands:
            self.hand_lost(now)
            self._maybe_debug(now, "no hand detected (MP)")
            return None

        # Pick which hand to feed the runner this frame. If we're
        # already holding a gesture, prefer the hand whose label
        # matches the held gesture's stored handedness so the timer
        # doesn't reset when the OTHER hand also enters frame.
        pick: Optional[Tuple[np.ndarray, Optional[str]]] = None
        if self._hold_name is not None and self._registry is not None:
            held_g = self._registry.get(self._hold_name)
            desired = held_g.handedness if held_g is not None else None
            if desired in ("Left", "Right"):
                for lm, lab in hands:
                    if lab == desired:
                        pick = (lm, lab)
                        break
        if pick is None:
            pick = hands[0]

        return self.process(pick[0], now, handedness=pick[1])

    @staticmethod
    def _hold_seconds_for(gesture: CustomGesture, default: float) -> float:
        """Per-gesture hold-to-activate duration, stored in the action
        payload by the wizard. Falls back to the runner's default."""
        try:
            payload = gesture.action.payload or {}
            value = payload.get("hold_s")
            if value is None:
                return default
            return max(0.05, float(value))
        except Exception:
            return default

    def hand_lost(self, now: float) -> None:
        """Called from the worker when there's no tracked hand this
        frame. Drops the hold after the grace window so a brief
        MediaPipe dropout doesn't reset the timer."""
        if self._hold_name is None:
            return
        if now - self._last_match_at >= self._grace:
            self._hold_name = None
            self._fired_for_hold = False

    def _maybe_debug(self, now: float, *parts: object) -> None:
        """Print at most one diagnostic line every 2 s. Caller passes
        the live state as positional args for 'live=...' / 'score=...'
        formatting. Silenced via HGR_CUSTOM_GESTURES_DEBUG=0."""
        if not self._debug_enabled:
            return
        if now - self._last_debug_log_at < 2.0:
            return
        self._last_debug_log_at = now
        print("[custom-gestures]", *parts)

    def process(
        self,
        landmarks_21x3: np.ndarray,
        now: float,
        handedness: Optional[str] = None,
    ) -> Optional[str]:
        """Classify the current landmarks and advance hold state. Returns
        the name of the gesture whose action just fired this frame, or
        None. Safe to call when no gestures are registered (returns None).

        `handedness` is the MediaPipe label of the tracked hand for
        this frame ("Left" / "Right" / None). A gesture only fires
        when its stored handedness matches (or when either side is
        None — legacy gestures fire on any hand).
        """
        if self._classifier is None:
            self._maybe_debug(now, "no classifier (registry empty?)")
            return None
        if landmarks_21x3 is None:
            self.hand_lost(now)
            return None
        try:
            match = self._classifier.classify(
                landmarks_21x3, sticky_name=self._hold_name
            )
        except Exception as exc:
            self._maybe_debug(now, f"classify error: {exc}")
            return None

        if match is None:
            self.hand_lost(now)
            # Diagnostic best-score lookup is gated on (1) debug being
            # enabled at all and (2) the 2-second throttle being clear,
            # because best_score_for() runs a full classifier pass — at
            # 30 FPS with a hand in frame it was burning ~3-5 ms/frame
            # on numpy work that nobody saw most of the time.
            should_log = self._debug_enabled and (now - self._last_debug_log_at >= 2.0)
            if should_log:
                try:
                    top_name, top_score = self._classifier.best_score_for(landmarks_21x3)
                except Exception:
                    top_name, top_score = (None, 0.0)
                self._maybe_debug(
                    now,
                    f"hand={handedness} no match — best={top_name!r} "
                    f"score={top_score:.2f} (threshold {self._classifier.threshold:.2f})",
                )
            return None

        # Handedness gate. Both sides None = legacy/either-hand → allow.
        # If both are concrete labels and they differ, drop the match —
        # treat it as a no-match this frame so the hold timer resets,
        # so a quick L/R swap doesn't fire a wrong-hand gesture.
        gesture_hand = match.gesture.handedness
        if (
            gesture_hand in ("Left", "Right")
            and handedness in ("Left", "Right")
            and gesture_hand != handedness
        ):
            self.hand_lost(now)
            self._maybe_debug(
                now,
                f"matched {match.gesture.name!r} score={match.score:.2f} "
                f"but live hand is {handedness!r} — gesture is bound to {gesture_hand!r}",
            )
            return None

        self._last_match_at = now
        if self._hold_name != match.gesture.name:
            self._hold_name = match.gesture.name
            self._hold_started_at = now
            self._fired_for_hold = False

        held = now - self._hold_started_at
        hold_duration = self._hold_seconds_for(match.gesture, self._default_hold)
        self._maybe_debug(
            now,
            f"matched {match.gesture.name!r} score={match.score:.2f} "
            f"hand={handedness} held={held:.2f}s/{hold_duration:.2f}s "
            f"fired={self._fired_for_hold}",
        )
        if not self._fired_for_hold and held >= hold_duration:
            # Binding-resolver path: if the user has remapped this custom
            # gesture to fire a different action via Settings → Gesture
            # Binds, the engine handles the dispatch and we suppress the
            # runner's own fire_once. We still mark the hold as fired so
            # we don't re-trigger every frame for the rest of the hold.
            if self._binding_resolver is not None:
                try:
                    handled = bool(self._binding_resolver(match.gesture.name))
                except Exception as exc:
                    handled = False
                    print(
                        f"[custom-gestures] binding_resolver error for "
                        f"{match.gesture.name!r}: {exc}"
                    )
                if handled:
                    self._fired_for_hold = True
                    print(
                        f"[custom-gestures] FIRED-REMAPPED {match.gesture.name!r} "
                        f"(score={match.score:.2f}, hand={handedness})"
                    )
                    return match.gesture.name
            # Qt-coupled action: route to the image-overlay handler
            # registered by the engine. fire_once would just no-op
            # this kind (returns True without doing anything), so
            # the bridge has to live here. The handler talks to the
            # GUI thread via a queued Qt signal.
            if match.gesture.action.kind == "show_overlay_drawing":
                payload = match.gesture.action.payload or {}
                filename = str(payload.get("filename", ""))
                # `path` is set by the wizard at gesture-creation time
                # after the user resolved any duplicate-name ambiguity.
                # Empty string here = legacy gesture made before the
                # creation-time validation existed; main_window falls
                # back to filesystem search.
                resolved_path = str(payload.get("path", ""))
                if self._image_overlay_handler is not None:
                    try:
                        self._image_overlay_handler(filename, resolved_path)
                    except Exception as exc:
                        print(
                            f"[custom-gestures] image_overlay_handler error "
                            f"for {match.gesture.name!r}: {exc}"
                        )
                self._fired_for_hold = True
                print(
                    f"[custom-gestures] FIRED-OVERLAY {match.gesture.name!r} "
                    f"file={filename!r} (score={match.score:.2f}, hand={handedness})"
                )
                return match.gesture.name
            if fire_once(match.gesture.name, match.gesture.action):
                self._fired_for_hold = True
                print(
                    f"[custom-gestures] FIRED {match.gesture.name!r} "
                    f"(score={match.score:.2f}, hand={handedness})"
                )
                return match.gesture.name
            else:
                self._maybe_debug(
                    now,
                    f"hold complete for {match.gesture.name!r} but fire_once "
                    f"returned False (cooldown still active or action failed)",
                )
        return None

# Author: Konstantin Markov
